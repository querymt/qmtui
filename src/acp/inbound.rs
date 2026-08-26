use std::sync::Arc;

use agent_client_protocol::{self as acp_sdk, JsonRpcMessage, schema::v1 as acp};
use serde_json::{Value, json};

use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};

use super::elicitation::{self, PendingResponse};
use super::events::EventSink;
use super::extensions::{delegation, mesh, models};
use super::notification;
use super::runtime::RuntimeState;
use super::transport::jsonrpc::{Envelope, Peer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExtensionNotification {
    Delegation,
    ModelsChanged,
    MeshChanged,
}

pub(super) fn extension_notification(method: &str) -> Option<ExtensionNotification> {
    match method {
        "querymt/session/delegationUpdate" | "_querymt/session/delegationUpdate" => {
            Some(ExtensionNotification::Delegation)
        }
        "querymt/models/changed" => Some(ExtensionNotification::ModelsChanged),
        "querymt/mesh/nodesChanged" | "querymt/mesh/joined" | "querymt/mesh/peerExpired" => {
            Some(ExtensionNotification::MeshChanged)
        }
        _ => None,
    }
}

pub(super) async fn session_notification(
    state: &Arc<RuntimeState>,
    events: &EventSink,
    notification: acp::SessionNotification,
) {
    let (session_id, translated) = notification::translate(notification);
    notification::apply(state, events, session_id, translated).await;
}

pub(super) fn delegation_notification(events: &EventSink, params: Value) {
    match delegation::from_value(params) {
        Ok(Some(update)) => events.send(AcpAppEvent::DelegationUpdate(update)),
        Ok(None) => events.info("delegation", "ignored delegation notification version"),
        Err(err) => events.info(
            "delegation",
            format!("invalid delegation notification: {err}"),
        ),
    }
}

pub(super) async fn websocket_text(
    peer: &Peer,
    state: &Arc<RuntimeState>,
    events: &EventSink,
    text: &str,
) -> Result<(), acp_sdk::Error> {
    let envelope: Envelope =
        serde_json::from_str(text).map_err(acp_sdk::Error::into_internal_error)?;
    if envelope.method.is_none() {
        peer.resolve(envelope).await;
        return Ok(());
    }
    let method = envelope.method.clone().unwrap_or_default();
    match method.as_str() {
        "session/update" => {
            let notification = acp::SessionNotification::parse_message(&method, &envelope.params)?;
            session_notification(state, events, notification).await;
        }
        "session/request_permission" => {
            let request = acp::RequestPermissionRequest::parse_message(&method, &envelope.params)?;
            if let Some(id) = envelope.id {
                peer.respond(
                    id,
                    serde_json::to_value(elicitation::permission_response(&request))
                        .map_err(acp_sdk::Error::into_internal_error),
                )?;
            }
        }
        "elicitation/create" => {
            let request = acp::CreateElicitationRequest::parse_message(&method, &envelope.params)?;
            if let Some(id) = envelope.id {
                register_direct_elicitation(state, events, peer.clone(), id, request).await;
            }
        }
        "elicitation/requested" => {
            register_result_elicitation(state, events, peer.clone(), envelope.params).await;
        }
        _ => match extension_notification(&method) {
            Some(ExtensionNotification::Delegation) => {
                delegation_notification(events, envelope.params)
            }
            Some(ExtensionNotification::ModelsChanged) => {
                spawn_model_refresh(peer.clone(), state.clone(), events.clone())
            }
            Some(ExtensionNotification::MeshChanged) => {
                spawn_mesh_refresh(peer.clone(), events.clone())
            }
            None => {}
        },
    }
    Ok(())
}

fn spawn_model_refresh(peer: Peer, state: Arc<RuntimeState>, events: EventSink) {
    tokio::spawn(async move {
        if let Ok(response) = models::list(&peer, false).await {
            state.set_models(response.models.clone()).await;
            events.send(AcpAppEvent::Models {
                models: response
                    .models
                    .iter()
                    .map(models::Model::to_app_model)
                    .collect(),
                meta: response
                    .meta
                    .as_ref()
                    .map(|meta| crate::acp_state::AcpModelsMetaInfo {
                        remote_node_count: meta.remote_node_count,
                        remote_timeout_count: meta.remote_timeout_count,
                    }),
            });
        }
    });
}

fn spawn_mesh_refresh(peer: Peer, events: EventSink) {
    tokio::spawn(async move {
        if let Ok(Some(nodes)) = mesh::nodes(&peer).await {
            events.send(AcpAppEvent::MeshNodes(nodes));
        }
    });
}

async fn register_direct_elicitation(
    state: &Arc<RuntimeState>,
    events: &EventSink,
    peer: Peer,
    id: Value,
    request: acp::CreateElicitationRequest,
) {
    let session_id = elicitation::request_session_id(&request);
    notification::flush_assistant(state, events, &session_id).await;
    let elicitation_id = id
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| id.to_string());
    let update = elicitation::requested_update(elicitation_id.clone(), request, "acp-ws");
    state
        .elicitations
        .insert(
            elicitation_id,
            PendingResponse::WebSocketResponse { peer, id },
        )
        .await;
    events.session_update(&session_id, update);
}

async fn register_result_elicitation(
    state: &Arc<RuntimeState>,
    events: &EventSink,
    peer: Peer,
    params: Value,
) {
    let elicitation_id = string_alias(&params, "elicitationId", "elicitation_id")
        .unwrap_or_else(|| "elicitation".to_string());
    let session_id =
        string_alias(&params, "sessionId", "session_id").unwrap_or_else(|| "request".to_string());
    let message = params
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Input requested")
        .to_string();
    let requested_schema = params
        .get("requestedSchema")
        .or_else(|| params.get("requested_schema"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let source = params
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("acp-ws")
        .to_string();
    let querymt = params
        .get("_meta")
        .or_else(|| params.get("meta"))
        .and_then(|meta| meta.get("querymt"));
    let allow_custom = source == "builtin:question"
        || params
            .get("allow_custom")
            .or_else(|| params.get("allowCustom"))
            .or_else(|| querymt.and_then(|meta| meta.get("allow_custom")))
            .or_else(|| querymt.and_then(|meta| meta.get("allowCustom")))
            .and_then(Value::as_bool)
            .unwrap_or(false);
    notification::flush_assistant(state, events, &session_id).await;
    state
        .elicitations
        .insert(
            elicitation_id.clone(),
            PendingResponse::WebSocketResult(peer),
        )
        .await;
    events.session_update(
        &session_id,
        AcpSessionUpdate::ElicitationRequested {
            elicitation_id,
            message,
            requested_schema,
            source,
            allow_custom,
        },
    );
}

fn string_alias(params: &Value, camel: &str, snake: &str) -> Option<String> {
    params
        .get(camel)
        .or_else(|| params.get(snake))
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_routing_documents_shared_and_websocket_only_styles() {
        assert_eq!(
            extension_notification("querymt/session/delegationUpdate"),
            Some(ExtensionNotification::Delegation)
        );
        assert_eq!(
            extension_notification("_querymt/session/delegationUpdate"),
            Some(ExtensionNotification::Delegation)
        );
        assert_eq!(
            extension_notification("querymt/models/changed"),
            Some(ExtensionNotification::ModelsChanged)
        );
        assert_eq!(
            extension_notification("querymt/mesh/joined"),
            Some(ExtensionNotification::MeshChanged)
        );
        assert_eq!(extension_notification("session/update"), None);
    }
}
