use std::sync::Arc;

use agent_client_protocol::{self as acp_sdk, JsonRpcMessage, schema::v1 as acp};
use serde_json::{Value, json};

use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};

use super::connection::AcpConnection;
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
    let version = params.get("version").and_then(Value::as_u64);
    match delegation::from_value(params) {
        Ok(Some(update)) => events.send(AcpAppEvent::DelegationUpdate(update)),
        Ok(None) => events.info(
            "delegation",
            version.map_or_else(
                || "ignored delegation notification version".to_string(),
                |version| format!("ignored delegation notification version {version}"),
            ),
        ),
        Err(err) => events.info(
            "delegation",
            format!("invalid delegation notification: {err}"),
        ),
    }
}

pub(super) async fn websocket_text<C: AcpConnection>(
    peer: &Peer,
    connection: &C,
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
                spawn_model_refresh(connection.clone(), state.clone(), events.clone())
            }
            Some(ExtensionNotification::MeshChanged) => {
                spawn_mesh_refresh(connection.clone(), events.clone())
            }
            None => {}
        },
    }
    Ok(())
}

fn spawn_model_refresh<C: AcpConnection>(
    connection: C,
    state: Arc<RuntimeState>,
    events: EventSink,
) {
    let task_connection = connection.clone();
    let _ = connection.spawn(async move {
        if let Ok(response) = models::list(&task_connection, false).await {
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
        Ok(())
    });
}

fn spawn_mesh_refresh<C: AcpConnection>(connection: C, events: EventSink) {
    let task_connection = connection.clone();
    let _ = connection.spawn(async move {
        if let Ok(Some(nodes)) = mesh::nodes(&task_connection).await {
            events.send(AcpAppEvent::MeshNodes(nodes));
        }
        Ok(())
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
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::Mutex;

    use agent_client_protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
    use tokio::sync::mpsc;
    use tokio_tungstenite::tungstenite::Message;

    use super::*;
    use crate::runtime_events::ServerChannelMsg;

    #[derive(Clone, Default)]
    struct TestConnection {
        messages: Arc<Mutex<Vec<(String, Value)>>>,
        responses: Arc<Mutex<VecDeque<Value>>>,
    }

    impl TestConnection {
        fn with_responses(responses: impl IntoIterator<Item = Value>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(responses.into_iter().collect())),
                ..Self::default()
            }
        }

        fn messages(&self) -> Vec<(String, Value)> {
            self.messages
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl AcpConnection for TestConnection {
        async fn request<R>(&self, request: R) -> Result<R::Response, acp_sdk::Error>
        where
            R: JsonRpcRequest + Send + Sync + 'static,
            R::Response: Send + 'static,
        {
            let message = request.to_untyped_message()?;
            self.messages
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((message.method.clone(), message.params));
            let response = self
                .responses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop_front()
                .unwrap_or_else(|| json!({}));
            R::Response::from_value(&message.method, response)
        }

        fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
        where
            N: JsonRpcNotification + Send + Sync + 'static,
        {
            let message = notification.to_untyped_message()?;
            self.messages
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((message.method, message.params));
            Ok(())
        }

        fn spawn(
            &self,
            future: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
        ) -> Result<(), acp_sdk::Error> {
            tokio::spawn(async move {
                let _ = future.await;
            });
            Ok(())
        }
    }

    fn websocket_harness() -> (
        Peer,
        mpsc::UnboundedReceiver<Message>,
        Arc<RuntimeState>,
        EventSink,
        mpsc::UnboundedReceiver<ServerChannelMsg>,
    ) {
        let (wire_tx, wire_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        (
            Peer::new(wire_tx),
            wire_rx,
            Arc::new(RuntimeState::new(None)),
            EventSink::new(event_tx),
            event_rx,
        )
    }

    fn envelope(method: &str, id: Option<Value>, params: Value) -> String {
        serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .expect("envelope JSON")
    }

    fn delegation_value(version: u32) -> Value {
        json!({
            "version": version,
            "sessionId": "parent",
            "delegationId": "d1",
            "state": "requested",
            "targetAgentId": "coder",
            "objective": "implement",
            "requestedAt": 1,
            "updatedAt": 1
        })
    }

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
        for method in [
            "querymt/mesh/nodesChanged",
            "querymt/mesh/joined",
            "querymt/mesh/peerExpired",
        ] {
            assert_eq!(
                extension_notification(method),
                Some(ExtensionNotification::MeshChanged)
            );
        }
        assert_eq!(extension_notification("session/update"), None);
    }

    #[test]
    fn delegation_notification_emits_supported_domain_update() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        delegation_notification(&EventSink::new(tx), delegation_value(1));
        assert!(matches!(
            rx.try_recv().expect("delegation event"),
            crate::runtime_events::ServerChannelMsg::Acp(AcpAppEvent::DelegationUpdate(update))
                if update.delegation_id == "d1"
        ));
    }

    #[test]
    fn delegation_notification_logs_exact_unsupported_version() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        delegation_notification(&EventSink::new(tx), delegation_value(2));
        assert!(matches!(
            rx.try_recv().expect("delegation log"),
            crate::runtime_events::ServerChannelMsg::Acp(AcpAppEvent::InfoLog { target, message })
                if target == "delegation"
                    && message == "ignored delegation notification version 2"
        ));
    }

    #[test]
    fn delegation_notification_logs_malformed_payload() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        delegation_notification(&EventSink::new(tx), json!({ "version": 1 }));
        assert!(matches!(
            rx.try_recv().expect("delegation log"),
            crate::runtime_events::ServerChannelMsg::Acp(AcpAppEvent::InfoLog { target, message })
                if target == "delegation"
                    && message.starts_with("invalid delegation notification: ")
        ));
    }

    #[tokio::test]
    async fn mesh_aliases_route_to_the_same_refresh_request_and_event() {
        for method in [
            "querymt/mesh/nodesChanged",
            "querymt/mesh/joined",
            "querymt/mesh/peerExpired",
        ] {
            let connection = TestConnection::with_responses([json!({
                "nodes": [{ "id": "node-1", "label": "Remote" }]
            })]);
            let (peer, _wire_rx, state, events, mut event_rx) = websocket_harness();
            websocket_text(
                &peer,
                &connection,
                &state,
                &events,
                &envelope(method, None, json!({})),
            )
            .await
            .expect("mesh notification");
            tokio::task::yield_now().await;

            assert_eq!(
                connection.messages(),
                vec![("_querymt/mesh/nodes".into(), json!({}))]
            );
            assert!(matches!(
                event_rx.try_recv().expect("mesh event"),
                ServerChannelMsg::Acp(AcpAppEvent::MeshNodes(nodes))
                    if nodes.nodes[0].id == "node-1"
            ));
            assert!(event_rx.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn malformed_known_input_errors_while_unknown_methods_are_ignored() {
        let connection = TestConnection::default();
        let (peer, mut wire_rx, state, events, mut event_rx) = websocket_harness();
        assert!(
            websocket_text(&peer, &connection, &state, &events, "{")
                .await
                .is_err()
        );
        assert!(
            websocket_text(
                &peer,
                &connection,
                &state,
                &events,
                &envelope("session/update", None, json!({ "sessionId": 7 })),
            )
            .await
            .is_err()
        );
        websocket_text(
            &peer,
            &connection,
            &state,
            &events,
            &envelope("querymt/unknown", Some(json!(9)), json!({ "bad": true })),
        )
        .await
        .expect("unknown ignored");

        assert!(connection.messages().is_empty());
        assert!(wire_rx.try_recv().is_err());
        assert!(event_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn response_envelopes_resolve_the_matching_peer_request() {
        let connection = TestConnection::default();
        let (peer, mut wire_rx, state, events, _event_rx) = websocket_harness();
        let request_peer = peer.clone();
        let pending = tokio::spawn(async move {
            request_peer
                .request("querymt/test", json!({ "value": 1 }))
                .await
        });
        let Message::Text(text) = wire_rx.recv().await.expect("request frame") else {
            panic!("text request");
        };
        let request: Value = serde_json::from_str(&text).expect("request JSON");

        websocket_text(
            &peer,
            &connection,
            &state,
            &events,
            &serde_json::to_string(&json!({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": { "ok": true }
            }))
            .expect("response JSON"),
        )
        .await
        .expect("response dispatch");
        assert_eq!(
            pending
                .await
                .expect("request task")
                .expect("request result"),
            json!({ "ok": true })
        );
    }

    #[tokio::test]
    async fn permission_request_responds_with_allow_once_wire_shape() {
        let connection = TestConnection::default();
        let (peer, mut wire_rx, state, events, mut event_rx) = websocket_harness();
        let request = acp::RequestPermissionRequest::new(
            "session-1",
            acp::ToolCallUpdate::new("tool-1", acp::ToolCallUpdateFields::new()),
            vec![
                acp::PermissionOption::new(
                    "reject",
                    "Reject",
                    acp::PermissionOptionKind::RejectOnce,
                ),
                acp::PermissionOption::new("allow", "Allow", acp::PermissionOptionKind::AllowOnce),
            ],
        );
        websocket_text(
            &peer,
            &connection,
            &state,
            &events,
            &envelope(
                "session/request_permission",
                Some(json!(7)),
                serde_json::to_value(request).expect("permission params"),
            ),
        )
        .await
        .expect("permission request");

        let Message::Text(text) = wire_rx.try_recv().expect("permission response") else {
            panic!("text response");
        };
        assert_eq!(
            serde_json::from_str::<Value>(&text).expect("response JSON"),
            json!({
                "jsonrpc": "2.0",
                "id": 7,
                "result": { "outcome": { "outcome": "selected", "optionId": "allow" } }
            })
        );
        assert!(event_rx.try_recv().is_err());
    }

    fn direct_elicitation() -> acp::CreateElicitationRequest {
        acp::CreateElicitationRequest::new(
            acp::ElicitationFormMode::new(
                acp::ElicitationSessionScope::new("session-1"),
                acp::ElicitationSchema::new().string("selection", true),
            ),
            "Choose",
        )
        .meta(serde_json::Map::from_iter([(
            "querymt".to_string(),
            json!({ "source": "test", "allow_custom": true }),
        )]))
    }

    #[tokio::test]
    async fn direct_elicitation_registers_and_dispatches_exact_response() {
        let connection = TestConnection::default();
        let (peer, mut wire_rx, state, events, mut event_rx) = websocket_harness();
        websocket_text(
            &peer,
            &connection,
            &state,
            &events,
            &envelope(
                "elicitation/create",
                Some(json!("e-direct")),
                serde_json::to_value(direct_elicitation()).expect("elicitation params"),
            ),
        )
        .await
        .expect("direct elicitation");
        assert!(matches!(
            event_rx.try_recv().expect("elicitation event"),
            ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate {
                session_id,
                update: AcpSessionUpdate::ElicitationRequested {
                    elicitation_id,
                    message,
                    source,
                    allow_custom: true,
                    ..
                },
                is_replay: false,
            }) if session_id == "session-1"
                && elicitation_id == "e-direct"
                && message == "Choose"
                && source == "test"
        ));

        state
            .elicitations
            .respond("e-direct", "accept", Some(json!({ "selection": "yes" })))
            .await;
        let Message::Text(text) = wire_rx.try_recv().expect("elicitation response") else {
            panic!("text response");
        };
        assert_eq!(
            serde_json::from_str::<Value>(&text).expect("response JSON"),
            json!({
                "jsonrpc": "2.0",
                "id": "e-direct",
                "result": { "action": "accept", "content": { "selection": "yes" } }
            })
        );
    }

    #[tokio::test]
    async fn requested_elicitation_accepts_aliases_and_exact_custom_metadata() {
        for params in [
            json!({
                "elicitationId": "e-camel",
                "sessionId": "session-camel",
                "message": "Camel",
                "requestedSchema": { "type": "object" },
                "source": "extension",
                "allowCustom": true
            }),
            json!({
                "elicitation_id": "e-snake",
                "session_id": "session-snake",
                "message": "Snake",
                "requested_schema": { "type": "string" },
                "source": "extension",
                "_meta": { "querymt": { "allow_custom": true } }
            }),
        ] {
            let expected_id = params
                .get("elicitationId")
                .or_else(|| params.get("elicitation_id"))
                .and_then(Value::as_str)
                .expect("elicitation id")
                .to_string();
            let expected_session = params
                .get("sessionId")
                .or_else(|| params.get("session_id"))
                .and_then(Value::as_str)
                .expect("session id")
                .to_string();
            let expected_schema = params
                .get("requestedSchema")
                .or_else(|| params.get("requested_schema"))
                .cloned()
                .expect("schema");
            let connection = TestConnection::default();
            let (peer, _wire_rx, state, events, mut event_rx) = websocket_harness();
            websocket_text(
                &peer,
                &connection,
                &state,
                &events,
                &envelope("elicitation/requested", None, params),
            )
            .await
            .expect("requested elicitation");

            assert!(matches!(
                event_rx.try_recv().expect("elicitation event"),
                ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate {
                    session_id,
                    update: AcpSessionUpdate::ElicitationRequested {
                        elicitation_id,
                        requested_schema,
                        source,
                        allow_custom: true,
                        ..
                    },
                    is_replay: false,
                }) if session_id == expected_session
                    && elicitation_id == expected_id
                    && requested_schema == expected_schema
                    && source == "extension"
            ));
        }
    }
}
