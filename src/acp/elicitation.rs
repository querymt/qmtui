use std::collections::{BTreeMap, HashMap};

use agent_client_protocol::{self as acp_sdk, schema::v1 as acp};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::transport::jsonrpc::Peer;

pub(super) enum PendingResponse {
    Sdk(acp_sdk::Responder<acp::CreateElicitationResponse>),
    WebSocketResult(Peer),
    WebSocketResponse { peer: Peer, id: Value },
}

#[derive(Default)]
pub(super) struct ElicitationRegistry {
    pending: Mutex<HashMap<String, PendingResponse>>,
}

impl ElicitationRegistry {
    pub(super) async fn insert(&self, id: String, response: PendingResponse) {
        self.pending.lock().await.insert(id, response);
    }

    pub(super) async fn respond(&self, id: &str, action: &str, content: Option<Value>) {
        let Some(response) = self.pending.lock().await.remove(id) else {
            return;
        };
        match response {
            PendingResponse::Sdk(responder) => {
                let response = match action {
                    "accept" => {
                        acp::CreateElicitationResponse::new(acp::ElicitationAction::Accept(
                            acp::ElicitationAcceptAction::new()
                                .content(elicitation_content(content)),
                        ))
                    }
                    "decline" => {
                        acp::CreateElicitationResponse::new(acp::ElicitationAction::Decline)
                    }
                    _ => acp::CreateElicitationResponse::new(acp::ElicitationAction::Cancel),
                };
                let _ = responder.respond(response);
            }
            PendingResponse::WebSocketResult(peer) => {
                let _ = peer
                    .request(
                        "elicitation_result",
                        json!({
                            "elicitation_id": id,
                            "action": action,
                            "content": content,
                        }),
                    )
                    .await;
            }
            PendingResponse::WebSocketResponse { peer, id } => {
                let _ = peer.respond(
                    id,
                    Ok(json!({
                        "action": action,
                        "content": content,
                    })),
                );
            }
        }
    }

    pub(super) async fn clear(&self) {
        self.pending.lock().await.clear();
    }

    #[cfg(test)]
    async fn len(&self) -> usize {
        self.pending.lock().await.len()
    }
}

pub(super) fn request_session_id(request: &acp::CreateElicitationRequest) -> String {
    match &request.mode {
        acp::ElicitationMode::Form(form) => scope_session_id(&form.scope),
        acp::ElicitationMode::Url(url) => scope_session_id(&url.scope),
        _ => "request".to_string(),
    }
}

pub(super) fn metadata(
    request: &acp::CreateElicitationRequest,
    default_source: &str,
) -> (String, bool) {
    let querymt = request.meta.as_ref().and_then(|meta| meta.get("querymt"));
    let source = querymt
        .and_then(|meta| meta.get("source"))
        .and_then(Value::as_str)
        .unwrap_or(default_source)
        .to_string();
    let allow_custom = source == "builtin:question"
        || querymt
            .and_then(|meta| meta.get("allow_custom").or_else(|| meta.get("allowCustom")))
            .and_then(Value::as_bool)
            .unwrap_or(false);
    (source, allow_custom)
}

pub(super) fn requested_update(
    id: String,
    request: acp::CreateElicitationRequest,
    default_source: &str,
) -> crate::acp_state::AcpSessionUpdate {
    let (metadata_source, allow_custom) = metadata(&request, default_source);
    let (requested_schema, source) = match &request.mode {
        acp::ElicitationMode::Form(form) => (
            serde_json::to_value(&form.requested_schema).unwrap_or_else(|_| json!({})),
            metadata_source,
        ),
        acp::ElicitationMode::Url(url) => (json!({}), format!("acp-url:{}", url.url)),
        _ => (json!({}), metadata_source),
    };
    crate::acp_state::AcpSessionUpdate::ElicitationRequested {
        elicitation_id: id,
        message: request.message,
        requested_schema,
        source,
        allow_custom,
    }
}

fn scope_session_id(scope: &acp::ElicitationScope) -> String {
    match scope {
        acp::ElicitationScope::Session(session) => session.session_id.to_string(),
        acp::ElicitationScope::Request(_) => "request".to_string(),
        _ => "request".to_string(),
    }
}

fn elicitation_content(
    content: Option<Value>,
) -> Option<BTreeMap<String, acp::ElicitationContentValue>> {
    let object = content?.as_object()?.clone();
    Some(
        object
            .into_iter()
            .filter_map(|(key, value)| json_to_value(value).map(|value| (key, value)))
            .collect(),
    )
}

fn json_to_value(value: Value) -> Option<acp::ElicitationContentValue> {
    match value {
        Value::String(value) => Some(acp::ElicitationContentValue::String(value)),
        Value::Bool(value) => Some(acp::ElicitationContentValue::Boolean(value)),
        Value::Number(value) => value
            .as_i64()
            .map(acp::ElicitationContentValue::Integer)
            .or_else(|| value.as_f64().map(acp::ElicitationContentValue::Number)),
        Value::Array(values) => Some(acp::ElicitationContentValue::StringArray(
            values
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
        )),
        _ => None,
    }
}

pub(super) fn permission_response(
    request: &acp::RequestPermissionRequest,
) -> acp::RequestPermissionResponse {
    let allow = request
        .options
        .iter()
        .find(|option| matches!(option.kind, acp::PermissionOptionKind::AllowOnce))
        .or_else(|| request.options.first());
    match allow {
        Some(option) => {
            acp::RequestPermissionResponse::new(acp::RequestPermissionOutcome::Selected(
                acp::SelectedPermissionOutcome::new(option.option_id.clone()),
            ))
        }
        None => acp::RequestPermissionResponse::new(acp::RequestPermissionOutcome::Cancelled),
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;
    use tokio_tungstenite::tungstenite::Message;

    use super::*;

    fn form_request() -> acp::CreateElicitationRequest {
        acp::CreateElicitationRequest::new(
            acp::ElicitationFormMode::new(
                acp::ElicitationSessionScope::new("session-1"),
                acp::ElicitationSchema::new().string("selection", true),
            ),
            "choose",
        )
        .meta(serde_json::Map::from_iter([(
            "querymt".to_string(),
            json!({ "source": "builtin:question" }),
        )]))
    }

    #[test]
    fn request_normalization_preserves_session_source_schema_and_custom_policy() {
        let request = form_request();
        assert_eq!(request_session_id(&request), "session-1");
        assert_eq!(metadata(&request, "acp"), ("builtin:question".into(), true));
        assert!(matches!(
            requested_update("e1".into(), request, "acp"),
            crate::acp_state::AcpSessionUpdate::ElicitationRequested {
                elicitation_id,
                source,
                allow_custom: true,
                ..
            } if elicitation_id == "e1" && source == "builtin:question"
        ));
    }

    #[tokio::test]
    async fn websocket_direct_response_is_correlated_once_with_action_and_content() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let registry = ElicitationRegistry::default();
        registry
            .insert(
                "e1".into(),
                PendingResponse::WebSocketResponse {
                    peer: Peer::new(tx),
                    id: json!(7),
                },
            )
            .await;
        registry
            .respond("e1", "accept", Some(json!({ "selection": "yes" })))
            .await;
        registry.respond("e1", "decline", None).await;
        let Message::Text(text) = rx.recv().await.expect("response") else {
            panic!("text response");
        };
        let value: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(value["id"], 7);
        assert_eq!(value["result"]["action"], "accept");
        assert_eq!(value["result"]["content"]["selection"], "yes");
        assert!(rx.try_recv().is_err());
        assert_eq!(registry.len().await, 0);
    }

    #[tokio::test]
    async fn websocket_result_method_is_correlated_once() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let registry = ElicitationRegistry::default();
        let peer = Peer::new(tx);
        registry
            .insert("e2".into(), PendingResponse::WebSocketResult(peer.clone()))
            .await;
        let response_peer = peer.clone();
        let responder = tokio::spawn(async move {
            let Message::Text(text) = rx.recv().await.expect("request") else {
                panic!("text request");
            };
            let envelope: crate::acp::transport::jsonrpc::Envelope =
                serde_json::from_str(&text).expect("json");
            assert_eq!(envelope.method.as_deref(), Some("elicitation_result"));
            assert_eq!(envelope.params["elicitation_id"], "e2");
            assert_eq!(envelope.params["action"], "decline");
            response_peer
                .resolve(crate::acp::transport::jsonrpc::Envelope {
                    jsonrpc: "2.0".into(),
                    id: envelope.id,
                    method: None,
                    params: Value::Null,
                    result: Some(json!({})),
                    error: None,
                })
                .await;
        });
        registry.respond("e2", "decline", None).await;
        responder.await.expect("responder");
        assert_eq!(registry.len().await, 0);
    }
}
