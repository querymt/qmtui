use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use agent_client_protocol as acp_sdk;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;

use super::super::connection::internal_error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Envelope {
    pub(super) jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) method: Option<String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub(super) params: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<Value>,
}

impl Envelope {
    pub(super) fn request(id: i64, method: &str, params: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(id)),
            method: Some(method.to_string()),
            params,
            result: None,
            error: None,
        }
    }

    pub(super) fn notification(method: &str, params: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: Some(method.to_string()),
            params,
            result: None,
            error: None,
        }
    }

    pub(super) fn response(id: Value, result: Result<Value, acp_sdk::Error>) -> Self {
        let (result, error) = match result {
            Ok(value) => (Some(value), None),
            Err(err) => (
                None,
                Some(serde_json::to_value(err).unwrap_or_else(|_| {
                    json!({
                        "code": -32603,
                        "message": "internal error"
                    })
                })),
            ),
        };
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: None,
            params: Value::Null,
            result,
            error,
        }
    }
}

struct PendingRequest {
    method: String,
    tx: oneshot::Sender<Result<Value, acp_sdk::Error>>,
}

#[derive(Clone)]
pub(super) struct Peer {
    tx: mpsc::UnboundedSender<Message>,
    pending: Arc<Mutex<HashMap<i64, PendingRequest>>>,
    next_id: Arc<AtomicI64>,
}

impl Peer {
    pub(super) fn new(tx: mpsc::UnboundedSender<Message>) -> Self {
        Self {
            tx,
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicI64::new(1)),
        }
    }

    pub(super) async fn request(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Value, acp_sdk::Error> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(
            id,
            PendingRequest {
                method: method.to_string(),
                tx,
            },
        );
        if let Err(err) = self.send(Envelope::request(id, method, params)) {
            self.pending.lock().await.remove(&id);
            return Err(err);
        }
        rx.await
            .map_err(|_| internal_error(format!("acp websocket request dropped: {method}")))?
    }

    pub(super) fn notify(&self, method: &str, params: Value) -> Result<(), acp_sdk::Error> {
        self.send(Envelope::notification(method, params))
    }

    pub(super) fn respond(
        &self,
        id: Value,
        result: Result<Value, acp_sdk::Error>,
    ) -> Result<(), acp_sdk::Error> {
        self.send(Envelope::response(id, result))
    }

    fn send(&self, envelope: Envelope) -> Result<(), acp_sdk::Error> {
        let text = serde_json::to_string(&envelope).map_err(acp_sdk::Error::into_internal_error)?;
        self.tx
            .send(Message::Text(text.into()))
            .map_err(|err| internal_error(format!("acp websocket send failed: {err}")))
    }

    pub(super) async fn resolve(&self, envelope: Envelope) {
        let Some(id) = envelope.id.and_then(|id| id.as_i64()) else {
            return;
        };
        let Some(pending) = self.pending.lock().await.remove(&id) else {
            return;
        };
        let result = if let Some(error) = envelope.error {
            Err(internal_error(format!(
                "acp websocket {} failed: {error}",
                pending.method
            )))
        } else {
            Ok(envelope.result.unwrap_or(Value::Null))
        };
        let _ = pending.tx.send(result);
    }

    pub(super) async fn fail_all(&self, reason: &str) {
        let pending = std::mem::take(&mut *self.pending.lock().await);
        for (_, request) in pending {
            let _ = request.tx.send(Err(internal_error(format!(
                "acp websocket {} failed: {reason}",
                request.method
            ))));
        }
    }

    #[cfg(test)]
    async fn pending_len(&self) -> usize {
        self.pending.lock().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn request_with_wire(
        peer: &Peer,
        rx: &mut mpsc::UnboundedReceiver<Message>,
        method: &'static str,
    ) -> (
        tokio::task::JoinHandle<Result<Value, acp_sdk::Error>>,
        Envelope,
    ) {
        let request_peer = peer.clone();
        let task = tokio::spawn(async move { request_peer.request(method, json!({})).await });
        let message = rx.recv().await.expect("request frame");
        let Message::Text(text) = message else {
            panic!("expected text frame");
        };
        let envelope = serde_json::from_str(&text).expect("request envelope");
        (task, envelope)
    }

    #[tokio::test]
    async fn ids_start_at_one_and_out_of_order_responses_correlate() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let peer = Peer::new(tx);
        let (first, first_wire) = request_with_wire(&peer, &mut rx, "first").await;
        let (second, second_wire) = request_with_wire(&peer, &mut rx, "second").await;
        assert_eq!(first_wire.id, Some(json!(1)));
        assert_eq!(second_wire.id, Some(json!(2)));

        peer.resolve(Envelope {
            result: Some(json!("second-result")),
            method: None,
            params: Value::Null,
            error: None,
            ..second_wire
        })
        .await;
        peer.resolve(Envelope {
            result: Some(json!("first-result")),
            method: None,
            params: Value::Null,
            error: None,
            ..first_wire
        })
        .await;

        assert_eq!(
            first.await.expect("first task").expect("first result"),
            "first-result"
        );
        assert_eq!(
            second.await.expect("second task").expect("second result"),
            "second-result"
        );
        assert_eq!(peer.pending_len().await, 0);
    }

    #[tokio::test]
    async fn errors_precede_results_and_include_the_method() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let peer = Peer::new(tx);
        let (task, wire) = request_with_wire(&peer, &mut rx, "querymt/models").await;
        peer.resolve(Envelope {
            result: Some(json!("ignored")),
            error: Some(json!({ "code": -1, "message": "nope" })),
            method: None,
            params: Value::Null,
            ..wire
        })
        .await;
        let error = task
            .await
            .expect("task")
            .expect_err("request error")
            .to_string();
        assert!(error.contains("querymt/models"));
        assert!(error.contains("nope"));
    }

    #[tokio::test]
    async fn missing_result_is_null_and_duplicate_unknown_late_ids_are_ignored() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let peer = Peer::new(tx);
        let (task, wire) = request_with_wire(&peer, &mut rx, "empty").await;
        let response = Envelope {
            method: None,
            params: Value::Null,
            result: None,
            error: None,
            ..wire.clone()
        };
        peer.resolve(Envelope {
            id: Some(json!(999)),
            ..response.clone()
        })
        .await;
        peer.resolve(response.clone()).await;
        peer.resolve(response).await;
        peer.resolve(Envelope {
            jsonrpc: "2.0".into(),
            id: Some(json!("nonnumeric")),
            method: None,
            params: Value::Null,
            result: Some(json!(1)),
            error: None,
        })
        .await;
        assert_eq!(task.await.expect("task").expect("result"), Value::Null);
        assert_eq!(peer.pending_len().await, 0);
    }

    #[tokio::test]
    async fn send_failure_unregisters_the_request() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let peer = Peer::new(tx);
        let error = peer
            .request("closed", json!({}))
            .await
            .expect_err("send failure");
        assert!(error.to_string().contains("send failed"));
        assert_eq!(peer.pending_len().await, 0);
    }

    #[tokio::test]
    async fn fail_all_drains_each_pending_request_once_and_late_responses_are_ignored() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let peer = Peer::new(tx);
        let (first, first_wire) = request_with_wire(&peer, &mut rx, "first").await;
        let (second, _) = request_with_wire(&peer, &mut rx, "second").await;
        peer.fail_all("socket closed").await;
        peer.fail_all("duplicate close").await;
        peer.resolve(Envelope {
            method: None,
            params: Value::Null,
            result: Some(json!("late")),
            error: None,
            ..first_wire
        })
        .await;

        let first_error = first
            .await
            .expect("first task")
            .expect_err("first error")
            .to_string();
        let second_error = second
            .await
            .expect("second task")
            .expect_err("second error")
            .to_string();
        assert!(first_error.contains("first"));
        assert!(first_error.contains("socket closed"));
        assert!(second_error.contains("second"));
        assert_eq!(peer.pending_len().await, 0);
    }
}
