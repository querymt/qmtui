use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::{
    self as acp_sdk, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::app::ConnectionEvent;
use crate::command::Command;
use crate::runtime_events::{ConnectionManagerEvent, ServerChannelMsg};

use super::super::commands;
use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::events::EventSink;
use super::super::inbound;
use super::super::runtime::RuntimeState;
use super::jsonrpc::Peer;

#[derive(Clone)]
struct WebSocketConnection {
    peer: Peer,
    spawned: Arc<Mutex<Option<JoinSet<()>>>>,
}

impl WebSocketConnection {
    fn new(peer: Peer) -> Self {
        Self {
            peer,
            spawned: Arc::new(Mutex::new(Some(JoinSet::new()))),
        }
    }

    #[cfg(test)]
    fn abort_spawned(&self) {
        if let Some(spawned) = self
            .spawned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            spawned.abort_all();
        }
    }

    async fn shutdown(&self, reason: &str) {
        self.peer.fail_all(reason).await;
        let mut spawned = self
            .spawned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .unwrap_or_default();
        let drained = tokio::time::timeout(Duration::from_secs(1), async {
            while spawned.join_next().await.is_some() {}
        })
        .await;
        if drained.is_err() {
            spawned.abort_all();
            while spawned.join_next().await.is_some() {}
        }
    }
}

impl AcpConnection for WebSocketConnection {
    async fn request<R>(&self, request: R) -> Result<R::Response, acp_sdk::Error>
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static,
    {
        let message = request.to_untyped_message()?;
        let method = message.method.clone();
        let result = self.peer.request(&message.method, message.params).await?;
        R::Response::from_value(&method, result)
    }

    fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
    where
        N: JsonRpcNotification + Send + Sync + 'static,
    {
        let message = notification.to_untyped_message()?;
        self.peer.notify(&message.method, message.params)
    }

    fn spawn(
        &self,
        future: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
    ) -> Result<(), acp_sdk::Error> {
        if let Some(spawned) = self
            .spawned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            spawned.spawn(async move {
                let _ = future.await;
            });
        }
        Ok(())
    }
}

pub(in crate::acp) async fn run(
    url: String,
    cmd_rx: &mut mpsc::UnboundedReceiver<Command>,
    srv_tx: mpsc::UnboundedSender<ServerChannelMsg>,
    conn_tx: mpsc::UnboundedSender<ConnectionManagerEvent>,
    launch_cwd: Option<String>,
) -> Result<(), acp_sdk::Error> {
    let state = Arc::new(RuntimeState::new(launch_cwd));
    let events = EventSink::new(srv_tx);
    let (socket, _) = connect_async(&url)
        .await
        .map_err(acp_sdk::Error::into_internal_error)?;
    let (mut socket_write, mut socket_read) = socket.split();
    let (write_tx, mut write_rx) = mpsc::unbounded_channel::<Message>();
    let peer = Peer::new(write_tx);
    let connection = WebSocketConnection::new(peer.clone());

    let writer_peer = peer.clone();
    let writer = tokio::spawn(async move {
        while let Some(message) = write_rx.recv().await {
            if let Err(err) = socket_write.send(message).await {
                let reason = format!("writer failed: {err}");
                writer_peer.fail_all(&reason).await;
                return Err(reason);
            }
        }
        writer_peer.fail_all("writer channel closed").await;
        Ok(())
    });

    let reader_peer = peer.clone();
    let reader_connection = connection.clone();
    let reader_state = state.clone();
    let reader_events = events.clone();
    let reader = tokio::spawn(async move {
        let mut reason = "socket eof".to_string();
        while let Some(message) = socket_read.next().await {
            match message {
                Ok(Message::Text(text)) => {
                    if let Err(err) = inbound::websocket_text(
                        &reader_peer,
                        &reader_connection,
                        &reader_state,
                        &reader_events,
                        text.as_ref(),
                    )
                    .await
                    {
                        reader_events.error(format!("ACP WebSocket message failed: {err:?}"));
                    }
                }
                Ok(Message::Close(frame)) => {
                    reason = frame
                        .map(|frame| format!("socket closed: {}", frame.reason))
                        .unwrap_or_else(|| "socket closed".to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => {
                    reason = format!("read failed: {err}");
                    reader_events.error(format!("ACP WebSocket read failed: {err}"));
                    break;
                }
            }
        }
        reader_peer.fail_all(&reason).await;
        reason
    });

    let _ = conn_tx.send(ConnectionManagerEvent::State(ConnectionEvent::Connected));
    tokio::pin!(reader);
    tokio::pin!(writer);
    let result = loop {
        tokio::select! {
            command = cmd_rx.recv() => {
                let Some(command) = command else {
                    break Ok(());
                };
                let context = CommandContext {
                    connection: &connection,
                    state: &state,
                    events: &events,
                };
                if let Err(err) = commands::dispatch(context, command).await {
                    events.error(format!("ACP request failed: {err:?}"));
                }
            }
            reader_result = &mut reader => {
                let reason = reader_result.unwrap_or_else(|err| err.to_string());
                break Err(super::super::connection::internal_error(format!(
                    "ACP WebSocket connection closed: {reason}"
                )));
            }
            writer_result = &mut writer => {
                let reason = writer_result
                    .map(|result| result.err().unwrap_or_else(|| "writer ended".to_string()))
                    .unwrap_or_else(|err| err.to_string());
                peer.fail_all(&reason).await;
                break Err(super::super::connection::internal_error(format!(
                    "ACP WebSocket connection closed: {reason}"
                )));
            }
        }
    };

    connection.shutdown("connection shutdown").await;
    reader.as_mut().abort();
    writer.as_mut().abort();
    state.elicitations.clear().await;
    state.assistants.clear().await;
    result
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_tungstenite::accept_async;

    use crate::acp_state::AcpAppEvent;
    use crate::command::PromptBlock;

    use super::*;

    #[tokio::test]
    async fn shutdown_publishes_prompt_failure_before_returning() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind WebSocket listener");
        let url = format!("ws://{}", listener.local_addr().expect("listener address"));
        let (prompt_seen_tx, prompt_seen_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept WebSocket client");
            let mut socket = accept_async(stream).await.expect("accept WebSocket");

            let message = socket
                .next()
                .await
                .expect("new-session request")
                .expect("valid new-session message");
            let Message::Text(text) = message else {
                panic!("expected text new-session request");
            };
            let request: serde_json::Value =
                serde_json::from_str(text.as_ref()).expect("new-session JSON");
            assert_eq!(request["method"], "session/new");
            socket
                .send(Message::Text(
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": { "sessionId": "session-1" }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .expect("send new-session response");

            let message = socket
                .next()
                .await
                .expect("prompt request")
                .expect("valid prompt message");
            let Message::Text(text) = message else {
                panic!("expected text prompt request");
            };
            let request: serde_json::Value =
                serde_json::from_str(text.as_ref()).expect("prompt JSON");
            assert_eq!(request["method"], "session/prompt");
            prompt_seen_tx.send(()).expect("signal pending prompt");
            std::future::pending::<()>().await;
        });

        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        let (srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (conn_tx, _conn_rx) = mpsc::unbounded_channel();
        let mut connection_task =
            tokio::spawn(async move { run(url, &mut cmd_rx, srv_tx, conn_tx, None).await });
        cmd_tx
            .send(Command::NewSession {
                cwd: None,
                profile_id: None,
            })
            .expect("send new-session command");
        cmd_tx
            .send(Command::Prompt {
                prompt: vec![PromptBlock::Text {
                    text: "hello".to_string(),
                }],
                local_id: "local-1".to_string(),
            })
            .expect("send prompt command");
        tokio::time::timeout(Duration::from_secs(1), prompt_seen_rx)
            .await
            .expect("prompt request reached WebSocket")
            .expect("prompt signal sender");

        drop(cmd_tx);
        let prompt_message = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                tokio::select! {
                    biased;
                    message = srv_rx.recv() => {
                        if let Some(ServerChannelMsg::Acp(AcpAppEvent::PromptFailed {
                            local_id,
                            message,
                        })) = message
                        {
                            break (local_id, message);
                        }
                    }
                    result = &mut connection_task => {
                        panic!("WebSocket teardown returned before PromptFailed: {result:?}");
                    }
                }
            }
        })
        .await
        .expect("prompt failure published during teardown");
        assert_eq!(prompt_message.0, "local-1");
        assert!(prompt_message.1.contains("connection shutdown"));
        connection_task
            .await
            .expect("WebSocket connection task")
            .expect("command-channel shutdown");

        let duplicate_failures = std::iter::from_fn(|| srv_rx.try_recv().ok())
            .filter(|message| {
                matches!(
                    message,
                    ServerChannelMsg::Acp(AcpAppEvent::PromptFailed { .. })
                )
            })
            .count();
        assert_eq!(duplicate_failures, 0);
        server.abort();
    }

    #[tokio::test]
    async fn socket_close_fails_pending_prompt_once_before_connection_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind WebSocket listener");
        let url = format!("ws://{}", listener.local_addr().expect("listener address"));
        let (prompt_seen_tx, prompt_seen_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept WebSocket client");
            let mut socket = accept_async(stream).await.expect("accept WebSocket");

            let Message::Text(text) = socket
                .next()
                .await
                .expect("new-session request")
                .expect("valid new-session message")
            else {
                panic!("text new-session request");
            };
            let request: serde_json::Value = serde_json::from_str(&text).expect("request JSON");
            socket
                .send(Message::Text(
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": { "sessionId": "session-1" }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .expect("send new-session response");

            let Message::Text(text) = socket
                .next()
                .await
                .expect("prompt request")
                .expect("valid prompt message")
            else {
                panic!("text prompt request");
            };
            let request: serde_json::Value = serde_json::from_str(&text).expect("prompt JSON");
            assert_eq!(request["method"], "session/prompt");
            prompt_seen_tx.send(()).expect("signal pending prompt");
            socket.close(None).await.expect("close socket");
        });

        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        let (srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (conn_tx, _conn_rx) = mpsc::unbounded_channel();
        let connection_task =
            tokio::spawn(async move { run(url, &mut cmd_rx, srv_tx, conn_tx, None).await });
        cmd_tx
            .send(Command::NewSession {
                cwd: None,
                profile_id: None,
            })
            .expect("send new-session command");
        cmd_tx
            .send(Command::Prompt {
                prompt: vec![PromptBlock::Text {
                    text: "hello".to_string(),
                }],
                local_id: "local-close".to_string(),
            })
            .expect("send prompt command");
        tokio::time::timeout(Duration::from_secs(1), prompt_seen_rx)
            .await
            .expect("prompt reached server")
            .expect("prompt signal");

        let error = tokio::time::timeout(Duration::from_secs(1), connection_task)
            .await
            .expect("connection returned")
            .expect("connection task")
            .expect_err("socket closure error");
        assert!(error.to_string().contains("connection closed"));
        let failures = std::iter::from_fn(|| srv_rx.try_recv().ok())
            .filter_map(|message| match message {
                ServerChannelMsg::Acp(AcpAppEvent::PromptFailed { local_id, message }) => {
                    Some((local_id, message))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "local-close");
        assert!(failures[0].1.contains("socket closed"));
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn abort_spawned_cancels_owned_prompt_work() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let connection = WebSocketConnection::new(Peer::new(tx));
        let dropped = Arc::new(AtomicBool::new(false));
        let task_dropped = dropped.clone();
        connection
            .spawn(async move {
                struct DropFlag(Arc<AtomicBool>);
                impl Drop for DropFlag {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::SeqCst);
                    }
                }
                let _flag = DropFlag(task_dropped);
                std::future::pending::<()>().await;
                Ok(())
            })
            .expect("spawn owned task");
        tokio::task::yield_now().await;

        connection.abort_spawned();

        tokio::time::timeout(Duration::from_secs(1), async {
            while !dropped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("owned task aborted");
    }
}
