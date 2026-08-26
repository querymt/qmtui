use std::future::Future;
use std::sync::{Arc, Mutex};

use agent_client_protocol::{
    self as acp_sdk, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
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
    spawned: Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
}

impl WebSocketConnection {
    fn new(peer: Peer) -> Self {
        Self {
            peer,
            spawned: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn abort_spawned(&self) {
        let handles = std::mem::take(
            &mut *self
                .spawned
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for handle in handles {
            handle.abort();
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
        let handle = tokio::spawn(async move {
            let _ = future.await;
        });
        self.spawned
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(handle.abort_handle());
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

    reader.as_mut().abort();
    writer.as_mut().abort();
    connection.abort_spawned();
    peer.fail_all("connection shutdown").await;
    state.elicitations.clear().await;
    state.assistants.clear().await;
    result
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;

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
