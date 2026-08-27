use tokio::sync::mpsc;

use crate::{
    acp::{self, AcpEndpoint},
    app::ConnectionEvent,
    command::Command,
    server_manager::ServerEvent,
};

use super::{ConnectionManagerEvent, ServerChannelMsg};

pub(super) async fn connection_manager(
    endpoint: AcpEndpoint,
    srv_tx: mpsc::UnboundedSender<ServerChannelMsg>,
    mut cmd_rx: mpsc::UnboundedReceiver<Command>,
    conn_tx: mpsc::UnboundedSender<ConnectionManagerEvent>,
    sup_event_tx: mpsc::UnboundedSender<ServerEvent>,
    launch_cwd: Option<String>,
) {
    match endpoint {
        AcpEndpoint::Stdio { argv } => {
            let _ = sup_event_tx.send(ServerEvent::Starting);
            let endpoint = AcpEndpoint::Stdio { argv };
            if let Err(err) = endpoint.validate() {
                let message = format!("invalid ACP stdio command: {err:?}");
                let _ = sup_event_tx.send(ServerEvent::StartFailed {
                    error: message.clone(),
                });
                let _ = conn_tx.send(ConnectionManagerEvent::State(
                    ConnectionEvent::Disconnected { reason: message },
                ));
                return;
            }
            let _ = sup_event_tx.send(ServerEvent::Started);
            let result = acp::run(endpoint, &mut cmd_rx, srv_tx, conn_tx.clone(), launch_cwd).await;

            match result {
                Ok(()) => {
                    let reason = "ACP stdio connection ended".to_string();
                    let _ = sup_event_tx.send(ServerEvent::Stopped {
                        reason: reason.clone(),
                    });
                    let _ = conn_tx.send(ConnectionManagerEvent::State(
                        ConnectionEvent::Disconnected { reason },
                    ));
                }
                Err(err) => {
                    let reason = format!("ACP stdio connection failed: {err:?}");
                    let _ = sup_event_tx.send(ServerEvent::StartFailed {
                        error: reason.clone(),
                    });
                    let _ = conn_tx.send(ConnectionManagerEvent::State(
                        ConnectionEvent::Disconnected { reason },
                    ));
                }
            }
        }
        AcpEndpoint::WebSocket { url } => {
            let mut attempt = 0u32;
            loop {
                if attempt > 0 {
                    let delay_ms = acp::websocket_retry_delay(attempt - 1).as_millis() as u64;
                    let _ =
                        conn_tx.send(ConnectionManagerEvent::State(ConnectionEvent::Connecting {
                            attempt,
                            delay_ms,
                        }));
                    tokio::time::sleep(acp::websocket_retry_delay(attempt - 1)).await;
                }

                let _ = sup_event_tx.send(ServerEvent::Starting);
                let result = acp::run(
                    AcpEndpoint::WebSocket { url: url.clone() },
                    &mut cmd_rx,
                    srv_tx.clone(),
                    conn_tx.clone(),
                    launch_cwd.clone(),
                )
                .await;

                match result {
                    Ok(()) => {
                        let reason = "ACP WebSocket connection ended".to_string();
                        let _ = sup_event_tx.send(ServerEvent::Stopped {
                            reason: reason.clone(),
                        });
                        let _ = conn_tx.send(ConnectionManagerEvent::State(
                            ConnectionEvent::Disconnected { reason },
                        ));
                        return;
                    }
                    Err(err) => {
                        let reason = format!("ACP WebSocket connection failed ({url}): {err:?}");
                        let _ = sup_event_tx.send(ServerEvent::StartFailed {
                            error: reason.clone(),
                        });
                        let _ = conn_tx.send(ConnectionManagerEvent::State(
                            ConnectionEvent::Disconnected { reason },
                        ));
                        attempt = attempt.saturating_add(1).max(1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn invalid_stdio_command_fails_before_started() {
        let (srv_tx, _srv_rx) = mpsc::unbounded_channel();
        let (_cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (sup_tx, mut sup_rx) = mpsc::unbounded_channel();

        connection_manager(
            AcpEndpoint::Stdio { argv: Vec::new() },
            srv_tx,
            cmd_rx,
            conn_tx,
            sup_tx,
            None,
        )
        .await;

        assert_eq!(sup_rx.try_recv().expect("starting"), ServerEvent::Starting);
        assert!(matches!(
            sup_rx.try_recv().expect("start failed"),
            ServerEvent::StartFailed { error }
                if error.starts_with("invalid ACP stdio command: ")
        ));
        assert!(sup_rx.try_recv().is_err());
        assert!(matches!(
            conn_rx.try_recv().expect("disconnected"),
            ConnectionManagerEvent::State(ConnectionEvent::Disconnected { reason })
                if reason.starts_with("invalid ACP stdio command: ")
        ));
    }
}
