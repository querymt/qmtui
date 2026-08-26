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
            let _ = sup_event_tx.send(ServerEvent::Started);
            let result = acp::run(
                AcpEndpoint::Stdio { argv },
                &mut cmd_rx,
                srv_tx,
                conn_tx.clone(),
                launch_cwd,
            )
            .await;

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
