use std::time::Duration;

use tokio::sync::mpsc;

use crate::{
    acp_client::{self, AcpEndpoint},
    app::ConnectionEvent,
    protocol::ClientMsg,
    server_manager::ServerEvent,
};

use super::{ConnectionManagerEvent, ServerChannelMsg};

fn reconnect_delay_ms(attempt: u32) -> u64 {
    let capped = attempt.min(5);
    250 * (1u64 << capped)
}

pub(super) async fn connection_manager(
    endpoint: AcpEndpoint,
    srv_tx: mpsc::UnboundedSender<ServerChannelMsg>,
    mut cmd_rx: mpsc::UnboundedReceiver<ClientMsg>,
    conn_tx: mpsc::UnboundedSender<ConnectionManagerEvent>,
    sup_event_tx: mpsc::UnboundedSender<ServerEvent>,
    launch_cwd: Option<String>,
) {
    match endpoint {
        AcpEndpoint::Stdio { argv } => {
            let _ = sup_event_tx.send(ServerEvent::Starting);
            let agent = match agent_client_protocol::AcpAgent::from_args(argv) {
                Ok(agent) => agent,
                Err(err) => {
                    let message = format!("invalid ACP stdio command: {err:?}");
                    let _ = sup_event_tx.send(ServerEvent::StartFailed {
                        error: message.clone(),
                    });
                    let _ = conn_tx.send(ConnectionManagerEvent::State(
                        ConnectionEvent::Disconnected { reason: message },
                    ));
                    return;
                }
            };

            let _ = sup_event_tx.send(ServerEvent::Started);
            let result = acp_client::run_stdio_agent(
                agent,
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
                    let delay_ms = reconnect_delay_ms(attempt - 1);
                    let _ =
                        conn_tx.send(ConnectionManagerEvent::State(ConnectionEvent::Connecting {
                            attempt,
                            delay_ms,
                        }));
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }

                let _ = sup_event_tx.send(ServerEvent::Starting);
                let result = acp_client::run_websocket_agent(
                    url.clone(),
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
    use super::reconnect_delay_ms;

    #[test]
    fn reconnect_delay_caps_after_five_steps() {
        assert_eq!(reconnect_delay_ms(0), 250);
        assert_eq!(reconnect_delay_ms(1), 500);
        assert_eq!(reconnect_delay_ms(2), 1000);
        assert_eq!(reconnect_delay_ms(3), 2000);
        assert_eq!(reconnect_delay_ms(4), 4000);
        assert_eq!(reconnect_delay_ms(5), 8000);
        assert_eq!(reconnect_delay_ms(8), 8000);
    }
}
