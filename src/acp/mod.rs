use std::time::Duration;

use tokio::sync::mpsc;

use crate::command::Command;
use crate::runtime_events::{ConnectionManagerEvent, ServerChannelMsg};

mod assistant_buffer;
mod commands;
mod configuration;
mod connection;
mod context;
mod elicitation;
mod events;
mod extensions;
mod inbound;
mod notification;
mod replay;
mod retry;
mod runtime;
mod transport;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AcpEndpoint {
    Stdio { argv: Vec<String> },
    WebSocket { url: String },
}

pub(crate) use retry::websocket_delay as websocket_retry_delay;

pub(crate) async fn run(
    endpoint: AcpEndpoint,
    cmd_rx: &mut mpsc::UnboundedReceiver<Command>,
    srv_tx: mpsc::UnboundedSender<ServerChannelMsg>,
    conn_tx: mpsc::UnboundedSender<ConnectionManagerEvent>,
    launch_cwd: Option<String>,
) -> Result<(), agent_client_protocol::Error> {
    match endpoint {
        AcpEndpoint::Stdio { argv } => {
            transport::stdio::run(argv, cmd_rx, srv_tx, conn_tx, launch_cwd).await
        }
        AcpEndpoint::WebSocket { url } => {
            transport::websocket::run(url, cmd_rx, srv_tx, conn_tx, launch_cwd).await
        }
    }
}

pub(crate) async fn probe_websocket(url: &str, timeout: Duration) -> bool {
    tokio::time::timeout(timeout, tokio_tungstenite::connect_async(url))
        .await
        .is_ok_and(|result| result.is_ok())
}
