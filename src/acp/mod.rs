use std::time::Duration;

use tokio::sync::mpsc;

use crate::acp_client;
use crate::command::Command;
use crate::runtime_events::{ConnectionManagerEvent, ServerChannelMsg};

mod assistant_buffer;
mod configuration;
mod connection;
mod elicitation;
mod events;
mod extensions;
mod replay;
mod retry;
mod runtime;
mod transport;

pub(crate) use crate::acp_client::AcpEndpoint;

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
            let agent = agent_client_protocol::AcpAgent::from_args(argv)?;
            acp_client::run_stdio_agent(agent, cmd_rx, srv_tx, conn_tx, launch_cwd).await
        }
        AcpEndpoint::WebSocket { url } => {
            acp_client::run_websocket_agent(url, cmd_rx, srv_tx, conn_tx, launch_cwd).await
        }
    }
}

pub(crate) async fn probe_websocket(url: &str, timeout: Duration) -> bool {
    acp_client::probe_websocket(url, timeout).await
}
