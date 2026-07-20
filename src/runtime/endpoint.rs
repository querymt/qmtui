use std::ffi::OsString;

use clap::Parser;

use crate::{acp_client::AcpEndpoint, config, server_manager};

pub(super) const DEFAULT_ACP_WS_HOST: &str = "127.0.0.1";
const DEFAULT_ACP_WS_PORT: &str = "3030";
const DEFAULT_ACP_WS_PATH: &str = "/ws";

#[derive(Parser)]
#[command(name = "qmtui")]
#[command(version = env!("QMTUI_BUILD_VERSION"))]
#[command(about = "querymt terminal interface")]
pub(super) struct Cli {
    /// Override the qmtcode binary used for ACP stdio for this run.
    #[arg(short = 'b', long = "acp-binary")]
    pub(super) acp_binary: Option<String>,

    /// Connect to an ACP WebSocket server; defaults to 127.0.0.1:3030.
    #[arg(short = 'w', long = "ws", value_name = "addr", num_args = 0..=1, default_missing_value = DEFAULT_ACP_WS_HOST)]
    pub(super) ws: Option<String>,

    /// Backcompat alias for --ws.
    #[arg(long, hide = true)]
    pub(super) acp_websocket: Option<String>,

    /// Restore a session by id.
    #[arg(short = 's', long)]
    pub(super) session: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum EndpointSelection {
    Endpoint {
        endpoint: AcpEndpoint,
        state: server_manager::ServerState,
        discovered_ws: Option<String>,
        missing_binary_fallback: bool,
    },
    BinaryNotFound,
    Disabled,
}

pub(super) fn normalize_acp_ws_url(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return default_acp_ws_url();
    }

    let mut url = if trimmed.starts_with("ws://") || trimmed.starts_with("wss://") {
        trimmed.to_string()
    } else {
        format!("ws://{trimmed}")
    };

    let scheme_end = url.find("://").map(|idx| idx + 3).unwrap_or(0);
    let authority_end = url[scheme_end..]
        .find('/')
        .map(|idx| scheme_end + idx)
        .unwrap_or(url.len());
    if !url[scheme_end..authority_end].contains(':') {
        url.insert_str(authority_end, &format!(":{DEFAULT_ACP_WS_PORT}"));
    }

    let path_start = url[scheme_end..].find('/').map(|idx| scheme_end + idx);
    match path_start {
        Some(path_start) if url[path_start..].trim_end_matches('/').is_empty() => {
            url.truncate(path_start);
            url.push_str(DEFAULT_ACP_WS_PATH);
        }
        Some(_) => {}
        None => url.push_str(DEFAULT_ACP_WS_PATH),
    }
    url
}

pub(super) fn default_acp_ws_url() -> String {
    normalize_acp_ws_url(DEFAULT_ACP_WS_HOST)
}

pub(super) fn select_acp_endpoint(
    cli: &Cli,
    cfg: &config::TuiConfig,
    default_ws_available: bool,
) -> EndpointSelection {
    if let Some(url) = cli
        .ws
        .as_deref()
        .or(cli.acp_websocket.as_deref())
        .map(normalize_acp_ws_url)
    {
        return EndpointSelection::Endpoint {
            endpoint: AcpEndpoint::WebSocket { url },
            state: server_manager::ServerState::Starting,
            discovered_ws: None,
            missing_binary_fallback: false,
        };
    }

    if let Some(binary_path) = cli.acp_binary.as_deref() {
        return EndpointSelection::Endpoint {
            endpoint: AcpEndpoint::Stdio {
                argv: server_manager::build_acp_argv(OsString::from(binary_path), cfg.acp_args()),
            },
            state: server_manager::ServerState::Starting,
            discovered_ws: None,
            missing_binary_fallback: false,
        };
    }

    if let Some(url) = cfg.acp.websocket_url.as_deref().map(normalize_acp_ws_url) {
        return EndpointSelection::Endpoint {
            endpoint: AcpEndpoint::WebSocket { url },
            state: server_manager::ServerState::Starting,
            discovered_ws: None,
            missing_binary_fallback: false,
        };
    }

    let transport = cfg.acp.transport.unwrap_or_default();
    if transport == config::AcpTransportMode::WebSocket {
        return EndpointSelection::Endpoint {
            endpoint: AcpEndpoint::WebSocket {
                url: default_acp_ws_url(),
            },
            state: server_manager::ServerState::Starting,
            discovered_ws: None,
            missing_binary_fallback: false,
        };
    }

    if !cfg.acp.auto_start.unwrap_or(true) {
        return EndpointSelection::Disabled;
    }

    let default_ws_url = default_acp_ws_url();
    if default_ws_available {
        return EndpointSelection::Endpoint {
            endpoint: AcpEndpoint::WebSocket {
                url: default_ws_url.clone(),
            },
            state: server_manager::ServerState::Starting,
            discovered_ws: Some(default_ws_url),
            missing_binary_fallback: false,
        };
    }

    let discovery = server_manager::find_binary_info(cfg.acp.binary_path.as_deref());
    discovery.binary.map_or_else(
        || EndpointSelection::Endpoint {
            endpoint: AcpEndpoint::WebSocket {
                url: default_ws_url,
            },
            state: server_manager::ServerState::Starting,
            discovered_ws: None,
            missing_binary_fallback: true,
        },
        |binary| EndpointSelection::Endpoint {
            endpoint: AcpEndpoint::Stdio {
                argv: server_manager::build_acp_argv(binary, cfg.acp_args()),
            },
            state: server_manager::ServerState::Starting,
            discovered_ws: None,
            missing_binary_fallback: false,
        },
    )
}

pub(super) fn detect_launch_cwd() -> Option<String> {
    std::env::current_dir()
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
}
