//! Quick-and-dirty ACP WebSocket CLI.
//!
//! Reuses qmtui's `agent_client_protocol` v1 types with a minimal probe-client capability set.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use agent_client_protocol::{
    JsonRpcMessage, JsonRpcRequest, JsonRpcResponse, schema::ProtocolVersion, schema::v1 as acp,
};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use url::Url;

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 3030;
const DEFAULT_PATH: &str = "/ws";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const PROMPT_TIMEOUT: Duration = Duration::from_secs(10 * 60);

type RpcResult = Result<Value, String>;
type PendingRequests = Arc<Mutex<HashMap<i64, oneshot::Sender<RpcResult>>>>;

#[derive(Parser)]
#[command(name = "qmtacp")]
#[command(about = "Dirty ACP WebSocket client using qmtui protocol types")]
struct Cli {
    /// ACP WebSocket URL or host[:port][/path]. Defaults to ws://127.0.0.1:3030/ws
    #[arg(short, long, value_name = "url")]
    url: Option<String>,

    /// Keep reading notifications after the last request, in milliseconds.
    #[arg(short, long, default_value_t = 2000)]
    listen: u64,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// initialize + session/new, then listen (default)
    Handshake {
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long)]
        profile: Option<String>,
    },
    /// initialize only
    Init,
    /// initialize + session/new
    New {
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long)]
        profile: Option<String>,
    },
    /// initialize + session/load
    Load {
        session_id: String,
        #[arg(long)]
        cwd: Option<PathBuf>,
    },
    /// initialize + session/list
    List,
    /// initialize + session/prompt
    Prompt { session_id: String, text: String },
    /// initialize, then send a raw JSON-RPC method
    Call {
        method: String,
        #[arg(default_value = "{}")]
        params: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let url = normalize_acp_ws_url(cli.url.as_deref().unwrap_or(DEFAULT_HOST))?;
    let listen = Duration::from_millis(cli.listen);
    let command = cli.command.unwrap_or(Command::Handshake {
        cwd: None,
        profile: None,
    });

    eprintln!("connecting {url}");
    let client = AcpClient::connect(&url).await?;
    match command {
        Command::Init => {
            print_json("initialize", &client.initialize().await?)?;
        }
        Command::Handshake { cwd, profile } | Command::New { cwd, profile } => {
            print_json("initialize", &client.initialize().await?)?;
            print_json("session/new", &client.new_session(cwd, profile).await?)?;
        }
        Command::Load { session_id, cwd } => {
            print_json("initialize", &client.initialize().await?)?;
            print_json("session/load", &client.load_session(session_id, cwd).await?)?;
        }
        Command::List => {
            print_json("initialize", &client.initialize().await?)?;
            print_json("session/list", &client.list_sessions().await?)?;
        }
        Command::Prompt { session_id, text } => {
            print_json("initialize", &client.initialize().await?)?;
            print_json("session/prompt", &client.prompt(session_id, text).await?)?;
        }
        Command::Call { method, params } => {
            print_json("initialize", &client.initialize().await?)?;
            let params: Value = serde_json::from_str(&params).context("params JSON")?;
            print_json(&method, &client.call(&method, params).await?)?;
        }
    }

    if !listen.is_zero() {
        eprintln!("listening {}ms", listen.as_millis());
        tokio::time::sleep(listen).await;
    }
    Ok(())
}

struct AcpClient {
    tx: mpsc::UnboundedSender<Message>,
    pending: PendingRequests,
    next_id: AtomicI64,
}

impl AcpClient {
    async fn connect(url: &str) -> Result<Self> {
        let (socket, _) = connect_async(url)
            .await
            .with_context(|| format!("connect {url}"))?;
        let (mut write, mut read) = socket.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();
        let pending: PendingRequests = Arc::new(Mutex::new(HashMap::new()));

        let pending_write = pending.clone();
        tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                if let Err(err) = write.send(message).await {
                    fail_pending(&pending_write, format!("WebSocket write failed: {err}")).await;
                    return;
                }
            }
            fail_pending(&pending_write, "WebSocket writer stopped").await;
        });

        let pending_read = pending.clone();
        let tx_read = tx.clone();
        tokio::spawn(async move {
            while let Some(message) = read.next().await {
                match message {
                    Ok(Message::Text(text)) => {
                        if let Err(err) =
                            handle_inbound(&pending_read, &tx_read, text.as_ref()).await
                        {
                            eprintln!("inbound error: {err:#}");
                        }
                    }
                    Ok(Message::Close(_)) => break,
                    Ok(_) => {}
                    Err(err) => {
                        eprintln!("WebSocket read failed: {err}");
                        break;
                    }
                }
            }
            fail_pending(&pending_read, "WebSocket connection closed").await;
        });

        Ok(Self {
            tx,
            pending,
            next_id: AtomicI64::new(1),
        })
    }

    async fn initialize(&self) -> Result<acp::InitializeResponse> {
        self.request(
            acp::InitializeRequest::new(ProtocolVersion::V1)
                .client_capabilities(client_capabilities())
                .client_info(acp::Implementation::new(
                    "qmtacp",
                    env!("CARGO_PKG_VERSION"),
                )),
        )
        .await
    }

    async fn new_session(
        &self,
        cwd: Option<PathBuf>,
        profile: Option<String>,
    ) -> Result<acp::NewSessionResponse> {
        let mut request = acp::NewSessionRequest::new(cwd.unwrap_or_else(default_cwd));
        if let Some(profile_id) = profile.as_deref() {
            request = request.meta(profile_meta(profile_id));
        }
        self.request(request).await
    }

    async fn load_session(
        &self,
        session_id: String,
        cwd: Option<PathBuf>,
    ) -> Result<acp::LoadSessionResponse> {
        self.request(acp::LoadSessionRequest::new(
            session_id,
            cwd.unwrap_or_else(default_cwd),
        ))
        .await
    }

    async fn list_sessions(&self) -> Result<acp::ListSessionsResponse> {
        self.request(acp::ListSessionsRequest::new()).await
    }

    async fn prompt(&self, session_id: String, text: String) -> Result<acp::PromptResponse> {
        self.request_with_timeout(
            acp::PromptRequest::new(
                session_id,
                vec![acp::ContentBlock::Text(acp::TextContent::new(text))],
            ),
            PROMPT_TIMEOUT,
        )
        .await
    }

    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.request_raw(method, params, REQUEST_TIMEOUT).await
    }

    async fn request<R>(&self, request: R) -> Result<R::Response>
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static,
    {
        self.request_with_timeout(request, REQUEST_TIMEOUT).await
    }

    async fn request_with_timeout<R>(&self, request: R, timeout: Duration) -> Result<R::Response>
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static,
    {
        let message = request.to_untyped_message()?;
        let method = message.method.clone();
        let result = self.request_raw(&method, message.params, timeout).await?;
        Ok(R::Response::from_value(&method, result)?)
    }

    async fn request_raw(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        let envelope = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        if self
            .tx
            .send(Message::Text(envelope.to_string().into()))
            .is_err()
        {
            self.pending.lock().await.remove(&id);
            bail!("WebSocket writer is closed");
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result.map_err(anyhow::Error::msg),
            Ok(Err(_)) => bail!("request {method} was dropped"),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                bail!("request {method} timed out after {}s", timeout.as_secs());
            }
        }
    }
}

async fn fail_pending(pending: &PendingRequests, message: impl Into<String>) {
    let message = message.into();
    for (_, waiter) in pending.lock().await.drain() {
        let _ = waiter.send(Err(message.clone()));
    }
}

async fn handle_inbound(
    pending: &PendingRequests,
    tx: &mpsc::UnboundedSender<Message>,
    text: &str,
) -> Result<()> {
    let value: Value = serde_json::from_str(text).context("inbound JSON")?;
    if value.get("method").is_none() {
        let Some(id) = value.get("id").and_then(Value::as_i64) else {
            eprintln!("<= {text}");
            return Ok(());
        };
        let result = if let Some(error) = value.get("error") {
            Err(error.to_string())
        } else {
            Ok(value.get("result").cloned().unwrap_or(Value::Null))
        };
        if let Some(waiter) = pending.lock().await.remove(&id) {
            let _ = waiter.send(result);
        }
        return Ok(());
    }

    let method = value
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let params = value.get("params").cloned().unwrap_or(Value::Null);
    print_notification(&method, &params);

    if let Some(id) = value.get("id").cloned() {
        let reply = inbound_request_reply(id, &method, &params)?;
        tx.send(Message::Text(reply.to_string().into()))
            .context("reply to inbound request")?;
    }
    Ok(())
}

fn inbound_request_reply(id: Value, method: &str, params: &Value) -> Result<Value> {
    if acp::RequestPermissionRequest::matches_method(method) {
        // A non-interactive probe must not authorize agent tool use on the user's behalf.
        let request = acp::RequestPermissionRequest::parse_message(method, params)?;
        let outcome = request
            .options
            .iter()
            .find(|option| {
                matches!(
                    option.kind,
                    acp::PermissionOptionKind::RejectOnce | acp::PermissionOptionKind::RejectAlways
                )
            })
            .map(|option| {
                acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(
                    option.option_id.clone(),
                ))
            })
            .unwrap_or(acp::RequestPermissionOutcome::Cancelled);
        let result = acp::RequestPermissionResponse::new(outcome).into_json(method)?;
        return Ok(json!({ "jsonrpc": "2.0", "id": id, "result": result }));
    }

    Ok(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32601, "message": format!("qmtacp does not support {method}") },
    }))
}

fn print_notification(method: &str, params: &Value) {
    if method == "session/update" {
        match acp::SessionNotification::parse_message(method, params) {
            Ok(notification) => {
                if let acp::SessionUpdate::AvailableCommandsUpdate(update) = &notification.update {
                    let names: Vec<_> = update
                        .available_commands
                        .iter()
                        .map(|command| command.name.as_str())
                        .collect();
                    println!(
                        "<= availableCommandsUpdate session={} commands={names:?}",
                        notification.session_id
                    );
                    return;
                }
                println!(
                    "<= session/update session={} {:?}",
                    notification.session_id, notification.update
                );
            }
            Err(err) => eprintln!("<= session/update parse error: {err}\n{params}"),
        }
        return;
    }
    println!("<= {method} {params}");
}

fn print_json(label: &str, value: &impl serde::Serialize) -> Result<()> {
    println!(
        "=> {label}\n{}",
        serde_json::to_string_pretty(value).context("serialize")?
    );
    Ok(())
}

fn client_capabilities() -> acp::ClientCapabilities {
    acp::ClientCapabilities::new().terminal(false)
}

fn profile_meta(profile_id: &str) -> serde_json::Map<String, Value> {
    let mut meta = serde_json::Map::new();
    meta.insert("querymt".to_string(), json!({ "profile_id": profile_id }));
    meta
}

fn default_cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn normalize_acp_ws_url(value: &str) -> Result<String> {
    let trimmed = value.trim();
    let trimmed = if trimmed.is_empty() {
        DEFAULT_HOST
    } else {
        trimmed
    };
    let candidate = if trimmed.starts_with("ws://") || trimmed.starts_with("wss://") {
        trimmed.to_string()
    } else {
        if trimmed.contains("://") {
            bail!("WebSocket URL must use ws:// or wss://");
        }
        let suffix_start = trimmed.find(['/', '?', '#']).unwrap_or(trimmed.len());
        let (authority, suffix) = trimmed.split_at(suffix_start);
        let authority = if authority.matches(':').count() > 1 && !authority.starts_with('[') {
            format!("[{authority}]")
        } else {
            authority.to_string()
        };
        format!("ws://{authority}{suffix}")
    };

    let explicit_port = explicit_port(&candidate)?;
    let mut url =
        Url::parse(&candidate).with_context(|| format!("invalid WebSocket URL: {value}"))?;
    if !matches!(url.scheme(), "ws" | "wss") {
        bail!("WebSocket URL must use ws:// or wss://");
    }
    if url.host_str().is_none() {
        bail!("WebSocket URL is missing a host");
    }
    if explicit_port.is_none() {
        url.set_port(Some(DEFAULT_PORT))
            .map_err(|_| anyhow::anyhow!("cannot set default WebSocket port"))?;
    }
    if url.path().is_empty() || url.path() == "/" {
        url.set_path(DEFAULT_PATH);
    }

    let normalized = url.to_string();
    match (explicit_port, url.port()) {
        (Some(port), None) => Ok(insert_port(&normalized, port)),
        _ => Ok(normalized),
    }
}

fn explicit_port(value: &str) -> Result<Option<u16>> {
    let authority_start = value
        .find("://")
        .map(|index| index + 3)
        .ok_or_else(|| anyhow::anyhow!("WebSocket URL is missing a scheme"))?;
    let authority_end = value[authority_start..]
        .find(['/', '?', '#'])
        .map(|index| authority_start + index)
        .unwrap_or(value.len());
    let authority = value[authority_start..authority_end]
        .rsplit_once('@')
        .map_or(&value[authority_start..authority_end], |(_, host)| host);
    let port = if let Some(bracket_end) = authority.find(']') {
        authority[bracket_end + 1..].strip_prefix(':')
    } else {
        authority.rsplit_once(':').map(|(_, port)| port)
    };
    port.map(|port| {
        port.parse::<u16>()
            .with_context(|| format!("invalid WebSocket port: {port}"))
    })
    .transpose()
}

fn insert_port(value: &str, port: u16) -> String {
    let authority_start = value.find("://").map_or(0, |index| index + 3);
    let authority_end = value[authority_start..]
        .find(['/', '?', '#'])
        .map(|index| authority_start + index)
        .unwrap_or(value.len());
    let mut value = value.to_string();
    value.insert_str(authority_end, &format!(":{port}"));
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_websocket_urls_structurally() {
        let cases = [
            ("127.0.0.1", "ws://127.0.0.1:3030/ws"),
            ("localhost", "ws://localhost:3030/ws"),
            ("::1", "ws://[::1]:3030/ws"),
            ("ws://[::1]", "ws://[::1]:3030/ws"),
            ("ws://host:80", "ws://host:80/ws"),
            ("wss://host:443", "wss://host:443/ws"),
            ("ws://[::1]:80", "ws://[::1]:80/ws"),
            ("ws://host:9000", "ws://host:9000/ws"),
            ("host/custom", "ws://host:3030/custom"),
            ("host?token=test", "ws://host:3030/ws?token=test"),
            (
                "wss://host:9443/custom?token=test",
                "wss://host:9443/custom?token=test",
            ),
        ];

        for (input, expected) in cases {
            assert_eq!(normalize_acp_ws_url(input).unwrap(), expected);
        }
    }

    #[test]
    fn rejects_non_websocket_urls() {
        assert!(normalize_acp_ws_url("https://example.com").is_err());
    }

    #[test]
    fn permission_requests_choose_a_reject_option() {
        let request = acp::RequestPermissionRequest::new(
            "session-1",
            acp::ToolCallUpdate::new("tool-1", acp::ToolCallUpdateFields::new()),
            vec![
                acp::PermissionOption::new("allow", "Allow", acp::PermissionOptionKind::AllowOnce),
                acp::PermissionOption::new(
                    "reject",
                    "Reject",
                    acp::PermissionOptionKind::RejectOnce,
                ),
            ],
        );
        let params = request.to_untyped_message().unwrap().params;
        let reply = inbound_request_reply(json!(7), "session/request_permission", &params).unwrap();

        assert_eq!(reply["id"], 7);
        assert_eq!(reply["result"]["outcome"]["outcome"], "selected");
        assert_eq!(reply["result"]["outcome"]["optionId"], "reject");
    }
}
