use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1 as acp;
use agent_client_protocol::{
    self as acp_sdk, AcpAgent, Agent, Client, ConnectionTo, JsonRpcMessage, JsonRpcNotification,
    JsonRpcRequest, JsonRpcResponse, UntypedMessage,
};
use futures_util::{SinkExt, StreamExt as FuturesStreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::ServerChannelMsg;
use crate::acp_state::{AcpAppEvent, AcpModelsMetaInfo, AcpSessionUpdate};
use crate::command::{Command, PromptBlock, SessionListRequest};
use crate::domain::activity::{DelegationState, DelegationUpdate};
use crate::domain::auth::{OAuthFlow, OAuthResult, OAuthResultStatus};
use crate::domain::mesh::{
    MeshInviteCreatedInfo, MeshNodesInfo, MeshScopeInfo, MeshStatusInfo, RemoteNodeInfo,
    RemoteSessionAttachInfo, RemoteSessionInfo, RemoteSessionListInfo,
};
use crate::domain::model::ModelEntry;
use crate::domain::profile::{AgentInfo, ProfileInfo};
use crate::domain::session::{
    ForkResult, RedoResult, SessionGroup, SessionListPage, SessionSummary, UndoResult,
    UndoStackSnapshot,
};
use crate::protocol::{
    AuthProvidersData, DelegationUpdateDto, DelegationUpdateStateDto, MeshInviteCreatedDto,
    MeshNodesDto, MeshScopeDto, MeshStatusDto, OAuthFlowDto, OAuthResultDto, RedoResultData,
    RemoteNodeDto, RemoteSessionAttachDto, RemoteSessionDto, RemoteSessionListDto, UndoResultData,
    UndoStackFrame,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcpEndpoint {
    Stdio { argv: Vec<String> },
    WebSocket { url: String },
}

#[derive(Debug, Clone)]
struct AgentIdentity {
    id: String,
    name: String,
}

impl Default for AgentIdentity {
    fn default() -> Self {
        Self {
            id: "querymt".to_string(),
            name: "QueryMT".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct AcpModelEntry {
    id: String,
    label: String,
    #[serde(default)]
    source: Option<String>,
    provider: String,
    model: String,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    node_label: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    quant: Option<String>,
}

impl AcpModelEntry {
    fn to_app_model(&self) -> ModelEntry {
        ModelEntry {
            id: self.id.clone(),
            label: self.label.clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            node_id: self.node_id.clone(),
            node_label: self.node_label.clone(),
            family: self.family.clone(),
            quant: self.quant.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct AcpModelsMeta {
    #[serde(default)]
    stale: bool,
    #[serde(default)]
    refresh_in_progress: bool,
    #[serde(default)]
    remote_node_count: u32,
    #[serde(default)]
    remote_timeout_count: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct AcpModelsResponse {
    models: Vec<AcpModelEntry>,
    meta: Option<AcpModelsMeta>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct AcpProfilesResponse {
    profiles: Vec<ProfileInfo>,
    #[serde(default)]
    active_profile_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AcpProfileAgentsResponse {
    profile_id: String,
    agents: Vec<AgentInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DelegateModelOverrideInfo {
    pub model_id: String,
    #[serde(default)]
    pub node_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct AcpDelegateModelResponse {
    session_id: String,
    agent_id: String,
    #[serde(default)]
    model: Option<DelegateModelOverrideInfo>,
}

fn normalize_profile_agents_response(
    response: Value,
) -> Result<AcpProfileAgentsResponse, serde_json::Error> {
    serde_json::from_value(ext_payload(&response).clone())
}

fn normalize_delegate_model_response(
    response: Value,
) -> Result<AcpDelegateModelResponse, serde_json::Error> {
    serde_json::from_value(ext_payload(&response).clone())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotProviderChange {
    provider: String,
    model: String,
    context_limit: Option<u64>,
    provider_node_id: Option<String>,
}

impl AcpModelsResponse {
    fn should_retry_empty(&self) -> bool {
        self.models.is_empty()
            || self
                .meta
                .as_ref()
                .is_some_and(|meta| meta.stale || meta.refresh_in_progress)
    }
}

#[derive(Debug, Default)]
struct AssistantBuffer {
    content: String,
    thinking: String,
    content_message_id: Option<String>,
    thinking_message_id: Option<String>,
}

enum PendingElicitationResponse {
    Sdk(acp_sdk::Responder<acp::CreateElicitationResponse>),
    WsResultMethod(WsAcpConnection),
    WsJsonRpcResponse {
        connection: WsAcpConnection,
        id: Value,
    },
}

#[derive(Default)]
struct AcpRuntimeState {
    agent: Mutex<AgentIdentity>,
    current_session_id: Mutex<Option<String>>,
    loading_sessions: Mutex<HashSet<String>>,
    replay_updates: Mutex<HashMap<String, Vec<AcpSessionUpdate>>>,
    assistant_buffers: Mutex<HashMap<String, AssistantBuffer>>,
    pending_elicitations: Mutex<HashMap<String, PendingElicitationResponse>>,
    models: Mutex<Vec<AcpModelEntry>>,
    selected_model_id: Mutex<Option<String>>,
    launch_cwd: Option<String>,
}

impl AcpRuntimeState {
    fn new(launch_cwd: Option<String>) -> Self {
        Self {
            launch_cwd,
            ..Self::default()
        }
    }

    async fn agent_identity(&self) -> AgentIdentity {
        self.agent.lock().await.clone()
    }

    async fn set_agent_identity(&self, identity: AgentIdentity) {
        *self.agent.lock().await = identity;
    }

    async fn current_session_id(&self) -> Option<String> {
        self.current_session_id.lock().await.clone()
    }

    async fn set_current_session_id(&self, session_id: impl Into<String>) {
        *self.current_session_id.lock().await = Some(session_id.into());
    }

    async fn begin_loading(&self, session_id: &str) {
        self.loading_sessions
            .lock()
            .await
            .insert(session_id.to_string());
    }

    async fn end_loading(&self, session_id: &str) -> Vec<AcpSessionUpdate> {
        self.loading_sessions.lock().await.remove(session_id);
        self.replay_updates
            .lock()
            .await
            .remove(session_id)
            .unwrap_or_default()
    }

    async fn is_loading(&self, session_id: &str) -> bool {
        self.loading_sessions.lock().await.contains(session_id)
    }

    async fn queue_replay_update(&self, session_id: &str, update: AcpSessionUpdate) {
        self.replay_updates
            .lock()
            .await
            .entry(session_id.to_string())
            .or_default()
            .push(update);
    }

    async fn set_models(&self, models: Vec<AcpModelEntry>) {
        *self.models.lock().await = models;
    }

    async fn model_by_id(&self, model_id: &str) -> Option<AcpModelEntry> {
        self.models
            .lock()
            .await
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
    }

    async fn select_model(&self, model_id: impl Into<String>) {
        *self.selected_model_id.lock().await = Some(model_id.into());
    }

    async fn selected_or_default_model(&self) -> Option<AcpModelEntry> {
        let selected = self.selected_model_id.lock().await.clone();
        let models = self.models.lock().await;
        selected
            .as_deref()
            .and_then(|id| models.iter().find(|model| model.id == id))
            .or_else(|| models.first())
            .cloned()
    }

    fn default_cwd(&self) -> PathBuf {
        self.launch_cwd
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

trait AcpConnection: Clone + Send + Sync + 'static {
    fn request<R>(
        &self,
        request: R,
    ) -> impl Future<Output = Result<R::Response, acp_sdk::Error>> + Send
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static;

    fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
    where
        N: JsonRpcNotification + Send + Sync + 'static;

    fn spawn(
        &self,
        fut: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
    ) -> Result<(), acp_sdk::Error>;
}

fn acp_internal_error(message: impl ToString) -> acp_sdk::Error {
    acp_sdk::Error::internal_error().data(message.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcEnvelope {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    method: Option<String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    params: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

#[derive(Clone)]
struct WsAcpConnection {
    tx: mpsc::UnboundedSender<Message>,
    pending: Arc<Mutex<HashMap<i64, PendingWsRequest>>>,
    next_id: Arc<AtomicI64>,
}

struct PendingWsRequest {
    method: String,
    tx: oneshot::Sender<Result<Value, acp_sdk::Error>>,
}

impl WsAcpConnection {
    fn new(tx: mpsc::UnboundedSender<Message>) -> Self {
        Self {
            tx,
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicI64::new(1)),
        }
    }

    async fn request_value(&self, method: &str, params: Value) -> Result<Value, acp_sdk::Error> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(
            id,
            PendingWsRequest {
                method: method.to_string(),
                tx,
            },
        );
        let envelope = JsonRpcEnvelope {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(id)),
            method: Some(method.to_string()),
            params,
            result: None,
            error: None,
        };
        if let Err(err) = self.send_envelope(envelope) {
            self.pending.lock().await.remove(&id);
            return Err(err);
        }
        rx.await
            .map_err(|_| acp_internal_error(format!("ACP WebSocket request dropped: {method}")))?
    }

    fn notify_value(&self, method: &str, params: Value) -> Result<(), acp_sdk::Error> {
        self.send_envelope(JsonRpcEnvelope {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: Some(method.to_string()),
            params,
            result: None,
            error: None,
        })
    }

    fn respond_value(
        &self,
        id: Value,
        result: Result<Value, acp_sdk::Error>,
    ) -> Result<(), acp_sdk::Error> {
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
        self.send_envelope(JsonRpcEnvelope {
            jsonrpc: "2.0".to_string(),
            id: Some(id),
            method: None,
            params: Value::Null,
            result,
            error,
        })
    }

    fn send_envelope(&self, envelope: JsonRpcEnvelope) -> Result<(), acp_sdk::Error> {
        let text = serde_json::to_string(&envelope).map_err(acp_sdk::Error::into_internal_error)?;
        self.tx
            .send(Message::Text(text.into()))
            .map_err(|err| acp_internal_error(format!("ACP WebSocket send failed: {err}")))
    }

    async fn resolve_response(&self, envelope: JsonRpcEnvelope) -> Result<(), acp_sdk::Error> {
        let Some(id) = envelope.id.and_then(|id| id.as_i64()) else {
            return Ok(());
        };
        let Some(pending) = self.pending.lock().await.remove(&id) else {
            return Ok(());
        };
        let result = if let Some(error) = envelope.error {
            Err(acp_internal_error(format!(
                "ACP WebSocket {} failed: {error}",
                pending.method
            )))
        } else {
            Ok(envelope.result.unwrap_or(Value::Null))
        };
        let _ = pending.tx.send(result);
        Ok(())
    }

    async fn fail_all(&self, message: &str) {
        let pending = std::mem::take(&mut *self.pending.lock().await);
        for (_, request) in pending {
            let _ = request.tx.send(Err(acp_internal_error(format!(
                "ACP WebSocket {} failed: {message}",
                request.method
            ))));
        }
    }
}

impl AcpConnection for ConnectionTo<Agent> {
    async fn request<R>(&self, request: R) -> Result<R::Response, acp_sdk::Error>
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static,
    {
        self.send_request(request).block_task().await
    }

    fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
    where
        N: JsonRpcNotification + Send + Sync + 'static,
    {
        self.send_notification(notification)
    }

    fn spawn(
        &self,
        fut: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
    ) -> Result<(), acp_sdk::Error> {
        self.spawn(fut)
    }
}

impl AcpConnection for WsAcpConnection {
    async fn request<R>(&self, request: R) -> Result<R::Response, acp_sdk::Error>
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static,
    {
        let message = request.to_untyped_message()?;
        let method = message.method.clone();
        let result = self.request_value(&message.method, message.params).await?;
        R::Response::from_value(&method, result)
    }

    fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
    where
        N: JsonRpcNotification + Send + Sync + 'static,
    {
        let message = notification.to_untyped_message()?;
        self.notify_value(&message.method, message.params)
    }

    fn spawn(
        &self,
        fut: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
    ) -> Result<(), acp_sdk::Error> {
        tokio::spawn(async move {
            let _ = fut.await;
        });
        Ok(())
    }
}

pub async fn probe_websocket(url: &str, timeout: Duration) -> bool {
    tokio::time::timeout(timeout, connect_async(url))
        .await
        .is_ok_and(|result| result.is_ok())
}

pub(crate) async fn run_websocket_agent(
    url: String,
    cmd_rx: &mut mpsc::UnboundedReceiver<Command>,
    srv_tx: mpsc::UnboundedSender<ServerChannelMsg>,
    conn_tx: mpsc::UnboundedSender<crate::ConnectionManagerEvent>,
    launch_cwd: Option<String>,
) -> Result<(), acp_sdk::Error> {
    let state = Arc::new(AcpRuntimeState::new(launch_cwd));
    let (socket, _) = connect_async(&url)
        .await
        .map_err(acp_sdk::Error::into_internal_error)?;
    let (mut ws_write, mut ws_read) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();
    let connection = WsAcpConnection::new(tx);

    let writer = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if ws_write.send(message).await.is_err() {
                break;
            }
        }
    });

    let reader_connection = connection.clone();
    let reader_state = state.clone();
    let reader_srv_tx = srv_tx.clone();
    let reader = tokio::spawn(async move {
        let mut close_reason = "socket closed".to_string();
        while let Some(message) = FuturesStreamExt::next(&mut ws_read).await {
            match message {
                Ok(Message::Text(text)) => {
                    if let Err(err) = handle_websocket_text(
                        &reader_connection,
                        &reader_state,
                        &reader_srv_tx,
                        text.as_ref(),
                    )
                    .await
                    {
                        send_error(
                            &reader_srv_tx,
                            format!("ACP WebSocket message failed: {err:?}"),
                        );
                    }
                }
                Ok(Message::Close(frame)) => {
                    close_reason = frame
                        .map(|frame| format!("socket closed: {}", frame.reason))
                        .unwrap_or_else(|| "socket closed".to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => {
                    close_reason = err.to_string();
                    send_error(&reader_srv_tx, format!("ACP WebSocket read failed: {err}"));
                    break;
                }
            }
        }
        reader_connection.fail_all(&close_reason).await;
        close_reason
    });

    let _ = conn_tx.send(crate::ConnectionManagerEvent::State(
        crate::app::ConnectionEvent::Connected,
    ));

    let mut reader = Box::pin(reader);
    let result = loop {
        tokio::select! {
            cmd = cmd_rx.recv() => {
                let Some(cmd) = cmd else {
                    break Ok(());
                };
                if let Err(err) = handle_command(&connection, &state, &srv_tx, cmd).await {
                    send_error(&srv_tx, format!("ACP request failed: {err:?}"));
                }
            }
            reader_result = &mut reader => {
                let reason = reader_result.unwrap_or_else(|err| err.to_string());
                break Err(acp_internal_error(format!("ACP WebSocket connection closed: {reason}")));
            }
        }
    };

    writer.abort();
    result
}

async fn handle_websocket_text(
    connection: &WsAcpConnection,
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    text: &str,
) -> Result<(), acp_sdk::Error> {
    let envelope: JsonRpcEnvelope =
        serde_json::from_str(text).map_err(acp_sdk::Error::into_internal_error)?;
    if envelope.method.is_none() {
        return connection.resolve_response(envelope).await;
    }

    let method = envelope.method.clone().unwrap_or_default();
    match method.as_str() {
        "session/update" => {
            let notification = acp::SessionNotification::parse_message(&method, &envelope.params)?;
            handle_session_notification(state, srv_tx, notification).await;
        }
        "session/request_permission" => {
            let request = acp::RequestPermissionRequest::parse_message(&method, &envelope.params)?;
            let response = permission_response_for(&request);
            if let Some(id) = envelope.id {
                connection.respond_value(
                    id,
                    serde_json::to_value(response).map_err(acp_sdk::Error::into_internal_error),
                )?;
            }
        }
        "elicitation/create" => {
            let request = acp::CreateElicitationRequest::parse_message(&method, &envelope.params)?;
            if let Some(id) = envelope.id {
                handle_ws_elicitation_request(state, srv_tx, connection.clone(), id, request).await;
            }
        }
        "elicitation/requested" => {
            handle_ws_elicitation_requested(state, srv_tx, connection.clone(), envelope.params)
                .await;
        }
        "querymt/models/changed" => {
            if let Ok(response) = call_acp_models(connection, false).await {
                send_models(srv_tx, &response);
            }
        }
        "querymt/session/delegationUpdate" | "_querymt/session/delegationUpdate" => {
            handle_delegation_notification(srv_tx, envelope.params);
        }
        "querymt/mesh/nodesChanged" | "querymt/mesh/joined" | "querymt/mesh/peerExpired" => {
            if let Ok(nodes_resp) =
                call_querymt_ext(connection, "querymt/mesh/nodes", json!({})).await
                && let Ok(nodes) =
                    serde_json::from_value::<MeshNodesDto>(ext_payload(&nodes_resp).clone())
            {
                send_acp(srv_tx, AcpAppEvent::MeshNodes(mesh_nodes_from_wire(nodes)));
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_delegation_notification_method(method: &str) -> bool {
    matches!(
        method,
        "querymt/session/delegationUpdate" | "_querymt/session/delegationUpdate"
    )
}

fn handle_delegation_notification(srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>, params: Value) {
    match serde_json::from_value::<DelegationUpdateDto>(params) {
        Ok(update) => {
            let version = update.version;
            match delegation_update_from_wire(update) {
                Some(update) => send_acp(srv_tx, AcpAppEvent::DelegationUpdate(update)),
                None => send_acp(
                    srv_tx,
                    AcpAppEvent::InfoLog {
                        target: "delegation",
                        message: format!("ignored delegation notification version {version}"),
                    },
                ),
            }
        }
        Err(err) => send_acp(
            srv_tx,
            AcpAppEvent::InfoLog {
                target: "delegation",
                message: format!("invalid delegation notification: {err}"),
            },
        ),
    }
}

async fn handle_ws_elicitation_requested(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    connection: WsAcpConnection,
    params: Value,
) {
    let elicitation_id = params
        .get("elicitationId")
        .or_else(|| params.get("elicitation_id"))
        .and_then(Value::as_str)
        .unwrap_or("elicitation")
        .to_string();
    let session_id = params
        .get("sessionId")
        .or_else(|| params.get("session_id"))
        .and_then(Value::as_str)
        .unwrap_or("request")
        .to_string();
    let message = params
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Input requested")
        .to_string();
    let requested_schema = params
        .get("requestedSchema")
        .or_else(|| params.get("requested_schema"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let source = params
        .get("source")
        .and_then(Value::as_str)
        .unwrap_or("acp-ws")
        .to_string();
    let querymt = params
        .get("_meta")
        .or_else(|| params.get("meta"))
        .and_then(|meta| meta.get("querymt"));
    let allow_custom = source == "builtin:question"
        || params
            .get("allow_custom")
            .or_else(|| params.get("allowCustom"))
            .or_else(|| querymt.and_then(|meta| meta.get("allow_custom")))
            .or_else(|| querymt.and_then(|meta| meta.get("allowCustom")))
            .and_then(Value::as_bool)
            .unwrap_or(false);

    state.pending_elicitations.lock().await.insert(
        elicitation_id.clone(),
        PendingElicitationResponse::WsResultMethod(connection),
    );
    send_session_update(
        srv_tx,
        &session_id,
        AcpSessionUpdate::ElicitationRequested {
            elicitation_id,
            message,
            requested_schema,
            source,
            allow_custom,
        },
        false,
    );
}

async fn handle_ws_elicitation_request(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    connection: WsAcpConnection,
    id: Value,
    request: acp::CreateElicitationRequest,
) {
    let session_id = elicitation_request_session_id(&request);
    flush_assistant_buffer(state, srv_tx, &session_id).await;
    let elicitation_id = id
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| id.to_string());
    let (metadata_source, allow_custom) = elicitation_metadata(&request, "acp-ws");
    let (requested_schema, source) = match &request.mode {
        acp::ElicitationMode::Form(form) => (
            serde_json::to_value(&form.requested_schema).unwrap_or_else(|_| json!({})),
            metadata_source,
        ),
        acp::ElicitationMode::Url(url) => (json!({}), format!("acp-url:{}", url.url)),
        _ => (json!({}), metadata_source),
    };
    state.pending_elicitations.lock().await.insert(
        elicitation_id.clone(),
        PendingElicitationResponse::WsJsonRpcResponse { connection, id },
    );
    send_session_update(
        srv_tx,
        &session_id,
        AcpSessionUpdate::ElicitationRequested {
            elicitation_id,
            message: request.message,
            requested_schema,
            source,
            allow_custom,
        },
        false,
    );
}

pub(crate) async fn run_stdio_agent(
    agent: AcpAgent,
    cmd_rx: &mut mpsc::UnboundedReceiver<Command>,
    srv_tx: mpsc::UnboundedSender<ServerChannelMsg>,
    conn_tx: mpsc::UnboundedSender<crate::ConnectionManagerEvent>,
    launch_cwd: Option<String>,
) -> Result<(), acp_sdk::Error> {
    let state = Arc::new(AcpRuntimeState::new(launch_cwd));

    Client
        .builder()
        .on_receive_notification(
            {
                let state = state.clone();
                let srv_tx = srv_tx.clone();
                async move |notification: acp::SessionNotification, _cx| {
                    handle_session_notification(&state, &srv_tx, notification).await;
                    Ok(())
                }
            },
            acp_sdk::on_receive_notification!(),
        )
        .on_receive_notification(
            {
                let srv_tx = srv_tx.clone();
                async move |notification: UntypedMessage, cx| {
                    let (method, params) = notification.clone().into_parts();
                    if is_delegation_notification_method(&method) {
                        handle_delegation_notification(&srv_tx, params);
                        Ok(acp_sdk::Handled::Yes)
                    } else {
                        Ok(acp_sdk::Handled::No {
                            message: (notification, cx),
                            retry: false,
                        })
                    }
                }
            },
            acp_sdk::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: acp::RequestPermissionRequest, responder, _cx| {
                let response = permission_response_for(&request);
                responder.respond(response)
            },
            acp_sdk::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                let srv_tx = srv_tx.clone();
                async move |request: acp::CreateElicitationRequest, responder, _cx| {
                    handle_elicitation_request(&state, &srv_tx, request, responder).await;
                    Ok(())
                }
            },
            acp_sdk::on_receive_request!(),
        )
        .connect_with(agent, |connection: ConnectionTo<Agent>| async move {
            let _ = conn_tx.send(crate::ConnectionManagerEvent::State(
                crate::app::ConnectionEvent::Connected,
            ));

            while let Some(cmd) = cmd_rx.recv().await {
                if let Err(err) = handle_command(&connection, &state, &srv_tx, cmd).await {
                    send_error(&srv_tx, format!("ACP request failed: {err:?}"));
                }
            }

            Ok(())
        })
        .await
}

async fn handle_command<C: AcpConnection>(
    connection: &C,
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    cmd: Command,
) -> Result<(), acp_sdk::Error> {
    match cmd {
        Command::Init => {
            let response = connection
                .request(
                    acp::InitializeRequest::new(ProtocolVersion::V1)
                        .client_capabilities(client_capabilities())
                        .client_info(acp::Implementation::new("qmtui", env!("CARGO_PKG_VERSION"))),
                )
                .await?;
            let identity = response
                .agent_info
                .map(|info| AgentIdentity {
                    id: info.name.clone(),
                    name: info.title.unwrap_or(info.name),
                })
                .unwrap_or_default();
            state.set_agent_identity(identity.clone()).await;
            send_state(
                srv_tx,
                &identity,
                Vec::new(),
                None,
                Some("build"),
                Some(None),
            );
            post_connect_diagnostics(connection, srv_tx).await;
        }
        Command::ListSessions { request, cursor } => {
            let mut req = acp::ListSessionsRequest::new().cursor(cursor);
            if let Some(cwd) = request.cwd() {
                req = req.cwd(PathBuf::from(cwd));
            }
            match connection.request(req).await {
                Ok(response) => send_session_list(srv_tx, request, response),
                Err(err) => send_acp(
                    srv_tx,
                    AcpAppEvent::SessionListFailed {
                        request,
                        message: format!("ACP session/list failed: {err:?}"),
                    },
                ),
            }
        }
        Command::NewSession { cwd, profile_id } => {
            let mut req = acp::NewSessionRequest::new(
                cwd.map(PathBuf::from)
                    .unwrap_or_else(|| state.default_cwd()),
            );
            if let Some(profile_id) = profile_id.as_deref() {
                req = req.meta(profile_meta(profile_id));
            }
            let response = connection.request(req).await?;
            let session_id = response.session_id.to_string();
            state.set_current_session_id(session_id.clone()).await;
            let identity = state.agent_identity().await;
            send_acp(
                srv_tx,
                AcpAppEvent::SessionCreated {
                    agent_id: identity.id.clone(),
                    session_id,
                    profile_id,
                },
            );
            if let Some(config_options) = response.config_options {
                send_config_updates(state, srv_tx, config_options).await;
            }
        }
        Command::LoadSession { session_id, cwd } => {
            state.set_current_session_id(session_id.clone()).await;
            state.begin_loading(&session_id).await;
            let load_cwd = load_session_cwd(cwd.as_deref(), state.default_cwd());
            let req = acp::LoadSessionRequest::new(session_id.clone(), load_cwd);

            let result = connection.request(req).await;
            let replay_updates = state.end_loading(&session_id).await;
            let response = result?;
            let response_value = serde_json::to_value(&response).unwrap_or(Value::Null);
            let snapshot_provider_change =
                snapshot_provider_change_from_load_value(&response_value);
            let snapshot_updates = snapshot_updates_from_load_value(&response_value);
            let delegation_updates = delegation_updates_from_load_value(&response_value);

            let identity = state.agent_identity().await;
            let profile_id = response
                .config_options
                .as_ref()
                .and_then(|opts| profile_id_from_config_options(opts.as_slice()));

            send_acp(
                srv_tx,
                AcpAppEvent::SessionLoaded {
                    session_id: session_id.clone(),
                    agent_id: identity.id.clone(),
                    profile_id,
                },
            );
            let history_updates =
                merge_replay_with_snapshot_stats(replay_updates, snapshot_updates);
            if !history_updates.is_empty() {
                send_acp(
                    srv_tx,
                    AcpAppEvent::SessionReplay {
                        session_id: session_id.clone(),
                        updates: history_updates,
                    },
                );
            }
            if !delegation_updates.is_empty() {
                send_acp(
                    srv_tx,
                    AcpAppEvent::DelegationReplay {
                        session_id: session_id.clone(),
                        updates: delegation_updates,
                    },
                );
            }
            if let Some(change) = snapshot_provider_change {
                send_acp(srv_tx, snapshot_provider_change_event(change));
            }

            if let Some(config_options) = response.config_options {
                send_config_updates(state, srv_tx, config_options).await;
            }
            if let Ok(undo_stack) = fetch_undo_stack(connection, &session_id).await {
                send_acp(srv_tx, AcpAppEvent::UndoStack(undo_stack));
            }
        }
        Command::Prompt { prompt, local_id } => {
            let Some(session_id) = state.current_session_id().await else {
                send_error(srv_tx, "cannot prompt before a session is loaded");
                return Ok(());
            };
            send_session_update(srv_tx, &session_id, AcpSessionUpdate::TurnStarted, false);

            let prompt_connection = connection.clone();
            let prompt_state = state.clone();
            let prompt_srv_tx = srv_tx.clone();
            connection.spawn(async move {
                let req = acp::PromptRequest::new(session_id.clone(), prompt_blocks(prompt));
                match prompt_connection.request(req).await {
                    Ok(response) => {
                        finish_prompt(
                            &prompt_state,
                            &prompt_srv_tx,
                            &session_id,
                            response.stop_reason,
                        )
                        .await;
                    }
                    Err(err) => {
                        send_acp(
                            &prompt_srv_tx,
                            AcpAppEvent::PromptFailed {
                                local_id,
                                message: format!("ACP prompt failed: {err:?}"),
                            },
                        );
                    }
                }
                Ok(())
            })?;
        }
        Command::CancelSession => {
            let Some(session_id) = state.current_session_id().await else {
                send_acp(
                    srv_tx,
                    AcpAppEvent::InfoLog {
                        target: "acp",
                        message: "session/cancel skipped: no active session".to_string(),
                    },
                );
                return Ok(());
            };
            connection.notify(acp::CancelNotification::new(session_id.clone()))?;
            send_acp(
                srv_tx,
                AcpAppEvent::InfoLog {
                    target: "acp",
                    message: format!("sent session/cancel for {session_id}"),
                },
            );
        }
        Command::DeleteSession { session_id } => {
            connection
                .request(acp::DeleteSessionRequest::new(session_id))
                .await?;
        }
        Command::SetAgentMode { mode } => {
            set_config_option(connection, state, srv_tx, "mode", &mode, None).await?;
        }
        Command::SetReasoningEffort { reasoning_effort } => {
            set_config_option(
                connection,
                state,
                srv_tx,
                "reasoning_effort",
                &reasoning_effort,
                None,
            )
            .await?;
        }
        Command::SetSessionModel {
            session_id,
            model_id,
            node_id,
        } => {
            let model = state
                .model_by_id(&model_id)
                .await
                .unwrap_or_else(|| fallback_model_entry(&model_id));
            let effective_node = node_id.as_deref().or(model.node_id.as_deref());
            let node_part = effective_node
                .map(|n| format!(" node={n}"))
                .unwrap_or_default();
            send_acp(
                srv_tx,
                AcpAppEvent::InfoLog {
                    target: "acp",
                    message: format!(
                        "ACP SetSessionModel: provider={} model={} id={}{node_part}",
                        model.provider, model.model, model_id
                    ),
                },
            );
            let meta = model_entry_meta(&model, node_id.as_deref());
            let response = connection
                .request(
                    acp::SetSessionConfigOptionRequest::new(session_id, "model", model_id.as_str())
                        .meta(meta),
                )
                .await?;
            state.select_model(model_id).await;
            send_provider_changed(srv_tx, &model);
            send_config_updates(state, srv_tx, response.config_options).await;
        }
        Command::ListAllModels { refresh } => {
            let response = load_acp_models(connection, refresh).await?;
            state.set_models(response.models.clone()).await;
            send_models(srv_tx, &response);
            if let Some(model) = state.selected_or_default_model().await {
                state.select_model(model.id.clone()).await;
                send_provider_changed(srv_tx, &model);
            }
        }
        Command::ListProfiles => match load_acp_profiles(connection).await {
            Ok(response) => send_profiles(srv_tx, response),
            Err(err) => send_acp(
                srv_tx,
                AcpAppEvent::InfoLog {
                    target: "profiles",
                    message: format!("profile catalog unavailable: {err}"),
                },
            ),
        },
        Command::ListProfileAgents { profile_id } => {
            match load_acp_profile_agents(connection, &profile_id).await {
                Ok(response) => send_acp(
                    srv_tx,
                    AcpAppEvent::ProfileAgents {
                        profile_id: response.profile_id,
                        agents: response.agents,
                    },
                ),
                Err(err) => send_acp(
                    srv_tx,
                    AcpAppEvent::InfoLog {
                        target: "profiles",
                        message: format!("profile agents unavailable for {profile_id}: {err}"),
                    },
                ),
            }
        }
        Command::SetDelegateModel {
            session_id,
            agent_id,
            model_id,
            node_id,
        } => {
            let response = call_querymt_ext(
                connection,
                "querymt/session/setDelegateModel",
                json!({
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "model_id": model_id,
                    "node_id": node_id,
                }),
            )
            .await?;
            let response = normalize_delegate_model_response(response)
                .map_err(acp_sdk::Error::into_internal_error)?;
            send_acp(
                srv_tx,
                AcpAppEvent::DelegateModelSet {
                    session_id: response.session_id,
                    agent_id: response.agent_id,
                    model: response.model,
                },
            );
        }
        Command::ListAuthProviders => {
            let response = call_querymt_ext(connection, "querymt/auth/status", json!({})).await?;
            if let Ok(auth) =
                serde_json::from_value::<AuthProvidersData>(ext_payload(&response).clone())
            {
                send_acp(srv_tx, AcpAppEvent::AuthProviders(auth.providers));
            }
        }
        Command::StartOAuthLogin { provider } => {
            let response = call_querymt_ext(
                connection,
                "querymt/auth/start",
                json!({ "provider": provider }),
            )
            .await?;
            if let Ok(flow) = serde_json::from_value::<OAuthFlowDto>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::OAuthFlowStarted(oauth_flow_from_wire(flow)),
                );
            }
        }
        Command::CompleteOAuthLogin { flow_id, response } => {
            let response = call_querymt_ext(
                connection,
                "querymt/auth/complete",
                json!({ "flow_id": flow_id, "response": response }),
            )
            .await?;
            if let Ok(result) =
                serde_json::from_value::<OAuthResultDto>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::OAuthResult(oauth_result_from_wire(result)),
                );
            }
        }
        Command::DisconnectOAuth { provider } => {
            let response = call_querymt_ext(
                connection,
                "querymt/auth/logout",
                json!({ "provider": provider }),
            )
            .await?;
            if let Ok(result) =
                serde_json::from_value::<OAuthResultDto>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::OAuthResult(oauth_result_from_wire(result)),
                );
            }
        }
        Command::ElicitationResponse {
            elicitation_id,
            action,
            content,
        } => {
            respond_to_elicitation(state, &elicitation_id, &action, content).await;
        }
        Command::ForkSession { message_id } => {
            let Some(session_id) = state.current_session_id().await else {
                send_acp(
                    srv_tx,
                    AcpAppEvent::ForkResult(ForkResult::Failed {
                        source_session_id: None,
                        message: Some("cannot fork before a session is loaded".to_string()),
                    }),
                );
                return Ok(());
            };
            let req = acp::ForkSessionRequest::new(session_id.clone(), state.default_cwd())
                .meta(fork_session_meta(&message_id));
            match connection.request(req).await {
                Ok(response) => send_acp(
                    srv_tx,
                    AcpAppEvent::ForkResult(ForkResult::Succeeded {
                        source_session_id: Some(session_id),
                        forked_session_id: Some(response.session_id.to_string()),
                        message: None,
                    }),
                ),
                Err(err) => send_acp(
                    srv_tx,
                    AcpAppEvent::ForkResult(ForkResult::Failed {
                        source_session_id: Some(session_id),
                        message: Some(format!("ACP fork failed: {err:?}")),
                    }),
                ),
            }
        }
        Command::SubscribeSession { .. } => {}
        Command::GetFileIndex => {
            // TODO(ACP parity): replace the deprecated UI file-index endpoint with
            // an ACP/QueryMT extension or client-side workspace indexing.
            send_error(
                srv_tx,
                "file mentions are not exposed in the ACP subset yet",
            );
        }
        Command::Undo { message_id } => {
            let Some(session_id) = state.current_session_id().await else {
                send_error(srv_tx, "cannot undo before a session is loaded");
                return Ok(());
            };
            let response = call_querymt_ext(
                connection,
                "querymt/session/undo",
                json!({ "session_id": session_id, "message_id": message_id }),
            )
            .await?;
            if let Ok(result) =
                serde_json::from_value::<UndoResultData>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::UndoResult(undo_result_from_wire(result)),
                );
            }
        }
        Command::Redo => {
            let Some(session_id) = state.current_session_id().await else {
                send_error(srv_tx, "cannot redo before a session is loaded");
                return Ok(());
            };
            let response = call_querymt_ext(
                connection,
                "querymt/session/redo",
                json!({ "session_id": session_id }),
            )
            .await?;
            if let Ok(result) =
                serde_json::from_value::<RedoResultData>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::RedoResult(redo_result_from_wire(result)),
                );
            }
        }
        Command::ListRemoteNodes => {
            let status_resp =
                call_querymt_ext(connection, "querymt/mesh/status", json!({})).await?;
            if let Ok(status) =
                serde_json::from_value::<MeshStatusDto>(ext_payload(&status_resp).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::MeshStatus(mesh_status_from_wire(status)),
                );
            }
            let nodes_resp = call_querymt_ext(connection, "querymt/mesh/nodes", json!({})).await?;
            if let Ok(nodes) =
                serde_json::from_value::<MeshNodesDto>(ext_payload(&nodes_resp).clone())
            {
                send_acp(srv_tx, AcpAppEvent::MeshNodes(mesh_nodes_from_wire(nodes)));
            }
        }
        Command::ListRemoteSessions {
            node_id,
            offset,
            limit,
        } => {
            let response = call_querymt_ext(
                connection,
                "querymt/remote/sessions",
                json!({ "node_id": node_id, "offset": offset, "limit": limit }),
            )
            .await?;
            if let Ok(list) =
                serde_json::from_value::<RemoteSessionListDto>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::RemoteSessions(remote_session_list_from_wire(list)),
                );
            }
        }
        Command::CreateRemoteSession { node_id, cwd } => {
            let response = call_querymt_ext(
                connection,
                "querymt/remote/createSession",
                json!({ "node_id": node_id, "cwd": cwd, "attach": true }),
            )
            .await?;
            if let Ok(attached) =
                serde_json::from_value::<RemoteSessionAttachDto>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::RemoteSessionAttached(remote_session_attach_from_wire(attached)),
                );
            }
        }
        Command::AttachRemoteSession {
            node_id,
            session_id,
        } => {
            let response = call_querymt_ext(
                connection,
                "querymt/remote/attachSession",
                json!({ "node_id": node_id, "session_id": session_id }),
            )
            .await?;
            if let Ok(attached) =
                serde_json::from_value::<RemoteSessionAttachDto>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::RemoteSessionAttached(remote_session_attach_from_wire(attached)),
                );
            }
        }
        Command::CreateMeshInvite {
            mesh_name,
            ttl,
            max_uses,
        } => {
            let response = call_querymt_ext(
                connection,
                "querymt/mesh/createInvite",
                json!({ "mesh_name": mesh_name, "ttl": ttl, "max_uses": max_uses }),
            )
            .await?;
            if let Ok(invite) =
                serde_json::from_value::<MeshInviteCreatedDto>(ext_payload(&response).clone())
            {
                send_acp(
                    srv_tx,
                    AcpAppEvent::MeshInviteCreated(mesh_invite_created_from_wire(invite)),
                );
            }
        }
        Command::ListSessionChildren { .. }
        | Command::DismissRemoteSession { .. }
        | Command::SetApiToken { .. }
        | Command::ClearApiToken { .. } => {
            // TODO(ACP parity): these actions relied on QueryMT UI-API-only
            // methods. Keep them explicit instead of silently falling back.
            send_error(
                srv_tx,
                format!("unsupported in the current ACP subset: {cmd:?}"),
            );
        }
    }

    Ok(())
}

async fn set_config_option<C: AcpConnection>(
    connection: &C,
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    config_id: &str,
    value: &str,
    meta: Option<serde_json::Map<String, Value>>,
) -> Result<(), acp_sdk::Error> {
    let Some(session_id) = state.current_session_id().await else {
        send_error(
            srv_tx,
            format!("cannot set {config_id} before a session is loaded"),
        );
        return Ok(());
    };
    let response = connection
        .request(
            acp::SetSessionConfigOptionRequest::new(
                session_id,
                config_id.to_string(),
                acp::SessionConfigOptionValue::from(value),
            )
            .meta(meta),
        )
        .await?;
    send_config_updates(state, srv_tx, response.config_options).await;
    Ok(())
}

async fn call_querymt_ext<C: AcpConnection>(
    connection: &C,
    method: &str,
    params: Value,
) -> Result<Value, acp_sdk::Error> {
    let wire_method = format!("_{method}");
    connection
        .request(UntypedMessage::new(&wire_method, params)?)
        .await
}

fn ext_payload(response: &Value) -> &Value {
    response.get("data").unwrap_or(response)
}

async fn load_acp_profiles<C: AcpConnection>(
    connection: &C,
) -> Result<AcpProfilesResponse, acp_sdk::Error> {
    let response = call_querymt_ext(connection, "querymt/profiles", json!({})).await?;
    normalize_profiles_response(response).map_err(acp_sdk::Error::into_internal_error)
}

fn normalize_profiles_response(response: Value) -> Result<AcpProfilesResponse, serde_json::Error> {
    serde_json::from_value(ext_payload(&response).clone())
}

async fn load_acp_profile_agents<C: AcpConnection>(
    connection: &C,
    profile_id: &str,
) -> Result<AcpProfileAgentsResponse, acp_sdk::Error> {
    let response = call_querymt_ext(
        connection,
        "querymt/profile/agents",
        json!({ "profile_id": profile_id }),
    )
    .await?;
    normalize_profile_agents_response(response).map_err(acp_sdk::Error::into_internal_error)
}

fn send_profiles(srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>, response: AcpProfilesResponse) {
    send_acp(
        srv_tx,
        AcpAppEvent::Profiles {
            profiles: response.profiles,
            active_profile_id: response.active_profile_id,
        },
    );
}

fn load_session_cwd(cwd: Option<&str>, default_cwd: PathBuf) -> PathBuf {
    cwd.and_then(|cwd| (!cwd.trim().is_empty()).then(|| PathBuf::from(cwd)))
        .unwrap_or(default_cwd)
}

fn mesh_status_from_wire(status: MeshStatusDto) -> MeshStatusInfo {
    MeshStatusInfo {
        enabled: status.enabled,
        peer_id: status.peer_id,
        transport: status.transport,
        known_peer_count: status.known_peer_count,
        has_invite_store: status.has_invite_store,
        has_mesh_state_store: status.has_mesh_state_store,
        scopes: status
            .scopes
            .into_iter()
            .map(mesh_scope_from_wire)
            .collect(),
    }
}

fn mesh_scope_from_wire(scope: MeshScopeDto) -> MeshScopeInfo {
    MeshScopeInfo {
        kind: scope.kind,
        id: scope.id,
    }
}

fn remote_node_from_wire(node: RemoteNodeDto) -> RemoteNodeInfo {
    RemoteNodeInfo {
        id: node.id,
        label: node.label,
        capabilities: node.capabilities,
        active_sessions: node.active_sessions,
        transport: node.transport,
        last_seen_at: node.last_seen_at,
    }
}

fn mesh_nodes_from_wire(nodes: MeshNodesDto) -> MeshNodesInfo {
    MeshNodesInfo {
        nodes: nodes.nodes.into_iter().map(remote_node_from_wire).collect(),
    }
}

fn remote_session_from_wire(session: RemoteSessionDto) -> RemoteSessionInfo {
    RemoteSessionInfo {
        id: session.id,
        node_id: session.node_id,
        node_label: session.node_label,
        title: session.title,
        cwd: session.cwd,
        updated_at: session.updated_at,
        profile_id: session.profile_id,
        model_id: session.model_id,
    }
}

fn remote_session_list_from_wire(list: RemoteSessionListDto) -> RemoteSessionListInfo {
    RemoteSessionListInfo {
        node_id: list.node_id,
        sessions: list
            .sessions
            .into_iter()
            .map(remote_session_from_wire)
            .collect(),
        next_offset: list.next_offset,
        total_count: list.total_count,
    }
}

fn remote_session_attach_from_wire(attached: RemoteSessionAttachDto) -> RemoteSessionAttachInfo {
    RemoteSessionAttachInfo {
        session_id: attached.session_id,
        node_id: attached.node_id,
        attached: attached.attached,
        config_options: attached.config_options,
        snapshot: attached.snapshot,
    }
}

fn mesh_invite_created_from_wire(invite: MeshInviteCreatedDto) -> MeshInviteCreatedInfo {
    MeshInviteCreatedInfo {
        invite_id: invite.invite_id,
        url: invite.url,
        qr_code: invite.qr_code,
        expires_at: invite.expires_at,
        max_uses: invite.max_uses,
        mesh_name: invite.mesh_name,
    }
}

fn delegation_update_from_wire(update: DelegationUpdateDto) -> Option<DelegationUpdate> {
    if update.version != 1 {
        return None;
    }

    Some(DelegationUpdate {
        session_id: update.session_id,
        delegation_id: update.delegation_id,
        tool_call_id: update.tool_call_id,
        state: match update.state {
            DelegationUpdateStateDto::Requested => DelegationState::Requested,
            DelegationUpdateStateDto::Forked => DelegationState::Forked,
            DelegationUpdateStateDto::Completed => DelegationState::Completed,
            DelegationUpdateStateDto::Failed => DelegationState::Failed,
            DelegationUpdateStateDto::Cancelled => DelegationState::Cancelled,
        },
        target_agent_id: update.target_agent_id,
        objective: update.objective,
        child_session_id: update.child_session_id,
        requested_at: update.requested_at,
        forked_at: update.forked_at,
        finished_at: update.finished_at,
        updated_at: update.updated_at,
        result_summary: update.result_summary,
        error: update.error,
    })
}

fn oauth_flow_from_wire(flow: OAuthFlowDto) -> OAuthFlow {
    OAuthFlow {
        flow_id: flow.flow_id,
        provider: flow.provider,
        authorization_url: flow.authorization_url,
        flow_kind: flow.flow_kind,
    }
}

fn oauth_result_from_wire(result: OAuthResultDto) -> OAuthResult {
    OAuthResult {
        provider: result.provider,
        status: if result.success {
            OAuthResultStatus::Success
        } else {
            OAuthResultStatus::Failure
        },
        message: result.message,
    }
}

fn undo_stack_snapshot_from_wire(frames: Vec<UndoStackFrame>) -> UndoStackSnapshot {
    UndoStackSnapshot {
        message_ids: frames.into_iter().map(|frame| frame.message_id).collect(),
    }
}

fn undo_result_from_wire(result: UndoResultData) -> UndoResult {
    let stack = undo_stack_snapshot_from_wire(result.undo_stack);
    if result.success {
        UndoResult::Applied {
            target_message_id: result.message_id,
            reverted_files: result.reverted_files,
            message: result.message,
            stack,
        }
    } else {
        UndoResult::Rejected {
            target_message_id: result.message_id,
            message: result.message,
            stack,
        }
    }
}

fn redo_result_from_wire(result: RedoResultData) -> RedoResult {
    let stack = undo_stack_snapshot_from_wire(result.undo_stack);
    if result.success {
        RedoResult::Applied {
            message: result.message,
            stack,
        }
    } else {
        RedoResult::Rejected {
            message: result.message,
            stack,
        }
    }
}

async fn fetch_undo_stack<C: AcpConnection>(
    connection: &C,
    session_id: &str,
) -> Result<UndoStackSnapshot, acp_sdk::Error> {
    let response = call_querymt_ext(
        connection,
        "querymt/session/undoStack",
        json!({ "session_id": session_id }),
    )
    .await?;
    let frames = ext_payload(&response)
        .get("undo_stack")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| serde_json::from_value::<UndoStackFrame>(item.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    Ok(undo_stack_snapshot_from_wire(frames))
}

async fn post_connect_diagnostics<C: AcpConnection>(
    connection: &C,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
) {
    match call_querymt_ext(connection, "querymt/capabilities", json!({})).await {
        Ok(response) => {
            let payload = ext_payload(&response).clone();
            send_acp(srv_tx, AcpAppEvent::ControlCapabilities(payload.clone()));
            let methods = payload
                .get("methods")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if methods.iter().any(|m| m == "querymt/mesh/nodes")
                && let Ok(nodes_resp) =
                    call_querymt_ext(connection, "querymt/mesh/nodes", json!({})).await
                && let Ok(nodes) =
                    serde_json::from_value::<MeshNodesDto>(ext_payload(&nodes_resp).clone())
            {
                send_acp(srv_tx, AcpAppEvent::MeshNodes(mesh_nodes_from_wire(nodes)));
            }
            if methods.iter().any(|m| m == "querymt/profiles") {
                match load_acp_profiles(connection).await {
                    Ok(profiles) => send_profiles(srv_tx, profiles),
                    Err(err) => send_error(srv_tx, format!("failed to load profiles: {err}")),
                }
            } else if payload
                .get("features")
                .and_then(|features| features.get("profiles"))
                .and_then(Value::as_bool)
                == Some(false)
            {
                send_profiles(srv_tx, AcpProfilesResponse::default());
            }
        }
        Err(err) => {
            send_acp(
                srv_tx,
                AcpAppEvent::ControlCapabilitiesUnavailable(err.to_string()),
            );
        }
    }
}

fn client_capabilities() -> acp::ClientCapabilities {
    acp::ClientCapabilities::new()
        .fs(acp::FileSystemCapabilities::new())
        .terminal(false)
        .elicitation(
            acp::ElicitationCapabilities::new().form(acp::ElicitationFormCapabilities::new()),
        )
}

fn prompt_blocks(blocks: Vec<PromptBlock>) -> Vec<acp::ContentBlock> {
    blocks
        .into_iter()
        .map(|block| match block {
            PromptBlock::Text { text } => acp::ContentBlock::Text(acp::TextContent::new(text)),
            PromptBlock::ResourceLink { name, uri } => {
                acp::ContentBlock::ResourceLink(acp::ResourceLink::new(name, uri))
            }
        })
        .collect()
}

fn profile_meta(profile_id: &str) -> serde_json::Map<String, Value> {
    let mut meta = serde_json::Map::new();
    meta.insert("querymt".to_string(), json!({ "profile_id": profile_id }));
    meta
}

fn fork_session_meta(message_id: &str) -> serde_json::Map<String, Value> {
    let mut meta = serde_json::Map::new();
    // ACP session/fork has no native fork-point field. QueryMT agents honor
    // this metadata hint and fall back to latest-message fork when it is absent.
    meta.insert("querymt".to_string(), json!({ "message_id": message_id }));
    meta
}

const SESSION_LOAD_SNAPSHOT_META_KEY: &str = "querymt/sessionLoadSnapshot.v1";

fn session_load_snapshot_from_load_value(response: &Value) -> Option<&Value> {
    response
        .get("_meta")
        .or_else(|| response.get("meta"))
        .and_then(|meta| meta.get(SESSION_LOAD_SNAPSHOT_META_KEY))
}

fn session_load_audit_from_load_value(response: &Value) -> Value {
    let snapshot = session_load_snapshot_from_load_value(response);
    match snapshot {
        Some(snapshot) => snapshot
            .get("audit")
            .cloned()
            .unwrap_or_else(|| json!({ "events": [] })),
        None => json!({ "events": [] }),
    }
}

fn delegation_updates_from_load_value(response: &Value) -> Vec<DelegationUpdate> {
    session_load_snapshot_from_load_value(response)
        .and_then(|snapshot| snapshot.get("delegationUpdates"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|update| serde_json::from_value::<DelegationUpdateDto>(update.clone()).ok())
        .filter_map(delegation_update_from_wire)
        .collect()
}

fn snapshot_provider_change_from_load_value(response: &Value) -> Option<SnapshotProviderChange> {
    let audit = session_load_audit_from_load_value(response);
    let events = audit.get("events").and_then(Value::as_array)?;
    events
        .iter()
        .filter_map(snapshot_provider_change_from_event)
        .next_back()
}

fn snapshot_provider_change_from_event(event: &Value) -> Option<SnapshotProviderChange> {
    let kind = event.get("kind")?;
    if kind.get("type").and_then(Value::as_str) != Some("provider_changed") {
        return None;
    }
    let data = kind.get("data")?;
    let provider = snapshot_string_field(data, "provider")?;
    let model = snapshot_string_field(data, "model")?;
    Some(SnapshotProviderChange {
        provider,
        model,
        context_limit: snapshot_u64(data, "context_limit"),
        provider_node_id: snapshot_string_field(data, "provider_node_id"),
    })
}

fn snapshot_provider_change_event(change: SnapshotProviderChange) -> AcpAppEvent {
    AcpAppEvent::ProviderChanged {
        provider: change.provider,
        model: change.model,
        context_limit: change.context_limit,
        provider_node_id: change.provider_node_id,
    }
}

fn merge_replay_with_snapshot_stats(
    replay_updates: Vec<AcpSessionUpdate>,
    snapshot_updates: Vec<AcpSessionUpdate>,
) -> Vec<AcpSessionUpdate> {
    if replay_updates.is_empty() {
        return snapshot_updates;
    }

    let replay_has_usage = replay_updates
        .iter()
        .any(|update| matches!(update, AcpSessionUpdate::UsageUpdate { .. }));
    let replay_has_timing = replay_updates
        .iter()
        .any(|update| matches!(update, AcpSessionUpdate::TimingUpdate { .. }));

    let mut merged = replay_updates;
    // Backend ACP replay currently sends visual history but not usage stats.
    // Snapshot stats come from QueryMT load metadata and keep status-bar context
    // accurate for loaded/forked sessions without duplicating chat entries.
    merged.extend(snapshot_updates.into_iter().filter(|update| match update {
        AcpSessionUpdate::UsageUpdate { .. } => !replay_has_usage,
        AcpSessionUpdate::TimingUpdate { .. } => !replay_has_timing,
        _ => false,
    }));
    merged
}

fn snapshot_updates_from_load_value(response: &Value) -> Vec<AcpSessionUpdate> {
    let audit = session_load_audit_from_load_value(response);
    let Some(events) = audit.get("events").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut updates = Vec::new();
    let mut context_limit = 0;
    let mut llm_started_at: Option<i64> = None;
    for event in events {
        let Some(kind) = event.get("kind") else {
            continue;
        };
        let Some(kind_type) = kind.get("type").and_then(Value::as_str) else {
            continue;
        };
        let data = kind.get("data").unwrap_or(&Value::Null);
        let timestamp = snapshot_i64(event, "timestamp");
        match kind_type {
            "llm_request_start" => {
                llm_started_at = timestamp;
            }
            "provider_changed" => {
                if let Some(limit) = snapshot_u64(data, "context_limit") {
                    context_limit = limit;
                    if limit > 0 {
                        updates.push(AcpSessionUpdate::UsageUpdate {
                            used: 0,
                            size: limit,
                            cost_usd: None,
                        });
                    }
                }
            }
            "llm_request_end" => {
                if let (Some(started), Some(ended)) = (llm_started_at.take(), timestamp)
                    && ended >= started
                {
                    updates.push(AcpSessionUpdate::TimingUpdate {
                        duration_secs: (ended - started) as u64,
                    });
                }
                updates.push(AcpSessionUpdate::UsageUpdate {
                    used: snapshot_u64(data, "context_tokens").unwrap_or(0),
                    size: context_limit,
                    cost_usd: snapshot_f64(data, "cumulative_cost_usd"),
                });
                updates.push(AcpSessionUpdate::Finished {
                    finish_reason: snapshot_string(data, "finish_reason")
                        .unwrap_or_else(|| "completed".to_string()),
                });
            }
            "cancelled" | "error" => {
                if let (Some(started), Some(ended)) = (llm_started_at.take(), timestamp)
                    && ended >= started
                {
                    updates.push(AcpSessionUpdate::TimingUpdate {
                        duration_secs: (ended - started) as u64,
                    });
                }
                if let Some(update) = snapshot_event_to_update(kind_type, data) {
                    updates.push(update);
                }
            }
            _ => {
                if let Some(update) = snapshot_event_to_update(kind_type, data) {
                    updates.push(update);
                }
            }
        }
    }
    updates
}

fn snapshot_event_to_update(kind_type: &str, data: &Value) -> Option<AcpSessionUpdate> {
    match kind_type {
        "prompt_received" => Some(AcpSessionUpdate::UserMessage {
            content: data.get("content").cloned().unwrap_or(Value::Null),
            message_id: snapshot_string(data, "message_id"),
        }),
        "assistant_content_delta" => Some(AcpSessionUpdate::AssistantContentDelta {
            content: snapshot_string(data, "content").unwrap_or_default(),
            message_id: snapshot_string(data, "message_id"),
        }),
        "assistant_thinking_delta" => Some(AcpSessionUpdate::AssistantThinkingDelta {
            content: snapshot_string(data, "content").unwrap_or_default(),
            message_id: snapshot_string(data, "message_id"),
        }),
        "assistant_message_stored" => Some(AcpSessionUpdate::AssistantMessage {
            content: snapshot_string(data, "content").unwrap_or_default(),
            thinking: snapshot_string(data, "thinking").filter(|text| !text.is_empty()),
            message_id: snapshot_string(data, "message_id"),
        }),
        "tool_call_start" => Some(AcpSessionUpdate::ToolCallStart {
            tool_call_id: snapshot_string(data, "tool_call_id"),
            name: snapshot_string(data, "tool_name").unwrap_or_else(|| "tool".to_string()),
            arguments: data.get("arguments").or_else(|| data.get("input")).cloned(),
        }),
        "tool_call_end" => Some(AcpSessionUpdate::ToolCallEnd {
            tool_call_id: snapshot_string(data, "tool_call_id"),
            name: snapshot_string(data, "tool_name").unwrap_or_else(|| "tool".to_string()),
            is_error: data
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            result: snapshot_string(data, "result")
                .or_else(|| snapshot_string(data, "output"))
                .or_else(|| snapshot_string(data, "content")),
        }),
        "cancelled" => Some(AcpSessionUpdate::Cancelled),
        _ => None,
    }
}

fn snapshot_i64(data: &Value, key: &str) -> Option<i64> {
    match data.get(key)? {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn snapshot_u64(data: &Value, key: &str) -> Option<u64> {
    match data.get(key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn snapshot_f64(data: &Value, key: &str) -> Option<f64> {
    match data.get(key)? {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn snapshot_string(data: &Value, key: &str) -> Option<String> {
    match data.get(key)? {
        Value::String(text) => Some(text.clone()),
        Value::Null => None,
        other => serde_json::to_string(other).ok(),
    }
}

fn snapshot_string_field(data: &Value, key: &str) -> Option<String> {
    match data.get(key)? {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

fn profile_id_from_config_options(config_options: &[acp::SessionConfigOption]) -> Option<String> {
    let options_json = serde_json::to_value(config_options).ok()?;
    let options = options_json.as_array()?;
    for option in options {
        let id = option.get("id").and_then(Value::as_str).unwrap_or_default();
        let category = option
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let name = option
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if normalize_config_key(id, category, name) == "profile" {
            return option
                .get("currentValue")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
    }
    None
}

async fn load_acp_models<C: AcpConnection>(
    connection: &C,
    refresh: bool,
) -> Result<AcpModelsResponse, acp_sdk::Error> {
    let mut response = if refresh {
        call_acp_models(connection, true).await?
    } else {
        call_acp_models(connection, false).await?
    };

    if !refresh && response.should_retry_empty() {
        for attempt in 0..3 {
            tokio::time::sleep(std::time::Duration::from_millis(250 * (attempt + 1))).await;
            response = call_acp_models(connection, false).await?;
            if !response.should_retry_empty() {
                return Ok(response);
            }
        }
        response = call_acp_models(connection, true).await?;
    }

    if response.should_retry_empty() {
        for attempt in 0..3 {
            tokio::time::sleep(std::time::Duration::from_millis(300 * (attempt + 1))).await;
            response = call_acp_models(connection, false).await?;
            if !response.should_retry_empty() {
                break;
            }
        }
    }

    Ok(response)
}

async fn call_acp_models<C: AcpConnection>(
    connection: &C,
    refresh: bool,
) -> Result<AcpModelsResponse, acp_sdk::Error> {
    let method = if refresh {
        "querymt/refreshModels"
    } else {
        "querymt/models"
    };
    let params = if refresh {
        json!({ "wait_for_completion": true })
    } else {
        json!({})
    };
    let response = call_querymt_ext(connection, method, params).await?;
    Ok(normalize_models_response(response))
}

fn normalize_models_response(response: Value) -> AcpModelsResponse {
    let payload = response.get("data").unwrap_or(&response);
    let models = payload
        .get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|model| serde_json::from_value::<AcpModelEntry>(model.clone()).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let meta = payload
        .get("meta")
        .or_else(|| response.get("meta"))
        .and_then(|meta| serde_json::from_value::<AcpModelsMeta>(meta.clone()).ok());

    AcpModelsResponse { models, meta }
}

async fn handle_session_notification(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    notification: acp::SessionNotification,
) {
    let session_id = notification.session_id.to_string();
    let loading = state.is_loading(&session_id).await;

    match notification.update {
        acp::SessionUpdate::UserMessageChunk(chunk) => {
            emit_or_queue_update(
                state,
                srv_tx,
                &session_id,
                AcpSessionUpdate::UserMessage {
                    content: content_block_to_json(&chunk.content),
                    message_id: chunk.message_id.map(|id| id.to_string()),
                },
                loading,
            )
            .await;
        }
        acp::SessionUpdate::AgentMessageChunk(chunk) => {
            let text = content_block_text(&chunk.content);
            let message_id = chunk.message_id.as_ref().map(ToString::to_string);
            if loading {
                emit_or_queue_update(
                    state,
                    srv_tx,
                    &session_id,
                    AcpSessionUpdate::AssistantMessage {
                        content: text,
                        thinking: None,
                        message_id,
                    },
                    true,
                )
                .await;
            } else {
                flush_assistant_buffer_for_message(
                    state,
                    srv_tx,
                    &session_id,
                    message_id.as_deref(),
                )
                .await;
                remember_assistant_chunk(state, &session_id, message_id.clone(), &text, false)
                    .await;
                send_session_update(
                    srv_tx,
                    &session_id,
                    AcpSessionUpdate::AssistantContentDelta {
                        content: text,
                        message_id,
                    },
                    false,
                );
            }
        }
        acp::SessionUpdate::AgentThoughtChunk(chunk) => {
            let text = content_block_text(&chunk.content);
            let message_id = chunk.message_id.as_ref().map(ToString::to_string);
            if !loading {
                flush_assistant_buffer_for_message(
                    state,
                    srv_tx,
                    &session_id,
                    message_id.as_deref(),
                )
                .await;
                remember_assistant_chunk(state, &session_id, message_id.clone(), &text, true).await;
            }
            emit_or_queue_update(
                state,
                srv_tx,
                &session_id,
                AcpSessionUpdate::AssistantThinkingDelta {
                    content: text,
                    message_id,
                },
                loading,
            )
            .await;
        }
        acp::SessionUpdate::ToolCall(tool_call) => {
            if !loading {
                flush_assistant_buffer(state, srv_tx, &session_id).await;
            }
            emit_or_queue_update(
                state,
                srv_tx,
                &session_id,
                tool_start_update(&tool_call),
                loading,
            )
            .await;
        }
        acp::SessionUpdate::ToolCallUpdate(update) => {
            emit_or_queue_update(
                state,
                srv_tx,
                &session_id,
                tool_call_update(update),
                loading,
            )
            .await;
        }
        acp::SessionUpdate::CurrentModeUpdate(update) => {
            send_acp(
                srv_tx,
                AcpAppEvent::AgentMode {
                    mode: update.current_mode_id.to_string(),
                },
            );
        }
        acp::SessionUpdate::ConfigOptionUpdate(update) => {
            send_config_updates(state, srv_tx, update.config_options).await;
        }
        acp::SessionUpdate::UsageUpdate(update) => {
            emit_or_queue_update(state, srv_tx, &session_id, usage_update(update), loading).await;
        }
        // TODO(ACP parity): map native session info into sidebar/header state.
        acp::SessionUpdate::SessionInfoUpdate(_) => {}
        // TODO(ACP parity): add native plan/available-command rendering once the TUI UX is defined.
        acp::SessionUpdate::Plan(_) | acp::SessionUpdate::AvailableCommandsUpdate(_) => {}
        _ => {}
    }
}

async fn remember_assistant_chunk(
    state: &Arc<AcpRuntimeState>,
    session_id: &str,
    message_id: Option<String>,
    text: &str,
    thinking: bool,
) {
    if text.is_empty() {
        return;
    }
    let mut buffers = state.assistant_buffers.lock().await;
    let buffer = buffers.entry(session_id.to_string()).or_default();
    if thinking {
        if buffer.thinking_message_id.is_none() && message_id.is_some() {
            buffer.thinking_message_id = message_id;
        }
        buffer.thinking.push_str(text);
    } else {
        if buffer.content_message_id.is_none() && message_id.is_some() {
            buffer.content_message_id = message_id;
        }
        buffer.content.push_str(text);
    }
}

fn assistant_buffer_message_id(buffer: &AssistantBuffer) -> Option<String> {
    buffer
        .content_message_id
        .clone()
        .or_else(|| buffer.thinking_message_id.clone())
}

fn assistant_buffer_has_different_message(
    buffer: &AssistantBuffer,
    incoming_message_id: Option<&str>,
) -> bool {
    let Some(incoming) = incoming_message_id else {
        return false;
    };
    assistant_buffer_message_id(buffer)
        .as_deref()
        .is_some_and(|current| current != incoming)
}

async fn flush_assistant_buffer_for_message(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    session_id: &str,
    incoming_message_id: Option<&str>,
) {
    let should_flush = state
        .assistant_buffers
        .lock()
        .await
        .get(session_id)
        .is_some_and(|buffer| assistant_buffer_has_different_message(buffer, incoming_message_id));
    if should_flush {
        flush_assistant_buffer(state, srv_tx, session_id).await;
    }
}

async fn flush_assistant_buffer(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    session_id: &str,
) {
    let buffer = state.assistant_buffers.lock().await.remove(session_id);
    if let Some(buffer) = buffer
        && (!buffer.content.is_empty() || !buffer.thinking.is_empty())
    {
        let message_id = assistant_buffer_message_id(&buffer);
        send_session_update(
            srv_tx,
            session_id,
            AcpSessionUpdate::AssistantMessage {
                content: buffer.content,
                thinking: (!buffer.thinking.is_empty()).then_some(buffer.thinking),
                message_id,
            },
            false,
        );
    }
}

async fn finish_prompt(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    session_id: &str,
    stop_reason: acp::StopReason,
) {
    flush_assistant_buffer(state, srv_tx, session_id).await;

    if matches!(stop_reason, acp::StopReason::Cancelled) {
        send_session_update(srv_tx, session_id, AcpSessionUpdate::Cancelled, false);
    } else {
        send_session_update(
            srv_tx,
            session_id,
            AcpSessionUpdate::Finished {
                finish_reason: format!("{stop_reason:?}"),
            },
            false,
        );
    }
}

fn tool_start_update(tool_call: &acp::ToolCall) -> AcpSessionUpdate {
    AcpSessionUpdate::ToolCallStart {
        tool_call_id: Some(tool_call.tool_call_id.to_string()),
        name: tool_name_from_title(&tool_call.title),
        arguments: tool_call.raw_input.clone(),
    }
}

fn usage_update(update: acp::UsageUpdate) -> AcpSessionUpdate {
    let cost_usd = update
        .cost
        .filter(|cost| cost.currency.eq_ignore_ascii_case("USD"))
        .map(|cost| cost.amount);
    AcpSessionUpdate::UsageUpdate {
        used: update.used,
        size: update.size,
        cost_usd,
    }
}

fn tool_call_update(update: acp::ToolCallUpdate) -> AcpSessionUpdate {
    let status = update.fields.status;
    let tool_name = update
        .fields
        .title
        .as_deref()
        .map(tool_name_from_title)
        .unwrap_or_else(|| "tool".to_string());

    if matches!(
        status,
        Some(acp::ToolCallStatus::Completed | acp::ToolCallStatus::Failed)
    ) {
        AcpSessionUpdate::ToolCallEnd {
            tool_call_id: Some(update.tool_call_id.to_string()),
            name: tool_name,
            is_error: matches!(status, Some(acp::ToolCallStatus::Failed)),
            result: tool_update_result(&update.fields),
        }
    } else {
        AcpSessionUpdate::ToolCallStart {
            tool_call_id: Some(update.tool_call_id.to_string()),
            name: tool_name,
            arguments: update.fields.raw_input.clone(),
        }
    }
}

fn tool_update_result(fields: &acp::ToolCallUpdateFields) -> Option<String> {
    if let Some(value) = fields.raw_output.as_ref() {
        return Some(value_to_text(value));
    }
    fields.content.as_ref().map(|content| {
        content
            .iter()
            .map(|entry| {
                serde_json::to_value(entry)
                    .map(|value| value_to_text(&value))
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn tool_name_from_title(title: &str) -> String {
    title.strip_prefix("Run ").unwrap_or(title).to_string()
}

async fn send_config_updates(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    config_options: Vec<acp::SessionConfigOption>,
) {
    let options_json = serde_json::to_value(&config_options).unwrap_or(Value::Null);
    let Some(options) = options_json.as_array() else {
        return;
    };

    let mut profiles = Vec::new();
    let mut active_profile_id = None;

    for option in options {
        let id = option.get("id").and_then(Value::as_str).unwrap_or_default();
        let category = option
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let name = option
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let current = option.get("currentValue").and_then(Value::as_str);
        let normalized = normalize_config_key(id, category, name);

        if normalized == "model" {
            if let Some(model_id) = current {
                state.select_model(model_id).await;
                if let Some(model) = state
                    .model_by_id(model_id)
                    .await
                    .or_else(|| model_entry_from_config_option(option, model_id))
                {
                    send_provider_changed(srv_tx, &model);
                }
            }
        } else if normalized == "mode" {
            if let Some(mode) = current {
                send_acp(
                    srv_tx,
                    AcpAppEvent::AgentMode {
                        mode: mode.to_string(),
                    },
                );
            }
        } else if matches!(
            normalized.as_str(),
            "thought_level" | "reasoning" | "reasoning_effort" | "thought"
        ) {
            if let Some(effort) = current {
                send_acp(
                    srv_tx,
                    AcpAppEvent::ReasoningEffort {
                        reasoning_effort: Some(effort.to_string()),
                    },
                );
            }
        } else if normalized == "profile" {
            active_profile_id = current.map(str::to_string);
            profiles = profile_infos_from_option(option);
        }
    }

    if !profiles.is_empty() {
        send_acp(
            srv_tx,
            AcpAppEvent::Profiles {
                profiles,
                active_profile_id,
            },
        );
    }
}

fn model_entry_meta(
    model: &AcpModelEntry,
    node_id_override: Option<&str>,
) -> Option<serde_json::Map<String, Value>> {
    let mut entry = serde_json::to_value(model).ok()?;
    if let Some(node_id) = node_id_override
        && let Some(object) = entry.as_object_mut()
    {
        object.insert("node_id".to_string(), Value::String(node_id.to_string()));
    }
    let mut meta = serde_json::Map::new();
    meta.insert("querymt".to_string(), json!({ "modelEntry": entry }));
    Some(meta)
}

fn fallback_model_entry(model_id: &str) -> AcpModelEntry {
    let (provider, model) = model_id
        .split_once('/')
        .map(|(provider, model)| (provider.to_string(), model.to_string()))
        .unwrap_or_else(|| ("unknown".to_string(), model_id.to_string()));
    AcpModelEntry {
        id: model_id.to_string(),
        label: model.to_string(),
        source: Some("qmtui-fallback".to_string()),
        provider,
        model,
        node_id: None,
        node_label: None,
        family: None,
        quant: None,
    }
}

fn model_entry_from_config_option(option: &Value, model_id: &str) -> Option<AcpModelEntry> {
    flatten_select_entries(option)
        .into_iter()
        .find(|entry| entry.get("value").and_then(Value::as_str) == Some(model_id))
        .map(|entry| {
            let label = entry
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(model_id)
                .to_string();
            let (provider, model) = model_id
                .split_once('/')
                .map(|(provider, model)| (provider.to_string(), model.to_string()))
                .unwrap_or_else(|| ("unknown".to_string(), label.clone()));
            AcpModelEntry {
                id: model_id.to_string(),
                label,
                source: None,
                provider,
                model,
                node_id: None,
                node_label: None,
                family: None,
                quant: None,
            }
        })
}

fn send_provider_changed(srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>, model: &AcpModelEntry) {
    send_acp(
        srv_tx,
        AcpAppEvent::ProviderChanged {
            provider: model.provider.clone(),
            model: model.model.clone(),
            context_limit: None,
            provider_node_id: model.node_id.clone(),
        },
    );
}

fn normalize_config_key(id: &str, category: &str, name: &str) -> String {
    for value in [id, category, name] {
        let normalized = value
            .to_ascii_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
            .trim_matches('_')
            .to_string();
        if !normalized.is_empty() {
            return normalized;
        }
    }
    String::new()
}

fn profile_infos_from_option(option: &Value) -> Vec<ProfileInfo> {
    flatten_select_entries(option)
        .into_iter()
        .filter_map(|entry| {
            let id = entry.get("value")?.as_str()?;
            let name = entry.get("name").and_then(Value::as_str).unwrap_or(id);
            Some(ProfileInfo {
                id: id.to_string(),
                name: name.to_string(),
                description: entry
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                tags: Vec::new(),
                source: None,
                config_kind: None,
                fingerprint: None,
            })
        })
        .collect()
}

fn flatten_select_entries(option: &Value) -> Vec<&Value> {
    option
        .get("options")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|entry| {
            if let Some(group_options) = entry.get("options").and_then(Value::as_array) {
                group_options.iter().collect::<Vec<_>>()
            } else {
                vec![entry]
            }
        })
        .collect()
}

fn session_list_page_from_acp(
    request: &SessionListRequest,
    response: acp::ListSessionsResponse,
) -> SessionListPage {
    let mut groups: BTreeMap<Option<String>, Vec<SessionSummary>> = BTreeMap::new();
    let response_cursor = response.next_cursor.map(|cursor| cursor.to_string());

    for session in response.sessions {
        let cwd = session.cwd.to_string_lossy().to_string();
        let group_key = request
            .cwd()
            .map(str::to_string)
            .or_else(|| (!cwd.is_empty()).then_some(cwd.clone()));
        groups.entry(group_key).or_default().push(SessionSummary {
            session_id: session.session_id.to_string(),
            name: session.title.clone(),
            title: session.title,
            cwd: (!cwd.is_empty()).then_some(cwd),
            created_at: None,
            updated_at: session.updated_at,
            parent_session_id: None,
            fork_origin: None,
            session_kind: None,
            has_children: false,
            fork_count: 0,
            children: Vec::new(),
            children_next_cursor: None,
            children_total_count: None,
            node: None,
            node_id: None,
            attached: None,
            runtime_state: None,
        });
    }

    if let Some(cwd) = request.cwd() {
        groups.entry(Some(cwd.to_string())).or_default();
    }

    let workspace_cursor = request
        .cwd()
        .is_some()
        .then_some(response_cursor.clone())
        .flatten();
    let groups = groups
        .into_iter()
        .map(|(cwd, sessions)| SessionGroup {
            cwd,
            latest_activity: sessions
                .first()
                .and_then(|session| session.updated_at.clone()),
            total_count: None,
            next_cursor: workspace_cursor.clone(),
            sessions,
        })
        .collect::<Vec<_>>();

    SessionListPage {
        groups,
        next_cursor: response_cursor,
        total_count: None,
    }
}

fn send_session_list(
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    request: SessionListRequest,
    response: acp::ListSessionsResponse,
) {
    let page = session_list_page_from_acp(&request, response);
    send_acp(srv_tx, AcpAppEvent::SessionList { request, page });
}

fn send_state(
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    identity: &AgentIdentity,
    profiles: Vec<ProfileInfo>,
    active_profile_id: Option<String>,
    agent_mode: Option<&str>,
    reasoning_effort: Option<Option<String>>,
) {
    send_acp(
        srv_tx,
        AcpAppEvent::Initialized {
            agent_id: identity.id.clone(),
            agent_name: identity.name.clone(),
            profiles,
            active_profile_id,
            agent_mode: agent_mode.map(str::to_string),
            reasoning_effort,
        },
    );
}

async fn handle_elicitation_request(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    request: acp::CreateElicitationRequest,
    responder: acp_sdk::Responder<acp::CreateElicitationResponse>,
) {
    let session_id = elicitation_request_session_id(&request);
    flush_assistant_buffer(state, srv_tx, &session_id).await;
    let responder_id = responder.id();
    let elicitation_id = responder_id
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| responder_id.to_string());
    let (metadata_source, allow_custom) = elicitation_metadata(&request, "acp");
    let (requested_schema, source) = match &request.mode {
        acp::ElicitationMode::Form(form) => (
            serde_json::to_value(&form.requested_schema).unwrap_or_else(|_| json!({})),
            metadata_source,
        ),
        acp::ElicitationMode::Url(url) => (json!({}), format!("acp-url:{}", url.url)),
        _ => (json!({}), metadata_source),
    };

    state.pending_elicitations.lock().await.insert(
        elicitation_id.clone(),
        PendingElicitationResponse::Sdk(responder),
    );

    send_session_update(
        srv_tx,
        &session_id,
        AcpSessionUpdate::ElicitationRequested {
            elicitation_id,
            message: request.message,
            requested_schema,
            source,
            allow_custom,
        },
        false,
    );
}

fn elicitation_request_session_id(request: &acp::CreateElicitationRequest) -> String {
    match &request.mode {
        acp::ElicitationMode::Form(form) => elicitation_scope_session_id(&form.scope),
        acp::ElicitationMode::Url(url) => elicitation_scope_session_id(&url.scope),
        _ => "request".to_string(),
    }
}

fn elicitation_metadata(
    request: &acp::CreateElicitationRequest,
    default_source: &str,
) -> (String, bool) {
    let querymt = request.meta.as_ref().and_then(|meta| meta.get("querymt"));
    let source = querymt
        .and_then(|meta| meta.get("source"))
        .and_then(Value::as_str)
        .unwrap_or(default_source)
        .to_string();
    let allow_custom = source == "builtin:question"
        || querymt
            .and_then(|meta| meta.get("allow_custom").or_else(|| meta.get("allowCustom")))
            .and_then(Value::as_bool)
            .unwrap_or(false);
    (source, allow_custom)
}

fn elicitation_scope_session_id(scope: &acp::ElicitationScope) -> String {
    match scope {
        acp::ElicitationScope::Session(session) => session.session_id.to_string(),
        acp::ElicitationScope::Request(_) => "request".to_string(),
        _ => "request".to_string(),
    }
}

async fn respond_to_elicitation(
    state: &Arc<AcpRuntimeState>,
    elicitation_id: &str,
    action: &str,
    content: Option<Value>,
) {
    let responder = state
        .pending_elicitations
        .lock()
        .await
        .remove(elicitation_id);
    let Some(responder) = responder else {
        return;
    };

    match responder {
        PendingElicitationResponse::Sdk(responder) => {
            let response = match action {
                "accept" => acp::CreateElicitationResponse::new(acp::ElicitationAction::Accept(
                    acp::ElicitationAcceptAction::new().content(elicitation_content(content)),
                )),
                "decline" => acp::CreateElicitationResponse::new(acp::ElicitationAction::Decline),
                _ => acp::CreateElicitationResponse::new(acp::ElicitationAction::Cancel),
            };
            let _ = responder.respond(response);
        }
        PendingElicitationResponse::WsResultMethod(connection) => {
            let _ = connection
                .request_value(
                    "elicitation_result",
                    json!({
                        "elicitation_id": elicitation_id,
                        "action": action,
                        "content": content,
                    }),
                )
                .await;
        }
        PendingElicitationResponse::WsJsonRpcResponse { connection, id } => {
            let result = json!({
                "action": action,
                "content": content,
            });
            let _ = connection.respond_value(id, Ok(result));
        }
    }
}

fn elicitation_content(
    content: Option<Value>,
) -> Option<BTreeMap<String, acp::ElicitationContentValue>> {
    let object = content?.as_object()?.clone();
    let mut result = BTreeMap::new();
    for (key, value) in object {
        if let Some(value) = json_to_elicitation_value(value) {
            result.insert(key, value);
        }
    }
    Some(result)
}

fn json_to_elicitation_value(value: Value) -> Option<acp::ElicitationContentValue> {
    match value {
        Value::String(value) => Some(acp::ElicitationContentValue::String(value)),
        Value::Bool(value) => Some(acp::ElicitationContentValue::Boolean(value)),
        Value::Number(value) => value
            .as_i64()
            .map(acp::ElicitationContentValue::Integer)
            .or_else(|| value.as_f64().map(acp::ElicitationContentValue::Number)),
        Value::Array(values) => Some(acp::ElicitationContentValue::StringArray(
            values
                .into_iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect(),
        )),
        _ => None,
    }
}

fn permission_response_for(
    request: &acp::RequestPermissionRequest,
) -> acp::RequestPermissionResponse {
    let allow = request
        .options
        .iter()
        .find(|option| matches!(option.kind, acp::PermissionOptionKind::AllowOnce))
        .or_else(|| request.options.first());

    match allow {
        Some(option) => {
            acp::RequestPermissionResponse::new(acp::RequestPermissionOutcome::Selected(
                acp::SelectedPermissionOutcome::new(option.option_id.clone()),
            ))
        }
        None => acp::RequestPermissionResponse::new(acp::RequestPermissionOutcome::Cancelled),
    }
}

async fn emit_or_queue_update(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    session_id: &str,
    update: AcpSessionUpdate,
    queue: bool,
) {
    if queue {
        state.queue_replay_update(session_id, update).await;
    } else {
        send_session_update(srv_tx, session_id, update, false);
    }
}

fn send_session_update(
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    session_id: &str,
    update: AcpSessionUpdate,
    is_replay: bool,
) {
    send_acp(
        srv_tx,
        AcpAppEvent::SessionUpdate {
            session_id: session_id.to_string(),
            update,
            is_replay,
        },
    );
}

fn send_models(srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>, response: &AcpModelsResponse) {
    send_acp(
        srv_tx,
        AcpAppEvent::Models {
            models: response
                .models
                .iter()
                .map(AcpModelEntry::to_app_model)
                .collect(),
            meta: response.meta.as_ref().map(|meta| AcpModelsMetaInfo {
                remote_node_count: meta.remote_node_count,
                remote_timeout_count: meta.remote_timeout_count,
            }),
        },
    );
}

fn send_error(srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>, message: impl Into<String>) {
    send_acp(
        srv_tx,
        AcpAppEvent::Error {
            message: message.into(),
        },
    );
}

fn send_acp(srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>, event: AcpAppEvent) {
    let _ = srv_tx.send(ServerChannelMsg::Acp(event));
}

fn content_block_to_json(block: &acp::ContentBlock) -> Value {
    serde_json::to_value(block).unwrap_or(Value::Null)
}

fn content_block_text(block: &acp::ContentBlock) -> String {
    match block {
        acp::ContentBlock::Text(text) => text.text.clone(),
        acp::ContentBlock::ResourceLink(link) => link.uri.clone(),
        other => serde_json::to_value(other)
            .map(|value| value_to_text(&value))
            .unwrap_or_default(),
    }
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn model_value(id: &str, provider: &str, model: &str) -> Value {
        json!({
            "id": id,
            "label": model,
            "source": "catalog",
            "provider": provider,
            "model": model,
        })
    }

    fn wire_stack(ids: &[&str]) -> Vec<UndoStackFrame> {
        ids.iter()
            .map(|message_id| UndoStackFrame {
                message_id: (*message_id).into(),
            })
            .collect()
    }

    fn delegation_wire_value(version: u32) -> Value {
        json!({
            "version": version,
            "sessionId": "parent",
            "delegationId": "delegation-1",
            "toolCallId": "call-1",
            "state": "completed",
            "targetAgentId": "coder",
            "objective": "Implement it",
            "childSessionId": "child-1",
            "requestedAt": 100,
            "forkedAt": 110,
            "finishedAt": 120,
            "updatedAt": 120,
            "resultSummary": "done",
            "error": "reported after completion"
        })
    }

    fn expected_delegation_update() -> DelegationUpdate {
        DelegationUpdate {
            session_id: "parent".into(),
            delegation_id: "delegation-1".into(),
            tool_call_id: Some("call-1".into()),
            state: DelegationState::Completed,
            target_agent_id: "coder".into(),
            objective: "Implement it".into(),
            child_session_id: Some("child-1".into()),
            requested_at: 100,
            forked_at: Some(110),
            finished_at: Some(120),
            updated_at: 120,
            result_summary: Some("done".into()),
            error: Some("reported after completion".into()),
        }
    }

    #[test]
    fn prompt_blocks_convert_semantic_text_and_resource_links() {
        let blocks = prompt_blocks(vec![
            PromptBlock::Text {
                text: "inspect this".into(),
            },
            PromptBlock::ResourceLink {
                name: "main.rs".into(),
                uri: "file:///repo/src/main.rs".into(),
            },
        ]);

        assert!(matches!(
            blocks.as_slice(),
            [
                acp::ContentBlock::Text(text),
                acp::ContentBlock::ResourceLink(link),
            ] if text.text == "inspect this"
                && link.name == "main.rs"
                && link.uri == "file:///repo/src/main.rs"
        ));
    }

    #[test]
    fn mesh_wire_adapters_preserve_every_semantic_field() {
        let status = mesh_status_from_wire(MeshStatusDto {
            enabled: true,
            peer_id: Some("peer-1".into()),
            transport: Some("webrtc".into()),
            known_peer_count: 2,
            has_invite_store: true,
            has_mesh_state_store: true,
            scopes: vec![MeshScopeDto {
                kind: "team".into(),
                id: "scope-1".into(),
            }],
        });
        assert!(status.enabled);
        assert_eq!(status.peer_id.as_deref(), Some("peer-1"));
        assert_eq!(status.transport.as_deref(), Some("webrtc"));
        assert_eq!(status.known_peer_count, 2);
        assert!(status.has_invite_store);
        assert!(status.has_mesh_state_store);
        assert_eq!(status.scopes[0].kind, "team");
        assert_eq!(status.scopes[0].id, "scope-1");

        let nodes = mesh_nodes_from_wire(MeshNodesDto {
            nodes: vec![RemoteNodeDto {
                id: "node-1".into(),
                label: "Framework".into(),
                capabilities: vec!["sessions".into(), "attach".into()],
                active_sessions: 3,
                transport: "relay".into(),
                last_seen_at: Some("2025-01-01T00:00:00Z".into()),
            }],
        });
        let node = &nodes.nodes[0];
        assert_eq!(node.id, "node-1");
        assert_eq!(node.label, "Framework");
        assert_eq!(node.capabilities, ["sessions", "attach"]);
        assert_eq!(node.active_sessions, 3);
        assert_eq!(node.transport, "relay");
        assert_eq!(node.last_seen_at.as_deref(), Some("2025-01-01T00:00:00Z"));

        let list = remote_session_list_from_wire(RemoteSessionListDto {
            node_id: "node-1".into(),
            sessions: vec![RemoteSessionDto {
                id: "session-1".into(),
                node_id: "node-1".into(),
                node_label: Some("Framework".into()),
                title: Some("Fix boundary".into()),
                cwd: Some("/repo".into()),
                updated_at: Some("now".into()),
                profile_id: Some("profile-1".into()),
                model_id: Some("model-1".into()),
            }],
            next_offset: Some(50),
            total_count: 51,
        });
        let session = &list.sessions[0];
        assert_eq!(list.node_id, "node-1");
        assert_eq!(list.next_offset, Some(50));
        assert_eq!(list.total_count, 51);
        assert_eq!(session.id, "session-1");
        assert_eq!(session.node_id, "node-1");
        assert_eq!(session.node_label.as_deref(), Some("Framework"));
        assert_eq!(session.title.as_deref(), Some("Fix boundary"));
        assert_eq!(session.cwd.as_deref(), Some("/repo"));
        assert_eq!(session.updated_at.as_deref(), Some("now"));
        assert_eq!(session.profile_id.as_deref(), Some("profile-1"));
        assert_eq!(session.model_id.as_deref(), Some("model-1"));

        let config_options = vec![json!({ "id": "profile", "value": { "nested": [1, 2] } })];
        let snapshot = json!({
            "audit": [{ "kind": "message", "data": { "nested": [1, 2] } }],
            "cursor": { "position": 1 },
            "delegationUpdates": [{ "id": "delegate-1" }]
        });
        let attached = remote_session_attach_from_wire(RemoteSessionAttachDto {
            session_id: "session-1".into(),
            node_id: "node-1".into(),
            attached: true,
            config_options: config_options.clone(),
            snapshot: Some(snapshot.clone()),
        });
        assert_eq!(attached.session_id, "session-1");
        assert_eq!(attached.node_id, "node-1");
        assert!(attached.attached);
        assert_eq!(attached.config_options, config_options);
        assert_eq!(attached.snapshot, Some(snapshot));

        let detached = remote_session_attach_from_wire(RemoteSessionAttachDto {
            session_id: "session-2".into(),
            node_id: "node-1".into(),
            attached: false,
            config_options: Vec::new(),
            snapshot: None,
        });
        assert_eq!(detached.snapshot, None);

        let invite = mesh_invite_created_from_wire(MeshInviteCreatedDto {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: Some("QR".into()),
            expires_at: 123,
            max_uses: 2,
            mesh_name: Some("Team".into()),
        });
        assert_eq!(invite.invite_id, "invite-1");
        assert_eq!(invite.url, "qmt://mesh/join/token");
        assert_eq!(invite.qr_code.as_deref(), Some("QR"));
        assert_eq!(invite.expires_at, 123);
        assert_eq!(invite.max_uses, 2);
        assert_eq!(invite.mesh_name.as_deref(), Some("Team"));
    }

    #[test]
    fn delegation_update_from_wire_maps_every_semantic_field() {
        let wire = serde_json::from_value(delegation_wire_value(1)).expect("wire update");

        let update = delegation_update_from_wire(wire).expect("supported update");

        assert_eq!(update, expected_delegation_update());
    }

    #[test]
    fn delegation_update_from_wire_rejects_unsupported_version() {
        let wire = serde_json::from_value(delegation_wire_value(2)).expect("wire update");

        assert_eq!(delegation_update_from_wire(wire), None);
    }

    #[test]
    fn oauth_flow_from_wire_preserves_semantic_fields() {
        let flow = oauth_flow_from_wire(OAuthFlowDto {
            flow_id: "flow-123".into(),
            provider: "openai".into(),
            authorization_url: "https://auth.example.com/authorize".into(),
            flow_kind: crate::domain::auth::OAuthFlowKind::DevicePoll,
        });

        assert_eq!(flow.flow_id, "flow-123");
        assert_eq!(flow.provider, "openai");
        assert_eq!(flow.authorization_url, "https://auth.example.com/authorize");
        assert_eq!(
            flow.flow_kind,
            crate::domain::auth::OAuthFlowKind::DevicePoll
        );
    }

    #[test]
    fn oauth_result_from_wire_maps_status_and_preserves_provider() {
        let success = oauth_result_from_wire(OAuthResultDto {
            provider: "openai".into(),
            success: true,
            message: "connected".into(),
        });
        assert_eq!(success.provider, "openai");
        assert_eq!(success.status, OAuthResultStatus::Success);
        assert_eq!(success.message, "connected");

        let failure = oauth_result_from_wire(OAuthResultDto {
            provider: "anthropic".into(),
            success: false,
            message: "authorization denied".into(),
        });
        assert_eq!(failure.provider, "anthropic");
        assert_eq!(failure.status, OAuthResultStatus::Failure);
        assert_eq!(failure.message, "authorization denied");
    }

    #[test]
    fn undo_stack_snapshot_from_wire_preserves_message_order() {
        let snapshot = undo_stack_snapshot_from_wire(wire_stack(&["message-2", "message-1"]));

        assert_eq!(snapshot.message_ids, ["message-2", "message-1"]);
    }

    #[test]
    fn undo_result_from_wire_maps_applied_details() {
        let result = undo_result_from_wire(UndoResultData {
            success: true,
            message_id: Some("message-2".into()),
            reverted_files: vec!["src/a.rs".into()],
            message: Some("undone".into()),
            undo_stack: wire_stack(&["message-1", "message-2"]),
        });

        assert_eq!(
            result,
            UndoResult::Applied {
                target_message_id: Some("message-2".into()),
                reverted_files: vec!["src/a.rs".into()],
                message: Some("undone".into()),
                stack: UndoStackSnapshot {
                    message_ids: vec!["message-1".into(), "message-2".into()],
                },
            }
        );
    }

    #[test]
    fn undo_result_from_wire_preserves_rejected_target_and_discards_files() {
        let result = undo_result_from_wire(UndoResultData {
            success: false,
            message_id: Some("message-1".into()),
            reverted_files: vec!["ignored.rs".into()],
            message: Some("undo rejected".into()),
            undo_stack: wire_stack(&["message-1"]),
        });

        assert_eq!(
            result,
            UndoResult::Rejected {
                target_message_id: Some("message-1".into()),
                message: Some("undo rejected".into()),
                stack: UndoStackSnapshot {
                    message_ids: vec!["message-1".into()],
                },
            }
        );
    }

    #[test]
    fn redo_result_from_wire_maps_applied_and_rejected_results() {
        let applied = redo_result_from_wire(RedoResultData {
            success: true,
            message: Some("redone".into()),
            undo_stack: wire_stack(&["message-1", "message-2"]),
        });
        assert_eq!(
            applied,
            RedoResult::Applied {
                message: Some("redone".into()),
                stack: UndoStackSnapshot {
                    message_ids: vec!["message-1".into(), "message-2".into()],
                },
            }
        );

        let rejected = redo_result_from_wire(RedoResultData {
            success: false,
            message: Some("redo rejected".into()),
            undo_stack: wire_stack(&["message-1"]),
        });
        assert_eq!(
            rejected,
            RedoResult::Rejected {
                message: Some("redo rejected".into()),
                stack: UndoStackSnapshot {
                    message_ids: vec!["message-1".into()],
                },
            }
        );
    }

    #[tokio::test]
    async fn assistant_buffer_flushes_segments_in_order_around_tool_boundary() {
        let state = Arc::new(AcpRuntimeState::default());
        let (tx, mut rx) = mpsc::unbounded_channel();

        remember_assistant_chunk(&state, "session-1", Some("a1".into()), "before", false).await;
        flush_assistant_buffer(&state, &tx, "session-1").await;
        send_session_update(
            &tx,
            "session-1",
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                arguments: None,
            },
            false,
        );
        remember_assistant_chunk(&state, "session-1", Some("a2".into()), "after", false).await;
        flush_assistant_buffer(&state, &tx, "session-1").await;

        let updates = (0..3)
            .map(|_| match rx.try_recv().expect("ordered update") {
                ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate { update, .. }) => update,
                other => panic!("unexpected event: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            updates.as_slice(),
            [
                AcpSessionUpdate::AssistantMessage { content: before, .. },
                AcpSessionUpdate::ToolCallStart { tool_call_id: Some(tool_id), .. },
                AcpSessionUpdate::AssistantMessage { content: after, .. },
            ] if before == "before" && tool_id == "tool-1" && after == "after"
        ));
    }

    #[tokio::test]
    async fn assistant_message_id_change_flushes_previous_segment() {
        let state = Arc::new(AcpRuntimeState::default());
        let (tx, mut rx) = mpsc::unbounded_channel();
        remember_assistant_chunk(&state, "session-1", Some("a1".into()), "first", false).await;

        flush_assistant_buffer_for_message(&state, &tx, "session-1", Some("a2")).await;

        assert!(matches!(
            rx.try_recv().expect("flushed message"),
            ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate {
                update: AcpSessionUpdate::AssistantMessage {
                    content,
                    message_id: Some(message_id),
                    ..
                },
                ..
            }) if content == "first" && message_id == "a1"
        ));
    }

    #[test]
    fn send_session_list_emits_request_and_domain_page() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let request = SessionListRequest::WorkspaceContinuation {
            cwd: "/repo".to_string(),
        };
        let response = acp::ListSessionsResponse::new(vec![acp::SessionInfo::new(
            acp::SessionId::from("session-1"),
            Path::new("/from-response"),
        )])
        .next_cursor(Some("cursor-2".to_string()));

        send_session_list(&tx, request.clone(), response);

        match rx.try_recv().expect("session list event") {
            ServerChannelMsg::Acp(AcpAppEvent::SessionList {
                request: emitted_request,
                page,
            }) => {
                assert_eq!(emitted_request, request);
                assert_eq!(page.next_cursor.as_deref(), Some("cursor-2"));
                assert_eq!(page.groups.len(), 1);
                assert_eq!(page.groups[0].cwd.as_deref(), Some("/repo"));
                assert_eq!(page.groups[0].next_cursor.as_deref(), Some("cursor-2"));
                assert_eq!(page.groups[0].sessions.len(), 1);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn acp_root_session_list_preserves_only_the_global_cursor() {
        let response = acp::ListSessionsResponse::new(vec![
            acp::SessionInfo::new(acp::SessionId::from("s1"), Path::new("/repo")),
            acp::SessionInfo::new(acp::SessionId::from("s2"), Path::new("/repo")),
            acp::SessionInfo::new(acp::SessionId::from("s3"), Path::new("/other")),
        ])
        .next_cursor(Some("100".to_string()));

        let page = session_list_page_from_acp(&SessionListRequest::Discovery, response);

        assert_eq!(page.next_cursor.as_deref(), Some("100"));
        assert_eq!(page.total_count, None);
        assert_eq!(page.groups.len(), 2);
        assert!(page.groups.iter().all(|group| group.next_cursor.is_none()));
    }

    #[test]
    fn acp_cwd_session_list_maps_next_cursor_to_group_cursor() {
        let response = acp::ListSessionsResponse::new(vec![
            acp::SessionInfo::new(acp::SessionId::from("s1"), Path::new("/from-response"))
                .title(Some("One".to_string()))
                .updated_at(Some("2024-01-01T00:00:00Z".to_string())),
        ])
        .next_cursor(Some("cursor-2".to_string()));
        let request = SessionListRequest::WorkspaceContinuation {
            cwd: "/repo".to_string(),
        };

        let page = session_list_page_from_acp(&request, response);

        assert_eq!(page.next_cursor.as_deref(), Some("cursor-2"));
        assert_eq!(page.groups.len(), 1);
        assert_eq!(page.groups[0].cwd.as_deref(), Some("/repo"));
        assert_eq!(page.groups[0].next_cursor.as_deref(), Some("cursor-2"));
        assert_eq!(page.groups[0].sessions[0].session_id, "s1");
        assert_eq!(
            page.groups[0].sessions[0].cwd.as_deref(),
            Some("/from-response")
        );
    }

    #[test]
    fn elicitation_metadata_enables_builtin_custom_answers() {
        let request = acp::CreateElicitationRequest::new(
            acp::ElicitationFormMode::new(
                acp::ElicitationSessionScope::new("session-1"),
                acp::ElicitationSchema::new().string("selection", true),
            ),
            "Choose",
        )
        .meta(serde_json::Map::from_iter([(
            "querymt".to_string(),
            json!({ "source": "builtin:question" }),
        )]));

        assert_eq!(
            elicitation_metadata(&request, "acp"),
            ("builtin:question".to_string(), true)
        );
    }

    #[test]
    fn elicitation_metadata_respects_explicit_custom_flag_but_keeps_mcp_strict() {
        let mode = || {
            acp::ElicitationFormMode::new(
                acp::ElicitationSessionScope::new("session-1"),
                acp::ElicitationSchema::new().string("selection", true),
            )
        };
        let strict = acp::CreateElicitationRequest::new(mode(), "Choose").meta(
            serde_json::Map::from_iter([(
                "querymt".to_string(),
                json!({ "source": "mcp:strict-server" }),
            )]),
        );
        let opted_in = acp::CreateElicitationRequest::new(mode(), "Choose").meta(
            serde_json::Map::from_iter([(
                "querymt".to_string(),
                json!({ "source": "mcp:flexible-server", "allow_custom": true }),
            )]),
        );

        assert_eq!(
            elicitation_metadata(&strict, "acp"),
            ("mcp:strict-server".to_string(), false)
        );
        assert_eq!(
            elicitation_metadata(&opted_in, "acp"),
            ("mcp:flexible-server".to_string(), true)
        );
    }

    #[test]
    fn fork_session_meta_uses_querymt_message_id_hint() {
        let meta = fork_session_meta("msg-123");
        assert_eq!(
            meta.get("querymt")
                .and_then(|value| value.get("message_id"))
                .and_then(Value::as_str),
            Some("msg-123")
        );
    }

    #[test]
    fn session_load_audit_reads_querymt_snapshot_from_meta() {
        let response = json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "kind": {
                                    "type": "prompt_received",
                                    "data": { "content": "hello", "message_id": "m1" }
                                },
                                "timestamp": 1
                            }
                        ]
                    },
                    "cursor": { "local_seq": 1, "remote_seq_by_source": {} }
                }
            }
        });
        let audit = session_load_audit_from_load_value(&response);
        let events = audit
            .get("events")
            .and_then(Value::as_array)
            .expect("events");
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn session_load_delegation_updates_skip_malformed_and_unsupported_snapshots() {
        let response = json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": { "events": [] },
                    "delegationUpdates": [
                        delegation_wire_value(1),
                        { "version": 1, "sessionId": "malformed" },
                        delegation_wire_value(2)
                    ]
                }
            }
        });

        let updates = delegation_updates_from_load_value(&response);

        assert_eq!(updates, [expected_delegation_update()]);
    }

    #[test]
    fn delegation_notification_method_accepts_stdio_and_websocket_shapes() {
        assert!(is_delegation_notification_method(
            "querymt/session/delegationUpdate"
        ));
        assert!(is_delegation_notification_method(
            "_querymt/session/delegationUpdate"
        ));
        assert!(!is_delegation_notification_method("session/update"));
    }

    #[test]
    fn delegation_notification_handler_emits_domain_update() {
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_delegation_notification(&tx, delegation_wire_value(1));

        assert!(matches!(
            rx.try_recv().expect("delegation event"),
            ServerChannelMsg::Acp(AcpAppEvent::DelegationUpdate(update))
                if update == expected_delegation_update()
        ));
    }

    #[test]
    fn delegation_notification_handler_logs_unsupported_version() {
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_delegation_notification(&tx, delegation_wire_value(2));

        assert!(matches!(
            rx.try_recv().expect("delegation version log"),
            ServerChannelMsg::Acp(AcpAppEvent::InfoLog { target, message })
                if target == "delegation"
                    && message == "ignored delegation notification version 2"
        ));
    }

    #[test]
    fn delegation_notification_handler_logs_malformed_payload() {
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_delegation_notification(&tx, json!({ "version": 1 }));

        assert!(matches!(
            rx.try_recv().expect("invalid delegation log"),
            ServerChannelMsg::Acp(AcpAppEvent::InfoLog { target, message })
                if target == "delegation"
                    && message.starts_with("invalid delegation notification: ")
        ));
    }

    #[test]
    fn session_load_audit_missing_snapshot_returns_empty_events() {
        let audit = session_load_audit_from_load_value(&json!({}));
        assert_eq!(
            audit.get("events").and_then(Value::as_array).map(Vec::len),
            Some(0)
        );
    }

    #[test]
    fn snapshot_provider_change_uses_last_provider_changed_event() {
        let change = snapshot_provider_change_from_load_value(&json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "kind": {
                                    "type": "provider_changed",
                                    "data": {
                                        "provider": "anthropic",
                                        "model": "claude-3-5",
                                        "context_limit": 200000
                                    }
                                }
                            },
                            {
                                "kind": { "type": "assistant_message_stored", "data": { "content": "hello" } }
                            },
                            {
                                "kind": {
                                    "type": "provider_changed",
                                    "data": {
                                        "provider": "openrouter",
                                        "model": "anthropic/claude-sonnet-4",
                                        "context_limit": "1000000",
                                        "provider_node_id": "node-1"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }))
        .expect("provider change");

        assert_eq!(change.provider, "openrouter");
        assert_eq!(change.model, "anthropic/claude-sonnet-4");
        assert_eq!(change.context_limit, Some(1_000_000));
        assert_eq!(change.provider_node_id.as_deref(), Some("node-1"));
    }

    #[test]
    fn snapshot_provider_change_ignores_malformed_provider_changed_event() {
        let change = snapshot_provider_change_from_load_value(&json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "kind": {
                                    "type": "provider_changed",
                                    "data": {
                                        "provider": "anthropic",
                                        "context_limit": 200000
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }));

        assert!(change.is_none());
    }

    #[test]
    fn snapshot_provider_change_event_maps_to_acp_provider_changed() {
        let event = snapshot_provider_change_event(SnapshotProviderChange {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4".to_string(),
            context_limit: Some(200000),
            provider_node_id: Some("node-1".to_string()),
        });

        assert!(matches!(
            event,
            AcpAppEvent::ProviderChanged { provider, model, context_limit, provider_node_id }
                if provider == "anthropic"
                    && model == "claude-sonnet-4"
                    && context_limit == Some(200000)
                    && provider_node_id.as_deref() == Some("node-1")
        ));
    }

    #[test]
    fn snapshot_updates_restore_basic_user_and_assistant_history() {
        let updates = snapshot_updates_from_load_value(&json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "kind": {
                                    "type": "prompt_received",
                                    "data": { "content": "hello", "message_id": "u1" }
                                }
                            },
                            {
                                "kind": {
                                    "type": "assistant_message_stored",
                                    "data": { "content": "world", "thinking": "notes", "message_id": "a1" }
                                }
                            }
                        ]
                    }
                }
            }
        }));

        assert_eq!(updates.len(), 2);
        assert!(matches!(
            &updates[0],
            AcpSessionUpdate::UserMessage { content, message_id }
                if content == "hello" && message_id.as_deref() == Some("u1")
        ));
        assert!(matches!(
            &updates[1],
            AcpSessionUpdate::AssistantMessage { content, thinking, message_id }
                if content == "world"
                    && thinking.as_deref() == Some("notes")
                    && message_id.as_deref() == Some("a1")
        ));
    }

    #[test]
    fn snapshot_updates_restore_tool_history() {
        let updates = snapshot_updates_from_load_value(&json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "kind": {
                                    "type": "tool_call_start",
                                    "data": {
                                        "tool_call_id": "tool-1",
                                        "tool_name": "read_tool",
                                        "arguments": { "path": "src/main.rs" }
                                    }
                                }
                            },
                            {
                                "kind": {
                                    "type": "tool_call_end",
                                    "data": {
                                        "tool_call_id": "tool-1",
                                        "tool_name": "read_tool",
                                        "is_error": false,
                                        "result": "ok"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }));

        assert_eq!(updates.len(), 2);
        assert!(matches!(
            &updates[0],
            AcpSessionUpdate::ToolCallStart { tool_call_id, name, arguments }
                if tool_call_id.as_deref() == Some("tool-1")
                    && name == "read_tool"
                    && arguments.as_ref().and_then(|v| v.get("path")).and_then(Value::as_str) == Some("src/main.rs")
        ));
        assert!(matches!(
            &updates[1],
            AcpSessionUpdate::ToolCallEnd { tool_call_id, name, is_error, result }
                if tool_call_id.as_deref() == Some("tool-1")
                    && name == "read_tool"
                    && !is_error
                    && result.as_deref() == Some("ok")
        ));
    }

    #[test]
    fn snapshot_updates_restore_usage_and_active_time_from_acp_load_metadata() {
        let updates = snapshot_updates_from_load_value(&json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "timestamp": 100,
                                "kind": { "type": "session_created" }
                            },
                            {
                                "timestamp": 120,
                                "kind": {
                                    "type": "provider_changed",
                                    "data": { "context_limit": 8192 }
                                }
                            },
                            {
                                "timestamp": 130,
                                "kind": { "type": "llm_request_start", "data": { "message_count": 1 } }
                            },
                            {
                                "timestamp": 160,
                                "kind": {
                                    "type": "llm_request_end",
                                    "data": {
                                        "context_tokens": 2048,
                                        "cumulative_cost_usd": 0.0123,
                                        "finish_reason": "stop"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }));

        assert_eq!(updates.len(), 4);
        assert!(matches!(
            &updates[0],
            AcpSessionUpdate::UsageUpdate {
                used: 0,
                size: 8192,
                cost_usd: None
            }
        ));
        assert!(matches!(
            &updates[1],
            AcpSessionUpdate::TimingUpdate { duration_secs: 30 }
        ));
        assert!(matches!(
            &updates[2],
            AcpSessionUpdate::UsageUpdate { used: 2048, size: 8192, cost_usd: Some(cost) }
                if (*cost - 0.0123).abs() < f64::EPSILON
        ));
        assert!(matches!(
            &updates[3],
            AcpSessionUpdate::Finished { finish_reason } if finish_reason == "stop"
        ));
    }

    #[test]
    fn merge_replay_with_snapshot_stats_preserves_replay_and_adds_usage() {
        let replay = vec![AcpSessionUpdate::UserMessage {
            content: json!({ "text": "hello" }),
            message_id: Some("u1".into()),
        }];
        let snapshot = vec![
            AcpSessionUpdate::UserMessage {
                content: json!({ "text": "duplicate" }),
                message_id: Some("u1".into()),
            },
            AcpSessionUpdate::TimingUpdate { duration_secs: 30 },
            AcpSessionUpdate::UsageUpdate {
                used: 2048,
                size: 8192,
                cost_usd: Some(0.0123),
            },
            AcpSessionUpdate::Finished {
                finish_reason: "stop".into(),
            },
        ];

        let merged = merge_replay_with_snapshot_stats(replay, snapshot);

        assert_eq!(merged.len(), 3);
        assert!(matches!(
            &merged[0],
            AcpSessionUpdate::UserMessage { content, message_id }
                if content == &json!({ "text": "hello" }) && message_id.as_deref() == Some("u1")
        ));
        assert!(matches!(
            &merged[1],
            AcpSessionUpdate::TimingUpdate { duration_secs: 30 }
        ));
        assert!(matches!(
            &merged[2],
            AcpSessionUpdate::UsageUpdate { used: 2048, size: 8192, cost_usd: Some(cost) }
                if (*cost - 0.0123).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn merge_replay_with_snapshot_stats_does_not_duplicate_native_usage() {
        let replay = vec![AcpSessionUpdate::UsageUpdate {
            used: 1024,
            size: 8192,
            cost_usd: None,
        }];
        let snapshot = vec![
            AcpSessionUpdate::TimingUpdate { duration_secs: 30 },
            AcpSessionUpdate::UsageUpdate {
                used: 2048,
                size: 8192,
                cost_usd: Some(0.0123),
            },
        ];

        let merged = merge_replay_with_snapshot_stats(replay, snapshot);

        assert_eq!(merged.len(), 2);
        assert!(matches!(
            &merged[0],
            AcpSessionUpdate::UsageUpdate {
                used: 1024,
                size: 8192,
                cost_usd: None
            }
        ));
        assert!(matches!(
            &merged[1],
            AcpSessionUpdate::TimingUpdate { duration_secs: 30 }
        ));
    }

    #[test]
    fn snapshot_updates_do_not_use_wall_clock_session_age() {
        let updates = snapshot_updates_from_load_value(&json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "timestamp": 200,
                                "kind": { "type": "session_created" }
                            },
                            {
                                "timestamp": 250,
                                "kind": { "type": "assistant_message_stored", "data": { "content": "hello" } }
                            }
                        ]
                    }
                }
            }
        }));

        assert!(
            !updates
                .iter()
                .any(|update| matches!(update, AcpSessionUpdate::TimingUpdate { .. }))
        );
    }

    #[test]
    fn snapshot_updates_restore_usage_even_without_context_limit() {
        let updates = snapshot_updates_from_load_value(&json!({
            "_meta": {
                "querymt/sessionLoadSnapshot.v1": {
                    "audit": {
                        "events": [
                            {
                                "kind": {
                                    "type": "llm_request_end",
                                    "data": { "context_tokens": "512" }
                                }
                            }
                        ]
                    }
                }
            }
        }));

        assert_eq!(updates.len(), 2);
        assert!(matches!(
            &updates[0],
            AcpSessionUpdate::UsageUpdate {
                used: 512,
                size: 0,
                cost_usd: None
            }
        ));
    }

    #[test]
    fn usage_update_maps_context_limit_and_usd_cost() {
        let update =
            usage_update(acp::UsageUpdate::new(2048, 8192).cost(acp::Cost::new(0.25, "USD")));

        assert!(matches!(
            update,
            AcpSessionUpdate::UsageUpdate {
                used: 2048,
                size: 8192,
                cost_usd: Some(0.25)
            }
        ));
    }

    #[test]
    fn normalize_profiles_response_accepts_direct_full_metadata() {
        let response = normalize_profiles_response(json!({
            "profiles": [{
                "id": "coder-delegate",
                "name": "Coder Delegate",
                "description": "Delegates coding tasks",
                "tags": ["coding", "delegate"],
                "source": "local:/tmp/coder.toml",
                "config_kind": "toml",
                "fingerprint": "sha256:abc"
            }],
            "active_profile_id": "coder-delegate"
        }))
        .expect("profile response");

        assert_eq!(
            response.active_profile_id.as_deref(),
            Some("coder-delegate")
        );
        assert_eq!(response.profiles.len(), 1);
        let profile = &response.profiles[0];
        assert_eq!(profile.name, "Coder Delegate");
        assert_eq!(profile.tags, ["coding", "delegate"]);
        assert_eq!(profile.source.as_deref(), Some("local:/tmp/coder.toml"));
        assert_eq!(profile.config_kind.as_deref(), Some("toml"));
        assert_eq!(profile.fingerprint.as_deref(), Some("sha256:abc"));
    }

    #[test]
    fn normalize_profiles_response_accepts_wrapped_and_empty_responses() {
        let wrapped = normalize_profiles_response(json!({
            "data": {
                "profiles": [{ "id": "fast", "name": "Fast" }],
                "active_profile_id": "fast"
            }
        }))
        .expect("wrapped profile response");
        assert_eq!(wrapped.profiles[0].id, "fast");

        let empty = normalize_profiles_response(json!({
            "profiles": [],
            "active_profile_id": null
        }))
        .expect("empty profile response");
        assert!(empty.profiles.is_empty());
        assert!(empty.active_profile_id.is_none());
        assert!(normalize_profiles_response(json!({})).is_err());
    }

    #[test]
    fn normalize_profile_agents_response_accepts_direct_and_wrapped_payloads() {
        let direct = normalize_profile_agents_response(json!({
            "profile_id": "quorum",
            "agents": [
                { "id": "primary", "name": "Session" },
                {
                    "id": "coder",
                    "name": "Coder",
                    "description": "Writes code",
                    "capabilities": ["coding"]
                }
            ]
        }))
        .expect("profile agents");
        assert_eq!(direct.profile_id, "quorum");
        assert_eq!(direct.agents[1].id, "coder");
        assert_eq!(direct.agents[1].capabilities, ["coding"]);

        let wrapped = normalize_profile_agents_response(json!({
            "data": { "profile_id": "single", "agents": [{ "id": "primary", "name": "Session" }] }
        }))
        .expect("wrapped profile agents");
        assert_eq!(wrapped.profile_id, "single");
        assert!(normalize_profile_agents_response(json!({ "profile_id": "bad" })).is_err());
    }

    #[test]
    fn normalize_delegate_model_response_preserves_model_and_node() {
        let response = normalize_delegate_model_response(json!({
            "data": {
                "session_id": "parent",
                "agent_id": "coder",
                "model": { "model_id": "openai/gpt-5", "node_id": "node-1" }
            }
        }))
        .expect("delegate model response");
        assert_eq!(response.session_id, "parent");
        assert_eq!(response.agent_id, "coder");
        let model = response.model.expect("model");
        assert_eq!(model.model_id, "openai/gpt-5");
        assert_eq!(model.node_id.as_deref(), Some("node-1"));
    }

    #[test]
    fn normalize_models_response_accepts_direct_response_with_meta() {
        let response = normalize_models_response(json!({
            "models": [model_value("openai/gpt-4o", "openai", "gpt-4o")],
            "meta": { "stale": true, "refresh_in_progress": false }
        }));

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].provider, "openai");
        assert!(response.meta.expect("meta").stale);
    }

    #[test]
    fn normalize_models_response_accepts_wrapped_response_with_meta() {
        let response = normalize_models_response(json!({
            "data": {
                "models": [model_value("anthropic/claude", "anthropic", "claude")],
                "meta": { "stale": false, "refresh_in_progress": true }
            }
        }));

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].model, "claude");
        assert!(response.meta.expect("meta").refresh_in_progress);
    }

    #[test]
    fn model_entry_meta_uses_full_acp_model_entry() {
        let model = normalize_models_response(json!({
            "models": [model_value("openrouter/openai/gpt-4o", "openrouter", "openai/gpt-4o")]
        }))
        .models
        .remove(0);

        let meta = model_entry_meta(&model, Some("node-1")).expect("meta");
        let model_entry = meta
            .get("querymt")
            .and_then(|value| value.get("modelEntry"))
            .expect("modelEntry");

        assert_eq!(
            model_entry.get("provider").and_then(Value::as_str),
            Some("openrouter")
        );
        assert_eq!(
            model_entry.get("model").and_then(Value::as_str),
            Some("openai/gpt-4o")
        );
        assert_eq!(
            model_entry.get("node_id").and_then(Value::as_str),
            Some("node-1")
        );
    }

    #[test]
    fn model_entry_from_config_option_reads_select_choices() {
        let option = json!({
            "id": "model",
            "currentValue": "openai/gpt-4o",
            "options": [
                { "value": "anthropic/claude", "name": "Claude" },
                { "group": "OpenAI", "options": [
                    { "value": "openai/gpt-4o", "name": "GPT 4o" }
                ] }
            ]
        });

        let model = model_entry_from_config_option(&option, "openai/gpt-4o").expect("model");
        assert_eq!(model.provider, "openai");
        assert_eq!(model.model, "gpt-4o");
        assert_eq!(model.label, "GPT 4o");
    }
}
