use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1 as acp;
use agent_client_protocol::{
    self as acp_sdk, AcpAgent, Agent, Client, ConnectionTo, UntypedMessage,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};

use crate::ServerChannelMsg;
use crate::acp_state::{AcpAppEvent, AcpModelsMetaInfo, AcpSessionUpdate};
use crate::protocol::{
    AuthProvidersData, ClientMsg, OAuthFlowData, OAuthResultData, ProfileInfo, SessionGroup,
    SessionSummary,
};

#[derive(Debug, Clone)]
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
    fn to_app_model(&self) -> crate::protocol::ModelEntry {
        crate::protocol::ModelEntry {
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
    message_id: Option<String>,
}

#[derive(Default)]
struct AcpRuntimeState {
    agent: Mutex<AgentIdentity>,
    current_session_id: Mutex<Option<String>>,
    loading_sessions: Mutex<HashSet<String>>,
    replay_updates: Mutex<HashMap<String, Vec<AcpSessionUpdate>>>,
    assistant_buffers: Mutex<HashMap<String, AssistantBuffer>>,
    pending_elicitations:
        Mutex<HashMap<String, acp_sdk::Responder<acp::CreateElicitationResponse>>>,
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

pub async fn run_stdio_agent(
    agent: AcpAgent,
    cmd_rx: &mut mpsc::UnboundedReceiver<ClientMsg>,
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
                if let Err(err) = handle_client_msg(&connection, &state, &srv_tx, cmd).await {
                    send_error(&srv_tx, format!("ACP request failed: {err:?}"));
                }
            }

            Ok(())
        })
        .await
}

async fn handle_client_msg(
    connection: &ConnectionTo<Agent>,
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    cmd: ClientMsg,
) -> Result<(), acp_sdk::Error> {
    match cmd {
        ClientMsg::Init => {
            let response = connection
                .send_request(
                    acp::InitializeRequest::new(ProtocolVersion::V1)
                        .client_capabilities(client_capabilities())
                        .client_info(acp::Implementation::new("qmtui", env!("CARGO_PKG_VERSION"))),
                )
                .block_task()
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
        ClientMsg::ListSessions { cursor, cwd, .. } => {
            let mut req = acp::ListSessionsRequest::new().cursor(cursor);
            if let Some(cwd) = cwd.and_then(|cwd| (cwd != "__none__").then_some(cwd)) {
                req = req.cwd(PathBuf::from(cwd));
            }
            let response = connection.send_request(req).block_task().await?;
            send_session_list(srv_tx, response);
        }
        ClientMsg::NewSession {
            cwd, profile_id, ..
        } => {
            let mut req = acp::NewSessionRequest::new(
                cwd.map(PathBuf::from)
                    .unwrap_or_else(|| state.default_cwd()),
            );
            if let Some(profile_id) = profile_id.as_deref() {
                req = req.meta(profile_meta(profile_id));
            }
            let response = connection.send_request(req).block_task().await?;
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
        ClientMsg::LoadSession { session_id } => {
            state.set_current_session_id(session_id.clone()).await;
            state.begin_loading(&session_id).await;

            let req = acp::LoadSessionRequest::new(session_id.clone(), state.default_cwd());
            let result = connection.send_request(req).block_task().await;
            let replay_updates = state.end_loading(&session_id).await;
            let response = result?;
            let snapshot_updates = if replay_updates.is_empty() {
                snapshot_updates_from_load_response(&response)
            } else {
                Vec::new()
            };

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
            let history_updates = if replay_updates.is_empty() {
                snapshot_updates
            } else {
                replay_updates
            };
            for update in history_updates {
                send_acp(
                    srv_tx,
                    AcpAppEvent::SessionUpdate {
                        session_id: session_id.clone(),
                        update,
                        is_replay: true,
                    },
                );
            }

            if let Some(config_options) = response.config_options {
                send_config_updates(state, srv_tx, config_options).await;
            }
        }
        ClientMsg::Prompt { prompt } => {
            let Some(session_id) = state.current_session_id().await else {
                send_error(srv_tx, "cannot prompt before a session is loaded");
                return Ok(());
            };
            send_session_update(srv_tx, &session_id, AcpSessionUpdate::TurnStarted, false);
            let req = acp::PromptRequest::new(session_id.clone(), prompt_blocks(prompt));
            let response = connection.send_request(req).block_task().await?;
            finish_prompt(state, srv_tx, &session_id, response.stop_reason).await;
        }
        ClientMsg::CancelSession => {
            let Some(session_id) = state.current_session_id().await else {
                return Ok(());
            };
            connection.send_notification(acp::CancelNotification::new(session_id))?;
        }
        ClientMsg::DeleteSession { session_id } => {
            connection
                .send_request(acp::DeleteSessionRequest::new(session_id))
                .block_task()
                .await?;
        }
        ClientMsg::SetAgentMode { mode } => {
            set_config_option(connection, state, srv_tx, "mode", &mode, None).await?;
        }
        ClientMsg::SetReasoningEffort { reasoning_effort } => {
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
        ClientMsg::SetSessionModel {
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
                .send_request(
                    acp::SetSessionConfigOptionRequest::new(session_id, "model", model_id.as_str())
                        .meta(meta),
                )
                .block_task()
                .await?;
            state.select_model(model_id).await;
            send_provider_changed(srv_tx, &model);
            send_config_updates(state, srv_tx, response.config_options).await;
        }
        ClientMsg::ListAllModels { refresh } => {
            let response = load_acp_models(connection, refresh).await?;
            state.set_models(response.models.clone()).await;
            send_models(srv_tx, &response);
            if let Some(model) = state.selected_or_default_model().await {
                state.select_model(model.id.clone()).await;
                send_provider_changed(srv_tx, &model);
            }
        }
        ClientMsg::ListAuthProviders => {
            let response = call_querymt_ext(connection, "querymt/auth/status", json!({})).await?;
            if let Ok(auth) =
                serde_json::from_value::<AuthProvidersData>(ext_payload(&response).clone())
            {
                send_acp(srv_tx, AcpAppEvent::AuthProviders(auth.providers));
            }
        }
        ClientMsg::StartOAuthLogin { provider } => {
            let response = call_querymt_ext(
                connection,
                "querymt/auth/start",
                json!({ "provider": provider }),
            )
            .await?;
            if let Ok(flow) =
                serde_json::from_value::<OAuthFlowData>(ext_payload(&response).clone())
            {
                send_acp(srv_tx, AcpAppEvent::OAuthFlowStarted(flow));
            }
        }
        ClientMsg::CompleteOAuthLogin { flow_id, response } => {
            let response = call_querymt_ext(
                connection,
                "querymt/auth/complete",
                json!({ "flow_id": flow_id, "response": response }),
            )
            .await?;
            if let Ok(result) =
                serde_json::from_value::<OAuthResultData>(ext_payload(&response).clone())
            {
                send_acp(srv_tx, AcpAppEvent::OAuthResult(result));
            }
        }
        ClientMsg::DisconnectOAuth { provider } => {
            let response = call_querymt_ext(
                connection,
                "querymt/auth/logout",
                json!({ "provider": provider }),
            )
            .await?;
            if let Ok(result) =
                serde_json::from_value::<OAuthResultData>(ext_payload(&response).clone())
            {
                send_acp(srv_tx, AcpAppEvent::OAuthResult(result));
            }
        }
        ClientMsg::ElicitationResponse {
            elicitation_id,
            action,
            content,
        } => {
            respond_to_elicitation(state, &elicitation_id, &action, content).await;
        }
        ClientMsg::SubscribeSession { .. } => {}
        ClientMsg::GetAgentMode => {}
        ClientMsg::GetFileIndex => {
            // TODO(ACP parity): replace the deprecated UI file-index endpoint with
            // an ACP/QueryMT extension or client-side workspace indexing.
            send_error(
                srv_tx,
                "file mentions are not exposed in the ACP subset yet",
            );
        }
        ClientMsg::ForkSession { .. }
        | ClientMsg::Undo { .. }
        | ClientMsg::Redo
        | ClientMsg::ListSessionChildren { .. }
        | ClientMsg::ListRemoteNodes
        | ClientMsg::ListRemoteSessions { .. }
        | ClientMsg::CreateRemoteSession { .. }
        | ClientMsg::AttachRemoteSession { .. }
        | ClientMsg::DismissRemoteSession { .. }
        | ClientMsg::ListProfiles
        | ClientMsg::SetActiveProfile { .. }
        | ClientMsg::SetApiToken { .. }
        | ClientMsg::ClearApiToken { .. }
        | ClientMsg::SetAuthMethod { .. } => {
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

async fn set_config_option(
    connection: &ConnectionTo<Agent>,
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
        .send_request(
            acp::SetSessionConfigOptionRequest::new(
                session_id,
                config_id.to_string(),
                acp::SessionConfigOptionValue::from(value),
            )
            .meta(meta),
        )
        .block_task()
        .await?;
    send_config_updates(state, srv_tx, response.config_options).await;
    Ok(())
}

async fn call_querymt_ext(
    connection: &ConnectionTo<Agent>,
    method: &str,
    params: Value,
) -> Result<Value, acp_sdk::Error> {
    let wire_method = format!("_{method}");
    connection
        .send_request(UntypedMessage::new(&wire_method, params)?)
        .block_task()
        .await
}

fn ext_payload<'a>(response: &'a Value) -> &'a Value {
    response.get("data").unwrap_or(response)
}

async fn post_connect_diagnostics(
    connection: &ConnectionTo<Agent>,
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
            if methods.iter().any(|m| m == "querymt/mesh/nodes") {
                if let Ok(nodes_resp) =
                    call_querymt_ext(connection, "querymt/mesh/nodes", json!({})).await
                {
                    send_acp(
                        srv_tx,
                        AcpAppEvent::MeshNodes(ext_payload(&nodes_resp).clone()),
                    );
                }
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

fn prompt_blocks(blocks: Vec<crate::protocol::PromptBlock>) -> Vec<acp::ContentBlock> {
    blocks
        .into_iter()
        .map(|block| match block {
            crate::protocol::PromptBlock::Text { text } => {
                acp::ContentBlock::Text(acp::TextContent::new(text))
            }
            crate::protocol::PromptBlock::ResourceLink { name, uri } => {
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

const SESSION_LOAD_SNAPSHOT_META_KEY: &str = "querymt/sessionLoadSnapshot.v1";

fn session_load_audit_from_load_value(response: &Value) -> Value {
    let meta = response.get("_meta").or_else(|| response.get("meta"));
    let snapshot = meta.and_then(|m| m.get(SESSION_LOAD_SNAPSHOT_META_KEY));
    match snapshot {
        Some(snapshot) => snapshot
            .get("audit")
            .cloned()
            .unwrap_or_else(|| json!({ "events": [] })),
        None => json!({ "events": [] }),
    }
}

fn snapshot_updates_from_load_response(
    response: &acp::LoadSessionResponse,
) -> Vec<AcpSessionUpdate> {
    serde_json::to_value(response)
        .map(|value| snapshot_updates_from_load_value(&value))
        .unwrap_or_default()
}

fn snapshot_updates_from_load_value(response: &Value) -> Vec<AcpSessionUpdate> {
    let audit = session_load_audit_from_load_value(response);
    let Some(events) = audit.get("events").and_then(Value::as_array) else {
        return Vec::new();
    };

    events
        .iter()
        .filter_map(|event| {
            let kind = event.get("kind")?;
            let kind_type = kind.get("type").and_then(Value::as_str)?;
            let data = kind.get("data").unwrap_or(&Value::Null);
            snapshot_event_to_update(kind_type, data)
        })
        .collect()
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
        "llm_request_end" => Some(AcpSessionUpdate::Finished {
            finish_reason: snapshot_string(data, "finish_reason")
                .unwrap_or_else(|| "completed".to_string()),
        }),
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

async fn load_acp_models(
    connection: &ConnectionTo<Agent>,
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

async fn call_acp_models(
    connection: &ConnectionTo<Agent>,
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
        // TODO(ACP parity): map these native ACP updates into status/sidebar state.
        acp::SessionUpdate::SessionInfoUpdate(_) | acp::SessionUpdate::UsageUpdate(_) => {}
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
    if buffer.message_id.is_none() && message_id.is_some() {
        buffer.message_id = message_id;
    }
    if thinking {
        buffer.thinking.push_str(text);
    } else {
        buffer.content.push_str(text);
    }
}

async fn finish_prompt(
    state: &Arc<AcpRuntimeState>,
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    session_id: &str,
    stop_reason: acp::StopReason,
) {
    let buffer = state.assistant_buffers.lock().await.remove(session_id);
    if let Some(buffer) = buffer {
        if !buffer.content.is_empty() || !buffer.thinking.is_empty() {
            send_session_update(
                srv_tx,
                session_id,
                AcpSessionUpdate::AssistantMessage {
                    content: buffer.content,
                    thinking: (!buffer.thinking.is_empty()).then_some(buffer.thinking),
                    message_id: buffer.message_id,
                },
                false,
            );
        }
    }

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

fn tool_call_update(update: acp::ToolCallUpdate) -> AcpSessionUpdate {
    let status = update.fields.status.clone();
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
    if let Some(node_id) = node_id_override {
        if let Some(object) = entry.as_object_mut() {
            object.insert("node_id".to_string(), Value::String(node_id.to_string()));
        }
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

fn send_session_list(
    srv_tx: &mpsc::UnboundedSender<ServerChannelMsg>,
    response: acp::ListSessionsResponse,
) {
    let mut groups: BTreeMap<Option<String>, Vec<SessionSummary>> = BTreeMap::new();
    for session in response.sessions {
        let cwd = session.cwd.to_string_lossy().to_string();
        let group_key = (!cwd.is_empty()).then_some(cwd.clone());
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

    let total_count: usize = groups.values().map(Vec::len).sum();
    let groups = groups
        .into_iter()
        .map(|(cwd, sessions)| SessionGroup {
            cwd,
            latest_activity: sessions
                .first()
                .and_then(|session| session.updated_at.clone()),
            total_count: Some(sessions.len() as u64),
            next_cursor: None,
            sessions,
        })
        .collect::<Vec<_>>();

    send_acp(
        srv_tx,
        AcpAppEvent::SessionList {
            groups,
            next_cursor: response.next_cursor.map(|cursor| cursor.to_string()),
            total_count: Some(total_count as u64),
        },
    );
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
    let responder_id = responder.id();
    let elicitation_id = responder_id
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| responder_id.to_string());
    let (session_id, requested_schema, source) = match &request.mode {
        acp::ElicitationMode::Form(form) => (
            elicitation_scope_session_id(&form.scope),
            serde_json::to_value(&form.requested_schema).unwrap_or_else(|_| json!({})),
            "acp".to_string(),
        ),
        acp::ElicitationMode::Url(url) => (
            elicitation_scope_session_id(&url.scope),
            json!({}),
            format!("acp-url:{}", url.url),
        ),
        _ => ("request".to_string(), json!({}), "acp".to_string()),
    };

    state
        .pending_elicitations
        .lock()
        .await
        .insert(elicitation_id.clone(), responder);

    send_session_update(
        srv_tx,
        &session_id,
        AcpSessionUpdate::ElicitationRequested {
            elicitation_id,
            message: request.message,
            requested_schema,
            source,
        },
        false,
    );
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

    let response = match action {
        "accept" => acp::CreateElicitationResponse::new(acp::ElicitationAction::Accept(
            acp::ElicitationAcceptAction::new().content(elicitation_content(content)),
        )),
        "decline" => acp::CreateElicitationResponse::new(acp::ElicitationAction::Decline),
        _ => acp::CreateElicitationResponse::new(acp::ElicitationAction::Cancel),
    };
    let _ = responder.respond(response);
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

    fn model_value(id: &str, provider: &str, model: &str) -> Value {
        json!({
            "id": id,
            "label": model,
            "source": "catalog",
            "provider": provider,
            "model": model,
        })
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
    fn session_load_audit_missing_snapshot_returns_empty_events() {
        let audit = session_load_audit_from_load_value(&json!({}));
        assert_eq!(
            audit.get("events").and_then(Value::as_array).map(Vec::len),
            Some(0)
        );
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
    fn normalize_models_response_accepts_direct_response_with_meta() {
        let response = normalize_models_response(json!({
            "models": [model_value("openai/gpt-4o", "openai", "gpt-4o")],
            "meta": { "stale": true, "refresh_in_progress": false }
        }));

        assert_eq!(response.models.len(), 1);
        assert_eq!(response.models[0].provider, "openai");
        assert_eq!(response.meta.expect("meta").stale, true);
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
        assert_eq!(response.meta.expect("meta").refresh_in_progress, true);
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
