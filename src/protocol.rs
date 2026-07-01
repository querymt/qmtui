use serde::{Deserialize, Serialize};

// --- Client → Server messages ---

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionScope {
    Root,
    Forks,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ClientMsg {
    Init,
    ListSessions {
        #[serde(skip_serializing_if = "Option::is_none")]
        mode: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cursor: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        query: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        include_remote: Option<bool>,
        session_scope: SessionScope,
    },
    ListRemoteNodes,
    ListRemoteSessions {
        node_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        offset: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
    },
    CreateRemoteSession {
        node_id: String,
        cwd: Option<String>,
        request_id: Option<String>,
    },
    AttachRemoteSession {
        node_id: String,
        session_id: String,
    },
    DismissRemoteSession {
        session_id: String,
    },
    ListSessionChildren {
        parent_session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cursor: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
        session_scope: SessionScope,
    },
    SetReasoningEffort {
        reasoning_effort: String,
    },
    ListProfiles,
    SetActiveProfile {
        profile_id: String,
    },
    NewSession {
        cwd: Option<String>,
        request_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        profile_id: Option<String>,
    },
    LoadSession {
        session_id: String,
    },
    Prompt {
        prompt: Vec<PromptBlock>,
    },
    CancelSession,
    ListAllModels {
        refresh: bool,
    },
    SetSessionModel {
        session_id: String,
        model_id: String,
        node_id: Option<String>,
    },
    SubscribeSession {
        session_id: String,
        agent_id: Option<String>,
    },
    DeleteSession {
        session_id: String,
    },
    ForkSession {
        message_id: String,
    },
    Undo {
        message_id: String,
    },
    Redo,
    GetFileIndex,
    SetAgentMode {
        mode: String,
    },
    GetAgentMode,
    ElicitationResponse {
        elicitation_id: String,
        action: String, // "accept", "decline", "cancel"
        content: Option<serde_json::Value>,
    },
    ListAuthProviders,
    #[serde(rename = "start_oauth_login")]
    StartOAuthLogin {
        provider: String,
    },
    #[serde(rename = "complete_oauth_login")]
    CompleteOAuthLogin {
        flow_id: String,
        response: String,
    },
    #[serde(rename = "disconnect_oauth")]
    DisconnectOAuth {
        provider: String,
    },
    SetApiToken {
        provider: String,
        api_key: String,
    },
    ClearApiToken {
        provider: String,
    },
    SetAuthMethod {
        provider: String,
        method: AuthMethod,
    },
}

impl ClientMsg {
    pub fn list_sessions_browse() -> Self {
        Self::ListSessions {
            mode: None,
            cursor: None,
            limit: None,
            cwd: None,
            query: None,
            include_remote: Some(true),
            session_scope: SessionScope::Root,
        }
    }

    pub fn list_sessions_group(cwd: Option<String>, cursor: String, limit: u32) -> Self {
        Self::ListSessions {
            mode: Some("group".to_string()),
            cursor: Some(cursor),
            limit: Some(limit),
            cwd: Some(cwd.unwrap_or_else(|| "__none__".to_string())),
            query: None,
            include_remote: None,
            session_scope: SessionScope::Root,
        }
    }

    pub fn list_session_children(
        parent_session_id: String,
        cursor: Option<String>,
        limit: u32,
    ) -> Self {
        Self::ListSessionChildren {
            parent_session_id,
            cursor,
            limit: Some(limit),
            session_scope: SessionScope::Forks,
        }
    }
}

#[cfg(test)]
mod client_msg_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn list_sessions_browse_serializes_root_session_scope() {
        let value = serde_json::to_value(ClientMsg::list_sessions_browse()).unwrap();
        assert_eq!(
            value,
            json!({
                "type": "list_sessions",
                "data": {
                    "include_remote": true,
                    "session_scope": "root"
                }
            })
        );
    }

    #[test]
    fn list_sessions_group_omits_include_remote_for_pagination() {
        let value = serde_json::to_value(ClientMsg::list_sessions_group(
            Some("/workspace/project".to_string()),
            "cursor-1".to_string(),
            10,
        ))
        .unwrap();
        assert!(value["data"].get("include_remote").is_none());
    }

    #[test]
    fn remote_session_messages_serialize() {
        assert_eq!(
            serde_json::to_value(ClientMsg::ListRemoteNodes).unwrap(),
            json!({ "type": "list_remote_nodes" })
        );
        assert_eq!(
            serde_json::to_value(ClientMsg::ListRemoteSessions {
                node_id: "node-1".to_string(),
                offset: Some(20),
                limit: Some(10),
            })
            .unwrap(),
            json!({
                "type": "list_remote_sessions",
                "data": { "node_id": "node-1", "offset": 20, "limit": 10 }
            })
        );
        assert_eq!(
            serde_json::to_value(ClientMsg::CreateRemoteSession {
                node_id: "node-1".to_string(),
                cwd: Some("/repo".to_string()),
                request_id: Some("req-1".to_string()),
            })
            .unwrap(),
            json!({
                "type": "create_remote_session",
                "data": { "node_id": "node-1", "cwd": "/repo", "request_id": "req-1" }
            })
        );
        assert_eq!(
            serde_json::to_value(ClientMsg::AttachRemoteSession {
                node_id: "node-1".to_string(),
                session_id: "s1".to_string(),
            })
            .unwrap(),
            json!({
                "type": "attach_remote_session",
                "data": { "node_id": "node-1", "session_id": "s1" }
            })
        );
        assert_eq!(
            serde_json::to_value(ClientMsg::DismissRemoteSession {
                session_id: "s1".to_string(),
            })
            .unwrap(),
            json!({
                "type": "dismiss_remote_session",
                "data": { "session_id": "s1" }
            })
        );
    }

    #[test]
    fn list_sessions_group_serializes_backend_pagination_fields() {
        let value = serde_json::to_value(ClientMsg::list_sessions_group(
            Some("/workspace/project".to_string()),
            "cursor-1".to_string(),
            10,
        ))
        .unwrap();
        assert_eq!(
            value,
            json!({
                "type": "list_sessions",
                "data": {
                    "mode": "group",
                    "cursor": "cursor-1",
                    "limit": 10,
                    "cwd": "/workspace/project",
                    "session_scope": "root"
                }
            })
        );
    }

    #[test]
    fn list_session_children_serializes_forks_scope() {
        let value = serde_json::to_value(ClientMsg::list_session_children(
            "root-1".to_string(),
            Some("child-cursor".to_string()),
            10,
        ))
        .unwrap();
        assert_eq!(
            value,
            json!({
                "type": "list_session_children",
                "data": {
                    "parent_session_id": "root-1",
                    "cursor": "child-cursor",
                    "limit": 10,
                    "session_scope": "forks"
                }
            })
        );
    }

    #[test]
    fn fork_session_serializes_message_id() {
        let value = serde_json::to_value(ClientMsg::ForkSession {
            message_id: "msg-123".to_string(),
        })
        .unwrap();
        assert_eq!(
            value,
            json!({
                "type": "fork_session",
                "data": {
                    "message_id": "msg-123"
                }
            })
        );
    }

    #[test]
    fn profile_messages_serialize() {
        let list = serde_json::to_value(ClientMsg::ListProfiles).unwrap();
        assert_eq!(list, json!({ "type": "list_profiles" }));

        let set = serde_json::to_value(ClientMsg::SetActiveProfile {
            profile_id: "fast".to_string(),
        })
        .unwrap();
        assert_eq!(
            set,
            json!({
                "type": "set_active_profile",
                "data": { "profile_id": "fast" }
            })
        );
    }

    #[test]
    fn new_session_serializes_optional_profile_id() {
        let without_profile = serde_json::to_value(ClientMsg::NewSession {
            cwd: None,
            request_id: None,
            profile_id: None,
        })
        .unwrap();
        assert_eq!(
            without_profile,
            json!({
                "type": "new_session",
                "data": { "cwd": null, "request_id": null }
            })
        );

        let with_profile = serde_json::to_value(ClientMsg::NewSession {
            cwd: Some("/repo".to_string()),
            request_id: None,
            profile_id: Some("fast".to_string()),
        })
        .unwrap();
        assert_eq!(
            with_profile,
            json!({
                "type": "new_session",
                "data": { "cwd": "/repo", "request_id": null, "profile_id": "fast" }
            })
        );
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum PromptBlock {
    Text { text: String },
    ResourceLink { name: String, uri: String },
}

// --- Server → Client messages ---
// We use a loose approach: parse the "type" tag, then decode known fields.

#[derive(Debug, Deserialize)]
pub struct RawServerMsg {
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
}

// Structured types we extract from RawServerMsg.data

#[derive(Debug, Deserialize)]
pub struct StateData {
    pub active_session_id: Option<String>,
    #[serde(default)]
    pub profiles: Vec<ProfileInfo>,
    #[serde(default)]
    pub active_profile_id: Option<String>,
    pub agents: Vec<AgentInfo>,
    pub agent_mode: Option<String>,
    /// Current reasoning effort level. `None` means "auto". Absent key means
    /// the server did not report it — callers should leave existing state intact.
    #[serde(default, deserialize_with = "deserialize_reasoning_effort")]
    pub reasoning_effort: ReasoningEffortField,
}

/// Three-state field for `reasoning_effort` in the `state` message:
/// - `Absent` — key was not present in JSON (leave existing TUI state alone)
/// - `Auto`   — key was `null` or `"auto"` (set to None / auto)
/// - `Set(s)` — key was a non-auto string like `"low"`, `"high"`, etc.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReasoningEffortField {
    #[default]
    Absent,
    Auto,
    Set(String),
}

fn deserialize_reasoning_effort<'de, D>(d: D) -> Result<ReasoningEffortField, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let v = Option::<String>::deserialize(d)?;
    Ok(match v.as_deref() {
        None | Some("auto") => ReasoningEffortField::Auto,
        Some(s) => ReasoningEffortField::Set(s.to_string()),
    })
}

#[derive(Debug, Deserialize)]
pub struct ReasoningEffortData {
    /// `None` or `"auto"` both map to the "auto" (no effort override) state.
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProfileInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub config_kind: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct SessionCreatedData {
    pub agent_id: String,
    pub session_id: String,
    pub request_id: Option<String>,
    #[serde(default)]
    pub profile_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct SessionListData {
    #[serde(default)]
    pub groups: Vec<SessionGroup>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub total_count: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
pub struct SessionChildrenData {
    pub parent_session_id: String,
    #[serde(default)]
    pub sessions: Vec<SessionSummary>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub total_count: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionGroup {
    pub cwd: Option<String>,
    #[serde(default)]
    pub sessions: Vec<SessionSummary>,
    /// ISO 8601 timestamp of the most recent activity in this group.
    #[serde(default)]
    pub latest_activity: Option<String>,
    #[serde(default)]
    pub total_count: Option<u64>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    /// Working directory for this session (may differ from group cwd for remote sessions).
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    /// Parent session ID if this is a forked session.
    #[serde(default)]
    pub parent_session_id: Option<String>,
    #[serde(default)]
    pub fork_origin: Option<String>,
    #[serde(default)]
    pub session_kind: Option<String>,
    /// Whether this session has child (forked) sessions.
    #[serde(default)]
    pub has_children: bool,
    /// Number of direct forked child sessions.
    #[serde(default)]
    pub fork_count: u64,
    #[serde(default)]
    pub children: Vec<SessionSummary>,
    #[serde(default)]
    pub children_next_cursor: Option<String>,
    #[serde(default)]
    pub children_total_count: Option<u64>,
    #[serde(default)]
    pub node: Option<String>,
    #[serde(default)]
    pub node_id: Option<String>,
    #[serde(default)]
    pub attached: Option<bool>,
    #[serde(default)]
    pub runtime_state: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SessionLoadedData {
    pub session_id: String,
    pub agent_id: String,
    pub audit: serde_json::Value,
    #[serde(default)]
    pub undo_stack: Vec<UndoStackFrame>,
    #[serde(default)]
    pub profile_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FileIndexEntry {
    pub path: String,
    pub is_dir: bool,
}

#[derive(Debug, Deserialize)]
pub struct FileIndexData {
    pub files: Vec<FileIndexEntry>,
    pub generated_at: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UndoStackFrame {
    pub message_id: String,
}

#[derive(Debug, Deserialize)]
pub struct UndoResultData {
    pub success: bool,
    pub message_id: Option<String>,
    #[serde(default)]
    pub reverted_files: Vec<String>,
    pub message: Option<String>,
    #[serde(default)]
    pub undo_stack: Vec<UndoStackFrame>,
}

#[derive(Debug, Deserialize)]
pub struct RedoResultData {
    pub success: bool,
    pub message: Option<String>,
    #[serde(default)]
    pub undo_stack: Vec<UndoStackFrame>,
}

#[derive(Debug, Deserialize)]
pub struct ForkResultData {
    pub success: bool,
    #[serde(default)]
    pub source_session_id: Option<String>,
    #[serde(default)]
    pub forked_session_id: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

#[cfg(test)]
mod fork_result_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fork_result_deserializes_optional_fields() {
        let result: ForkResultData = serde_json::from_value(json!({
            "success": true,
            "source_session_id": "source-1",
            "forked_session_id": "fork-1",
            "message": "forked"
        }))
        .unwrap();

        assert!(result.success);
        assert_eq!(result.source_session_id.as_deref(), Some("source-1"));
        assert_eq!(result.forked_session_id.as_deref(), Some("fork-1"));
        assert_eq!(result.message.as_deref(), Some("forked"));
    }
}

#[derive(Debug, Deserialize)]
pub struct EventData {
    pub agent_id: String,
    pub session_id: String,
    #[serde(default)]
    pub profile_id: Option<String>,
    pub event: EventEnvelope,
}

/// Like [`EventData`] but keeps the event as raw JSON so an unknown
/// event kind doesn't prevent routing the message entirely.
#[derive(Debug, Deserialize)]
pub struct EventDataRaw {
    pub agent_id: String,
    pub session_id: String,
    #[serde(default)]
    pub profile_id: Option<String>,
    pub event: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct SessionEventsData {
    pub session_id: String,
    pub agent_id: String,
    #[serde(default)]
    pub profile_id: Option<String>,
    pub events: Vec<EventEnvelope>,
}

/// Like [`SessionEventsData`] but with raw JSON values for events so unknown
/// event kinds don't blow up deserialization of the whole batch.
#[derive(Debug, Deserialize)]
pub struct SessionEventsDataRaw {
    pub session_id: String,
    pub agent_id: String,
    #[serde(default)]
    pub profile_id: Option<String>,
    pub events: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum EventEnvelope {
    Durable(InnerEvent),
    Ephemeral(InnerEvent),
}

impl EventEnvelope {
    pub fn kind(&self) -> &EventKind {
        match self {
            Self::Durable(e) | Self::Ephemeral(e) => &e.kind,
        }
    }

    pub fn timestamp(&self) -> Option<i64> {
        match self {
            Self::Durable(e) | Self::Ephemeral(e) => e.timestamp,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct InnerEvent {
    pub kind: EventKind,
    pub timestamp: Option<i64>,
}

/// Flat event shape used in AuditView.events (not wrapped in EventEnvelope).
#[derive(Debug, Deserialize)]
pub struct AgentEvent {
    pub kind: EventKind,
    pub timestamp: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum EventKind {
    TurnStarted,
    PromptReceived {
        content: serde_json::Value,
        message_id: Option<String>,
    },
    UserMessageStored {
        content: serde_json::Value,
    },
    AssistantMessageStored {
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    },
    AssistantContentDelta {
        content: String,
        message_id: Option<String>,
    },
    AssistantThinkingDelta {
        content: String,
        message_id: Option<String>,
    },
    CompactionStart {
        token_estimate: u32,
    },
    CompactionEnd {
        summary: String,
        summary_len: u32,
    },
    LlmRequestStart {
        message_count: Option<u32>,
    },
    LlmRequestEnd {
        finish_reason: Option<String>,
        cost_usd: Option<f64>,
        cumulative_cost_usd: Option<f64>,
        context_tokens: Option<u64>,
        tool_calls: Option<u32>,
        metrics: Option<serde_json::Value>,
    },
    ToolCallStart {
        tool_call_id: Option<String>,
        tool_name: String,
        arguments: Option<serde_json::Value>,
    },
    ToolCallEnd {
        tool_call_id: Option<String>,
        tool_name: String,
        is_error: Option<bool>,
        result: Option<String>,
    },
    SnapshotStart {
        policy: String,
    },
    SnapshotEnd {
        summary: Option<String>,
    },
    ProgressRecorded {
        progress_entry: ProgressEntry,
    },
    ArtifactRecorded {
        artifact: ArtifactInfo,
    },
    SessionQueued {
        reason: String,
    },
    SessionConfigured {
        cwd: Option<String>,
        #[serde(default)]
        mcp_servers: Vec<serde_json::Value>,
        limits: Option<SessionLimits>,
    },
    ToolsAvailable {
        #[serde(default)]
        tools: Vec<ToolInfo>,
        #[serde(default)]
        tools_hash: Option<serde_json::Value>,
    },
    ProviderChanged {
        provider: String,
        model: String,
        config_id: Option<i64>,
        context_limit: Option<u64>,
        /// Mesh node hosting the provider when the session routes LLM calls remotely.
        #[serde(default)]
        provider_node_id: Option<String>,
    },
    ElicitationRequested {
        elicitation_id: String,
        session_id: String,
        message: String,
        requested_schema: serde_json::Value,
        source: String,
    },
    /// Emitted when a session's mode changes (per-session mode in actor model).
    /// Durable — appears in the audit journal and replayed on session load.
    /// The last occurrence in a session's audit gives the session's last-used mode.
    SessionModeChanged {
        mode: String,
    },
    Error {
        message: String,
    },
    Cancelled,
    SessionCreated,
    DelegationRequested {
        delegation: DelegationData,
    },
    DelegationCompleted {
        delegation_id: String,
        #[serde(default)]
        result: Option<String>,
    },
    DelegationFailed {
        delegation_id: String,
        #[serde(default)]
        error: Option<String>,
    },
    DelegationCancelled {
        delegation_id: String,
    },
    SessionForked {
        #[serde(default)]
        child_session_id: Option<String>,
        #[serde(default)]
        origin: Option<String>,
        /// Delegation public_id when origin="delegation".
        #[serde(default)]
        fork_point_ref: Option<String>,
        /// The agent the child session was delegated to.
        #[serde(default)]
        target_agent_id: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProgressEntry {
    pub kind: ProgressKind,
    pub content: String,
    pub metadata: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArtifactInfo {
    pub kind: String,
    pub uri: Option<String>,
    pub path: Option<String>,
    pub summary: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgressKind {
    ToolCall,
    Artifact,
    Note,
    Checkpoint,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SessionLimits {
    pub max_steps: Option<u32>,
    pub max_turns: Option<u32>,
    pub max_cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolInfo {
    #[serde(rename = "type", default)]
    pub tool_type: String,
    #[serde(default)]
    pub function: Option<FunctionToolInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionToolInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// Subset of the server-side `Delegation` struct that we care about.
#[derive(Debug, Clone, Deserialize)]
pub struct DelegationData {
    pub public_id: String,
    #[serde(default)]
    pub target_agent_id: Option<String>,
    #[serde(default)]
    pub objective: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AudioCapabilitiesData {
    #[serde(default)]
    pub stt_models: Vec<AudioModelInfo>,
    #[serde(default)]
    pub tts_models: Vec<AudioModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct AudioModelInfo {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct ProviderCapabilitiesData {
    #[serde(default)]
    pub providers: Vec<ProviderCapabilityEntry>,
}

#[derive(Debug, Deserialize)]
pub struct ProviderCapabilityEntry {
    pub provider: String,
    pub supports_custom_models: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub label: String,
    pub provider: String,
    pub model: String,
    pub node_id: Option<String>,
    #[serde(default)]
    pub node_label: Option<String>,
    pub family: Option<String>,
    pub quant: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AgentModeData {
    pub mode: String,
}

#[derive(Debug, Deserialize)]
pub struct ErrorData {
    pub message: String,
}

// ── Auth / token types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    #[serde(rename = "oauth")]
    OAuth,
    ApiKey,
    EnvVar,
}

impl std::fmt::Display for AuthMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OAuth => write!(f, "OAuth"),
            Self::ApiKey => write!(f, "API Key"),
            Self::EnvVar => write!(f, "Env"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OAuthStatus {
    Connected,
    Expired,
    NotAuthenticated,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AuthProviderEntry {
    pub provider: String,
    pub display_name: String,
    pub oauth_status: Option<OAuthStatus>,
    pub has_stored_api_key: bool,
    pub has_env_api_key: bool,
    pub env_var_name: Option<String>,
    pub supports_oauth: bool,
    pub preferred_method: Option<AuthMethod>,
}

impl AuthProviderEntry {
    /// Provider supports only OAuth (no API key env var).
    pub fn is_oauth_only(&self) -> bool {
        self.supports_oauth && self.env_var_name.is_none()
    }

    /// Provider supports only API key (no OAuth).
    pub fn is_api_key_only(&self) -> bool {
        !self.supports_oauth && self.env_var_name.is_some()
    }

    /// Provider supports multiple auth methods (both OAuth and API key).
    pub fn has_multiple_auth_methods(&self) -> bool {
        self.supports_oauth && self.env_var_name.is_some()
    }

    /// Provider requires OAuth but the build doesn't include it.
    pub fn is_unconfigurable(&self) -> bool {
        !self.supports_oauth && self.env_var_name.is_none()
    }

    /// Resolve which auth method is effectively active.
    pub fn effective_auth(&self) -> Option<AuthMethod> {
        let pref = self.preferred_method;
        let order: &[AuthMethod] = if let Some(p) = pref {
            // Preferred first, then defaults
            match p {
                AuthMethod::OAuth => &[AuthMethod::OAuth, AuthMethod::ApiKey, AuthMethod::EnvVar],
                AuthMethod::ApiKey => &[AuthMethod::ApiKey, AuthMethod::OAuth, AuthMethod::EnvVar],
                AuthMethod::EnvVar => &[AuthMethod::EnvVar, AuthMethod::OAuth, AuthMethod::ApiKey],
            }
        } else if self.supports_oauth {
            &[AuthMethod::OAuth, AuthMethod::ApiKey, AuthMethod::EnvVar]
        } else {
            &[AuthMethod::ApiKey, AuthMethod::EnvVar]
        };

        for method in order {
            match method {
                AuthMethod::OAuth => {
                    if self.oauth_status == Some(OAuthStatus::Connected) {
                        return Some(AuthMethod::OAuth);
                    }
                }
                AuthMethod::ApiKey => {
                    if self.has_stored_api_key {
                        return Some(AuthMethod::ApiKey);
                    }
                }
                AuthMethod::EnvVar => {
                    if self.has_env_api_key {
                        return Some(AuthMethod::EnvVar);
                    }
                }
            }
        }
        None
    }

    /// Badge label for current auth state.
    pub fn auth_badge_label(&self) -> &'static str {
        if self.is_unconfigurable() {
            return "OAuth required";
        }
        if self.oauth_status == Some(OAuthStatus::Expired) {
            return "Expired";
        }
        match self.effective_auth() {
            Some(AuthMethod::OAuth) => "OAuth",
            Some(AuthMethod::ApiKey) => "API Key",
            Some(AuthMethod::EnvVar) => "Env",
            None => "Not configured",
        }
    }

    /// Whether the badge indicates a successful/active auth.
    pub fn is_auth_active(&self) -> bool {
        self.effective_auth().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OAuthFlowKind {
    RedirectCode,
    DevicePoll,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OAuthFlowData {
    pub flow_id: String,
    pub provider: String,
    pub authorization_url: String,
    pub flow_kind: OAuthFlowKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OAuthResultData {
    pub provider: String,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ApiTokenResultData {
    pub provider: String,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthProvidersData {
    pub providers: Vec<AuthProviderEntry>,
}
