use serde::Deserialize;

use crate::domain::auth::{AuthProviderEntry, OAuthFlowKind};

#[derive(Debug, Clone, Deserialize)]
pub struct UndoStackFrame {
    pub message_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UndoResultData {
    pub success: bool,
    pub message_id: Option<String>,
    #[serde(default)]
    pub reverted_files: Vec<String>,
    pub message: Option<String>,
    #[serde(default)]
    pub undo_stack: Vec<UndoStackFrame>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RedoResultData {
    pub success: bool,
    pub message: Option<String>,
    #[serde(default)]
    pub undo_stack: Vec<UndoStackFrame>,
}

#[cfg(test)]
mod session_mutation_data_tests {
    use super::{RedoResultData, UndoResultData};
    use serde_json::json;

    #[test]
    fn undo_result_deserializes_success_fields_and_stack_order() {
        let result: UndoResultData = serde_json::from_value(json!({
            "success": true,
            "message_id": "message-2",
            "reverted_files": ["src/a.rs", "src/b.rs"],
            "message": "undone",
            "undo_stack": [
                { "message_id": "message-1" },
                { "message_id": "message-2" }
            ]
        }))
        .unwrap();

        assert!(result.success);
        assert_eq!(result.message_id.as_deref(), Some("message-2"));
        assert_eq!(result.reverted_files, ["src/a.rs", "src/b.rs"]);
        assert_eq!(result.message.as_deref(), Some("undone"));
        assert_eq!(result.undo_stack[0].message_id, "message-1");
        assert_eq!(result.undo_stack[1].message_id, "message-2");
    }

    #[test]
    fn undo_result_deserializes_failure_with_default_collections() {
        let result: UndoResultData = serde_json::from_value(json!({
            "success": false,
            "message_id": null,
            "message": "undo rejected"
        }))
        .unwrap();

        assert!(!result.success);
        assert!(result.reverted_files.is_empty());
        assert!(result.undo_stack.is_empty());
        assert_eq!(result.message.as_deref(), Some("undo rejected"));
    }

    #[test]
    fn redo_result_deserializes_success_fields_and_stack_order() {
        let result: RedoResultData = serde_json::from_value(json!({
            "success": true,
            "message": "redone",
            "undo_stack": [
                { "message_id": "message-1" },
                { "message_id": "message-2" }
            ]
        }))
        .unwrap();

        assert!(result.success);
        assert_eq!(result.message.as_deref(), Some("redone"));
        assert_eq!(result.undo_stack[0].message_id, "message-1");
        assert_eq!(result.undo_stack[1].message_id, "message-2");
    }

    #[test]
    fn redo_result_deserializes_failure_with_default_stack() {
        let result: RedoResultData = serde_json::from_value(json!({
            "success": false,
            "message": "redo rejected"
        }))
        .unwrap();

        assert!(!result.success);
        assert!(result.undo_stack.is_empty());
        assert_eq!(result.message.as_deref(), Some("redo rejected"));
    }
}

#[allow(dead_code)] // Active event wire contract retains fields consumed across targets.
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

#[allow(dead_code)] // Nested active EventKind payload.
#[derive(Debug, Clone, Deserialize)]
pub struct ProgressEntry {
    pub kind: ProgressKind,
    pub content: String,
    pub metadata: Option<String>,
    pub created_at: String,
}

#[allow(dead_code)] // Nested active EventKind payload.
#[derive(Debug, Clone, Deserialize)]
pub struct ArtifactInfo {
    pub kind: String,
    pub uri: Option<String>,
    pub path: Option<String>,
    pub summary: Option<String>,
    pub created_at: String,
}

#[allow(dead_code)] // Nested active EventKind payload.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgressKind {
    ToolCall,
    Artifact,
    Note,
    Checkpoint,
}

#[allow(dead_code)] // Nested active EventKind payload.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionLimits {
    pub max_steps: Option<u32>,
    pub max_turns: Option<u32>,
    pub max_cost_usd: Option<f64>,
}

#[allow(dead_code)] // Nested active EventKind payload.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolInfo {
    #[serde(rename = "type", default)]
    pub tool_type: String,
    #[serde(default)]
    pub function: Option<FunctionToolInfo>,
}

#[allow(dead_code)] // Nested active EventKind payload.
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
#[allow(dead_code)] // Nested active EventKind payload.
#[derive(Debug, Clone, Deserialize)]
pub struct DelegationData {
    pub public_id: String,
    #[serde(default)]
    pub target_agent_id: Option<String>,
    #[serde(default)]
    pub objective: Option<String>,
}

#[allow(dead_code)] // Mesh status preserves server fields not currently rendered.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshStatusInfo {
    pub enabled: bool,
    #[serde(default)]
    pub peer_id: Option<String>,
    #[serde(default)]
    pub transport: Option<String>,
    #[serde(default)]
    pub known_peer_count: u32,
    #[serde(default)]
    pub has_invite_store: bool,
    #[serde(default)]
    pub has_mesh_state_store: bool,
    #[serde(default)]
    pub scopes: Vec<MeshScopeInfo>,
}

#[allow(dead_code)] // Nested active mesh status payload.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshScopeInfo {
    pub kind: String,
    pub id: String,
}

#[allow(dead_code)] // Remote node DTO preserves active mesh metadata.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteNodeInfo {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub active_sessions: u32,
    #[serde(default)]
    pub transport: String,
    #[serde(default)]
    pub last_seen_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshNodesInfo {
    #[serde(default)]
    pub nodes: Vec<RemoteNodeInfo>,
}

#[allow(dead_code)] // Remote session DTO preserves active mesh metadata.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteSessionInfo {
    pub id: String,
    pub node_id: String,
    #[serde(default)]
    pub node_label: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
}

#[allow(dead_code)] // Active paged remote-session response boundary.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteSessionListInfo {
    pub node_id: String,
    #[serde(default)]
    pub sessions: Vec<RemoteSessionInfo>,
    #[serde(default)]
    pub next_offset: Option<u32>,
    #[serde(default)]
    pub total_count: u32,
}

#[allow(dead_code)] // Active attach response preserves snapshot and config metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct RemoteSessionAttachInfo {
    pub session_id: String,
    pub node_id: String,
    #[serde(default)]
    pub attached: bool,
    #[serde(default)]
    pub config_options: Vec<serde_json::Value>,
    #[serde(default)]
    pub snapshot: serde_json::Value,
}

#[allow(dead_code)] // Active invite response preserves server metadata.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshInviteCreatedInfo {
    pub invite_id: String,
    pub url: String,
    #[serde(default)]
    pub qr_code: Option<String>,
    #[serde(default)]
    pub expires_at: u64,
    #[serde(default)]
    pub max_uses: u32,
    #[serde(default)]
    pub mesh_name: Option<String>,
}

// ── Auth / token types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OAuthFlowDto {
    pub flow_id: String,
    pub provider: String,
    pub authorization_url: String,
    pub flow_kind: OAuthFlowKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OAuthResultDto {
    pub provider: String,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthProvidersData {
    pub providers: Vec<AuthProviderEntry>,
}

#[cfg(test)]
mod auth_data_tests {
    use super::{AuthProvidersData, OAuthFlowDto, OAuthResultDto};
    use crate::domain::auth::{AuthMethod, OAuthFlowKind, OAuthStatus};
    use serde_json::json;

    #[test]
    fn oauth_flow_dto_deserializes_both_flow_kinds_and_requires_fields() {
        for (wire_kind, expected_kind) in [
            ("redirect_code", OAuthFlowKind::RedirectCode),
            ("device_poll", OAuthFlowKind::DevicePoll),
        ] {
            let flow: OAuthFlowDto = serde_json::from_value(json!({
                "flow_id": "flow-123",
                "provider": "openai",
                "authorization_url": "https://example.test/authorize",
                "flow_kind": wire_kind
            }))
            .unwrap();
            assert_eq!(flow.flow_id, "flow-123");
            assert_eq!(flow.provider, "openai");
            assert_eq!(flow.authorization_url, "https://example.test/authorize");
            assert_eq!(flow.flow_kind, expected_kind);
        }

        let complete = json!({
            "flow_id": "flow-123",
            "provider": "openai",
            "authorization_url": "https://example.test/authorize",
            "flow_kind": "redirect_code"
        });
        for field in ["flow_id", "provider", "authorization_url", "flow_kind"] {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<OAuthFlowDto>(missing).is_err());
        }
    }

    #[test]
    fn oauth_result_dto_deserializes_success_failure_and_requires_fields() {
        let success: OAuthResultDto = serde_json::from_value(json!({
            "provider": "openai",
            "success": true,
            "message": "connected"
        }))
        .unwrap();
        assert_eq!(success.provider, "openai");
        assert!(success.success);
        assert_eq!(success.message, "connected");

        let failure: OAuthResultDto = serde_json::from_value(json!({
            "provider": "anthropic",
            "success": false,
            "message": "authorization denied"
        }))
        .unwrap();
        assert_eq!(failure.provider, "anthropic");
        assert!(!failure.success);
        assert_eq!(failure.message, "authorization denied");

        let complete = json!({
            "provider": "openai",
            "success": true,
            "message": "connected"
        });
        for field in ["provider", "success", "message"] {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<OAuthResultDto>(missing).is_err());
        }
    }

    #[test]
    fn auth_providers_data_deserializes_mixed_providers() {
        let data: AuthProvidersData = serde_json::from_value(json!({
            "providers": [
                {
                    "provider": "openai",
                    "display_name": "OpenAI",
                    "oauth_status": "not_authenticated",
                    "has_stored_api_key": false,
                    "has_env_api_key": true,
                    "env_var_name": "OPENAI_API_KEY",
                    "supports_oauth": true,
                    "preferred_method": "oauth"
                },
                {
                    "provider": "groq",
                    "display_name": "Groq",
                    "oauth_status": null,
                    "has_stored_api_key": true,
                    "has_env_api_key": false,
                    "env_var_name": "GROQ_API_KEY",
                    "supports_oauth": false,
                    "preferred_method": "api_key"
                },
                {
                    "provider": "local",
                    "display_name": "Local",
                    "oauth_status": null,
                    "has_stored_api_key": false,
                    "has_env_api_key": false,
                    "env_var_name": null,
                    "supports_oauth": false,
                    "preferred_method": null
                }
            ]
        }))
        .unwrap();

        assert_eq!(data.providers.len(), 3);
        assert_eq!(
            data.providers[0].oauth_status,
            Some(OAuthStatus::NotAuthenticated)
        );
        assert_eq!(data.providers[0].preferred_method, Some(AuthMethod::OAuth));
        assert_eq!(data.providers[1].preferred_method, Some(AuthMethod::ApiKey));
        assert_eq!(data.providers[2].preferred_method, None);
    }
}
