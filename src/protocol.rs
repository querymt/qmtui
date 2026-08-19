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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshStatusDto {
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
    pub scopes: Vec<MeshScopeDto>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshScopeDto {
    pub kind: String,
    pub id: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteNodeDto {
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
pub struct MeshNodesDto {
    #[serde(default)]
    pub nodes: Vec<RemoteNodeDto>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteSessionDto {
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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteSessionListDto {
    pub node_id: String,
    #[serde(default)]
    pub sessions: Vec<RemoteSessionDto>,
    #[serde(default)]
    pub next_offset: Option<u32>,
    #[serde(default)]
    pub total_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RemoteSessionAttachDto {
    pub session_id: String,
    pub node_id: String,
    #[serde(default)]
    pub attached: bool,
    #[serde(default)]
    pub config_options: Vec<serde_json::Value>,
    #[serde(default)]
    pub snapshot: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshInviteCreatedDto {
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

#[cfg(test)]
mod mesh_dto_tests {
    use super::{
        MeshInviteCreatedDto, MeshNodesDto, MeshStatusDto, RemoteNodeDto, RemoteSessionAttachDto,
        RemoteSessionDto, RemoteSessionListDto,
    };
    use serde_json::json;

    #[test]
    fn mesh_dtos_deserialize_representative_values_and_ignore_unknown_fields() {
        let status: MeshStatusDto = serde_json::from_value(json!({
            "enabled": true,
            "peer_id": "peer-1",
            "transport": "webrtc",
            "known_peer_count": 2,
            "has_invite_store": true,
            "has_mesh_state_store": true,
            "scopes": [{ "kind": "team", "id": "scope-1" }],
            "unknown": "ignored"
        }))
        .unwrap();
        assert!(status.enabled);
        assert_eq!(status.scopes[0].kind, "team");
        assert_eq!(status.scopes[0].id, "scope-1");

        let nodes: MeshNodesDto = serde_json::from_value(json!({
            "nodes": [{
                "id": "node-1",
                "label": "Framework",
                "capabilities": ["sessions"],
                "active_sessions": 3,
                "transport": "relay",
                "last_seen_at": "2025-01-01T00:00:00Z"
            }]
        }))
        .unwrap();
        assert_eq!(nodes.nodes[0].id, "node-1");
        assert_eq!(nodes.nodes[0].capabilities, ["sessions"]);
        assert_eq!(nodes.nodes[0].active_sessions, 3);

        let node_defaults: RemoteNodeDto = serde_json::from_value(json!({
            "id": "node-2",
            "label": "Defaults"
        }))
        .unwrap();
        assert!(node_defaults.capabilities.is_empty());
        assert_eq!(node_defaults.active_sessions, 0);
        assert!(node_defaults.transport.is_empty());
        assert_eq!(node_defaults.last_seen_at, None);

        let sessions: RemoteSessionListDto = serde_json::from_value(json!({
            "node_id": "node-1",
            "sessions": [{
                "id": "session-1",
                "node_id": "node-1",
                "node_label": "Framework",
                "title": "Fix boundary",
                "cwd": "/repo",
                "updated_at": "now",
                "profile_id": "profile-1",
                "model_id": "model-1"
            }],
            "next_offset": 50,
            "total_count": 51
        }))
        .unwrap();
        assert_eq!(sessions.sessions[0].model_id.as_deref(), Some("model-1"));
        assert_eq!(sessions.next_offset, Some(50));
        assert_eq!(sessions.total_count, 51);

        let session_defaults: RemoteSessionDto = serde_json::from_value(json!({
            "id": "session-2",
            "node_id": "node-1"
        }))
        .unwrap();
        assert_eq!(session_defaults.title, None);
        assert_eq!(session_defaults.cwd, None);
        assert_eq!(session_defaults.model_id, None);

        let attach: RemoteSessionAttachDto = serde_json::from_value(json!({
            "session_id": "session-1",
            "node_id": "node-1",
            "attached": true,
            "config_options": [],
            "snapshot": {
                "audit": [{ "kind": "message", "data": { "nested": [1, 2] } }],
                "cursor": { "position": 1 },
                "delegationUpdates": [{ "id": "delegate-1" }]
            }
        }))
        .unwrap();
        assert!(attach.config_options.is_empty());
        assert_eq!(
            attach.snapshot.as_ref().unwrap()["audit"][0]["data"]["nested"],
            json!([1, 2])
        );
        assert_eq!(
            attach.snapshot.as_ref().unwrap()["delegationUpdates"][0]["id"],
            "delegate-1"
        );

        let detached: RemoteSessionAttachDto = serde_json::from_value(json!({
            "session_id": "session-2",
            "node_id": "node-1",
            "attached": false,
            "config_options": []
        }))
        .unwrap();
        assert_eq!(detached.snapshot, None);

        let null_snapshot: RemoteSessionAttachDto = serde_json::from_value(json!({
            "session_id": "session-3",
            "node_id": "node-1",
            "snapshot": null
        }))
        .unwrap();
        assert_eq!(null_snapshot.snapshot, None);

        let invite: MeshInviteCreatedDto = serde_json::from_value(json!({
            "invite_id": "invite-1",
            "url": "qmt://mesh/join/token",
            "qr_code": "QR",
            "expires_at": 123,
            "max_uses": 2,
            "mesh_name": "Team"
        }))
        .unwrap();
        assert_eq!(invite.mesh_name.as_deref(), Some("Team"));
    }

    #[test]
    fn mesh_dto_defaults_preserve_existing_wire_behavior() {
        let status: MeshStatusDto = serde_json::from_value(json!({ "enabled": false })).unwrap();
        assert!(status.scopes.is_empty());
        assert_eq!(status.known_peer_count, 0);

        let nodes: MeshNodesDto = serde_json::from_value(json!({})).unwrap();
        assert!(nodes.nodes.is_empty());

        let sessions: RemoteSessionListDto =
            serde_json::from_value(json!({ "node_id": "node-1" })).unwrap();
        assert!(sessions.sessions.is_empty());
        assert_eq!(sessions.next_offset, None);
        assert_eq!(sessions.total_count, 0);

        let attach: RemoteSessionAttachDto =
            serde_json::from_value(json!({ "session_id": "session-1", "node_id": "node-1" }))
                .unwrap();
        assert!(!attach.attached);
        assert!(attach.config_options.is_empty());
        assert_eq!(attach.snapshot, None);

        let invite: MeshInviteCreatedDto = serde_json::from_value(json!({
            "invite_id": "invite-1",
            "url": "qmt://mesh/join/token"
        }))
        .unwrap();
        assert_eq!(invite.expires_at, 0);
        assert_eq!(invite.max_uses, 0);
        assert_eq!(invite.qr_code, None);
    }
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
