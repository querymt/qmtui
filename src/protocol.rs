use serde::{Deserialize, Serialize};

use crate::command::Command;
use crate::domain::auth::{AuthMethod, AuthProviderEntry, OAuthFlowKind};

pub(crate) use crate::command::SessionListRequest;

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
        #[serde(skip)]
        request: SessionListRequest,
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
    CreateMeshInvite {
        #[serde(skip_serializing_if = "Option::is_none")]
        mesh_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ttl: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
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
    ListProfileAgents {
        profile_id: String,
    },
    SetDelegateModel {
        session_id: String,
        agent_id: String,
        model_id: Option<String>,
        node_id: Option<String>,
    },
    NewSession {
        cwd: Option<String>,
        request_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        profile_id: Option<String>,
    },
    LoadSession {
        session_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
    },
    Prompt {
        prompt: Vec<PromptBlock>,
        #[serde(skip)]
        local_id: String,
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

impl From<Command> for ClientMsg {
    fn from(command: Command) -> Self {
        match command {
            Command::ListSessions { request, cursor } => {
                let (mode, limit, cwd, include_remote) = match &request {
                    SessionListRequest::Discovery => (None, None, None, Some(true)),
                    SessionListRequest::WorkspaceFirstPage { cwd }
                    | SessionListRequest::WorkspaceContinuation { cwd } => {
                        (Some("group".to_string()), Some(10), Some(cwd.clone()), None)
                    }
                };
                Self::ListSessions {
                    request,
                    mode,
                    cursor,
                    limit,
                    cwd,
                    query: None,
                    include_remote,
                    session_scope: SessionScope::Root,
                }
            }
            Command::ListRemoteSessions {
                node_id,
                offset,
                limit,
            } => Self::ListRemoteSessions {
                node_id,
                offset: Some(offset),
                limit: Some(limit),
            },
            Command::CreateMeshInvite {
                mesh_name,
                ttl,
                max_uses,
            } => Self::CreateMeshInvite {
                mesh_name,
                ttl,
                max_uses,
            },
            Command::ListSessionChildren {
                parent_session_id,
                cursor,
                limit,
            } => Self::ListSessionChildren {
                parent_session_id,
                cursor,
                limit: Some(limit),
                session_scope: SessionScope::Forks,
            },
            Command::SetReasoningEffort { reasoning_effort } => {
                Self::SetReasoningEffort { reasoning_effort }
            }
            Command::ListProfileAgents { profile_id } => Self::ListProfileAgents { profile_id },
            Command::SetDelegateModel {
                session_id,
                agent_id,
                model_id,
                node_id,
            } => Self::SetDelegateModel {
                session_id,
                agent_id,
                model_id,
                node_id,
            },
            Command::LoadSession { session_id, cwd } => Self::LoadSession { session_id, cwd },
            Command::SubscribeSession {
                session_id,
                agent_id,
            } => Self::SubscribeSession {
                session_id,
                agent_id,
            },
            Command::GetFileIndex => Self::GetFileIndex,
            Command::SetAgentMode { mode } => Self::SetAgentMode { mode },
            Command::ListAuthProviders => Self::ListAuthProviders,
        }
    }
}

impl ClientMsg {
    pub fn list_sessions_browse() -> Self {
        Command::list_sessions_browse().into()
    }

    pub fn list_sessions_discovery(cursor: Option<String>) -> Self {
        Command::list_sessions_discovery(cursor).into()
    }

    pub fn list_sessions_workspace(cwd: String) -> Self {
        Command::list_sessions_workspace(cwd).into()
    }

    pub fn list_sessions_group(cwd: String, cursor: String) -> Self {
        Command::list_sessions_group(cwd, cursor).into()
    }

    pub fn list_session_children(
        parent_session_id: String,
        cursor: Option<String>,
        limit: u32,
    ) -> Self {
        Command::list_session_children(parent_session_id, cursor, limit).into()
    }
}

#[cfg(test)]
mod command_conversion_tests {
    use super::{ClientMsg, SessionListRequest, SessionScope};
    use crate::command::Command;

    #[test]
    fn list_session_commands_preserve_semantics_and_bound_wire_fields() {
        let discovery = ClientMsg::from(Command::list_sessions_discovery(Some("root-2".into())));
        assert!(matches!(
            discovery,
            ClientMsg::ListSessions {
                request: SessionListRequest::Discovery,
                mode: None,
                cursor: Some(cursor),
                limit: None,
                cwd: None,
                query: None,
                include_remote: Some(true),
                session_scope: SessionScope::Root,
            } if cursor == "root-2"
        ));

        let first = ClientMsg::from(Command::list_sessions_workspace("/repo".into()));
        assert!(matches!(
            first,
            ClientMsg::ListSessions {
                request: SessionListRequest::WorkspaceFirstPage { cwd: request_cwd },
                mode: Some(mode),
                cursor: None,
                limit: Some(10),
                cwd: Some(wire_cwd),
                query: None,
                include_remote: None,
                session_scope: SessionScope::Root,
            } if request_cwd == "/repo" && mode == "group" && wire_cwd == "/repo"
        ));

        let continuation = ClientMsg::from(Command::list_sessions_group(
            "/repo".into(),
            "workspace-2".into(),
        ));
        assert!(matches!(
            continuation,
            ClientMsg::ListSessions {
                request: SessionListRequest::WorkspaceContinuation { cwd: request_cwd },
                mode: Some(mode),
                cursor: Some(cursor),
                limit: Some(10),
                cwd: Some(wire_cwd),
                query: None,
                include_remote: None,
                session_scope: SessionScope::Root,
            } if request_cwd == "/repo"
                && mode == "group"
                && cursor == "workspace-2"
                && wire_cwd == "/repo"
        ));

        let children = ClientMsg::from(Command::list_session_children(
            "parent-1".into(),
            Some("child-2".into()),
            25,
        ));
        assert!(matches!(
            children,
            ClientMsg::ListSessionChildren {
                parent_session_id,
                cursor: Some(cursor),
                limit: Some(25),
                session_scope: SessionScope::Forks,
            } if parent_session_id == "parent-1" && cursor == "child-2"
        ));
    }

    #[test]
    fn mesh_and_model_commands_convert_exactly() {
        let remote = ClientMsg::from(Command::ListRemoteSessions {
            node_id: "node-1".into(),
            offset: 5,
            limit: 50,
        });
        assert!(matches!(
            remote,
            ClientMsg::ListRemoteSessions {
                node_id,
                offset: Some(5),
                limit: Some(50),
            } if node_id == "node-1"
        ));

        let invite = ClientMsg::from(Command::CreateMeshInvite {
            mesh_name: Some("team".into()),
            ttl: Some("24h".into()),
            max_uses: Some(3),
        });
        assert!(matches!(
            invite,
            ClientMsg::CreateMeshInvite {
                mesh_name: Some(mesh_name),
                ttl: Some(ttl),
                max_uses: Some(3),
            } if mesh_name == "team" && ttl == "24h"
        ));

        let effort = ClientMsg::from(Command::SetReasoningEffort {
            reasoning_effort: "high".into(),
        });
        assert!(matches!(
            effort,
            ClientMsg::SetReasoningEffort { reasoning_effort } if reasoning_effort == "high"
        ));

        let agents = ClientMsg::from(Command::ListProfileAgents {
            profile_id: "code".into(),
        });
        assert!(matches!(
            agents,
            ClientMsg::ListProfileAgents { profile_id } if profile_id == "code"
        ));

        let delegate = ClientMsg::from(Command::SetDelegateModel {
            session_id: "session-1".into(),
            agent_id: "reviewer".into(),
            model_id: Some("provider/model".into()),
            node_id: Some("node-2".into()),
        });
        assert!(matches!(
            delegate,
            ClientMsg::SetDelegateModel {
                session_id,
                agent_id,
                model_id: Some(model_id),
                node_id: Some(node_id),
            } if session_id == "session-1"
                && agent_id == "reviewer"
                && model_id == "provider/model"
                && node_id == "node-2"
        ));
    }

    #[test]
    fn session_and_input_commands_convert_exactly() {
        let load = ClientMsg::from(Command::LoadSession {
            session_id: "session-1".into(),
            cwd: Some("/repo".into()),
        });
        assert!(matches!(
            load,
            ClientMsg::LoadSession {
                session_id,
                cwd: Some(cwd),
            } if session_id == "session-1" && cwd == "/repo"
        ));

        let subscribe = ClientMsg::from(Command::SubscribeSession {
            session_id: "session-1".into(),
            agent_id: Some("agent-1".into()),
        });
        assert!(matches!(
            subscribe,
            ClientMsg::SubscribeSession {
                session_id,
                agent_id: Some(agent_id),
            } if session_id == "session-1" && agent_id == "agent-1"
        ));

        assert!(matches!(
            ClientMsg::from(Command::GetFileIndex),
            ClientMsg::GetFileIndex
        ));

        let mode = ClientMsg::from(Command::SetAgentMode {
            mode: "plan".into(),
        });
        assert!(matches!(
            mode,
            ClientMsg::SetAgentMode { mode } if mode == "plan"
        ));

        assert!(matches!(
            ClientMsg::from(Command::ListAuthProviders),
            ClientMsg::ListAuthProviders
        ));
    }
}

#[cfg(test)]
mod client_msg_tests {
    use super::ClientMsg;
    use crate::domain::auth::AuthMethod;
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
    fn list_sessions_workspace_serializes_cwd_without_cursor() {
        let value = serde_json::to_value(ClientMsg::list_sessions_workspace(
            "/workspace/project".to_string(),
        ))
        .unwrap();
        assert_eq!(
            value,
            json!({
                "type": "list_sessions",
                "data": {
                    "mode": "group",
                    "limit": 10,
                    "cwd": "/workspace/project",
                    "session_scope": "root"
                }
            })
        );
    }

    #[test]
    fn list_sessions_group_omits_include_remote_for_pagination() {
        let value = serde_json::to_value(ClientMsg::list_sessions_group(
            "/workspace/project".to_string(),
            "cursor-1".to_string(),
        ))
        .unwrap();
        assert!(value["data"].get("include_remote").is_none());
    }

    #[test]
    fn list_sessions_discovery_serializes_opaque_cursor() {
        let value = serde_json::to_value(ClientMsg::list_sessions_discovery(Some(
            "opaque-root-2".to_string(),
        )))
        .unwrap();
        assert_eq!(
            value,
            json!({
                "type": "list_sessions",
                "data": {
                    "cursor": "opaque-root-2",
                    "include_remote": true,
                    "session_scope": "root"
                }
            })
        );
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
            "/workspace/project".to_string(),
            "cursor-1".to_string(),
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
    fn list_profiles_serializes() {
        let list = serde_json::to_value(ClientMsg::ListProfiles).unwrap();
        assert_eq!(list, json!({ "type": "list_profiles" }));
    }

    #[test]
    fn list_auth_providers_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::ListAuthProviders).unwrap(),
            json!({ "type": "list_auth_providers" })
        );
    }

    #[test]
    fn start_oauth_login_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::StartOAuthLogin {
                provider: "codex".into(),
            })
            .unwrap(),
            json!({
                "type": "start_oauth_login",
                "data": { "provider": "codex" }
            })
        );
    }

    #[test]
    fn complete_oauth_login_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::CompleteOAuthLogin {
                flow_id: "flow-1".into(),
                response: "code-123".into(),
            })
            .unwrap(),
            json!({
                "type": "complete_oauth_login",
                "data": { "flow_id": "flow-1", "response": "code-123" }
            })
        );
    }

    #[test]
    fn disconnect_oauth_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::DisconnectOAuth {
                provider: "openai".into(),
            })
            .unwrap(),
            json!({
                "type": "disconnect_oauth",
                "data": { "provider": "openai" }
            })
        );
    }

    #[test]
    fn set_api_token_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::SetApiToken {
                provider: "openai".into(),
                api_key: "sk-123".into(),
            })
            .unwrap(),
            json!({
                "type": "set_api_token",
                "data": { "provider": "openai", "api_key": "sk-123" }
            })
        );
    }

    #[test]
    fn clear_api_token_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::ClearApiToken {
                provider: "openai".into(),
            })
            .unwrap(),
            json!({
                "type": "clear_api_token",
                "data": { "provider": "openai" }
            })
        );
    }

    #[test]
    fn set_auth_method_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::SetAuthMethod {
                provider: "openai".into(),
                method: AuthMethod::ApiKey,
            })
            .unwrap(),
            json!({
                "type": "set_auth_method",
                "data": { "provider": "openai", "method": "api_key" }
            })
        );
    }

    #[test]
    fn set_oauth_auth_method_serializes_exact_shape() {
        assert_eq!(
            serde_json::to_value(ClientMsg::SetAuthMethod {
                provider: "openai".into(),
                method: AuthMethod::OAuth,
            })
            .unwrap(),
            json!({
                "type": "set_auth_method",
                "data": { "provider": "openai", "method": "oauth" }
            })
        );
    }

    #[test]
    fn delegate_profile_messages_serialize() {
        let agents = serde_json::to_value(ClientMsg::ListProfileAgents {
            profile_id: "quorum".into(),
        })
        .unwrap();
        assert_eq!(
            agents,
            json!({
                "type": "list_profile_agents",
                "data": { "profile_id": "quorum" }
            })
        );

        let set = serde_json::to_value(ClientMsg::SetDelegateModel {
            session_id: "parent".into(),
            agent_id: "coder".into(),
            model_id: Some("openai/gpt-5".into()),
            node_id: Some("node-1".into()),
        })
        .unwrap();
        assert_eq!(
            set,
            json!({
                "type": "set_delegate_model",
                "data": {
                    "session_id": "parent",
                    "agent_id": "coder",
                    "model_id": "openai/gpt-5",
                    "node_id": "node-1"
                }
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

#[derive(Debug, Deserialize)]
pub struct ReasoningEffortData {
    /// `None` or `"auto"` both map to the "auto" (no effort override) state.
    pub reasoning_effort: Option<String>,
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
    pub groups: Vec<SessionGroupData>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub total_count: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
pub struct SessionGroupData {
    pub cwd: Option<String>,
    #[serde(default)]
    pub sessions: Vec<SessionSummaryData>,
    /// ISO 8601 timestamp of the most recent activity in this group.
    #[serde(default)]
    pub latest_activity: Option<String>,
    #[serde(default)]
    pub total_count: Option<u64>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct SessionSummaryData {
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
    pub children: Vec<SessionSummaryData>,
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

#[derive(Debug, Default, Deserialize)]
pub struct SessionChildrenData {
    pub parent_session_id: String,
    #[serde(default)]
    pub sessions: Vec<SessionSummaryData>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub total_count: Option<u64>,
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

#[derive(Debug, Clone, Deserialize)]
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
mod session_page_data_tests {
    use super::{SessionChildrenData, SessionListData};
    use serde_json::json;

    #[test]
    fn session_pages_deserialize_recursive_wire_fields_and_defaults() {
        let list: SessionListData = serde_json::from_value(json!({
            "groups": [{
                "cwd": "/workspace/project",
                "latest_activity": "2024-02-01T00:00:00Z",
                "total_count": 4,
                "next_cursor": "group-next",
                "unknown_group_field": true,
                "sessions": [{
                    "session_id": "root",
                    "name": "Session One",
                    "title": "Root",
                    "cwd": "/workspace/project",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-02-01T00:00:00Z",
                    "parent_session_id": null,
                    "fork_origin": "manual",
                    "session_kind": "interactive",
                    "has_children": true,
                    "fork_count": 1,
                    "children_next_cursor": "child-next",
                    "children_total_count": 2,
                    "node": "remote",
                    "node_id": "node-1",
                    "attached": true,
                    "runtime_state": "running",
                    "unknown_session_field": "ignored",
                    "children": [{
                        "session_id": "child",
                        "parent_session_id": "root",
                        "fork_origin": "manual",
                        "children_next_cursor": "grandchild-next",
                        "children_total_count": 1,
                        "node_id": "node-2",
                        "children": [{ "session_id": "grandchild" }]
                    }]
                }]
            }],
            "next_cursor": "list-next",
            "total_count": 9,
            "unknown_page_field": null
        }))
        .expect("session list wire shape should deserialize");

        assert_eq!(list.next_cursor.as_deref(), Some("list-next"));
        assert_eq!(list.total_count, Some(9));
        let group = &list.groups[0];
        assert_eq!(group.cwd.as_deref(), Some("/workspace/project"));
        assert_eq!(
            group.latest_activity.as_deref(),
            Some("2024-02-01T00:00:00Z")
        );
        assert_eq!(group.total_count, Some(4));
        assert_eq!(group.next_cursor.as_deref(), Some("group-next"));
        let root = &group.sessions[0];
        assert_eq!(root.session_id, "root");
        assert_eq!(root.name.as_deref(), Some("Session One"));
        assert_eq!(root.title.as_deref(), Some("Root"));
        assert_eq!(root.cwd.as_deref(), Some("/workspace/project"));
        assert_eq!(root.created_at.as_deref(), Some("2024-01-01T00:00:00Z"));
        assert_eq!(root.updated_at.as_deref(), Some("2024-02-01T00:00:00Z"));
        assert_eq!(root.parent_session_id, None);
        assert_eq!(root.fork_origin.as_deref(), Some("manual"));
        assert_eq!(root.session_kind.as_deref(), Some("interactive"));
        assert!(root.has_children);
        assert_eq!(root.fork_count, 1);
        assert_eq!(root.children_next_cursor.as_deref(), Some("child-next"));
        assert_eq!(root.children_total_count, Some(2));
        assert_eq!(root.node.as_deref(), Some("remote"));
        assert_eq!(root.node_id.as_deref(), Some("node-1"));
        assert_eq!(root.attached, Some(true));
        assert_eq!(root.runtime_state.as_deref(), Some("running"));

        let child = &root.children[0];
        assert_eq!(child.session_id, "child");
        assert_eq!(child.parent_session_id.as_deref(), Some("root"));
        assert_eq!(child.fork_origin.as_deref(), Some("manual"));
        assert_eq!(
            child.children_next_cursor.as_deref(),
            Some("grandchild-next")
        );
        assert_eq!(child.children_total_count, Some(1));
        assert_eq!(child.node_id.as_deref(), Some("node-2"));
        assert_eq!(child.children[0].session_id, "grandchild");
        assert_eq!(child.children[0].fork_count, 0);
        assert!(child.children[0].children.is_empty());

        let children: SessionChildrenData = serde_json::from_value(json!({
            "parent_session_id": "root",
            "sessions": [{
                "session_id": "child",
                "node_id": "node-2",
                "children_next_cursor": "nested-next",
                "children_total_count": 1,
                "children": [{ "session_id": "grandchild" }]
            }],
            "next_cursor": "children-next",
            "total_count": 3
        }))
        .expect("session children wire shape should deserialize");
        assert_eq!(children.parent_session_id, "root");
        assert_eq!(children.next_cursor.as_deref(), Some("children-next"));
        assert_eq!(children.total_count, Some(3));
        assert_eq!(children.sessions[0].node_id.as_deref(), Some("node-2"));
        assert_eq!(
            children.sessions[0].children_next_cursor.as_deref(),
            Some("nested-next")
        );
        assert_eq!(children.sessions[0].children_total_count, Some(1));
        assert_eq!(children.sessions[0].children[0].session_id, "grandchild");

        let defaulted_children: SessionChildrenData = serde_json::from_value(json!({
            "parent_session_id": "root",
            "sessions": [{ "session_id": "child" }]
        }))
        .expect("session children defaults should deserialize");
        assert_eq!(defaulted_children.sessions[0].fork_count, 0);
        assert!(!defaulted_children.sessions[0].has_children);
        assert!(defaulted_children.sessions[0].children.is_empty());
        assert_eq!(defaulted_children.next_cursor, None);
        assert_eq!(defaulted_children.total_count, None);

        let empty_list: SessionListData =
            serde_json::from_value(json!({})).expect("session list defaults should deserialize");
        assert!(empty_list.groups.is_empty());
        assert_eq!(empty_list.next_cursor, None);
        assert_eq!(empty_list.total_count, None);
    }
}

#[cfg(test)]
mod session_mutation_data_tests {
    use super::{ForkResultData, RedoResultData, UndoResultData};
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

    #[test]
    fn fork_result_deserializes_present_and_missing_optional_fields() {
        let succeeded: ForkResultData = serde_json::from_value(json!({
            "success": true,
            "source_session_id": "source-1",
            "forked_session_id": "fork-1",
            "message": "forked"
        }))
        .unwrap();
        assert!(succeeded.success);
        assert_eq!(succeeded.source_session_id.as_deref(), Some("source-1"));
        assert_eq!(succeeded.forked_session_id.as_deref(), Some("fork-1"));
        assert_eq!(succeeded.message.as_deref(), Some("forked"));

        let failed: ForkResultData = serde_json::from_value(json!({ "success": false })).unwrap();
        assert!(!failed.success);
        assert_eq!(failed.source_session_id, None);
        assert_eq!(failed.forked_session_id, None);
        assert_eq!(failed.message, None);
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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshScopeInfo {
    pub kind: String,
    pub id: String,
}

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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshInviteInfo {
    pub invite_id: String,
    #[serde(default)]
    pub mesh_name: Option<String>,
    #[serde(default)]
    pub expires_at: u64,
    #[serde(default)]
    pub max_uses: u32,
    #[serde(default)]
    pub uses_remaining: u32,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub used_by: Vec<String>,
    #[serde(default)]
    pub created_at: u64,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshInviteListInfo {
    #[serde(default)]
    pub invites: Vec<MeshInviteInfo>,
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
