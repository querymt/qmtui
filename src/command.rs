#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionListRequest {
    Discovery,
    WorkspaceFirstPage { cwd: String },
    WorkspaceContinuation { cwd: String },
}

impl SessionListRequest {
    pub fn cwd(&self) -> Option<&str> {
        match self {
            Self::Discovery => None,
            Self::WorkspaceFirstPage { cwd } | Self::WorkspaceContinuation { cwd } => Some(cwd),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptBlock {
    Text { text: String },
    ResourceLink { name: String, uri: String },
}

#[derive(Clone, PartialEq, Eq)]
pub enum Command {
    Init,
    ListSessions {
        request: SessionListRequest,
        cursor: Option<String>,
    },
    ListRemoteNodes,
    ListRemoteSessions {
        node_id: String,
        offset: u32,
        limit: u32,
    },
    CreateRemoteSession {
        node_id: String,
        cwd: Option<String>,
    },
    AttachRemoteSession {
        node_id: String,
        session_id: String,
    },
    DismissRemoteSession {
        session_id: String,
    },
    CreateMeshInvite {
        mesh_name: Option<String>,
        ttl: Option<String>,
        max_uses: Option<u32>,
    },
    ListSessionChildren {
        parent_session_id: String,
        cursor: Option<String>,
        limit: u32,
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
        profile_id: Option<String>,
    },
    LoadSession {
        session_id: String,
        cwd: Option<String>,
    },
    Prompt {
        prompt: Vec<PromptBlock>,
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
    ElicitationResponse {
        elicitation_id: String,
        action: String,
        content: Option<serde_json::Value>,
    },
    ListAuthProviders,
    StartOAuthLogin {
        provider: String,
    },
    CompleteOAuthLogin {
        flow_id: String,
        response: String,
    },
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
}

impl Command {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Init => "Init",
            Self::ListSessions { .. } => "ListSessions",
            Self::ListRemoteNodes => "ListRemoteNodes",
            Self::ListRemoteSessions { .. } => "ListRemoteSessions",
            Self::CreateRemoteSession { .. } => "CreateRemoteSession",
            Self::AttachRemoteSession { .. } => "AttachRemoteSession",
            Self::DismissRemoteSession { .. } => "DismissRemoteSession",
            Self::CreateMeshInvite { .. } => "CreateMeshInvite",
            Self::ListSessionChildren { .. } => "ListSessionChildren",
            Self::SetReasoningEffort { .. } => "SetReasoningEffort",
            Self::ListProfiles => "ListProfiles",
            Self::ListProfileAgents { .. } => "ListProfileAgents",
            Self::SetDelegateModel { .. } => "SetDelegateModel",
            Self::NewSession { .. } => "NewSession",
            Self::LoadSession { .. } => "LoadSession",
            Self::Prompt { .. } => "Prompt",
            Self::CancelSession => "CancelSession",
            Self::ListAllModels { .. } => "ListAllModels",
            Self::SetSessionModel { .. } => "SetSessionModel",
            Self::SubscribeSession { .. } => "SubscribeSession",
            Self::DeleteSession { .. } => "DeleteSession",
            Self::ForkSession { .. } => "ForkSession",
            Self::Undo { .. } => "Undo",
            Self::Redo => "Redo",
            Self::GetFileIndex => "GetFileIndex",
            Self::SetAgentMode { .. } => "SetAgentMode",
            Self::ElicitationResponse { .. } => "ElicitationResponse",
            Self::ListAuthProviders => "ListAuthProviders",
            Self::StartOAuthLogin { .. } => "StartOAuthLogin",
            Self::CompleteOAuthLogin { .. } => "CompleteOAuthLogin",
            Self::DisconnectOAuth { .. } => "DisconnectOAuth",
            Self::SetApiToken { .. } => "SetApiToken",
            Self::ClearApiToken { .. } => "ClearApiToken",
        }
    }

    pub fn list_sessions_browse() -> Self {
        Self::list_sessions_discovery(None)
    }

    pub fn list_sessions_discovery(cursor: Option<String>) -> Self {
        Self::ListSessions {
            request: SessionListRequest::Discovery,
            cursor,
        }
    }

    pub fn list_sessions_workspace(cwd: String) -> Self {
        Self::ListSessions {
            request: SessionListRequest::WorkspaceFirstPage { cwd },
            cursor: None,
        }
    }

    pub fn list_sessions_group(cwd: String, cursor: String) -> Self {
        Self::ListSessions {
            request: SessionListRequest::WorkspaceContinuation { cwd },
            cursor: Some(cursor),
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
            limit,
        }
    }

    pub fn load_session_commands(
        session_id: String,
        cwd: Option<String>,
        agent_id: Option<String>,
    ) -> [Self; 2] {
        [
            Self::LoadSession {
                session_id: session_id.clone(),
                cwd,
            },
            Self::SubscribeSession {
                session_id,
                agent_id,
            },
        ]
    }
}

impl std::fmt::Debug for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::Command;

    #[test]
    fn command_debug_redacts_api_tokens() {
        let command = Command::SetApiToken {
            provider: "provider".into(),
            api_key: "sentinel-secret-api-key".into(),
        };

        let debug = format!("{command:?}");
        assert_eq!(debug, "SetApiToken");
        assert!(!debug.contains("sentinel-secret-api-key"));
    }

    #[test]
    fn load_session_commands_preserve_order_and_fields() {
        let commands = Command::load_session_commands(
            "session-1".into(),
            Some("/repo".into()),
            Some("agent-1".into()),
        );

        assert_eq!(
            commands,
            [
                Command::LoadSession {
                    session_id: "session-1".into(),
                    cwd: Some("/repo".into()),
                },
                Command::SubscribeSession {
                    session_id: "session-1".into(),
                    agent_id: Some("agent-1".into()),
                },
            ]
        );
    }
}
