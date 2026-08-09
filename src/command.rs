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
pub enum Command {
    ListSessions {
        request: SessionListRequest,
        cursor: Option<String>,
    },
    ListRemoteSessions {
        node_id: String,
        offset: u32,
        limit: u32,
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
    ListProfileAgents {
        profile_id: String,
    },
    SetDelegateModel {
        session_id: String,
        agent_id: String,
        model_id: Option<String>,
        node_id: Option<String>,
    },
    LoadSession {
        session_id: String,
        cwd: Option<String>,
    },
    SubscribeSession {
        session_id: String,
        agent_id: Option<String>,
    },
    GetFileIndex,
    SetAgentMode {
        mode: String,
    },
    ListAuthProviders,
}

impl Command {
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
}
