use serde_json::Value;

// Wire fields are retained even when the current UI projects only a subset.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct MeshStatusInfo {
    pub enabled: bool,
    pub peer_id: Option<String>,
    pub transport: Option<String>,
    pub known_peer_count: u32,
    pub has_invite_store: bool,
    pub has_mesh_state_store: bool,
    pub scopes: Vec<MeshScopeInfo>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct MeshScopeInfo {
    pub kind: String,
    pub id: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct RemoteNodeInfo {
    pub id: String,
    pub label: String,
    pub capabilities: Vec<String>,
    pub active_sessions: u32,
    pub transport: String,
    pub last_seen_at: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MeshNodesInfo {
    pub nodes: Vec<RemoteNodeInfo>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct RemoteSessionInfo {
    pub id: String,
    pub node_id: String,
    pub node_label: Option<String>,
    pub title: Option<String>,
    pub cwd: Option<String>,
    pub updated_at: Option<String>,
    pub profile_id: Option<String>,
    pub model_id: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct RemoteSessionListInfo {
    pub node_id: String,
    pub sessions: Vec<RemoteSessionInfo>,
    pub next_offset: Option<u32>,
    pub total_count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct RemoteSessionLocation {
    pub node_id: String,
    pub cwd: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RemoteSessionAttachInfo {
    pub session_id: String,
    pub node_id: String,
    pub attached: bool,
    pub config_options: Vec<Value>,
    pub snapshot: Option<Value>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct MeshInviteCreatedInfo {
    pub invite_id: String,
    pub url: String,
    pub qr_code: Option<String>,
    pub expires_at: u64,
    pub max_uses: u32,
    pub mesh_name: Option<String>,
}
