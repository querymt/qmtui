use serde::Deserialize;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoableTurn {
    pub turn_id: String,
    pub message_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForkBoundaryKind {
    Assistant,
    User,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForkTurnItem {
    pub turn_index: usize,
    pub message_id: String,
    pub boundary_kind: ForkBoundaryKind,
    pub user_preview: String,
    pub assistant_preview: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UndoFrameStatus {
    Pending,
    Confirmed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoFrame {
    pub turn_id: String,
    pub message_id: String,
    pub status: UndoFrameStatus,
    pub reverted_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UndoState {
    pub stack: Vec<UndoFrame>,
    pub frontier_message_id: Option<String>,
}
