#[derive(Debug, Clone, Default)]
pub struct SessionGroup {
    pub cwd: Option<String>,
    pub sessions: Vec<SessionSummary>,
    /// ISO 8601 timestamp of the most recent activity in this group.
    pub latest_activity: Option<String>,
    pub total_count: Option<u64>,
    pub next_cursor: Option<String>,
}

// Session discovery retains backend metadata beyond the current projection.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SessionSummary {
    pub session_id: String,
    pub name: Option<String>,
    pub title: Option<String>,
    /// Working directory for this session (may differ from group cwd for remote sessions).
    pub cwd: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    /// Parent session ID if this is a forked session.
    pub parent_session_id: Option<String>,
    pub fork_origin: Option<String>,
    pub session_kind: Option<String>,
    /// Whether this session has child (forked) sessions.
    pub has_children: bool,
    /// Number of direct forked child sessions.
    pub fork_count: u64,
    pub children: Vec<SessionSummary>,
    pub children_next_cursor: Option<String>,
    pub children_total_count: Option<u64>,
    pub node: Option<String>,
    pub node_id: Option<String>,
    pub attached: Option<bool>,
    pub runtime_state: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SessionListPage {
    pub groups: Vec<SessionGroup>,
    pub next_cursor: Option<String>,
    pub total_count: Option<u64>,
}

#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub struct SessionChildrenPage {
    pub parent_session_id: String,
    pub sessions: Vec<SessionSummary>,
    pub next_cursor: Option<String>,
    pub total_count: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UndoStackSnapshot {
    pub message_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UndoResult {
    Applied {
        target_message_id: Option<String>,
        reverted_files: Vec<String>,
        message: Option<String>,
        stack: UndoStackSnapshot,
    },
    Rejected {
        target_message_id: Option<String>,
        message: Option<String>,
        stack: UndoStackSnapshot,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedoResult {
    Applied {
        message: Option<String>,
        stack: UndoStackSnapshot,
    },
    Rejected {
        message: Option<String>,
        stack: UndoStackSnapshot,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForkResult {
    Succeeded {
        source_session_id: Option<String>,
        forked_session_id: Option<String>,
        message: Option<String>,
    },
    Failed {
        source_session_id: Option<String>,
        message: Option<String>,
    },
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
