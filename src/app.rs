use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::domain::activity::{
    ActivityState, DelegateChildState, DelegateEntry, DelegateStats, PendingDelegateToolCall,
    SessionActivity, SessionOp, SessionStatsLite,
};
use crate::domain::auth::{AuthProviderEntry, OAuthFlow, OAuthResult};
use crate::domain::chat::{ChatEntry, format_outcome_labels};
use crate::domain::elicitation::ElicitationState;
use crate::domain::model::{DelegateModelPreference, ModelEntry};
use crate::domain::profile::{AgentInfo, ProfileInfo};
use crate::domain::session::{
    ForkBoundaryKind, ForkTurnItem, SessionGroup, UndoFrame, UndoFrameStatus, UndoStackSnapshot,
    UndoState, UndoableTurn,
};
use crate::highlight::Highlighter;
use crate::markdown::CardBlock;
use crate::mesh::{MeshFocus, MeshInviteFormField};
use crate::protocol::{
    ClientMsg, EventKind, MeshInviteCreatedInfo, MeshStatusInfo, RemoteNodeInfo, RemoteSessionInfo,
};
use crate::ui::{CardCache, ElicitationUiState};

/// Cache for rendered streaming markdown to avoid re-parsing every frame.
/// Invalidated when `streaming_content` grows or is cleared.
pub struct StreamingCache {
    /// Length of `streaming_content` at the time of last render.
    rendered_len: usize,
    /// Cached rendered blocks (without the spinner).
    blocks: Vec<CardBlock>,
}

impl StreamingCache {
    pub fn new() -> Self {
        Self {
            rendered_len: 0,
            blocks: Vec::new(),
        }
    }

    /// Returns cached blocks if content length hasn't changed, otherwise None.
    pub fn get(&self, content_len: usize) -> Option<&[CardBlock]> {
        if content_len > 0 && content_len == self.rendered_len {
            Some(&self.blocks)
        } else {
            None
        }
    }

    /// Store freshly rendered blocks and the content length they correspond to.
    pub fn store(&mut self, content_len: usize, blocks: Vec<CardBlock>) {
        self.rendered_len = content_len;
        self.blocks = blocks;
    }

    /// Reset the cache (call when streaming_content is cleared).
    pub fn invalidate(&mut self) {
        self.rendered_len = 0;
        self.blocks.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Screen {
    Sessions,
    Chat,
    /// Read-only view for delegate child sessions (no input box).
    Delegate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Popup {
    None,
    CommandPalette,
    Mesh,
    MeshInvite,
    MeshInviteQr,
    ModelSelect,
    SessionSelect,
    NewSession,
    ThemeSelect,
    Help,
    Log,
    ProviderAuth,
    ForkTurnSelect,
    ProfileSelect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandPaletteAction {
    OpenMesh,
    AttachRemoteSession,
    CreateRemoteSession,
    CreateMeshInvite,
    ModelSelect,
    SessionSelect,
    DelegateSessions,
    NewSession,
    ThemeSelect,
    Help,
    Log,
    ProviderAuth,
    ForkTurnSelect,
    ProfileSelect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandPaletteCommand {
    pub title: &'static str,
    pub description: &'static str,
    pub shortcut: &'static str,
    pub action: CommandPaletteAction,
    pub chat_only: bool,
}

pub const COMMAND_PALETTE_COMMANDS: &[CommandPaletteCommand] = &[
    CommandPaletteCommand {
        title: "Open Mesh",
        description: "View mesh nodes and remote sessions",
        shortcut: "",
        action: CommandPaletteAction::OpenMesh,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Attach remote session",
        description: "Attach an existing session from a mesh node",
        shortcut: "",
        action: CommandPaletteAction::AttachRemoteSession,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Create remote session",
        description: "Start a new session on a mesh node",
        shortcut: "",
        action: CommandPaletteAction::CreateRemoteSession,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Create mesh invite",
        description: "Generate a mesh invite link and QR code",
        shortcut: "",
        action: CommandPaletteAction::CreateMeshInvite,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Model selector",
        description: "Choose the model for this session or delegates",
        shortcut: "C-x m",
        action: CommandPaletteAction::ModelSelect,
        chat_only: true,
    },
    CommandPaletteCommand {
        title: "Session switcher",
        description: "Browse and load sessions",
        shortcut: "C-x l",
        action: CommandPaletteAction::SessionSelect,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Delegate sessions",
        description: "Browse delegate child sessions",
        shortcut: "",
        action: CommandPaletteAction::DelegateSessions,
        chat_only: true,
    },
    CommandPaletteCommand {
        title: "New session",
        description: "Start a new session in a directory",
        shortcut: "C-x n",
        action: CommandPaletteAction::NewSession,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Theme picker",
        description: "Change the UI theme",
        shortcut: "C-x t",
        action: CommandPaletteAction::ThemeSelect,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Help",
        description: "Show keyboard shortcuts and commands",
        shortcut: "C-x ?",
        action: CommandPaletteAction::Help,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Logs",
        description: "Show in-memory logs",
        shortcut: "Ctrl+l",
        action: CommandPaletteAction::Log,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Provider auth",
        description: "Manage provider authentication",
        shortcut: "C-x a",
        action: CommandPaletteAction::ProviderAuth,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Fork selector",
        description: "Choose a turn to fork from",
        shortcut: "C-x f",
        action: CommandPaletteAction::ForkTurnSelect,
        chat_only: true,
    },
    CommandPaletteCommand {
        title: "Profile selector",
        description: "Choose the active profile for new sessions",
        shortcut: "C-x p",
        action: CommandPaletteAction::ProfileSelect,
        chat_only: false,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Trace => Self::Debug,
            Self::Debug => Self::Info,
            Self::Info => Self::Warn,
            Self::Warn => Self::Error,
            Self::Error => Self::Trace,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppLogEntry {
    pub elapsed: Duration,
    pub level: LogLevel,
    pub target: &'static str,
    pub message: String,
}

// ── Delegation tracking ───────────────────────────────────────────────────────

/// Update per-delegation stats from a single event arriving on a child session.
pub(crate) fn accumulate_delegate_stats(stats: &mut DelegateStats, kind: &EventKind) {
    match kind {
        EventKind::ToolCallStart { .. } => {
            stats.tool_calls = stats.tool_calls.saturating_add(1);
        }
        EventKind::AssistantMessageStored { .. } => {
            stats.messages = stats.messages.saturating_add(1);
        }
        EventKind::LlmRequestEnd {
            cost_usd,
            context_tokens,
            ..
        } => {
            if let Some(c) = cost_usd {
                stats.cost_usd += c;
            }
            if let Some(ctx) = context_tokens {
                stats.context_tokens = *ctx;
            }
        }
        EventKind::ProviderChanged {
            context_limit: Some(limit),
            ..
        } => {
            stats.context_limit = *limit;
        }
        EventKind::LlmRequestStart { .. }
        | EventKind::SnapshotStart { .. }
        | EventKind::SnapshotEnd { .. }
        | EventKind::ProgressRecorded { .. }
        | EventKind::ArtifactRecorded { .. }
        | EventKind::SessionQueued { .. }
        | EventKind::SessionConfigured { .. }
        | EventKind::ToolsAvailable { .. }
        | EventKind::SessionCreated
        | EventKind::Unknown => {}
        _ => {}
    }
}

pub(crate) fn update_delegate_child_state(state: &mut DelegateChildState, kind: &EventKind) {
    match kind {
        EventKind::ElicitationRequested {
            elicitation_id,
            message,
            source,
            requested_schema,
            ..
        } => {
            *state = DelegateChildState::PendingElicitation {
                elicitation_id: elicitation_id.clone(),
                message: message.clone(),
                requested_schema: requested_schema.clone(),
                source: source.clone(),
            };
        }
        EventKind::ToolCallEnd { tool_name, .. } if tool_name == "question" => {
            *state = DelegateChildState::QuestionToolFinished;
        }
        EventKind::AssistantMessageStored { .. } => {
            *state = DelegateChildState::AssistantMessage;
        }
        EventKind::UserMessageStored { .. } => {
            *state = DelegateChildState::UserMessage;
        }
        EventKind::ToolCallStart { .. }
        | EventKind::AssistantContentDelta { .. }
        | EventKind::AssistantThinkingDelta { .. }
        | EventKind::PromptReceived { .. }
        | EventKind::LlmRequestStart { .. }
        | EventKind::LlmRequestEnd { .. }
        | EventKind::CompactionStart { .. }
        | EventKind::CompactionEnd { .. }
        | EventKind::SnapshotStart { .. }
        | EventKind::SnapshotEnd { .. }
        | EventKind::ProgressRecorded { .. }
        | EventKind::ArtifactRecorded { .. }
        | EventKind::SessionQueued { .. }
        | EventKind::ProviderChanged { .. }
        | EventKind::Error { .. }
        | EventKind::Cancelled => {
            *state = DelegateChildState::OtherProgress;
        }
        EventKind::TurnStarted
        | EventKind::SessionModeChanged { .. }
        | EventKind::SessionConfigured { .. }
        | EventKind::ToolsAvailable { .. }
        | EventKind::SessionCreated
        | EventKind::DelegationRequested { .. }
        | EventKind::DelegationCompleted { .. }
        | EventKind::DelegationFailed { .. }
        | EventKind::DelegationCancelled { .. }
        | EventKind::SessionForked { .. }
        | EventKind::Unknown => {}
        EventKind::ToolCallEnd { .. } => {
            *state = DelegateChildState::OtherProgress;
        }
    }
}

pub(crate) fn backfill_elicitation_outcomes(messages: &mut [ChatEntry], result_str: &str) {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(result_str) else {
        return;
    };
    let Some(answers) = val.get("answers").and_then(|a| a.as_array()) else {
        return;
    };

    let mut answer_iter = answers.iter();
    for entry in messages.iter_mut() {
        let ChatEntry::Elicitation { outcome, .. } = entry else {
            continue;
        };
        if outcome.as_deref() != Some("responded") {
            continue;
        }
        let Some(answer_entry) = answer_iter.next() else {
            break;
        };
        let labels = answer_entry
            .get("answers")
            .and_then(|a| a.as_array())
            .map(|answers| answers.iter().filter_map(|answer| answer.as_str()))
            .into_iter()
            .flatten();
        *outcome = Some(format_outcome_labels(labels));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileIndexEntryLite {
    pub path: String,
    pub is_dir: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MentionState {
    pub trigger_start: usize,
    pub query: String,
    pub selected_index: usize,
    pub results: Vec<FileIndexEntryLite>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCompletionState {
    /// The text typed after the leading `/` (e.g. `"mo"` while typing `/mo`).
    pub query: String,
    pub selected_index: usize,
    pub results: Vec<&'static crate::slash::SlashCommandDef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathCompletionState {
    pub query: String,
    pub selected_index: usize,
    pub results: Vec<FileIndexEntryLite>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnState {
    Connecting,
    Connected,
    Disconnected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionEvent {
    Connecting { attempt: u32, delay_ms: u64 },
    Connected,
    Disconnected { reason: String },
}

const CANCEL_CONFIRM_TIMEOUT: Duration = Duration::from_millis(1000);

/// A single visible row on the start-page session list.
///
/// Built by [`App::visible_start_items`] each render frame, respecting the
/// current filter and per-group collapse state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartPageItem {
    /// A group header row (cwd label + count + collapsed state).
    GroupHeader {
        /// The cwd key used to look up collapse state (mirrors `SessionGroup::cwd`).
        cwd: Option<String>,
        /// Loaded sessions in this group (unfiltered).
        session_count: usize,
        /// Total sessions in this group from the backend, when known.
        session_total: Option<u64>,
        /// Whether the group is currently collapsed.
        collapsed: bool,
    },
    /// A session row inside an expanded group.
    Session {
        /// Index into `App::session_groups`.
        group_idx: usize,
        /// Index path from root session to this visible session.
        path: Vec<usize>,
        /// Visual nesting depth: 0 for roots, 1+ for fork children.
        depth: usize,
    },
    /// A "... show all" row shown when a group has more sessions than are
    /// currently visible on the start page.
    ShowMore {
        /// Index into `App::session_groups`.
        group_idx: usize,
        /// Number of loaded sessions hidden beyond the first `MAX_RECENT_SESSIONS`.
        remaining: usize,
        /// Whether the backend has another page for this group.
        has_more: bool,
    },
}

/// Maximum number of recent sessions shown per group on the start page.
pub const MAX_RECENT_SESSIONS: usize = 3;
/// Maximum discovery-preview sessions retained before the workspace page arrives.
pub const POPUP_SESSION_PAGE_TARGET: usize = 10;
pub const SESSION_CHILD_PAGE_LIMIT: u32 = 10;

/// Maximum number of workspace groups shown on the start page.
/// Groups beyond this cap are hidden from the start page but remain accessible
/// through the session popup.
pub const MAX_VISIBLE_GROUPS: usize = 3;

/// A single visible row in the sessions popup.
///
/// Built by [`App::visible_popup_items`]. Unlike [`StartPageItem`] there is no
/// `ShowMore` variant — the popup always shows all sessions and all groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PopupItem {
    /// A group header row (cwd label + count + collapsed state).
    GroupHeader {
        /// The cwd key used to look up collapse state (mirrors `SessionGroup::cwd`).
        cwd: Option<String>,
        /// Loaded sessions in this group (unfiltered).
        session_count: usize,
        /// Total sessions in this group from the backend, when known.
        session_total: Option<u64>,
        /// Whether the group is currently collapsed in the popup.
        collapsed: bool,
    },
    /// A session row inside an expanded group.
    Session {
        /// Index into `App::session_groups`.
        group_idx: usize,
        /// Index path from root session to this visible session.
        path: Vec<usize>,
        /// Visual nesting depth: 0 for roots, 1+ for fork children.
        depth: usize,
    },
    /// A "... load more..." row that fetches the next page for one group or parent fork.
    LoadMore {
        /// Index into `App::session_groups`.
        group_idx: usize,
        /// Parent session path when loading more fork children; empty for a cwd group page.
        parent_path: Vec<usize>,
    },
}

pub fn session_group_count_text(session_count: usize, session_total: Option<u64>) -> String {
    session_total
        .map(|total| format!("{session_count}/{total}"))
        .unwrap_or_else(|| session_count.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelPopupItem {
    ProviderHeader {
        provider: String,
        model_count: usize,
        /// When set, header shows `@ {node}` right-aligned (remote mesh group).
        node_suffix: Option<String>,
    },
    Model {
        model_idx: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthUiNotice {
    pub provider: Option<String>,
    pub success: bool,
    pub message: String,
}

/// Which sub-panel is active in the provider auth popup.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AuthPanel {
    /// Browsing the provider list.
    #[default]
    List,
    /// Editing an API key for the selected provider.
    ApiKeyInput,
    /// Active OAuth flow — showing URL and callback/device-poll input.
    OAuthFlow,
}

pub struct App {
    pub screen: Screen,
    pub popup: Popup,
    pub chord: bool, // true after ctrl+x pressed, waiting for second key
    pub command_palette_cursor: usize,
    pub command_palette_filter: String,

    // sessions
    /// Session groups as received from the server (preserve group structure for start page).
    pub session_groups: Vec<SessionGroup>,
    pub session_cursor: usize,
    pub session_filter: String,
    /// Active tab in the session popup: 0 = sessions, 1 = delegates.
    pub session_popup_tab: usize,
    /// Groups whose header has been collapsed by the user on the start page.
    pub collapsed_groups: HashSet<String>,
    /// Groups whose header has been collapsed by the user in the session popup.
    /// Separate from `collapsed_groups` so start-page and popup states are independent.
    pub popup_collapsed_groups: HashSet<String>,
    /// Whether global ACP session discovery has another request in flight.
    pub session_discovery_in_progress: bool,
    /// Opaque global cursors already queued during the current discovery pass.
    pub session_discovery_cursors: HashSet<String>,
    /// CWDs with an outstanding first-page or continuation request.
    pub pending_session_group_loads: HashSet<Option<String>>,
    /// CWDs whose authoritative first workspace page has been loaded.
    pub hydrated_session_groups: HashSet<String>,
    /// Root session IDs expanded to show fork children.
    pub expanded_session_children: HashSet<String>,
    /// Parent session IDs with an outstanding child-page request.
    pub pending_session_child_loads: HashSet<String>,
    /// Scroll offset for the start-page session list (in visible rows).
    pub start_page_scroll: usize,
    /// Last rendered visible row count for the sessions tab in the popup.
    pub session_popup_visible_rows: usize,
    /// Last rendered visible row count for the delegates tab in the popup.
    pub delegate_popup_visible_rows: usize,

    // active session
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub agent_mode: String,
    pub mode_before_review: Option<String>,
    pub launch_cwd: Option<String>,
    pub new_session_path: String,
    pub new_session_cursor: usize,
    pub new_session_completion: Option<PathCompletionState>,
    pub session_activity: HashMap<String, SessionActivity>,
    /// Remote sessions the user attached/dismissed before the refreshed list arrives.
    pub remote_session_nodes: HashMap<String, String>,

    // chat
    pub messages: Vec<ChatEntry>,
    /// Monotonic identifier source for locally rendered prompts awaiting ACP echo.
    pub pending_prompt_seq: u64,
    pub input: String,
    pub input_cursor: usize,
    pub input_scroll: u16,
    pub input_line_width: usize,
    pub input_preferred_col: Option<usize>,
    pub fork_filter: String,
    pub fork_cursor: usize,
    pub pending_fork_message_id: Option<String>,
    pub scroll_offset: u16,
    /// Total content height (in rows) from the last render frame.
    /// Used to compensate `scroll_offset` when content grows while the user
    /// is scrolled up, so the viewport stays at the same absolute position.
    pub prev_total_height: u16,
    pub activity: ActivityState,
    pub streaming_content: String,
    pub streaming_content_message_id: Option<String>,
    pub streaming_cache: StreamingCache,
    pub streaming_thinking: String,
    pub streaming_thinking_message_id: Option<String>,
    pub streaming_thinking_cache: StreamingCache,
    pub file_index: Vec<FileIndexEntryLite>,
    pub file_index_generated_at: Option<u64>,
    pub file_index_loading: bool,
    pub file_index_error: Option<String>,
    pub mention_state: Option<MentionState>,
    pub slash_state: Option<SlashCompletionState>,
    pub last_compaction_token_estimate: Option<u32>,
    /// Active elicitation request waiting for user response.
    pub elicitation: Option<ElicitationState>,
    /// Editor, cursor, and layout state for the active elicitation panel.
    pub elicitation_ui: Option<ElicitationUiState>,

    // thinking display
    pub show_thinking: bool,

    // reasoning effort
    /// Current reasoning-effort level. `None` = "auto" (server default).
    /// Matches `reasoningEffort: string | null` in the web UI.
    pub reasoning_effort: Option<String>,
    // profile info
    pub profiles: Vec<ProfileInfo>,
    pub active_profile_id: Option<String>,
    pub session_profiles: HashMap<String, String>,
    pub profile_cursor: usize,
    pub profile_filter: String,

    // model info
    pub current_model: Option<String>,
    pub current_provider: Option<String>,
    /// Mesh node for the active catalog model (`None` = local provider host).
    pub current_model_node_id: Option<String>,
    pub models: Vec<ModelEntry>,
    pub model_cursor: usize,
    pub model_filter: String,

    // delegate agent model preferences
    /// Agents for `agents_profile_id`. Index zero is the primary session agent.
    pub agents: Vec<AgentInfo>,
    pub agents_profile_id: Option<String>,
    /// Currently selected tab index: zero is the session, remaining tabs are delegates.
    pub model_popup_agent_tab: usize,
    /// Explicit model preferences scoped by profile ID and delegate agent ID.
    pub delegate_model_preferences: HashMap<String, HashMap<String, DelegateModelPreference>>,

    // theme selector
    pub theme_cursor: usize,
    pub theme_filter: String,

    // help popup
    pub help_scroll: usize,

    // in-memory logs popup
    pub started_at: Instant,
    pub logs: Vec<AppLogEntry>,
    pub log_cursor: usize,
    pub log_filter: String,
    pub log_level_filter: LogLevel,

    // Undo/redo state mirrors the web UI semantics: a server-authoritative stack
    // of reverted turns plus a frontier that marks the current branch point.
    pub undo_state: Option<UndoState>,
    pub undoable_turns: Vec<UndoableTurn>,
    /// Tracks the latest prompt text so the follow-up `user_message_stored`
    /// event can behave like a backfill instead of a duplicate row.
    pub recent_prompt_text: Option<String>,

    // session stats
    pub cumulative_cost: Option<f64>,
    pub context_limit: u64,
    pub session_stats: SessionStatsLite,
    pub pending_cancel_confirm_until: Option<Instant>,

    // status line
    pub status: String,

    // mesh / remote (from ACP extensions)
    /// Count of mesh nodes from the last `querymt/mesh/nodes` fetch.
    pub mesh_node_count: Option<u32>,
    pub mesh_status: Option<MeshStatusInfo>,
    pub mesh_nodes: Vec<RemoteNodeInfo>,
    pub remote_sessions_by_node: HashMap<String, Vec<RemoteSessionInfo>>,
    pub mesh_node_cursor: usize,
    pub remote_session_cursor: usize,
    pub mesh_focus: MeshFocus,
    pub mesh_error: Option<String>,
    pub mesh_error_until: Option<Instant>,
    pub mesh_invite: Option<MeshInviteCreatedInfo>,
    pub mesh_invite_name: String,
    pub mesh_invite_ttl: String,
    pub mesh_invite_max_uses: String,
    pub mesh_invite_form_field: MeshInviteFormField,
    pub mesh_clipboard_fallback: Option<String>,

    // connection
    pub conn: ConnState,
    pub reconnect_attempt: u32,
    pub reconnect_delay_ms: Option<u64>,

    // server lifecycle (managed by server_manager::supervisor)
    pub server_state: crate::server_manager::ServerState,

    // syntax highlighting
    pub hl: Highlighter,

    // card cache for incremental rendering
    pub(crate) card_cache: CardCache,

    // auth popup state
    pub auth_providers: Vec<AuthProviderEntry>,
    pub auth_cursor: usize,
    pub auth_filter: String,
    pub auth_selected: Option<usize>,
    pub auth_panel: AuthPanel,
    pub auth_api_key_input: String,
    pub auth_api_key_cursor: usize,
    pub auth_api_key_masked: bool,
    pub auth_oauth_flow: Option<OAuthFlow>,
    pub auth_oauth_response: String,
    pub auth_oauth_response_cursor: usize,
    pub auth_last_result: Option<OAuthResult>,
    pub auth_ui_notice: Option<AuthUiNotice>,
    /// When clipboard copy fails, store the URL here for a fallback display popup.
    pub auth_clipboard_fallback: Option<String>,

    // delegate session listing (built from event stream)
    pub delegate_entries: Vec<DelegateEntry>,
    pub delegate_cursor: usize,
    pub delegate_filter: String,
    /// Parent session ID (set when viewing a delegate child session).
    pub parent_session_id: Option<String>,
    /// Staging field: set by delegate popup before LoadSession, consumed by session_loaded.
    pub pending_parent_session_id: Option<String>,
    /// Set after DelegationCompleted/DelegationFailed; consumed by the next
    /// UserMessageStored to suppress the noisy batch-result message.
    pub suppress_delegation_result: bool,
    /// Commands queued by event handlers (e.g. SubscribeSession for child sessions).
    /// Drained by native ACP after each event/replay batch.
    pub pending_commands: Vec<ClientMsg>,
    /// Child-session state observed before a delegation entry can be linked.
    pub pending_delegate_child_states: HashMap<String, DelegateChildState>,
    pub pending_delegate_child_stats: HashMap<String, DelegateStats>,
    pub delegate_child_message_ids: HashMap<String, HashSet<String>>,
    /// Latest lifecycle timestamp and bounded terminal metadata by delegation ID.
    pub delegation_update_times: HashMap<String, i64>,
    pub delegation_result_summaries: HashMap<String, String>,
    pub delegation_errors: HashMap<String, String>,
    /// Parent delegate ToolCallStart records awaiting DelegationRequested linkage.
    pub pending_delegate_tool_calls: Vec<PendingDelegateToolCall>,
    /// While a reverted frontier turn is being suppressed, ignore any
    /// follow-up assistant/tool/cancelled events until a new prompt arrives.
    pub suppress_turn_output: bool,

    pub tick: u64,
    pub should_quit: bool,
}

/// Validate and normalize a reasoning-effort string.
///
/// * Returns `Some(Some(normalized))` for valid explicit levels:
///   `"low"`, `"medium"` (also accepts alias `"med"`), `"high"`, `"max"`.
/// * Returns `Some(None)` for `"auto"`, empty string, or `None`.
/// * Returns `None` for any invalid/unrecognized level.
pub fn validate_reasoning_effort(s: Option<&str>) -> Option<Option<String>> {
    match s {
        None | Some("auto") | Some("") => Some(None),
        Some("low") => Some(Some("low".to_string())),
        Some("medium") | Some("med") => Some(Some("medium".to_string())),
        Some("high") => Some(Some("high".to_string())),
        Some("max") => Some(Some("max".to_string())),
        Some(_) => None,
    }
}

fn move_wrapping_cursor(cursor: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        0
    } else {
        (cursor as isize + delta).rem_euclid(len as isize) as usize
    }
}

impl App {
    pub fn begin_session_discovery(&mut self) -> Option<ClientMsg> {
        if self.session_discovery_in_progress || !self.pending_session_group_loads.is_empty() {
            return None;
        }
        self.session_groups.clear();
        self.session_discovery_cursors.clear();
        self.pending_session_group_loads.clear();
        self.hydrated_session_groups.clear();
        self.session_discovery_in_progress = true;
        Some(ClientMsg::list_sessions_browse())
    }

    pub fn session_group_page_request(&mut self, group_idx: usize) -> Option<ClientMsg> {
        let group = self.session_groups.get(group_idx)?;
        let cursor = group.next_cursor.clone()?;
        let cwd = group.cwd.clone()?;
        if !self.pending_session_group_loads.insert(Some(cwd.clone())) {
            return None;
        }
        Some(ClientMsg::list_sessions_group(cwd, cursor))
    }

    pub fn session_child_page_request(
        &mut self,
        group_idx: usize,
        parent_path: &[usize],
    ) -> Option<ClientMsg> {
        let parent = self.session_by_path(group_idx, parent_path)?;
        let parent_session_id = parent.session_id.clone();
        let cursor = parent.children_next_cursor.clone();
        self.pending_session_child_loads
            .insert(parent_session_id.clone());
        Some(ClientMsg::list_session_children(
            parent_session_id,
            cursor,
            SESSION_CHILD_PAGE_LIMIT,
        ))
    }

    pub fn new() -> Self {
        Self {
            screen: Screen::Sessions,
            popup: Popup::None,
            chord: false,
            command_palette_cursor: 0,
            command_palette_filter: String::new(),
            session_groups: Vec::new(),
            session_cursor: 0,
            session_filter: String::new(),
            session_popup_tab: 0,
            collapsed_groups: HashSet::new(),
            popup_collapsed_groups: HashSet::new(),
            session_discovery_in_progress: false,
            session_discovery_cursors: HashSet::new(),
            pending_session_group_loads: HashSet::new(),
            hydrated_session_groups: HashSet::new(),
            expanded_session_children: HashSet::new(),
            pending_session_child_loads: HashSet::new(),
            start_page_scroll: 0,
            session_popup_visible_rows: 0,
            delegate_popup_visible_rows: 0,
            session_id: None,
            agent_id: None,
            agent_mode: "build".into(),
            mode_before_review: None,
            launch_cwd: None,
            new_session_path: String::new(),
            new_session_cursor: 0,
            new_session_completion: None,
            session_activity: HashMap::new(),
            remote_session_nodes: HashMap::new(),
            messages: Vec::new(),
            pending_prompt_seq: 0,
            input: String::new(),
            input_cursor: 0,
            input_scroll: 0,
            input_line_width: 1,
            input_preferred_col: None,
            fork_filter: String::new(),
            fork_cursor: 0,
            pending_fork_message_id: None,
            scroll_offset: 0,
            prev_total_height: 0,
            activity: ActivityState::Idle,
            streaming_content: String::new(),
            streaming_content_message_id: None,
            streaming_cache: StreamingCache::new(),
            streaming_thinking: String::new(),
            streaming_thinking_message_id: None,
            streaming_thinking_cache: StreamingCache::new(),
            file_index: Vec::new(),
            file_index_generated_at: None,
            file_index_loading: false,
            file_index_error: None,
            mention_state: None,
            slash_state: None,
            last_compaction_token_estimate: None,
            elicitation: None,
            elicitation_ui: None,
            show_thinking: true,
            reasoning_effort: None,
            profiles: Vec::new(),
            active_profile_id: None,
            session_profiles: HashMap::new(),
            profile_cursor: 0,
            profile_filter: String::new(),
            current_model: None,
            current_provider: None,
            current_model_node_id: None,
            models: Vec::new(),
            model_cursor: 0,
            model_filter: String::new(),
            agents: Vec::new(),
            agents_profile_id: None,
            model_popup_agent_tab: 0,
            delegate_model_preferences: HashMap::new(),
            theme_cursor: 0,
            theme_filter: String::new(),
            help_scroll: 0,
            started_at: Instant::now(),
            logs: Vec::new(),
            log_cursor: 0,
            log_filter: String::new(),
            log_level_filter: LogLevel::Info,
            undo_state: None,
            undoable_turns: Vec::new(),
            recent_prompt_text: None,
            cumulative_cost: None,
            context_limit: 0,
            session_stats: SessionStatsLite::default(),
            pending_cancel_confirm_until: None,
            mesh_node_count: None,
            mesh_status: None,
            mesh_nodes: Vec::new(),
            remote_sessions_by_node: HashMap::new(),
            mesh_node_cursor: 0,
            remote_session_cursor: 0,
            mesh_focus: MeshFocus::Nodes,
            mesh_error: None,
            mesh_error_until: None,
            mesh_invite: None,
            mesh_invite_name: String::new(),
            mesh_invite_ttl: "24h".into(),
            mesh_invite_max_uses: "1".into(),
            mesh_invite_form_field: MeshInviteFormField::MeshName,
            mesh_clipboard_fallback: None,
            conn: ConnState::Connecting,
            reconnect_attempt: 0,
            reconnect_delay_ms: None,
            server_state: crate::server_manager::ServerState::default(),
            hl: Highlighter::new(),
            card_cache: CardCache::new(),
            auth_providers: Vec::new(),
            auth_cursor: 0,
            auth_filter: String::new(),
            auth_selected: None,
            auth_panel: AuthPanel::default(),
            auth_api_key_input: String::new(),
            auth_api_key_cursor: 0,
            auth_api_key_masked: true,
            auth_oauth_flow: None,
            auth_oauth_response: String::new(),
            auth_oauth_response_cursor: 0,
            auth_last_result: None,
            auth_ui_notice: None,
            auth_clipboard_fallback: None,
            delegate_entries: Vec::new(),
            delegate_cursor: 0,
            delegate_filter: String::new(),
            parent_session_id: None,
            pending_parent_session_id: None,
            suppress_delegation_result: false,
            pending_commands: Vec::new(),
            pending_delegate_child_states: HashMap::new(),
            pending_delegate_child_stats: HashMap::new(),
            delegate_child_message_ids: HashMap::new(),
            delegation_update_times: HashMap::new(),
            delegation_result_summaries: HashMap::new(),
            delegation_errors: HashMap::new(),
            pending_delegate_tool_calls: Vec::new(),
            suppress_turn_output: false,
            status: "connecting...".into(),
            tick: 0,
            should_quit: false,
        }
    }

    pub fn open_command_palette(&mut self) {
        self.popup = Popup::CommandPalette;
        self.command_palette_filter.clear();
        self.command_palette_cursor = 0;
    }

    pub fn filtered_command_palette_commands(&self) -> Vec<&'static CommandPaletteCommand> {
        let q = self.command_palette_filter.trim().to_lowercase();
        COMMAND_PALETTE_COMMANDS
            .iter()
            .filter(|command| !command.chat_only || matches!(self.screen, Screen::Chat))
            .filter(|command| {
                q.is_empty()
                    || command.title.to_lowercase().contains(&q)
                    || command.description.to_lowercase().contains(&q)
                    || command.shortcut.to_lowercase().contains(&q)
            })
            .collect()
    }

    pub fn move_command_palette_cursor(&mut self, delta: isize) {
        self.command_palette_cursor = move_wrapping_cursor(
            self.command_palette_cursor,
            self.filtered_command_palette_commands().len(),
            delta,
        );
    }

    pub fn selected_command_palette_action(&self) -> Option<CommandPaletteAction> {
        self.filtered_command_palette_commands()
            .get(self.command_palette_cursor)
            .map(|command| command.action)
    }

    pub fn command_palette_filter_insert(&mut self, c: char) {
        self.command_palette_filter.push(c);
        self.command_palette_cursor = 0;
    }

    pub fn command_palette_filter_backspace(&mut self) {
        self.command_palette_filter.pop();
        self.command_palette_cursor = 0;
    }

    /// Invalidate both streaming caches and clear the thinking buffer.
    ///
    /// Call this when a streaming turn ends (assistant message finalized,
    /// new turn starts, session reloaded, etc.) so stale markdown renders
    /// are discarded.
    pub fn invalidate_streaming_caches(&mut self) {
        self.streaming_cache.invalidate();
        self.streaming_thinking.clear();
        self.streaming_thinking_message_id = None;
        self.streaming_thinking_cache.invalidate();
    }

    pub fn invalidate_delegate_render_cache(&mut self) {
        self.card_cache.invalidate();
    }

    /// Short display label for the current reasoning effort level.
    /// Matches the five values used in the web UI: auto / low / medium / high / max.
    pub fn reasoning_effort_label(&self) -> &str {
        self.reasoning_effort.as_deref().unwrap_or("auto")
    }

    /// Valid reasoning effort levels (excluding "auto" which maps to `None`).
    pub const EFFORT_LEVELS: &[&str] = &["low", "medium", "high", "max"];

    /// Cycle through `[auto, low, medium, high, max]` (wraps around).
    /// Updates `self.reasoning_effort` optimistically, saves the new value as
    /// the preference for the current `(mode, provider, model)` context, and
    /// returns the [`ClientMsg`] to forward to the server.
    ///
    /// Returns `None` if the current value is not a recognized level; in that
    /// case the state is left unchanged and no message is emitted (the caller
    /// should surface a warning to the user instead of silently coercing the
    /// unknown value to `low`).
    pub fn cycle_reasoning_effort(&mut self) -> Option<ClientMsg> {
        const LEVELS: &[Option<&str>] =
            &[None, Some("low"), Some("medium"), Some("high"), Some("max")];
        let current = self.reasoning_effort.as_deref();
        let Some(idx) = LEVELS.iter().position(|l| l.as_deref() == current) else {
            // Unknown current value: leave state unchanged and let the caller
            // surface a warning to the user instead of silently coercing to low.
            return None;
        };
        let next = LEVELS[(idx + 1) % LEVELS.len()];
        Some(
            self.set_reasoning_effort(next)
                .expect("cycle always produces a valid level"),
        )
    }

    /// Set the reasoning effort to a specific level.
    /// `None` or `Some("auto")` both map to the "auto" (no override) state.
    /// Updates `self.reasoning_effort` and returns the [`ClientMsg`] to forward to the server.
    /// Returns `None` if the level is invalid (state is unchanged).
    pub fn set_reasoning_effort(&mut self, level: Option<&str>) -> Option<ClientMsg> {
        match validate_reasoning_effort(level) {
            Some(normalized) => {
                self.reasoning_effort = normalized;
                let effort_str = self
                    .reasoning_effort
                    .as_deref()
                    .unwrap_or("auto")
                    .to_string();
                Some(ClientMsg::SetReasoningEffort {
                    reasoning_effort: effort_str,
                })
            }
            None => None,
        }
    }

    /// Filtered auth providers matching the current `auth_filter`.
    pub fn filtered_auth_providers(&self) -> Vec<(usize, &AuthProviderEntry)> {
        if self.auth_filter.is_empty() {
            self.auth_providers.iter().enumerate().collect()
        } else {
            let q = self.auth_filter.to_lowercase();
            self.auth_providers
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    p.display_name.to_lowercase().contains(&q)
                        || p.provider.to_lowercase().contains(&q)
                })
                .collect()
        }
    }

    /// Reset auth popup state for a fresh open.
    pub fn open_auth_popup(&mut self) {
        self.popup = Popup::ProviderAuth;
        self.auth_cursor = 0;
        self.auth_filter.clear();
        self.auth_selected = None;
        self.auth_panel = AuthPanel::List;
        self.auth_api_key_input.clear();
        self.auth_api_key_cursor = 0;
        self.auth_api_key_masked = true;
        self.auth_oauth_flow = None;
        self.auth_oauth_response.clear();
        self.auth_oauth_response_cursor = 0;
        self.auth_last_result = None;
        self.auth_ui_notice = None;
        self.auth_clipboard_fallback = None;
    }

    /// Reset auth detail panel state (when switching providers or going back).
    pub fn auth_close_detail(&mut self) {
        self.auth_selected = None;
        self.auth_panel = AuthPanel::List;
        self.auth_api_key_input.clear();
        self.auth_api_key_cursor = 0;
        self.auth_oauth_flow = None;
        self.auth_oauth_response.clear();
        self.auth_oauth_response_cursor = 0;
        self.auth_last_result = None;
        self.auth_ui_notice = None;
        self.auth_clipboard_fallback = None;
    }

    pub fn auth_feedback_for_provider(&self, provider: &str) -> Option<(bool, &str)> {
        if let Some(notice) = self.auth_ui_notice.as_ref().filter(|notice| {
            notice
                .provider
                .as_deref()
                .is_none_or(|notice_provider| notice_provider == provider)
        }) {
            return Some((notice.success, notice.message.as_str()));
        }

        self.auth_last_result
            .as_ref()
            .filter(|result| result.provider == provider)
            .map(|result| (result.is_success(), result.message.as_str()))
    }

    pub fn profile_by_id(&self, profile_id: &str) -> Option<&ProfileInfo> {
        self.profiles
            .iter()
            .find(|profile| profile.id == profile_id)
    }

    pub fn active_profile(&self) -> Option<&ProfileInfo> {
        self.active_profile_id
            .as_deref()
            .and_then(|profile_id| self.profile_by_id(profile_id))
    }

    pub fn current_session_profile_id(&self) -> Option<&str> {
        self.session_id
            .as_deref()
            .and_then(|session_id| self.session_profiles.get(session_id).map(String::as_str))
    }

    pub fn current_session_profile(&self) -> Option<&ProfileInfo> {
        self.current_session_profile_id()
            .and_then(|profile_id| self.profile_by_id(profile_id))
    }

    pub fn profile_display_name(&self, profile_id: &str) -> String {
        self.profile_by_id(profile_id)
            .map(|profile| profile.name.clone())
            .unwrap_or_else(|| profile_id.to_string())
    }

    fn profile_label_or(
        &self,
        profile_id: Option<&str>,
        fallback: impl FnOnce() -> String,
    ) -> String {
        profile_id
            .map(|profile_id| self.profile_display_name(profile_id))
            .unwrap_or_else(fallback)
    }

    pub fn active_profile_label(&self) -> String {
        self.profile_label_or(self.active_profile_id.as_deref(), || "default".to_string())
    }

    pub fn current_profile_label(&self) -> String {
        self.profile_label_or(self.current_session_profile_id(), || {
            if self.current_session_is_remote() {
                "remote".to_string()
            } else {
                self.active_profile_label()
            }
        })
    }

    pub fn filtered_profiles(&self) -> Vec<&ProfileInfo> {
        if self.profile_filter.is_empty() {
            self.profiles.iter().collect()
        } else {
            let matcher = SkimMatcherV2::default();
            let mut scored: Vec<(i64, &ProfileInfo)> = self
                .profiles
                .iter()
                .filter_map(|profile| {
                    let score = [
                        matcher.fuzzy_match(&profile.name, &self.profile_filter),
                        matcher.fuzzy_match(&profile.id, &self.profile_filter),
                        profile.description.as_deref().and_then(|description| {
                            matcher.fuzzy_match(description, &self.profile_filter)
                        }),
                    ]
                    .into_iter()
                    .flatten()
                    .max();
                    score.map(|s| (s, profile))
                })
                .collect();
            scored.sort_by_key(|item| std::cmp::Reverse(item.0));
            scored.into_iter().map(|(_, profile)| profile).collect()
        }
    }

    pub fn move_profile_cursor(&mut self, delta: isize) {
        self.profile_cursor =
            move_wrapping_cursor(self.profile_cursor, self.filtered_profiles().len(), delta);
    }

    pub fn selected_profile(&self) -> Option<&ProfileInfo> {
        self.filtered_profiles().get(self.profile_cursor).copied()
    }

    pub fn open_profile_popup(&mut self) {
        self.popup = Popup::ProfileSelect;
        self.profile_filter.clear();
        self.profile_cursor = self
            .active_profile_id
            .as_deref()
            .and_then(|active_id| {
                self.profiles
                    .iter()
                    .position(|profile| profile.id == active_id)
            })
            .unwrap_or(0);
    }

    pub fn find_profile_id(&self, query: &str) -> Option<String> {
        let needle = query.trim();
        if needle.is_empty() {
            return None;
        }
        self.profiles
            .iter()
            .find(|profile| profile.id == needle || profile.name.eq_ignore_ascii_case(needle))
            .map(|profile| profile.id.clone())
    }

    pub fn filtered_models(&self) -> Vec<&ModelEntry> {
        if self.model_filter.is_empty() {
            self.models.iter().collect()
        } else {
            let matcher = SkimMatcherV2::default();
            let mut scored: Vec<(i64, &ModelEntry)> = self
                .models
                .iter()
                .filter_map(|m| {
                    let score = [&m.label, &m.provider, &m.model]
                        .iter()
                        .filter_map(|field| matcher.fuzzy_match(field, &self.model_filter))
                        .max();
                    score.map(|s| (s, m))
                })
                .collect();
            scored.sort_by_key(|item| std::cmp::Reverse(item.0));
            scored.into_iter().map(|(_, m)| m).collect()
        }
    }

    fn model_index_for_entry(&self, entry: &ModelEntry) -> Option<usize> {
        self.models
            .iter()
            .position(|m| m.id == entry.id && m.node_id == entry.node_id)
    }

    /// Match catalog row to a provider/model pair, disambiguating local vs mesh.
    pub fn model_entry_matches_node(
        entry: &ModelEntry,
        provider: &str,
        model: &str,
        node_id: Option<&str>,
    ) -> bool {
        if entry.provider != provider || entry.model != model {
            return false;
        }
        match node_id {
            Some(node) => entry.node_id.as_deref() == Some(node),
            None => entry.node_id.is_none(),
        }
    }

    pub fn apply_model_selection_from_entry(&mut self, entry: &ModelEntry) {
        self.current_provider = Some(entry.provider.clone());
        self.current_model = Some(entry.model.clone());
        self.current_model_node_id = entry.node_id.clone();
    }

    pub fn live_model_selection_matches_entry(&self, entry: &ModelEntry) -> bool {
        let (Some(cp), Some(cm)) = (
            self.current_provider.as_deref(),
            self.current_model.as_deref(),
        ) else {
            return false;
        };
        Self::model_entry_matches_node(entry, cp, cm, self.current_model_node_id.as_deref())
    }

    pub fn model_popup_open_cursor(&self) -> usize {
        let items = self.visible_model_popup_items();
        items
            .iter()
            .position(|item| match item {
                ModelPopupItem::Model { model_idx } => self
                    .models
                    .get(*model_idx)
                    .is_some_and(|e| self.live_model_selection_matches_entry(e)),
                _ => false,
            })
            .or_else(|| {
                items
                    .iter()
                    .position(|item| matches!(item, ModelPopupItem::Model { .. }))
            })
            .unwrap_or(0)
    }

    pub fn visible_model_popup_items(&self) -> Vec<ModelPopupItem> {
        let filtered = self.filtered_models();
        let mut items = Vec::new();

        #[derive(Eq, PartialEq, Ord, PartialOrd)]
        struct GroupKey {
            provider: String,
            node_id: Option<String>,
        }

        let mut groups: std::collections::BTreeMap<GroupKey, Vec<&ModelEntry>> =
            std::collections::BTreeMap::new();
        for model in filtered.iter() {
            let key = GroupKey {
                provider: model.provider.clone(),
                node_id: model.node_id.clone(),
            };
            groups.entry(key).or_default().push(model);
        }

        for (key, models_in_group) in groups {
            let node_suffix = key.node_id.as_ref().map(|nid| {
                models_in_group
                    .first()
                    .and_then(|m| m.node_label.clone())
                    .unwrap_or_else(|| nid.clone())
            });
            items.push(ModelPopupItem::ProviderHeader {
                provider: key.provider.clone(),
                model_count: models_in_group.len(),
                node_suffix,
            });
            for model in models_in_group {
                if let Some(model_idx) = self.model_index_for_entry(model) {
                    items.push(ModelPopupItem::Model { model_idx });
                }
            }
        }

        items
    }

    #[cfg(test)]
    pub(crate) fn test_model_entry(
        id: &str,
        provider: &str,
        model: &str,
        node_id: Option<&str>,
        node_label: Option<&str>,
    ) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            label: model.into(),
            provider: provider.into(),
            model: model.into(),
            node_id: node_id.map(str::to_string),
            node_label: node_label.map(str::to_string),
            family: None,
            quant: None,
        }
    }

    pub fn push_log(&mut self, level: LogLevel, target: &'static str, message: impl Into<String>) {
        let message = message.into();
        if self.logs.last().is_some_and(|entry| {
            entry.level == level && entry.target == target && entry.message == message
        }) {
            return;
        }
        self.logs.push(AppLogEntry {
            elapsed: self.started_at.elapsed(),
            level,
            target,
            message,
        });
    }

    pub fn set_status(
        &mut self,
        level: LogLevel,
        target: &'static str,
        message: impl Into<String>,
    ) {
        let message = message.into();
        self.status = message.clone();
        self.push_log(level, target, message);
    }

    pub fn filtered_logs(&self) -> Vec<&AppLogEntry> {
        let query = self.log_filter.to_lowercase();
        self.logs
            .iter()
            .filter(|entry| entry.level >= self.log_level_filter)
            .filter(|entry| {
                query.is_empty()
                    || entry.message.to_lowercase().contains(&query)
                    || entry.target.to_lowercase().contains(&query)
                    || entry.level.label().to_lowercase().contains(&query)
            })
            .collect()
    }

    pub fn cycle_log_level_filter(&mut self) {
        self.log_level_filter = self.log_level_filter.next();
    }

    pub fn cancel_confirm_active(&self) -> bool {
        self.pending_cancel_confirm_until
            .map(|deadline| Instant::now() <= deadline)
            .unwrap_or(false)
    }

    pub fn arm_cancel_confirm(&mut self) {
        self.pending_cancel_confirm_until = Some(Instant::now() + CANCEL_CONFIRM_TIMEOUT);
        self.set_status(LogLevel::Warn, "input", "press Esc again to stop");
    }

    pub fn clear_cancel_confirm(&mut self) {
        self.pending_cancel_confirm_until = None;
    }

    pub fn is_turn_active(&self) -> bool {
        matches!(
            self.activity,
            ActivityState::Thinking
                | ActivityState::Streaming
                | ActivityState::RunningTool { .. }
                | ActivityState::Compacting { .. }
        )
    }

    /// Adjust `scroll_offset` to compensate for content growth so the
    /// viewport stays at the same absolute position when the user is
    /// scrolled up.  No-op when `scroll_offset == 0` (auto-following).
    ///
    /// Call from the renderer after computing the new `total_height`.
    pub fn compensate_scroll_for_growth(&mut self, total_height: u16) {
        let growth = total_height.saturating_sub(self.prev_total_height);
        if self.scroll_offset > 0 && growth > 0 {
            self.scroll_offset = self.scroll_offset.saturating_add(growth);
        }
        self.prev_total_height = total_height;
    }

    pub fn has_cancellable_activity(&self) -> bool {
        self.is_turn_active()
    }

    pub fn has_pending_session_op(&self) -> bool {
        matches!(self.activity, ActivityState::SessionOp(_))
    }

    pub fn forkable_turns(&self) -> Vec<ForkTurnItem> {
        let mut turns = Vec::new();
        let mut current_user: Option<(Option<String>, String)> = None;
        let mut current_assistant: Option<(String, String)> = None;

        for entry in &self.messages {
            match entry {
                ChatEntry::User { text, message_id } => {
                    if let Some((user_id, user_text)) = current_user.take()
                        && let Some(item) = Self::fork_turn_item(
                            turns.len() + 1,
                            user_id,
                            user_text,
                            current_assistant.take(),
                        )
                    {
                        turns.push(item);
                    }
                    current_user = Some((message_id.clone(), text.clone()));
                    current_assistant = None;
                }
                ChatEntry::Assistant {
                    content,
                    message_id,
                    ..
                } => {
                    if let Some(id) = message_id.clone() {
                        current_assistant = Some((id, content.clone()));
                    }
                }
                _ => {}
            }
        }

        if let Some((user_id, user_text)) = current_user.take()
            && let Some(item) = Self::fork_turn_item(
                turns.len() + 1,
                user_id,
                user_text,
                current_assistant.take(),
            )
        {
            turns.push(item);
        }

        turns
    }

    fn fork_turn_item(
        turn_index: usize,
        user_id: Option<String>,
        user_text: String,
        assistant: Option<(String, String)>,
    ) -> Option<ForkTurnItem> {
        let (message_id, boundary_kind, assistant_preview) = match assistant {
            Some((assistant_id, assistant_text)) => (
                Some(assistant_id),
                ForkBoundaryKind::Assistant,
                assistant_text,
            ),
            None => (user_id, ForkBoundaryKind::User, String::new()),
        };

        let message_id = message_id.filter(|message_id| !message_id.is_empty())?;

        Some(ForkTurnItem {
            turn_index,
            message_id,
            boundary_kind,
            user_preview: user_text,
            assistant_preview,
        })
    }

    pub fn filtered_fork_turns(&self) -> Vec<ForkTurnItem> {
        let query = self.fork_filter.trim().to_lowercase();
        self.forkable_turns()
            .into_iter()
            .filter(|turn| {
                query.is_empty()
                    || turn.user_preview.to_lowercase().contains(&query)
                    || turn.assistant_preview.to_lowercase().contains(&query)
            })
            .collect()
    }

    pub fn visible_fork_turns(&self) -> Vec<ForkTurnItem> {
        self.filtered_fork_turns().into_iter().rev().collect()
    }

    pub fn latest_fork_boundary(&self) -> Option<ForkTurnItem> {
        if self.is_turn_active() {
            None
        } else {
            self.forkable_turns().into_iter().last()
        }
    }

    pub fn open_fork_turn_popup(&mut self) {
        self.popup = Popup::ForkTurnSelect;
        self.fork_filter.clear();
        self.fork_cursor = 0;
    }

    pub fn move_fork_cursor(&mut self, delta: isize) {
        self.fork_cursor =
            move_wrapping_cursor(self.fork_cursor, self.visible_fork_turns().len(), delta);
    }

    pub fn fork_filter_insert(&mut self, c: char) {
        self.fork_filter.push(c);
        self.fork_cursor = 0;
    }

    pub fn fork_filter_backspace(&mut self) {
        self.fork_filter.pop();
        self.fork_cursor = 0;
    }

    pub fn selected_fork_turn(&self) -> Option<ForkTurnItem> {
        self.visible_fork_turns().get(self.fork_cursor).cloned()
    }

    pub fn push_pending_prompt(&mut self, text: String) -> String {
        self.pending_prompt_seq = self.pending_prompt_seq.saturating_add(1);
        let local_id = format!("local:pending:{}", self.pending_prompt_seq);
        self.messages.push(ChatEntry::User {
            text,
            message_id: Some(local_id.clone()),
        });
        self.card_cache.invalidate();
        self.scroll_offset = 0;
        local_id
    }

    pub fn input_blocked_by_activity(&self) -> bool {
        self.elicitation.is_some()
            || self.has_pending_session_op()
            || self.pending_cancel_confirm_until.is_some()
    }

    pub fn should_hide_input_contents(&self) -> bool {
        self.input_blocked_by_activity()
    }

    pub fn activity_status_text(&self) -> Option<String> {
        match &self.activity {
            ActivityState::Idle => None,
            ActivityState::Thinking => Some("thinking...".into()),
            ActivityState::Streaming => Some("streaming...".into()),
            ActivityState::RunningTool { name } => Some(format!("tool: {name}")),
            ActivityState::Compacting { token_estimate } => {
                Some(format!("compacting context (~{token_estimate} tokens)"))
            }
            ActivityState::SessionOp(SessionOp::Undo) => Some("undoing...".into()),
            ActivityState::SessionOp(SessionOp::Redo) => Some("redoing...".into()),
        }
    }

    pub fn refresh_transient_status(&mut self) {
        if self.pending_cancel_confirm_until.is_some() {
            return;
        }
        if self.elicitation.is_some() {
            self.set_status(
                LogLevel::Debug,
                "elicitation",
                "question - answer in the panel above input",
            );
        } else if let Some(activity_status) = self.activity_status_text() {
            self.set_status(LogLevel::Debug, "activity", activity_status);
        } else if self.conn == ConnState::Connected {
            self.set_status(LogLevel::Debug, "activity", "ready");
        }
    }

    pub fn clear_expired_cancel_confirm(&mut self) {
        if self.pending_cancel_confirm_until.is_some() && !self.cancel_confirm_active() {
            self.clear_cancel_confirm();
            self.refresh_transient_status();
        }
    }

    pub fn begin_llm_request_span(&mut self, timestamp: Option<i64>) {
        if self.session_stats.open_llm_request_ts.is_none() {
            self.session_stats.open_llm_request_ts = timestamp;
            self.session_stats.open_llm_request_instant = Some(Instant::now());
        }
    }

    pub fn end_llm_request_span(&mut self, timestamp: Option<i64>) {
        let duration = match (self.session_stats.open_llm_request_ts, timestamp) {
            (Some(started), Some(ended)) if ended >= started => {
                Some(Duration::from_secs((ended - started) as u64))
            }
            _ => self
                .session_stats
                .open_llm_request_instant
                .map(|started| started.elapsed()),
        };
        if let Some(duration) = duration {
            self.session_stats.active_llm_duration += duration;
        }
        self.session_stats.open_llm_request_ts = None;
        self.session_stats.open_llm_request_instant = None;
    }

    pub fn apply_event_stats(&mut self, kind: &EventKind, timestamp: Option<i64>) {
        match kind {
            EventKind::ToolCallStart { .. } => {
                self.session_stats.total_tool_calls =
                    self.session_stats.total_tool_calls.saturating_add(1);
            }
            EventKind::LlmRequestStart { .. } => {
                self.begin_llm_request_span(timestamp);
            }
            EventKind::LlmRequestEnd { context_tokens, .. } => {
                self.end_llm_request_span(timestamp);
                if let Some(ctx) = context_tokens {
                    self.session_stats.latest_context_tokens = Some(*ctx);
                }
            }
            EventKind::Cancelled | EventKind::Error { .. } => {
                self.end_llm_request_span(timestamp);
            }
            _ => {}
        }
    }

    pub fn llm_request_elapsed(&self) -> Option<Duration> {
        let mut elapsed = self.session_stats.active_llm_duration;
        if let Some(started) = self.session_stats.open_llm_request_instant {
            elapsed += started.elapsed();
        }
        if elapsed.is_zero() {
            None
        } else {
            Some(elapsed)
        }
    }

    pub fn handle_connection_event(&mut self, event: ConnectionEvent) {
        self.clear_cancel_confirm();
        match event {
            ConnectionEvent::Connecting { attempt, delay_ms } => {
                self.conn = ConnState::Connecting;
                self.reconnect_attempt = attempt;
                self.reconnect_delay_ms = Some(delay_ms);
                let secs = delay_ms as f64 / 1000.0;
                self.set_status(
                    LogLevel::Warn,
                    "connection",
                    format!("waiting for server - retry {attempt} in {secs:.1}s"),
                );
            }
            ConnectionEvent::Connected => {
                self.conn = ConnState::Connected;
                self.reconnect_attempt = 0;
                self.reconnect_delay_ms = None;
                self.set_status(
                    LogLevel::Info,
                    "connection",
                    if self.session_id.is_some() {
                        "reconnected".to_string()
                    } else {
                        "connected".to_string()
                    },
                );
            }
            ConnectionEvent::Disconnected { reason } => {
                self.conn = ConnState::Disconnected;
                self.reconnect_delay_ms = None;
                self.session_discovery_in_progress = false;
                self.pending_session_group_loads.clear();
                self.set_status(
                    LogLevel::Warn,
                    "connection",
                    format!("connection lost - {reason}"),
                );
            }
        }
    }

    pub fn has_pending_undo(&self) -> bool {
        self.undo_state
            .as_ref()
            .map(|state| {
                state
                    .stack
                    .iter()
                    .any(|frame| frame.status == UndoFrameStatus::Pending)
            })
            .unwrap_or(false)
    }

    pub fn pending_session_label(&self) -> Option<&'static str> {
        match self.activity {
            ActivityState::SessionOp(SessionOp::Undo) => Some("undoing"),
            ActivityState::SessionOp(SessionOp::Redo) => Some("redoing"),
            _ => None,
        }
    }

    pub fn current_undo_target(&self) -> Option<&UndoableTurn> {
        let frontier_message_id = self
            .undo_state
            .as_ref()
            .and_then(|state| state.frontier_message_id.as_deref());

        let mut start_index = self.undoable_turns.len();
        if let Some(frontier_message_id) = frontier_message_id
            && let Some(frontier_index) = self
                .undoable_turns
                .iter()
                .position(|turn| turn.message_id == frontier_message_id)
        {
            start_index = frontier_index;
        }

        self.undoable_turns[..start_index]
            .iter()
            .rev()
            .find(|turn| !turn.message_id.is_empty())
    }

    pub fn can_redo(&self) -> bool {
        self.undo_state
            .as_ref()
            .map(|state| !state.stack.is_empty())
            .unwrap_or(false)
    }

    pub fn push_pending_undo(&mut self, turn: &UndoableTurn) {
        let mut stack = self
            .undo_state
            .as_ref()
            .map(|state| state.stack.clone())
            .unwrap_or_default();
        stack.push(UndoFrame {
            turn_id: turn.turn_id.clone(),
            message_id: turn.message_id.clone(),
            status: UndoFrameStatus::Pending,
            reverted_files: Vec::new(),
        });
        self.undo_state = Some(UndoState {
            stack,
            frontier_message_id: Some(turn.message_id.clone()),
        });
    }

    pub fn build_undo_state_from_server_stack(
        &self,
        undo_stack: &UndoStackSnapshot,
        preferred_frontier_message_id: Option<&str>,
        reverted_files: Option<&[String]>,
    ) -> Option<UndoState> {
        if undo_stack.message_ids.is_empty() {
            return None;
        }

        let previous_state = self.undo_state.as_ref();
        let mut previous_by_message_id = std::collections::HashMap::new();
        if let Some(previous_state) = previous_state {
            for frame in &previous_state.stack {
                previous_by_message_id.insert(frame.message_id.clone(), frame.clone());
            }
        }

        let stack: Vec<UndoFrame> = undo_stack
            .message_ids
            .iter()
            .map(|message_id| {
                let previous = previous_by_message_id.get(message_id);
                let reverted_files = if preferred_frontier_message_id == Some(message_id.as_str()) {
                    reverted_files
                        .map(|files| files.to_vec())
                        .or_else(|| previous.map(|frame| frame.reverted_files.clone()))
                        .unwrap_or_default()
                } else {
                    previous
                        .map(|frame| frame.reverted_files.clone())
                        .unwrap_or_default()
                };
                let turn_id = previous
                    .map(|frame| frame.turn_id.clone())
                    .or_else(|| {
                        self.undoable_turns
                            .iter()
                            .find(|turn| turn.message_id == *message_id)
                            .map(|turn| turn.turn_id.clone())
                    })
                    .unwrap_or_else(|| message_id.clone());
                UndoFrame {
                    turn_id,
                    message_id: message_id.clone(),
                    status: UndoFrameStatus::Confirmed,
                    reverted_files,
                }
            })
            .collect();

        let has_message = |message_id: Option<&str>| {
            message_id
                .map(|message_id| stack.iter().any(|frame| frame.message_id == message_id))
                .unwrap_or(false)
        };

        let frontier_message_id = if has_message(preferred_frontier_message_id) {
            preferred_frontier_message_id.map(ToOwned::to_owned)
        } else if has_message(previous_state.and_then(|state| state.frontier_message_id.as_deref()))
        {
            previous_state.and_then(|state| state.frontier_message_id.clone())
        } else {
            stack.last().map(|frame| frame.message_id.clone())
        };

        Some(UndoState {
            stack,
            frontier_message_id,
        })
    }

    /// Mark the pending elicitation chat card with an outcome and clear the active state.
    pub fn resolve_elicitation(&mut self, elicitation_id: &str, outcome: &str) {
        for entry in &mut self.messages {
            if let ChatEntry::Elicitation {
                elicitation_id: eid,
                outcome: out,
                ..
            } = entry
                && eid == elicitation_id
            {
                *out = Some(outcome.to_string());
                break;
            }
        }
        self.elicitation = None;
        self.elicitation_ui = None;
        self.card_cache.invalidate();
        self.refresh_transient_status();
    }

    pub fn next_mode(&self) -> String {
        match self.agent_mode.as_str() {
            "build" => "plan".into(),
            "plan" => "build".into(),
            "review" => self
                .mode_before_review
                .clone()
                .unwrap_or_else(|| "build".into()),
            _ => "build".into(),
        }
    }

    // ── delegate model preferences ───────────────────────────────────────────

    /// Whether there are multiple agents (multi-agent / delegation mode).
    pub fn is_multi_agent(&self) -> bool {
        self.agents.len() > 1
    }

    /// Tabs: 0 = session model; 1.. = delegate agents when `agents.len() > 1`.
    pub fn model_popup_tab_count(&self) -> usize {
        if self.agents.len() > 1 {
            self.agents.len()
        } else {
            1
        }
    }

    pub fn model_popup_has_tabs(&self) -> bool {
        self.agents.len() > 1
    }

    pub fn model_popup_tab_label(&self, tab_idx: usize) -> &str {
        if tab_idx == 0 {
            "session"
        } else {
            self.agents
                .get(tab_idx)
                .map(|a| a.name.as_str())
                .unwrap_or("???")
        }
    }

    /// `None` on session tab; delegate agent id on agent tabs.
    pub fn model_popup_tab_agent_id(&self, tab_idx: usize) -> Option<&str> {
        if tab_idx == 0 {
            None
        } else {
            self.agents.get(tab_idx).map(|a| a.id.as_str())
        }
    }

    pub fn model_popup_is_session_tab(&self, tab_idx: usize) -> bool {
        tab_idx == 0
    }

    pub fn delegate_preference_profile_id(&self) -> Option<&str> {
        self.current_session_profile_id()
            .or(self.active_profile_id.as_deref())
    }

    pub fn desired_agents_profile_id(&self) -> Option<&str> {
        self.delegate_preference_profile_id()
    }

    pub fn set_delegate_model_preference(
        &mut self,
        profile_id: &str,
        agent_id: &str,
        model: &ModelEntry,
    ) {
        self.delegate_model_preferences
            .entry(profile_id.to_string())
            .or_default()
            .insert(
                agent_id.to_string(),
                DelegateModelPreference {
                    model_id: model.id.clone(),
                    provider: model.provider.clone(),
                    model: model.model.clone(),
                    node_id: model.node_id.clone(),
                },
            );
    }

    pub fn clear_delegate_model_preference(&mut self, profile_id: &str, agent_id: &str) {
        if let Some(preferences) = self.delegate_model_preferences.get_mut(profile_id) {
            preferences.remove(agent_id);
            if preferences.is_empty() {
                self.delegate_model_preferences.remove(profile_id);
            }
        }
    }

    pub fn get_delegate_model_preference(
        &self,
        profile_id: &str,
        agent_id: &str,
    ) -> Option<&DelegateModelPreference> {
        self.delegate_model_preferences
            .get(profile_id)
            .and_then(|preferences| preferences.get(agent_id))
    }

    pub fn delegate_model_commands_for_session(
        &self,
        session_id: &str,
        profile_id: &str,
    ) -> Vec<ClientMsg> {
        let known_agents: HashSet<&str> =
            self.agents.iter().skip(1).map(|a| a.id.as_str()).collect();
        self.delegate_model_preferences
            .get(profile_id)
            .into_iter()
            .flat_map(|preferences| preferences.iter())
            .filter(|(agent_id, _)| known_agents.contains(agent_id.as_str()))
            .map(|(agent_id, preference)| ClientMsg::SetDelegateModel {
                session_id: session_id.to_string(),
                agent_id: agent_id.clone(),
                model_id: Some(preference.model_id.clone()),
                node_id: preference.node_id.clone(),
            })
            .collect()
    }

    /// Cursor position for a delegate agent's preferred model in the popup list.
    pub fn delegate_model_cursor(&self, agent_id: &str) -> usize {
        let items = self.visible_model_popup_items();
        let Some(profile_id) = self.delegate_preference_profile_id() else {
            return items
                .iter()
                .position(|item| matches!(item, ModelPopupItem::Model { .. }))
                .unwrap_or(0);
        };
        let Some(preference) = self.get_delegate_model_preference(profile_id, agent_id) else {
            return items
                .iter()
                .position(|item| matches!(item, ModelPopupItem::Model { .. }))
                .unwrap_or(0);
        };
        items
            .iter()
            .position(|item| match item {
                ModelPopupItem::Model { model_idx } => {
                    self.models[*model_idx].id == preference.model_id
                        && self.models[*model_idx].node_id == preference.node_id
                }
                _ => false,
            })
            .unwrap_or(0)
    }
}

// ── reasoning_effort_tests ────────────────────────────────────────────────────

#[cfg(test)]
mod reasoning_effort_tests {
    use super::*;
    use crate::domain::session::SessionSummary;

    // ── reasoning_effort_label ────────────────────────────────────────────────

    #[test]
    fn label_none_is_auto() {
        let app = App::new();
        assert_eq!(app.reasoning_effort_label(), "auto");
    }

    #[test]
    fn label_low() {
        let mut app = App::new();
        app.reasoning_effort = Some("low".into());
        assert_eq!(app.reasoning_effort_label(), "low");
    }

    #[test]
    fn label_medium() {
        let mut app = App::new();
        app.reasoning_effort = Some("medium".into());
        assert_eq!(app.reasoning_effort_label(), "medium");
    }

    #[test]
    fn label_high() {
        let mut app = App::new();
        app.reasoning_effort = Some("high".into());
        assert_eq!(app.reasoning_effort_label(), "high");
    }

    #[test]
    fn label_max() {
        let mut app = App::new();
        app.reasoning_effort = Some("max".into());
        assert_eq!(app.reasoning_effort_label(), "max");
    }

    #[test]
    fn label_unknown_passes_through() {
        let mut app = App::new();
        app.reasoning_effort = Some("ultra".into());
        assert_eq!(app.reasoning_effort_label(), "ultra");
    }

    // ── cycle_reasoning_effort ────────────────────────────────────────────────

    #[test]
    fn cycle_from_auto_to_low() {
        let mut app = App::new();
        assert_eq!(app.reasoning_effort, None);
        app.cycle_reasoning_effort();
        assert_eq!(app.reasoning_effort, Some("low".into()));
    }

    #[test]
    fn cycle_from_low_to_medium() {
        let mut app = App::new();
        app.reasoning_effort = Some("low".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.reasoning_effort, Some("medium".into()));
    }

    #[test]
    fn cycle_from_medium_to_high() {
        let mut app = App::new();
        app.reasoning_effort = Some("medium".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.reasoning_effort, Some("high".into()));
    }

    #[test]
    fn cycle_from_high_to_max() {
        let mut app = App::new();
        app.reasoning_effort = Some("high".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.reasoning_effort, Some("max".into()));
    }

    #[test]
    fn cycle_from_max_wraps_to_auto() {
        let mut app = App::new();
        app.reasoning_effort = Some("max".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.reasoning_effort, None);
    }

    #[test]
    fn cycle_full_round_trip() {
        let mut app = App::new();
        // auto → low → medium → high → max → auto
        for _ in 0..5 {
            app.cycle_reasoning_effort();
        }
        assert_eq!(app.reasoning_effort, None);
    }

    #[test]
    fn cycle_reasoning_effort_unknown_value_noop() {
        let mut app = App::new();
        app.reasoning_effort = Some("invalid_level".into());
        let result = app.cycle_reasoning_effort();
        // unknown current value → no-op: returns None and leaves state unchanged
        assert!(
            result.is_none(),
            "cycling an unknown value should return None"
        );
        assert_eq!(
            app.reasoning_effort,
            Some("invalid_level".into()),
            "state must not change when cycling an unknown value"
        );
    }

    #[test]
    fn cycle_returns_correct_client_msg() {
        let mut app = App::new(); // starts at auto
        let msg = app.cycle_reasoning_effort().expect("auto is a valid level");
        // auto → low: should send "low"
        match msg {
            ClientMsg::SetReasoningEffort { reasoning_effort } => {
                assert_eq!(reasoning_effort, "low");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn cycle_to_auto_sends_auto_string() {
        let mut app = App::new();
        app.reasoning_effort = Some("max".into());
        let msg = app.cycle_reasoning_effort().expect("max is a valid level");
        // max → auto: server expects "auto" string (not null)
        match msg {
            ClientMsg::SetReasoningEffort { reasoning_effort } => {
                assert_eq!(reasoning_effort, "auto");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    // ── set_reasoning_effort ────────────────────────────────────────────────────

    #[test]
    fn set_reasoning_effort_high_returns_correct_msg() {
        let mut app = App::new();
        let msg = app.set_reasoning_effort(Some("high"));
        assert_eq!(app.reasoning_effort, Some("high".into()));
        match msg {
            Some(ClientMsg::SetReasoningEffort { reasoning_effort }) => {
                assert_eq!(reasoning_effort, "high");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn set_reasoning_effort_auto_clears_to_none() {
        let mut app = App::new();
        app.reasoning_effort = Some("max".into());
        let msg = app.set_reasoning_effort(Some("auto"));
        assert_eq!(app.reasoning_effort, None);
        match msg {
            Some(ClientMsg::SetReasoningEffort { reasoning_effort }) => {
                assert_eq!(reasoning_effort, "auto");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn set_reasoning_effort_none_clears_to_auto() {
        let mut app = App::new();
        app.reasoning_effort = Some("low".into());
        let msg = app.set_reasoning_effort(None);
        assert_eq!(app.reasoning_effort, None);
        match msg {
            Some(ClientMsg::SetReasoningEffort { reasoning_effort }) => {
                assert_eq!(reasoning_effort, "auto");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn set_reasoning_effort_invalid_value_rejected() {
        let mut app = App::new();
        app.reasoning_effort = Some("medium".into());
        let msg = app.set_reasoning_effort(Some("ultra"));
        assert_eq!(app.reasoning_effort, Some("medium".into()));
        assert!(msg.is_none());
    }

    #[test]
    fn validate_reasoning_effort_normalizes_med() {
        assert_eq!(
            validate_reasoning_effort(Some("med")),
            Some(Some("medium".to_string()))
        );
        assert_eq!(
            validate_reasoning_effort(Some("MED")),
            None // case-sensitive
        );
    }

    // ── state message populates reasoning_effort ──────────────────────────────

    // ── reasoning_effort push notification ────────────────────────────────────

    #[test]
    fn active_session_count_requires_multiple_recent_sessions() {
        let mut app = App::new();
        app.note_session_activity("session-a");
        assert_eq!(app.active_session_count(), 1);

        app.note_session_activity("session-b");
        assert_eq!(app.active_session_count(), 2);
    }

    #[test]
    fn other_active_session_count_excludes_current_session() {
        let mut app = App::new();
        app.session_id = Some("session-a".into());
        app.note_session_activity("session-a");
        app.note_session_activity("session-b");
        app.note_session_activity("session-c");

        assert_eq!(app.other_active_session_count(), 2);
    }

    #[test]
    fn other_active_session_count_shows_other_session_when_current_is_idle() {
        let mut app = App::new();
        app.session_id = Some("session-a".into());
        app.note_session_activity("session-b");

        assert_eq!(app.other_active_session_count(), 1);
    }

    #[test]
    fn active_session_count_excludes_stale_sessions() {
        let mut app = App::new();
        app.note_session_activity("session-a");
        app.session_activity.insert(
            "session-b".into(),
            SessionActivity {
                last_event_at: Instant::now() - Duration::from_secs(6),
            },
        );

        assert_eq!(app.active_session_count(), 1);
        assert_eq!(app.other_active_session_count(), 1);
    }

    #[test]
    fn resolve_new_session_default_cwd_prefers_active_session_cwd_then_group_then_launch() {
        let mut app = App::new();
        app.launch_cwd = Some("/launch".into());
        app.session_id = Some("session-a".into());
        app.session_groups = vec![SessionGroup {
            cwd: Some("/group".into()),
            latest_activity: None,
            sessions: vec![SessionSummary {
                session_id: "session-a".into(),
                title: Some("Session A".into()),
                cwd: Some("/session".into()),
                created_at: None,
                updated_at: None,
                parent_session_id: None,
                has_children: false,
                ..Default::default()
            }],
            ..Default::default()
        }];
        assert_eq!(
            app.resolve_new_session_default_cwd().as_deref(),
            Some("/session")
        );

        app.session_groups[0].sessions[0].cwd = None;
        assert_eq!(
            app.resolve_new_session_default_cwd().as_deref(),
            Some("/group")
        );

        app.session_groups.clear();
        assert_eq!(
            app.resolve_new_session_default_cwd().as_deref(),
            Some("/launch")
        );
    }

    #[test]
    fn open_new_session_popup_prefills_path_and_cursor() {
        let mut app = App::new();
        app.launch_cwd = Some("/launch".into());

        app.open_new_session_popup();

        assert_eq!(app.popup, Popup::NewSession);
        assert_eq!(app.new_session_path, "/launch");
        assert_eq!(app.new_session_cursor, "/launch".len());
    }

    #[test]
    fn normalize_new_session_path_uses_launch_cwd_for_relative_paths() {
        let mut app = App::new();
        app.launch_cwd = Some("/launch/base".into());

        assert_eq!(
            app.normalize_new_session_path("proj/subdir").as_deref(),
            Some("/launch/base/proj/subdir")
        );
        assert_eq!(
            app.normalize_new_session_path("../proj/./subdir/..",)
                .as_deref(),
            Some("/launch/proj")
        );
        assert_eq!(
            app.normalize_new_session_path("/absolute/path/../clean")
                .as_deref(),
            Some("/absolute/clean")
        );
    }

    #[test]
    fn normalize_new_session_path_expands_tilde() {
        let app = App::new();
        let home = dirs::home_dir().expect("home dir available for test");
        let expected = home.join("workspace").to_string_lossy().into_owned();

        assert_eq!(
            app.normalize_new_session_path("~/workspace").as_deref(),
            Some(expected.as_str())
        );
    }

    #[test]
    fn accept_selected_new_session_completion_replaces_input() {
        let mut app = App::new();
        app.new_session_completion = Some(PathCompletionState {
            query: "pro".into(),
            selected_index: 0,
            results: vec![FileIndexEntryLite {
                path: "/launch/project/../project-two".into(),
                is_dir: true,
            }],
        });

        assert!(app.accept_selected_new_session_completion());
        assert_eq!(app.new_session_path, "/launch/project-two/");
        assert!(app.new_session_completion.is_none());
    }

    #[test]
    fn rank_path_completion_matches_filters_out_files() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("qmt-app-tests-path-complete-{pid}-{nanos}"));
        std::fs::create_dir_all(dir.join("project-dir")).unwrap();
        std::fs::write(dir.join("project-file.txt"), "x").unwrap();

        let mut app = App::new();
        app.launch_cwd = Some(dir.to_string_lossy().into_owned());
        let results = app.rank_path_completion_matches("project");

        assert!(results.iter().all(|entry| entry.is_dir));
        assert!(
            results
                .iter()
                .any(|entry| entry.path.ends_with("project-dir"))
        );
        assert!(
            !results
                .iter()
                .any(|entry| entry.path.ends_with("project-file.txt"))
        );
    }
}

// ── delegate_entry_tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod delegate_entry_tests {
    use super::*;
    use crate::domain::activity::DelegateStatus;

    fn make_entry(delegation_id: &str, objective: &str, status: DelegateStatus) -> DelegateEntry {
        DelegateEntry {
            delegation_id: delegation_id.into(),
            child_session_id: Some(format!("child-{delegation_id}")),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: objective.into(),
            status,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        }
    }

    // ── visible_delegate_entries ───────────────────────────────────────────────

    #[test]
    fn visible_entries_empty_when_no_entries() {
        let app = App::new();
        assert!(app.visible_delegate_entries().is_empty());
    }

    #[test]
    fn visible_entries_returns_all_when_no_filter() {
        let mut app = App::new();
        app.delegate_entries = vec![
            make_entry("d1", "Build feature", DelegateStatus::Completed),
            make_entry("d2", "Fix tests", DelegateStatus::InProgress),
        ];
        assert_eq!(app.visible_delegate_entries().len(), 2);
    }

    #[test]
    fn visible_entries_filters_by_objective() {
        let mut app = App::new();
        app.delegate_entries = vec![
            make_entry("d1", "Build feature", DelegateStatus::Completed),
            make_entry("d2", "Fix tests", DelegateStatus::InProgress),
        ];
        app.delegate_filter = "build".into();
        let entries = app.visible_delegate_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].delegation_id, "d1");
    }

    #[test]
    fn visible_entries_filters_by_delegation_id() {
        let mut app = App::new();
        app.delegate_entries = vec![
            make_entry("abc123", "Build feature", DelegateStatus::Completed),
            make_entry("xyz789", "Fix tests", DelegateStatus::InProgress),
        ];
        app.delegate_filter = "xyz".into();
        let entries = app.visible_delegate_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].delegation_id, "xyz789");
    }

    #[test]
    fn visible_entries_filters_by_target_agent() {
        let mut app = App::new();
        app.delegate_entries = vec![
            DelegateEntry {
                delegation_id: "d1".into(),
                child_session_id: None,
                delegate_tool_call_id: None,
                target_agent_id: Some("planner".into()),
                objective: "Plan work".into(),
                status: DelegateStatus::Completed,
                stats: DelegateStats::default(),
                started_at: None,
                ended_at: None,
                child_state: DelegateChildState::None,
            },
            DelegateEntry {
                delegation_id: "d2".into(),
                child_session_id: None,
                delegate_tool_call_id: None,
                target_agent_id: Some("coder".into()),
                objective: "Write code".into(),
                status: DelegateStatus::InProgress,
                stats: DelegateStats::default(),
                started_at: None,
                ended_at: None,
                child_state: DelegateChildState::None,
            },
        ];
        app.delegate_filter = "planner".into();
        let entries = app.visible_delegate_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].delegation_id, "d1");
    }

    #[test]
    fn visible_entries_filter_is_case_insensitive() {
        let mut app = App::new();
        app.delegate_entries = vec![make_entry("d1", "Build Feature", DelegateStatus::Completed)];
        app.delegate_filter = "BUILD".into();
        assert_eq!(app.visible_delegate_entries().len(), 1);
    }

    // ── delegation event processing ───────────────────────────────────────────

    // ── DelegateStats accumulation ────────────────────────────────────────────

    #[test]
    fn stats_tool_call_increments() {
        let mut stats = DelegateStats::default();
        accumulate_delegate_stats(
            &mut stats,
            &EventKind::ToolCallStart {
                tool_call_id: None,
                tool_name: "read_tool".into(),
                arguments: None,
            },
        );
        assert_eq!(stats.tool_calls, 1);
    }

    #[test]
    fn stats_message_increments_on_assistant_message() {
        let mut stats = DelegateStats::default();
        accumulate_delegate_stats(
            &mut stats,
            &EventKind::AssistantMessageStored {
                content: "hello".into(),
                thinking: None,
                message_id: None,
            },
        );
        assert_eq!(stats.messages, 1);
    }

    #[test]
    fn stats_cost_accumulates_across_llm_requests() {
        let mut stats = DelegateStats::default();
        accumulate_delegate_stats(
            &mut stats,
            &EventKind::LlmRequestEnd {
                finish_reason: None,
                cost_usd: Some(0.01),
                cumulative_cost_usd: None,
                context_tokens: None,
                tool_calls: None,
                metrics: None,
            },
        );
        accumulate_delegate_stats(
            &mut stats,
            &EventKind::LlmRequestEnd {
                finish_reason: None,
                cost_usd: Some(0.02),
                cumulative_cost_usd: None,
                context_tokens: None,
                tool_calls: None,
                metrics: None,
            },
        );
        assert!((stats.cost_usd - 0.03).abs() < 1e-9);
    }

    #[test]
    fn stats_context_tokens_takes_latest_value() {
        let mut stats = DelegateStats::default();
        accumulate_delegate_stats(
            &mut stats,
            &EventKind::LlmRequestEnd {
                finish_reason: None,
                cost_usd: None,
                cumulative_cost_usd: None,
                context_tokens: Some(1000),
                tool_calls: None,
                metrics: None,
            },
        );
        accumulate_delegate_stats(
            &mut stats,
            &EventKind::LlmRequestEnd {
                finish_reason: None,
                cost_usd: None,
                cumulative_cost_usd: None,
                context_tokens: Some(2048),
                tool_calls: None,
                metrics: None,
            },
        );
        assert_eq!(stats.context_tokens, 2048);
    }

    #[test]
    fn stats_context_limit_set_from_provider_changed() {
        let mut stats = DelegateStats::default();
        accumulate_delegate_stats(
            &mut stats,
            &EventKind::ProviderChanged {
                provider: "anthropic".into(),
                model: "claude-sonnet".into(),
                config_id: None,
                context_limit: Some(200_000),
                provider_node_id: None,
            },
        );
        assert_eq!(stats.context_limit, 200_000);
    }

    #[test]
    fn child_state_tracks_pending_elicitation_and_terminal_progress() {
        let mut state = DelegateChildState::None;
        update_delegate_child_state(
            &mut state,
            &EventKind::ElicitationRequested {
                elicitation_id: "elic-1".into(),
                session_id: "child-1".into(),
                message: "Choose".into(),
                requested_schema: serde_json::json!({"type":"string"}),
                source: "builtin:question".into(),
            },
        );
        assert!(matches!(
            state,
            DelegateChildState::PendingElicitation { ref elicitation_id, ref message, .. }
                if elicitation_id == "elic-1" && message == "Choose"
        ));

        update_delegate_child_state(
            &mut state,
            &EventKind::AssistantMessageStored {
                content: "answer".into(),
                thinking: None,
                message_id: None,
            },
        );
        assert_eq!(state, DelegateChildState::AssistantMessage);

        update_delegate_child_state(
            &mut state,
            &EventKind::ToolCallEnd {
                tool_call_id: None,
                tool_name: "question".into(),
                result: None,
                is_error: Some(false),
            },
        );
        assert_eq!(state, DelegateChildState::QuestionToolFinished);

        update_delegate_child_state(
            &mut state,
            &EventKind::ToolCallEnd {
                tool_call_id: None,
                tool_name: "read_tool".into(),
                result: None,
                is_error: Some(false),
            },
        );
        assert_eq!(state, DelegateChildState::OtherProgress);
    }

    #[test]
    fn stats_context_pct_computes_correctly() {
        let stats = DelegateStats {
            context_tokens: 50_000,
            context_limit: 200_000,
            ..DelegateStats::default()
        };
        assert_eq!(stats.context_pct(), Some(25));
    }

    #[test]
    fn stats_context_pct_none_when_no_limit() {
        let stats = DelegateStats {
            context_tokens: 1000,
            context_limit: 0,
            ..DelegateStats::default()
        };
        assert_eq!(stats.context_pct(), None);
    }

    // ── child session event routing ───────────────────────────────────────────

    // ── subscribe on SessionForked ────────────────────────────────────────────

    // ── delegation view (parent tracking, Screen::Delegate) ──────────────────

    // ── delegation result suppression ────────────────────────────────────────

    // ── delegation duration tracking ─────────────────────────────────────────
}

#[cfg(test)]
mod model_node_selection_tests {
    use super::*;

    #[test]
    fn live_selection_matches_only_one_of_local_and_remote_duplicate() {
        let mut app = App::new();
        let local = App::test_model_entry("codex/local", "codex", "gpt-5", None, None);
        let remote = App::test_model_entry(
            "codex@n1/remote",
            "codex",
            "gpt-5",
            Some("n1"),
            Some("peer"),
        );
        app.models = vec![local.clone(), remote.clone()];
        app.apply_model_selection_from_entry(&remote);

        assert!(app.live_model_selection_matches_entry(&remote));
        assert!(!app.live_model_selection_matches_entry(&local));
    }

    #[test]
    fn model_popup_open_cursor_points_at_remote_when_node_id_set() {
        let mut app = App::new();
        app.models = vec![
            App::test_model_entry("codex/local", "codex", "gpt-5", None, None),
            App::test_model_entry("codex@n1/gpt-5", "codex", "gpt-5", Some("n1"), Some("box")),
        ];
        let remote = app.models[1].clone();
        app.apply_model_selection_from_entry(&remote);
        let items = app.visible_model_popup_items();
        let cursor = app.model_popup_open_cursor();
        let item = &items[cursor];
        assert!(
            matches!(item, ModelPopupItem::Model { model_idx } if app.models[*model_idx].node_id.is_some()),
            "cursor should be on remote row"
        );
    }
}

// ── session_mode_tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod session_mode_tests {
    use super::*;

    // ── SessionModeChanged in live events ─────────────────────────────────────

    #[test]
    fn next_mode_exits_review_to_previous_mode() {
        let mut app = App::new();
        app.agent_mode = "review".into();
        app.mode_before_review = Some("plan".into());
        assert_eq!(app.next_mode(), "plan");
    }

    #[test]
    fn next_mode_from_review_defaults_to_build_without_previous_mode() {
        let mut app = App::new();
        app.agent_mode = "review".into();
        app.mode_before_review = None;
        assert_eq!(app.next_mode(), "build");
    }

    // ── SessionModeChanged in audit replay ────────────────────────────────────

    // ── session_loaded returns SetAgentMode ───────────────────────────────────

    // ── session_loaded: model/effort from audit only (no TUI cache) ─────────────
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::OUTCOME_BULLET;
    use crate::protocol::{AgentEvent, ProgressKind};

    fn make_turn(message_id: &str) -> UndoableTurn {
        UndoableTurn {
            turn_id: format!("turn-{message_id}"),
            message_id: message_id.into(),
            text: format!("prompt {message_id}"),
        }
    }

    fn make_stack(ids: &[&str]) -> UndoStackSnapshot {
        UndoStackSnapshot {
            message_ids: ids.iter().map(|id| (*id).into()).collect(),
        }
    }

    #[test]
    fn backend_next_protocol_events_deserialize() {
        let session_queued = serde_json::json!({
            "kind": {
                "type": "session_queued",
                "data": { "reason": "waiting for previous operation to complete" }
            },
            "timestamp": null
        });
        let session_configured = serde_json::json!({
            "kind": {
                "type": "session_configured",
                "data": {
                    "cwd": "/workspace/project",
                    "mcp_servers": [],
                    "limits": {
                        "max_steps": 200,
                        "max_turns": 50,
                        "max_cost_usd": null
                    }
                }
            },
            "timestamp": null
        });
        let tools_available = serde_json::json!({
            "kind": {
                "type": "tools_available",
                "data": {
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "search_text",
                            "description": "Search file contents",
                            "parameters": { "type": "object" }
                        }
                    }],
                    "tools_hash": "123456789"
                }
            },
            "timestamp": null
        });
        let artifact_recorded = serde_json::json!({
            "kind": {
                "type": "artifact_recorded",
                "data": {
                    "artifact": {
                        "kind": "file",
                        "uri": null,
                        "path": "src/generated.txt",
                        "summary": "Produced by write_file",
                        "created_at": "2026-04-29T14:25:09Z"
                    }
                }
            },
            "timestamp": null
        });

        let queued: AgentEvent = serde_json::from_value(session_queued).unwrap();
        assert!(
            matches!(queued.kind, EventKind::SessionQueued { reason } if reason == "waiting for previous operation to complete")
        );

        let configured: AgentEvent = serde_json::from_value(session_configured).unwrap();
        assert!(
            matches!(configured.kind, EventKind::SessionConfigured { cwd, mcp_servers, limits } if cwd.as_deref() == Some("/workspace/project") && mcp_servers.is_empty() && limits.as_ref().and_then(|l| l.max_steps) == Some(200))
        );

        let available: AgentEvent = serde_json::from_value(tools_available).unwrap();
        assert!(
            matches!(available.kind, EventKind::ToolsAvailable { tools, tools_hash } if tools.first().and_then(|tool| tool.function.as_ref()).map(|function| function.name.as_str()) == Some("search_text") && tools_hash.is_some())
        );

        let artifact: AgentEvent = serde_json::from_value(artifact_recorded).unwrap();
        assert!(
            matches!(artifact.kind, EventKind::ArtifactRecorded { artifact } if artifact.kind == "file" && artifact.path.as_deref() == Some("src/generated.txt") && artifact.summary.as_deref() == Some("Produced by write_file"))
        );
    }

    #[test]
    fn backend_snapshot_and_progress_events_deserialize() {
        let snapshot_start = serde_json::json!({
            "kind": {
                "type": "snapshot_start",
                "data": { "policy": "diff" }
            },
            "timestamp": null
        });
        let snapshot_end = serde_json::json!({
            "kind": {
                "type": "snapshot_end",
                "data": { "summary": "1 modified" }
            },
            "timestamp": null
        });
        let progress_recorded = serde_json::json!({
            "kind": {
                "type": "progress_recorded",
                "data": {
                    "progress_entry": {
                        "kind": "tool_call",
                        "content": "Calling tool: shell",
                        "metadata": "{\"tool\":\"shell\"}",
                        "created_at": "2026-04-13T00:00:00Z"
                    }
                }
            },
            "timestamp": null
        });

        let start: AgentEvent = serde_json::from_value(snapshot_start).unwrap();
        assert!(matches!(start.kind, EventKind::SnapshotStart { policy } if policy == "diff"));

        let end: AgentEvent = serde_json::from_value(snapshot_end).unwrap();
        assert!(
            matches!(end.kind, EventKind::SnapshotEnd { summary } if summary.as_deref() == Some("1 modified"))
        );

        let progress: AgentEvent = serde_json::from_value(progress_recorded).unwrap();
        assert!(
            matches!(progress.kind, EventKind::ProgressRecorded { progress_entry } if progress_entry.kind == ProgressKind::ToolCall && progress_entry.content == "Calling tool: shell")
        );
    }

    #[test]
    fn fork_boundary_derivation_uses_last_assistant_then_user_fallback() {
        let mut app = App::new();
        app.messages = vec![
            ChatEntry::User {
                text: "first prompt".into(),
                message_id: Some("user-1".into()),
            },
            ChatEntry::Assistant {
                content: "first reply".into(),
                thinking: None,
                message_id: Some("asst-1a".into()),
            },
            ChatEntry::Assistant {
                content: "final reply".into(),
                thinking: None,
                message_id: Some("asst-1b".into()),
            },
            ChatEntry::User {
                text: "second prompt".into(),
                message_id: Some("user-2".into()),
            },
        ];

        let turns = app.forkable_turns();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].message_id, "asst-1b");
        assert_eq!(turns[0].boundary_kind, ForkBoundaryKind::Assistant);
        assert_eq!(turns[1].message_id, "user-2");
        assert_eq!(turns[1].boundary_kind, ForkBoundaryKind::User);
    }

    #[test]
    fn fork_boundary_derivation_includes_turn_when_only_assistant_has_message_id() {
        let mut app = App::new();
        app.messages = vec![
            ChatEntry::User {
                text: "first prompt".into(),
                message_id: None,
            },
            ChatEntry::Assistant {
                content: "first reply".into(),
                thinking: None,
                message_id: Some("asst-1".into()),
            },
            ChatEntry::User {
                text: "second prompt".into(),
                message_id: Some("user-2".into()),
            },
            ChatEntry::Assistant {
                content: "second reply".into(),
                thinking: None,
                message_id: None,
            },
            ChatEntry::User {
                text: "third prompt".into(),
                message_id: None,
            },
            ChatEntry::Assistant {
                content: "third reply".into(),
                thinking: None,
                message_id: None,
            },
        ];

        let turns = app.forkable_turns();
        assert_eq!(turns.len(), 2);

        assert_eq!(turns[0].turn_index, 1);
        assert_eq!(turns[0].message_id, "asst-1");
        assert_eq!(turns[0].boundary_kind, ForkBoundaryKind::Assistant);
        assert_eq!(turns[0].user_preview, "first prompt");
        assert_eq!(turns[0].assistant_preview, "first reply");

        assert_eq!(turns[1].turn_index, 2);
        assert_eq!(turns[1].message_id, "user-2");
        assert_eq!(turns[1].boundary_kind, ForkBoundaryKind::User);
        assert_eq!(turns[1].user_preview, "second prompt");
        assert_eq!(turns[1].assistant_preview, "");
    }

    #[test]
    fn latest_fork_boundary_selects_latest_eligible_turn() {
        let mut app = App::new();
        app.messages = vec![
            ChatEntry::User {
                text: "old".into(),
                message_id: Some("user-old".into()),
            },
            ChatEntry::Assistant {
                content: "old reply".into(),
                thinking: None,
                message_id: Some("asst-old".into()),
            },
            ChatEntry::User {
                text: "new".into(),
                message_id: None,
            },
            ChatEntry::Assistant {
                content: "new reply".into(),
                thinking: None,
                message_id: Some("asst-new".into()),
            },
        ];

        let latest = app.latest_fork_boundary().expect("latest fork boundary");
        assert_eq!(latest.message_id, "asst-new");
        assert_eq!(latest.boundary_kind, ForkBoundaryKind::Assistant);

        app.activity = ActivityState::Streaming;
        assert!(app.latest_fork_boundary().is_none());
    }

    #[test]
    fn current_undo_target_moves_left_of_frontier() {
        let mut app = App::new();
        app.undoable_turns = vec![make_turn("msg-1"), make_turn("msg-2"), make_turn("msg-3")];

        assert_eq!(
            app.current_undo_target()
                .map(|turn| turn.message_id.as_str()),
            Some("msg-3")
        );

        app.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-3".into(),
                message_id: "msg-3".into(),
                status: UndoFrameStatus::Confirmed,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-3".into()),
        });

        assert_eq!(
            app.current_undo_target()
                .map(|turn| turn.message_id.as_str()),
            Some("msg-2")
        );
    }

    #[test]
    fn build_undo_state_confirms_frames_and_preserves_frontier() {
        let mut app = App::new();
        app.undoable_turns = vec![make_turn("msg-1"), make_turn("msg-2")];
        app.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-1".into(),
                message_id: "msg-1".into(),
                status: UndoFrameStatus::Pending,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-1".into()),
        });

        let next = app
            .build_undo_state_from_server_stack(
                &make_stack(&["msg-1", "msg-2"]),
                Some("msg-2"),
                Some(&["a.rs".into(), "b.rs".into()]),
            )
            .expect("undo state");

        assert_eq!(next.frontier_message_id.as_deref(), Some("msg-2"));
        assert_eq!(next.stack.len(), 2);
        assert!(
            next.stack
                .iter()
                .all(|frame| frame.status == UndoFrameStatus::Confirmed)
        );
        assert_eq!(next.stack[1].turn_id, "turn-msg-2");
        assert_eq!(next.stack[1].reverted_files, vec!["a.rs", "b.rs"]);
    }

    #[test]
    fn build_undo_state_preserves_order_previous_turn_and_unknown_fallback() {
        let mut app = App::new();
        app.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "previous-turn".into(),
                message_id: "msg-1".into(),
                status: UndoFrameStatus::Pending,
                reverted_files: vec!["preserved.rs".into()],
            }],
            frontier_message_id: Some("msg-1".into()),
        });

        let state = app
            .build_undo_state_from_server_stack(&make_stack(&["msg-1", "unknown"]), None, None)
            .expect("undo state");

        assert_eq!(state.stack[0].message_id, "msg-1");
        assert_eq!(state.stack[0].turn_id, "previous-turn");
        assert_eq!(state.stack[0].status, UndoFrameStatus::Confirmed);
        assert_eq!(state.stack[0].reverted_files, ["preserved.rs"]);
        assert_eq!(state.stack[1].message_id, "unknown");
        assert_eq!(state.stack[1].turn_id, "unknown");
    }

    #[test]
    fn build_undo_state_returns_none_for_empty_stack() {
        let app = App::new();
        assert_eq!(
            app.build_undo_state_from_server_stack(&UndoStackSnapshot::default(), None, None),
            None
        );
    }

    #[test]
    fn pending_guard_tracks_pending_frames() {
        let mut app = App::new();
        let turn = make_turn("msg-1");
        app.push_pending_undo(&turn);

        assert!(app.has_pending_undo());
        assert_eq!(
            app.undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("msg-1")
        );
        assert_eq!(
            app.undo_state.as_ref().map(|state| state.stack.len()),
            Some(1)
        );
        assert_eq!(
            app.undo_state
                .as_ref()
                .map(|state| state.stack[0].status.clone()),
            Some(UndoFrameStatus::Pending)
        );
    }

    #[test]
    fn pending_session_label_stays_reserved_for_undo_and_redo() {
        let mut app = App::new();
        app.activity = ActivityState::Compacting {
            token_estimate: 9_000,
        };
        assert_eq!(app.pending_session_label(), None);

        app.activity = ActivityState::SessionOp(SessionOp::Undo);
        assert_eq!(app.pending_session_label(), Some("undoing"));
    }

    #[test]
    fn push_log_deduplicates_consecutive_entries() {
        let mut app = App::new();

        app.push_log(LogLevel::Info, "server", "starting local server");
        app.push_log(LogLevel::Info, "server", "starting local server");
        app.push_log(LogLevel::Warn, "server", "waiting for lock");

        assert_eq!(app.logs.len(), 2);
        assert_eq!(app.logs[0].level, LogLevel::Info);
        assert_eq!(app.logs[0].target, "server");
        assert_eq!(app.logs[0].message, "starting local server");
        assert_eq!(app.logs[1].level, LogLevel::Warn);
    }

    #[test]
    fn set_status_updates_visible_status_and_appends_log() {
        let mut app = App::new();

        app.set_status(LogLevel::Info, "connection", "connected");

        assert_eq!(app.status, "connected");
        let last = app.logs.last().expect("missing log entry");
        assert_eq!(last.level, LogLevel::Info);
        assert_eq!(last.target, "connection");
        assert_eq!(last.message, "connected");
    }

    #[test]
    fn filtered_logs_apply_level_threshold_and_text_filter() {
        let mut app = App::new();
        app.push_log(LogLevel::Debug, "activity", "ready");
        app.push_log(LogLevel::Warn, "server", "waiting for lock");
        app.push_log(LogLevel::Error, "server", "start failed");

        app.log_level_filter = LogLevel::Warn;
        let filtered = app.filtered_logs();
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|entry| entry.level >= LogLevel::Warn));

        app.log_filter = "failed".into();
        let filtered = app.filtered_logs();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].message, "start failed");
    }

    #[test]
    fn cancel_confirm_arms_expires_and_restores_status() {
        let mut app = App::new();
        app.activity = ActivityState::Thinking;

        app.arm_cancel_confirm();
        assert!(app.cancel_confirm_active());
        assert_eq!(app.status, "press Esc again to stop");
        assert!(
            matches!(app.logs.last(), Some(entry) if entry.message == "press Esc again to stop")
        );

        app.pending_cancel_confirm_until = Some(Instant::now() - Duration::from_millis(1));
        app.clear_expired_cancel_confirm();
        assert!(!app.cancel_confirm_active());
        assert_eq!(app.status, "thinking...");
        assert!(matches!(app.logs.last(), Some(entry) if entry.message == "thinking..."));
    }

    #[test]
    fn refresh_transient_status_preserves_connection_and_operation_precedence() {
        let mut app = App::new();
        app.conn = ConnState::Disconnected;
        app.set_status(LogLevel::Warn, "connection", "connection lost - retrying");
        app.refresh_transient_status();
        assert_eq!(app.status, "connection lost - retrying");

        app.conn = ConnState::Connected;
        app.activity = ActivityState::Thinking;
        app.refresh_transient_status();
        assert_eq!(app.status, "thinking...");

        app.activity = ActivityState::Compacting {
            token_estimate: 2048,
        };
        app.refresh_transient_status();
        assert_eq!(app.status, "compacting context (~2048 tokens)");

        app.activity = ActivityState::SessionOp(SessionOp::Redo);
        app.refresh_transient_status();
        assert_eq!(app.status, "redoing...");
    }

    #[test]
    fn session_stats_track_llm_request_elapsed_context_and_tool_calls_from_events() {
        let mut app = App::new();
        app.apply_event_stats(
            &EventKind::PromptReceived {
                content: serde_json::json!("hi"),
                message_id: None,
            },
            Some(100),
        );
        app.apply_event_stats(
            &EventKind::LlmRequestStart {
                message_count: Some(2),
            },
            Some(120),
        );
        app.apply_event_stats(
            &EventKind::ToolCallStart {
                tool_call_id: Some("call-1".into()),
                tool_name: "read_tool".into(),
                arguments: None,
            },
            Some(130),
        );
        app.apply_event_stats(
            &EventKind::LlmRequestEnd {
                finish_reason: None,
                cost_usd: None,
                cumulative_cost_usd: None,
                context_tokens: Some(2048),
                tool_calls: Some(99),
                metrics: None,
            },
            Some(160),
        );

        assert_eq!(app.session_stats.latest_context_tokens, Some(2048));
        assert_eq!(app.session_stats.total_tool_calls, 1);
        assert_eq!(app.llm_request_elapsed(), Some(Duration::from_secs(40)));
    }

    #[test]
    fn cancelled_closes_open_llm_request_span() {
        let mut app = App::new();
        app.apply_event_stats(
            &EventKind::LlmRequestStart {
                message_count: Some(1),
            },
            Some(200),
        );
        app.apply_event_stats(&EventKind::Cancelled, Some(215));
        assert_eq!(app.llm_request_elapsed(), Some(Duration::from_secs(15)));
        assert_eq!(app.session_stats.open_llm_request_ts, None);
        assert_eq!(app.session_stats.open_llm_request_instant, None);
    }

    #[test]
    fn active_mention_query_detects_trigger_and_ignores_email() {
        let app = App::new();

        assert_eq!(
            app.active_mention_query_from("fix @src/ma", "fix @src/ma".len()),
            Some((4, "src/ma".into()))
        );
        assert_eq!(
            app.active_mention_query_from("email@test.com", "email@test.com".len()),
            None
        );
        assert_eq!(
            app.active_mention_query_from("foo @", 5),
            Some((4, String::new()))
        );
        assert_eq!(
            app.active_mention_query_from("foo @bar baz", 8),
            Some((4, "bar".into()))
        );
        assert_eq!(app.active_mention_query_from("foo @bar baz", 12), None);
    }

    #[test]
    fn mention_results_rank_prefix_before_loose_matches() {
        let mut app = App::new();
        app.file_index = vec![
            FileIndexEntryLite {
                path: "src/main.rs".into(),
                is_dir: false,
            },
            FileIndexEntryLite {
                path: "tests/main_spec.rs".into(),
                is_dir: false,
            },
            FileIndexEntryLite {
                path: "src/manifest.toml".into(),
                is_dir: false,
            },
            FileIndexEntryLite {
                path: "src".into(),
                is_dir: true,
            },
        ];

        let results = app.rank_file_matches("ma");
        let ranked: Vec<&str> = results.iter().map(|entry| entry.path.as_str()).collect();
        assert_eq!(ranked[0], "src/main.rs");
        assert!(ranked.contains(&"src/manifest.toml"));
        assert!(ranked.contains(&"tests/main_spec.rs"));
    }

    #[test]
    fn input_up_visual_moves_to_previous_wrapped_row() {
        let mut app = App::new();
        app.input = "abcdef".into();
        app.input_cursor = 4;
        app.input_line_width = 4;

        app.input_up_visual(2);

        assert_eq!(app.input_cursor, 2);
        assert_eq!(app.input_preferred_col, Some(2));
    }

    #[test]
    fn input_down_visual_moves_to_next_wrapped_row() {
        let mut app = App::new();
        app.input = "abcdef".into();
        app.input_cursor = 2;
        app.input_line_width = 4;

        app.input_down_visual(2);

        assert_eq!(app.input_cursor, 4);
        assert_eq!(app.input_preferred_col, Some(2));
    }

    #[test]
    fn input_down_visual_crosses_newline_boundary() {
        let mut app = App::new();
        app.input = "ab\ncd".into();
        app.input_cursor = 1;
        app.input_line_width = 6;

        app.input_down_visual(2);

        assert_eq!(app.input_cursor, 4);
    }

    #[test]
    fn input_horizontal_move_resets_preferred_column() {
        let mut app = App::new();
        app.input = "abcdef".into();
        app.input_cursor = 4;
        app.input_preferred_col = Some(2);

        app.input_left();

        assert_eq!(app.input_cursor, 3);
        assert_eq!(app.input_preferred_col, None);
    }

    #[test]
    fn accept_selected_mention_replaces_query_with_friendly_token() {
        let mut app = App::new();
        app.input = "open @src/ma now".into();
        app.input_cursor = "open @src/ma".len();
        app.file_index = vec![FileIndexEntryLite {
            path: "src/main.rs".into(),
            is_dir: false,
        }];
        app.refresh_mention_state();

        let accepted = app.accept_selected_mention();
        assert!(accepted);
        assert_eq!(app.input, "open @src/main.rs  now");
        assert_eq!(app.input_cursor, "open @src/main.rs ".len());
        assert!(app.mention_state.is_none());
    }

    #[test]
    fn build_prompt_text_converts_friendly_mentions_to_markup_and_links() {
        let app = App::new();
        let (text, links) =
            app.build_prompt_text_and_links("check @src/main.rs and @src/lib.rs then @src/main.rs");
        assert_eq!(text, "check @src/main.rs and @src/lib.rs then @src/main.rs");
        assert_eq!(links, vec!["src/main.rs", "src/lib.rs"]);
    }

    #[test]
    fn activity_helpers_report_turn_and_session_state() {
        let mut app = App::new();
        assert!(!app.is_turn_active());
        assert!(!app.has_pending_session_op());
        assert!(!app.input_blocked_by_activity());
        assert!(!app.should_hide_input_contents());
        assert_eq!(app.pending_session_label(), None);

        app.activity = ActivityState::SessionOp(SessionOp::Undo);
        assert!(!app.is_turn_active());
        assert!(app.has_pending_session_op());
        assert!(app.input_blocked_by_activity());
        assert!(app.should_hide_input_contents());
        assert_eq!(app.pending_session_label(), Some("undoing"));

        app.activity = ActivityState::SessionOp(SessionOp::Redo);
        assert!(!app.is_turn_active());
        assert!(app.has_pending_session_op());
        assert!(app.input_blocked_by_activity());
        assert!(app.should_hide_input_contents());
        assert_eq!(app.pending_session_label(), Some("redoing"));

        app.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };
        assert!(app.is_turn_active());
        assert!(app.has_cancellable_activity());
        assert!(!app.has_pending_session_op());
        assert!(!app.input_blocked_by_activity());
        assert!(!app.should_hide_input_contents());
        assert_eq!(app.pending_session_label(), None);

        app.arm_cancel_confirm();
        assert!(app.input_blocked_by_activity());
        assert!(app.should_hide_input_contents());
    }

    #[test]
    fn connection_events_update_status_and_retry_metadata() {
        let mut app = App::new();
        app.handle_connection_event(ConnectionEvent::Connecting {
            attempt: 3,
            delay_ms: 2000,
        });
        assert_eq!(app.conn, ConnState::Connecting);
        assert_eq!(app.reconnect_attempt, 3);
        assert_eq!(app.reconnect_delay_ms, Some(2000));
        assert_eq!(app.status, "waiting for server - retry 3 in 2.0s");
        assert!(
            matches!(app.logs.last(), Some(entry) if entry.target == "connection" && entry.level == LogLevel::Warn)
        );

        app.handle_connection_event(ConnectionEvent::Disconnected {
            reason: "socket closed".into(),
        });
        assert_eq!(app.conn, ConnState::Disconnected);
        assert_eq!(app.reconnect_delay_ms, None);
        assert_eq!(app.status, "connection lost - socket closed");
        assert!(
            matches!(app.logs.last(), Some(entry) if entry.message == "connection lost - socket closed")
        );

        app.session_id = Some("session-1".into());
        app.handle_connection_event(ConnectionEvent::Connected);
        assert_eq!(app.conn, ConnState::Connected);
        assert_eq!(app.reconnect_attempt, 0);
        assert_eq!(app.reconnect_delay_ms, None);
        assert_eq!(app.status, "reconnected");
        assert!(
            matches!(app.logs.last(), Some(entry) if entry.level == LogLevel::Info && entry.message == "reconnected")
        );
    }

    // ── Elicitation: event handling ───────────────────────────────────────────

    // ── backfill_elicitation_outcomes ─────────────────────────────────────────

    #[test]
    fn backfill_single_answer_sets_outcome() {
        let mut messages = vec![ChatEntry::Elicitation {
            elicitation_id: "e1".into(),
            message: "Pick one".into(),
            source: "builtin:question".into(),
            outcome: Some("responded".into()),
        }];
        let result = r#"{"answers":[{"question":"Pick one","answers":["Beta"]}]}"#;
        backfill_elicitation_outcomes(&mut messages, result);
        assert!(matches!(&messages[0],
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}Beta")
        ));
    }

    #[test]
    fn backfill_multi_answer_joins_with_newline() {
        let mut messages = vec![ChatEntry::Elicitation {
            elicitation_id: "e1".into(),
            message: "Pick many".into(),
            source: "builtin:question".into(),
            outcome: Some("responded".into()),
        }];
        let result = r#"{"answers":[{"question":"Pick many","answers":["X","Z"]}]}"#;
        backfill_elicitation_outcomes(&mut messages, result);
        assert!(matches!(&messages[0],
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}X\n{OUTCOME_BULLET}Z")
        ));
    }

    #[test]
    fn backfill_multiple_questions_each_card_gets_its_own_answer() {
        let mut messages = vec![
            ChatEntry::Elicitation {
                elicitation_id: "e1".into(),
                message: "Q1".into(),
                source: "builtin:question".into(),
                outcome: Some("responded".into()),
            },
            ChatEntry::Elicitation {
                elicitation_id: "e2".into(),
                message: "Q2".into(),
                source: "builtin:question".into(),
                outcome: Some("responded".into()),
            },
        ];
        let result = r#"{"answers":[{"question":"Q1","answers":["Alpha"]},{"question":"Q2","answers":["Yes"]}]}"#;
        backfill_elicitation_outcomes(&mut messages, result);
        assert!(matches!(&messages[0],
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}Alpha")
        ));
        assert!(matches!(&messages[1],
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}Yes")
        ));
    }

    #[test]
    fn backfill_skips_already_resolved_cards() {
        let mut messages = vec![
            ChatEntry::Elicitation {
                elicitation_id: "e1".into(),
                message: "Q1".into(),
                source: "builtin:question".into(),
                outcome: Some(format!("{OUTCOME_BULLET}AlreadySet")),
            },
            ChatEntry::Elicitation {
                elicitation_id: "e2".into(),
                message: "Q2".into(),
                source: "builtin:question".into(),
                outcome: Some("responded".into()),
            },
        ];
        let result = r#"{"answers":[{"question":"Q2","answers":["Beta"]}]}"#;
        backfill_elicitation_outcomes(&mut messages, result);
        // First card unchanged
        assert!(matches!(&messages[0],
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}AlreadySet")
        ));
        // Second card updated
        assert!(matches!(&messages[1],
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}Beta")
        ));
    }
}

// ── Start-page session grouping tests ─────────────────────────────────────────

#[cfg(test)]
mod start_page_tests {
    use super::*;
    use crate::domain::session::SessionSummary;

    fn make_group(cwd: Option<&str>, ids: &[(&str, Option<&str>)]) -> SessionGroup {
        SessionGroup {
            cwd: cwd.map(String::from),
            latest_activity: None,
            sessions: ids
                .iter()
                .map(|(id, updated_at)| SessionSummary {
                    session_id: id.to_string(),
                    title: Some(format!("Session {id}")),
                    cwd: cwd.map(String::from),
                    created_at: None,
                    updated_at: updated_at.map(String::from),
                    parent_session_id: None,
                    has_children: false,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    // ── visible_start_items: no sessions ─────────────────────────────────────

    #[test]
    fn visible_items_empty_when_no_sessions() {
        let app = App::new();
        let items = app.visible_start_items();
        assert!(items.is_empty());
    }

    // ── visible_start_items: basic structure ─────────────────────────────────

    #[test]
    fn visible_items_header_then_sessions_expanded() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &[("s1", None), ("s2", None)])];

        let items = app.visible_start_items();
        // 1 header + 2 sessions
        assert_eq!(items.len(), 3);
        assert!(matches!(&items[0], StartPageItem::GroupHeader { .. }));
        assert!(matches!(&items[1], StartPageItem::Session { .. }));
        assert!(matches!(&items[2], StartPageItem::Session { .. }));
    }

    // ── visible_start_items: collapse hides children ─────────────────────────

    #[test]
    fn visible_items_collapsed_group_hides_sessions() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &[("s1", None), ("s2", None)])];
        app.collapsed_groups.insert("/a".to_string());

        let items = app.visible_start_items();
        // only the header
        assert_eq!(items.len(), 1);
        assert!(matches!(
            &items[0],
            StartPageItem::GroupHeader {
                collapsed: true,
                ..
            }
        ));
    }

    // ── visible_start_items: multiple groups ─────────────────────────────────

    #[test]
    fn visible_items_multiple_groups() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None), ("s3", None)]),
        ];

        let items = app.visible_start_items();
        // group /a: 1 header + 1 session = 2
        // group /b: 1 header + 2 sessions = 3
        assert_eq!(items.len(), 5);
    }

    // ── visible_start_items: mixed collapse ───────────────────────────────────

    #[test]
    fn visible_items_one_group_collapsed_other_expanded() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None), ("s3", None)]),
        ];
        app.collapsed_groups.insert("/a".to_string());

        let items = app.visible_start_items();
        // /a collapsed: 1 header
        // /b expanded:  1 header + 2 sessions
        assert_eq!(items.len(), 4);
        assert!(matches!(
            &items[0],
            StartPageItem::GroupHeader {
                collapsed: true,
                ..
            }
        ));
        assert!(matches!(
            &items[1],
            StartPageItem::GroupHeader {
                collapsed: false,
                ..
            }
        ));
    }

    // ── visible_start_items: filter hides non-matching sessions ──────────────

    #[test]
    fn visible_items_filter_hides_non_matching_sessions() {
        let mut app = App::new();
        app.session_groups = vec![make_group(
            Some("/a"),
            &[("aaa", None), ("bbb", None), ("aab", None)],
        )];
        app.session_filter = "aa".to_string();

        let items = app.visible_start_items();
        // header + "aaa" + "aab" (bbb filtered out by session_id)
        assert_eq!(items.len(), 3);
    }

    // ── visible_start_items: filter hides empty groups ────────────────────────

    #[test]
    fn visible_items_filter_hides_groups_with_no_matches() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("aaa", None)]),
            make_group(Some("/b"), &[("bbb", None)]),
        ];
        app.session_filter = "bbb".to_string();

        let items = app.visible_start_items();
        // group /a has no matches → hidden entirely
        // group /b: header + "bbb"
        assert_eq!(items.len(), 2);
        if let StartPageItem::GroupHeader { cwd, .. } = &items[0] {
            assert_eq!(cwd.as_deref(), Some("/b"));
        } else {
            panic!("expected GroupHeader");
        }
    }

    // ── visible_start_items: session indices are correct ─────────────────────

    #[test]
    fn visible_items_session_indices_correct() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("s0", None), ("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
        ];

        let items = app.visible_start_items();
        // items[0]: GroupHeader /a
        // items[1]: Session group_idx=0, session_idx=0
        // items[2]: Session group_idx=0, session_idx=1
        // items[3]: GroupHeader /b
        // items[4]: Session group_idx=1, session_idx=0
        assert!(matches!(
            &items[1],
            StartPageItem::Session {
                group_idx: 0,
                path,
                depth: 0,
            } if path == &vec![0]
        ));
        assert!(matches!(
            &items[2],
            StartPageItem::Session {
                group_idx: 0,
                path,
                depth: 0,
            } if path == &vec![1]
        ));
        assert!(matches!(
            &items[4],
            StartPageItem::Session {
                group_idx: 1,
                path,
                depth: 0,
            } if path == &vec![0]
        ));
    }

    #[test]
    fn toggle_session_children_only_signals_load_without_marking_pending() {
        let mut app = App::new();
        let mut group = make_group(Some("/a"), &[("root", Some("2024-01-04T00:00:00Z"))]);
        group.sessions[0].fork_count = 1;
        app.session_groups = vec![group];
        app.session_cursor = 1;

        let action = crate::handlers::apply_session_fork_toggle_key(&mut app, false);
        assert_eq!(
            action,
            crate::handlers::SessionKeyAction::LoadMoreSessions {
                group_idx: 0,
                parent_path: vec![0],
            }
        );
        assert!(app.expanded_session_children.contains("root"));
        assert!(app.pending_session_child_loads.is_empty());

        let action = crate::handlers::apply_session_fork_toggle_key(&mut app, false);
        assert_eq!(action, crate::handlers::SessionKeyAction::None);
        assert!(app.pending_session_child_loads.is_empty());
    }

    #[test]
    fn remote_node_sessions_are_not_expandable() {
        let mut app = App::new();
        let mut group = make_group(Some("/a"), &[("remote", Some("2024-01-04T00:00:00Z"))]);
        group.sessions[0].fork_count = 1;
        group.sessions[0].node = Some("remote-host".to_string());
        app.session_groups = vec![group];

        assert!(!app.expandable_root_session(0, &[0]));
        assert!(!app.toggle_session_children(0, &[0]));
        assert!(app.expanded_session_children.is_empty());
        assert!(app.pending_session_child_loads.is_empty());
    }

    #[test]
    fn expanded_root_children_are_visible_with_load_more_row() {
        let mut app = App::new();
        let mut group = make_group(Some("/a"), &[("root", None)]);
        group.sessions[0].fork_count = 2;
        group.sessions[0].children_next_cursor = Some("next".to_string());
        group.sessions[0].children = vec![SessionSummary {
            session_id: "child".to_string(),
            title: Some("Child".to_string()),
            parent_session_id: Some("root".to_string()),
            ..Default::default()
        }];
        app.expanded_session_children.insert("root".to_string());
        app.session_groups = vec![group];

        let items = app.visible_popup_items();

        assert!(matches!(
            &items[2],
            PopupItem::Session {
                group_idx: 0,
                path,
                depth: 1,
            } if path == &vec![0, 0]
        ));
        assert!(matches!(
            &items[3],
            PopupItem::LoadMore {
                group_idx: 0,
                parent_path,
            } if parent_path == &vec![0]
        ));
    }

    // ── filtered_sessions still works (for popup compat) ─────────────────────

    #[test]
    fn filtered_sessions_returns_flat_list_for_popup() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None), ("s3", None)]),
        ];

        let flat = app.filtered_sessions();
        assert_eq!(flat.len(), 3);
    }

    #[test]
    fn filtered_sessions_applies_filter() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &[("aaa", None), ("bbb", None)])];
        app.session_filter = "aaa".to_string();

        let flat = app.filtered_sessions();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].session_id, "aaa");
    }

    // ── GroupHeader carries correct session_count ─────────────────────────────

    #[test]
    fn group_header_session_count_reflects_total_not_filtered() {
        let mut app = App::new();
        app.session_groups = vec![make_group(
            Some("/a"),
            &[("s1", None), ("s2", None), ("s3", None)],
        )];
        app.session_groups[0].total_count = Some(8);

        let items = app.visible_start_items();
        assert!(matches!(
            &items[0],
            StartPageItem::GroupHeader {
                session_count: 3,
                session_total: Some(8),
                ..
            }
        ));
    }

    #[test]
    fn group_header_session_total_falls_back_to_unknown() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &[("s1", None)])];

        let items = app.visible_start_items();
        assert!(matches!(
            &items[0],
            StartPageItem::GroupHeader {
                session_count: 1,
                session_total: None,
                ..
            }
        ));
    }

    // ── toggle_group_collapse ─────────────────────────────────────────────────

    #[test]
    fn toggle_group_collapse_collapses_then_expands() {
        let mut app = App::new();
        let key = "/a".to_string();
        assert!(!app.collapsed_groups.contains(&key));

        app.toggle_group_collapse(Some("/a"));
        assert!(app.collapsed_groups.contains(&key));

        app.toggle_group_collapse(Some("/a"));
        assert!(!app.collapsed_groups.contains(&key));
    }

    #[test]
    fn toggle_group_collapse_none_cwd_uses_empty_string_key() {
        let mut app = App::new();
        app.toggle_group_collapse(None);
        assert!(app.collapsed_groups.contains(""));

        app.toggle_group_collapse(None);
        assert!(!app.collapsed_groups.contains(""));
    }

    // ── MAX_RECENT_SESSIONS cap ───────────────────────────────────────────────

    #[test]
    fn visible_items_group_with_three_sessions_shows_no_show_more() {
        let mut app = App::new();
        app.session_groups = vec![make_group(
            Some("/a"),
            &[("s1", None), ("s2", None), ("s3", None)],
        )];
        let items = app.visible_start_items();
        // header + 3 sessions, no ShowMore
        assert_eq!(items.len(), 4);
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, StartPageItem::ShowMore { .. }))
        );
    }

    #[test]
    fn visible_items_group_with_four_sessions_shows_show_more() {
        let mut app = App::new();
        app.session_groups = vec![make_group(
            Some("/a"),
            &[("s1", None), ("s2", None), ("s3", None), ("s4", None)],
        )];
        let items = app.visible_start_items();
        // header + 3 sessions + ShowMore
        assert_eq!(items.len(), 5);
        assert!(matches!(
            items.last(),
            Some(StartPageItem::ShowMore { remaining: 1, .. })
        ));
    }

    #[test]
    fn visible_items_show_more_remaining_is_total_minus_three() {
        let mut app = App::new();
        app.session_groups = vec![make_group(
            Some("/a"),
            &[
                ("s1", None),
                ("s2", None),
                ("s3", None),
                ("s4", None),
                ("s5", None),
                ("s6", None),
                ("s7", None),
                ("s8", None),
                ("s9", None),
                ("s10", None),
                ("s11", None),
            ],
        )];
        let items = app.visible_start_items();
        assert!(matches!(
            items.last(),
            Some(StartPageItem::ShowMore {
                remaining: 8,
                has_more: false,
                ..
            })
        ));
    }

    #[test]
    fn visible_items_filter_active_still_caps_sessions() {
        let mut app = App::new();
        app.session_groups = vec![make_group(
            Some("/a"),
            &[
                ("aaa1", None),
                ("aaa2", None),
                ("aaa3", None),
                ("aaa4", None),
                ("aaa5", None),
                ("aaa6", None),
                ("aaa7", None),
                ("aaa8", None),
                ("aaa9", None),
                ("aaa10", None),
                ("aaa11", None),
            ],
        )];
        app.session_filter = "aaa".to_string();
        let items = app.visible_start_items();
        assert_eq!(items.len(), 5);
        assert!(matches!(
            items.last(),
            Some(StartPageItem::ShowMore { remaining: 8, .. })
        ));
    }

    // ── MAX_VISIBLE_GROUPS cap ────────────────────────────────────────────────

    #[test]
    fn visible_items_three_groups_shows_no_trailing_show_more() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
            make_group(Some("/c"), &[("s3", None)]),
        ];
        let items = app.visible_start_items();
        // 3 headers + 3 sessions = 6, no trailing ShowMore
        assert_eq!(items.len(), 6);
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, StartPageItem::ShowMore { .. }))
        );
    }

    #[test]
    fn visible_items_four_groups_caps_at_three_no_trailing_show_more() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
            make_group(Some("/c"), &[("s3", None)]),
            make_group(Some("/d"), &[("s4", None)]),
        ];
        let items = app.visible_start_items();
        // 3 groups (3 headers + 3 sessions) = 6, no trailing ShowMore
        assert_eq!(items.len(), 6);
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, StartPageItem::ShowMore { .. }))
        );
    }

    #[test]
    fn visible_items_six_groups_caps_at_three_no_trailing_show_more() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
            make_group(Some("/c"), &[("s3", None)]),
            make_group(Some("/d"), &[("s4", None)]),
            make_group(Some("/e"), &[("s5", None)]),
            make_group(Some("/f"), &[("s6", None)]),
        ];
        let items = app.visible_start_items();
        // 3 shown groups (3 headers + 3 sessions) = 6, no trailing ShowMore
        assert_eq!(items.len(), 6);
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, StartPageItem::ShowMore { .. }))
        );
    }

    #[test]
    fn visible_items_group_cap_applied_with_filter_active() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &[("aaa1", None)]),
            make_group(Some("/b"), &[("aaa2", None)]),
            make_group(Some("/c"), &[("aaa3", None)]),
            make_group(Some("/d"), &[("aaa4", None)]),
        ];
        app.session_filter = "aaa".to_string();
        let items = app.visible_start_items();
        // Filter active but group cap still applies → 3 groups, no trailing ShowMore
        let headers = items
            .iter()
            .filter(|i| matches!(i, StartPageItem::GroupHeader { .. }))
            .count();
        assert_eq!(headers, 3);
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, StartPageItem::ShowMore { .. }))
        );
    }
}

// ── popup_item_tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod popup_item_tests {
    use super::*;
    use crate::domain::session::SessionSummary;

    fn make_group(cwd: Option<&str>, ids: &[&str]) -> SessionGroup {
        SessionGroup {
            cwd: cwd.map(String::from),
            latest_activity: None,
            sessions: ids
                .iter()
                .map(|id| SessionSummary {
                    session_id: id.to_string(),
                    title: Some(format!("Session {id}")),
                    cwd: cwd.map(String::from),
                    created_at: None,
                    updated_at: None,
                    parent_session_id: None,
                    has_children: false,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    // ── empty state ───────────────────────────────────────────────────────────

    #[test]
    fn popup_items_empty_when_no_sessions() {
        let app = App::new();
        assert!(app.visible_popup_items().is_empty());
    }

    // ── basic structure: header then sessions ─────────────────────────────────

    #[test]
    fn popup_items_header_then_sessions() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        let items = app.visible_popup_items();
        // 1 header + 2 sessions
        assert_eq!(items.len(), 3);
        assert!(matches!(&items[0], PopupItem::GroupHeader { .. }));
        assert!(matches!(&items[1], PopupItem::Session { .. }));
        assert!(matches!(&items[2], PopupItem::Session { .. }));
    }

    // ── no MAX_RECENT_SESSIONS cap ────────────────────────────────────────────

    #[test]
    fn popup_items_shows_all_sessions_beyond_cap() {
        let mut app = App::new();
        // 10 sessions - all should appear, no cap like start page
        let ids: Vec<&str> = vec!["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"];
        app.session_groups = vec![make_group(Some("/a"), &ids)];
        let items = app.visible_popup_items();
        // 1 header + 10 sessions = 11
        assert_eq!(items.len(), 11);
        assert!(
            !items
                .iter()
                .any(|i| matches!(i, PopupItem::LoadMore { .. }))
        );
    }

    #[test]
    fn popup_items_include_load_more_when_group_has_next_cursor() {
        let mut app = App::new();
        let mut group = make_group(Some("/workspace/project"), &["s1"]);
        group.next_cursor = Some("cursor-1".to_string());
        app.session_groups = vec![group];

        let items = app.visible_popup_items();

        assert!(matches!(
            items.last(),
            Some(PopupItem::LoadMore {
                group_idx: 0,
                parent_path,
            }) if parent_path.is_empty()
        ));
    }

    // ── no MAX_VISIBLE_GROUPS cap ─────────────────────────────────────────────

    #[test]
    fn popup_items_hide_group_load_more_while_request_is_pending() {
        let mut app = App::new();
        let mut group = make_group(Some("/workspace/project"), &["s1"]);
        group.next_cursor = Some("opaque-workspace-2".to_string());
        app.session_groups = vec![group];
        app.pending_session_group_loads
            .insert(Some("/workspace/project".to_string()));

        assert!(!app.visible_popup_items().iter().any(
            |item| matches!(item, PopupItem::LoadMore { parent_path, .. } if parent_path.is_empty())
        ));
    }

    #[test]
    fn popup_items_shows_all_groups_beyond_cap() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &["s1"]),
            make_group(Some("/b"), &["s2"]),
            make_group(Some("/c"), &["s3"]),
            make_group(Some("/d"), &["s4"]),
            make_group(Some("/e"), &["s5"]),
        ];
        let items = app.visible_popup_items();
        let headers = items
            .iter()
            .filter(|i| matches!(i, PopupItem::GroupHeader { .. }))
            .count();
        // All 5 groups shown (start page would cap at MAX_VISIBLE_GROUPS=3)
        assert_eq!(headers, 5);
    }

    // ── collapse is separate from start page ──────────────────────────────────

    #[test]
    fn popup_collapsed_is_independent_of_start_page_collapsed() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        // Collapse on the start page should NOT affect the popup
        app.collapsed_groups.insert("/a".to_string());
        let items = app.visible_popup_items();
        // Popup uses popup_collapsed_groups, not collapsed_groups
        // /a is expanded in popup → header + 2 sessions = 3
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn popup_collapsed_hides_sessions() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.popup_collapsed_groups.insert("/a".to_string());
        let items = app.visible_popup_items();
        // Only the header visible
        assert_eq!(items.len(), 1);
        assert!(matches!(
            &items[0],
            PopupItem::GroupHeader {
                collapsed: true,
                ..
            }
        ));
    }

    #[test]
    fn popup_expanded_shows_sessions() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        // Not in popup_collapsed_groups → expanded
        let items = app.visible_popup_items();
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0],
            PopupItem::GroupHeader {
                collapsed: false,
                ..
            }
        ));
    }

    // ── multiple groups, mixed collapse ───────────────────────────────────────

    #[test]
    fn popup_items_multiple_groups() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &["s1"]),
            make_group(Some("/b"), &["s2", "s3"]),
        ];
        let items = app.visible_popup_items();
        // /a: 1 header + 1 session; /b: 1 header + 2 sessions = 5
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn popup_one_group_collapsed_other_expanded() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &["s1"]),
            make_group(Some("/b"), &["s2", "s3"]),
        ];
        app.popup_collapsed_groups.insert("/a".to_string());
        let items = app.visible_popup_items();
        // /a collapsed: 1 header; /b expanded: 1 header + 2 sessions = 4
        assert_eq!(items.len(), 4);
        assert!(matches!(
            &items[0],
            PopupItem::GroupHeader {
                collapsed: true,
                ..
            }
        ));
        assert!(matches!(
            &items[1],
            PopupItem::GroupHeader {
                collapsed: false,
                ..
            }
        ));
    }

    // ── filter hides non-matching sessions ────────────────────────────────────

    #[test]
    fn popup_filter_hides_non_matching_sessions() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["aaa", "bbb", "aab"])];
        app.session_filter = "aa".to_string();
        let items = app.visible_popup_items();
        // header + "aaa" + "aab" (bbb filtered out by session_id)
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn popup_filter_hides_groups_with_no_matches() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &["aaa"]),
            make_group(Some("/b"), &["bbb"]),
        ];
        app.session_filter = "bbb".to_string();
        let items = app.visible_popup_items();
        // /a has no matches → hidden; /b: header + "bbb" = 2
        assert_eq!(items.len(), 2);
        if let PopupItem::GroupHeader { cwd, .. } = &items[0] {
            assert_eq!(cwd.as_deref(), Some("/b"));
        } else {
            panic!("expected GroupHeader");
        }
    }

    // ── session indices are correct ───────────────────────────────────────────

    #[test]
    fn popup_items_session_indices_correct() {
        let mut app = App::new();
        app.session_groups = vec![
            make_group(Some("/a"), &["s0", "s1"]),
            make_group(Some("/b"), &["s2"]),
        ];
        let items = app.visible_popup_items();
        // items[0]: GroupHeader /a
        // items[1]: Session group_idx=0, session_idx=0
        // items[2]: Session group_idx=0, session_idx=1
        // items[3]: GroupHeader /b
        // items[4]: Session group_idx=1, session_idx=0
        assert!(matches!(
            &items[1],
            PopupItem::Session {
                group_idx: 0,
                path,
                depth: 0,
            } if path == &vec![0]
        ));
        assert!(matches!(
            &items[2],
            PopupItem::Session {
                group_idx: 0,
                path,
                depth: 0,
            } if path == &vec![1]
        ));
        assert!(matches!(
            &items[4],
            PopupItem::Session {
                group_idx: 1,
                path,
                depth: 0,
            } if path == &vec![0]
        ));
    }

    // ── group header carries correct session_count ────────────────────────────

    #[test]
    fn popup_group_header_session_count_reflects_total() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3"])];
        app.session_groups[0].total_count = Some(8);
        let items = app.visible_popup_items();
        assert!(matches!(
            &items[0],
            PopupItem::GroupHeader {
                session_count: 3,
                session_total: Some(8),
                ..
            }
        ));
    }

    // ── toggle_popup_group_collapse ───────────────────────────────────────────

    #[test]
    fn toggle_popup_collapse_collapses_then_expands() {
        let mut app = App::new();
        assert!(!app.popup_collapsed_groups.contains("/a"));
        app.toggle_popup_group_collapse(Some("/a"));
        assert!(app.popup_collapsed_groups.contains("/a"));
        app.toggle_popup_group_collapse(Some("/a"));
        assert!(!app.popup_collapsed_groups.contains("/a"));
    }

    #[test]
    fn toggle_popup_collapse_none_cwd_uses_empty_string_key() {
        let mut app = App::new();
        app.toggle_popup_group_collapse(None);
        assert!(app.popup_collapsed_groups.contains(""));
        app.toggle_popup_group_collapse(None);
        assert!(!app.popup_collapsed_groups.contains(""));
    }

    #[test]
    fn toggle_popup_collapse_does_not_affect_start_page_state() {
        let mut app = App::new();
        app.toggle_popup_group_collapse(Some("/a"));
        assert!(app.popup_collapsed_groups.contains("/a"));
        // start page collapsed_groups should be untouched
        assert!(!app.collapsed_groups.contains("/a"));
    }

    // ── command palette ───────────────────────────────────────────────────────

    #[test]
    fn command_palette_filters_commands_and_hides_chat_only_outside_chat() {
        let mut app = App::new();
        app.screen = Screen::Sessions;
        app.command_palette_filter = "selector".into();

        let titles: Vec<&str> = app
            .filtered_command_palette_commands()
            .iter()
            .map(|command| command.title)
            .collect();

        assert_eq!(titles, vec!["Profile selector"]);
    }

    #[test]
    fn command_palette_includes_chat_only_commands_in_chat() {
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.command_palette_filter = "model".into();

        let actions: Vec<CommandPaletteAction> = app
            .filtered_command_palette_commands()
            .iter()
            .map(|command| command.action)
            .collect();

        assert!(actions.contains(&CommandPaletteAction::ModelSelect));
    }

    #[test]
    fn command_palette_cursor_wraps() {
        let mut app = App::new();
        app.screen = Screen::Chat;

        app.move_command_palette_cursor(-1);

        assert_eq!(
            app.command_palette_cursor,
            app.filtered_command_palette_commands().len() - 1
        );
    }

    // ── slash completion state ─────────────────────────────────────────────────

    #[test]
    fn slash_query_cursor_at_zero_returns_none() {
        let mut app = App::new();
        app.input = "/model".into();
        app.input_cursor = 0;
        assert_eq!(app.active_slash_query(), None);
    }

    #[test]
    fn slash_query_only_slash_typed() {
        let mut app = App::new();
        app.input = "/".into();
        app.input_cursor = 1;
        assert_eq!(app.active_slash_query(), Some(String::new()));
    }

    #[test]
    fn slash_query_partial_command() {
        let mut app = App::new();
        app.input = "/mo".into();
        app.input_cursor = 3;
        assert_eq!(app.active_slash_query(), Some("mo".into()));
    }

    #[test]
    fn slash_query_full_command_no_space() {
        let mut app = App::new();
        app.input = "/model".into();
        app.input_cursor = 6;
        assert_eq!(app.active_slash_query(), Some("model".into()));
    }

    #[test]
    fn slash_query_after_space_returns_none() {
        let mut app = App::new();
        app.input = "/model ".into();
        app.input_cursor = 7;
        assert_eq!(app.active_slash_query(), None);
    }

    #[test]
    fn slash_query_non_slash_input_returns_none() {
        let mut app = App::new();
        app.input = "hello".into();
        app.input_cursor = 5;
        assert_eq!(app.active_slash_query(), None);
    }

    #[test]
    fn refresh_slash_state_filters_by_prefix() {
        let mut app = App::new();
        app.input = "/mo".into();
        app.input_cursor = 3;
        app.refresh_slash_state();
        let state = app
            .slash_state
            .as_ref()
            .expect("slash_state should be Some");
        assert!(!state.results.is_empty());
        assert!(state.results.iter().all(|c| c.name.starts_with("mo")));
        assert!(state.results.iter().any(|c| c.name == "model"));
    }

    #[test]
    fn refresh_slash_state_clears_on_no_match() {
        let mut app = App::new();
        app.input = "/zzz".into();
        app.input_cursor = 4;
        app.refresh_slash_state();
        assert!(app.slash_state.is_none());
    }

    #[test]
    fn refresh_slash_state_clears_on_cursor_at_zero() {
        let mut app = App::new();
        app.input = "/model".into();
        app.input_cursor = 0;
        app.refresh_slash_state(); // must not panic
        assert!(app.slash_state.is_none());
    }

    #[test]
    fn move_slash_selection_wraps_down_to_first() {
        let mut app = App::new();
        app.input = "/".into();
        app.input_cursor = 1;
        app.refresh_slash_state();
        let total = app.slash_state.as_ref().unwrap().results.len();
        app.slash_state.as_mut().unwrap().selected_index = total - 1;
        app.move_slash_selection(1);
        assert_eq!(app.slash_state.as_ref().unwrap().selected_index, 0);
    }

    #[test]
    fn move_slash_selection_wraps_up_to_last() {
        let mut app = App::new();
        app.input = "/".into();
        app.input_cursor = 1;
        app.refresh_slash_state();
        let total = app.slash_state.as_ref().unwrap().results.len();
        app.slash_state.as_mut().unwrap().selected_index = 0;
        app.move_slash_selection(-1);
        assert_eq!(app.slash_state.as_ref().unwrap().selected_index, total - 1);
    }

    #[test]
    fn accept_slash_completion_replaces_partial_token() {
        let mut app = App::new();
        app.input = "/mo".into();
        app.input_cursor = 3;
        app.refresh_slash_state();
        let idx = app
            .slash_state
            .as_ref()
            .unwrap()
            .results
            .iter()
            .position(|c| c.name == "model")
            .expect("model should be in results");
        app.slash_state.as_mut().unwrap().selected_index = idx;

        let accepted = app.accept_selected_slash_completion();
        assert!(accepted);
        assert_eq!(app.input, "/model ");
        assert_eq!(app.input_cursor, "/model ".len());
        assert!(app.slash_state.is_none());
    }

    #[test]
    fn accept_slash_completion_no_state_returns_false() {
        let mut app = App::new();
        app.input = "/model".into();
        app.input_cursor = 6;
        app.slash_state = None;
        assert!(!app.accept_selected_slash_completion());
    }
}

// ── delegate_model_preference_tests ──────────────────────────────────────────

#[cfg(test)]
mod delegate_model_preference_tests {
    use super::*;
    use crate::domain::model::ModelEntry;
    use crate::domain::profile::AgentInfo;

    fn make_agent(id: &str, name: &str) -> AgentInfo {
        AgentInfo {
            id: id.into(),
            name: name.into(),
            description: None,
            capabilities: Vec::new(),
        }
    }

    fn make_model(provider: &str, model: &str) -> ModelEntry {
        ModelEntry {
            id: format!("{provider}/{model}"),
            label: format!("{provider}/{model}"),
            provider: provider.into(),
            model: model.into(),
            node_id: None,
            node_label: None,
            family: None,
            quant: None,
        }
    }

    #[test]
    fn is_multi_agent_false_when_no_agents() {
        let app = App::new();
        assert!(!app.is_multi_agent());
    }

    #[test]
    fn is_multi_agent_false_when_single_agent() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main")];
        assert!(!app.is_multi_agent());
    }

    #[test]
    fn is_multi_agent_true_when_two_agents() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        assert!(app.is_multi_agent());
    }

    #[test]
    fn tab_label_session_at_zero() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_label(0), "session");
    }

    #[test]
    fn tab_label_delegate_agent_at_one_when_multi_agent() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        assert_eq!(app.model_popup_tab_label(1), "Coder");
    }

    #[test]
    fn tab_agent_id_none_on_session_tab() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_agent_id(0), None);
    }

    #[test]
    fn tab_agent_id_some_for_delegate_tab() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        assert_eq!(app.model_popup_tab_agent_id(1), Some("coder"));
    }

    #[test]
    fn tab_count_single_agent_or_empty() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_count(), 1);
        let mut single = App::new();
        single.agents = vec![make_agent("main", "Main")];
        assert_eq!(single.model_popup_tab_count(), 1);
    }

    #[test]
    fn tab_count_multi_agent_matches_agent_list() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        assert_eq!(app.model_popup_tab_count(), 2);
        assert!(app.model_popup_has_tabs());
        app.agents.push(make_agent("reviewer", "Reviewer"));
        assert_eq!(app.model_popup_tab_count(), 3);
        assert!(app.model_popup_has_tabs());
    }

    #[test]
    fn delegate_pref_round_trip() {
        let mut app = App::new();
        let model = make_model("anthropic", "claude-sonnet");
        app.set_delegate_model_preference("profile", "coder", &model);
        assert_eq!(
            app.get_delegate_model_preference("profile", "coder")
                .map(|preference| preference.model_id.as_str()),
            Some("anthropic/claude-sonnet")
        );
    }

    #[test]
    fn delegate_pref_missing_returns_none() {
        let app = App::new();
        assert_eq!(app.get_delegate_model_preference("profile", "coder"), None);
    }

    #[test]
    fn delegate_model_cursor_with_preference() {
        let mut app = App::new();
        app.models = vec![
            make_model("openai", "gpt-4o"),
            make_model("anthropic", "claude-sonnet"),
        ];
        app.active_profile_id = Some("profile".into());
        app.set_delegate_model_preference("profile", "coder", &app.models[1].clone());
        let cursor = app.delegate_model_cursor("coder");
        // Should point to the second item (index 1 in models, but popup items
        // include provider headers — exact index depends on visible_model_popup_items).
        let items = app.visible_model_popup_items();
        match &items[cursor] {
            ModelPopupItem::Model { model_idx } => {
                assert_eq!(app.models[*model_idx].model, "claude-sonnet");
            }
            _ => panic!("expected Model item at cursor"),
        }
    }

    #[test]
    fn delegate_model_cursor_without_preference() {
        let mut app = App::new();
        app.models = vec![
            make_model("openai", "gpt-4o"),
            make_model("anthropic", "claude-sonnet"),
        ];
        let cursor = app.delegate_model_cursor("coder");
        // Should land on the first Model item.
        let items = app.visible_model_popup_items();
        assert!(matches!(&items[cursor], ModelPopupItem::Model { .. }));
    }
}
