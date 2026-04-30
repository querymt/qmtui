use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use ratatui::text::Line;

use crate::highlight::Highlighter;
use crate::markdown::CardBlock;
use crate::protocol::*;
use crate::ui::{CardCache, OUTCOME_BULLET};

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
    ModelSelect,
    SessionSelect,
    NewSession,
    ThemeSelect,
    Help,
    Log,
    ProviderAuth,
}

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

/// Per-delegation stats accumulated from child session events.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DelegateStats {
    pub tool_calls: u32,
    pub messages: u32,
    /// Cumulative cost in USD from LlmRequestEnd events.
    pub cost_usd: f64,
    /// Latest context token count (last LlmRequestEnd wins).
    pub context_tokens: u64,
    /// Context limit from ProviderChanged events.
    pub context_limit: u64,
}

impl DelegateStats {
    pub fn context_pct(&self) -> Option<u32> {
        if self.context_limit > 0 {
            Some(
                ((self.context_tokens as f64 / self.context_limit as f64) * 100.0)
                    .round()
                    .min(100.0) as u32,
            )
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DelegateChildState {
    #[default]
    None,
    PendingElicitation {
        elicitation_id: String,
        message: String,
        requested_schema: serde_json::Value,
        source: String,
    },
    QuestionToolFinished,
    AssistantMessage,
    UserMessage,
    OtherProgress,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DelegateEntry {
    pub delegation_id: String,
    pub child_session_id: Option<String>,
    /// Parent delegate tool call this row renders for, when known.
    pub delegate_tool_call_id: Option<String>,
    pub target_agent_id: Option<String>,
    pub objective: String,
    pub status: DelegateStatus,
    pub stats: DelegateStats,
    /// Server timestamp (unix seconds) when delegation was requested.
    pub started_at: Option<i64>,
    /// Server timestamp (unix seconds) when delegation completed/failed/cancelled.
    pub ended_at: Option<i64>,
    /// Compact state derived from the latest significant child-session event.
    pub child_state: DelegateChildState,
}

impl DelegateEntry {
    pub fn awaiting_input(&self) -> bool {
        self.status == DelegateStatus::InProgress
            && matches!(
                self.child_state,
                DelegateChildState::PendingElicitation { .. }
            )
    }

    pub fn pending_elicitation(&self) -> Option<(&str, &str, &str)> {
        match &self.child_state {
            DelegateChildState::PendingElicitation {
                elicitation_id,
                message,
                source,
                ..
            } => Some((elicitation_id.as_str(), message.as_str(), source.as_str())),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegateStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingDelegateToolCall {
    pub tool_call_id: String,
    pub target_agent_id: Option<String>,
    pub objective: String,
}

#[derive(Debug, Clone)]
pub enum ChatEntry {
    User {
        text: String,
        message_id: Option<String>,
    },
    Assistant {
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    },
    ToolCall {
        tool_call_id: Option<String>,
        name: String,
        is_error: bool,
        detail: ToolDetail,
    },
    CompactionStart {
        token_estimate: u32,
    },
    CompactionEnd {
        token_estimate: Option<u32>,
        summary: String,
        summary_len: u32,
    },
    Info(String),
    Error(String),
    Elicitation {
        elicitation_id: String,
        message: String,
        source: String,
        /// None = pending; Some = responded with this outcome label.
        outcome: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoableTurn {
    pub turn_id: String,
    pub message_id: String,
    pub text: String,
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

// ── Elicitation types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ElicitationOption {
    pub value: serde_json::Value,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElicitationFieldKind {
    SingleSelect { options: Vec<ElicitationOption> },
    MultiSelect { options: Vec<ElicitationOption> },
    TextInput,
    NumberInput { integer: bool },
    BooleanToggle,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElicitationField {
    pub name: String,
    pub title: String,
    pub description: Option<String>,
    pub required: bool,
    pub kind: ElicitationFieldKind,
}

#[derive(Debug, Clone)]
pub struct ElicitationState {
    pub elicitation_id: String,
    pub message: String,
    pub source: String,
    pub fields: Vec<ElicitationField>,
    /// Which field is active (for multi-field forms, currently always 0).
    pub field_cursor: usize,
    /// Which option within the current select field is highlighted.
    pub option_cursor: usize,
    /// Accumulated selections (field name → value).
    pub selected: HashMap<String, serde_json::Value>,
    /// Text buffer for TextInput / NumberInput fields.
    pub text_input: String,
    pub text_cursor: usize,
}

impl ElicitationState {
    /// Parse a JSON Schema `properties` object into a flat list of fields.
    /// Mirrors `parseSchema` in `ElicitationCard.tsx`.
    pub fn parse_schema(schema: &serde_json::Value) -> Vec<ElicitationField> {
        let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
            return Vec::new();
        };
        let required: std::collections::HashSet<&str> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        let mut fields = Vec::new();
        for (name, prop) in props {
            let title = prop
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or(name)
                .to_string();
            let description = prop
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let typ = prop
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("string");

            let kind = if let Some(one_of) = prop.get("oneOf").and_then(|v| v.as_array()) {
                ElicitationFieldKind::SingleSelect {
                    options: one_of
                        .iter()
                        .map(|opt| ElicitationOption {
                            value: opt.get("const").cloned().unwrap_or(serde_json::Value::Null),
                            label: opt
                                .get("title")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            description: opt
                                .get("description")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                        })
                        .collect(),
                }
            } else if let Some(enum_vals) = prop.get("enum").and_then(|v| v.as_array()) {
                ElicitationFieldKind::SingleSelect {
                    options: enum_vals
                        .iter()
                        .map(|v| ElicitationOption {
                            value: v.clone(),
                            label: v.as_str().unwrap_or("").to_string(),
                            description: None,
                        })
                        .collect(),
                }
            } else if typ == "array" {
                let items = prop.get("items");
                let item_opts = items
                    .and_then(|i| i.get("anyOf").or_else(|| i.get("oneOf")))
                    .and_then(|v| v.as_array());
                if let Some(opts) = item_opts {
                    ElicitationFieldKind::MultiSelect {
                        options: opts
                            .iter()
                            .map(|opt| ElicitationOption {
                                value: opt.get("const").cloned().unwrap_or(serde_json::Value::Null),
                                label: opt
                                    .get("title")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                description: opt
                                    .get("description")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string()),
                            })
                            .collect(),
                    }
                } else {
                    ElicitationFieldKind::TextInput
                }
            } else if typ == "boolean" {
                ElicitationFieldKind::BooleanToggle
            } else if typ == "integer" {
                ElicitationFieldKind::NumberInput { integer: true }
            } else if typ == "number" {
                ElicitationFieldKind::NumberInput { integer: false }
            } else {
                ElicitationFieldKind::TextInput
            };

            fields.push(ElicitationField {
                name: name.clone(),
                title,
                description,
                required: required.contains(name.as_str()),
                kind,
            });
        }
        fields
    }

    /// Current active field, if the schema produced at least one supported field.
    pub fn current_field(&self) -> Option<&ElicitationField> {
        self.fields
            .get(self.field_cursor.min(self.fields.len().saturating_sub(1)))
    }

    /// Number of options in the current field's select list (0 for non-select).
    pub fn current_option_count(&self) -> usize {
        match self.current_field().map(|field| &field.kind) {
            Some(ElicitationFieldKind::SingleSelect { options }) => options.len(),
            Some(ElicitationFieldKind::MultiSelect { options }) => options.len(),
            _ => 0,
        }
    }

    /// Move the option cursor by `delta`, clamped to valid range.
    pub fn move_cursor(&mut self, delta: i32) {
        let max = self.current_option_count().saturating_sub(1);
        self.option_cursor = (self.option_cursor as i32 + delta).clamp(0, max as i32) as usize;
    }

    /// For SingleSelect: record the highlighted option as the field's value.
    pub fn select_current_option(&mut self) {
        let Some(field) = self.current_field() else {
            return;
        };
        if let ElicitationFieldKind::SingleSelect { options } = &field.kind
            && let Some(opt) = options.get(self.option_cursor)
        {
            let name = field.name.clone();
            let value = opt.value.clone();
            self.selected.insert(name, value);
        }
    }

    /// For MultiSelect: toggle the highlighted option in the field's array value.
    /// For BooleanToggle: flip between explicit true and false.
    pub fn toggle_current_option(&mut self) {
        let Some(field) = self.current_field() else {
            return;
        };
        match &field.kind {
            ElicitationFieldKind::MultiSelect { options } => {
                if let Some(opt) = options.get(self.option_cursor) {
                    let name = field.name.clone();
                    let val = opt.value.clone();
                    let arr = self
                        .selected
                        .entry(name)
                        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                    if let serde_json::Value::Array(items) = arr {
                        if let Some(pos) = items.iter().position(|v| v == &val) {
                            items.remove(pos);
                        } else {
                            items.push(val);
                        }
                    }
                }
            }
            ElicitationFieldKind::BooleanToggle => {
                let name = field.name.clone();
                let next = self
                    .selected
                    .get(&name)
                    .and_then(serde_json::Value::as_bool)
                    .map(|value| !value)
                    .unwrap_or(true);
                self.selected.insert(name, serde_json::Value::Bool(next));
            }
            ElicitationFieldKind::SingleSelect { .. }
            | ElicitationFieldKind::TextInput
            | ElicitationFieldKind::NumberInput { .. } => {}
        }
    }

    /// Build the `content` object to send with an accept response.
    pub fn build_accept_content(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        for field in &self.fields {
            match &field.kind {
                ElicitationFieldKind::SingleSelect { .. }
                | ElicitationFieldKind::MultiSelect { .. } => {
                    if let Some(v) = self.selected.get(&field.name) {
                        obj.insert(field.name.clone(), v.clone());
                    }
                }
                ElicitationFieldKind::TextInput => {
                    if !self.text_input.is_empty() {
                        obj.insert(
                            field.name.clone(),
                            serde_json::Value::String(self.text_input.clone()),
                        );
                    }
                }
                ElicitationFieldKind::NumberInput { integer } => {
                    if !self.text_input.is_empty() {
                        let v = if *integer {
                            self.text_input
                                .parse::<i64>()
                                .map(|n| serde_json::json!(n))
                                .unwrap_or(serde_json::Value::Null)
                        } else {
                            self.text_input
                                .parse::<f64>()
                                .map(|n| serde_json::json!(n))
                                .unwrap_or(serde_json::Value::Null)
                        };
                        obj.insert(field.name.clone(), v);
                    }
                }
                ElicitationFieldKind::BooleanToggle => {
                    if let Some(v) = self.selected.get(&field.name) {
                        obj.insert(field.name.clone(), v.clone());
                    }
                }
            }
        }
        serde_json::Value::Object(obj)
    }

    /// Returns true if all required fields have a value.
    pub fn is_valid(&self) -> bool {
        for field in &self.fields {
            if !field.required {
                continue;
            }
            match &field.kind {
                ElicitationFieldKind::SingleSelect { .. }
                | ElicitationFieldKind::MultiSelect { .. }
                | ElicitationFieldKind::BooleanToggle => {
                    if !self.selected.contains_key(&field.name) {
                        return false;
                    }
                }
                ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
                    if self.text_input.is_empty() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Source label for display (strips the "mcp:" / "builtin:" prefix).
    pub fn source_label(&self) -> &str {
        if let Some(rest) = self.source.strip_prefix("mcp:") {
            rest
        } else if self.source == "builtin:question" {
            "built-in"
        } else {
            &self.source
        }
    }

    /// Returns a human-readable summary of what the user selected/entered,
    /// for display in the resolved chat card.
    ///
    /// - SingleSelect  → label of the chosen option
    /// - MultiSelect   → comma-joined labels of checked options
    /// - TextInput / NumberInput → the raw text
    /// - BooleanToggle → "Yes" or "No"
    pub fn selected_display(&self) -> String {
        let Some(field) = self.fields.first() else {
            return String::new();
        };
        match &field.kind {
            ElicitationFieldKind::SingleSelect { options } => {
                let val = self.selected.get(&field.name);
                options
                    .iter()
                    .find(|o| Some(&o.value) == val)
                    .map(|o| format!("{OUTCOME_BULLET}{}", o.label))
                    .unwrap_or_default()
            }
            ElicitationFieldKind::MultiSelect { options } => {
                if let Some(serde_json::Value::Array(arr)) = self.selected.get(&field.name) {
                    options
                        .iter()
                        .filter(|o| arr.contains(&o.value))
                        .map(|o| format!("{OUTCOME_BULLET}{}", o.label))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    String::new()
                }
            }
            ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
                self.text_input.clone()
            }
            ElicitationFieldKind::BooleanToggle => {
                match self.selected.get(&field.name).and_then(|v| v.as_bool()) {
                    Some(true) => "Yes".into(),
                    Some(false) => "No".into(),
                    None => String::new(),
                }
            }
        }
    }

    /// Constructor used by unit tests.
    #[cfg(test)]
    pub fn new_for_test(fields: Vec<ElicitationField>) -> Self {
        Self {
            elicitation_id: "test-id".into(),
            message: "Test question".into(),
            source: "builtin:question".into(),
            fields,
            field_cursor: 0,
            option_cursor: 0,
            selected: HashMap::new(),
            text_input: String::new(),
            text_cursor: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SessionStatsLite {
    pub active_llm_duration: Duration,
    pub open_llm_request_ts: Option<i64>,
    pub open_llm_request_instant: Option<Instant>,
    pub latest_context_tokens: Option<u64>,
    pub total_tool_calls: u32,
}

#[derive(Debug, Clone)]
pub struct SessionActivity {
    pub last_event_at: Instant,
}

#[derive(Debug, Clone)]
pub enum ToolDetail {
    None,
    /// Compact one-liner info for display after tool name
    Summary(String),
    /// One-liner header + indented output lines below
    SummaryWithOutput {
        header: String,
        output: String,
    },
    Edit {
        file: String,
        old: String,
        new: String,
        start_line: Option<usize>,
        /// Pre-computed diff lines (avoids re-running TextDiff on every render).
        cached_lines: Vec<Line<'static>>,
    },
    WriteFile {
        path: String,
        content: String,
        /// Pre-computed write preview lines.
        cached_lines: Vec<Line<'static>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnState {
    Connecting,
    Connected,
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionOp {
    Undo,
    Redo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivityState {
    Idle,
    Thinking,
    Streaming,
    RunningTool { name: String },
    Compacting { token_estimate: u32 },
    SessionOp(SessionOp),
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
        /// Total sessions in this group (unfiltered).
        session_count: usize,
        /// Whether the group is currently collapsed.
        collapsed: bool,
    },
    /// A session row inside an expanded group.
    Session {
        /// Index into `App::session_groups`.
        group_idx: usize,
        /// Index into `App::session_groups[group_idx].sessions`.
        session_idx: usize,
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
/// Target visible sessions per group when auto-filling the sessions popup.
pub const POPUP_SESSION_PAGE_TARGET: usize = 10;
pub const SESSION_GROUP_PAGE_LIMIT: u32 = 10;

/// Maximum number of workspace groups shown on the start page.
/// Groups beyond this cap are hidden from the start page but remain accessible
/// through the session popup.
pub const MAX_VISIBLE_GROUPS: usize = 3;

/// In-memory per-mode cached state within a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedModeState {
    /// `"provider/model"` e.g. `"anthropic/claude-sonnet-4-20250514"`
    pub model: String,
    /// Reasoning effort level. `None` = auto.
    pub effort: Option<String>,
}

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
        /// Total sessions in this group (unfiltered).
        session_count: usize,
        /// Whether the group is currently collapsed in the popup.
        collapsed: bool,
    },
    /// A session row inside an expanded group.
    Session {
        /// Index into `App::session_groups`.
        group_idx: usize,
        /// Index into `App::session_groups[group_idx].sessions`.
        session_idx: usize,
    },
    /// A "... load more..." row that fetches the next page for one group.
    LoadMore {
        /// Index into `App::session_groups`.
        group_idx: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelPopupItem {
    ProviderHeader {
        provider: String,
        model_count: usize,
    },
    Model {
        model_idx: usize,
    },
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
    /// CWDs with an outstanding group-page session request.
    pub pending_session_group_loads: HashSet<Option<String>>,
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

    // chat
    pub messages: Vec<ChatEntry>,
    pub input: String,
    pub input_cursor: usize,
    pub input_scroll: u16,
    pub input_line_width: usize,
    pub input_preferred_col: Option<usize>,
    pub scroll_offset: u16,
    /// Total content height (in rows) from the last render frame.
    /// Used to compensate `scroll_offset` when content grows while the user
    /// is scrolled up, so the viewport stays at the same absolute position.
    pub prev_total_height: u16,
    pub activity: ActivityState,
    pub streaming_content: String,
    pub streaming_cache: StreamingCache,
    pub streaming_thinking: String,
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

    // thinking display
    pub show_thinking: bool,

    // reasoning effort
    /// Current reasoning-effort level. `None` = "auto" (server default).
    /// Matches `reasoningEffort: string | null` in the web UI.
    pub reasoning_effort: Option<String>,
    /// Per-session, per-mode cache: session_id → mode → CachedModeState.
    /// Stores which model and reasoning effort were used in each mode within each
    /// session.  Loaded from `~/.cache/qmt/tui-cache.toml` on startup.
    pub session_cache: HashMap<String, HashMap<String, CachedModeState>>,

    // model info
    pub current_model: Option<String>,
    pub current_provider: Option<String>,
    pub models: Vec<ModelEntry>,
    pub model_cursor: usize,
    pub model_filter: String,
    /// Per-mode model preferences: mode -> (provider, model).
    /// Set when the user manually selects a model; applied automatically on mode switch.
    pub mode_model_preferences: HashMap<String, (String, String)>,

    // delegate agent model preferences
    /// Known agents from the server's `state` message.
    /// Single-element or empty in single-agent mode.
    pub agents: Vec<crate::protocol::AgentInfo>,
    /// Currently selected tab index in the model popup.
    /// 0 = "Planner" (plan mode), 1 = "Build" (build mode), 2..N = delegate agents.
    pub model_popup_agent_tab: usize,
    /// Per-agent-id model preferences: agent_id → (provider, model).
    /// Applied automatically when `SessionForked` creates a child for that agent.
    pub delegate_model_preferences: HashMap<String, (String, String)>,

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

    // connection
    pub conn: ConnState,
    pub reconnect_attempt: u32,
    pub reconnect_delay_ms: Option<u64>,

    // server lifecycle (managed by server_manager::supervisor)
    pub server_state: crate::server_manager::ServerState,

    // syntax highlighting
    pub hl: Highlighter,

    // card cache for incremental rendering
    pub card_cache: CardCache,

    // auth popup state
    pub auth_providers: Vec<crate::protocol::AuthProviderEntry>,
    pub auth_cursor: usize,
    pub auth_filter: String,
    pub auth_selected: Option<usize>,
    pub auth_panel: AuthPanel,
    pub auth_api_key_input: String,
    pub auth_api_key_cursor: usize,
    pub auth_api_key_masked: bool,
    pub auth_oauth_flow: Option<crate::protocol::OAuthFlowData>,
    pub auth_oauth_response: String,
    pub auth_oauth_response_cursor: usize,
    pub auth_result_message: Option<(bool, String)>,
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
    /// Drained by handle_server_msg after each event/replay batch.
    pub pending_commands: Vec<ClientMsg>,
    /// Child-session state observed before a delegation entry can be linked.
    pub pending_delegate_child_states: HashMap<String, DelegateChildState>,
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

impl App {
    pub fn session_group_page_request(&mut self, group_idx: usize) -> Option<ClientMsg> {
        let group = self.session_groups.get(group_idx)?;
        let cursor = group.next_cursor.clone()?;
        let cwd = group.cwd.clone();
        self.pending_session_group_loads.insert(cwd.clone());
        Some(ClientMsg::list_sessions_group(
            cwd,
            cursor,
            SESSION_GROUP_PAGE_LIMIT,
        ))
    }

    pub fn new() -> Self {
        Self {
            screen: Screen::Sessions,
            popup: Popup::None,
            chord: false,
            session_groups: Vec::new(),
            session_cursor: 0,
            session_filter: String::new(),
            session_popup_tab: 0,
            collapsed_groups: HashSet::new(),
            popup_collapsed_groups: HashSet::new(),
            pending_session_group_loads: HashSet::new(),
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
            messages: Vec::new(),
            input: String::new(),
            input_cursor: 0,
            input_scroll: 0,
            input_line_width: 1,
            input_preferred_col: None,
            scroll_offset: 0,
            prev_total_height: 0,
            activity: ActivityState::Idle,
            streaming_content: String::new(),
            streaming_cache: StreamingCache::new(),
            streaming_thinking: String::new(),
            streaming_thinking_cache: StreamingCache::new(),
            file_index: Vec::new(),
            file_index_generated_at: None,
            file_index_loading: false,
            file_index_error: None,
            mention_state: None,
            slash_state: None,
            last_compaction_token_estimate: None,
            elicitation: None,
            show_thinking: true,
            reasoning_effort: None,
            session_cache: HashMap::new(),
            current_model: None,
            current_provider: None,
            models: Vec::new(),
            model_cursor: 0,
            model_filter: String::new(),
            mode_model_preferences: HashMap::new(),
            agents: Vec::new(),
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
            auth_result_message: None,
            auth_clipboard_fallback: None,
            delegate_entries: Vec::new(),
            delegate_cursor: 0,
            delegate_filter: String::new(),
            parent_session_id: None,
            pending_parent_session_id: None,
            suppress_delegation_result: false,
            pending_commands: Vec::new(),
            pending_delegate_child_states: HashMap::new(),
            pending_delegate_tool_calls: Vec::new(),
            suppress_turn_output: false,
            status: "connecting...".into(),
            tick: 0,
            should_quit: false,
        }
    }

    /// Invalidate both streaming caches and clear the thinking buffer.
    ///
    /// Call this when a streaming turn ends (assistant message finalized,
    /// new turn starts, session reloaded, etc.) so stale markdown renders
    /// are discarded.
    pub fn invalidate_streaming_caches(&mut self) {
        self.streaming_cache.invalidate();
        self.streaming_thinking.clear();
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
    /// Updates `self.reasoning_effort`, caches the session mode state, and
    /// returns the [`ClientMsg`] to forward to the server.
    /// Returns `None` if the level is invalid (state is unchanged).
    pub fn set_reasoning_effort(&mut self, level: Option<&str>) -> Option<ClientMsg> {
        match validate_reasoning_effort(level) {
            Some(normalized) => {
                self.reasoning_effort = normalized;
                self.cache_session_mode_state();
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

    /// Save the current model + reasoning effort into the session cache for
    /// the current `session_id` + `agent_mode`.
    /// No-op if session_id, provider, or model are unknown.
    pub fn cache_session_mode_state(&mut self) {
        let (Some(sid), Some(provider), Some(model)) = (
            self.session_id.clone(),
            self.current_provider.clone(),
            self.current_model.clone(),
        ) else {
            return;
        };
        let model_key = format!("{provider}/{model}");
        self.session_cache.entry(sid).or_default().insert(
            self.agent_mode.clone(),
            CachedModeState {
                model: model_key,
                effort: self.reasoning_effort.clone(),
            },
        );
    }

    /// Update the cached model for a specific mode in the current session.
    /// Resets effort to `None` (auto) to match active-mode model-selection behaviour.
    /// No-op if there is no current session.
    pub fn update_cached_mode_model(&mut self, mode: &str, provider: &str, model: &str) {
        let Some(sid) = self.session_id.clone() else {
            return;
        };
        let model_key = format!("{provider}/{model}");
        self.session_cache.entry(sid).or_default().insert(
            mode.to_string(),
            CachedModeState {
                model: model_key,
                effort: None,
            },
        );
    }

    /// Look up the cached mode state for the current `session_id` +
    /// `agent_mode` and restore the model and effort from it.
    ///
    /// Returns a list of commands to send to the server:
    /// - `SetSessionModel` if the cached model differs from the current one
    /// - `SetReasoningEffort` if the cached effort differs from the current one
    ///
    /// Returns empty vec when there is no cache entry or nothing changed.
    pub fn apply_cached_mode_state(&mut self) -> Vec<ClientMsg> {
        let Some(sid) = self.session_id.clone() else {
            return vec![];
        };
        let cached = self
            .session_cache
            .get(&sid)
            .and_then(|modes| modes.get(&self.agent_mode))
            .cloned();
        let Some(cached) = cached else {
            return vec![];
        };

        let mut cmds = Vec::new();

        // Restore model if it differs from what the session currently has.
        let current_model_key = match (
            self.current_provider.as_deref(),
            self.current_model.as_deref(),
        ) {
            (Some(p), Some(m)) => format!("{p}/{m}"),
            _ => String::new(),
        };
        if cached.model != current_model_key {
            // Parse "provider/model" back into parts.
            if let Some((provider, model)) = cached.model.split_once('/') {
                // Find the model entry to get its full id + node_id.
                let model_entry = self
                    .models
                    .iter()
                    .find(|e| e.provider == provider && e.model == model);
                if let Some(entry) = model_entry {
                    cmds.push(ClientMsg::SetSessionModel {
                        session_id: sid.clone(),
                        model_id: entry.id.clone(),
                        node_id: entry.node_id.clone(),
                    });
                    self.current_provider = Some(provider.to_string());
                    self.current_model = Some(model.to_string());
                }
            }
        }

        // Restore effort if it differs.
        if cached.effort != self.reasoning_effort {
            self.reasoning_effort = cached.effort.clone();
            let effort_str = cached.effort.as_deref().unwrap_or("auto").to_string();
            cmds.push(ClientMsg::SetReasoningEffort {
                reasoning_effort: effort_str,
            });
        }

        cmds
    }

    /// Filtered auth providers matching the current `auth_filter`.
    pub fn filtered_auth_providers(&self) -> Vec<(usize, &crate::protocol::AuthProviderEntry)> {
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
        self.auth_result_message = None;
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
        self.auth_result_message = None;
        self.auth_clipboard_fallback = None;
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

    pub fn visible_model_popup_items(&self) -> Vec<ModelPopupItem> {
        let filtered = self.filtered_models();
        let mut items = Vec::new();
        let mut current_provider: Option<&str> = None;

        for model in filtered {
            if current_provider != Some(model.provider.as_str()) {
                current_provider = Some(model.provider.as_str());
                let provider_count = self
                    .filtered_models()
                    .into_iter()
                    .filter(|m| m.provider == model.provider)
                    .count();
                items.push(ModelPopupItem::ProviderHeader {
                    provider: model.provider.clone(),
                    model_count: provider_count,
                });
            }
            if let Some(model_idx) = self.models.iter().position(|m| m.id == model.id) {
                items.push(ModelPopupItem::Model { model_idx });
            }
        }

        items
    }

    pub fn current_mode_model_cursor(&self) -> usize {
        self.mode_model_cursor(&self.agent_mode.clone())
    }

    /// Cursor position for a given mode's preferred model in the popup list.
    pub fn mode_model_cursor(&self, mode: &str) -> usize {
        let target = self
            .get_mode_model_preference(mode)
            .or(if self.agent_mode == mode {
                match (
                    self.current_provider.as_deref(),
                    self.current_model.as_deref(),
                ) {
                    (Some(provider), Some(model)) => Some((provider, model)),
                    _ => None,
                }
            } else {
                None
            });

        let Some((provider, model)) = target else {
            return self
                .visible_model_popup_items()
                .iter()
                .position(|item| matches!(item, ModelPopupItem::Model { .. }))
                .unwrap_or(0);
        };

        self.visible_model_popup_items()
            .iter()
            .position(|item| match item {
                ModelPopupItem::Model { model_idx } => self
                    .models
                    .get(*model_idx)
                    .is_some_and(|entry| entry.provider == provider && entry.model == model),
                ModelPopupItem::ProviderHeader { .. } => false,
            })
            .unwrap_or_else(|| {
                self.visible_model_popup_items()
                    .iter()
                    .position(|item| matches!(item, ModelPopupItem::Model { .. }))
                    .unwrap_or(0)
            })
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
        undo_stack: &[UndoStackFrame],
        preferred_frontier_message_id: Option<&str>,
        reverted_files: Option<&[String]>,
    ) -> Option<UndoState> {
        if undo_stack.is_empty() {
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
            .iter()
            .map(|frame| {
                let previous = previous_by_message_id.get(&frame.message_id);
                let reverted_files =
                    if preferred_frontier_message_id == Some(frame.message_id.as_str()) {
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
                            .find(|turn| turn.message_id == frame.message_id)
                            .map(|turn| turn.turn_id.clone())
                    })
                    .unwrap_or_else(|| frame.message_id.clone());
                UndoFrame {
                    turn_id,
                    message_id: frame.message_id.clone(),
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
        self.card_cache.invalidate();
        self.refresh_transient_status();
    }

    pub fn set_mode_model_preference(&mut self, mode: &str, provider: &str, model: &str) {
        self.mode_model_preferences
            .insert(mode.to_string(), (provider.to_string(), model.to_string()));
    }

    pub fn get_mode_model_preference(&self, mode: &str) -> Option<(&str, &str)> {
        self.mode_model_preferences
            .get(mode)
            .map(|(p, m)| (p.as_str(), m.as_str()))
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

    /// Total number of tabs in the model popup.
    /// Always at least 3 (Plan + Build + Review), plus one tab per delegation agent.
    pub fn model_popup_tab_count(&self) -> usize {
        3 + self.agents.len().saturating_sub(1)
    }

    /// Label for a model popup tab.
    /// 0 = "plan", 1 = "build", 2 = "review", 3+ maps to `agents[1..].name`.
    pub fn model_popup_tab_label(&self, tab_idx: usize) -> &str {
        match tab_idx {
            0 => "plan",
            1 => "build",
            2 => "review",
            _ => self
                .agents
                .get(tab_idx - 2)
                .map(|a| a.name.as_str())
                .unwrap_or("???"),
        }
    }

    /// The agent_id for an agent tab (index 3+), None for mode tabs (0, 1, 2).
    pub fn model_popup_tab_agent_id(&self, tab_idx: usize) -> Option<&str> {
        if tab_idx < 3 {
            None
        } else {
            self.agents.get(tab_idx - 2).map(|a| a.id.as_str())
        }
    }

    /// The mode name for a mode tab (0 = "plan", 1 = "build", 2 = "review"), None for agent tabs.
    pub fn model_popup_tab_mode(&self, tab_idx: usize) -> Option<&'static str> {
        match tab_idx {
            0 => Some("plan"),
            1 => Some("build"),
            2 => Some("review"),
            _ => None,
        }
    }

    pub fn set_delegate_model_preference(&mut self, agent_id: &str, provider: &str, model: &str) {
        self.delegate_model_preferences.insert(
            agent_id.to_string(),
            (provider.to_string(), model.to_string()),
        );
    }

    pub fn get_delegate_model_preference(&self, agent_id: &str) -> Option<(&str, &str)> {
        self.delegate_model_preferences
            .get(agent_id)
            .map(|(p, m)| (p.as_str(), m.as_str()))
    }

    /// Cursor position for a delegate agent's preferred model in the popup list.
    pub fn delegate_model_cursor(&self, agent_id: &str) -> usize {
        let items = self.visible_model_popup_items();
        let Some((provider, model)) = self.get_delegate_model_preference(agent_id) else {
            return items
                .iter()
                .position(|item| matches!(item, ModelPopupItem::Model { .. }))
                .unwrap_or(0);
        };
        items
            .iter()
            .position(|item| match item {
                ModelPopupItem::Model { model_idx } => {
                    let m = &self.models[*model_idx];
                    m.provider == provider && m.model == model
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

    #[test]
    fn state_msg_sets_reasoning_effort() {
        let mut app = App::new();
        app.handle_server_msg(RawServerMsg {
            msg_type: "state".into(),
            data: Some(serde_json::json!({
                "active_session_id": null,
                "agents": [],
                "agent_mode": "build",
                "reasoning_effort": "high"
            })),
        });
        assert_eq!(app.reasoning_effort, Some("high".into()));
    }

    #[test]
    fn state_msg_with_null_reasoning_effort_sets_none() {
        let mut app = App::new();
        app.reasoning_effort = Some("medium".into());
        app.handle_server_msg(RawServerMsg {
            msg_type: "state".into(),
            data: Some(serde_json::json!({
                "active_session_id": null,
                "agents": [],
                "agent_mode": "build",
                "reasoning_effort": null
            })),
        });
        assert_eq!(app.reasoning_effort, None);
    }

    #[test]
    fn state_msg_missing_reasoning_effort_leaves_existing() {
        let mut app = App::new();
        app.reasoning_effort = Some("medium".into());
        app.handle_server_msg(RawServerMsg {
            msg_type: "state".into(),
            data: Some(serde_json::json!({
                "active_session_id": null,
                "agents": [],
                "agent_mode": "build"
                // reasoning_effort key absent → existing value preserved
            })),
        });
        assert_eq!(app.reasoning_effort, Some("medium".into()));
    }

    // ── reasoning_effort push notification ────────────────────────────────────

    #[test]
    fn reasoning_effort_push_updates_field() {
        let mut app = App::new();
        app.handle_server_msg(RawServerMsg {
            msg_type: "reasoning_effort".into(),
            data: Some(serde_json::json!({ "reasoning_effort": "max" })),
        });
        assert_eq!(app.reasoning_effort, Some("max".into()));
    }

    #[test]
    fn reasoning_effort_push_null_clears_field() {
        let mut app = App::new();
        app.reasoning_effort = Some("low".into());
        app.handle_server_msg(RawServerMsg {
            msg_type: "reasoning_effort".into(),
            data: Some(serde_json::json!({ "reasoning_effort": null })),
        });
        assert_eq!(app.reasoning_effort, None);
    }

    #[test]
    fn reasoning_effort_push_auto_string_clears_field() {
        let mut app = App::new();
        app.reasoning_effort = Some("high".into());
        app.handle_server_msg(RawServerMsg {
            msg_type: "reasoning_effort".into(),
            data: Some(serde_json::json!({ "reasoning_effort": "auto" })),
        });
        assert_eq!(app.reasoning_effort, None);
    }

    #[test]
    fn reasoning_effort_push_invalid_value_rejected() {
        let mut app = App::new();
        app.reasoning_effort = Some("medium".into());
        app.handle_server_msg(RawServerMsg {
            msg_type: "reasoning_effort".into(),
            data: Some(serde_json::json!({ "reasoning_effort": "ultra" })),
        });
        assert_eq!(app.reasoning_effort, Some("medium".into()));
    }

    #[test]
    fn event_message_ignores_non_active_session() {
        let mut app = App::new();
        app.session_id = Some("session-b".into());

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "agent_id": "agent-1",
                "session_id": "session-a",
                "event": {
                    "type": "ephemeral",
                    "data": {
                        "kind": {
                            "type": "assistant_content_delta",
                            "data": {
                                "content": "leaked text",
                                "message_id": null
                            }
                        },
                        "timestamp": null
                    }
                }
            })),
        });

        assert!(app.streaming_content.is_empty());
        assert!(app.messages.is_empty());
    }

    #[test]
    fn session_events_message_ignores_non_active_session() {
        let mut app = App::new();
        app.session_id = Some("session-b".into());

        app.handle_server_msg(RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({
                "session_id": "session-a",
                "agent_id": "agent-1",
                "events": [
                    {
                        "type": "ephemeral",
                        "data": {
                            "kind": {
                                "type": "assistant_content_delta",
                                "data": {
                                    "content": "leaked batch text",
                                    "message_id": null
                                }
                            },
                            "timestamp": null
                        }
                    }
                ]
            })),
        });

        assert!(app.streaming_content.is_empty());
        assert!(app.messages.is_empty());
    }

    #[test]
    fn event_message_applies_active_session() {
        let mut app = App::new();
        app.session_id = Some("session-a".into());

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "agent_id": "agent-1",
                "session_id": "session-a",
                "event": {
                    "type": "ephemeral",
                    "data": {
                        "kind": {
                            "type": "assistant_content_delta",
                            "data": {
                                "content": "visible text",
                                "message_id": null
                            }
                        },
                        "timestamp": null
                    }
                }
            })),
        });

        assert_eq!(app.streaming_content, "visible text");
    }

    #[test]
    fn non_active_session_event_still_counts_as_recent_activity() {
        let mut app = App::new();
        app.session_id = Some("session-b".into());

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "agent_id": "agent-1",
                "session_id": "session-a",
                "event": {
                    "type": "ephemeral",
                    "data": {
                        "kind": {
                            "type": "assistant_content_delta",
                            "data": {
                                "content": "hidden text",
                                "message_id": null
                            }
                        },
                        "timestamp": null
                    }
                }
            })),
        });

        assert_eq!(app.active_session_count(), 1);
        assert!(app.streaming_content.is_empty());
    }

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
    use crate::server_msg::accumulate_delegate_stats;

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

    #[test]
    fn delegation_requested_creates_entry() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::DelegationRequested {
                delegation: DelegationData {
                    public_id: "del-1".into(),
                    target_agent_id: Some("coder".into()),
                    objective: Some("Fix the bug".into()),
                },
            },
            false,
            None,
        );
        assert_eq!(app.delegate_entries.len(), 1);
        assert_eq!(app.delegate_entries[0].delegation_id, "del-1");
        assert_eq!(app.delegate_entries[0].objective, "Fix the bug");
        assert_eq!(
            app.delegate_entries[0].target_agent_id.as_deref(),
            Some("coder")
        );
        assert_eq!(app.delegate_entries[0].status, DelegateStatus::InProgress);
        assert!(app.delegate_entries[0].child_session_id.is_none());
    }

    #[test]
    fn session_forked_sets_child_session_id() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "Fix the bug".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: Some("del-1".into()),
                target_agent_id: None,
            },
            false,
            None,
        );
        assert_eq!(
            app.delegate_entries[0].child_session_id.as_deref(),
            Some("child-sess-1")
        );
    }

    #[test]
    fn session_forked_ignores_user_origin() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("user".into()),
                fork_point_ref: Some("del-1".into()),
                target_agent_id: None,
            },
            false,
            None,
        );
        assert!(app.delegate_entries[0].child_session_id.is_none());
    }

    #[test]
    fn delegation_completed_sets_status() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::DelegationCompleted {
                delegation_id: "del-1".into(),
                result: Some("done".into()),
            },
            false,
            None,
        );
        assert_eq!(app.delegate_entries[0].status, DelegateStatus::Completed);
    }

    #[test]
    fn delegation_failed_sets_status() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::DelegationFailed {
                delegation_id: "del-1".into(),
                error: Some("boom".into()),
            },
            false,
            None,
        );
        assert_eq!(app.delegate_entries[0].status, DelegateStatus::Failed);
    }

    #[test]
    fn delegation_completed_unknown_id_is_noop() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::DelegationCompleted {
                delegation_id: "unknown".into(),
                result: None,
            },
            false,
            None,
        );
        assert_eq!(app.delegate_entries[0].status, DelegateStatus::InProgress);
    }

    #[test]
    fn session_loaded_clears_delegate_entries() {
        let mut app = App::new();
        app.delegate_entries
            .push(make_entry("d1", "old", DelegateStatus::Completed));
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "audit": { "events": [] }
            })),
        });
        assert!(app.delegate_entries.is_empty());
    }

    #[test]
    fn session_created_clears_delegate_entries() {
        let mut app = App::new();
        app.delegate_entries
            .push(make_entry("d1", "old", DelegateStatus::Completed));
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_created".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1"
            })),
        });
        assert!(app.delegate_entries.is_empty());
    }

    #[test]
    fn replay_audit_builds_delegate_entries() {
        let mut app = App::new();
        let audit = serde_json::json!({
            "events": [
                {
                    "kind": {
                        "type": "delegation_requested",
                        "data": {
                            "delegation": {
                                "public_id": "del-1",
                                "target_agent_id": "coder",
                                "objective": "Fix bug"
                            }
                        }
                    }
                },
                {
                    "kind": {
                        "type": "session_forked",
                        "data": {
                            "parent_session_id": "parent",
                            "child_session_id": "child-1",
                            "target_agent_id": "coder",
                            "origin": "delegation",
                            "fork_point_type": "progress_entry",
                            "fork_point_ref": "del-1"
                        }
                    }
                },
                {
                    "kind": {
                        "type": "delegation_completed",
                        "data": {
                            "delegation_id": "del-1",
                            "result": "done"
                        }
                    }
                }
            ]
        });
        app.replay_audit(&audit, None);
        assert_eq!(app.delegate_entries.len(), 1);
        assert_eq!(app.delegate_entries[0].delegation_id, "del-1");
        assert_eq!(app.delegate_entries[0].objective, "Fix bug");
        assert_eq!(
            app.delegate_entries[0].child_session_id.as_deref(),
            Some("child-1")
        );
        assert_eq!(app.delegate_entries[0].status, DelegateStatus::Completed);
    }

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
            },
        );
        assert_eq!(stats.context_limit, 200_000);
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

    #[test]
    fn child_session_events_route_to_delegate_stats() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        // Simulate a session_events message for the child session
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "events": [
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "tool_call_start", "data": {
                                "tool_call_id": "t1",
                                "tool_name": "read_tool",
                                "arguments": null
                            }},
                            "timestamp": null
                        }
                    },
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "tool_call_start", "data": {
                                "tool_call_id": "t2",
                                "tool_name": "write_file",
                                "arguments": null
                            }},
                            "timestamp": null
                        }
                    }
                ]
            })),
        });
        assert_eq!(app.delegate_entries[0].stats.tool_calls, 2);
    }

    #[test]
    fn child_session_events_with_unknown_kind_still_counts_known_events() {
        // If the batch contains an unknown event kind with a data payload
        // (e.g. a new server-side event type the TUI hasn't seen), the known
        // events in the same batch must still be counted. Previously the whole
        // Vec<EventEnvelope> deserialization failed on the first unknown kind.
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "events": [
                    // known event — should be counted
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "tool_call_start", "data": {
                                "tool_call_id": "t1",
                                "tool_name": "read_tool",
                                "arguments": null
                            }},
                            "timestamp": null
                        }
                    },
                    // unknown event kind with a data payload — must NOT blow up the batch
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "brand_new_unknown_event_2099", "data": {
                                "some_field": "some_value",
                                "nested": { "deep": true }
                            }},
                            "timestamp": null
                        }
                    },
                    // another known event after the unknown one — must still be counted
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "tool_call_start", "data": {
                                "tool_call_id": "t2",
                                "tool_name": "write_file",
                                "arguments": null
                            }},
                            "timestamp": null
                        }
                    }
                ]
            })),
        });
        assert_eq!(
            app.delegate_entries[0].stats.tool_calls, 2,
            "both known tool_call_start events must be counted despite unknown event in batch"
        );
    }

    #[test]
    fn child_session_events_unknown_session_ignored() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({
                "session_id": "unknown-sess",
                "agent_id": "coder",
                "events": []
            })),
        });
        assert_eq!(app.delegate_entries[0].stats.tool_calls, 0);
    }

    #[test]
    fn child_elicitation_marks_delegate_awaiting_input() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "elicitation_requested", "data": {
                            "elicitation_id": "elic-1",
                            "session_id": "child-sess-1",
                            "message": "Need approval",
                            "requested_schema": {
                                "properties": { "choice": { "oneOf": [{ "const": "a", "title": "A" }] } },
                                "required": ["choice"]
                            },
                            "source": "builtin:question"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        let entry = &app.delegate_entries[0];
        assert!(entry.awaiting_input());
        assert_eq!(
            entry.pending_elicitation(),
            Some(("elic-1", "Need approval", "builtin:question"))
        );
    }

    #[test]
    fn child_elicitation_before_session_fork_marks_delegate_awaiting_after_link() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "elicitation_requested", "data": {
                            "elicitation_id": "elic-1",
                            "session_id": "child-sess-1",
                            "message": "Need approval",
                            "requested_schema": { "properties": {}, "required": [] },
                            "source": "builtin:question"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: Some("del-1".into()),
                target_agent_id: Some("coder".into()),
            },
            false,
            None,
        );

        let entry = &app.delegate_entries[0];
        assert_eq!(entry.child_session_id.as_deref(), Some("child-sess-1"));
        assert!(entry.awaiting_input());
    }

    #[test]
    fn session_forked_without_ref_links_single_matching_delegate_by_agent() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: None,
                target_agent_id: Some("coder".into()),
            },
            false,
            None,
        );
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "elicitation_requested", "data": {
                            "elicitation_id": "elic-1",
                            "session_id": "child-sess-1",
                            "message": "Need approval",
                            "requested_schema": { "properties": {}, "required": [] },
                            "source": "builtin:question"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        let entry = &app.delegate_entries[0];
        assert_eq!(entry.child_session_id.as_deref(), Some("child-sess-1"));
        assert!(entry.awaiting_input());
    }

    #[test]
    fn session_forked_without_ref_does_not_link_ambiguous_agent_match() {
        let mut app = App::new();
        for delegation_id in ["del-1", "del-2"] {
            app.delegate_entries.push(DelegateEntry {
                delegation_id: delegation_id.into(),
                child_session_id: None,
                delegate_tool_call_id: None,
                target_agent_id: Some("coder".into()),
                objective: "task".into(),
                status: DelegateStatus::InProgress,
                stats: DelegateStats::default(),
                started_at: None,
                ended_at: None,
                child_state: DelegateChildState::None,
            });
        }

        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: None,
                target_agent_id: Some("coder".into()),
            },
            false,
            None,
        );
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "elicitation_requested", "data": {
                            "elicitation_id": "elic-1",
                            "session_id": "child-sess-1",
                            "message": "Need approval",
                            "requested_schema": { "properties": {}, "required": [] },
                            "source": "builtin:question"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        assert!(
            app.delegate_entries
                .iter()
                .all(|e| e.child_session_id.is_none())
        );
        assert!(app.delegate_entries.iter().all(|e| !e.awaiting_input()));
        assert!(
            app.pending_delegate_child_states
                .contains_key("child-sess-1")
        );
    }

    #[test]
    fn buffered_child_elicitation_applies_when_session_fork_later_links() {
        let mut app = App::new();
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "elicitation_requested", "data": {
                            "elicitation_id": "elic-1",
                            "session_id": "child-sess-1",
                            "message": "Need approval",
                            "requested_schema": { "properties": {}, "required": [] },
                            "source": "builtin:question"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: None,
                target_agent_id: Some("coder".into()),
            },
            false,
            None,
        );

        assert!(app.delegate_entries[0].awaiting_input());
        assert!(
            !app.pending_delegate_child_states
                .contains_key("child-sess-1")
        );
    }

    #[test]
    fn child_question_tool_end_clears_pending_elicitation() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::PendingElicitation {
                elicitation_id: "elic-1".into(),
                message: "Need approval".into(),
                requested_schema: serde_json::json!({ "properties": {} }),
                source: "builtin:question".into(),
            },
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "tool_call_end", "data": {
                            "tool_call_id": "tool-1",
                            "tool_name": "question",
                            "is_error": false,
                            "result": "answered"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        let entry = &app.delegate_entries[0];
        assert!(!entry.awaiting_input());
        assert_eq!(entry.child_state, DelegateChildState::QuestionToolFinished);
    }

    #[test]
    fn child_assistant_message_clears_pending_elicitation() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::PendingElicitation {
                elicitation_id: "elic-1".into(),
                message: "Need approval".into(),
                requested_schema: serde_json::json!({ "properties": {} }),
                source: "builtin:question".into(),
            },
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-sess-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "assistant_message_stored", "data": {
                            "content": "Done",
                            "thinking": null,
                            "message_id": null
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        let entry = &app.delegate_entries[0];
        assert!(!entry.awaiting_input());
        assert_eq!(entry.child_state, DelegateChildState::AssistantMessage);
    }

    #[test]
    fn delegation_completion_clears_pending_elicitation() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::PendingElicitation {
                elicitation_id: "elic-1".into(),
                message: "Need approval".into(),
                requested_schema: serde_json::json!({ "properties": {} }),
                source: "builtin:question".into(),
            },
        });

        app.handle_event_kind(
            &EventKind::DelegationCompleted {
                delegation_id: "del-1".into(),
                result: Some("done".into()),
            },
            false,
            None,
        );

        let entry = &app.delegate_entries[0];
        assert_eq!(entry.status, DelegateStatus::Completed);
        assert!(!entry.awaiting_input());
        assert_eq!(entry.child_state, DelegateChildState::None);
    }

    #[test]
    fn repeated_child_elicitation_replay_is_idempotent() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-sess-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        let event = serde_json::json!({
            "session_id": "child-sess-1",
            "agent_id": "coder",
            "event": {
                "type": "durable",
                "data": {
                    "kind": { "type": "elicitation_requested", "data": {
                        "elicitation_id": "elic-1",
                        "session_id": "child-sess-1",
                        "message": "Need approval",
                        "requested_schema": {
                            "properties": { "choice": { "oneOf": [{ "const": "a", "title": "A" }] } },
                            "required": ["choice"]
                        },
                        "source": "builtin:question"
                    }},
                    "timestamp": null
                }
            }
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(event.clone()),
        });
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(event),
        });

        let entry = &app.delegate_entries[0];
        assert!(entry.awaiting_input());
        assert_eq!(
            entry.pending_elicitation(),
            Some(("elic-1", "Need approval", "builtin:question"))
        );
    }

    // ── subscribe on SessionForked ────────────────────────────────────────────

    #[test]
    fn session_forked_queues_subscribe_command() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: Some("del-1".into()),
                target_agent_id: None,
            },
            false,
            None,
        );
        assert_eq!(app.pending_commands.len(), 1);
        assert!(matches!(
            &app.pending_commands[0],
            ClientMsg::SubscribeSession { session_id, .. } if session_id == "child-sess-1"
        ));
    }

    #[test]
    fn session_forked_user_origin_does_not_queue_subscribe() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-sess-1".into()),
                origin: Some("user".into()),
                fork_point_ref: None,
                target_agent_id: None,
            },
            false,
            None,
        );
        assert!(app.pending_commands.is_empty());
    }

    #[test]
    fn replay_audit_emits_subscribe_for_delegation_forks() {
        let mut app = App::new();
        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "audit": {
                    "events": [
                        {
                            "kind": {
                                "type": "delegation_requested",
                                "data": { "delegation": {
                                    "public_id": "del-1",
                                    "target_agent_id": "coder",
                                    "objective": "task"
                                }}
                            }
                        },
                        {
                            "kind": {
                                "type": "session_forked",
                                "data": {
                                    "parent_session_id": "parent",
                                    "child_session_id": "child-1",
                                    "target_agent_id": "coder",
                                    "origin": "delegation",
                                    "fork_point_type": "progress_entry",
                                    "fork_point_ref": "del-1"
                                }
                            }
                        }
                    ]
                }
            })),
        });
        let subscribe_cmds: Vec<_> = cmds
            .iter()
            .filter(|c| matches!(c, ClientMsg::SubscribeSession { session_id, .. } if session_id == "child-1"))
            .collect();
        assert_eq!(subscribe_cmds.len(), 1);
    }

    #[test]
    fn session_forked_subscribe_uses_target_agent_id() {
        // The subscribe command sent for a child session must use the
        // delegation's target_agent_id, not the parent session's agent_id.
        // This matches the web UI which sends kindData.target_agent_id.
        let mut app = App::new();
        app.agent_id = Some("parent-agent".into()); // parent session's agent
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()), // delegation target
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: Some("del-1".into()),
                target_agent_id: Some("coder".into()),
            },
            false,
            None,
        );
        assert_eq!(app.pending_commands.len(), 1);
        match &app.pending_commands[0] {
            ClientMsg::SubscribeSession {
                session_id,
                agent_id,
            } => {
                assert_eq!(session_id, "child-1");
                // Must be the delegation target agent, not the parent agent
                assert_eq!(agent_id.as_deref(), Some("coder"));
                assert_ne!(agent_id.as_deref(), Some("parent-agent"));
            }
            other => panic!("expected SubscribeSession, got {other:?}"),
        }
    }

    #[test]
    fn session_forked_applies_delegate_model_preference() {
        let mut app = App::new();
        app.agent_id = Some("parent-agent".into());
        app.models = vec![crate::protocol::ModelEntry {
            id: "anthropic/claude-sonnet".into(),
            label: "claude-sonnet".into(),
            provider: "anthropic".into(),
            model: "claude-sonnet".into(),
            node_id: None,
            family: None,
            quant: None,
        }];
        app.set_delegate_model_preference("coder", "anthropic", "claude-sonnet");
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: Some("del-1".into()),
                target_agent_id: Some("coder".into()),
            },
            false,
            None,
        );
        // Should have SubscribeSession + SetSessionModel
        assert_eq!(app.pending_commands.len(), 2);
        assert!(
            app.pending_commands
                .iter()
                .any(|m| matches!(m, ClientMsg::SetSessionModel { session_id, .. } if session_id == "child-1")),
            "expected SetSessionModel for child-1: {:?}",
            app.pending_commands
        );
    }

    #[test]
    fn session_forked_no_preference_does_not_send_model() {
        let mut app = App::new();
        app.agent_id = Some("parent-agent".into());
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.handle_event_kind(
            &EventKind::SessionForked {
                child_session_id: Some("child-1".into()),
                origin: Some("delegation".into()),
                fork_point_ref: Some("del-1".into()),
                target_agent_id: Some("coder".into()),
            },
            false,
            None,
        );
        // Only SubscribeSession, no SetSessionModel
        assert_eq!(app.pending_commands.len(), 1);
        assert!(
            matches!(&app.pending_commands[0], ClientMsg::SubscribeSession { .. }),
            "expected only SubscribeSession: {:?}",
            app.pending_commands
        );
    }

    #[test]
    fn session_events_deser_failure_is_logged_not_silently_dropped() {
        // Garbled JSON must produce a log entry rather than being silently
        // ignored — otherwise debugging missing stats is impossible.
        let mut app = App::new();
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({ "bad": "shape" })),
        });
        // A log entry should have been pushed
        assert!(
            app.logs
                .iter()
                .any(|e| e.message.contains("session_events")),
            "expected a log warning about session_events parse failure"
        );
    }

    #[test]
    fn delegate_child_events_do_not_inflate_multi_session_badge() {
        // Subscribing to delegate child sessions must NOT add them to
        // session_activity — otherwise other_active_session_count() would
        // count every child as a separate "other" session, inflating the
        // multi-session badge (𐬽 N).
        let mut app = App::new();
        app.session_id = Some("parent".into());
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        // Simulate session_events arriving for the child (from subscribe replay)
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({
                "session_id": "child-1",
                "agent_id": "coder",
                "events": [
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "tool_call_start", "data": {
                                "tool_call_id": "t1",
                                "tool_name": "read_tool",
                                "arguments": null
                            }},
                            "timestamp": null
                        }
                    }
                ]
            })),
        });

        // Child events should still accumulate stats
        assert_eq!(app.delegate_entries[0].stats.tool_calls, 1);
        // But the child must NOT appear in session_activity
        assert_eq!(
            app.other_active_session_count(),
            0,
            "delegate child sessions must not inflate multi-session badge"
        );

        // Simulate a live "event" arriving for the child
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "child-1",
                "agent_id": "coder",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "tool_call_start", "data": {
                            "tool_call_id": "t2",
                            "tool_name": "write_file",
                            "arguments": null
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        assert_eq!(app.delegate_entries[0].stats.tool_calls, 2);
        assert_eq!(
            app.other_active_session_count(),
            0,
            "delegate child live events must not inflate multi-session badge"
        );
    }

    #[test]
    fn parent_replay_does_not_duplicate_delegate_entries() {
        // Regression: the parent session can be replayed twice:
        // 1) via session_loaded audit replay
        // 2) via subscribed parent session_events replay
        // DelegationRequested must be idempotent, otherwise the delegate badge
        // shows e.g. 4/8 instead of 4/4.
        let mut app = App::new();

        // First load the parent session (audit replay path)
        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "parent-agent",
                "audit": {
                    "events": [
                        {
                            "kind": { "type": "delegation_requested", "data": {
                                "delegation": {
                                    "public_id": "del-1",
                                    "target_agent_id": "coder",
                                    "objective": "task one"
                                }
                            }}
                        },
                        {
                            "kind": { "type": "session_forked", "data": {
                                "parent_session_id": "parent",
                                "child_session_id": "child-1",
                                "target_agent_id": "coder",
                                "origin": "delegation",
                                "fork_point_type": "progress_entry",
                                "fork_point_ref": "del-1"
                            }}
                        },
                        {
                            "kind": { "type": "delegation_completed", "data": {
                                "delegation_id": "del-1",
                                "result": "done"
                            }}
                        },
                        {
                            "kind": { "type": "delegation_requested", "data": {
                                "delegation": {
                                    "public_id": "del-2",
                                    "target_agent_id": "coder",
                                    "objective": "task two"
                                }
                            }}
                        },
                        {
                            "kind": { "type": "session_forked", "data": {
                                "parent_session_id": "parent",
                                "child_session_id": "child-2",
                                "target_agent_id": "coder",
                                "origin": "delegation",
                                "fork_point_type": "progress_entry",
                                "fork_point_ref": "del-2"
                            }}
                        },
                        {
                            "kind": { "type": "delegation_completed", "data": {
                                "delegation_id": "del-2",
                                "result": "done"
                            }}
                        }
                    ]
                }
            })),
        });

        // Sanity check after initial audit replay.
        assert_eq!(app.delegate_entries.len(), 2);
        assert_eq!(
            app.delegate_entries
                .iter()
                .filter(|e| e.status == DelegateStatus::Completed)
                .count(),
            2
        );

        // Simulate replay of the same parent session via session_events after subscribe.
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "parent-agent",
                "events": [
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "delegation_requested", "data": {
                                "delegation": {
                                    "public_id": "del-1",
                                    "target_agent_id": "coder",
                                    "objective": "task one"
                                }
                            }},
                            "timestamp": null
                        }
                    },
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "session_forked", "data": {
                                "parent_session_id": "parent",
                                "child_session_id": "child-1",
                                "target_agent_id": "coder",
                                "origin": "delegation",
                                "fork_point_type": "progress_entry",
                                "fork_point_ref": "del-1"
                            }},
                            "timestamp": null
                        }
                    },
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "delegation_completed", "data": {
                                "delegation_id": "del-1",
                                "result": "done"
                            }},
                            "timestamp": null
                        }
                    },
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "delegation_requested", "data": {
                                "delegation": {
                                    "public_id": "del-2",
                                    "target_agent_id": "coder",
                                    "objective": "task two"
                                }
                            }},
                            "timestamp": null
                        }
                    },
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "session_forked", "data": {
                                "parent_session_id": "parent",
                                "child_session_id": "child-2",
                                "target_agent_id": "coder",
                                "origin": "delegation",
                                "fork_point_type": "progress_entry",
                                "fork_point_ref": "del-2"
                            }},
                            "timestamp": null
                        }
                    },
                    {
                        "type": "durable",
                        "data": {
                            "kind": { "type": "delegation_completed", "data": {
                                "delegation_id": "del-2",
                                "result": "done"
                            }},
                            "timestamp": null
                        }
                    }
                ]
            })),
        });

        // Must remain unique by delegation_id, not duplicate to 4.
        assert_eq!(
            app.delegate_entries.len(),
            2,
            "parent replay must not duplicate delegate entries"
        );
        assert_eq!(
            app.delegate_entries
                .iter()
                .filter(|e| e.status == DelegateStatus::Completed)
                .count(),
            2,
            "completed count must remain 2/2, not 2/4"
        );

        // Also ensure the initial session_loaded returned the child subscriptions.
        let subscribe_count = cmds
            .iter()
            .filter(|c| matches!(c, ClientMsg::SubscribeSession { .. }))
            .count();
        assert_eq!(subscribe_count, 2);
    }

    // ── delegation view (parent tracking, Screen::Delegate) ──────────────────

    #[test]
    fn session_loaded_preserves_delegates_for_child() {
        let mut app = App::new();
        app.session_id = Some("parent".into());
        app.delegate_entries
            .push(make_entry("d1", "task one", DelegateStatus::InProgress));
        app.pending_parent_session_id = Some("parent".into());

        app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "child-1",
                "agent_id": "a1",
                "audit": { "events": [] }
            })),
        });

        assert_eq!(
            app.delegate_entries.len(),
            1,
            "delegate entries must be preserved"
        );
        assert_eq!(
            app.parent_session_id.as_deref(),
            Some("parent"),
            "parent_session_id must be set"
        );
        assert_eq!(
            app.screen,
            Screen::Delegate,
            "child session must use Delegate screen"
        );
    }

    #[test]
    fn session_loaded_clears_delegates_for_non_child() {
        let mut app = App::new();
        app.session_id = Some("old".into());
        app.delegate_entries
            .push(make_entry("d1", "task", DelegateStatus::Completed));

        app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "unrelated",
                "agent_id": "a1",
                "audit": { "events": [] }
            })),
        });

        assert!(
            app.delegate_entries.is_empty(),
            "delegate entries must be cleared"
        );
        assert!(
            app.parent_session_id.is_none(),
            "no parent for non-child session"
        );
        assert_eq!(app.screen, Screen::Chat, "non-child must use Chat screen");
    }

    #[test]
    fn session_loaded_resolves_parent_from_session_groups() {
        use crate::protocol::{SessionGroup, SessionSummary};
        let mut app = App::new();
        app.session_groups = vec![SessionGroup {
            cwd: Some("/test".into()),
            sessions: vec![SessionSummary {
                session_id: "child-1".into(),
                title: Some("child".into()),
                cwd: Some("/test".into()),
                created_at: None,
                updated_at: None,
                parent_session_id: Some("parent".into()),
                has_children: false,
                ..Default::default()
            }],
            latest_activity: None,
            ..Default::default()
        }];

        app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "child-1",
                "agent_id": "a1",
                "audit": { "events": [] }
            })),
        });

        assert_eq!(
            app.parent_session_id.as_deref(),
            Some("parent"),
            "parent resolved from session_groups"
        );
        assert_eq!(app.screen, Screen::Delegate);
    }

    // ── delegation result suppression ────────────────────────────────────────

    #[test]
    fn delegation_completed_suppresses_next_user_message() {
        let mut app = App::new();
        app.session_id = Some("parent".into());
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        // delegation_completed fires
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "delegation_completed", "data": {
                            "delegation_id": "del-1",
                            "result": "some result"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        // The immediately following user_message_stored is the noisy batch result
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "user_message_stored", "data": {
                            "content": "Delegation batch completed.\n\nResults:\n- del-1: completed"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        assert!(
            app.messages.is_empty(),
            "delegation batch result message must be suppressed"
        );

        // A subsequent real user message must NOT be suppressed
        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "user_message_stored", "data": {
                            "content": "real user message"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        assert_eq!(
            app.messages.len(),
            1,
            "real user message must not be suppressed"
        );
    }

    #[test]
    fn delegation_failed_suppresses_next_user_message() {
        let mut app = App::new();
        app.session_id = Some("parent".into());
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "delegation_failed", "data": {
                            "delegation_id": "del-1",
                            "error": "boom"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "user_message_stored", "data": {
                            "content": "Delegation batch failed.\n\nErrors:\n- del-1: boom"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        assert!(
            app.messages.is_empty(),
            "delegation failure result message must be suppressed"
        );
    }

    #[test]
    fn delegation_cancelled_sets_status() {
        let mut app = App::new();
        app.session_id = Some("parent".into());
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "delegation_cancelled", "data": {
                            "delegation_id": "del-1"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        assert_eq!(app.delegate_entries[0].status, DelegateStatus::Cancelled);
        assert!(
            app.suppress_delegation_result,
            "suppress flag must be set after cancellation"
        );
    }

    #[test]
    fn delegation_cancelled_suppresses_next_user_message() {
        let mut app = App::new();
        app.session_id = Some("parent".into());
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "delegation_cancelled", "data": {
                            "delegation_id": "del-1"
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "parent",
                "agent_id": "a1",
                "event": {
                    "type": "durable",
                    "data": {
                        "kind": { "type": "user_message_stored", "data": {
                            "content": "Delegation cancelled."
                        }},
                        "timestamp": null
                    }
                }
            })),
        });

        assert!(
            app.messages.is_empty(),
            "delegation cancellation result must be suppressed"
        );
    }

    // ── delegation duration tracking ─────────────────────────────────────────
}

// ── session_cache_tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod session_cache_tests {
    use super::*;

    fn set_ctx(app: &mut App, sid: &str, mode: &str, provider: &str, model: &str) {
        app.session_id = Some(sid.into());
        app.agent_mode = mode.into();
        app.current_provider = Some(provider.into());
        app.current_model = Some(model.into());
    }

    fn make_model_entry(provider: &str, model: &str) -> ModelEntry {
        ModelEntry {
            id: format!("{provider}/{model}"),
            label: model.into(),
            provider: provider.into(),
            model: model.into(),
            node_id: None,
            family: None,
            quant: None,
        }
    }

    // ── cache_session_mode_state ──────────────────────────────────────────────

    #[test]
    fn cache_stores_model_and_effort_under_session_and_mode() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = Some("high".into());

        app.cache_session_mode_state();

        let cms = &app.session_cache["s1"]["build"];
        assert_eq!(cms.model, "anthropic/claude-sonnet");
        assert_eq!(cms.effort, Some("high".into()));
    }

    #[test]
    fn cache_stores_auto_effort_as_none() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "plan", "openai", "gpt-4o");
        app.reasoning_effort = None;

        app.cache_session_mode_state();

        let cms = &app.session_cache["s1"]["plan"];
        assert_eq!(cms.model, "openai/gpt-4o");
        assert_eq!(cms.effort, None);
    }

    #[test]
    fn cache_noop_when_no_session_id() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        app.current_provider = Some("anthropic".into());
        app.current_model = Some("claude-sonnet".into());

        app.cache_session_mode_state();

        assert!(app.session_cache.is_empty());
    }

    #[test]
    fn cache_noop_when_no_provider_or_model() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();

        app.cache_session_mode_state();

        assert!(app.session_cache.is_empty());
    }

    #[test]
    fn cache_overwrites_existing_mode_entry() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = Some("low".into());
        app.cache_session_mode_state();

        // Switch model + effort, same session + mode
        app.current_model = Some("claude-opus".into());
        app.current_provider = Some("anthropic".into());
        app.reasoning_effort = Some("max".into());
        app.cache_session_mode_state();

        let cms = &app.session_cache["s1"]["build"];
        assert_eq!(cms.model, "anthropic/claude-opus");
        assert_eq!(cms.effort, Some("max".into()));
    }

    #[test]
    fn cache_different_modes_independent_within_session() {
        let mut app = App::new();

        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = Some("high".into());
        app.cache_session_mode_state();

        set_ctx(&mut app, "s1", "plan", "openai", "gpt-4o");
        app.reasoning_effort = Some("low".into());
        app.cache_session_mode_state();

        assert_eq!(
            app.session_cache["s1"]["build"].model,
            "anthropic/claude-sonnet"
        );
        assert_eq!(app.session_cache["s1"]["build"].effort, Some("high".into()));
        assert_eq!(app.session_cache["s1"]["plan"].model, "openai/gpt-4o");
        assert_eq!(app.session_cache["s1"]["plan"].effort, Some("low".into()));
    }

    #[test]
    fn cache_different_sessions_independent() {
        let mut app = App::new();

        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = Some("high".into());
        app.cache_session_mode_state();

        set_ctx(&mut app, "s2", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = Some("low".into());
        app.cache_session_mode_state();

        assert_eq!(app.session_cache["s1"]["build"].effort, Some("high".into()));
        assert_eq!(app.session_cache["s2"]["build"].effort, Some("low".into()));
    }

    // ── apply_cached_mode_state ───────────────────────────────────────────────

    #[test]
    fn apply_restores_effort_when_model_matches() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = None;

        app.session_cache.entry("s1".into()).or_default().insert(
            "build".into(),
            CachedModeState {
                model: "anthropic/claude-sonnet".into(),
                effort: Some("high".into()),
            },
        );

        let cmds = app.apply_cached_mode_state();
        assert_eq!(app.reasoning_effort, Some("high".into()));
        assert_eq!(cmds.len(), 1);
        assert!(
            matches!(&cmds[0], ClientMsg::SetReasoningEffort { reasoning_effort } if reasoning_effort == "high")
        );
    }

    #[test]
    fn apply_restores_model_and_effort_when_model_differs() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = None;
        // The cached state says build mode used opus with max effort
        app.session_cache.entry("s1".into()).or_default().insert(
            "build".into(),
            CachedModeState {
                model: "anthropic/claude-opus".into(),
                effort: Some("max".into()),
            },
        );
        // Need the model in the models list for the lookup
        app.models = vec![make_model_entry("anthropic", "claude-opus")];

        let cmds = app.apply_cached_mode_state();

        assert_eq!(app.current_provider.as_deref(), Some("anthropic"));
        assert_eq!(app.current_model.as_deref(), Some("claude-opus"));
        assert_eq!(app.reasoning_effort, Some("max".into()));
        assert_eq!(cmds.len(), 2);
        assert!(matches!(&cmds[0], ClientMsg::SetSessionModel { .. }));
        assert!(
            matches!(&cmds[1], ClientMsg::SetReasoningEffort { reasoning_effort } if reasoning_effort == "max")
        );
    }

    #[test]
    fn apply_returns_empty_when_no_cache_entry() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = Some("high".into());

        let cmds = app.apply_cached_mode_state();
        assert!(cmds.is_empty());
        // Nothing changed
        assert_eq!(app.reasoning_effort, Some("high".into()));
    }

    #[test]
    fn apply_returns_empty_when_everything_matches() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = Some("high".into());

        app.session_cache.entry("s1".into()).or_default().insert(
            "build".into(),
            CachedModeState {
                model: "anthropic/claude-sonnet".into(),
                effort: Some("high".into()),
            },
        );

        let cmds = app.apply_cached_mode_state();
        assert!(cmds.is_empty());
    }

    #[test]
    fn apply_returns_empty_when_no_session_id() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        app.current_provider = Some("anthropic".into());
        app.current_model = Some("claude-sonnet".into());
        app.reasoning_effort = Some("max".into());

        let cmds = app.apply_cached_mode_state();
        assert!(cmds.is_empty());
        assert_eq!(app.reasoning_effort, Some("max".into()));
    }

    #[test]
    fn apply_skips_model_switch_when_model_not_in_models_list() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");
        app.reasoning_effort = None;

        app.session_cache.entry("s1".into()).or_default().insert(
            "build".into(),
            CachedModeState {
                model: "anthropic/claude-opus".into(),
                effort: Some("max".into()),
            },
        );
        // models list is empty — can't resolve opus
        app.models = vec![];

        let cmds = app.apply_cached_mode_state();
        // Can't switch model, but effort still restored
        assert_eq!(app.current_model.as_deref(), Some("claude-sonnet")); // unchanged
        assert_eq!(app.reasoning_effort, Some("max".into()));
        assert_eq!(cmds.len(), 1);
        assert!(matches!(&cmds[0], ClientMsg::SetReasoningEffort { .. }));
    }

    // ── cycle auto-caches ─────────────────────────────────────────────────────

    #[test]
    fn cycle_caches_mode_state() {
        let mut app = App::new();
        set_ctx(&mut app, "s1", "build", "anthropic", "claude-sonnet");

        app.cycle_reasoning_effort();

        assert_eq!(app.reasoning_effort, Some("low".into()));
        let cms = &app.session_cache["s1"]["build"];
        assert_eq!(cms.model, "anthropic/claude-sonnet");
        assert_eq!(cms.effort, Some("low".into()));
    }

    #[test]
    fn cycle_does_not_cache_when_no_context() {
        let mut app = App::new();
        app.cycle_reasoning_effort();
        assert_eq!(app.reasoning_effort, Some("low".into()));
        assert!(app.session_cache.is_empty());
    }

    // ── update_cached_mode_model ──────────────────────────────────────────────

    #[test]
    fn update_cached_mode_model_stores_model_and_resets_effort() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.update_cached_mode_model("build", "anthropic", "claude-sonnet");

        let cms = &app.session_cache["s1"]["build"];
        assert_eq!(cms.model, "anthropic/claude-sonnet");
        assert_eq!(cms.effort, None);
    }

    #[test]
    fn update_cached_mode_model_preserves_other_modes() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.update_cached_mode_model("build", "anthropic", "claude-sonnet");
        app.update_cached_mode_model("plan", "openai", "gpt-4o");

        assert_eq!(
            app.session_cache["s1"]["build"].model,
            "anthropic/claude-sonnet"
        );
        assert_eq!(app.session_cache["s1"]["plan"].model, "openai/gpt-4o");
    }

    #[test]
    fn update_cached_mode_model_overwrites_existing_entry() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.session_cache.entry("s1".into()).or_default().insert(
            "build".into(),
            CachedModeState {
                model: "old/model".into(),
                effort: Some("high".into()),
            },
        );

        app.update_cached_mode_model("build", "anthropic", "claude-opus");

        let cms = &app.session_cache["s1"]["build"];
        assert_eq!(cms.model, "anthropic/claude-opus");
        assert_eq!(cms.effort, None);
    }

    #[test]
    fn update_cached_mode_model_noop_without_session() {
        let mut app = App::new();
        app.update_cached_mode_model("build", "anthropic", "claude-sonnet");
        assert!(app.session_cache.is_empty());
    }
}

// ── session_mode_tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod session_mode_tests {
    use super::*;

    fn provider_changed_event(provider: &str, model: &str) -> serde_json::Value {
        serde_json::json!({
            "kind": {
                "type": "provider_changed",
                "data": { "provider": provider, "model": model }
            }
        })
    }

    fn mode_changed_event(mode: &str) -> serde_json::Value {
        serde_json::json!({
            "kind": {
                "type": "session_mode_changed",
                "data": { "mode": mode }
            }
        })
    }

    fn make_audit(events: &[serde_json::Value]) -> serde_json::Value {
        serde_json::json!({ "events": events })
    }

    fn make_session_loaded(audit: serde_json::Value) -> RawServerMsg {
        RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "audit": audit,
                "undo_stack": []
            })),
        }
    }

    // ── SessionModeChanged in live events ─────────────────────────────────────

    #[test]
    fn live_session_mode_changed_updates_agent_mode() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        app.handle_event_kind(
            &EventKind::SessionModeChanged {
                mode: "plan".into(),
            },
            false,
            None,
        );
        assert_eq!(app.agent_mode, "plan");
    }

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

    #[test]
    fn live_session_mode_changed_to_build_updates_agent_mode() {
        let mut app = App::new();
        app.agent_mode = "plan".into();
        app.handle_event_kind(
            &EventKind::SessionModeChanged {
                mode: "build".into(),
            },
            false,
            None,
        );
        assert_eq!(app.agent_mode, "build");
    }

    // ── SessionModeChanged in audit replay ────────────────────────────────────

    #[test]
    fn replay_session_mode_changed_restores_mode() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        let audit = make_audit(&[mode_changed_event("plan")]);
        app.replay_audit(&audit, None);
        assert_eq!(app.agent_mode, "plan");
    }

    #[test]
    fn replay_last_session_mode_changed_wins() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        let audit = make_audit(&[
            mode_changed_event("plan"),
            mode_changed_event("build"),
            mode_changed_event("plan"),
        ]);
        app.replay_audit(&audit, None);
        assert_eq!(app.agent_mode, "plan");
    }

    #[test]
    fn replay_no_mode_change_leaves_agent_mode_unchanged() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        let audit = make_audit(&[provider_changed_event("anthropic", "claude-sonnet")]);
        app.replay_audit(&audit, None);
        assert_eq!(app.agent_mode, "build");
    }

    #[test]
    fn live_session_mode_changed_away_from_review_clears_previous_mode() {
        let mut app = App::new();
        app.agent_mode = "review".into();
        app.mode_before_review = Some("plan".into());
        app.handle_event_kind(
            &EventKind::SessionModeChanged {
                mode: "build".into(),
            },
            false,
            None,
        );
        assert_eq!(app.agent_mode, "build");
        assert_eq!(app.mode_before_review, None);
    }

    // ── session_loaded returns SetAgentMode ───────────────────────────────────

    #[test]
    fn session_loaded_returns_set_agent_mode_from_audit() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        let audit = make_audit(&[mode_changed_event("plan")]);
        let cmds = app.handle_server_msg(make_session_loaded(audit));
        assert!(
            cmds.iter().any(|m| matches!(
                m,
                ClientMsg::SetAgentMode { mode } if mode == "plan"
            )),
            "expected SetAgentMode(plan) in {cmds:?}"
        );
    }

    #[test]
    fn session_loaded_always_returns_set_agent_mode_even_without_mode_event() {
        let mut app = App::new();
        app.agent_mode = "build".into();
        let audit = make_audit(&[]);
        let cmds = app.handle_server_msg(make_session_loaded(audit));
        // No SessionModeChanged → agent_mode stays "build"; command still sent
        assert!(
            cmds.iter().any(|m| matches!(
                m,
                ClientMsg::SetAgentMode { mode } if mode == "build"
            )),
            "expected SetAgentMode(build) in {cmds:?}"
        );
    }

    #[test]
    fn session_loaded_review_mode_does_not_create_previous_mode() {
        let mut app = App::new();
        let audit = make_audit(&[mode_changed_event("review")]);
        let cmds = app.handle_server_msg(make_session_loaded(audit));
        assert!(
            cmds.iter().any(|m| matches!(
                m,
                ClientMsg::SetAgentMode { mode } if mode == "review"
            )),
            "expected SetAgentMode(review) in {cmds:?}"
        );
        assert_eq!(app.agent_mode, "review");
        assert_eq!(app.mode_before_review, None);
        assert_eq!(app.next_mode(), "build");
    }

    // ── session_loaded restores mode state from session cache ──────────────────

    #[test]
    fn session_loaded_restores_effort_from_session_cache() {
        let mut app = App::new();
        // Pre-cache: session s1, mode plan, model anthropic/claude-sonnet, effort high
        app.session_cache.entry("s1".into()).or_default().insert(
            "plan".into(),
            CachedModeState {
                model: "anthropic/claude-sonnet".into(),
                effort: Some("high".into()),
            },
        );

        let audit = make_audit(&[
            provider_changed_event("anthropic", "claude-sonnet"),
            mode_changed_event("plan"),
        ]);
        let cmds = app.handle_server_msg(make_session_loaded(audit));
        assert!(
            cmds.iter().any(|m| matches!(
                m,
                ClientMsg::SetReasoningEffort { reasoning_effort } if reasoning_effort == "high"
            )),
            "expected SetReasoningEffort(high) in {cmds:?}"
        );
        assert_eq!(app.reasoning_effort, Some("high".into()));
    }

    #[test]
    fn session_loaded_restores_model_from_session_cache() {
        let mut app = App::new();
        // Cache says plan mode used opus
        app.session_cache.entry("s1".into()).or_default().insert(
            "plan".into(),
            CachedModeState {
                model: "anthropic/claude-opus".into(),
                effort: Some("max".into()),
            },
        );
        // Need opus in the models list
        app.models = vec![ModelEntry {
            id: "anthropic/claude-opus".into(),
            label: "claude-opus".into(),
            provider: "anthropic".into(),
            model: "claude-opus".into(),
            node_id: None,
            family: None,
            quant: None,
        }];

        // Audit says session was in plan mode using sonnet (different from cache)
        let audit = make_audit(&[
            provider_changed_event("anthropic", "claude-sonnet"),
            mode_changed_event("plan"),
        ]);
        let cmds = app.handle_server_msg(make_session_loaded(audit));

        // Cache wins: model switched to opus
        assert!(
            cmds.iter()
                .any(|m| matches!(m, ClientMsg::SetSessionModel { .. })),
            "expected SetSessionModel in {cmds:?}"
        );
        assert_eq!(app.current_model.as_deref(), Some("claude-opus"));
    }

    #[test]
    fn session_loaded_no_cache_entry_returns_no_effort_or_model_cmds() {
        let mut app = App::new();
        app.reasoning_effort = None;
        let audit = make_audit(&[
            provider_changed_event("anthropic", "claude-sonnet"),
            mode_changed_event("plan"),
        ]);

        let cmds = app.handle_server_msg(make_session_loaded(audit));
        // Only SetAgentMode, no SetReasoningEffort or SetSessionModel
        assert!(
            !cmds
                .iter()
                .any(|m| matches!(m, ClientMsg::SetReasoningEffort { .. })),
            "expected no SetReasoningEffort: {cmds:?}"
        );
        assert!(
            !cmds
                .iter()
                .any(|m| matches!(m, ClientMsg::SetSessionModel { .. })),
            "expected no SetSessionModel: {cmds:?}"
        );
    }

    // ── handle_server_msg returns Vec now (backward compat for other msgs) ────

    #[test]
    fn state_msg_returns_empty_vec() {
        let mut app = App::new();
        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "state".into(),
            data: Some(serde_json::json!({
                "active_session_id": null,
                "agents": [],
                "agent_mode": "build"
            })),
        });
        assert!(cmds.is_empty());
    }

    #[test]
    fn session_created_returns_subscribe_in_vec() {
        let mut app = App::new();
        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "session_created".into(),
            data: Some(serde_json::json!({
                "session_id": "s99",
                "agent_id": "a1",
                "request_id": null
            })),
        });
        assert!(
            cmds.iter().any(|m| matches!(m, ClientMsg::SubscribeSession { session_id, .. } if session_id == "s99")),
            "expected SubscribeSession in {cmds:?}"
        );
    }

    #[test]
    fn session_created_applies_mode_model_preference() {
        let mut app = App::new();
        app.models = vec![crate::protocol::ModelEntry {
            id: "anthropic/claude-sonnet".into(),
            label: "claude-sonnet".into(),
            provider: "anthropic".into(),
            model: "claude-sonnet".into(),
            node_id: None,
            family: None,
            quant: None,
        }];
        app.set_mode_model_preference("build", "anthropic", "claude-sonnet");

        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "session_created".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "request_id": null
            })),
        });
        assert!(
            cmds.iter().any(
                |m| matches!(m, ClientMsg::SetSessionModel { session_id, .. } if session_id == "s1")
            ),
            "expected SetSessionModel for new session: {cmds:?}"
        );
        assert_eq!(app.current_provider.as_deref(), Some("anthropic"));
        assert_eq!(app.current_model.as_deref(), Some("claude-sonnet"));
    }

    #[test]
    fn session_created_no_preference_uses_server_default() {
        let mut app = App::new();
        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "session_created".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "request_id": null
            })),
        });
        // Only SubscribeSession, no SetSessionModel
        assert_eq!(cmds.len(), 1);
        assert!(
            matches!(&cmds[0], ClientMsg::SubscribeSession { .. }),
            "expected only SubscribeSession: {cmds:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server_msg::backfill_elicitation_outcomes;

    fn make_turn(message_id: &str) -> UndoableTurn {
        UndoableTurn {
            turn_id: format!("turn-{message_id}"),
            message_id: message_id.into(),
            text: format!("prompt {message_id}"),
        }
    }

    fn make_stack(ids: &[&str]) -> Vec<UndoStackFrame> {
        ids.iter()
            .map(|id| UndoStackFrame {
                message_id: (*id).into(),
            })
            .collect()
    }

    fn audit_event(kind_type: &str, data: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "kind": {
                "type": kind_type,
                "data": data
            },
            "timestamp": null
        })
    }

    fn durable_event(kind_type: &str, data: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "type": "durable",
            "data": audit_event(kind_type, data)
        })
    }

    fn durable_prompt(text: &str, message_id: &str) -> serde_json::Value {
        durable_event(
            "prompt_received",
            serde_json::json!({
                "content": text,
                "message_id": message_id,
            }),
        )
    }

    fn durable_user_stored(text: &str) -> serde_json::Value {
        durable_event(
            "user_message_stored",
            serde_json::json!({
                "content": text,
            }),
        )
    }

    fn durable_assistant(content: &str, message_id: &str) -> serde_json::Value {
        durable_event(
            "assistant_message_stored",
            serde_json::json!({
                "content": content,
                "thinking": null,
                "message_id": message_id,
            }),
        )
    }

    fn durable_tool_call_start(tool_call_id: &str, tool_name: &str) -> serde_json::Value {
        durable_event(
            "tool_call_start",
            serde_json::json!({
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": null,
            }),
        )
    }

    fn durable_tool_call_end(
        tool_call_id: &str,
        tool_name: &str,
        result: &str,
    ) -> serde_json::Value {
        durable_event(
            "tool_call_end",
            serde_json::json!({
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "is_error": false,
                "result": result,
            }),
        )
    }

    fn session_events_msg(events: Vec<serde_json::Value>) -> RawServerMsg {
        RawServerMsg {
            msg_type: "session_events".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "events": events,
            })),
        }
    }

    fn make_session_loaded(audit: serde_json::Value) -> RawServerMsg {
        RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "audit": audit,
                "undo_stack": []
            })),
        }
    }

    fn durable_progress_recorded() -> serde_json::Value {
        durable_event(
            "progress_recorded",
            serde_json::json!({
                "progress_entry": {
                    "kind": "note",
                    "content": "received response",
                    "metadata": null,
                    "created_at": "2026-04-13T00:00:00Z"
                }
            }),
        )
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
    fn raw_capability_messages_log_without_unknown_warning_or_chat() {
        let mut app = App::new();

        let audio_cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "audio_capabilities".into(),
            data: Some(serde_json::json!({
                "stt_models": [
                    { "provider": "izwi", "model": "stt-small" },
                    { "provider": "mistralrs", "model": "stt-local" }
                ],
                "tts_models": [
                    { "provider": "izwi", "model": "tts-small" }
                ]
            })),
        });
        let provider_cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "provider_capabilities".into(),
            data: Some(serde_json::json!({
                "providers": [
                    { "provider": "openai", "supports_custom_models": true },
                    { "provider": "anthropic", "supports_custom_models": false }
                ]
            })),
        });

        assert!(audio_cmds.is_empty());
        assert!(provider_cmds.is_empty());
        assert!(app.messages.is_empty());
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "audio"
                && entry.message == "audio: 2 STT, 1 TTS (izwi, mistralrs)"
        }));
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "models"
                && entry.message
                    == "models: 2 provider capability entrie(s), 1 support custom models"
        }));
        assert!(app.logs.iter().all(|entry| {
            !entry.message.contains("audio_capabilities")
                && !entry.message.contains("provider_capabilities")
        }));
    }

    #[test]
    fn unknown_server_message_type_logs_warning() {
        let mut app = App::new();

        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "future_server_message".into(),
            data: Some(serde_json::json!({ "ignored": true })),
        });

        assert!(cmds.is_empty());
        assert_eq!(app.status, "connecting...");
        let last = app.logs.last().expect("missing warning log entry");
        assert_eq!(last.level, LogLevel::Warn);
        assert_eq!(last.target, "protocol");
        assert!(last.message.contains("future_server_message"));
    }

    #[test]
    fn session_events_unknown_kind_logs_warning_and_keeps_known_events() {
        let mut app = App::new();
        app.session_id = Some("s1".into());

        app.handle_server_msg(session_events_msg(vec![
            durable_prompt("hello", "msg-1"),
            durable_event(
                "brand_new_unknown_event_2099",
                serde_json::json!({
                    "some_field": "some_value",
                    "nested": { "deep": true }
                }),
            ),
            durable_assistant("world", "msg-2"),
        ]));

        assert!(matches!(
            app.messages.first(),
            Some(ChatEntry::User { text, message_id }) if text == "hello" && message_id.as_deref() == Some("msg-1")
        ));
        assert!(matches!(
            app.messages.last(),
            Some(ChatEntry::Assistant { content, message_id, .. }) if content == "world" && message_id.as_deref() == Some("msg-2")
        ));
        let last = app.logs.last().expect("missing warning log entry");
        assert_eq!(last.level, LogLevel::Warn);
        assert_eq!(last.target, "protocol");
        assert!(last.message.contains("brand_new_unknown_event_2099"));
    }

    #[test]
    fn artifact_recorded_session_event_replay_skips_log_without_unknown_warning() {
        let mut app = App::new();
        app.session_id = Some("s1".into());

        app.handle_server_msg(session_events_msg(vec![durable_event(
            "artifact_recorded",
            serde_json::json!({
                "artifact": {
                    "kind": "file",
                    "uri": null,
                    "path": "src/generated.txt",
                    "summary": "Produced by write_file",
                    "created_at": "2026-04-29T14:25:09Z"
                }
            }),
        )]));

        assert!(app.messages.is_empty());
        assert!(app.logs.iter().all(|entry| entry.target != "artifact"));
        assert!(
            app.logs
                .iter()
                .all(|entry| !entry.message.contains("unknown session_events kind"))
        );
    }

    #[test]
    fn artifact_recorded_live_event_logs_only() {
        let mut app = App::new();
        app.session_id = Some("s1".into());

        app.handle_server_msg(RawServerMsg {
            msg_type: "event".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "event": durable_event(
                    "artifact_recorded",
                    serde_json::json!({
                        "artifact": {
                            "kind": "file",
                            "uri": null,
                            "path": "src/generated.txt",
                            "summary": "Produced by write_file",
                            "created_at": "2026-04-29T14:25:09Z"
                        }
                    })
                )
            })),
        });

        assert!(app.messages.is_empty());
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "artifact"
                && entry.message
                    == "artifact recorded: file src/generated.txt (Produced by write_file)"
        }));
        assert!(
            app.logs
                .iter()
                .all(|entry| !entry.message.contains("unknown event kind"))
        );
    }

    #[test]
    fn artifact_recorded_replay_skips_log_and_chat() {
        let mut app = App::new();
        app.session_id = Some("s1".into());

        app.handle_server_msg(make_session_loaded(serde_json::json!({
            "events": [audit_event(
                "artifact_recorded",
                serde_json::json!({
                    "artifact": {
                        "kind": "file",
                        "uri": null,
                        "path": "src/generated.txt",
                        "summary": "Produced by write_file",
                        "created_at": "2026-04-29T14:25:09Z"
                    }
                })
            )]
        })));

        assert!(app.messages.is_empty());
        assert!(app.logs.iter().all(
            |entry| entry.target != "artifact" && !entry.message.contains("artifact recorded")
        ));
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
    fn build_undo_state_returns_none_for_empty_stack() {
        let app = App::new();
        assert_eq!(
            app.build_undo_state_from_server_stack(&[], None, None),
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
    fn compaction_events_replace_live_indicator_with_summary_card() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::CompactionStart {
                token_estimate: 12_000,
            },
            false,
            None,
        );
        assert_eq!(
            app.activity,
            ActivityState::Compacting {
                token_estimate: 12_000
            }
        );
        assert!(matches!(
            app.messages.last(),
            Some(ChatEntry::CompactionStart {
                token_estimate: 12_000
            })
        ));

        app.handle_event_kind(
            &EventKind::CompactionEnd {
                summary: "Trimmed tool output".into(),
                summary_len: 19,
            },
            false,
            None,
        );

        assert_eq!(app.activity, ActivityState::Thinking);
        assert!(
            matches!(app.messages.last(), Some(ChatEntry::CompactionEnd { token_estimate: Some(12_000), summary, summary_len }) if summary == "Trimmed tool output" && *summary_len == 19)
        );
        assert!(
            !app.messages
                .iter()
                .any(|entry| matches!(entry, ChatEntry::CompactionStart { .. }))
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

    #[test]
    fn undo_and_redo_results_clear_pending_session_op() {
        let mut app = App::new();
        app.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.handle_server_msg(RawServerMsg {
            msg_type: "undo_result".into(),
            data: Some(serde_json::json!({
                "success": false,
                "message": "undo failed",
                "undo_stack": []
            })),
        });
        assert_eq!(app.activity, ActivityState::Idle);

        app.activity = ActivityState::SessionOp(SessionOp::Redo);
        app.handle_server_msg(RawServerMsg {
            msg_type: "redo_result".into(),
            data: Some(serde_json::json!({
                "success": false,
                "message": "redo failed",
                "undo_stack": []
            })),
        });
        assert_eq!(app.activity, ActivityState::Idle);
    }

    #[test]
    fn turn_activity_transitions_across_tool_and_completion_events() {
        let mut app = App::new();

        app.handle_event_kind(&EventKind::TurnStarted, false, None);
        assert_eq!(app.activity, ActivityState::Thinking);

        app.handle_event_kind(
            &EventKind::AssistantMessageStored {
                content: "draft".into(),
                thinking: None,
                message_id: None,
            },
            false,
            None,
        );
        assert_eq!(app.activity, ActivityState::Thinking);

        app.handle_event_kind(
            &EventKind::ToolCallStart {
                tool_call_id: Some("call-1".into()),
                tool_name: "read_tool".into(),
                arguments: None,
            },
            false,
            None,
        );
        assert_eq!(
            app.activity,
            ActivityState::RunningTool {
                name: "read_tool".into()
            }
        );

        app.handle_event_kind(
            &EventKind::LlmRequestEnd {
                finish_reason: None,
                cost_usd: None,
                cumulative_cost_usd: None,
                context_tokens: None,
                tool_calls: None,
                metrics: None,
            },
            false,
            None,
        );
        assert_eq!(app.activity, ActivityState::Idle);
    }

    #[test]
    fn replay_audit_prunes_frontier_and_later_events_after_undo() {
        let mut app = App::new();
        app.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-2".into(),
                message_id: "msg-2".into(),
                status: UndoFrameStatus::Confirmed,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-2".into()),
        });

        let audit = serde_json::json!({
            "events": [
                {
                    "kind": {
                        "type": "prompt_received",
                        "data": {
                            "content": [{ "type": "text", "text": "first" }],
                            "message_id": "msg-1"
                        }
                    }
                },
                {
                    "kind": {
                        "type": "assistant_message_stored",
                        "data": {
                            "content": "reply one",
                            "thinking": null,
                            "message_id": "a-1"
                        }
                    }
                },
                {
                    "kind": {
                        "type": "prompt_received",
                        "data": {
                            "content": [{ "type": "text", "text": "second" }],
                            "message_id": "msg-2"
                        }
                    }
                },
                {
                    "kind": {
                        "type": "assistant_message_stored",
                        "data": {
                            "content": "reply two",
                            "thinking": null,
                            "message_id": "a-2"
                        }
                    }
                }
            ]
        });

        app.replay_audit(&audit, None);

        assert_eq!(app.messages.len(), 2);
        assert!(
            matches!(&app.messages[0], ChatEntry::User { text, message_id: Some(message_id) } if text == "first" && message_id == "msg-1")
        );
        assert!(
            matches!(&app.messages[1], ChatEntry::Assistant { content, .. } if content == "reply one")
        );
        assert_eq!(app.undoable_turns.len(), 1);
        assert_eq!(app.undoable_turns[0].message_id, "msg-1");
        assert!(app.can_redo());
    }

    #[test]
    fn prompt_received_with_same_message_id_is_not_duplicated() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::PromptReceived {
                content: serde_json::json!("test ping"),
                message_id: Some("msg-1".into()),
            },
            false,
            None,
        );

        app.handle_event_kind(
            &EventKind::PromptReceived {
                content: serde_json::json!("test ping"),
                message_id: Some("msg-1".into()),
            },
            false,
            None,
        );

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.undoable_turns.len(), 1);
    }

    #[test]
    fn assistant_message_stored_with_same_message_id_is_not_duplicated() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::AssistantMessageStored {
                content: "Pong".into(),
                thinking: None,
                message_id: Some("a-1".into()),
            },
            false,
            None,
        );

        app.handle_event_kind(
            &EventKind::AssistantMessageStored {
                content: "Pong".into(),
                thinking: None,
                message_id: Some("a-1".into()),
            },
            false,
            None,
        );

        assert_eq!(app.messages.len(), 1);
    }

    #[test]
    fn user_message_stored_only_backfills_when_prompt_received_missing() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::PromptReceived {
                content: serde_json::json!("test ping"),
                message_id: Some("msg-1".into()),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::UserMessageStored {
                content: serde_json::json!("test ping"),
            },
            false,
            None,
        );

        assert_eq!(app.messages.len(), 1);
        assert!(matches!(
            &app.messages[0],
            ChatEntry::User { text, message_id: Some(message_id) }
                if text == "test ping" && message_id == "msg-1"
        ));

        let mut fallback_only = App::new();
        fallback_only.handle_event_kind(
            &EventKind::UserMessageStored {
                content: serde_json::json!("fallback prompt"),
            },
            false,
            None,
        );
        assert_eq!(fallback_only.messages.len(), 1);
        assert!(matches!(
            &fallback_only.messages[0],
            ChatEntry::User { text, message_id: None } if text == "fallback prompt"
        ));
    }

    #[test]
    fn live_frontier_events_do_not_resurrect_undone_prompt() {
        let mut app = App::new();
        app.messages.push(ChatEntry::User {
            text: "first".into(),
            message_id: Some("msg-1".into()),
        });
        app.messages.push(ChatEntry::Assistant {
            content: "reply one".into(),
            thinking: None,
            message_id: None,
        });
        app.undoable_turns = vec![UndoableTurn {
            turn_id: "turn-msg-1".into(),
            message_id: "msg-1".into(),
            text: "first".into(),
        }];
        app.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-2".into(),
                message_id: "msg-2".into(),
                status: UndoFrameStatus::Confirmed,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-2".into()),
        });

        app.handle_event_kind(
            &EventKind::PromptReceived {
                content: serde_json::json!([{ "type": "text", "text": "second" }]),
                message_id: Some("msg-2".into()),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::UserMessageStored {
                content: serde_json::json!("second"),
            },
            false,
            None,
        );

        assert_eq!(app.messages.len(), 2);
        assert!(app.messages.iter().all(
            |entry| !matches!(entry, ChatEntry::User { message_id: Some(message_id), .. } if message_id == "msg-2")
        ));
        assert!(
            app.messages
                .iter()
                .all(|entry| !matches!(entry, ChatEntry::User { text, .. } if text == "second"))
        );
        assert_eq!(app.undoable_turns.len(), 1);
        assert_eq!(app.undoable_turns[0].message_id, "msg-1");
        assert_eq!(
            app.undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("msg-2")
        );
    }

    #[test]
    fn parent_session_events_replay_is_idempotent_for_existing_history() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.agent_id = Some("a1".into());

        let events = vec![
            durable_prompt("test ping", "msg-1"),
            durable_user_stored("test ping"),
            durable_assistant("Pong", "a-1"),
            durable_prompt("nice", "msg-2"),
            durable_user_stored("nice"),
            durable_assistant("Great", "a-2"),
        ];

        app.handle_server_msg(session_events_msg(events.clone()));
        app.handle_server_msg(session_events_msg(events));

        assert_eq!(app.messages.len(), 4);
        assert_eq!(app.undoable_turns.len(), 2);
    }

    #[test]
    fn current_session_history_replay_keeps_tool_calls_with_their_turns() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.agent_id = Some("a1".into());

        let history = vec![
            durable_prompt("first", "msg-1"),
            durable_tool_call_start("tool-1", "read_tool"),
            durable_tool_call_end("tool-1", "read_tool", "first output"),
            durable_assistant("first reply", "a-1"),
            durable_prompt("second", "msg-2"),
            durable_tool_call_start("tool-2", "ls"),
            durable_tool_call_end("tool-2", "ls", "second output"),
            durable_assistant("second reply", "a-2"),
        ];

        app.handle_server_msg(make_session_loaded(serde_json::json!({
            "events": [
                audit_event(
                    "prompt_received",
                    serde_json::json!({
                        "content": "first",
                        "message_id": "msg-1"
                    }),
                ),
                audit_event(
                    "tool_call_start",
                    serde_json::json!({
                        "tool_call_id": "tool-1",
                        "tool_name": "read_tool",
                        "arguments": null
                    }),
                ),
                audit_event(
                    "tool_call_end",
                    serde_json::json!({
                        "tool_call_id": "tool-1",
                        "tool_name": "read_tool",
                        "is_error": false,
                        "result": "first output"
                    }),
                ),
                audit_event(
                    "assistant_message_stored",
                    serde_json::json!({
                        "content": "first reply",
                        "thinking": null,
                        "message_id": "a-1"
                    }),
                ),
                audit_event(
                    "prompt_received",
                    serde_json::json!({
                        "content": "second",
                        "message_id": "msg-2"
                    }),
                ),
                audit_event(
                    "tool_call_start",
                    serde_json::json!({
                        "tool_call_id": "tool-2",
                        "tool_name": "ls",
                        "arguments": null
                    }),
                ),
                audit_event(
                    "tool_call_end",
                    serde_json::json!({
                        "tool_call_id": "tool-2",
                        "tool_name": "ls",
                        "is_error": false,
                        "result": "second output"
                    }),
                ),
                audit_event(
                    "assistant_message_stored",
                    serde_json::json!({
                        "content": "second reply",
                        "thinking": null,
                        "message_id": "a-2"
                    }),
                ),
            ],
        })));
        app.handle_server_msg(session_events_msg(vec![
            history[0].clone(),
            durable_event(
                "brand_new_unknown_event_2099",
                serde_json::json!({ "some_field": true }),
            ),
            history[1].clone(),
            history[2].clone(),
            history[3].clone(),
            history[4].clone(),
            history[5].clone(),
            history[6].clone(),
            history[7].clone(),
        ]));

        assert_eq!(app.messages.len(), 6);
        assert!(matches!(
            &app.messages[0],
            ChatEntry::User { text, message_id: Some(message_id) }
                if text == "first" && message_id == "msg-1"
        ));
        assert!(matches!(
            &app.messages[1],
            ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), name, .. }
                if tool_call_id == "tool-1" && name == "read_tool"
        ));
        assert!(matches!(
            &app.messages[2],
            ChatEntry::Assistant { content, message_id: Some(message_id), .. }
                if content == "first reply" && message_id == "a-1"
        ));
        assert!(matches!(
            &app.messages[3],
            ChatEntry::User { text, message_id: Some(message_id) }
                if text == "second" && message_id == "msg-2"
        ));
        assert!(matches!(
            &app.messages[4],
            ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), name, .. }
                if tool_call_id == "tool-2" && name == "ls"
        ));
        assert!(matches!(
            &app.messages[5],
            ChatEntry::Assistant { content, message_id: Some(message_id), .. }
                if content == "second reply" && message_id == "a-2"
        ));
    }

    #[test]
    fn undo_then_session_loaded_prunes_reverted_turn_and_later_history() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.agent_id = Some("a1".into());
        app.messages = vec![
            ChatEntry::User {
                text: "test ping".into(),
                message_id: Some("msg-1".into()),
            },
            ChatEntry::Assistant {
                content: "Pong".into(),
                thinking: None,
                message_id: None,
            },
            ChatEntry::User {
                text: "nice".into(),
                message_id: Some("msg-2".into()),
            },
            ChatEntry::Assistant {
                content: "Great".into(),
                thinking: None,
                message_id: None,
            },
        ];
        app.undoable_turns = vec![
            UndoableTurn {
                turn_id: "turn-msg-1".into(),
                message_id: "msg-1".into(),
                text: "test ping".into(),
            },
            UndoableTurn {
                turn_id: "turn-msg-2".into(),
                message_id: "msg-2".into(),
                text: "nice".into(),
            },
        ];

        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "undo_result".into(),
            data: Some(serde_json::json!({
                "success": true,
                "message_id": "msg-2",
                "reverted_files": [],
                "undo_stack": [{ "message_id": "msg-2" }]
            })),
        });
        assert!(
            cmds.iter().any(
                |msg| matches!(msg, ClientMsg::LoadSession { session_id } if session_id == "s1")
            )
        );

        let loaded = RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "undo_stack": [{ "message_id": "msg-2" }],
                "audit": {
                    "events": [
                        {
                            "kind": {
                                "type": "prompt_received",
                                "data": {
                                    "content": "test ping",
                                    "message_id": "msg-1"
                                }
                            }
                        },
                        {
                            "kind": {
                                "type": "assistant_message_stored",
                                "data": {
                                    "content": "Pong",
                                    "thinking": null,
                                    "message_id": "a-1"
                                }
                            }
                        },
                        {
                            "kind": {
                                "type": "prompt_received",
                                "data": {
                                    "content": "nice",
                                    "message_id": "msg-2"
                                }
                            }
                        },
                        {
                            "kind": {
                                "type": "assistant_message_stored",
                                "data": {
                                    "content": "Great",
                                    "thinking": null,
                                    "message_id": "a-2"
                                }
                            }
                        }
                    ]
                }
            })),
        };
        app.handle_server_msg(loaded);

        assert_eq!(app.messages.len(), 2);
        assert!(matches!(
            &app.messages[0],
            ChatEntry::User { text, message_id: Some(message_id) }
                if text == "test ping" && message_id == "msg-1"
        ));
        assert!(matches!(
            &app.messages[1],
            ChatEntry::Assistant { content, .. } if content == "Pong"
        ));
        assert_eq!(app.undoable_turns.len(), 1);
        assert_eq!(app.undoable_turns[0].message_id, "msg-1");
    }

    #[test]
    fn undo_reload_clears_stale_tool_and_cancelled_rows() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.agent_id = Some("a1".into());
        app.messages = vec![
            ChatEntry::User {
                text: "test".into(),
                message_id: Some("msg-1".into()),
            },
            ChatEntry::Assistant {
                content: "Hello! I'm ready to help. What can I do for you?".into(),
                thinking: None,
                message_id: None,
            },
            ChatEntry::User {
                text: "wdyt does undo functionaliy works?".into(),
                message_id: Some("msg-2".into()),
            },
            ChatEntry::Assistant {
                content: "Let me take a look at the codebase to understand what undo functionality exists and how it works.".into(),
                thinking: None,
                message_id: None,
            },
            ChatEntry::ToolCall {
                tool_call_id: Some("tool-1".into()),
                name: "ls".into(),
                is_error: false,
                detail: ToolDetail::None,
            },
            ChatEntry::Assistant {
                content: "/projects/personal/test/\n(0 entries)\n [cancelled]".into(),
                thinking: None,
                message_id: None,
            },
        ];
        app.undoable_turns = vec![
            UndoableTurn {
                turn_id: "turn-msg-1".into(),
                message_id: "msg-1".into(),
                text: "test".into(),
            },
            UndoableTurn {
                turn_id: "turn-msg-2".into(),
                message_id: "msg-2".into(),
                text: "wdyt does undo functionaliy works?".into(),
            },
        ];

        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "undo_result".into(),
            data: Some(serde_json::json!({
                "success": true,
                "message_id": "msg-2",
                "reverted_files": [],
                "undo_stack": [{ "message_id": "msg-2" }]
            })),
        });
        assert!(
            cmds.iter().any(
                |msg| matches!(msg, ClientMsg::LoadSession { session_id } if session_id == "s1")
            )
        );

        app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "s1",
                "agent_id": "a1",
                "undo_stack": [{ "message_id": "msg-2" }],
                "audit": {
                    "events": [
                        {
                            "kind": {
                                "type": "prompt_received",
                                "data": {
                                    "content": "test",
                                    "message_id": "msg-1"
                                }
                            }
                        },
                        {
                            "kind": {
                                "type": "assistant_message_stored",
                                "data": {
                                    "content": "Hello! I'm ready to help. What can I do for you?",
                                    "thinking": null,
                                    "message_id": "a-1"
                                }
                            }
                        }
                    ]
                }
            })),
        });

        assert_eq!(app.messages.len(), 2);
        assert!(
            app.messages
                .iter()
                .all(|entry| !matches!(entry, ChatEntry::ToolCall { .. }))
        );
        assert!(app.messages.iter().all(
            |entry| !matches!(entry, ChatEntry::Assistant { content, .. } if content.contains("cancelled"))
        ));
        assert!(app.messages.iter().all(
            |entry| !matches!(entry, ChatEntry::User { text, .. } if text == "wdyt does undo functionaliy works?")
        ));
    }

    #[test]
    fn live_next_protocol_events_log_without_chat_messages() {
        let mut app = App::new();

        app.handle_event_kind(
            &EventKind::SessionQueued {
                reason: "waiting for previous operation to complete".into(),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::SessionConfigured {
                cwd: Some("/workspace/project".into()),
                mcp_servers: vec![],
                limits: Some(SessionLimits {
                    max_steps: Some(200),
                    max_turns: Some(50),
                    max_cost_usd: None,
                }),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::ToolsAvailable {
                tools: ["delegate", "ls", "question", "glob", "shell", "read_tool"]
                    .into_iter()
                    .map(|name| ToolInfo {
                        tool_type: "function".into(),
                        function: Some(FunctionToolInfo {
                            name: name.into(),
                            description: Some(format!("{name} tool")),
                            parameters: Some(serde_json::json!({ "type": "object" })),
                        }),
                    })
                    .collect(),
                tools_hash: Some(serde_json::json!("123456789")),
            },
            false,
            None,
        );

        assert!(app.messages.is_empty());
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Warn
                && entry.target == "session"
                && entry.message == "session queued: waiting for previous operation to complete"
        }));
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "session"
                && entry
                    .message
                    .contains("session configured: cwd /workspace/project")
        }));
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "tools"
                && entry.message
                    == "tools available: 6 tool(s) (delegate, ls, question, glob, shell, read_tool)"
        }));
        assert!(
            app.logs
                .iter()
                .all(|entry| !entry.message.contains("+2 more"))
        );
    }

    #[test]
    fn live_snapshot_and_progress_events_log_without_chat_messages() {
        let mut app = App::new();
        let initial_messages = app.messages.len();

        app.handle_event_kind(
            &EventKind::SnapshotStart {
                policy: "diff".into(),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::SnapshotEnd {
                summary: Some("1 modified".into()),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::ProgressRecorded {
                progress_entry: ProgressEntry {
                    kind: ProgressKind::Note,
                    content: "Received response from LLM".into(),
                    metadata: None,
                    created_at: "2026-04-13T00:00:00Z".into(),
                },
            },
            false,
            None,
        );

        assert_eq!(app.messages.len(), initial_messages);
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "snapshot"
                && entry.message == "starting diff snapshot"
        }));
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Info
                && entry.target == "snapshot"
                && entry.message == "snapshot: 1 modified"
        }));
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "progress"
                && entry.message == "progress: Received response from LLM"
        }));
    }

    #[test]
    fn replay_next_protocol_events_skip_logs_and_chat_messages() {
        let mut app = App::new();

        app.handle_event_kind(
            &EventKind::SessionQueued {
                reason: "waiting".into(),
            },
            true,
            None,
        );
        app.handle_event_kind(
            &EventKind::SessionConfigured {
                cwd: None,
                mcp_servers: vec![],
                limits: None,
            },
            true,
            None,
        );
        app.handle_event_kind(
            &EventKind::ToolsAvailable {
                tools: vec![],
                tools_hash: None,
            },
            true,
            None,
        );

        assert!(app.messages.is_empty());
        assert!(app.logs.is_empty());
    }

    #[test]
    fn replay_snapshot_and_progress_events_skip_logs_and_chat_messages() {
        let mut app = App::new();

        app.handle_event_kind(
            &EventKind::SnapshotEnd {
                summary: Some("No changes".into()),
            },
            true,
            None,
        );
        app.handle_event_kind(
            &EventKind::ProgressRecorded {
                progress_entry: ProgressEntry {
                    kind: ProgressKind::Note,
                    content: "Received response from LLM".into(),
                    metadata: None,
                    created_at: "2026-04-13T00:00:00Z".into(),
                },
            },
            true,
            None,
        );

        assert!(app.messages.is_empty());
        assert!(app.logs.is_empty());
    }

    #[test]
    fn parent_session_next_protocol_events_do_not_warn_or_add_chat() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.agent_id = Some("a1".into());
        app.messages.push(ChatEntry::User {
            text: "existing".into(),
            message_id: Some("existing-msg".into()),
        });

        app.handle_server_msg(session_events_msg(vec![
            durable_prompt("hello", "msg-1"),
            durable_event(
                "session_configured",
                serde_json::json!({
                    "cwd": "/workspace/project",
                    "mcp_servers": [],
                    "limits": { "max_steps": 200, "max_turns": 50, "max_cost_usd": null }
                }),
            ),
            durable_event(
                "tools_available",
                serde_json::json!({
                    "tools": [],
                    "tools_hash": "123456789"
                }),
            ),
            durable_assistant("reply", "a-1"),
        ]));

        assert_eq!(app.messages.len(), 3);
        assert!(app.logs.iter().all(|entry| {
            !entry.message.contains("session_configured")
                && !entry.message.contains("tools_available")
        }));
    }

    #[test]
    fn parent_session_progress_recorded_does_not_warn_or_add_chat() {
        let mut app = App::new();
        app.session_id = Some("s1".into());
        app.agent_id = Some("a1".into());
        app.messages.push(ChatEntry::User {
            text: "existing".into(),
            message_id: Some("existing-msg".into()),
        });

        app.handle_server_msg(session_events_msg(vec![
            durable_prompt("hello", "msg-1"),
            durable_progress_recorded(),
            durable_assistant("reply", "a-1"),
        ]));

        assert_eq!(app.messages.len(), 3);
        assert!(matches!(
            &app.messages[0],
            ChatEntry::User { text, message_id: Some(message_id) }
                if text == "existing" && message_id == "existing-msg"
        ));
        assert!(matches!(
            &app.messages[1],
            ChatEntry::User { text, message_id: Some(message_id) }
                if text == "hello" && message_id == "msg-1"
        ));
        assert!(matches!(
            &app.messages[2],
            ChatEntry::Assistant { content, message_id: Some(message_id), .. }
                if content == "reply" && message_id == "a-1"
        ));
        assert!(
            app.logs
                .iter()
                .all(|entry| !entry.message.contains("progress_recorded"))
        );
    }

    #[test]
    fn suppressed_frontier_turn_does_not_append_orphan_assistant_or_tools() {
        let mut app = App::new();
        app.messages.push(ChatEntry::User {
            text: "first".into(),
            message_id: Some("msg-1".into()),
        });
        app.messages.push(ChatEntry::Assistant {
            content: "reply one".into(),
            thinking: None,
            message_id: Some("a-1".into()),
        });
        app.undoable_turns = vec![UndoableTurn {
            turn_id: "turn-msg-1".into(),
            message_id: "msg-1".into(),
            text: "first".into(),
        }];
        app.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-2".into(),
                message_id: "msg-2".into(),
                status: UndoFrameStatus::Confirmed,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-2".into()),
        });

        app.handle_event_kind(
            &EventKind::PromptReceived {
                content: serde_json::json!("second"),
                message_id: Some("msg-2".into()),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::AssistantMessageStored {
                content: "orphaned reply".into(),
                thinking: None,
                message_id: Some("a-2".into()),
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::ToolCallStart {
                tool_call_id: Some("tool-2".into()),
                tool_name: "ls".into(),
                arguments: None,
            },
            false,
            None,
        );
        app.handle_event_kind(&EventKind::Cancelled, false, None);

        assert_eq!(app.messages.len(), 2);
        assert!(app.messages.iter().all(
            |entry| !matches!(entry, ChatEntry::Assistant { content, .. } if content == "orphaned reply")
        ));
        assert!(app.messages.iter().all(
            |entry| !matches!(entry, ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), .. } if tool_call_id == "tool-2")
        ));
        assert!(app.messages.iter().all(
            |entry| !matches!(entry, ChatEntry::Assistant { content, .. } if content.contains("[cancelled]"))
        ));
    }

    // ── ElicitationState::selected_display ────────────────────────────────────

    #[test]
    fn selected_display_single_select_returns_chosen_label() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "choice".into(),
            title: "Choice".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![
                    ElicitationOption {
                        value: serde_json::json!("a"),
                        label: "Alpha".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("b"),
                        label: "Beta".into(),
                        description: None,
                    },
                ],
            },
        }]);
        state
            .selected
            .insert("choice".into(), serde_json::json!("b"));
        assert_eq!(state.selected_display(), format!("{OUTCOME_BULLET}Beta"));
    }

    #[test]
    fn selected_display_multi_select_returns_bulleted_lines() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "tags".into(),
            title: "Tags".into(),
            description: None,
            required: false,
            kind: ElicitationFieldKind::MultiSelect {
                options: vec![
                    ElicitationOption {
                        value: serde_json::json!("x"),
                        label: "X".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("y"),
                        label: "Y".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("z"),
                        label: "Z".into(),
                        description: None,
                    },
                ],
            },
        }]);
        state
            .selected
            .insert("tags".into(), serde_json::json!(["x", "z"]));
        assert_eq!(
            state.selected_display(),
            format!("{OUTCOME_BULLET}X\n{OUTCOME_BULLET}Z")
        );
    }

    #[test]
    fn selected_display_text_input_returns_text() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "name".into(),
            title: "Name".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::TextInput,
        }]);
        state.text_input = "Alice".into();
        assert_eq!(state.selected_display(), "Alice");
    }

    #[test]
    fn selected_display_boolean_returns_yes_or_no() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "ok".into(),
            title: "OK".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::BooleanToggle,
        }]);
        state.selected.insert("ok".into(), serde_json::json!(true));
        assert_eq!(state.selected_display(), "Yes");
        state.selected.insert("ok".into(), serde_json::json!(false));
        assert_eq!(state.selected_display(), "No");
    }

    // ── ToolCallStart suppression for question ────────────────────────────────

    #[test]
    fn question_tool_call_start_does_not_push_chat_entry() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::ToolCallStart {
                tool_call_id: Some("call-1".into()),
                tool_name: "question".into(),
                arguments: None,
            },
            false,
            None,
        );
        assert!(
            !app.messages
                .iter()
                .any(|m| matches!(m, ChatEntry::ToolCall { .. }))
        );
    }

    #[test]
    fn other_tool_call_start_still_pushes_chat_entry() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::ToolCallStart {
                tool_call_id: Some("call-2".into()),
                tool_name: "read_tool".into(),
                arguments: None,
            },
            false,
            None,
        );
        assert!(
            app.messages
                .iter()
                .any(|m| matches!(m, ChatEntry::ToolCall { name, .. } if name == "read_tool"))
        );
    }

    // ── Elicitation: schema parsing ───────────────────────────────────────────

    #[test]
    fn parse_elicitation_schema_single_select() {
        let schema = serde_json::json!({
            "properties": {
                "choice": {
                    "title": "Pick one",
                    "description": "Your selection",
                    "oneOf": [
                        { "const": "a", "title": "Option A", "description": "First" },
                        { "const": "b", "title": "Option B" }
                    ]
                }
            },
            "required": ["choice"]
        });
        let fields = ElicitationState::parse_schema(&schema);
        assert_eq!(fields.len(), 1);
        let f = &fields[0];
        assert_eq!(f.name, "choice");
        assert_eq!(f.title, "Pick one");
        assert_eq!(f.description.as_deref(), Some("Your selection"));
        assert!(f.required);
        let ElicitationFieldKind::SingleSelect { options } = &f.kind else {
            panic!("expected SingleSelect");
        };
        assert_eq!(options.len(), 2);
        assert_eq!(options[0].label, "Option A");
        assert_eq!(options[0].description.as_deref(), Some("First"));
        assert_eq!(options[1].label, "Option B");
        assert!(options[1].description.is_none());
    }

    #[test]
    fn parse_elicitation_schema_multi_select() {
        let schema = serde_json::json!({
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            { "const": "x", "title": "X" },
                            { "const": "y", "title": "Y" }
                        ]
                    }
                }
            },
            "required": []
        });
        let fields = ElicitationState::parse_schema(&schema);
        assert_eq!(fields.len(), 1);
        let ElicitationFieldKind::MultiSelect { options } = &fields[0].kind else {
            panic!("expected MultiSelect");
        };
        assert_eq!(options.len(), 2);
        assert!(!fields[0].required);
    }

    #[test]
    fn parse_elicitation_schema_text_and_boolean() {
        let schema = serde_json::json!({
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" },
                "confirm": { "type": "boolean" }
            },
            "required": ["name"]
        });
        let fields = ElicitationState::parse_schema(&schema);
        assert_eq!(fields.len(), 3);
        let kinds: Vec<_> = fields.iter().map(|f| (&f.name, &f.kind)).collect();
        assert!(matches!(
            kinds.iter().find(|(n, _)| *n == "name").unwrap().1,
            ElicitationFieldKind::TextInput
        ));
        assert!(matches!(
            kinds.iter().find(|(n, _)| *n == "count").unwrap().1,
            ElicitationFieldKind::NumberInput { integer: true }
        ));
        assert!(matches!(
            kinds.iter().find(|(n, _)| *n == "confirm").unwrap().1,
            ElicitationFieldKind::BooleanToggle
        ));
    }

    #[test]
    fn parse_elicitation_schema_empty_returns_empty() {
        let fields = ElicitationState::parse_schema(&serde_json::json!({}));
        assert!(fields.is_empty());
    }

    // ── Elicitation: state navigation ─────────────────────────────────────────

    #[test]
    fn elicitation_move_cursor_wraps_within_options() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "q".into(),
            title: "Q".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![
                    ElicitationOption {
                        value: serde_json::json!("a"),
                        label: "A".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("b"),
                        label: "B".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("c"),
                        label: "C".into(),
                        description: None,
                    },
                ],
            },
        }]);
        assert_eq!(state.option_cursor, 0);
        state.move_cursor(1);
        assert_eq!(state.option_cursor, 1);
        state.move_cursor(1);
        assert_eq!(state.option_cursor, 2);
        state.move_cursor(1); // clamps at max
        assert_eq!(state.option_cursor, 2);
        state.move_cursor(-1);
        assert_eq!(state.option_cursor, 1);
        state.move_cursor(-10);
        assert_eq!(state.option_cursor, 0);
    }

    #[test]
    fn elicitation_build_accept_content_single_select() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "choice".into(),
            title: "Choice".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![
                    ElicitationOption {
                        value: serde_json::json!("yes"),
                        label: "Yes".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("no"),
                        label: "No".into(),
                        description: None,
                    },
                ],
            },
        }]);
        state.option_cursor = 0;
        state.select_current_option(); // select "yes"
        let content = state.build_accept_content();
        assert_eq!(content, serde_json::json!({ "choice": "yes" }));
    }

    #[test]
    fn elicitation_build_accept_content_text_input() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "name".into(),
            title: "Name".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::TextInput,
        }]);
        state.text_input = "Alice".into();
        let content = state.build_accept_content();
        assert_eq!(content, serde_json::json!({ "name": "Alice" }));
    }

    #[test]
    fn elicitation_is_valid_requires_required_fields() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "must".into(),
            title: "Must".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::TextInput,
        }]);
        assert!(!state.is_valid());
        state.text_input = "value".into();
        assert!(state.is_valid());
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

    #[test]
    fn toolcallend_question_replay_backfills_elicitation_cards() {
        let mut app = App::new();
        // Simulate replay of ElicitationRequested (pushes "responded" card)
        app.handle_event_kind(
            &EventKind::ElicitationRequested {
                elicitation_id: "e1".into(),
                session_id: "sess-1".into(),
                message: "Which?".into(),
                requested_schema: serde_json::json!({
                    "properties": { "choice": { "oneOf": [{ "const": "a", "title": "Alpha" }] } },
                    "required": ["choice"]
                }),
                source: "builtin:question".into(),
            },
            true,
            None,
        );
        // Simulate replay of ToolCallEnd for question
        app.handle_event_kind(
            &EventKind::ToolCallEnd {
                tool_call_id: Some("call-1".into()),
                tool_name: "question".into(),
                is_error: Some(false),
                result: Some(r#"{"answers":[{"question":"Which?","answers":["Alpha"]}]}"#.into()),
            },
            true,
            None,
        );
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}Alpha")
        )));
    }

    #[test]
    fn elicitation_requested_during_replay_does_not_open_popup() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::ElicitationRequested {
                elicitation_id: "elic-replay".into(),
                session_id: "sess-1".into(),
                message: "Which option?".into(),
                requested_schema: serde_json::json!({
                    "properties": {
                        "choice": { "oneOf": [{ "const": "a", "title": "A" }] }
                    },
                    "required": ["choice"]
                }),
                source: "builtin:question".into(),
            },
            true, // is_replay
            None,
        );

        // No popup should be opened
        assert!(app.elicitation.is_none());
        // Chat card should be present but already marked as resolved
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { elicitation_id, outcome: Some(_), .. }
            if elicitation_id == "elic-replay"
        )));
    }

    #[test]
    fn replayed_elicitation_request_is_idempotent_across_history_replays() {
        let mut app = App::new();
        let requested_schema = serde_json::json!({
            "properties": {
                "choice": {
                    "oneOf": [
                        { "const": "a", "title": "Alpha" },
                        { "const": "b", "title": "Beta" }
                    ]
                }
            },
            "required": ["choice"]
        });
        let replayed_request = EventKind::ElicitationRequested {
            elicitation_id: "elic-dup".into(),
            session_id: "sess-1".into(),
            message: "Which option?".into(),
            requested_schema,
            source: "builtin:question".into(),
        };

        // Simulate session_loaded replay, then current-session history replay of the same request.
        app.handle_event_kind(&replayed_request, true, None);
        app.handle_event_kind(&replayed_request, true, None);
        app.handle_event_kind(
            &EventKind::ToolCallEnd {
                tool_call_id: Some("call-1".into()),
                tool_name: "question".into(),
                is_error: Some(false),
                result: Some(
                    r#"{"answers":[{"question":"Which option?","answers":["Beta"]}]}"#.into(),
                ),
            },
            true,
            None,
        );

        let elicitation_cards: Vec<_> = app
            .messages
            .iter()
            .filter(|entry| {
                matches!(
                    entry,
                    ChatEntry::Elicitation { elicitation_id, .. } if elicitation_id == "elic-dup"
                )
            })
            .collect();
        assert_eq!(elicitation_cards.len(), 1);
        assert!(matches!(elicitation_cards[0],
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}Beta")
        ));
    }

    #[test]
    fn elicitation_requested_event_creates_state_and_chat_card() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::ElicitationRequested {
                elicitation_id: "elic-1".into(),
                session_id: "sess-1".into(),
                message: "Which option?".into(),
                requested_schema: serde_json::json!({
                    "properties": {
                        "choice": {
                            "oneOf": [
                                { "const": "a", "title": "Alpha" },
                                { "const": "b", "title": "Beta" }
                            ]
                        }
                    },
                    "required": ["choice"]
                }),
                source: "builtin:question".into(),
            },
            false,
            None,
        );

        // State should be populated
        let state = app.elicitation.as_ref().expect("elicitation state");
        assert_eq!(state.elicitation_id, "elic-1");
        assert_eq!(state.message, "Which option?");
        assert_eq!(state.fields.len(), 1);

        // A chat card should have been appended
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { elicitation_id, outcome: None, .. }
            if elicitation_id == "elic-1"
        )));
    }

    #[test]
    fn elicitation_requested_with_empty_schema_does_not_open_popup() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::ElicitationRequested {
                elicitation_id: "elic-empty".into(),
                session_id: "sess-1".into(),
                message: "Unsupported question".into(),
                requested_schema: serde_json::json!({}),
                source: "builtin:question".into(),
            },
            false,
            None,
        );

        assert!(app.elicitation.is_none());
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { elicitation_id, outcome: Some(outcome), .. }
            if elicitation_id == "elic-empty" && outcome == "unsupported schema - cannot answer in TUI"
        )));
        assert_eq!(app.status, "question skipped - unsupported schema");
    }

    #[test]
    fn replay_audit_does_not_clear_redo_stack() {
        let mut app = App::new();
        app.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-3".into(),
                message_id: "msg-3".into(),
                status: UndoFrameStatus::Confirmed,
                reverted_files: vec!["src/lib.rs".into()],
            }],
            frontier_message_id: Some("msg-3".into()),
        });

        let audit = serde_json::json!({
            "events": [
                {
                    "kind": {
                        "type": "prompt_received",
                        "data": {
                            "content": [{ "type": "text", "text": "one" }],
                            "message_id": "msg-1"
                        }
                    }
                },
                {
                    "kind": {
                        "type": "prompt_received",
                        "data": {
                            "content": [{ "type": "text", "text": "two" }],
                            "message_id": "msg-2"
                        }
                    }
                },
                {
                    "kind": {
                        "type": "prompt_received",
                        "data": {
                            "content": [{ "type": "text", "text": "three" }],
                            "message_id": "msg-3"
                        }
                    }
                }
            ]
        });

        app.replay_audit(&audit, None);

        assert!(app.can_redo());
        let state = app.undo_state.as_ref().expect("undo state");
        assert_eq!(state.frontier_message_id.as_deref(), Some("msg-3"));
        assert_eq!(state.stack.len(), 1);
        assert_eq!(state.stack[0].reverted_files, vec!["src/lib.rs"]);
    }
}

// ── Thinking content tests ────────────────────────────────────────────────────

#[cfg(test)]
mod thinking_content_tests {
    use super::*;

    #[test]
    fn thinking_delta_accumulates_in_streaming_thinking() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false, None);

        app.handle_event_kind(
            &EventKind::AssistantThinkingDelta {
                content: "Let me ".into(),
                message_id: None,
            },
            false,
            None,
        );
        assert_eq!(app.streaming_thinking, "Let me ");

        app.handle_event_kind(
            &EventKind::AssistantThinkingDelta {
                content: "think about this.".into(),
                message_id: None,
            },
            false,
            None,
        );
        assert_eq!(app.streaming_thinking, "Let me think about this.");
    }

    #[test]
    fn turn_started_clears_streaming_thinking() {
        let mut app = App::new();
        app.streaming_thinking = "old thinking".into();

        app.handle_event_kind(&EventKind::TurnStarted, false, None);

        assert!(app.streaming_thinking.is_empty());
    }

    #[test]
    fn assistant_message_stored_captures_thinking_field() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false, None);

        app.handle_event_kind(
            &EventKind::AssistantMessageStored {
                content: "The answer is 42.".into(),
                thinking: Some("I need to compute the answer.".into()),
                message_id: None,
            },
            false,
            None,
        );

        assert_eq!(app.messages.len(), 1);
        match &app.messages[0] {
            ChatEntry::Assistant {
                content, thinking, ..
            } => {
                assert_eq!(content, "The answer is 42.");
                assert_eq!(thinking.as_deref(), Some("I need to compute the answer."));
            }
            other => panic!("expected Assistant, got {:?}", other),
        }
    }

    #[test]
    fn assistant_message_stored_without_thinking_sets_none() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false, None);

        app.handle_event_kind(
            &EventKind::AssistantMessageStored {
                content: "Hello!".into(),
                thinking: None,
                message_id: None,
            },
            false,
            None,
        );

        assert_eq!(app.messages.len(), 1);
        match &app.messages[0] {
            ChatEntry::Assistant {
                content, thinking, ..
            } => {
                assert_eq!(content, "Hello!");
                assert!(thinking.is_none());
            }
            other => panic!("expected Assistant, got {:?}", other),
        }
    }

    #[test]
    fn streaming_thinking_falls_back_when_stored_thinking_is_none() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false, None);

        // Simulate thinking deltas arriving before the stored message
        app.handle_event_kind(
            &EventKind::AssistantThinkingDelta {
                content: "Streamed thinking.".into(),
                message_id: None,
            },
            false,
            None,
        );

        // AssistantMessageStored arrives without thinking field
        app.handle_event_kind(
            &EventKind::AssistantMessageStored {
                content: "Final answer.".into(),
                thinking: None,
                message_id: None,
            },
            false,
            None,
        );

        match &app.messages[0] {
            ChatEntry::Assistant {
                content, thinking, ..
            } => {
                assert_eq!(content, "Final answer.");
                assert_eq!(thinking.as_deref(), Some("Streamed thinking."));
            }
            other => panic!("expected Assistant, got {:?}", other),
        }
        // streaming_thinking should be cleared after capture
        assert!(app.streaming_thinking.is_empty());
    }

    #[test]
    fn cancelled_with_thinking_preserves_thinking_in_entry() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false, None);

        app.handle_event_kind(
            &EventKind::AssistantThinkingDelta {
                content: "Deep thought.".into(),
                message_id: None,
            },
            false,
            None,
        );
        app.handle_event_kind(
            &EventKind::AssistantContentDelta {
                content: "Partial answer".into(),
                message_id: None,
            },
            false,
            None,
        );
        app.handle_event_kind(&EventKind::Cancelled, false, None);

        assert_eq!(app.messages.len(), 1);
        match &app.messages[0] {
            ChatEntry::Assistant {
                content, thinking, ..
            } => {
                assert!(content.contains("Partial answer"));
                assert!(content.contains("[cancelled]"));
                assert_eq!(thinking.as_deref(), Some("Deep thought."));
            }
            other => panic!("expected Assistant, got {:?}", other),
        }
    }

    #[test]
    fn thinking_delta_keeps_activity_as_thinking() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false, None);
        assert_eq!(app.activity, ActivityState::Thinking);

        app.handle_event_kind(
            &EventKind::AssistantThinkingDelta {
                content: "hmm".into(),
                message_id: None,
            },
            false,
            None,
        );
        // Should still be Thinking (not Streaming) during thinking phase
        assert_eq!(app.activity, ActivityState::Thinking);
    }
}

// ── Start-page session grouping tests ─────────────────────────────────────────

#[cfg(test)]
mod start_page_tests {
    use super::*;

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
                session_idx: 0
            }
        ));
        assert!(matches!(
            &items[2],
            StartPageItem::Session {
                group_idx: 0,
                session_idx: 1
            }
        ));
        assert!(matches!(
            &items[4],
            StartPageItem::Session {
                group_idx: 1,
                session_idx: 0
            }
        ));
    }

    // ── session_list message preserves group structure ────────────────────────

    #[test]
    fn session_list_message_populates_session_groups() {
        let mut app = App::new();
        app.handle_server_msg(RawServerMsg {
            msg_type: "session_list".into(),
            data: Some(serde_json::json!({
                "groups": [
                    {
                        "cwd": "/home/user/proj",
                        "sessions": [
                            { "session_id": "s1", "title": "T1", "updated_at": "2024-01-01T00:00:00Z" }
                        ]
                    }
                ]
            })),
        });

        assert_eq!(app.session_groups.len(), 1);
        assert_eq!(
            app.session_groups[0].cwd.as_deref(),
            Some("/home/user/proj")
        );
        assert_eq!(app.session_groups[0].sessions.len(), 1);
        assert_eq!(app.session_groups[0].sessions[0].session_id, "s1");
    }

    #[test]
    fn backend_shaped_session_list_deserializes_with_new_fields() {
        let list: SessionListData = serde_json::from_value(serde_json::json!({
            "groups": [
                {
                    "cwd": "/workspace/project",
                    "latest_activity": "2024-02-01T00:00:00Z",
                    "total_count": 1,
                    "next_cursor": "group-next",
                    "sessions": [
                        {
                            "session_id": "s1",
                            "name": "Session One",
                            "cwd": "/workspace/project",
                            "title": "T1",
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-02-01T00:00:00Z",
                            "parent_session_id": null,
                            "fork_origin": null,
                            "session_kind": "interactive",
                            "has_children": true,
                            "node": "local",
                            "node_id": "node-1",
                            "attached": true,
                            "runtime_state": "running"
                        }
                    ]
                }
            ],
            "next_cursor": "list-next",
            "total_count": 1
        }))
        .expect("backend session_list shape should parse");

        assert_eq!(list.next_cursor.as_deref(), Some("list-next"));
        assert_eq!(list.total_count, Some(1));
        assert_eq!(list.groups[0].total_count, Some(1));
        assert_eq!(list.groups[0].next_cursor.as_deref(), Some("group-next"));
        let session = &list.groups[0].sessions[0];
        assert_eq!(session.name.as_deref(), Some("Session One"));
        assert_eq!(session.session_kind.as_deref(), Some("interactive"));
        assert_eq!(session.node.as_deref(), Some("local"));
        assert_eq!(session.node_id.as_deref(), Some("node-1"));
        assert_eq!(session.attached, Some(true));
        assert_eq!(session.runtime_state.as_deref(), Some("running"));
    }

    #[test]
    fn session_list_hides_only_delegation_children_and_preserves_delegate_entries() {
        let mut app = App::new();
        app.delegate_entries.push(DelegateEntry {
            delegation_id: "del-existing".into(),
            child_session_id: Some("delegate-child".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("agent".into()),
            objective: "Existing delegate".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::default(),
        });

        app.handle_server_msg(RawServerMsg {
            msg_type: "session_list".into(),
            data: Some(serde_json::json!({
                "groups": [
                    {
                        "cwd": "/workspace/project",
                        "sessions": [
                            { "session_id": "root", "title": "Root", "updated_at": "2024-01-04T00:00:00Z" },
                            { "session_id": "delegate-child", "title": "Delegation", "updated_at": "2024-01-03T00:00:00Z", "parent_session_id": "root", "fork_origin": "delegation" },
                            { "session_id": "user-child", "title": "User Fork", "updated_at": "2024-01-02T00:00:00Z", "parent_session_id": "root", "fork_origin": "user" },
                            { "session_id": "unknown-child", "title": "Unknown Fork", "updated_at": "2024-01-01T00:00:00Z", "parent_session_id": "root" }
                        ]
                    },
                    {
                        "cwd": "/workspace/project/empty-after-filter",
                        "sessions": [
                            { "session_id": "delegate-only", "title": "Delegate Only", "updated_at": "2024-01-05T00:00:00Z", "parent_session_id": "root", "fork_origin": "delegation" }
                        ]
                    }
                ],
                "total_count": 5
            })),
        });

        let visible_ids: Vec<&str> = app.session_groups[0]
            .sessions
            .iter()
            .map(|session| session.session_id.as_str())
            .collect();
        assert_eq!(app.session_groups.len(), 1);
        assert_eq!(visible_ids, vec!["root", "user-child", "unknown-child"]);
        assert!(!visible_ids.contains(&"delegate-child"));
        assert_eq!(app.status, "3 session(s)");
        assert_eq!(app.delegate_entries.len(), 1);
        assert_eq!(app.delegate_entries[0].delegation_id, "del-existing");

        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "session"
                && entry.message
                    == "session list: kept 2 non-delegation child session(s) (fork_origin=unknown: 1, user: 1)"
        }));
        assert!(app.logs.iter().any(|entry| {
            entry.level == LogLevel::Debug
                && entry.target == "session"
                && entry.message
                    == "session list: hid 2 delegation child session(s); visible=3, backend_total=5"
        }));
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

        let items = app.visible_start_items();
        assert!(matches!(
            &items[0],
            StartPageItem::GroupHeader {
                session_count: 3,
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
            Some(PopupItem::LoadMore { group_idx: 0 })
        ));
    }

    // ── no MAX_VISIBLE_GROUPS cap ─────────────────────────────────────────────

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
                session_idx: 0
            }
        ));
        assert!(matches!(
            &items[2],
            PopupItem::Session {
                group_idx: 0,
                session_idx: 1
            }
        ));
        assert!(matches!(
            &items[4],
            PopupItem::Session {
                group_idx: 1,
                session_idx: 0
            }
        ));
    }

    // ── group header carries correct session_count ────────────────────────────

    #[test]
    fn popup_group_header_session_count_reflects_total() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3"])];
        let items = app.visible_popup_items();
        assert!(matches!(
            &items[0],
            PopupItem::GroupHeader {
                session_count: 3,
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
    use crate::protocol::{AgentInfo, ModelEntry};

    fn make_agent(id: &str, name: &str) -> AgentInfo {
        AgentInfo {
            id: id.into(),
            name: name.into(),
        }
    }

    fn make_model(provider: &str, model: &str) -> ModelEntry {
        ModelEntry {
            id: format!("{provider}/{model}"),
            label: format!("{provider}/{model}"),
            provider: provider.into(),
            model: model.into(),
            node_id: None,
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
    fn tab_label_plan_at_zero() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_label(0), "plan");
    }

    #[test]
    fn tab_label_build_at_one() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_label(1), "build");
    }

    #[test]
    fn tab_label_review_at_two() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_label(2), "review");
    }

    #[test]
    fn tab_label_agent_at_three() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        assert_eq!(app.model_popup_tab_label(3), "Coder");
    }

    #[test]
    fn tab_agent_id_none_at_zero() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_agent_id(0), None);
    }

    #[test]
    fn tab_agent_id_none_at_one() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_agent_id(1), None);
    }

    #[test]
    fn tab_agent_id_none_at_two() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_agent_id(2), None);
    }

    #[test]
    fn tab_agent_id_some_at_three() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        assert_eq!(app.model_popup_tab_agent_id(3), Some("coder"));
    }

    #[test]
    fn tab_mode_review_at_two() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_mode(2), Some("review"));
    }

    #[test]
    fn tab_count_no_agents() {
        let app = App::new();
        assert_eq!(app.model_popup_tab_count(), 3);
    }

    #[test]
    fn tab_count_single_agent() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main")];
        assert_eq!(app.model_popup_tab_count(), 3);
    }

    #[test]
    fn tab_count_multi_agent() {
        let mut app = App::new();
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        assert_eq!(app.model_popup_tab_count(), 4);
    }

    #[test]
    fn delegate_pref_round_trip() {
        let mut app = App::new();
        app.set_delegate_model_preference("coder", "anthropic", "claude-sonnet");
        assert_eq!(
            app.get_delegate_model_preference("coder"),
            Some(("anthropic", "claude-sonnet"))
        );
    }

    #[test]
    fn delegate_pref_missing_returns_none() {
        let app = App::new();
        assert_eq!(app.get_delegate_model_preference("coder"), None);
    }

    #[test]
    fn delegate_model_cursor_with_preference() {
        let mut app = App::new();
        app.models = vec![
            make_model("openai", "gpt-4o"),
            make_model("anthropic", "claude-sonnet"),
        ];
        app.set_delegate_model_preference("coder", "anthropic", "claude-sonnet");
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
