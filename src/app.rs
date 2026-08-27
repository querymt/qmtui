use crate::auth_state::AuthState;
use crate::chat_state::ChatState;
use crate::command::Command;
use crate::composer_state::ComposerState;
use crate::connection_state::{ConnState, ConnectionState};
use crate::delegates_state::DelegatesState;
use crate::diagnostics::{AppLogEntry, DiagnosticsState, LogLevel};
use crate::mesh_state::MeshState;
use crate::models_state::ModelsState;
use crate::navigation_state::{NavigationState, Popup};
use crate::profiles_state::ProfilesState;
use crate::render_state::RenderState;
use crate::session_state::SessionsState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionEvent {
    Connecting { attempt: u32, delay_ms: u64 },
    Connected,
    Disconnected { reason: String },
}

pub struct App {
    pub(crate) navigation: NavigationState,

    // sessions
    pub(crate) sessions: SessionsState,
    pub(crate) delegates: DelegatesState,

    // chat
    pub(crate) chat: ChatState,
    pub(crate) composer: ComposerState,

    // profile info
    pub(crate) profiles: ProfilesState,

    // model, reasoning effort, and profile-agent state
    pub(crate) models: ModelsState,

    // in-memory logs popup and status line
    pub(crate) diagnostics: DiagnosticsState,

    // mesh / remote (from ACP extensions)
    pub(crate) mesh: MeshState,

    // connection and server lifecycle
    pub(crate) connection: ConnectionState,

    // temporary render-local composition
    pub(crate) render: RenderState,

    // auth popup state
    pub(crate) auth: AuthState,

    pub should_quit: bool,
}

impl App {
    pub fn begin_session_discovery(&mut self) -> Option<Command> {
        self.sessions
            .prepare_session_discovery()
            .then(Command::list_sessions_browse)
    }

    pub fn session_group_page_request(&mut self, group_idx: usize) -> Option<Command> {
        let (cwd, cursor) = self.sessions.prepare_session_group_page(group_idx)?;
        Some(Command::list_sessions_group(cwd, cursor))
    }

    pub fn session_child_page_request(
        &mut self,
        group_idx: usize,
        parent_path: &[usize],
    ) -> Option<Command> {
        let (parent_session_id, cursor) = self
            .sessions
            .prepare_session_child_page(group_idx, parent_path)?;
        Some(Command::list_session_children(
            parent_session_id,
            cursor,
            crate::session_state::SESSION_CHILD_PAGE_LIMIT,
        ))
    }

    pub fn new() -> Self {
        Self {
            navigation: NavigationState::new(),
            sessions: SessionsState::new(),
            delegates: DelegatesState::new(),
            chat: ChatState::new(),
            composer: ComposerState::new(),
            profiles: ProfilesState::new(),
            models: ModelsState::new(),
            diagnostics: DiagnosticsState::new(),
            mesh: MeshState::new(),
            connection: ConnectionState::new(),
            render: RenderState::new(),
            auth: AuthState::new(),
            should_quit: false,
        }
    }

    pub fn take_input(&mut self) -> String {
        self.composer.input_cursor = 0;
        self.composer.input_scroll = 0;
        self.composer.input_preferred_col = None;
        self.chat.scroll_offset = 0;
        self.composer.mention_state = None;
        self.composer.slash_state = None;
        std::mem::take(&mut self.composer.input)
    }

    /// Cycle through `[auto, low, medium, high, max]` (wraps around).
    /// Updates the nested reasoning state optimistically and
    /// returns the [`Command`] to forward to the server.
    ///
    /// Returns `None` if the current value is not a recognized level; in that
    /// case the state is left unchanged and no message is emitted (the caller
    /// should surface a warning to the user instead of silently coercing the
    /// unknown value to `low`).
    pub fn cycle_reasoning_effort(&mut self) -> Option<Command> {
        let reasoning_effort = self.models.cycle_reasoning_effort()?;
        Some(Command::SetReasoningEffort { reasoning_effort })
    }

    /// Set the reasoning effort to a specific level.
    /// `None` or `Some("auto")` both map to the "auto" (no override) state.
    /// Updates nested reasoning state and returns the [`Command`] to forward to the server.
    /// Returns `None` if the level is invalid (state is unchanged).
    pub fn set_reasoning_effort(&mut self, level: Option<&str>) -> Option<Command> {
        let reasoning_effort = self.models.set_reasoning_effort(level)?;
        Some(Command::SetReasoningEffort { reasoning_effort })
    }

    /// Route to a freshly reset auth popup while preserving provider data.
    pub fn open_auth_popup(&mut self) {
        self.navigation.popup = Popup::ProviderAuth;
        self.auth.reset_for_open();
    }

    pub fn current_session_profile_id(&self) -> Option<&str> {
        self.sessions
            .session_id
            .as_deref()
            .and_then(|session_id| self.profiles.session_profile_id(session_id))
    }

    pub fn current_profile_label(&self) -> String {
        self.current_session_profile_id()
            .map(|profile_id| self.profiles.profile_display_name(profile_id))
            .unwrap_or_else(|| {
                if self.sessions.current_session_is_remote() {
                    "remote".to_string()
                } else {
                    self.profiles.active_profile_label()
                }
            })
    }

    pub fn open_profile_popup(&mut self) {
        self.navigation.popup = Popup::ProfileSelect;
        self.profiles.reset_for_open();
    }

    // phase 11 compatibility forward
    pub fn push_log(&mut self, level: LogLevel, target: &'static str, message: impl Into<String>) {
        self.diagnostics.push_log(level, target, message);
    }

    // phase 11 compatibility forward
    pub fn set_status(
        &mut self,
        level: LogLevel,
        target: &'static str,
        message: impl Into<String>,
    ) {
        self.diagnostics.set_status(level, target, message);
    }

    // phase 11 compatibility forward
    pub fn filtered_logs(&self) -> Vec<&AppLogEntry> {
        self.diagnostics.filtered_logs()
    }

    // phase 11 compatibility forward
    pub fn cycle_log_level_filter(&mut self) {
        self.diagnostics.cycle_log_level_filter();
    }

    pub fn arm_cancel_confirm(&mut self) {
        self.chat.arm_cancel_confirm();
        self.set_status(LogLevel::Warn, "input", "press Esc again to stop");
    }

    pub fn open_fork_turn_popup(&mut self) {
        self.navigation.popup = Popup::ForkTurnSelect;
        self.chat.reset_fork_selector();
    }

    pub fn push_pending_prompt(&mut self, text: String) -> String {
        let local_id = self.chat.push_pending_prompt(text);
        self.render.invalidate_card_cache();
        local_id
    }

    pub fn refresh_transient_status(&mut self) {
        if self.chat.pending_cancel_confirm_until.is_some() {
            return;
        }
        if self.chat.elicitation.is_some() {
            self.set_status(
                LogLevel::Debug,
                "elicitation",
                "question - answer in the panel above input",
            );
        } else if let Some(activity_status) = self.chat.activity_status_text() {
            self.set_status(LogLevel::Debug, "activity", activity_status);
        } else if self.connection.conn == ConnState::Connected {
            self.set_status(LogLevel::Debug, "activity", "ready");
        }
    }

    pub fn clear_expired_cancel_confirm(&mut self) {
        if self.chat.clear_expired_cancel_confirm() {
            self.refresh_transient_status();
        }
    }

    pub fn handle_connection_event(&mut self, event: ConnectionEvent) {
        self.chat.clear_cancel_confirm();
        match event {
            ConnectionEvent::Connecting { attempt, delay_ms } => {
                self.connection.apply_connecting(attempt, delay_ms);
                let secs = delay_ms as f64 / 1000.0;
                self.set_status(
                    LogLevel::Warn,
                    "connection",
                    format!("waiting for server - retry {attempt} in {secs:.1}s"),
                );
            }
            ConnectionEvent::Connected => {
                self.connection.apply_connected();
                self.set_status(
                    LogLevel::Info,
                    "connection",
                    if self.sessions.session_id.is_some() {
                        "reconnected".to_string()
                    } else {
                        "connected".to_string()
                    },
                );
            }
            ConnectionEvent::Disconnected { reason } => {
                self.connection.apply_disconnected();
                self.sessions.session_discovery_in_progress = false;
                self.sessions.pending_session_group_loads.clear();
                self.set_status(
                    LogLevel::Warn,
                    "connection",
                    format!("connection lost - {reason}"),
                );
            }
        }
    }

    // ── delegate model preference coordination ───────────────────────────────

    pub fn delegate_preference_profile_id(&self) -> Option<&str> {
        self.current_session_profile_id()
            .or(self.profiles.active_profile_id.as_deref())
    }

    pub fn desired_agents_profile_id(&self) -> Option<&str> {
        self.delegate_preference_profile_id()
    }

    pub fn delegate_model_commands_for_session(
        &self,
        session_id: &str,
        profile_id: &str,
    ) -> Vec<Command> {
        self.models
            .delegate_model_commands_for_session(session_id, profile_id)
    }

    /// Cursor position for a delegate agent's preferred model in the popup list.
    pub fn delegate_model_cursor(&self, agent_id: &str) -> usize {
        self.models
            .delegate_model_cursor(self.delegate_preference_profile_id(), agent_id)
    }
}

// ── app coordinator and session tests ─────────────────────────────────────────

#[cfg(test)]
mod reasoning_effort_tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::domain::activity::SessionActivity;
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::session_state::PathCompletionState;

    // ── cycle_reasoning_effort ────────────────────────────────────────────────

    #[test]
    fn cycle_from_auto_to_low() {
        let mut app = App::new();
        assert_eq!(app.models.reasoning_effort, None);
        app.cycle_reasoning_effort();
        assert_eq!(app.models.reasoning_effort, Some("low".into()));
    }

    #[test]
    fn cycle_from_low_to_medium() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("low".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.models.reasoning_effort, Some("medium".into()));
    }

    #[test]
    fn cycle_from_medium_to_high() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("medium".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.models.reasoning_effort, Some("high".into()));
    }

    #[test]
    fn cycle_from_high_to_max() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("high".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.models.reasoning_effort, Some("max".into()));
    }

    #[test]
    fn cycle_from_max_wraps_to_auto() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("max".into());
        app.cycle_reasoning_effort();
        assert_eq!(app.models.reasoning_effort, None);
    }

    #[test]
    fn cycle_full_round_trip() {
        let mut app = App::new();
        // auto → low → medium → high → max → auto
        for _ in 0..5 {
            app.cycle_reasoning_effort();
        }
        assert_eq!(app.models.reasoning_effort, None);
    }

    #[test]
    fn cycle_reasoning_effort_unknown_value_noop() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("invalid_level".into());
        let result = app.cycle_reasoning_effort();
        // unknown current value → no-op: returns None and leaves state unchanged
        assert!(
            result.is_none(),
            "cycling an unknown value should return None"
        );
        assert_eq!(
            app.models.reasoning_effort,
            Some("invalid_level".into()),
            "state must not change when cycling an unknown value"
        );
    }

    #[test]
    fn cycle_returns_correct_command() {
        let mut app = App::new(); // starts at auto
        let msg = app.cycle_reasoning_effort().expect("auto is a valid level");
        // auto → low: should send "low"
        match msg {
            Command::SetReasoningEffort { reasoning_effort } => {
                assert_eq!(reasoning_effort, "low");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn cycle_to_auto_sends_auto_string() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("max".into());
        let msg = app.cycle_reasoning_effort().expect("max is a valid level");
        // max → auto: server expects "auto" string (not null)
        match msg {
            Command::SetReasoningEffort { reasoning_effort } => {
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
        assert_eq!(app.models.reasoning_effort, Some("high".into()));
        match msg {
            Some(Command::SetReasoningEffort { reasoning_effort }) => {
                assert_eq!(reasoning_effort, "high");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn set_reasoning_effort_auto_clears_to_none() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("max".into());
        let msg = app.set_reasoning_effort(Some("auto"));
        assert_eq!(app.models.reasoning_effort, None);
        match msg {
            Some(Command::SetReasoningEffort { reasoning_effort }) => {
                assert_eq!(reasoning_effort, "auto");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn set_reasoning_effort_none_clears_to_auto() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("low".into());
        let msg = app.set_reasoning_effort(None);
        assert_eq!(app.models.reasoning_effort, None);
        match msg {
            Some(Command::SetReasoningEffort { reasoning_effort }) => {
                assert_eq!(reasoning_effort, "auto");
            }
            other => panic!("expected SetReasoningEffort, got {other:?}"),
        }
    }

    #[test]
    fn set_reasoning_effort_invalid_value_rejected() {
        let mut app = App::new();
        app.models.reasoning_effort = Some("medium".into());
        let msg = app.set_reasoning_effort(Some("ultra"));
        assert_eq!(app.models.reasoning_effort, Some("medium".into()));
        assert!(msg.is_none());
    }

    // ── state message populates reasoning_effort ──────────────────────────────

    // ── reasoning_effort push notification ────────────────────────────────────

    #[test]
    fn active_session_count_requires_multiple_recent_sessions() {
        let mut app = App::new();
        app.sessions.note_session_activity("session-a");
        assert_eq!(app.sessions.active_session_count(), 1);

        app.sessions.note_session_activity("session-b");
        assert_eq!(app.sessions.active_session_count(), 2);
    }

    #[test]
    fn other_active_session_count_excludes_current_session() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-a".into());
        app.sessions.note_session_activity("session-a");
        app.sessions.note_session_activity("session-b");
        app.sessions.note_session_activity("session-c");

        assert_eq!(app.sessions.other_active_session_count(), 2);
    }

    #[test]
    fn other_active_session_count_shows_other_session_when_current_is_idle() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-a".into());
        app.sessions.note_session_activity("session-b");

        assert_eq!(app.sessions.other_active_session_count(), 1);
    }

    #[test]
    fn active_session_count_excludes_stale_sessions() {
        let mut app = App::new();
        app.sessions.note_session_activity("session-a");
        app.sessions.session_activity.insert(
            "session-b".into(),
            SessionActivity {
                last_event_at: Instant::now() - Duration::from_secs(6),
            },
        );

        assert_eq!(app.sessions.active_session_count(), 1);
        assert_eq!(app.sessions.other_active_session_count(), 1);
    }

    #[test]
    fn resolve_new_session_default_cwd_prefers_active_session_cwd_then_group_then_launch() {
        let mut app = App::new();
        app.connection.launch_cwd = Some("/launch".into());
        app.sessions.session_id = Some("session-a".into());
        app.sessions.session_groups = vec![SessionGroup {
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

        app.sessions.session_groups[0].sessions[0].cwd = None;
        assert_eq!(
            app.resolve_new_session_default_cwd().as_deref(),
            Some("/group")
        );

        app.sessions.session_groups.clear();
        assert_eq!(
            app.resolve_new_session_default_cwd().as_deref(),
            Some("/launch")
        );
    }

    #[test]
    fn open_new_session_popup_prefills_path_and_cursor() {
        let mut app = App::new();
        app.connection.launch_cwd = Some("/launch".into());

        app.open_new_session_popup();

        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert_eq!(app.sessions.new_session_path, "/launch");
        assert_eq!(app.sessions.new_session_cursor, "/launch".len());
    }

    #[test]
    fn normalize_new_session_path_uses_launch_cwd_for_relative_paths() {
        let mut app = App::new();
        app.connection.launch_cwd = Some("/launch/base".into());

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
        app.sessions.new_session_completion = Some(PathCompletionState {
            query: "pro".into(),
            selected_index: 0,
            results: vec![crate::composer_state::FileIndexEntryLite {
                path: "/launch/project/../project-two".into(),
                is_dir: true,
            }],
        });

        assert!(app.accept_selected_new_session_completion());
        assert_eq!(app.sessions.new_session_path, "/launch/project-two/");
        assert!(app.sessions.new_session_completion.is_none());
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
        app.connection.launch_cwd = Some(dir.to_string_lossy().into_owned());
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
    use crate::domain::activity::{
        DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus,
    };

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
        assert!(app.delegates.visible_entries().is_empty());
    }

    #[test]
    fn visible_entries_returns_all_when_no_filter() {
        let mut app = App::new();
        app.delegates.delegate_entries = vec![
            make_entry("d1", "Build feature", DelegateStatus::Completed),
            make_entry("d2", "Fix tests", DelegateStatus::InProgress),
        ];
        assert_eq!(app.delegates.visible_entries().len(), 2);
    }

    #[test]
    fn visible_entries_filters_by_objective() {
        let mut app = App::new();
        app.delegates.delegate_entries = vec![
            make_entry("d1", "Build feature", DelegateStatus::Completed),
            make_entry("d2", "Fix tests", DelegateStatus::InProgress),
        ];
        app.delegates.delegate_filter = "build".into();
        let entries = app.delegates.visible_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].delegation_id, "d1");
    }

    #[test]
    fn visible_entries_filters_by_delegation_id() {
        let mut app = App::new();
        app.delegates.delegate_entries = vec![
            make_entry("abc123", "Build feature", DelegateStatus::Completed),
            make_entry("xyz789", "Fix tests", DelegateStatus::InProgress),
        ];
        app.delegates.delegate_filter = "xyz".into();
        let entries = app.delegates.visible_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].delegation_id, "xyz789");
    }

    #[test]
    fn visible_entries_filters_by_target_agent() {
        let mut app = App::new();
        app.delegates.delegate_entries = vec![
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
        app.delegates.delegate_filter = "planner".into();
        let entries = app.delegates.visible_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].delegation_id, "d1");
    }

    #[test]
    fn visible_entries_filter_is_case_insensitive() {
        let mut app = App::new();
        app.delegates.delegate_entries =
            vec![make_entry("d1", "Build Feature", DelegateStatus::Completed)];
        app.delegates.delegate_filter = "BUILD".into();
        assert_eq!(app.delegates.visible_entries().len(), 1);
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

// ── session_mode_tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod session_mode_tests {
    use super::*;

    // ── SessionModeChanged in live events ─────────────────────────────────────

    #[test]
    fn next_mode_exits_review_to_previous_mode() {
        let mut app = App::new();
        app.sessions.agent_mode = "review".into();
        app.sessions.mode_before_review = Some("plan".into());
        assert_eq!(app.sessions.next_mode(), "plan");
    }

    #[test]
    fn next_mode_from_review_defaults_to_build_without_previous_mode() {
        let mut app = App::new();
        app.sessions.agent_mode = "review".into();
        app.sessions.mode_before_review = None;
        assert_eq!(app.sessions.next_mode(), "build");
    }

    // ── SessionModeChanged in audit replay ────────────────────────────────────

    // ── session_loaded returns SetAgentMode ───────────────────────────────────

    // ── session_loaded: model/effort from audit only (no TUI cache) ─────────────
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::connection_state::ServerState;
    use crate::domain::activity::{ActivityState, SessionOp};
    use crate::domain::chat::{ChatEntry, OUTCOME_BULLET};
    use crate::domain::session::{
        ForkBoundaryKind, UndoFrame, UndoFrameStatus, UndoStackSnapshot, UndoState, UndoableTurn,
    };

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
    fn fork_boundary_derivation_uses_last_assistant_then_user_fallback() {
        let mut app = App::new();
        app.chat.messages = vec![
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

        let turns = app.chat.forkable_turns();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].message_id, "asst-1b");
        assert_eq!(turns[0].boundary_kind, ForkBoundaryKind::Assistant);
        assert_eq!(turns[1].message_id, "user-2");
        assert_eq!(turns[1].boundary_kind, ForkBoundaryKind::User);
    }

    #[test]
    fn fork_boundary_derivation_includes_turn_when_only_assistant_has_message_id() {
        let mut app = App::new();
        app.chat.messages = vec![
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

        let turns = app.chat.forkable_turns();
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
        app.chat.messages = vec![
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

        let latest = app
            .chat
            .latest_fork_boundary()
            .expect("latest fork boundary");
        assert_eq!(latest.message_id, "asst-new");
        assert_eq!(latest.boundary_kind, ForkBoundaryKind::Assistant);

        app.chat.activity = ActivityState::Streaming;
        assert!(app.chat.latest_fork_boundary().is_none());
    }

    #[test]
    fn current_undo_target_moves_left_of_frontier() {
        let mut app = App::new();
        app.chat.undoable_turns = vec![make_turn("msg-1"), make_turn("msg-2"), make_turn("msg-3")];

        assert_eq!(
            app.chat
                .current_undo_target()
                .map(|turn| turn.message_id.as_str()),
            Some("msg-3")
        );

        app.chat.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-3".into(),
                message_id: "msg-3".into(),
                status: UndoFrameStatus::Confirmed,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-3".into()),
        });

        assert_eq!(
            app.chat
                .current_undo_target()
                .map(|turn| turn.message_id.as_str()),
            Some("msg-2")
        );
    }

    #[test]
    fn build_undo_state_confirms_frames_and_preserves_frontier() {
        let mut app = App::new();
        app.chat.undoable_turns = vec![make_turn("msg-1"), make_turn("msg-2")];
        app.chat.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-msg-1".into(),
                message_id: "msg-1".into(),
                status: UndoFrameStatus::Pending,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-1".into()),
        });

        let next = app
            .chat
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
        app.chat.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "previous-turn".into(),
                message_id: "msg-1".into(),
                status: UndoFrameStatus::Pending,
                reverted_files: vec!["preserved.rs".into()],
            }],
            frontier_message_id: Some("msg-1".into()),
        });

        let state = app
            .chat
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
            app.chat
                .build_undo_state_from_server_stack(&UndoStackSnapshot::default(), None, None),
            None
        );
    }

    #[test]
    fn pending_guard_tracks_pending_frames() {
        let mut app = App::new();
        let turn = make_turn("msg-1");
        app.chat.push_pending_undo(&turn);

        assert!(app.chat.has_pending_undo());
        assert_eq!(
            app.chat
                .undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("msg-1")
        );
        assert_eq!(
            app.chat.undo_state.as_ref().map(|state| state.stack.len()),
            Some(1)
        );
        assert_eq!(
            app.chat
                .undo_state
                .as_ref()
                .map(|state| state.stack[0].status.clone()),
            Some(UndoFrameStatus::Pending)
        );
    }

    #[test]
    fn pending_session_label_stays_reserved_for_undo_and_redo() {
        let mut app = App::new();
        app.chat.activity = ActivityState::Compacting {
            token_estimate: 9_000,
        };
        assert_eq!(app.chat.pending_session_label(), None);

        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        assert_eq!(app.chat.pending_session_label(), Some("undoing"));
    }

    #[test]
    fn cancel_confirm_arms_expires_and_restores_status() {
        let mut app = App::new();
        app.chat.activity = ActivityState::Thinking;

        app.arm_cancel_confirm();
        assert!(app.chat.cancel_confirm_active());
        assert_eq!(app.diagnostics.status, "press Esc again to stop");
        assert!(
            matches!(app.diagnostics.logs.last(), Some(entry) if entry.message == "press Esc again to stop")
        );

        app.chat.pending_cancel_confirm_until = Some(Instant::now() - Duration::from_millis(1));
        app.clear_expired_cancel_confirm();
        assert!(!app.chat.cancel_confirm_active());
        assert_eq!(app.diagnostics.status, "thinking...");
        assert!(
            matches!(app.diagnostics.logs.last(), Some(entry) if entry.message == "thinking...")
        );
    }

    #[test]
    fn refresh_transient_status_preserves_connection_and_operation_precedence() {
        let mut app = App::new();
        app.connection.conn = ConnState::Disconnected;
        app.set_status(LogLevel::Warn, "connection", "connection lost - retrying");
        app.refresh_transient_status();
        assert_eq!(app.diagnostics.status, "connection lost - retrying");

        app.connection.conn = ConnState::Connected;
        app.chat.activity = ActivityState::Thinking;
        app.refresh_transient_status();
        assert_eq!(app.diagnostics.status, "thinking...");

        app.chat.activity = ActivityState::Compacting {
            token_estimate: 2048,
        };
        app.refresh_transient_status();
        assert_eq!(app.diagnostics.status, "compacting context (~2048 tokens)");

        app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
        app.refresh_transient_status();
        assert_eq!(app.diagnostics.status, "redoing...");
    }

    #[test]
    fn activity_helpers_report_turn_and_session_state() {
        let mut app = App::new();
        assert!(!app.chat.is_turn_active());
        assert!(!app.chat.has_pending_session_op());
        assert!(!app.chat.input_blocked_by_activity());
        assert!(!app.chat.should_hide_input_contents());
        assert_eq!(app.chat.pending_session_label(), None);

        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        assert!(!app.chat.is_turn_active());
        assert!(app.chat.has_pending_session_op());
        assert!(app.chat.input_blocked_by_activity());
        assert!(app.chat.should_hide_input_contents());
        assert_eq!(app.chat.pending_session_label(), Some("undoing"));

        app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
        assert!(!app.chat.is_turn_active());
        assert!(app.chat.has_pending_session_op());
        assert!(app.chat.input_blocked_by_activity());
        assert!(app.chat.should_hide_input_contents());
        assert_eq!(app.chat.pending_session_label(), Some("redoing"));

        app.chat.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };
        assert!(app.chat.is_turn_active());
        assert!(app.chat.has_cancellable_activity());
        assert!(!app.chat.has_pending_session_op());
        assert!(!app.chat.input_blocked_by_activity());
        assert!(!app.chat.should_hide_input_contents());
        assert_eq!(app.chat.pending_session_label(), None);

        app.arm_cancel_confirm();
        assert!(app.chat.input_blocked_by_activity());
        assert!(app.chat.should_hide_input_contents());
    }

    #[test]
    fn connecting_event_updates_status_retry_metadata_and_clears_cancel_confirmation() {
        let mut app = App::new();
        app.arm_cancel_confirm();

        app.handle_connection_event(ConnectionEvent::Connecting {
            attempt: 3,
            delay_ms: 2000,
        });

        assert_eq!(app.connection.conn, ConnState::Connecting);
        assert_eq!(app.connection.reconnect_attempt, 3);
        assert_eq!(app.connection.reconnect_delay_ms, Some(2000));
        assert!(app.chat.pending_cancel_confirm_until.is_none());
        assert_eq!(
            app.diagnostics.status,
            "waiting for server - retry 3 in 2.0s"
        );
        assert!(
            matches!(app.diagnostics.logs.last(), Some(entry) if entry.target == "connection" && entry.level == LogLevel::Warn)
        );
    }

    #[test]
    fn connected_event_selects_connected_or_reconnected_and_resets_retry_metadata() {
        let mut app = App::new();
        app.connection.reconnect_attempt = 3;
        app.connection.reconnect_delay_ms = Some(2000);

        app.handle_connection_event(ConnectionEvent::Connected);
        assert_eq!(app.connection.conn, ConnState::Connected);
        assert_eq!(app.connection.reconnect_attempt, 0);
        assert_eq!(app.connection.reconnect_delay_ms, None);
        assert_eq!(app.diagnostics.status, "connected");

        app.sessions.session_id = Some("session-1".into());
        app.connection.reconnect_attempt = 2;
        app.connection.reconnect_delay_ms = Some(1000);
        app.handle_connection_event(ConnectionEvent::Connected);
        assert_eq!(app.connection.conn, ConnState::Connected);
        assert_eq!(app.connection.reconnect_attempt, 0);
        assert_eq!(app.connection.reconnect_delay_ms, None);
        assert_eq!(app.diagnostics.status, "reconnected");
        assert!(
            matches!(app.diagnostics.logs.last(), Some(entry) if entry.level == LogLevel::Info && entry.message == "reconnected")
        );
    }

    #[test]
    fn disconnected_event_clears_only_transient_session_discovery_and_delay_state() {
        let mut app = App::new();
        app.connection.launch_cwd = Some("/workspace".into());
        app.connection.reconnect_attempt = 4;
        app.connection.reconnect_delay_ms = Some(4000);
        app.connection.server_state = ServerState::Running;
        app.sessions.session_id = Some("session-1".into());
        app.sessions.session_discovery_in_progress = true;
        app.sessions
            .pending_session_group_loads
            .insert(Some("/workspace".into()));
        app.sessions
            .pending_session_child_loads
            .insert("session-1".into());
        app.composer.input = "retained prompt".into();
        app.arm_cancel_confirm();

        app.handle_connection_event(ConnectionEvent::Disconnected {
            reason: "socket closed".into(),
        });

        assert_eq!(app.connection.conn, ConnState::Disconnected);
        assert_eq!(app.connection.reconnect_attempt, 4);
        assert_eq!(app.connection.reconnect_delay_ms, None);
        assert_eq!(app.connection.launch_cwd.as_deref(), Some("/workspace"));
        assert_eq!(app.connection.server_state, ServerState::Running);
        assert_eq!(app.sessions.session_id.as_deref(), Some("session-1"));
        assert!(!app.sessions.session_discovery_in_progress);
        assert!(app.sessions.pending_session_group_loads.is_empty());
        assert!(
            app.sessions
                .pending_session_child_loads
                .contains("session-1")
        );
        assert_eq!(app.composer.input, "retained prompt");
        assert!(app.chat.pending_cancel_confirm_until.is_none());
        assert_eq!(app.diagnostics.status, "connection lost - socket closed");
        assert!(
            matches!(app.diagnostics.logs.last(), Some(entry) if entry.message == "connection lost - socket closed")
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
        let mut chat = ChatState::new();
        chat.messages = messages;
        chat.backfill_elicitation_outcomes(result);
        messages = chat.messages;
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
        let mut chat = ChatState::new();
        chat.messages = messages;
        chat.backfill_elicitation_outcomes(result);
        messages = chat.messages;
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
        let mut chat = ChatState::new();
        chat.messages = messages;
        chat.backfill_elicitation_outcomes(result);
        messages = chat.messages;
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
        let mut chat = ChatState::new();
        chat.messages = messages;
        chat.backfill_elicitation_outcomes(result);
        messages = chat.messages;
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
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::session_state::{PopupItem, StartPageItem};

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
        let items = app.sessions.visible_start_items();
        assert!(items.is_empty());
    }

    // ── visible_start_items: basic structure ─────────────────────────────────

    #[test]
    fn visible_items_header_then_sessions_expanded() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(Some("/a"), &[("s1", None), ("s2", None)])];

        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![make_group(Some("/a"), &[("s1", None), ("s2", None)])];
        app.sessions.collapsed_groups.insert("/a".to_string());

        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None), ("s3", None)]),
        ];

        let items = app.sessions.visible_start_items();
        // group /a: 1 header + 1 session = 2
        // group /b: 1 header + 2 sessions = 3
        assert_eq!(items.len(), 5);
    }

    // ── visible_start_items: mixed collapse ───────────────────────────────────

    #[test]
    fn visible_items_one_group_collapsed_other_expanded() {
        let mut app = App::new();
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None), ("s3", None)]),
        ];
        app.sessions.collapsed_groups.insert("/a".to_string());

        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![make_group(
            Some("/a"),
            &[("aaa", None), ("bbb", None), ("aab", None)],
        )];
        app.sessions.session_filter = "aa".to_string();

        let items = app.sessions.visible_start_items();
        // header + "aaa" + "aab" (bbb filtered out by session_id)
        assert_eq!(items.len(), 3);
    }

    // ── visible_start_items: filter hides empty groups ────────────────────────

    #[test]
    fn visible_items_filter_hides_groups_with_no_matches() {
        let mut app = App::new();
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("aaa", None)]),
            make_group(Some("/b"), &[("bbb", None)]),
        ];
        app.sessions.session_filter = "bbb".to_string();

        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("s0", None), ("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
        ];

        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![group];
        app.sessions.session_cursor = 1;

        let action = crate::handlers::apply_session_fork_toggle_key(&mut app, false);
        assert_eq!(
            action,
            crate::handlers::SessionKeyAction::LoadMoreSessions {
                group_idx: 0,
                parent_path: vec![0],
            }
        );
        assert!(app.sessions.expanded_session_children.contains("root"));
        assert!(app.sessions.pending_session_child_loads.is_empty());

        let action = crate::handlers::apply_session_fork_toggle_key(&mut app, false);
        assert_eq!(action, crate::handlers::SessionKeyAction::None);
        assert!(app.sessions.pending_session_child_loads.is_empty());
    }

    #[test]
    fn remote_node_sessions_are_not_expandable() {
        let mut app = App::new();
        let mut group = make_group(Some("/a"), &[("remote", Some("2024-01-04T00:00:00Z"))]);
        group.sessions[0].fork_count = 1;
        group.sessions[0].node = Some("remote-host".to_string());
        app.sessions.session_groups = vec![group];

        assert!(!app.sessions.expandable_root_session(0, &[0]));
        assert!(!app.sessions.toggle_session_children(0, &[0]));
        assert!(app.sessions.expanded_session_children.is_empty());
        assert!(app.sessions.pending_session_child_loads.is_empty());
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
        app.sessions
            .expanded_session_children
            .insert("root".to_string());
        app.sessions.session_groups = vec![group];

        let items = app.sessions.visible_popup_items();

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

    // ── GroupHeader carries correct session_count ─────────────────────────────

    #[test]
    fn group_header_session_count_reflects_total_not_filtered() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(
            Some("/a"),
            &[("s1", None), ("s2", None), ("s3", None)],
        )];
        app.sessions.session_groups[0].total_count = Some(8);

        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![make_group(Some("/a"), &[("s1", None)])];

        let items = app.sessions.visible_start_items();
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
        assert!(!app.sessions.collapsed_groups.contains(&key));

        app.sessions.toggle_group_collapse(Some("/a"));
        assert!(app.sessions.collapsed_groups.contains(&key));

        app.sessions.toggle_group_collapse(Some("/a"));
        assert!(!app.sessions.collapsed_groups.contains(&key));
    }

    #[test]
    fn toggle_group_collapse_none_cwd_uses_empty_string_key() {
        let mut app = App::new();
        app.sessions.toggle_group_collapse(None);
        assert!(app.sessions.collapsed_groups.contains(""));

        app.sessions.toggle_group_collapse(None);
        assert!(!app.sessions.collapsed_groups.contains(""));
    }

    // ── MAX_RECENT_SESSIONS cap ───────────────────────────────────────────────

    #[test]
    fn visible_items_group_with_three_sessions_shows_no_show_more() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(
            Some("/a"),
            &[("s1", None), ("s2", None), ("s3", None)],
        )];
        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![make_group(
            Some("/a"),
            &[("s1", None), ("s2", None), ("s3", None), ("s4", None)],
        )];
        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![make_group(
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
        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![make_group(
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
        app.sessions.session_filter = "aaa".to_string();
        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
            make_group(Some("/c"), &[("s3", None)]),
        ];
        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
            make_group(Some("/c"), &[("s3", None)]),
            make_group(Some("/d"), &[("s4", None)]),
        ];
        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("s1", None)]),
            make_group(Some("/b"), &[("s2", None)]),
            make_group(Some("/c"), &[("s3", None)]),
            make_group(Some("/d"), &[("s4", None)]),
            make_group(Some("/e"), &[("s5", None)]),
            make_group(Some("/f"), &[("s6", None)]),
        ];
        let items = app.sessions.visible_start_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &[("aaa1", None)]),
            make_group(Some("/b"), &[("aaa2", None)]),
            make_group(Some("/c"), &[("aaa3", None)]),
            make_group(Some("/d"), &[("aaa4", None)]),
        ];
        app.sessions.session_filter = "aaa".to_string();
        let items = app.sessions.visible_start_items();
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
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::session_state::PopupItem;

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
        assert!(app.sessions.visible_popup_items().is_empty());
    }

    // ── basic structure: header then sessions ─────────────────────────────────

    #[test]
    fn popup_items_header_then_sessions() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![make_group(Some("/a"), &ids)];
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![group];

        let items = app.sessions.visible_popup_items();

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
        app.sessions.session_groups = vec![group];
        app.sessions
            .pending_session_group_loads
            .insert(Some("/workspace/project".to_string()));

        assert!(app.session_group_page_request(0).is_none());
        assert!(!app.sessions.visible_popup_items().iter().any(
            |item| matches!(item, PopupItem::LoadMore { parent_path, .. } if parent_path.is_empty())
        ));
    }

    #[test]
    fn popup_items_shows_all_groups_beyond_cap() {
        let mut app = App::new();
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &["s1"]),
            make_group(Some("/b"), &["s2"]),
            make_group(Some("/c"), &["s3"]),
            make_group(Some("/d"), &["s4"]),
            make_group(Some("/e"), &["s5"]),
        ];
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        // Collapse on the start page should NOT affect the popup
        app.sessions.collapsed_groups.insert("/a".to_string());
        let items = app.sessions.visible_popup_items();
        // Popup uses popup_collapsed_groups, not collapsed_groups
        // /a is expanded in popup → header + 2 sessions = 3
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn popup_collapsed_hides_sessions() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.sessions.popup_collapsed_groups.insert("/a".to_string());
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1"])];
        // Not in popup_collapsed_groups → expanded
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &["s1"]),
            make_group(Some("/b"), &["s2", "s3"]),
        ];
        let items = app.sessions.visible_popup_items();
        // /a: 1 header + 1 session; /b: 1 header + 2 sessions = 5
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn popup_one_group_collapsed_other_expanded() {
        let mut app = App::new();
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &["s1"]),
            make_group(Some("/b"), &["s2", "s3"]),
        ];
        app.sessions.popup_collapsed_groups.insert("/a".to_string());
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![make_group(Some("/a"), &["aaa", "bbb", "aab"])];
        app.sessions.session_filter = "aa".to_string();
        let items = app.sessions.visible_popup_items();
        // header + "aaa" + "aab" (bbb filtered out by session_id)
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn popup_filter_hides_groups_with_no_matches() {
        let mut app = App::new();
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &["aaa"]),
            make_group(Some("/b"), &["bbb"]),
        ];
        app.sessions.session_filter = "bbb".to_string();
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![
            make_group(Some("/a"), &["s0", "s1"]),
            make_group(Some("/b"), &["s2"]),
        ];
        let items = app.sessions.visible_popup_items();
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
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3"])];
        app.sessions.session_groups[0].total_count = Some(8);
        let items = app.sessions.visible_popup_items();
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
        assert!(!app.sessions.popup_collapsed_groups.contains("/a"));
        app.sessions.toggle_popup_group_collapse(Some("/a"));
        assert!(app.sessions.popup_collapsed_groups.contains("/a"));
        app.sessions.toggle_popup_group_collapse(Some("/a"));
        assert!(!app.sessions.popup_collapsed_groups.contains("/a"));
    }

    #[test]
    fn toggle_popup_collapse_none_cwd_uses_empty_string_key() {
        let mut app = App::new();
        app.sessions.toggle_popup_group_collapse(None);
        assert!(app.sessions.popup_collapsed_groups.contains(""));
        app.sessions.toggle_popup_group_collapse(None);
        assert!(!app.sessions.popup_collapsed_groups.contains(""));
    }

    #[test]
    fn toggle_popup_collapse_does_not_affect_start_page_state() {
        let mut app = App::new();
        app.sessions.toggle_popup_group_collapse(Some("/a"));
        assert!(app.sessions.popup_collapsed_groups.contains("/a"));
        // start page collapsed_groups should be untouched
        assert!(!app.sessions.collapsed_groups.contains("/a"));
    }
}
