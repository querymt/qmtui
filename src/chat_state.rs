use std::time::{Duration, Instant};

use crate::application::Effect;
use crate::composer_state::build_input_visual_layout;
use crate::diagnostics::LogLevel;
use crate::domain::activity::{ActivityState, SessionOp, SessionStatsLite};
use crate::domain::chat::{ChatEntry, format_outcome_labels};
use crate::domain::elicitation::ElicitationState;
use crate::domain::session::{
    ForkBoundaryKind, ForkTurnItem, UndoFrame, UndoFrameStatus, UndoStackSnapshot, UndoState,
    UndoableTurn,
};
use crate::domain::tool::ToolDetail;

const CANCEL_CONFIRM_TIMEOUT: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Default)]
pub(crate) struct ElicitationUiState {
    pub(crate) field_cursor: usize,
    pub(crate) option_cursor: usize,
    pub(crate) text_cursor: usize,
    pub(crate) custom_active: bool,
    pub(crate) custom_cursor: usize,
    pub(crate) custom_line_width: usize,
    pub(crate) custom_scroll: u16,
}

impl ElicitationUiState {
    pub(crate) fn current_field_index(&self, field_count: usize) -> Option<usize> {
        (field_count > 0).then(|| self.field_cursor.min(field_count - 1))
    }

    pub(crate) fn custom_insert(&mut self, input: &mut String, character: char) {
        input.insert(self.custom_cursor, character);
        self.custom_cursor += character.len_utf8();
    }

    pub(crate) fn custom_backspace(&mut self, input: &mut String) {
        if self.custom_cursor == 0 {
            return;
        }
        let previous = input[..self.custom_cursor]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        input.drain(previous..self.custom_cursor);
        self.custom_cursor = previous;
    }

    pub(crate) fn custom_delete(&mut self, input: &mut String) {
        if self.custom_cursor >= input.len() {
            return;
        }
        let next = input[self.custom_cursor..]
            .char_indices()
            .nth(1)
            .map(|(index, _)| self.custom_cursor + index)
            .unwrap_or(input.len());
        input.drain(self.custom_cursor..next);
    }

    pub(crate) fn custom_left(&mut self, input: &str) {
        if self.custom_cursor > 0 {
            self.custom_cursor = input[..self.custom_cursor]
                .char_indices()
                .last()
                .map(|(index, _)| index)
                .unwrap_or(0);
        }
    }

    pub(crate) fn custom_right(&mut self, input: &str) {
        if self.custom_cursor < input.len() {
            self.custom_cursor = input[self.custom_cursor..]
                .char_indices()
                .nth(1)
                .map(|(index, _)| self.custom_cursor + index)
                .unwrap_or(input.len());
        }
    }

    pub(crate) fn custom_home(&mut self, input: &str) {
        self.custom_cursor = input[..self.custom_cursor]
            .rfind('\n')
            .map(|index| index + 1)
            .unwrap_or(0);
    }

    pub(crate) fn custom_end(&mut self, input: &str) {
        self.custom_cursor = input[self.custom_cursor..]
            .find('\n')
            .map(|index| self.custom_cursor + index)
            .unwrap_or(input.len());
    }

    pub(crate) fn custom_move_visual(&mut self, input: &str, delta: i32) {
        let layout =
            build_input_visual_layout(input, self.custom_cursor, self.custom_line_width.max(1), 2);
        let row = (layout.cursor_row as i32 + delta)
            .clamp(0, layout.total_rows().saturating_sub(1) as i32) as usize;
        self.custom_cursor = layout.cursor_offset_for_row_col(row, layout.cursor_text_col);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UserMessageTransition {
    Ignored,
    Duplicate,
    Reconciled,
    Appended,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StreamingDeltaTransition {
    pub(crate) finalized_previous: bool,
    pub(crate) ignored_duplicate: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ElicitationTransition {
    pub(crate) inserted: bool,
    pub(crate) finalized_streaming: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AssistantMessageTransition {
    Ignored,
    Duplicate,
    ReplacedThinking,
    Appended,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChatAction {
    TurnStarted {
        is_replay: bool,
    },
    UserMessage {
        text: String,
        message_id: Option<String>,
        is_replay: bool,
    },
    AssistantContentDelta {
        content: String,
        message_id: Option<String>,
    },
    AssistantThinkingDelta {
        content: String,
        message_id: Option<String>,
        is_replay: bool,
    },
    AssistantMessage {
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    },
    Cancelled {
        is_replay: bool,
    },
    Finished {
        finish_reason: String,
        is_replay: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatTransition {
    TurnStarted,
    UserMessage(UserMessageTransition),
    AssistantContentDelta(StreamingDeltaTransition),
    AssistantThinkingDelta(StreamingDeltaTransition),
    AssistantMessage(AssistantMessageTransition),
    Cancelled,
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChatCoordination {
    InvalidateContentCache,
    InvalidateThinkingCache,
    InvalidateCardCache,
    Status {
        level: LogLevel,
        target: &'static str,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatOutcome {
    pub(crate) transition: ChatTransition,
    pub(crate) coordination: Vec<ChatCoordination>,
    pub(crate) effects: Vec<Effect>,
}

pub(crate) struct ChatState {
    pub(crate) messages: Vec<ChatEntry>,
    pub(crate) pending_prompt_seq: u64,
    pub(crate) fork_filter: String,
    pub(crate) fork_cursor: usize,
    pub(crate) pending_fork_message_id: Option<String>,
    pub(crate) scroll_offset: u16,
    pub(crate) activity: ActivityState,
    pub(crate) streaming_content: String,
    pub(crate) streaming_content_message_id: Option<String>,
    pub(crate) streaming_thinking: String,
    pub(crate) streaming_thinking_message_id: Option<String>,
    pub(crate) last_compaction_token_estimate: Option<u32>,
    pub(crate) elicitation: Option<ElicitationState>,
    pub(crate) elicitation_ui: Option<ElicitationUiState>,
    pub(crate) show_thinking: bool,
    pub(crate) undo_state: Option<UndoState>,
    pub(crate) undoable_turns: Vec<UndoableTurn>,
    pub(crate) recent_prompt_text: Option<String>,
    pub(crate) cumulative_cost: Option<f64>,
    pub(crate) context_limit: u64,
    pub(crate) session_stats: SessionStatsLite,
    pub(crate) pending_cancel_confirm_until: Option<Instant>,
    pub(crate) suppress_turn_output: bool,
}

impl ChatState {
    pub(crate) fn reduce(&mut self, action: ChatAction) -> ChatOutcome {
        let (transition, coordination) = match action {
            ChatAction::TurnStarted { is_replay } => {
                self.begin_turn(is_replay);
                (
                    ChatTransition::TurnStarted,
                    vec![
                        ChatCoordination::InvalidateContentCache,
                        ChatCoordination::InvalidateThinkingCache,
                        ChatCoordination::Status {
                            level: LogLevel::Debug,
                            target: "activity",
                            message: "thinking...".into(),
                        },
                    ],
                )
            }
            ChatAction::UserMessage {
                text,
                message_id,
                is_replay,
            } => {
                let transition = self.push_user_message(text, message_id, is_replay);
                let coordination = matches!(transition, UserMessageTransition::Reconciled)
                    .then_some(ChatCoordination::InvalidateCardCache)
                    .into_iter()
                    .collect();
                (ChatTransition::UserMessage(transition), coordination)
            }
            ChatAction::AssistantContentDelta {
                content,
                message_id,
            } => {
                let transition = self.append_streaming_content(&content, message_id);
                let coordination = Self::stream_boundary_coordination(transition);
                (
                    ChatTransition::AssistantContentDelta(transition),
                    coordination,
                )
            }
            ChatAction::AssistantThinkingDelta {
                content,
                message_id,
                is_replay,
            } => {
                let transition = self.append_streaming_thinking(&content, message_id, is_replay);
                let coordination = Self::stream_boundary_coordination(transition);
                (
                    ChatTransition::AssistantThinkingDelta(transition),
                    coordination,
                )
            }
            ChatAction::AssistantMessage {
                content,
                thinking,
                message_id,
            } => {
                let transition = self.push_assistant_message(content, thinking, message_id);
                let mut coordination = vec![
                    ChatCoordination::InvalidateContentCache,
                    ChatCoordination::InvalidateThinkingCache,
                ];
                if transition == AssistantMessageTransition::ReplacedThinking {
                    coordination.push(ChatCoordination::InvalidateCardCache);
                }
                (ChatTransition::AssistantMessage(transition), coordination)
            }
            ChatAction::Cancelled { is_replay } => {
                self.cancel_turn(is_replay);
                (
                    ChatTransition::Cancelled,
                    vec![
                        ChatCoordination::InvalidateContentCache,
                        ChatCoordination::InvalidateThinkingCache,
                        ChatCoordination::Status {
                            level: LogLevel::Warn,
                            target: "activity",
                            message: "cancelled".into(),
                        },
                    ],
                )
            }
            ChatAction::Finished {
                finish_reason,
                is_replay,
            } => {
                self.finish_turn(is_replay);
                (
                    ChatTransition::Finished,
                    vec![
                        ChatCoordination::InvalidateContentCache,
                        ChatCoordination::InvalidateThinkingCache,
                        ChatCoordination::Status {
                            level: LogLevel::Debug,
                            target: "activity",
                            message: format!("finished: {finish_reason}"),
                        },
                    ],
                )
            }
        };
        ChatOutcome {
            transition,
            coordination,
            effects: Vec::new(),
        }
    }

    fn stream_boundary_coordination(transition: StreamingDeltaTransition) -> Vec<ChatCoordination> {
        if transition.finalized_previous {
            vec![
                ChatCoordination::InvalidateContentCache,
                ChatCoordination::InvalidateThinkingCache,
                ChatCoordination::InvalidateCardCache,
            ]
        } else {
            Vec::new()
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            messages: Vec::new(),
            pending_prompt_seq: 0,
            fork_filter: String::new(),
            fork_cursor: 0,
            pending_fork_message_id: None,
            scroll_offset: 0,
            activity: ActivityState::Idle,
            streaming_content: String::new(),
            streaming_content_message_id: None,
            streaming_thinking: String::new(),
            streaming_thinking_message_id: None,
            last_compaction_token_estimate: None,
            elicitation: None,
            elicitation_ui: None,
            show_thinking: true,
            undo_state: None,
            undoable_turns: Vec::new(),
            recent_prompt_text: None,
            cumulative_cost: None,
            context_limit: 0,
            session_stats: SessionStatsLite::default(),
            pending_cancel_confirm_until: None,
            suppress_turn_output: false,
        }
    }

    pub(crate) fn reset_for_session_switch(&mut self) {
        self.messages.clear();
        self.pending_prompt_seq = 0;
        self.streaming_content.clear();
        self.streaming_content_message_id = None;
        self.streaming_thinking.clear();
        self.streaming_thinking_message_id = None;
        self.scroll_offset = 0;
        self.undo_state = None;
        self.undoable_turns.clear();
        self.recent_prompt_text = None;
        self.suppress_turn_output = false;
        self.last_compaction_token_estimate = None;
        self.elicitation = None;
        self.elicitation_ui = None;
        self.pending_cancel_confirm_until = None;
        self.cumulative_cost = None;
        self.session_stats = SessionStatsLite::default();
    }

    pub(crate) fn cancel_confirm_active(&self) -> bool {
        self.cancel_confirm_active_at(Instant::now())
    }

    fn cancel_confirm_active_at(&self, now: Instant) -> bool {
        self.pending_cancel_confirm_until
            .map(|deadline| now <= deadline)
            .unwrap_or(false)
    }

    pub(crate) fn arm_cancel_confirm(&mut self) {
        self.pending_cancel_confirm_until = Some(Instant::now() + CANCEL_CONFIRM_TIMEOUT);
    }

    pub(crate) fn clear_cancel_confirm(&mut self) {
        self.pending_cancel_confirm_until = None;
    }

    pub(crate) fn clear_expired_cancel_confirm(&mut self) -> bool {
        if self.pending_cancel_confirm_until.is_some() && !self.cancel_confirm_active() {
            self.clear_cancel_confirm();
            true
        } else {
            false
        }
    }

    pub(crate) fn is_turn_active(&self) -> bool {
        matches!(
            self.activity,
            ActivityState::Thinking
                | ActivityState::Streaming
                | ActivityState::RunningTool { .. }
                | ActivityState::Compacting { .. }
        )
    }

    pub(crate) fn has_cancellable_activity(&self) -> bool {
        self.is_turn_active()
    }

    pub(crate) fn has_pending_session_op(&self) -> bool {
        matches!(self.activity, ActivityState::SessionOp(_))
    }

    pub(crate) fn input_blocked_by_activity(&self) -> bool {
        self.elicitation.is_some()
            || self.has_pending_session_op()
            || self.pending_cancel_confirm_until.is_some()
    }

    pub(crate) fn should_hide_input_contents(&self) -> bool {
        self.input_blocked_by_activity()
    }

    pub(crate) fn activity_status_text(&self) -> Option<String> {
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

    pub(crate) fn pending_session_label(&self) -> Option<&'static str> {
        match self.activity {
            ActivityState::SessionOp(SessionOp::Undo) => Some("undoing"),
            ActivityState::SessionOp(SessionOp::Redo) => Some("redoing"),
            _ => None,
        }
    }

    pub(crate) fn forkable_turns(&self) -> Vec<ForkTurnItem> {
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

    pub(crate) fn filtered_fork_turns(&self) -> Vec<ForkTurnItem> {
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

    pub(crate) fn visible_fork_turns(&self) -> Vec<ForkTurnItem> {
        self.filtered_fork_turns().into_iter().rev().collect()
    }

    pub(crate) fn latest_fork_boundary(&self) -> Option<ForkTurnItem> {
        if self.is_turn_active() {
            None
        } else {
            self.forkable_turns().into_iter().last()
        }
    }

    pub(crate) fn reset_fork_selector(&mut self) {
        self.fork_filter.clear();
        self.fork_cursor = 0;
    }

    pub(crate) fn move_fork_cursor(&mut self, delta: isize) {
        self.fork_cursor =
            move_wrapping_cursor(self.fork_cursor, self.visible_fork_turns().len(), delta);
    }

    pub(crate) fn fork_filter_insert(&mut self, character: char) {
        self.fork_filter.push(character);
        self.fork_cursor = 0;
    }

    pub(crate) fn fork_filter_backspace(&mut self) {
        self.fork_filter.pop();
        self.fork_cursor = 0;
    }

    pub(crate) fn selected_fork_turn(&self) -> Option<ForkTurnItem> {
        self.visible_fork_turns().get(self.fork_cursor).cloned()
    }

    pub(crate) fn push_pending_prompt(&mut self, text: String) -> String {
        self.pending_prompt_seq = self.pending_prompt_seq.saturating_add(1);
        let local_id = format!("local:pending:{}", self.pending_prompt_seq);
        self.messages.push(ChatEntry::User {
            text,
            message_id: Some(local_id.clone()),
        });
        self.scroll_offset = 0;
        local_id
    }

    pub(crate) fn rollback_pending_prompt(&mut self, local_id: &str) -> bool {
        let old_len = self.messages.len();
        self.messages.retain(|entry| {
            !matches!(
                entry,
                ChatEntry::User {
                    message_id: Some(message_id),
                    ..
                } if message_id == local_id
            )
        });
        old_len != self.messages.len()
    }

    pub(crate) fn begin_llm_request_span(&mut self, timestamp: Option<i64>) {
        if self.session_stats.open_llm_request_ts.is_none()
            && self.session_stats.open_llm_request_instant.is_none()
        {
            self.session_stats.open_llm_request_ts = timestamp;
            self.session_stats.open_llm_request_instant = Some(Instant::now());
        }
    }

    pub(crate) fn end_llm_request_span(&mut self, timestamp: Option<i64>) {
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

    pub(crate) fn record_tool_call(&mut self) {
        self.session_stats.total_tool_calls = self.session_stats.total_tool_calls.saturating_add(1);
    }

    pub(crate) fn record_context_tokens(&mut self, context_tokens: u64) {
        self.session_stats.latest_context_tokens = Some(context_tokens);
    }

    pub(crate) fn apply_usage(&mut self, used: u64, size: u64, cost_usd: Option<f64>) {
        if used > 0 {
            self.record_context_tokens(used);
        }
        if size > 0 {
            self.context_limit = size;
        }
        if let Some(cost_usd) = cost_usd {
            self.cumulative_cost = Some(cost_usd);
        }
    }

    pub(crate) fn add_active_llm_duration(&mut self, duration: Duration) {
        self.session_stats.active_llm_duration += duration;
    }

    pub(crate) fn llm_request_elapsed(&self) -> Option<Duration> {
        let mut elapsed = self.session_stats.active_llm_duration;
        if let Some(started) = self.session_stats.open_llm_request_instant {
            elapsed += started.elapsed();
        }
        (!elapsed.is_zero()).then_some(elapsed)
    }

    pub(crate) fn has_pending_undo(&self) -> bool {
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

    pub(crate) fn current_undo_target(&self) -> Option<&UndoableTurn> {
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

    pub(crate) fn can_redo(&self) -> bool {
        self.undo_state
            .as_ref()
            .map(|state| !state.stack.is_empty())
            .unwrap_or(false)
    }

    pub(crate) fn push_pending_undo(&mut self, turn: &UndoableTurn) {
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

    pub(crate) fn build_undo_state_from_server_stack(
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

    pub(crate) fn push_undoable_user_turn(&mut self, message_id: String, text: String) {
        if !self
            .undoable_turns
            .iter()
            .any(|turn| turn.message_id == message_id)
        {
            self.undoable_turns.push(UndoableTurn {
                turn_id: message_id.clone(),
                message_id,
                text,
            });
        }
    }

    pub(crate) fn begin_turn(&mut self, is_replay: bool) {
        self.clear_cancel_confirm();
        if !is_replay {
            self.begin_llm_request_span(None);
        }
        self.activity = ActivityState::Thinking;
        self.streaming_content.clear();
        self.streaming_content_message_id = None;
        self.clear_streaming_thinking();
    }

    pub(crate) fn append_streaming_content(
        &mut self,
        content: &str,
        message_id: Option<String>,
    ) -> StreamingDeltaTransition {
        if self.is_turn_active() {
            self.activity = ActivityState::Streaming;
        }
        let active_message_id = self
            .streaming_content_message_id
            .as_deref()
            .or(self.streaming_thinking_message_id.as_deref());
        let finalized_previous = message_id
            .as_deref()
            .is_some_and(|incoming| active_message_id.is_some_and(|active| active != incoming))
            && self.finalize_streaming_segment();
        if self.streaming_content.is_empty()
            || (self.streaming_content_message_id.is_none() && message_id.is_some())
        {
            self.streaming_content_message_id = message_id;
        }
        self.streaming_content.push_str(content);
        StreamingDeltaTransition {
            finalized_previous,
            ignored_duplicate: false,
        }
    }

    pub(crate) fn append_streaming_thinking(
        &mut self,
        content: &str,
        message_id: Option<String>,
        is_replay: bool,
    ) -> StreamingDeltaTransition {
        if is_replay
            && message_id.as_deref().is_some_and(|incoming| {
                self.messages.iter().any(|entry| {
                    matches!(entry, ChatEntry::Thinking { message_id: Some(existing), .. } if existing == incoming)
                }) || (self.streaming_thinking_message_id.as_deref() == Some(incoming)
                    && self.streaming_thinking == content)
            })
        {
            return StreamingDeltaTransition {
                finalized_previous: false,
                ignored_duplicate: true,
            };
        }

        let active_message_id = self
            .streaming_content_message_id
            .as_deref()
            .or(self.streaming_thinking_message_id.as_deref());
        let finalized_previous = message_id
            .as_deref()
            .is_some_and(|incoming| active_message_id.is_some_and(|active| active != incoming))
            && self.finalize_streaming_segment();
        if self.streaming_thinking.is_empty()
            || (self.streaming_thinking_message_id.is_none() && message_id.is_some())
        {
            self.streaming_thinking_message_id = message_id;
        }
        self.streaming_thinking.push_str(content);
        StreamingDeltaTransition {
            finalized_previous,
            ignored_duplicate: false,
        }
    }

    pub(crate) fn finalize_streaming_segment(&mut self) -> bool {
        if self.streaming_content.is_empty() && self.streaming_thinking.is_empty() {
            return false;
        }
        let content = std::mem::take(&mut self.streaming_content);
        let content_message_id = self.streaming_content_message_id.take();
        let thinking = (!self.streaming_thinking.is_empty())
            .then(|| std::mem::take(&mut self.streaming_thinking));
        let thinking_message_id = self.streaming_thinking_message_id.take();

        if content.is_empty() {
            if let Some(thinking) = thinking {
                self.messages.push(ChatEntry::Thinking {
                    content: thinking,
                    message_id: thinking_message_id,
                });
            }
        } else {
            self.messages.push(ChatEntry::Assistant {
                content,
                thinking,
                message_id: content_message_id.or(thinking_message_id),
            });
        }
        true
    }

    pub(crate) fn clear_streaming_thinking(&mut self) {
        self.streaming_thinking.clear();
        self.streaming_thinking_message_id = None;
    }

    pub(crate) fn push_streaming_thinking_entry(&mut self) -> bool {
        if self.streaming_thinking.is_empty() {
            return false;
        }
        self.messages.push(ChatEntry::Thinking {
            content: std::mem::take(&mut self.streaming_thinking),
            message_id: self.streaming_thinking_message_id.take(),
        });
        true
    }

    pub(crate) fn cancel_turn(&mut self, is_replay: bool) {
        if !is_replay {
            self.end_llm_request_span(None);
        }
        self.activity = ActivityState::Idle;
        self.streaming_content.clear();
        self.streaming_content_message_id = None;
        self.clear_streaming_thinking();
    }

    pub(crate) fn finish_turn(&mut self, is_replay: bool) {
        self.cancel_turn(is_replay);
    }

    pub(crate) fn push_user_message(
        &mut self,
        text: String,
        message_id: Option<String>,
        is_replay: bool,
    ) -> UserMessageTransition {
        if text.is_empty() {
            return UserMessageTransition::Ignored;
        }
        if let Some(message_id) = message_id.as_deref()
            && self.messages.iter().any(|entry| {
                matches!(entry, ChatEntry::User { message_id: Some(mid), .. } if mid == message_id)
            })
        {
            self.recent_prompt_text = Some(text);
            self.suppress_turn_output = false;
            return UserMessageTransition::Duplicate;
        }
        if !is_replay
            && let Some(entry) = self.messages.iter_mut().find(|entry| {
                matches!(
                    entry,
                    ChatEntry::User {
                        text: pending_text,
                        message_id: Some(pending_id),
                    } if pending_id.starts_with("local:pending:")
                        && pending_text.trim() == text.trim()
                )
            })
        {
            if let ChatEntry::User {
                text: pending_text,
                message_id: pending_id,
            } = entry
            {
                *pending_text = text.clone();
                *pending_id = message_id.clone();
            }
            self.recent_prompt_text = Some(text.clone());
            self.suppress_turn_output = false;
            if let Some(message_id) = message_id {
                self.push_undoable_user_turn(message_id, text);
            }
            return UserMessageTransition::Reconciled;
        }
        if !is_replay && (self.undo_state.is_some() || self.suppress_turn_output) {
            return UserMessageTransition::Ignored;
        }
        self.suppress_turn_output = false;
        self.messages.push(ChatEntry::User {
            text: text.clone(),
            message_id: message_id.clone(),
        });
        self.recent_prompt_text = Some(text.clone());
        if let Some(message_id) = message_id {
            self.push_undoable_user_turn(message_id, text);
        }
        UserMessageTransition::Appended
    }

    pub(crate) fn push_assistant_message(
        &mut self,
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    ) -> AssistantMessageTransition {
        let explicit_thinking = thinking.filter(|text| !text.is_empty());
        let thinking_message_id = message_id
            .clone()
            .or_else(|| self.streaming_thinking_message_id.clone());
        self.streaming_content.clear();
        self.streaming_content_message_id = None;
        if self.is_turn_active() {
            self.activity = ActivityState::Thinking;
        }
        if content.is_empty() && explicit_thinking.is_none() && self.streaming_thinking.is_empty() {
            self.streaming_thinking_message_id = None;
            return AssistantMessageTransition::Ignored;
        }
        self.recent_prompt_text = None;
        if self.suppress_turn_output {
            self.clear_streaming_thinking();
            return AssistantMessageTransition::Ignored;
        }

        if let Some(message_id) = message_id.as_deref() {
            if self.messages.iter().any(|entry| {
                matches!(entry, ChatEntry::Assistant { message_id: Some(mid), .. } if mid == message_id)
            }) {
                self.clear_streaming_thinking();
                return AssistantMessageTransition::Duplicate;
            }

            if let Some(index) = self.messages.iter().position(|entry| {
                matches!(entry, ChatEntry::Thinking { message_id: Some(mid), .. } if mid == message_id)
            }) {
                if content.is_empty() {
                    self.clear_streaming_thinking();
                    return AssistantMessageTransition::Duplicate;
                }
                let existing_thinking = match &self.messages[index] {
                    ChatEntry::Thinking { content, .. } => content.clone(),
                    _ => String::new(),
                };
                let streaming_thinking = (!self.streaming_thinking.is_empty())
                    .then(|| std::mem::take(&mut self.streaming_thinking));
                let thinking_text = explicit_thinking
                    .or_else(|| (!existing_thinking.is_empty()).then_some(existing_thinking))
                    .or(streaming_thinking);
                self.streaming_thinking_message_id = None;
                self.messages[index] = ChatEntry::Assistant {
                    content,
                    thinking: thinking_text,
                    message_id: Some(message_id.to_string()),
                };
                return AssistantMessageTransition::ReplacedThinking;
            }
        }

        let streaming_thinking = (!self.streaming_thinking.is_empty())
            .then(|| std::mem::take(&mut self.streaming_thinking));
        let thinking_text = explicit_thinking.or(streaming_thinking);
        self.streaming_thinking_message_id = None;
        if content.is_empty() {
            if let Some(thinking) = thinking_text {
                self.messages.push(ChatEntry::Thinking {
                    content: thinking,
                    message_id: thinking_message_id,
                });
            }
        } else {
            self.messages.push(ChatEntry::Assistant {
                content,
                thinking: thinking_text,
                message_id,
            });
        }
        AssistantMessageTransition::Appended
    }

    pub(crate) fn reconcile_tool_call_start(
        &mut self,
        tool_call_id: Option<&str>,
        tool_name: &str,
        detail: ToolDetail,
    ) -> bool {
        let Some(tool_call_id) = tool_call_id else {
            return false;
        };
        let fallback_name = format!("{tool_name} (failed)");
        for entry in self.messages.iter_mut().rev() {
            if let ChatEntry::ToolCall {
                tool_call_id: Some(existing),
                name,
                detail: existing_detail,
                ..
            } = entry
            {
                if existing != tool_call_id {
                    continue;
                }
                let is_failed_fallback = name == &fallback_name;
                if is_failed_fallback {
                    *name = tool_name.to_string();
                }
                if (is_failed_fallback || matches!(existing_detail, ToolDetail::None))
                    && !matches!(detail, ToolDetail::None)
                {
                    *existing_detail = detail;
                }
                return true;
            }
        }
        false
    }

    pub(crate) fn mark_tool_call_failed(
        &mut self,
        tool_call_id: Option<&str>,
        tool_name: &str,
    ) -> bool {
        let Some(tool_call_id) = tool_call_id else {
            return false;
        };
        let fallback_name = format!("{tool_name} (failed)");
        for entry in self.messages.iter_mut().rev() {
            if let ChatEntry::ToolCall {
                tool_call_id: Some(existing),
                name,
                is_error,
                ..
            } = entry
                && existing == tool_call_id
                && (name == tool_name || name == &fallback_name)
            {
                *is_error = true;
                return true;
            }
        }
        false
    }

    pub(crate) fn push_tool_call(
        &mut self,
        tool_call_id: Option<String>,
        name: String,
        is_error: bool,
        detail: ToolDetail,
    ) {
        self.messages.push(ChatEntry::ToolCall {
            tool_call_id,
            name,
            is_error,
            detail,
        });
    }

    pub(crate) fn push_error(&mut self, message: &str) -> bool {
        if self
            .messages
            .iter()
            .any(|entry| matches!(entry, ChatEntry::Error(existing) if existing == message))
        {
            false
        } else {
            self.messages.push(ChatEntry::Error(message.to_string()));
            true
        }
    }

    pub(crate) fn push_elicitation(
        &mut self,
        active: Option<ElicitationState>,
        elicitation_id: String,
        message: String,
        source: String,
        outcome: Option<String>,
    ) -> ElicitationTransition {
        if self.messages.iter().any(|entry| {
            matches!(
                entry,
                ChatEntry::Elicitation {
                    elicitation_id: existing_id,
                    ..
                } if existing_id == &elicitation_id
            )
        }) {
            return ElicitationTransition::default();
        }

        let finalized_streaming = self.finalize_streaming_segment();
        if let Some(active) = active {
            self.elicitation = Some(active);
            self.elicitation_ui = Some(ElicitationUiState::default());
        }
        self.messages.push(ChatEntry::Elicitation {
            elicitation_id,
            message,
            source,
            outcome,
        });
        self.scroll_offset = 0;
        ElicitationTransition {
            inserted: true,
            finalized_streaming,
        }
    }

    pub(crate) fn resolve_elicitation(&mut self, elicitation_id: &str, outcome: &str) -> bool {
        let mut resolved = false;
        for entry in &mut self.messages {
            if let ChatEntry::Elicitation {
                elicitation_id: existing_id,
                outcome: existing_outcome,
                ..
            } = entry
                && existing_id == elicitation_id
            {
                *existing_outcome = Some(outcome.to_string());
                resolved = true;
                break;
            }
        }
        self.elicitation = None;
        self.elicitation_ui = None;
        resolved
    }

    pub(crate) fn backfill_elicitation_outcomes(&mut self, result_str: &str) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(result_str) else {
            return;
        };
        let Some(answers) = value.get("answers").and_then(|answers| answers.as_array()) else {
            return;
        };

        let mut answer_iter = answers.iter();
        for entry in &mut self.messages {
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
                .and_then(|answers| answers.as_array())
                .map(|answers| answers.iter().filter_map(|answer| answer.as_str()))
                .into_iter()
                .flatten();
            *outcome = Some(format_outcome_labels(labels));
        }
    }
}

fn move_wrapping_cursor(cursor: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        0
    } else {
        (cursor as isize + delta).rem_euclid(len as isize) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::OUTCOME_BULLET;
    use crate::domain::tool::ToolDetail;

    fn user(text: &str, message_id: Option<&str>) -> ChatEntry {
        ChatEntry::User {
            text: text.into(),
            message_id: message_id.map(str::to_string),
        }
    }

    fn assistant(content: &str, message_id: Option<&str>) -> ChatEntry {
        ChatEntry::Assistant {
            content: content.into(),
            thinking: None,
            message_id: message_id.map(str::to_string),
        }
    }

    fn turn(message_id: &str) -> UndoableTurn {
        UndoableTurn {
            turn_id: format!("turn-{message_id}"),
            message_id: message_id.into(),
            text: format!("prompt {message_id}"),
        }
    }

    #[test]
    fn constructor_uses_exact_twenty_three_defaults() {
        let chat = ChatState::new();
        assert!(chat.messages.is_empty());
        assert_eq!(chat.pending_prompt_seq, 0);
        assert!(chat.fork_filter.is_empty());
        assert_eq!(chat.fork_cursor, 0);
        assert!(chat.pending_fork_message_id.is_none());
        assert_eq!(chat.scroll_offset, 0);
        assert_eq!(chat.activity, ActivityState::Idle);
        assert!(chat.streaming_content.is_empty());
        assert!(chat.streaming_content_message_id.is_none());
        assert!(chat.streaming_thinking.is_empty());
        assert!(chat.streaming_thinking_message_id.is_none());
        assert!(chat.last_compaction_token_estimate.is_none());
        assert!(chat.elicitation.is_none());
        assert!(chat.elicitation_ui.is_none());
        assert!(chat.show_thinking);
        assert!(chat.undo_state.is_none());
        assert!(chat.undoable_turns.is_empty());
        assert!(chat.recent_prompt_text.is_none());
        assert!(chat.cumulative_cost.is_none());
        assert_eq!(chat.context_limit, 0);
        assert_eq!(chat.session_stats, SessionStatsLite::default());
        assert!(chat.pending_cancel_confirm_until.is_none());
        assert!(!chat.suppress_turn_output);
    }

    #[test]
    fn session_switch_reset_clears_exact_scope_and_preserves_survivors() {
        let mut chat = ChatState::new();
        chat.messages.push(ChatEntry::Error("stale".into()));
        chat.pending_prompt_seq = 8;
        chat.fork_filter = "keep".into();
        chat.fork_cursor = 3;
        chat.pending_fork_message_id = Some("keep-fork".into());
        chat.scroll_offset = 7;
        chat.activity = ActivityState::Streaming;
        chat.streaming_content = "content".into();
        chat.streaming_content_message_id = Some("content-id".into());
        chat.streaming_thinking = "thinking".into();
        chat.streaming_thinking_message_id = Some("thinking-id".into());
        chat.last_compaction_token_estimate = Some(42);
        chat.elicitation = Some(ElicitationState::new_for_test(Vec::new()));
        chat.elicitation_ui = Some(ElicitationUiState::default());
        chat.show_thinking = false;
        chat.undo_state = Some(UndoState {
            stack: Vec::new(),
            frontier_message_id: Some("frontier".into()),
        });
        chat.undoable_turns.push(turn("turn"));
        chat.recent_prompt_text = Some("prompt".into());
        chat.cumulative_cost = Some(1.25);
        chat.context_limit = 200_000;
        chat.session_stats.total_tool_calls = 2;
        chat.pending_cancel_confirm_until = Some(Instant::now());
        chat.suppress_turn_output = true;

        chat.reset_for_session_switch();

        assert!(chat.messages.is_empty());
        assert_eq!(chat.pending_prompt_seq, 0);
        assert_eq!(chat.fork_filter, "keep");
        assert_eq!(chat.fork_cursor, 3);
        assert_eq!(chat.pending_fork_message_id.as_deref(), Some("keep-fork"));
        assert_eq!(chat.scroll_offset, 0);
        assert_eq!(chat.activity, ActivityState::Streaming);
        assert!(chat.streaming_content.is_empty());
        assert!(chat.streaming_content_message_id.is_none());
        assert!(chat.streaming_thinking.is_empty());
        assert!(chat.streaming_thinking_message_id.is_none());
        assert!(chat.last_compaction_token_estimate.is_none());
        assert!(chat.elicitation.is_none());
        assert!(chat.elicitation_ui.is_none());
        assert!(!chat.show_thinking);
        assert!(chat.undo_state.is_none());
        assert!(chat.undoable_turns.is_empty());
        assert!(chat.recent_prompt_text.is_none());
        assert!(chat.cumulative_cost.is_none());
        assert_eq!(chat.context_limit, 200_000);
        assert_eq!(chat.session_stats, SessionStatsLite::default());
        assert!(chat.pending_cancel_confirm_until.is_none());
        assert!(!chat.suppress_turn_output);
    }

    #[test]
    fn fork_filter_order_wrap_and_selection_preserve_boundaries() {
        let mut chat = ChatState::new();
        chat.messages = vec![
            user("first alpha", Some("user-1")),
            assistant("reply one", Some("assistant-1")),
            user("second beta", Some("user-2")),
            user("third alpha", Some("user-3")),
        ];
        let turns = chat.forkable_turns();
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].message_id, "assistant-1");
        assert_eq!(turns[0].boundary_kind, ForkBoundaryKind::Assistant);
        assert_eq!(turns[1].message_id, "user-2");
        assert_eq!(turns[1].boundary_kind, ForkBoundaryKind::User);

        chat.fork_filter_insert('a');
        assert_eq!(chat.visible_fork_turns().len(), 3);
        assert_eq!(chat.selected_fork_turn().unwrap().message_id, "user-3");
        chat.move_fork_cursor(-1);
        assert_eq!(chat.selected_fork_turn().unwrap().message_id, "assistant-1");
        chat.move_fork_cursor(1);
        assert_eq!(chat.selected_fork_turn().unwrap().message_id, "user-3");
        chat.fork_filter = "beta".into();
        chat.fork_cursor = 9;
        chat.move_fork_cursor(1);
        assert_eq!(chat.fork_cursor, 0);
        assert_eq!(chat.selected_fork_turn().unwrap().message_id, "user-2");
    }

    #[test]
    fn pending_prompt_ids_saturate_and_rollback_only_the_target() {
        let mut chat = ChatState::new();
        let first = chat.push_pending_prompt("first".into());
        let second = chat.push_pending_prompt("second".into());
        chat.messages.push(assistant("answer", Some("assistant")));
        assert_eq!(first, "local:pending:1");
        assert_eq!(second, "local:pending:2");
        assert!(chat.rollback_pending_prompt(&first));
        assert!(!chat.rollback_pending_prompt("missing"));
        assert!(chat.messages.iter().any(
            |entry| matches!(entry, ChatEntry::User { message_id: Some(id), .. } if id == &second)
        ));
        assert!(
            chat.messages
                .iter()
                .any(|entry| matches!(entry, ChatEntry::Assistant { .. }))
        );
    }

    #[test]
    fn cancel_deadline_and_activity_status_transitions_are_local() {
        let mut chat = ChatState::new();
        chat.activity = ActivityState::Compacting {
            token_estimate: 2048,
        };
        assert!(chat.is_turn_active());
        assert_eq!(
            chat.activity_status_text().as_deref(),
            Some("compacting context (~2048 tokens)")
        );
        chat.arm_cancel_confirm();
        assert!(chat.cancel_confirm_active());
        chat.pending_cancel_confirm_until = Some(Instant::now() - Duration::from_millis(1));
        assert!(chat.clear_expired_cancel_confirm());
        assert!(!chat.cancel_confirm_active());
        chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        assert!(chat.has_pending_session_op());
        assert_eq!(chat.pending_session_label(), Some("undoing"));
        assert!(chat.input_blocked_by_activity());
    }

    #[test]
    fn timing_usage_cost_context_and_tool_stats_accumulate() {
        let mut chat = ChatState::new();
        chat.begin_llm_request_span(Some(100));
        chat.record_tool_call();
        chat.end_llm_request_span(Some(140));
        chat.apply_usage(2048, 8192, Some(0.0123));
        chat.add_active_llm_duration(Duration::from_secs(2));
        assert_eq!(chat.llm_request_elapsed(), Some(Duration::from_secs(42)));
        assert_eq!(chat.session_stats.latest_context_tokens, Some(2048));
        assert_eq!(chat.session_stats.total_tool_calls, 1);
        assert_eq!(chat.context_limit, 8192);
        assert_eq!(chat.cumulative_cost, Some(0.0123));
    }

    #[test]
    fn undo_frontier_pending_and_server_rebuild_preserve_order() {
        let mut chat = ChatState::new();
        chat.undoable_turns = vec![turn("one"), turn("two"), turn("three")];
        assert_eq!(chat.current_undo_target().unwrap().message_id, "three");
        let target = chat.current_undo_target().unwrap().clone();
        chat.push_pending_undo(&target);
        assert!(chat.has_pending_undo());
        assert_eq!(chat.current_undo_target().unwrap().message_id, "two");

        let state = chat
            .build_undo_state_from_server_stack(
                &UndoStackSnapshot {
                    message_ids: vec!["three".into(), "unknown".into()],
                },
                Some("unknown"),
                Some(&["src/lib.rs".into()]),
            )
            .unwrap();
        assert_eq!(state.frontier_message_id.as_deref(), Some("unknown"));
        assert_eq!(state.stack[0].turn_id, "turn-three");
        assert_eq!(state.stack[1].turn_id, "unknown");
        assert_eq!(state.stack[1].reverted_files, ["src/lib.rs"]);
    }

    #[test]
    fn elicitation_resolution_and_backfill_update_durable_cards() {
        let mut chat = ChatState::new();
        let state = ElicitationState::new_for_test(Vec::new());
        let transition = chat.push_elicitation(
            Some(state),
            "elic-1".into(),
            "Pick".into(),
            "question".into(),
            None,
        );
        assert!(transition.inserted);
        assert!(chat.elicitation.is_some());
        assert!(chat.resolve_elicitation("elic-1", "responded"));
        assert!(chat.elicitation.is_none());
        chat.backfill_elicitation_outcomes(
            r#"{"answers":[{"question":"Pick","answers":["Alpha","Beta"]}]}"#,
        );
        assert!(matches!(
            &chat.messages[0],
            ChatEntry::Elicitation { outcome: Some(outcome), .. }
                if outcome == &format!("{OUTCOME_BULLET}Alpha\n{OUTCOME_BULLET}Beta")
        ));
    }

    #[test]
    fn elicitation_live_replay_duplicate_identity_is_idempotent() {
        let mut chat = ChatState::new();
        let replay = chat.push_elicitation(
            None,
            "elic-1".into(),
            "Question".into(),
            "source".into(),
            None,
        );
        assert!(replay.inserted);
        assert!(chat.elicitation.is_none());
        let duplicate = chat.push_elicitation(
            Some(ElicitationState::new_for_test(Vec::new())),
            "elic-1".into(),
            "Question".into(),
            "source".into(),
            None,
        );
        assert_eq!(duplicate, ElicitationTransition::default());
        assert_eq!(chat.messages.len(), 1);
        assert!(chat.elicitation.is_none());
    }

    #[test]
    fn streaming_content_and_thinking_keep_message_id_boundaries() {
        let mut chat = ChatState::new();
        chat.activity = ActivityState::Thinking;
        chat.append_streaming_thinking("plan", Some("one".into()), false);
        chat.append_streaming_content("answer", Some("one".into()));
        let transition = chat.append_streaming_content("second", Some("two".into()));
        assert!(transition.finalized_previous);
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(id) }]
                if content == "answer" && thinking == "plan" && id == "one"
        ));
        assert_eq!(chat.streaming_content, "second");
        assert_eq!(chat.streaming_content_message_id.as_deref(), Some("two"));
    }

    #[test]
    fn duplicate_replay_messages_and_final_assistants_are_suppressed() {
        let mut chat = ChatState::new();
        assert_eq!(
            chat.push_user_message("prompt".into(), Some("user-1".into()), true),
            UserMessageTransition::Appended
        );
        assert_eq!(
            chat.push_user_message("prompt".into(), Some("user-1".into()), true),
            UserMessageTransition::Duplicate
        );
        chat.push_assistant_message("answer".into(), None, Some("assistant-1".into()));
        chat.push_assistant_message("answer".into(), None, Some("assistant-1".into()));
        assert_eq!(chat.messages.len(), 2);
    }

    #[test]
    fn tool_boundary_preserves_streamed_thinking_before_tool_entry() {
        let mut chat = ChatState::new();
        chat.streaming_thinking = "inspect".into();
        chat.streaming_thinking_message_id = Some("assistant-1".into());
        assert!(chat.push_streaming_thinking_entry());
        chat.record_tool_call();
        chat.messages.push(ChatEntry::ToolCall {
            tool_call_id: Some("call-1".into()),
            name: "read_tool".into(),
            is_error: false,
            detail: ToolDetail::None,
        });
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::Thinking { message_id: Some(id), .. }, ChatEntry::ToolCall { tool_call_id: Some(call_id), .. }]
                if id == "assistant-1" && call_id == "call-1"
        ));
        assert_eq!(chat.session_stats.total_tool_calls, 1);
    }

    #[test]
    fn optimistic_prompt_reconciliation_is_submission_ordered() {
        let mut chat = ChatState::new();
        let first = chat.push_pending_prompt("repeat".into());
        let second = chat.push_pending_prompt("repeat".into());
        assert_eq!(
            chat.push_user_message("repeat".into(), Some("server-1".into()), false),
            UserMessageTransition::Reconciled
        );
        assert!(matches!(
            &chat.messages[0],
            ChatEntry::User { message_id: Some(id), .. } if id == "server-1"
        ));
        assert!(matches!(
            &chat.messages[1],
            ChatEntry::User { message_id: Some(id), .. } if id == &second
        ));
        assert_ne!(first, second);
    }

    #[test]
    fn reducer_turn_start_preserves_live_replay_timing_and_exact_coordination() {
        let mut live = ChatState::new();
        live.streaming_content = "discarded".into();
        live.streaming_thinking = "discarded thinking".into();
        live.arm_cancel_confirm();
        let outcome = live.reduce(ChatAction::TurnStarted { is_replay: false });
        assert_eq!(outcome.transition, ChatTransition::TurnStarted);
        assert_eq!(
            outcome.coordination,
            vec![
                ChatCoordination::InvalidateContentCache,
                ChatCoordination::InvalidateThinkingCache,
                ChatCoordination::Status {
                    level: LogLevel::Debug,
                    target: "activity",
                    message: "thinking...".into(),
                },
            ]
        );
        assert!(outcome.effects.is_empty());
        assert_eq!(live.activity, ActivityState::Thinking);
        assert!(live.streaming_content.is_empty());
        assert!(live.streaming_thinking.is_empty());
        assert!(live.pending_cancel_confirm_until.is_none());
        let opened = live.session_stats.open_llm_request_instant;
        assert!(opened.is_some());
        live.reduce(ChatAction::TurnStarted { is_replay: false });
        assert_eq!(live.session_stats.open_llm_request_instant, opened);

        let mut replay = ChatState::new();
        replay.reduce(ChatAction::TurnStarted { is_replay: true });
        assert_eq!(replay.activity, ActivityState::Thinking);
        assert!(replay.session_stats.open_llm_request_instant.is_none());
    }

    #[test]
    fn reducer_user_message_reports_noops_and_reconciles_in_submission_order() {
        let mut chat = ChatState::new();
        let ignored = chat.reduce(ChatAction::UserMessage {
            text: String::new(),
            message_id: Some("empty".into()),
            is_replay: false,
        });
        assert_eq!(
            ignored,
            ChatOutcome {
                transition: ChatTransition::UserMessage(UserMessageTransition::Ignored),
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );

        let first = chat.push_pending_prompt("  repeat\n".into());
        let second = chat.push_pending_prompt("repeat".into());
        let reconciled = chat.reduce(ChatAction::UserMessage {
            text: "repeat".into(),
            message_id: Some("server-1".into()),
            is_replay: false,
        });
        assert_eq!(
            reconciled.transition,
            ChatTransition::UserMessage(UserMessageTransition::Reconciled)
        );
        assert_eq!(
            reconciled.coordination,
            vec![ChatCoordination::InvalidateCardCache]
        );
        assert!(matches!(
            &chat.messages[0],
            ChatEntry::User { message_id: Some(id), .. } if id == "server-1"
        ));
        assert!(matches!(
            &chat.messages[1],
            ChatEntry::User { message_id: Some(id), .. } if id == &second
        ));
        assert_ne!(first, second);
        assert_eq!(chat.undoable_turns[0].message_id, "server-1");

        let duplicate = chat.reduce(ChatAction::UserMessage {
            text: "repeat".into(),
            message_id: Some("server-1".into()),
            is_replay: false,
        });
        assert_eq!(
            duplicate.transition,
            ChatTransition::UserMessage(UserMessageTransition::Duplicate)
        );
        assert!(duplicate.coordination.is_empty());

        chat.undo_state = Some(UndoState {
            stack: Vec::new(),
            frontier_message_id: None,
        });
        chat.suppress_turn_output = true;
        let replay = chat.reduce(ChatAction::UserMessage {
            text: "replayed".into(),
            message_id: Some("server-2".into()),
            is_replay: true,
        });
        assert_eq!(
            replay.transition,
            ChatTransition::UserMessage(UserMessageTransition::Appended)
        );
        assert!(replay.coordination.is_empty());
    }

    #[test]
    fn reducer_stream_deltas_preserve_boundaries_and_replay_suppression() {
        let mut chat = ChatState::new();
        chat.activity = ActivityState::Thinking;
        let thinking = chat.reduce(ChatAction::AssistantThinkingDelta {
            content: "plan".into(),
            message_id: Some("one".into()),
            is_replay: true,
        });
        assert_eq!(
            thinking.transition,
            ChatTransition::AssistantThinkingDelta(StreamingDeltaTransition::default())
        );
        assert!(thinking.coordination.is_empty());

        let duplicate = chat.reduce(ChatAction::AssistantThinkingDelta {
            content: "plan".into(),
            message_id: Some("one".into()),
            is_replay: true,
        });
        assert_eq!(
            duplicate.transition,
            ChatTransition::AssistantThinkingDelta(StreamingDeltaTransition {
                finalized_previous: false,
                ignored_duplicate: true,
            })
        );
        assert!(duplicate.coordination.is_empty());
        assert_eq!(chat.streaming_thinking, "plan");

        let content = chat.reduce(ChatAction::AssistantContentDelta {
            content: "answer".into(),
            message_id: Some("one".into()),
        });
        assert_eq!(
            content.transition,
            ChatTransition::AssistantContentDelta(StreamingDeltaTransition::default())
        );
        assert_eq!(chat.activity, ActivityState::Streaming);

        let boundary = chat.reduce(ChatAction::AssistantContentDelta {
            content: "second".into(),
            message_id: Some("two".into()),
        });
        assert_eq!(
            boundary.transition,
            ChatTransition::AssistantContentDelta(StreamingDeltaTransition {
                finalized_previous: true,
                ignored_duplicate: false,
            })
        );
        assert_eq!(
            boundary.coordination,
            vec![
                ChatCoordination::InvalidateContentCache,
                ChatCoordination::InvalidateThinkingCache,
                ChatCoordination::InvalidateCardCache,
            ]
        );
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(id) }]
                if content == "answer" && thinking == "plan" && id == "one"
        ));
        assert_eq!(chat.streaming_content, "second");
        assert_eq!(chat.streaming_content_message_id.as_deref(), Some("two"));
    }

    #[test]
    fn reducer_final_assistant_merges_replaces_and_deduplicates_exactly() {
        let expected_base_coordination = vec![
            ChatCoordination::InvalidateContentCache,
            ChatCoordination::InvalidateThinkingCache,
        ];
        let mut chat = ChatState::new();
        chat.streaming_content = "partial".into();
        chat.streaming_content_message_id = Some("assistant-1".into());
        chat.streaming_thinking = "streamed thinking".into();
        chat.streaming_thinking_message_id = Some("assistant-1".into());
        let appended = chat.reduce(ChatAction::AssistantMessage {
            content: "final".into(),
            thinking: None,
            message_id: Some("assistant-1".into()),
        });
        assert_eq!(
            appended.transition,
            ChatTransition::AssistantMessage(AssistantMessageTransition::Appended)
        );
        assert_eq!(appended.coordination, expected_base_coordination);
        assert!(chat.streaming_content.is_empty());
        assert!(chat.streaming_thinking.is_empty());
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(id) }]
                if content == "final" && thinking == "streamed thinking" && id == "assistant-1"
        ));

        let duplicate = chat.reduce(ChatAction::AssistantMessage {
            content: "duplicate".into(),
            thinking: Some("duplicate thinking".into()),
            message_id: Some("assistant-1".into()),
        });
        assert_eq!(
            duplicate.transition,
            ChatTransition::AssistantMessage(AssistantMessageTransition::Duplicate)
        );
        assert_eq!(duplicate.coordination, expected_base_coordination);
        assert_eq!(chat.messages.len(), 1);

        chat.messages.push(ChatEntry::Thinking {
            content: "old thinking".into(),
            message_id: Some("assistant-2".into()),
        });
        let replaced = chat.reduce(ChatAction::AssistantMessage {
            content: "replacement".into(),
            thinking: Some("explicit thinking".into()),
            message_id: Some("assistant-2".into()),
        });
        assert_eq!(
            replaced.transition,
            ChatTransition::AssistantMessage(AssistantMessageTransition::ReplacedThinking)
        );
        assert_eq!(
            replaced.coordination,
            vec![
                ChatCoordination::InvalidateContentCache,
                ChatCoordination::InvalidateThinkingCache,
                ChatCoordination::InvalidateCardCache,
            ]
        );
        assert!(matches!(
            &chat.messages[1],
            ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(id) }
                if content == "replacement" && thinking == "explicit thinking" && id == "assistant-2"
        ));
    }

    #[test]
    fn reducer_cancel_and_finish_discard_streams_close_live_timing_and_report_status() {
        let mut cancelled = ChatState::new();
        cancelled.begin_llm_request_span(None);
        cancelled.activity = ActivityState::Streaming;
        cancelled.streaming_content = "discard content".into();
        cancelled.streaming_thinking = "discard thinking".into();
        let outcome = cancelled.reduce(ChatAction::Cancelled { is_replay: false });
        assert_eq!(outcome.transition, ChatTransition::Cancelled);
        assert_eq!(
            outcome.coordination,
            vec![
                ChatCoordination::InvalidateContentCache,
                ChatCoordination::InvalidateThinkingCache,
                ChatCoordination::Status {
                    level: LogLevel::Warn,
                    target: "activity",
                    message: "cancelled".into(),
                },
            ]
        );
        assert_eq!(cancelled.activity, ActivityState::Idle);
        assert!(cancelled.session_stats.open_llm_request_instant.is_none());
        assert!(cancelled.streaming_content.is_empty());
        assert!(cancelled.streaming_thinking.is_empty());
        assert!(cancelled.messages.is_empty());

        let mut finished = ChatState::new();
        finished.begin_llm_request_span(None);
        finished.streaming_content = "discard content".into();
        finished.streaming_thinking = "discard thinking".into();
        let outcome = finished.reduce(ChatAction::Finished {
            finish_reason: "EndTurn".into(),
            is_replay: false,
        });
        assert_eq!(outcome.transition, ChatTransition::Finished);
        assert_eq!(
            outcome.coordination,
            vec![
                ChatCoordination::InvalidateContentCache,
                ChatCoordination::InvalidateThinkingCache,
                ChatCoordination::Status {
                    level: LogLevel::Debug,
                    target: "activity",
                    message: "finished: EndTurn".into(),
                },
            ]
        );
        assert_eq!(finished.activity, ActivityState::Idle);
        assert!(finished.session_stats.open_llm_request_instant.is_none());
        assert!(finished.streaming_content.is_empty());
        assert!(finished.streaming_thinking.is_empty());
        assert!(finished.messages.is_empty());

        let mut replay = ChatState::new();
        replay.begin_llm_request_span(None);
        let opened = replay.session_stats.open_llm_request_instant;
        replay.reduce(ChatAction::Finished {
            finish_reason: "Replay".into(),
            is_replay: true,
        });
        assert_eq!(replay.session_stats.open_llm_request_instant, opened);
    }

    #[test]
    fn elicitation_editor_reuses_composer_visual_layout() {
        let mut ui = ElicitationUiState {
            custom_cursor: 4,
            custom_line_width: 4,
            ..Default::default()
        };
        ui.custom_move_visual("abcdef", -1);
        assert_eq!(ui.custom_cursor, 2);
        ui.custom_move_visual("abcdef", 1);
        assert_eq!(ui.custom_cursor, 4);
    }
}
