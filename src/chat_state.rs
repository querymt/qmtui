use std::time::{Duration, Instant};

use crate::application::Effect;
use crate::diagnostics::LogLevel;
use crate::domain::activity::{ActivityState, SessionOp, SessionStatsLite};
use crate::domain::chat::{ChatEntry, ElicitationResponseOutcome};
use crate::domain::elicitation::ElicitationState;
use crate::domain::session::{
    ForkBoundaryKind, ForkResult, ForkTurnItem, RedoResult, UndoFrame, UndoFrameStatus, UndoResult,
    UndoStackSnapshot, UndoState, UndoableTurn,
};
use crate::domain::tool::ToolDetail;
use crate::input_layout::build_input_visual_layout;
use crate::tool_detail;

const CANCEL_CONFIRM_TIMEOUT: Duration = Duration::from_millis(1000);

#[derive(Debug, Clone, Default)]
pub(crate) struct ElicitationUiState {
    pub(crate) field_cursor: usize,
    pub(crate) option_cursor: usize,
    pub(crate) text_cursor: usize,
    pub(crate) custom_active: bool,
    pub(crate) custom_cursor: usize,
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

    pub(crate) fn custom_move_visual(&mut self, input: &str, line_width: usize, delta: i32) {
        let layout = build_input_visual_layout(input, self.custom_cursor, line_width, 2);
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ElicitationAction {
    Requested {
        elicitation_id: String,
        message: String,
        source: String,
        requested_schema: serde_json::Value,
        allow_custom: bool,
        is_replay: bool,
    },
    ResponseAcknowledged {
        elicitation_id: String,
        outcome: ElicitationResponseOutcome,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElicitationTransition {
    InsertedSupported {
        is_replay: bool,
        finalized_streaming: bool,
    },
    InsertedUnsupported {
        is_replay: bool,
        finalized_streaming: bool,
    },
    Duplicate,
    ResolvedActive,
    ResolvedStale,
    UnknownAcknowledgement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AssistantMessageTransition {
    Ignored,
    Duplicate,
    ReplacedThinking,
    Appended,
}

#[derive(Debug, Clone)]
pub(crate) enum ChatToolAction {
    PrepareToolStart {
        name: String,
    },
    InsertOrReconcileToolStart {
        tool_call_id: Option<String>,
        name: String,
        detail: ToolDetail,
    },
    ToolCallEnd {
        tool_call_id: Option<String>,
        name: String,
        is_error: bool,
        result: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ChatUsageAction {
    UsageUpdated {
        used: u64,
        size: u64,
        cost_usd: Option<f64>,
    },
    TimingUpdated {
        duration_secs: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatUsageTransition {
    UsageUpdated,
    TimingUpdated,
    NoOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolStartPreparationTransition {
    Suppressed,
    Prepared { finalized_streaming: bool },
    QuestionOnly { finalized_streaming: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolStartInsertionTransition {
    Reconciled,
    Inserted { moved_thinking: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolEndTransition {
    Updated {
        detail_updated: bool,
        marked_failed: bool,
    },
    FallbackInserted,
    NoOp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolResultDetailState {
    Shell(Option<crate::domain::tool::ShellOutput>),
    ReadTool(Option<u64>, Option<u64>),
    Edit(Option<usize>),
    MultiEdit(Vec<Option<usize>>),
    Unchanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatToolTransition {
    StartPrepared(ToolStartPreparationTransition),
    StartInserted(ToolStartInsertionTransition),
    Ended(ToolEndTransition),
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
    AcpError {
        message: String,
    },
    BackendPromptFailed {
        local_id: String,
        message: String,
    },
    RuntimePromptDispatchFailed {
        local_id: String,
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
    AcpError {
        error_inserted: bool,
    },
    BackendPromptFailed {
        prompt_rolled_back: bool,
        error_inserted: bool,
    },
    RuntimePromptDispatchFailed {
        prompt_rolled_back: bool,
    },
    Cancelled,
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HistoryAction {
    ReplaceStack(UndoStackSnapshot),
    UndoCompleted(UndoResult),
    RedoCompleted(RedoResult),
    ForkCompleted(ForkResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HistoryTransition {
    StackReplaced,
    UndoApplied,
    UndoRejected,
    RedoApplied,
    RedoRejected,
    ForkLoaded,
    ForkMissingSessionId,
    ForkFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HistoryCoordination {
    Status {
        level: LogLevel,
        target: &'static str,
        message: String,
    },
    ClosePopup,
    ReloadActiveSession,
    LoadForkedSession {
        session_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChatCoordination {
    Log {
        level: LogLevel,
        target: &'static str,
        message: String,
    },
    Status {
        level: LogLevel,
        target: &'static str,
        message: String,
    },
    RefreshTransientStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatOutcome {
    pub(crate) transition: ChatTransition,
    pub(crate) coordination: Vec<ChatCoordination>,
    pub(crate) effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatToolOutcome {
    pub(crate) transition: ChatToolTransition,
    pub(crate) coordination: Vec<ChatCoordination>,
    pub(crate) effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChatUsageOutcome {
    pub(crate) transition: ChatUsageTransition,
    pub(crate) coordination: Vec<ChatCoordination>,
    pub(crate) effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElicitationOutcome {
    pub(crate) transition: ElicitationTransition,
    pub(crate) coordination: Vec<ChatCoordination>,
    pub(crate) effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HistoryOutcome {
    pub(crate) transition: HistoryTransition,
    pub(crate) coordination: Vec<HistoryCoordination>,
    pub(crate) effects: Vec<Effect>,
}

pub(crate) struct ChatState {
    pub(crate) messages: Vec<ChatEntry>,
    pub(crate) pending_prompt_seq: u64,
    pub(crate) fork_filter: String,
    pub(crate) fork_cursor: usize,
    pub(crate) pending_fork_message_id: Option<String>,
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
                    vec![ChatCoordination::Status {
                        level: LogLevel::Debug,
                        target: "activity",
                        message: "thinking...".into(),
                    }],
                )
            }
            ChatAction::UserMessage {
                text,
                message_id,
                is_replay,
            } => {
                let transition = self.push_user_message(text, message_id, is_replay);
                (ChatTransition::UserMessage(transition), Vec::new())
            }
            ChatAction::AssistantContentDelta {
                content,
                message_id,
            } => {
                let transition = self.append_streaming_content(&content, message_id);
                (
                    ChatTransition::AssistantContentDelta(transition),
                    Vec::new(),
                )
            }
            ChatAction::AssistantThinkingDelta {
                content,
                message_id,
                is_replay,
            } => {
                let transition = self.append_streaming_thinking(&content, message_id, is_replay);
                (
                    ChatTransition::AssistantThinkingDelta(transition),
                    Vec::new(),
                )
            }
            ChatAction::AssistantMessage {
                content,
                thinking,
                message_id,
            } => {
                let transition = self.push_assistant_message(content, thinking, message_id);
                (ChatTransition::AssistantMessage(transition), Vec::new())
            }
            ChatAction::AcpError { message } => {
                self.end_llm_request_span(None);
                let error_inserted = self.push_error(&message);
                (
                    ChatTransition::AcpError { error_inserted },
                    vec![ChatCoordination::Status {
                        level: LogLevel::Error,
                        target: "acp",
                        message: format!("error: {message}"),
                    }],
                )
            }
            ChatAction::BackendPromptFailed { local_id, message } => {
                self.end_llm_request_span(None);
                let prompt_rolled_back = self.rollback_pending_prompt(&local_id);
                let error_inserted = self.push_error(&message);
                (
                    ChatTransition::BackendPromptFailed {
                        prompt_rolled_back,
                        error_inserted,
                    },
                    vec![ChatCoordination::Status {
                        level: LogLevel::Error,
                        target: "acp",
                        message: format!("error: {message}"),
                    }],
                )
            }
            ChatAction::RuntimePromptDispatchFailed { local_id } => {
                let prompt_rolled_back = self.rollback_pending_prompt(&local_id);
                (
                    ChatTransition::RuntimePromptDispatchFailed { prompt_rolled_back },
                    Vec::new(),
                )
            }
            ChatAction::Cancelled { is_replay } => {
                self.cancel_turn(is_replay);
                (
                    ChatTransition::Cancelled,
                    vec![ChatCoordination::Status {
                        level: LogLevel::Warn,
                        target: "activity",
                        message: "cancelled".into(),
                    }],
                )
            }
            ChatAction::Finished {
                finish_reason,
                is_replay,
            } => {
                self.finish_turn(is_replay);
                (
                    ChatTransition::Finished,
                    vec![ChatCoordination::Status {
                        level: LogLevel::Debug,
                        target: "activity",
                        message: format!("finished: {finish_reason}"),
                    }],
                )
            }
        };
        ChatOutcome {
            transition,
            coordination,
            effects: Vec::new(),
        }
    }

    pub(crate) fn reduce_history(&mut self, action: HistoryAction) -> HistoryOutcome {
        let (transition, coordination) = match action {
            HistoryAction::ReplaceStack(stack) => {
                self.undo_state = self.build_undo_state_from_server_stack(&stack, None, None);
                (HistoryTransition::StackReplaced, Vec::new())
            }
            HistoryAction::UndoCompleted(result) => {
                self.activity = ActivityState::Idle;
                match result {
                    UndoResult::Applied {
                        target_message_id,
                        reverted_files,
                        message: _,
                        stack,
                    } => {
                        let preferred =
                            target_message_id.or_else(|| stack.message_ids.last().cloned());
                        self.undo_state = self.build_undo_state_from_server_stack(
                            &stack,
                            preferred.as_deref(),
                            Some(&reverted_files),
                        );
                        self.recent_prompt_text = None;
                        self.streaming_content.clear();
                        self.streaming_content_message_id = None;
                        (
                            HistoryTransition::UndoApplied,
                            vec![
                                HistoryCoordination::Status {
                                    level: LogLevel::Info,
                                    target: "session",
                                    message: "undone - reloading session".into(),
                                },
                                HistoryCoordination::ReloadActiveSession,
                            ],
                        )
                    }
                    UndoResult::Rejected {
                        target_message_id,
                        message,
                        stack,
                    } => {
                        let preferred =
                            target_message_id.or_else(|| stack.message_ids.last().cloned());
                        self.undo_state = self.build_undo_state_from_server_stack(
                            &stack,
                            preferred.as_deref(),
                            None,
                        );
                        (
                            HistoryTransition::UndoRejected,
                            vec![HistoryCoordination::Status {
                                level: LogLevel::Warn,
                                target: "session",
                                message: message.unwrap_or_else(|| "undo failed".into()),
                            }],
                        )
                    }
                }
            }
            HistoryAction::RedoCompleted(result) => {
                self.activity = ActivityState::Idle;
                match result {
                    RedoResult::Applied { message: _, stack } => {
                        self.undo_state =
                            self.build_undo_state_from_server_stack(&stack, None, None);
                        (
                            HistoryTransition::RedoApplied,
                            vec![
                                HistoryCoordination::Status {
                                    level: LogLevel::Info,
                                    target: "session",
                                    message: "redone - reloading session".into(),
                                },
                                HistoryCoordination::ReloadActiveSession,
                            ],
                        )
                    }
                    RedoResult::Rejected { message, stack } => {
                        self.undo_state =
                            self.build_undo_state_from_server_stack(&stack, None, None);
                        (
                            HistoryTransition::RedoRejected,
                            vec![HistoryCoordination::Status {
                                level: LogLevel::Warn,
                                target: "session",
                                message: message.unwrap_or_else(|| "redo failed".into()),
                            }],
                        )
                    }
                }
            }
            HistoryAction::ForkCompleted(result) => {
                self.pending_fork_message_id = None;
                match result {
                    ForkResult::Succeeded {
                        source_session_id: _,
                        forked_session_id: Some(session_id),
                        message: _,
                    } => (
                        HistoryTransition::ForkLoaded,
                        vec![
                            HistoryCoordination::ClosePopup,
                            HistoryCoordination::Status {
                                level: LogLevel::Info,
                                target: "fork",
                                message: "forked - loading session".into(),
                            },
                            HistoryCoordination::LoadForkedSession { session_id },
                        ],
                    ),
                    ForkResult::Succeeded {
                        source_session_id: _,
                        forked_session_id: None,
                        message,
                    } => (
                        HistoryTransition::ForkMissingSessionId,
                        vec![HistoryCoordination::Status {
                            level: LogLevel::Warn,
                            target: "fork",
                            message: message
                                .unwrap_or_else(|| "fork succeeded without session id".into()),
                        }],
                    ),
                    ForkResult::Failed {
                        source_session_id: _,
                        message,
                    } => (
                        HistoryTransition::ForkFailed,
                        vec![HistoryCoordination::Status {
                            level: LogLevel::Warn,
                            target: "fork",
                            message: message.unwrap_or_else(|| "fork failed".into()),
                        }],
                    ),
                }
            }
        };
        HistoryOutcome {
            transition,
            coordination,
            effects: Vec::new(),
        }
    }

    pub(crate) fn reduce_tool(&mut self, action: ChatToolAction) -> ChatToolOutcome {
        let (transition, coordination) = match action {
            ChatToolAction::PrepareToolStart { name } => {
                if self.suppress_turn_output {
                    return ChatToolOutcome {
                        transition: ChatToolTransition::StartPrepared(
                            ToolStartPreparationTransition::Suppressed,
                        ),
                        coordination: Vec::new(),
                        effects: Vec::new(),
                    };
                }

                self.activity = ActivityState::RunningTool { name: name.clone() };
                let finalized_streaming = self.finalize_streaming_segment();
                let coordination = vec![ChatCoordination::Status {
                    level: LogLevel::Debug,
                    target: "tool",
                    message: format!("tool: {name}"),
                }];
                let preparation = if name == "question" {
                    ToolStartPreparationTransition::QuestionOnly {
                        finalized_streaming,
                    }
                } else {
                    ToolStartPreparationTransition::Prepared {
                        finalized_streaming,
                    }
                };
                (ChatToolTransition::StartPrepared(preparation), coordination)
            }
            ChatToolAction::InsertOrReconcileToolStart {
                tool_call_id,
                name,
                detail,
            } => {
                if tool_detail::reconcile_tool_call_start(
                    &mut self.messages,
                    tool_call_id.as_deref(),
                    &name,
                    detail.clone(),
                ) {
                    self.clear_streaming_thinking();
                    (
                        ChatToolTransition::StartInserted(ToolStartInsertionTransition::Reconciled),
                        Vec::new(),
                    )
                } else {
                    self.record_tool_call();
                    let moved_thinking = self.push_streaming_thinking_entry();
                    self.push_tool_call(tool_call_id, name, false, detail);
                    (
                        ChatToolTransition::StartInserted(ToolStartInsertionTransition::Inserted {
                            moved_thinking,
                        }),
                        Vec::new(),
                    )
                }
            }
            ChatToolAction::ToolCallEnd {
                tool_call_id,
                name,
                is_error,
                result,
            } => {
                let detail_before = self
                    .tool_call_detail(tool_call_id.as_deref())
                    .map(tool_result_detail_state);
                let detail_updated = result.as_deref().is_some_and(|result| {
                    let found = tool_detail::update_tool_detail(
                        &mut self.messages,
                        tool_call_id.as_deref(),
                        result,
                    );
                    found
                        && detail_before.as_ref().is_some_and(|before| {
                            self.tool_call_detail(tool_call_id.as_deref())
                                .is_some_and(|after| before != &tool_result_detail_state(after))
                        })
                });
                let failure_before = is_error
                    .then(|| self.tool_call_error_state(tool_call_id.as_deref(), &name))
                    .flatten();
                let marked_failed = if failure_before.is_some() {
                    self.mark_tool_call_failed(tool_call_id.as_deref(), &name);
                    failure_before == Some(false)
                } else {
                    false
                };
                let transition = if is_error && failure_before.is_none() {
                    self.push_tool_call(
                        tool_call_id,
                        format!("{name} (failed)"),
                        true,
                        result
                            .map(|result| ToolDetail::Generic {
                                input: None,
                                result: Some(result),
                            })
                            .unwrap_or(ToolDetail::None),
                    );
                    ToolEndTransition::FallbackInserted
                } else if detail_updated || marked_failed {
                    ToolEndTransition::Updated {
                        detail_updated,
                        marked_failed,
                    }
                } else {
                    ToolEndTransition::NoOp
                };
                (ChatToolTransition::Ended(transition), Vec::new())
            }
        };
        ChatToolOutcome {
            transition,
            coordination,
            effects: Vec::new(),
        }
    }

    pub(crate) fn reduce_usage(&mut self, action: ChatUsageAction) -> ChatUsageOutcome {
        let (transition, coordination) = match action {
            ChatUsageAction::UsageUpdated {
                used,
                size,
                cost_usd,
            } => {
                self.apply_usage(used, size, cost_usd);
                let percentage = if used > 0 && size > 0 {
                    format!(" ({}%)", (used as f64 / size as f64 * 100.0) as u32)
                } else {
                    String::new()
                };
                let cost = cost_usd
                    .map(|amount| format!(", cost ${amount:.4}"))
                    .unwrap_or_default();
                (
                    ChatUsageTransition::UsageUpdated,
                    vec![ChatCoordination::Log {
                        level: LogLevel::Info,
                        target: "usage",
                        message: format!("usage: context {used}/{size} tokens{percentage}{cost}"),
                    }],
                )
            }
            ChatUsageAction::TimingUpdated { duration_secs } if duration_secs > 0 => {
                self.add_active_llm_duration(Duration::from_secs(duration_secs));
                (
                    ChatUsageTransition::TimingUpdated,
                    vec![ChatCoordination::Log {
                        level: LogLevel::Info,
                        target: "usage",
                        message: format!("usage: active time {duration_secs}s"),
                    }],
                )
            }
            ChatUsageAction::TimingUpdated { .. } => (ChatUsageTransition::NoOp, Vec::new()),
        };
        ChatUsageOutcome {
            transition,
            coordination,
            effects: Vec::new(),
        }
    }

    pub(crate) fn reduce_elicitation(&mut self, action: ElicitationAction) -> ElicitationOutcome {
        let (transition, coordination) = match action {
            ElicitationAction::Requested {
                elicitation_id,
                message,
                source,
                requested_schema,
                allow_custom,
                is_replay,
            } => {
                if self.messages.iter().any(|entry| {
                    matches!(
                        entry,
                        ChatEntry::Elicitation {
                            elicitation_id: existing_id,
                            ..
                        } if existing_id == &elicitation_id
                    )
                }) {
                    return ElicitationOutcome {
                        transition: ElicitationTransition::Duplicate,
                        coordination: Vec::new(),
                        effects: Vec::new(),
                    };
                }

                let fields = ElicitationState::parse_schema(&requested_schema);
                let supported = !fields.is_empty();
                let finalized_streaming = self.finalize_streaming_segment();
                if supported && !is_replay {
                    self.elicitation = Some(ElicitationState {
                        elicitation_id: elicitation_id.clone(),
                        message: message.clone(),
                        source: source.clone(),
                        fields,
                        selected: std::collections::HashMap::new(),
                        text_input: String::new(),
                        custom_input: String::new(),
                        allow_custom,
                    });
                    self.elicitation_ui = Some(ElicitationUiState::default());
                }
                self.messages.push(ChatEntry::Elicitation {
                    elicitation_id,
                    message,
                    source,
                    outcome: (!supported).then_some(ElicitationResponseOutcome::UnsupportedSchema),
                });

                let (transition, status) = if supported {
                    (
                        ElicitationTransition::InsertedSupported {
                            is_replay,
                            finalized_streaming,
                        },
                        ChatCoordination::Status {
                            level: LogLevel::Info,
                            target: "elicitation",
                            message: "question - answer in the panel above input".into(),
                        },
                    )
                } else {
                    (
                        ElicitationTransition::InsertedUnsupported {
                            is_replay,
                            finalized_streaming,
                        },
                        ChatCoordination::Status {
                            level: LogLevel::Warn,
                            target: "elicitation",
                            message: "question skipped - unsupported schema".into(),
                        },
                    )
                };
                (transition, vec![status])
            }
            ElicitationAction::ResponseAcknowledged {
                elicitation_id,
                outcome,
            } => {
                let is_active = self
                    .elicitation
                    .as_ref()
                    .is_some_and(|state| state.elicitation_id == elicitation_id);
                let Some(card_outcome) = self.messages.iter_mut().find_map(|entry| match entry {
                    ChatEntry::Elicitation {
                        elicitation_id: existing_id,
                        outcome,
                        ..
                    } if existing_id == &elicitation_id => Some(outcome),
                    _ => None,
                }) else {
                    return ElicitationOutcome {
                        transition: ElicitationTransition::UnknownAcknowledgement,
                        coordination: Vec::new(),
                        effects: Vec::new(),
                    };
                };
                *card_outcome = Some(outcome);

                if is_active {
                    self.elicitation = None;
                    self.elicitation_ui = None;
                    (
                        ElicitationTransition::ResolvedActive,
                        vec![ChatCoordination::RefreshTransientStatus],
                    )
                } else {
                    (ElicitationTransition::ResolvedStale, Vec::new())
                }
            }
        };
        ElicitationOutcome {
            transition,
            coordination,
            effects: Vec::new(),
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            messages: Vec::new(),
            pending_prompt_seq: 0,
            fork_filter: String::new(),
            fork_cursor: 0,
            pending_fork_message_id: None,
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

    fn tool_call_detail(&self, tool_call_id: Option<&str>) -> Option<&ToolDetail> {
        let tool_call_id = tool_call_id?;
        self.messages.iter().rev().find_map(|entry| match entry {
            ChatEntry::ToolCall {
                tool_call_id: Some(existing),
                detail,
                ..
            } if existing == tool_call_id => Some(detail),
            _ => None,
        })
    }

    fn tool_call_error_state(&self, tool_call_id: Option<&str>, tool_name: &str) -> Option<bool> {
        let tool_call_id = tool_call_id?;
        let fallback_name = format!("{tool_name} (failed)");
        self.messages.iter().rev().find_map(|entry| match entry {
            ChatEntry::ToolCall {
                tool_call_id: Some(existing),
                name,
                is_error,
                ..
            } if existing == tool_call_id && (name == tool_name || name == &fallback_name) => {
                Some(*is_error)
            }
            _ => None,
        })
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
}

fn tool_result_detail_state(detail: &ToolDetail) -> ToolResultDetailState {
    match detail {
        ToolDetail::Shell { output, .. } => ToolResultDetailState::Shell(output.clone()),
        ToolDetail::ReadTool {
            start_line,
            end_line,
            ..
        } => ToolResultDetailState::ReadTool(*start_line, *end_line),
        ToolDetail::Edit { start_line, .. } => ToolResultDetailState::Edit(*start_line),
        ToolDetail::MultiEdit { sections, .. } => ToolResultDetailState::MultiEdit(
            sections.iter().map(|section| section.start_line).collect(),
        ),
        _ => ToolResultDetailState::Unchanged,
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
    fn usage_reducer_replaces_metrics_and_emits_exact_log_intent() {
        let mut chat = ChatState::new();
        chat.apply_usage(1024, 4096, Some(1.25));

        let outcome = chat.reduce_usage(ChatUsageAction::UsageUpdated {
            used: 2048,
            size: 8192,
            cost_usd: Some(0.0123),
        });

        assert_eq!(chat.session_stats.latest_context_tokens, Some(2048));
        assert_eq!(chat.context_limit, 8192);
        assert_eq!(chat.cumulative_cost, Some(0.0123));
        assert_eq!(
            outcome,
            ChatUsageOutcome {
                transition: ChatUsageTransition::UsageUpdated,
                coordination: vec![ChatCoordination::Log {
                    level: LogLevel::Info,
                    target: "usage",
                    message: "usage: context 2048/8192 tokens (25%), cost $0.0123".into(),
                }],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn usage_reducer_preserves_absent_values_and_omits_zero_size_percentage() {
        let mut chat = ChatState::new();
        chat.apply_usage(128, 8192, Some(1.25));

        let outcome = chat.reduce_usage(ChatUsageAction::UsageUpdated {
            used: 512,
            size: 0,
            cost_usd: None,
        });

        assert_eq!(chat.session_stats.latest_context_tokens, Some(512));
        assert_eq!(chat.context_limit, 8192);
        assert_eq!(chat.cumulative_cost, Some(1.25));
        assert_eq!(
            outcome,
            ChatUsageOutcome {
                transition: ChatUsageTransition::UsageUpdated,
                coordination: vec![ChatCoordination::Log {
                    level: LogLevel::Info,
                    target: "usage",
                    message: "usage: context 512/0 tokens".into(),
                }],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn timing_reducer_adds_repeated_durations_and_types_zero_as_noop() {
        let mut chat = ChatState::new();
        chat.add_active_llm_duration(Duration::from_secs(2));

        let first = chat.reduce_usage(ChatUsageAction::TimingUpdated { duration_secs: 3 });
        assert_eq!(
            first,
            ChatUsageOutcome {
                transition: ChatUsageTransition::TimingUpdated,
                coordination: vec![ChatCoordination::Log {
                    level: LogLevel::Info,
                    target: "usage",
                    message: "usage: active time 3s".into(),
                }],
                effects: Vec::new(),
            }
        );
        let second = chat.reduce_usage(ChatUsageAction::TimingUpdated { duration_secs: 4 });
        assert_eq!(second.transition, ChatUsageTransition::TimingUpdated);
        assert_eq!(
            second.coordination,
            vec![ChatCoordination::Log {
                level: LogLevel::Info,
                target: "usage",
                message: "usage: active time 4s".into(),
            }]
        );
        assert!(second.effects.is_empty());
        assert_eq!(chat.llm_request_elapsed(), Some(Duration::from_secs(9)));

        let zero = chat.reduce_usage(ChatUsageAction::TimingUpdated { duration_secs: 0 });
        assert_eq!(
            zero,
            ChatUsageOutcome {
                transition: ChatUsageTransition::NoOp,
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );
        assert_eq!(chat.llm_request_elapsed(), Some(Duration::from_secs(9)));
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
    fn history_stack_replacement_rebuilds_only_undo_state() {
        let mut chat = ChatState::new();
        chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        chat.streaming_content = "content".into();
        chat.streaming_content_message_id = Some("content-id".into());
        chat.streaming_thinking = "thinking".into();
        chat.streaming_thinking_message_id = Some("thinking-id".into());
        chat.recent_prompt_text = Some("prompt".into());
        chat.pending_fork_message_id = Some("fork-target".into());
        chat.undoable_turns = vec![turn("one")];

        let outcome = chat.reduce_history(HistoryAction::ReplaceStack(UndoStackSnapshot {
            message_ids: vec!["one".into()],
        }));

        assert_eq!(
            outcome,
            HistoryOutcome {
                transition: HistoryTransition::StackReplaced,
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );
        assert_eq!(chat.activity, ActivityState::SessionOp(SessionOp::Undo));
        assert_eq!(chat.streaming_content, "content");
        assert_eq!(
            chat.streaming_content_message_id.as_deref(),
            Some("content-id")
        );
        assert_eq!(chat.streaming_thinking, "thinking");
        assert_eq!(
            chat.streaming_thinking_message_id.as_deref(),
            Some("thinking-id")
        );
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("prompt"));
        assert_eq!(chat.pending_fork_message_id.as_deref(), Some("fork-target"));
        let state = chat.undo_state.as_ref().expect("rebuilt undo state");
        assert_eq!(state.frontier_message_id.as_deref(), Some("one"));
        assert_eq!(state.stack[0].turn_id, "turn-one");
    }

    #[test]
    fn history_applied_undo_prefers_explicit_target_and_clears_only_undo_stream_scope() {
        let mut chat = ChatState::new();
        chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        chat.undoable_turns = vec![turn("one"), turn("two")];
        chat.streaming_content = "discard content".into();
        chat.streaming_content_message_id = Some("content-id".into());
        chat.streaming_thinking = "keep thinking".into();
        chat.streaming_thinking_message_id = Some("thinking-id".into());
        chat.recent_prompt_text = Some("discard prompt".into());

        let outcome = chat.reduce_history(HistoryAction::UndoCompleted(UndoResult::Applied {
            target_message_id: Some("one".into()),
            reverted_files: vec!["src/lib.rs".into()],
            message: Some("ignored success".into()),
            stack: UndoStackSnapshot {
                message_ids: vec!["one".into(), "two".into()],
            },
        }));

        assert_eq!(chat.activity, ActivityState::Idle);
        assert!(chat.streaming_content.is_empty());
        assert!(chat.streaming_content_message_id.is_none());
        assert_eq!(chat.streaming_thinking, "keep thinking");
        assert_eq!(
            chat.streaming_thinking_message_id.as_deref(),
            Some("thinking-id")
        );
        assert!(chat.recent_prompt_text.is_none());
        let state = chat.undo_state.as_ref().expect("rebuilt undo state");
        assert_eq!(state.frontier_message_id.as_deref(), Some("one"));
        assert_eq!(state.stack[0].reverted_files, ["src/lib.rs"]);
        assert!(state.stack[1].reverted_files.is_empty());
        assert_eq!(
            outcome,
            HistoryOutcome {
                transition: HistoryTransition::UndoApplied,
                coordination: vec![
                    HistoryCoordination::Status {
                        level: LogLevel::Info,
                        target: "session",
                        message: "undone - reloading session".into(),
                    },
                    HistoryCoordination::ReloadActiveSession,
                ],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn history_applied_undo_uses_stack_tail_for_reverted_file_attribution() {
        let mut chat = ChatState::new();
        chat.undoable_turns = vec![turn("one"), turn("two")];

        chat.reduce_history(HistoryAction::UndoCompleted(UndoResult::Applied {
            target_message_id: None,
            reverted_files: vec!["src/main.rs".into()],
            message: None,
            stack: UndoStackSnapshot {
                message_ids: vec!["one".into(), "two".into()],
            },
        }));

        let state = chat.undo_state.as_ref().expect("rebuilt undo state");
        assert_eq!(state.frontier_message_id.as_deref(), Some("two"));
        assert!(state.stack[0].reverted_files.is_empty());
        assert_eq!(state.stack[1].reverted_files, ["src/main.rs"]);
    }

    #[test]
    fn history_rejected_undo_uses_fallback_frontier_and_preserves_live_state() {
        let mut chat = ChatState::new();
        chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        chat.undoable_turns = vec![turn("one"), turn("two")];
        chat.streaming_content = "keep content".into();
        chat.streaming_content_message_id = Some("content-id".into());
        chat.streaming_thinking = "keep thinking".into();
        chat.streaming_thinking_message_id = Some("thinking-id".into());
        chat.recent_prompt_text = Some("keep prompt".into());

        let outcome = chat.reduce_history(HistoryAction::UndoCompleted(UndoResult::Rejected {
            target_message_id: None,
            message: None,
            stack: UndoStackSnapshot {
                message_ids: vec!["one".into(), "two".into()],
            },
        }));

        assert_eq!(chat.activity, ActivityState::Idle);
        assert_eq!(chat.streaming_content, "keep content");
        assert_eq!(
            chat.streaming_content_message_id.as_deref(),
            Some("content-id")
        );
        assert_eq!(chat.streaming_thinking, "keep thinking");
        assert_eq!(
            chat.streaming_thinking_message_id.as_deref(),
            Some("thinking-id")
        );
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("keep prompt"));
        assert_eq!(
            chat.undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("two")
        );
        assert_eq!(
            outcome,
            HistoryOutcome {
                transition: HistoryTransition::UndoRejected,
                coordination: vec![HistoryCoordination::Status {
                    level: LogLevel::Warn,
                    target: "session",
                    message: "undo failed".into(),
                }],
                effects: Vec::new(),
            }
        );

        let outcome = chat.reduce_history(HistoryAction::UndoCompleted(UndoResult::Rejected {
            target_message_id: Some("one".into()),
            message: Some("explicit rejection".into()),
            stack: UndoStackSnapshot {
                message_ids: vec!["one".into(), "two".into()],
            },
        }));
        assert_eq!(
            chat.undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("one")
        );
        assert_eq!(
            outcome.coordination,
            vec![HistoryCoordination::Status {
                level: LogLevel::Warn,
                target: "session",
                message: "explicit rejection".into(),
            }]
        );
    }

    #[test]
    fn history_redo_results_rebuild_without_undo_cleanup_and_coordinate_exactly() {
        let mut chat = ChatState::new();
        chat.activity = ActivityState::SessionOp(SessionOp::Redo);
        chat.streaming_content = "keep content".into();
        chat.streaming_content_message_id = Some("content-id".into());
        chat.streaming_thinking = "keep thinking".into();
        chat.streaming_thinking_message_id = Some("thinking-id".into());
        chat.recent_prompt_text = Some("keep prompt".into());

        let applied = chat.reduce_history(HistoryAction::RedoCompleted(RedoResult::Applied {
            message: Some("ignored success".into()),
            stack: UndoStackSnapshot {
                message_ids: vec!["one".into()],
            },
        }));

        assert_eq!(chat.activity, ActivityState::Idle);
        assert_eq!(chat.streaming_content, "keep content");
        assert_eq!(
            chat.streaming_content_message_id.as_deref(),
            Some("content-id")
        );
        assert_eq!(chat.streaming_thinking, "keep thinking");
        assert_eq!(
            chat.streaming_thinking_message_id.as_deref(),
            Some("thinking-id")
        );
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("keep prompt"));
        assert_eq!(
            applied,
            HistoryOutcome {
                transition: HistoryTransition::RedoApplied,
                coordination: vec![
                    HistoryCoordination::Status {
                        level: LogLevel::Info,
                        target: "session",
                        message: "redone - reloading session".into(),
                    },
                    HistoryCoordination::ReloadActiveSession,
                ],
                effects: Vec::new(),
            }
        );

        chat.activity = ActivityState::SessionOp(SessionOp::Redo);
        let rejected = chat.reduce_history(HistoryAction::RedoCompleted(RedoResult::Rejected {
            message: None,
            stack: UndoStackSnapshot {
                message_ids: vec!["one".into(), "two".into()],
            },
        }));
        assert_eq!(chat.activity, ActivityState::Idle);
        assert_eq!(
            chat.undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("one")
        );
        assert_eq!(
            rejected,
            HistoryOutcome {
                transition: HistoryTransition::RedoRejected,
                coordination: vec![HistoryCoordination::Status {
                    level: LogLevel::Warn,
                    target: "session",
                    message: "redo failed".into(),
                }],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn history_fork_results_clear_pending_and_preserve_ignored_success_fields() {
        let mut chat = ChatState::new();
        chat.pending_fork_message_id = Some("fork-target".into());

        let loaded = chat.reduce_history(HistoryAction::ForkCompleted(ForkResult::Succeeded {
            source_session_id: Some("ignored-source".into()),
            forked_session_id: Some("forked".into()),
            message: Some("ignored success".into()),
        }));
        assert!(chat.pending_fork_message_id.is_none());
        assert_eq!(
            loaded,
            HistoryOutcome {
                transition: HistoryTransition::ForkLoaded,
                coordination: vec![
                    HistoryCoordination::ClosePopup,
                    HistoryCoordination::Status {
                        level: LogLevel::Info,
                        target: "fork",
                        message: "forked - loading session".into(),
                    },
                    HistoryCoordination::LoadForkedSession {
                        session_id: "forked".into(),
                    },
                ],
                effects: Vec::new(),
            }
        );

        chat.pending_fork_message_id = Some("fork-target".into());
        let missing_id = chat.reduce_history(HistoryAction::ForkCompleted(ForkResult::Succeeded {
            source_session_id: Some("ignored-source".into()),
            forked_session_id: None,
            message: None,
        }));
        assert!(chat.pending_fork_message_id.is_none());
        assert_eq!(
            missing_id,
            HistoryOutcome {
                transition: HistoryTransition::ForkMissingSessionId,
                coordination: vec![HistoryCoordination::Status {
                    level: LogLevel::Warn,
                    target: "fork",
                    message: "fork succeeded without session id".into(),
                }],
                effects: Vec::new(),
            }
        );

        chat.pending_fork_message_id = Some("fork-target".into());
        let failed = chat.reduce_history(HistoryAction::ForkCompleted(ForkResult::Failed {
            source_session_id: Some("ignored-source".into()),
            message: None,
        }));
        assert!(chat.pending_fork_message_id.is_none());
        assert_eq!(
            failed,
            HistoryOutcome {
                transition: HistoryTransition::ForkFailed,
                coordination: vec![HistoryCoordination::Status {
                    level: LogLevel::Warn,
                    target: "fork",
                    message: "fork failed".into(),
                }],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn history_rejections_preserve_supplied_diagnostic_messages() {
        let mut chat = ChatState::new();

        let undo = chat.reduce_history(HistoryAction::UndoCompleted(UndoResult::Rejected {
            target_message_id: None,
            message: Some("undo rejected by server".into()),
            stack: UndoStackSnapshot::default(),
        }));
        assert_eq!(
            undo.coordination,
            vec![HistoryCoordination::Status {
                level: LogLevel::Warn,
                target: "session",
                message: "undo rejected by server".into(),
            }]
        );

        let redo = chat.reduce_history(HistoryAction::RedoCompleted(RedoResult::Rejected {
            message: Some("redo rejected by server".into()),
            stack: UndoStackSnapshot::default(),
        }));
        assert_eq!(
            redo.coordination,
            vec![HistoryCoordination::Status {
                level: LogLevel::Warn,
                target: "session",
                message: "redo rejected by server".into(),
            }]
        );

        let missing_fork =
            chat.reduce_history(HistoryAction::ForkCompleted(ForkResult::Succeeded {
                source_session_id: None,
                forked_session_id: None,
                message: Some("fork response omitted an id".into()),
            }));
        assert_eq!(
            missing_fork.coordination,
            vec![HistoryCoordination::Status {
                level: LogLevel::Warn,
                target: "fork",
                message: "fork response omitted an id".into(),
            }]
        );

        let failed = chat.reduce_history(HistoryAction::ForkCompleted(ForkResult::Failed {
            source_session_id: None,
            message: Some("fork rejected by server".into()),
        }));
        assert_eq!(
            failed.coordination,
            vec![HistoryCoordination::Status {
                level: LogLevel::Warn,
                target: "fork",
                message: "fork rejected by server".into(),
            }]
        );
    }

    #[test]
    fn elicitation_live_replay_duplicate_identity_is_idempotent() {
        let mut chat = ChatState::new();
        let request = |is_replay| ElicitationAction::Requested {
            elicitation_id: "elic-1".into(),
            message: "Question".into(),
            source: "source".into(),
            requested_schema: serde_json::json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } }
            }),
            allow_custom: false,
            is_replay,
        };
        let replay = chat.reduce_elicitation(request(true));
        assert_eq!(
            replay.transition,
            ElicitationTransition::InsertedSupported {
                is_replay: true,
                finalized_streaming: false,
            }
        );
        assert!(chat.elicitation.is_none());
        let duplicate = chat.reduce_elicitation(request(false));
        assert_eq!(duplicate.transition, ElicitationTransition::Duplicate);
        assert!(duplicate.coordination.is_empty());
        assert_eq!(chat.messages.len(), 1);
        assert!(chat.elicitation.is_none());
    }

    #[test]
    fn elicitation_supported_live_and_replay_have_exact_state_and_coordination() {
        let request = |is_replay| ElicitationAction::Requested {
            elicitation_id: "elic-1".into(),
            message: "Choose a target".into(),
            source: "builtin:question".into(),
            requested_schema: serde_json::json!({
                "type": "object",
                "properties": { "target": { "type": "string", "enum": ["staging"] } },
                "required": ["target"]
            }),
            allow_custom: true,
            is_replay,
        };

        let mut live = ChatState::new();
        let live_outcome = live.reduce_elicitation(request(false));
        assert_eq!(
            live_outcome,
            ElicitationOutcome {
                transition: ElicitationTransition::InsertedSupported {
                    is_replay: false,
                    finalized_streaming: false,
                },
                coordination: vec![ChatCoordination::Status {
                    level: LogLevel::Info,
                    target: "elicitation",
                    message: "question - answer in the panel above input".into(),
                }],
                effects: Vec::new(),
            }
        );
        let active = live.elicitation.as_ref().expect("live active elicitation");
        assert_eq!(active.elicitation_id, "elic-1");
        assert_eq!(active.message, "Choose a target");
        assert_eq!(active.source, "builtin:question");
        assert_eq!(active.fields.len(), 1);
        assert!(active.allow_custom);
        assert!(live.elicitation_ui.is_some());
        assert!(matches!(
            live.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, message, source, outcome: None }]
                if elicitation_id == "elic-1"
                    && message == "Choose a target"
                    && source == "builtin:question"
        ));

        let mut replay = ChatState::new();
        let replay_outcome = replay.reduce_elicitation(request(true));
        assert_eq!(
            replay_outcome,
            ElicitationOutcome {
                transition: ElicitationTransition::InsertedSupported {
                    is_replay: true,
                    finalized_streaming: false,
                },
                coordination: vec![ChatCoordination::Status {
                    level: LogLevel::Info,
                    target: "elicitation",
                    message: "question - answer in the panel above input".into(),
                }],
                effects: Vec::new(),
            }
        );
        assert!(replay.elicitation.is_none());
        assert!(replay.elicitation_ui.is_none());
        assert!(matches!(
            replay.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: None, .. }]
                if elicitation_id == "elic-1"
        ));
    }

    #[test]
    fn elicitation_unsupported_live_and_replay_insert_outcome_without_active_state() {
        for is_replay in [false, true] {
            let mut chat = ChatState::new();
            let outcome = chat.reduce_elicitation(ElicitationAction::Requested {
                elicitation_id: format!("unsupported-{is_replay}"),
                message: "Upload a file".into(),
                source: "extension:file-picker".into(),
                requested_schema: serde_json::json!({ "type": "array" }),
                allow_custom: false,
                is_replay,
            });

            assert_eq!(
                outcome,
                ElicitationOutcome {
                    transition: ElicitationTransition::InsertedUnsupported {
                        is_replay,
                        finalized_streaming: false,
                    },
                    coordination: vec![ChatCoordination::Status {
                        level: LogLevel::Warn,
                        target: "elicitation",
                        message: "question skipped - unsupported schema".into(),
                    }],
                    effects: Vec::new(),
                }
            );
            assert!(chat.elicitation.is_none());
            assert!(chat.elicitation_ui.is_none());
            assert!(matches!(
                chat.messages.as_slice(),
                [ChatEntry::Elicitation {
                    outcome: Some(ElicitationResponseOutcome::UnsupportedSchema),
                    ..
                }]
            ));
        }
    }

    #[test]
    fn elicitation_request_finalizes_stream_before_card_and_distinct_live_replaces_active() {
        let request = |id: &str| ElicitationAction::Requested {
            elicitation_id: id.into(),
            message: format!("Question {id}"),
            source: "builtin:question".into(),
            requested_schema: serde_json::json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } }
            }),
            allow_custom: false,
            is_replay: false,
        };
        let mut chat = ChatState::new();
        chat.streaming_thinking = "plan".into();
        chat.streaming_thinking_message_id = Some("assistant-1".into());
        chat.streaming_content = "answer".into();
        chat.streaming_content_message_id = Some("assistant-1".into());

        let first = chat.reduce_elicitation(request("elic-1"));
        assert_eq!(
            first.transition,
            ElicitationTransition::InsertedSupported {
                is_replay: false,
                finalized_streaming: true,
            }
        );
        assert_eq!(
            first.coordination,
            vec![ChatCoordination::Status {
                level: LogLevel::Info,
                target: "elicitation",
                message: "question - answer in the panel above input".into(),
            },]
        );
        assert!(first.effects.is_empty());
        assert!(matches!(
            chat.messages.as_slice(),
            [
                ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(message_id) },
                ChatEntry::Elicitation { elicitation_id, outcome: None, .. },
            ] if content == "answer"
                && thinking == "plan"
                && message_id == "assistant-1"
                && elicitation_id == "elic-1"
        ));

        chat.streaming_content = "must remain pending".into();
        let duplicate = chat.reduce_elicitation(request("elic-1"));
        assert_eq!(duplicate.transition, ElicitationTransition::Duplicate);
        assert!(duplicate.coordination.is_empty());
        assert!(duplicate.effects.is_empty());
        assert_eq!(chat.streaming_content, "must remain pending");
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(
            chat.elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-1")
        );

        chat.streaming_content.clear();
        let second = chat.reduce_elicitation(request("elic-2"));
        assert_eq!(
            second.transition,
            ElicitationTransition::InsertedSupported {
                is_replay: false,
                finalized_streaming: false,
            }
        );
        assert_eq!(
            chat.elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-2")
        );
        assert!(chat.elicitation_ui.is_some());
        assert!(matches!(
            chat.messages.as_slice(),
            [
                ChatEntry::Assistant { .. },
                ChatEntry::Elicitation { elicitation_id: first, outcome: None, .. },
                ChatEntry::Elicitation { elicitation_id: second, outcome: None, .. },
            ] if first == "elic-1" && second == "elic-2"
        ));
    }

    #[test]
    fn elicitation_acknowledgements_type_unknown_stale_active_and_duplicate_behavior() {
        let mut chat = ChatState::new();
        for id in ["elic-old", "elic-new"] {
            chat.messages.push(ChatEntry::Elicitation {
                elicitation_id: id.into(),
                message: format!("Question {id}"),
                source: "builtin:question".into(),
                outcome: None,
            });
        }
        let mut active = ElicitationState::new_for_test(Vec::new());
        active.elicitation_id = "elic-new".into();
        chat.elicitation = Some(active);
        chat.elicitation_ui = Some(ElicitationUiState::default());

        let unknown = chat.reduce_elicitation(ElicitationAction::ResponseAcknowledged {
            elicitation_id: "missing".into(),
            outcome: ElicitationResponseOutcome::Text("accepted".into()),
        });
        assert_eq!(
            unknown,
            ElicitationOutcome {
                transition: ElicitationTransition::UnknownAcknowledgement,
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );
        assert_eq!(
            chat.elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-new")
        );

        let stale = chat.reduce_elicitation(ElicitationAction::ResponseAcknowledged {
            elicitation_id: "elic-old".into(),
            outcome: ElicitationResponseOutcome::Declined,
        });
        assert_eq!(stale.transition, ElicitationTransition::ResolvedStale);
        assert_eq!(stale.coordination, Vec::new());
        assert!(stale.effects.is_empty());
        assert_eq!(
            chat.elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-new")
        );
        assert!(chat.elicitation_ui.is_some());

        let active = chat.reduce_elicitation(ElicitationAction::ResponseAcknowledged {
            elicitation_id: "elic-new".into(),
            outcome: ElicitationResponseOutcome::Text("accepted".into()),
        });
        assert_eq!(active.transition, ElicitationTransition::ResolvedActive);
        assert_eq!(
            active.coordination,
            vec![ChatCoordination::RefreshTransientStatus,]
        );
        assert!(active.effects.is_empty());
        assert!(chat.elicitation.is_none());
        assert!(chat.elicitation_ui.is_none());

        let duplicate = chat.reduce_elicitation(ElicitationAction::ResponseAcknowledged {
            elicitation_id: "elic-new".into(),
            outcome: ElicitationResponseOutcome::Text("accepted".into()),
        });
        assert_eq!(duplicate.transition, ElicitationTransition::ResolvedStale);
        assert_eq!(duplicate.coordination, Vec::new());
        assert!(duplicate.effects.is_empty());
        assert!(matches!(
            chat.messages.as_slice(),
            [
                ChatEntry::Elicitation { elicitation_id: old, outcome: Some(old_outcome), .. },
                ChatEntry::Elicitation { elicitation_id: new, outcome: Some(new_outcome), .. },
            ] if old == "elic-old"
                && old_outcome == &ElicitationResponseOutcome::Declined
                && new == "elic-new"
                && new_outcome == &ElicitationResponseOutcome::Text("accepted".into())
        ));
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
    fn tool_reducer_suppression_is_a_typed_noop() {
        let mut chat = ChatState::new();
        chat.suppress_turn_output = true;
        chat.activity = ActivityState::Thinking;
        chat.streaming_content = "preserved".into();
        chat.streaming_content_message_id = Some("assistant-1".into());
        chat.session_stats.total_tool_calls = 4;

        let outcome = chat.reduce_tool(ChatToolAction::PrepareToolStart {
            name: "shell".into(),
        });

        assert_eq!(
            outcome,
            ChatToolOutcome {
                transition: ChatToolTransition::StartPrepared(
                    ToolStartPreparationTransition::Suppressed,
                ),
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );
        assert_eq!(chat.activity, ActivityState::Thinking);
        assert_eq!(chat.streaming_content, "preserved");
        assert_eq!(
            chat.streaming_content_message_id.as_deref(),
            Some("assistant-1")
        );
        assert!(chat.messages.is_empty());
        assert_eq!(chat.session_stats.total_tool_calls, 4);
    }

    #[test]
    fn tool_reducer_prepare_sets_status_and_finalizes_stream_in_exact_order() {
        let mut chat = ChatState::new();
        chat.streaming_content = "answer".into();
        chat.streaming_content_message_id = Some("assistant-1".into());
        chat.streaming_thinking = "plan".into();
        chat.streaming_thinking_message_id = Some("assistant-1".into());

        let outcome = chat.reduce_tool(ChatToolAction::PrepareToolStart {
            name: "shell".into(),
        });

        assert_eq!(
            outcome,
            ChatToolOutcome {
                transition: ChatToolTransition::StartPrepared(
                    ToolStartPreparationTransition::Prepared {
                        finalized_streaming: true,
                    },
                ),
                coordination: vec![ChatCoordination::Status {
                    level: LogLevel::Debug,
                    target: "tool",
                    message: "tool: shell".into(),
                },],
                effects: Vec::new(),
            }
        );
        assert_eq!(
            chat.activity,
            ActivityState::RunningTool {
                name: "shell".into()
            }
        );
        assert!(chat.streaming_content.is_empty());
        assert!(chat.streaming_thinking.is_empty());
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(id) }]
                if content == "answer" && thinking == "plan" && id == "assistant-1"
        ));
    }

    #[test]
    fn tool_reducer_question_prepare_excludes_card_and_count() {
        let mut chat = ChatState::new();

        let outcome = chat.reduce_tool(ChatToolAction::PrepareToolStart {
            name: "question".into(),
        });

        assert_eq!(
            outcome.transition,
            ChatToolTransition::StartPrepared(ToolStartPreparationTransition::QuestionOnly {
                finalized_streaming: false,
            })
        );
        assert_eq!(
            outcome.coordination,
            vec![ChatCoordination::Status {
                level: LogLevel::Debug,
                target: "tool",
                message: "tool: question".into(),
            }]
        );
        assert!(outcome.effects.is_empty());
        assert_eq!(
            chat.activity,
            ActivityState::RunningTool {
                name: "question".into()
            }
        );
        assert!(chat.messages.is_empty());
        assert_eq!(chat.session_stats.total_tool_calls, 0);
    }

    #[test]
    fn tool_reducer_inserts_after_pending_thinking_and_counts_once() {
        let mut chat = ChatState::new();
        chat.streaming_thinking = "inspect".into();
        chat.streaming_thinking_message_id = Some("assistant-1".into());
        chat.session_stats.total_tool_calls = 7;

        let outcome = chat.reduce_tool(ChatToolAction::InsertOrReconcileToolStart {
            tool_call_id: Some("tool-1".into()),
            name: "read_tool".into(),
            detail: ToolDetail::ReadTool {
                path: "src/lib.rs".into(),
                start_line: Some(1),
                end_line: Some(5),
            },
        });

        assert_eq!(
            outcome.transition,
            ChatToolTransition::StartInserted(ToolStartInsertionTransition::Inserted {
                moved_thinking: true,
            })
        );
        assert!(outcome.coordination.is_empty());
        assert!(outcome.effects.is_empty());
        assert_eq!(chat.session_stats.total_tool_calls, 8);
        assert!(matches!(
            chat.messages.as_slice(),
            [
                ChatEntry::Thinking { content, message_id: Some(message_id) },
                ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), name, is_error: false, .. },
            ] if content == "inspect"
                && message_id == "assistant-1"
                && tool_call_id == "tool-1"
                && name == "read_tool"
        ));

        chat.session_stats.total_tool_calls = u32::MAX;
        let saturated = chat.reduce_tool(ChatToolAction::InsertOrReconcileToolStart {
            tool_call_id: None,
            name: "shell".into(),
            detail: ToolDetail::None,
        });
        assert_eq!(
            saturated.transition,
            ChatToolTransition::StartInserted(ToolStartInsertionTransition::Inserted {
                moved_thinking: false,
            })
        );
        assert_eq!(chat.session_stats.total_tool_calls, u32::MAX);
    }

    #[test]
    fn tool_reducer_reconciles_failed_start_without_append_or_count() {
        let mut chat = ChatState::new();
        chat.session_stats.total_tool_calls = 7;
        chat.streaming_thinking = "discard duplicate thinking".into();
        chat.push_tool_call(
            Some("tool-1".into()),
            "shell (failed)".into(),
            true,
            ToolDetail::Generic {
                input: None,
                result: Some("failed".into()),
            },
        );

        let outcome = chat.reduce_tool(ChatToolAction::InsertOrReconcileToolStart {
            tool_call_id: Some("tool-1".into()),
            name: "shell".into(),
            detail: ToolDetail::Shell {
                command: "echo late".into(),
                arguments: Vec::new(),
                workdir: None,
                output: None,
            },
        });

        assert_eq!(
            outcome.transition,
            ChatToolTransition::StartInserted(ToolStartInsertionTransition::Reconciled)
        );
        assert_eq!(outcome.coordination, vec![]);
        assert!(outcome.effects.is_empty());
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.session_stats.total_tool_calls, 7);
        assert!(chat.streaming_thinking.is_empty());
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::ToolCall { tool_call_id: Some(id), name, is_error: true, detail: ToolDetail::Shell { command, .. } }]
                if id == "tool-1" && name == "shell" && command == "echo late"
        ));
    }

    #[test]
    fn tool_reducer_failed_fallback_dedupes_and_late_start_restores() {
        let mut chat = ChatState::new();
        let failed = ChatToolAction::ToolCallEnd {
            tool_call_id: Some("tool-1".into()),
            name: "shell".into(),
            is_error: true,
            result: Some("failed".into()),
        };

        let inserted = chat.reduce_tool(failed.clone());
        assert_eq!(
            inserted.transition,
            ChatToolTransition::Ended(ToolEndTransition::FallbackInserted)
        );
        assert_eq!(inserted.coordination, Vec::new());
        assert!(inserted.effects.is_empty());

        let repeated = chat.reduce_tool(failed);
        assert_eq!(
            repeated.transition,
            ChatToolTransition::Ended(ToolEndTransition::NoOp)
        );
        assert!(repeated.coordination.is_empty());
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.session_stats.total_tool_calls, 0);

        let reconciled = chat.reduce_tool(ChatToolAction::InsertOrReconcileToolStart {
            tool_call_id: Some("tool-1".into()),
            name: "shell".into(),
            detail: ToolDetail::Shell {
                command: "echo late".into(),
                arguments: Vec::new(),
                workdir: None,
                output: None,
            },
        });
        assert_eq!(
            reconciled.transition,
            ChatToolTransition::StartInserted(ToolStartInsertionTransition::Reconciled)
        );
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.session_stats.total_tool_calls, 0);
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::ToolCall { name, is_error: true, detail: ToolDetail::Shell { command, .. }, .. }]
                if name == "shell" && command == "echo late"
        ));
    }

    #[test]
    fn tool_reducer_success_before_start_is_a_typed_noop() {
        let mut chat = ChatState::new();

        let outcome = chat.reduce_tool(ChatToolAction::ToolCallEnd {
            tool_call_id: Some("missing".into()),
            name: "shell".into(),
            is_error: false,
            result: Some("done".into()),
        });

        assert_eq!(
            outcome,
            ChatToolOutcome {
                transition: ChatToolTransition::Ended(ToolEndTransition::NoOp),
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );
        assert!(chat.messages.is_empty());
    }

    #[test]
    fn tool_reducer_existing_end_updates_detail_then_failure_once() {
        let mut chat = ChatState::new();
        chat.push_tool_call(
            Some("tool-1".into()),
            "shell".into(),
            false,
            ToolDetail::Shell {
                command: "cargo check".into(),
                arguments: Vec::new(),
                workdir: Some("/repo".into()),
                output: None,
            },
        );
        let end = ChatToolAction::ToolCallEnd {
            tool_call_id: Some("tool-1".into()),
            name: "shell".into(),
            is_error: true,
            result: Some("line one\nline two".into()),
        };

        let updated = chat.reduce_tool(end.clone());
        assert_eq!(
            updated.transition,
            ChatToolTransition::Ended(ToolEndTransition::Updated {
                detail_updated: true,
                marked_failed: true,
            })
        );
        assert_eq!(updated.coordination, Vec::new());
        assert!(updated.effects.is_empty());
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::ToolCall { is_error: true, detail: ToolDetail::Shell { output: Some(output), .. }, .. }]
                if output.stdout == "line one\nline two" && output.stderr.is_empty()
        ));

        let repeated = chat.reduce_tool(end);
        assert_eq!(
            repeated.transition,
            ChatToolTransition::Ended(ToolEndTransition::NoOp)
        );
        assert!(repeated.coordination.is_empty());
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
    fn error_reducer_closes_live_timing_and_types_error_deduplication() {
        let mut chat = ChatState::new();
        chat.recent_prompt_text = Some("preserved prompt".into());
        chat.session_stats.open_llm_request_instant = Some(Instant::now() - Duration::from_secs(2));

        let inserted = chat.reduce(ChatAction::AcpError {
            message: "connection lost".into(),
        });

        assert_eq!(
            inserted,
            ChatOutcome {
                transition: ChatTransition::AcpError {
                    error_inserted: true,
                },
                coordination: vec![ChatCoordination::Status {
                    level: LogLevel::Error,
                    target: "acp",
                    message: "error: connection lost".into(),
                }],
                effects: Vec::new(),
            }
        );
        assert!(chat.session_stats.open_llm_request_instant.is_none());
        assert!(chat.session_stats.active_llm_duration >= Duration::from_secs(2));
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("preserved prompt"));
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "connection lost"
        ));

        chat.session_stats.open_llm_request_instant = Some(Instant::now());
        let duplicate = chat.reduce(ChatAction::AcpError {
            message: "connection lost".into(),
        });
        assert_eq!(
            duplicate.transition,
            ChatTransition::AcpError {
                error_inserted: false,
            }
        );
        assert_eq!(duplicate.coordination, inserted.coordination);
        assert!(duplicate.effects.is_empty());
        assert!(chat.session_stats.open_llm_request_instant.is_none());
        assert_eq!(chat.messages.len(), 1);
    }

    #[test]
    fn backend_prompt_failure_reducer_preserves_matching_and_dedupe_semantics() {
        let mut chat = ChatState::new();
        let failed_id = chat.push_pending_prompt("failed".into());
        let retained_id = chat.push_pending_prompt("retained".into());
        chat.recent_prompt_text = Some("preserved prompt".into());
        chat.session_stats.open_llm_request_instant = Some(Instant::now() - Duration::from_secs(2));

        let failed = chat.reduce(ChatAction::BackendPromptFailed {
            local_id: failed_id.clone(),
            message: "backend rejected prompt".into(),
        });

        assert_eq!(
            failed,
            ChatOutcome {
                transition: ChatTransition::BackendPromptFailed {
                    prompt_rolled_back: true,
                    error_inserted: true,
                },
                coordination: vec![ChatCoordination::Status {
                    level: LogLevel::Error,
                    target: "acp",
                    message: "error: backend rejected prompt".into(),
                },],
                effects: Vec::new(),
            }
        );
        assert!(chat.session_stats.open_llm_request_instant.is_none());
        assert!(chat.session_stats.active_llm_duration >= Duration::from_secs(2));
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("preserved prompt"));
        assert!(matches!(
            chat.messages.as_slice(),
            [
                ChatEntry::User { text, message_id: Some(message_id) },
                ChatEntry::Error(message),
            ] if text == "retained"
                && message_id == &retained_id
                && message == "backend rejected prompt"
        ));

        chat.session_stats.open_llm_request_instant = Some(Instant::now());
        let nonmatching = chat.reduce(ChatAction::BackendPromptFailed {
            local_id: "missing".into(),
            message: "backend rejected prompt".into(),
        });
        assert_eq!(
            nonmatching.transition,
            ChatTransition::BackendPromptFailed {
                prompt_rolled_back: false,
                error_inserted: false,
            }
        );
        assert_eq!(nonmatching.coordination, failed.coordination);
        assert!(nonmatching.effects.is_empty());
        assert!(chat.session_stats.open_llm_request_instant.is_none());
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("preserved prompt"));
    }

    #[test]
    fn runtime_prompt_dispatch_failure_reducer_only_coordinates_matching_rollback() {
        let mut chat = ChatState::new();
        let failed_id = chat.push_pending_prompt("failed".into());
        let retained_id = chat.push_pending_prompt("retained".into());
        chat.recent_prompt_text = Some("preserved prompt".into());
        chat.session_stats.open_llm_request_instant = Some(Instant::now());
        let open_timing = chat.session_stats.open_llm_request_instant;

        let failed = chat.reduce(ChatAction::RuntimePromptDispatchFailed {
            local_id: failed_id,
        });

        assert_eq!(
            failed,
            ChatOutcome {
                transition: ChatTransition::RuntimePromptDispatchFailed {
                    prompt_rolled_back: true,
                },
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );
        assert_eq!(chat.session_stats.open_llm_request_instant, open_timing);
        assert_eq!(chat.session_stats.active_llm_duration, Duration::ZERO);
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("preserved prompt"));
        assert!(matches!(
            chat.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "retained" && message_id == &retained_id
        ));

        let nonmatching = chat.reduce(ChatAction::RuntimePromptDispatchFailed {
            local_id: "missing".into(),
        });
        assert_eq!(
            nonmatching,
            ChatOutcome {
                transition: ChatTransition::RuntimePromptDispatchFailed {
                    prompt_rolled_back: false,
                },
                coordination: Vec::new(),
                effects: Vec::new(),
            }
        );
        assert_eq!(chat.session_stats.open_llm_request_instant, open_timing);
        assert_eq!(chat.session_stats.active_llm_duration, Duration::ZERO);
        assert_eq!(chat.recent_prompt_text.as_deref(), Some("preserved prompt"));
        assert!(
            chat.messages
                .iter()
                .all(|entry| !matches!(entry, ChatEntry::Error(_)))
        );
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
            vec![ChatCoordination::Status {
                level: LogLevel::Debug,
                target: "activity",
                message: "thinking...".into(),
            },]
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
        assert_eq!(reconciled.coordination, Vec::new());
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
        assert_eq!(boundary.coordination, vec![]);
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
        let expected_base_coordination = vec![];
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
        assert_eq!(replaced.coordination, vec![]);
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
            vec![ChatCoordination::Status {
                level: LogLevel::Warn,
                target: "activity",
                message: "cancelled".into(),
            },]
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
            vec![ChatCoordination::Status {
                level: LogLevel::Debug,
                target: "activity",
                message: "finished: EndTurn".into(),
            },]
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
    fn elicitation_custom_editor_preserves_utf8_editing_and_hard_line_navigation() {
        let mut ui = ElicitationUiState::default();
        let mut input = String::new();

        ui.custom_insert(&mut input, '界');
        ui.custom_insert(&mut input, 'é');
        assert_eq!(ui.custom_cursor, input.len());
        ui.custom_left(&input);
        assert_eq!(ui.custom_cursor, "界".len());
        ui.custom_delete(&mut input);
        assert_eq!(input, "界");
        ui.custom_backspace(&mut input);
        assert!(input.is_empty());
        assert_eq!(ui.custom_cursor, 0);

        input = "ab\n界d".into();
        ui.custom_cursor = input.len();
        ui.custom_home(&input);
        assert_eq!(ui.custom_cursor, "ab\n".len());
        ui.custom_end(&input);
        assert_eq!(ui.custom_cursor, input.len());
        ui.custom_move_visual(&input, 20, -1);
        assert_eq!(ui.custom_cursor, 2);
    }

    #[test]
    fn elicitation_editor_reuses_composer_visual_layout() {
        let mut ui = ElicitationUiState {
            custom_cursor: 4,
            ..Default::default()
        };
        ui.custom_move_visual("abcdef", 4, -1);
        assert_eq!(ui.custom_cursor, 2);
        ui.custom_move_visual("abcdef", 4, 1);
        assert_eq!(ui.custom_cursor, 4);
    }
}
