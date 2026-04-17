//! Server message handling for the TUI application.
//!
//! Contains `handle_server_msg`, `handle_event_kind`, `replay_audit`, and
//! helper functions for parsing tool details, updating tool results, and
//! building diff/write content lines.

use std::collections::HashMap;

use crate::app::*;
use crate::protocol::*;
use crate::ui::{ELLIPSIS, OUTCOME_BULLET, build_diff_lines, build_write_lines};

impl App {
    pub fn handle_server_msg(&mut self, raw: RawServerMsg) -> Vec<ClientMsg> {
        match raw.msg_type.as_str() {
            "state" => {
                if let Some(data) = raw.data
                    && let Ok(state) = serde_json::from_value::<StateData>(data)
                {
                    self.agent_id = state.agents.first().map(|a| a.id.clone());
                    if let Some(mode) = state.agent_mode {
                        self.agent_mode = mode;
                    }
                    // Only update reasoning_effort when the key was present in
                    // the JSON; absent means the server didn't report it.
                    match state.reasoning_effort {
                        ReasoningEffortField::Absent => {}
                        ReasoningEffortField::Auto => self.reasoning_effort = None,
                        ReasoningEffortField::Set(s) => self.reasoning_effort = Some(s),
                    }
                    self.conn = ConnState::Connected;
                    self.set_status(LogLevel::Info, "connection", "connected");
                }
                vec![]
            }
            "reasoning_effort" => {
                if let Some(data) = raw.data
                    && let Ok(re) = serde_json::from_value::<ReasoningEffortData>(data)
                {
                    self.reasoning_effort = match re.reasoning_effort.as_deref() {
                        None | Some("auto") => None,
                        Some(s) => Some(s.to_string()),
                    };
                    // Server is authoritative — cache so this session + mode
                    // remembers the level across restarts.
                    self.cache_session_mode_state();
                }
                vec![]
            }
            "agent_mode" => {
                if let Some(data) = raw.data
                    && let Ok(am) = serde_json::from_value::<AgentModeData>(data)
                {
                    self.agent_mode = am.mode;
                }
                vec![]
            }
            "file_index" => {
                if let Some(data) = raw.data
                    && let Ok(fi) = serde_json::from_value::<FileIndexData>(data)
                {
                    self.file_index = fi
                        .files
                        .into_iter()
                        .map(|entry| FileIndexEntryLite {
                            path: entry.path,
                            is_dir: entry.is_dir,
                        })
                        .collect();
                    self.file_index_generated_at = Some(fi.generated_at);
                    self.file_index_loading = false;
                    self.file_index_error = None;
                    self.refresh_mention_state();
                }
                vec![]
            }
            "undo_result" => {
                self.activity = ActivityState::Idle;
                if let Some(data) = raw.data
                    && let Ok(ur) = serde_json::from_value::<UndoResultData>(data)
                {
                    let message_id_for_files = ur
                        .message_id
                        .clone()
                        .or_else(|| ur.undo_stack.last().map(|frame| frame.message_id.clone()));
                    let next = self.build_undo_state_from_server_stack(
                        &ur.undo_stack,
                        message_id_for_files.as_deref(),
                        if ur.success {
                            Some(ur.reverted_files.as_slice())
                        } else {
                            None
                        },
                    );
                    self.undo_state = next;

                    if ur.success {
                        self.recent_prompt_text = None;
                        self.streaming_content.clear();
                        self.streaming_cache.invalidate();
                        self.set_status(LogLevel::Info, "session", "undone - reloading session");
                        if let Some(ref sid) = self.session_id {
                            return vec![ClientMsg::LoadSession {
                                session_id: sid.clone(),
                            }];
                        }
                    } else {
                        self.set_status(
                            LogLevel::Warn,
                            "session",
                            ur.message.unwrap_or_else(|| "undo failed".into()),
                        );
                    }
                }
                vec![]
            }
            "redo_result" => {
                self.activity = ActivityState::Idle;
                if let Some(data) = raw.data
                    && let Ok(rr) = serde_json::from_value::<RedoResultData>(data)
                {
                    self.undo_state =
                        self.build_undo_state_from_server_stack(&rr.undo_stack, None, None);
                    if rr.success {
                        self.set_status(LogLevel::Info, "session", "redone - reloading session");
                        if let Some(ref sid) = self.session_id {
                            return vec![ClientMsg::LoadSession {
                                session_id: sid.clone(),
                            }];
                        }
                    } else {
                        self.set_status(
                            LogLevel::Warn,
                            "session",
                            rr.message.unwrap_or_else(|| "redo failed".into()),
                        );
                    }
                }
                vec![]
            }
            "session_list" => {
                if let Some(data) = raw.data
                    && let Ok(list) = serde_json::from_value::<SessionListData>(data)
                {
                    // Sort sessions within each group by updated_at descending.
                    let mut groups = list.groups;
                    for group in &mut groups {
                        group
                            .sessions
                            .sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                    }
                    // Sort groups by their most-recent session activity descending.
                    groups.sort_by(|a, b| {
                        let a_latest = a.sessions.first().and_then(|s| s.updated_at.as_deref());
                        let b_latest = b.sessions.first().and_then(|s| s.updated_at.as_deref());
                        b_latest.cmp(&a_latest)
                    });

                    let total: usize = groups.iter().map(|g| g.sessions.len()).sum();
                    self.session_groups = groups;

                    // Clamp cursor to the new visible item count.
                    let visible_len = self.visible_start_items().len();
                    if self.session_cursor >= visible_len && visible_len > 0 {
                        self.session_cursor = visible_len - 1;
                    }
                    self.set_status(LogLevel::Info, "session", format!("{} session(s)", total));
                }
                vec![]
            }
            "session_created" => {
                if let Some(data) = raw.data
                    && let Ok(sc) = serde_json::from_value::<SessionCreatedData>(data)
                {
                    self.session_id = Some(sc.session_id.clone());
                    self.agent_id = Some(sc.agent_id);
                    self.messages.clear();
                    self.streaming_content.clear();
                    self.streaming_cache.invalidate();
                    self.scroll_offset = 0;
                    self.undo_state = None;
                    self.undoable_turns.clear();
                    self.recent_prompt_text = None;
                    self.suppress_turn_output = false;
                    self.delegate_entries.clear();
                    self.parent_session_id = None;
                    self.pending_parent_session_id = None;
                    self.suppress_delegation_result = false;
                    self.file_index.clear();
                    self.file_index_generated_at = None;
                    self.file_index_loading = false;
                    self.file_index_error = None;
                    self.mention_state = None;
                    self.last_compaction_token_estimate = None;
                    self.elicitation = None;
                    self.clear_cancel_confirm();
                    self.cumulative_cost = None;
                    self.session_stats = SessionStatsLite::default();
                    self.screen = Screen::Chat;
                    self.set_status(LogLevel::Info, "session", "session created");
                    return vec![ClientMsg::SubscribeSession {
                        session_id: sc.session_id,
                        agent_id: self.agent_id.clone(),
                    }];
                }
                vec![]
            }
            "session_loaded" => {
                if let Some(data) = raw.data {
                    match serde_json::from_value::<SessionLoadedData>(data.clone()) {
                        Err(e) => {
                            self.activity = ActivityState::Idle;
                            self.set_status(LogLevel::Error, "session", format!("load error: {e}"));
                        }
                        Ok(sl) => {
                            self.activity = ActivityState::Idle;
                            // Resolve parent before moving sl.session_id: prefer explicit
                            // pending value (from delegate popup), fall back to session_groups.
                            self.parent_session_id =
                                self.pending_parent_session_id.take().or_else(|| {
                                    self.session_groups
                                        .iter()
                                        .flat_map(|g| &g.sessions)
                                        .find(|s| s.session_id == sl.session_id)
                                        .and_then(|s| s.parent_session_id.clone())
                                });
                            self.session_id = Some(sl.session_id);
                            self.agent_id = Some(sl.agent_id);
                            self.messages.clear();
                            self.streaming_content.clear();
                            self.streaming_cache.invalidate();
                            self.scroll_offset = 0;
                            self.cumulative_cost = None;
                            self.session_stats = SessionStatsLite::default();
                            self.screen = if self.parent_session_id.is_some() {
                                Screen::Delegate
                            } else {
                                Screen::Chat
                            };
                            self.undoable_turns.clear();
                            self.recent_prompt_text = None;
                            self.suppress_turn_output = false;
                            self.suppress_delegation_result = false;
                            // Keep parent's delegate entries when navigating to a child
                            // session; otherwise clear for unrelated session switches.
                            if self.parent_session_id.is_none() {
                                self.delegate_entries.clear();
                            }
                            self.file_index.clear();
                            self.file_index_generated_at = None;
                            self.file_index_loading = false;
                            self.file_index_error = None;
                            self.mention_state = None;
                            self.last_compaction_token_estimate = None;
                            self.elicitation = None;
                            self.clear_cancel_confirm();
                            self.undo_state =
                                self.build_undo_state_from_server_stack(&sl.undo_stack, None, None);
                            self.set_status(LogLevel::Debug, "activity", "ready");
                            // Replay audit: sets current_provider/model (ProviderChanged)
                            // and agent_mode (SessionModeChanged).
                            self.replay_audit(&sl.audit);

                            // Restore the session's mode on the server.
                            let mut cmds = vec![ClientMsg::SetAgentMode {
                                mode: self.agent_mode.clone(),
                            }];
                            // Restore cached model + effort for this session + mode.
                            cmds.extend(self.apply_cached_mode_state());
                            // Drain any subscribe commands queued during audit replay
                            // (e.g. SubscribeSession for delegation child sessions).
                            cmds.extend(self.drain_pending());
                            return cmds;
                        }
                    }
                }
                vec![]
            }
            "session_events" => {
                if let Some(data) = raw.data {
                    if let Ok(parsed) = serde_json::from_value::<SessionEventsData>(data.clone()) {
                        if self.session_id.as_deref() == Some(parsed.session_id.as_str()) {
                            self.note_session_activity(&parsed.session_id);
                            for envelope in parsed.events {
                                self.handle_event(&envelope);
                            }
                        } else if let Some(entry) = self.delegate_entries.iter_mut().find(|e| {
                            e.child_session_id.as_deref() == Some(parsed.session_id.as_str())
                        }) {
                            // Delegate child: accumulate stats without inflating
                            // the multi-session activity badge.
                            for envelope in &parsed.events {
                                accumulate_delegate_stats(&mut entry.stats, envelope.kind());
                            }
                        } else {
                            self.note_session_activity(&parsed.session_id);
                        }
                    } else {
                        match serde_json::from_value::<SessionEventsDataRaw>(data) {
                            Err(e) => {
                                // Structural error (missing session_id etc.) — log it.
                                self.push_log(
                                    LogLevel::Warn,
                                    "session_events",
                                    format!("session_events parse error: {e}"),
                                );
                            }
                            Ok(se) => {
                                if let Some(entry) = self.delegate_entries.iter_mut().find(|e| {
                                    e.child_session_id.as_deref() == Some(se.session_id.as_str())
                                }) {
                                    for val in se.events {
                                        // Child sessions keep the raw parse path so
                                        // unknown kinds do not block known stats updates.
                                        if let Ok(envelope) =
                                            serde_json::from_value::<EventEnvelope>(val)
                                        {
                                            accumulate_delegate_stats(
                                                &mut entry.stats,
                                                envelope.kind(),
                                            );
                                        }
                                    }
                                } else {
                                    self.note_session_activity(&se.session_id);
                                }
                            }
                        }
                    }
                }
                self.drain_pending()
            }
            "event" => {
                if let Some(data) = raw.data {
                    if let Ok(parsed) = serde_json::from_value::<EventData>(data.clone()) {
                        if self.session_id.as_deref() == Some(parsed.session_id.as_str()) {
                            self.note_session_activity(&parsed.session_id);
                            self.handle_event(&parsed.event);
                        } else if let Some(entry) = self.delegate_entries.iter_mut().find(|e| {
                            e.child_session_id.as_deref() == Some(parsed.session_id.as_str())
                        }) {
                            accumulate_delegate_stats(&mut entry.stats, parsed.event.kind());
                        } else {
                            self.note_session_activity(&parsed.session_id);
                        }
                    } else if let Ok(ed) = serde_json::from_value::<EventDataRaw>(data) {
                        if let Some(entry) = self
                            .delegate_entries
                            .iter_mut()
                            .find(|e| e.child_session_id.as_deref() == Some(ed.session_id.as_str()))
                        {
                            if let Ok(envelope) = serde_json::from_value::<EventEnvelope>(ed.event)
                            {
                                accumulate_delegate_stats(&mut entry.stats, envelope.kind());
                            }
                        } else {
                            self.note_session_activity(&ed.session_id);
                        }
                    }
                }
                self.drain_pending()
            }
            "all_models_list" => {
                if let Some(data) = raw.data
                    && let Ok(ml) = serde_json::from_value::<AllModelsData>(data)
                {
                    self.models = ml.models;
                }
                vec![]
            }
            "auth_providers" => {
                if let Some(data) = raw.data
                    && let Ok(ap) = serde_json::from_value::<AuthProvidersData>(data)
                {
                    self.auth_providers = ap.providers;
                    self.push_log(
                        LogLevel::Debug,
                        "auth",
                        format!("{} auth provider(s)", self.auth_providers.len()),
                    );
                }
                vec![]
            }
            "oauth_flow_started" => {
                if let Some(data) = raw.data
                    && let Ok(flow) = serde_json::from_value::<OAuthFlowData>(data)
                {
                    self.push_log(
                        LogLevel::Info,
                        "auth",
                        format!("OAuth flow started for {}", flow.provider),
                    );
                    self.auth_oauth_flow = Some(flow);
                    self.auth_panel = AuthPanel::OAuthFlow;
                    self.auth_oauth_response.clear();
                    self.auth_oauth_response_cursor = 0;
                    self.auth_result_message = None;
                }
                vec![]
            }
            "oauth_result" => {
                if let Some(data) = raw.data
                    && let Ok(result) = serde_json::from_value::<OAuthResultData>(data)
                {
                    let level = if result.success {
                        LogLevel::Info
                    } else {
                        LogLevel::Warn
                    };
                    self.push_log(level, "auth", &result.message);
                    self.auth_result_message = Some((result.success, result.message));
                    if result.success {
                        self.auth_oauth_flow = None;
                        self.auth_panel = AuthPanel::List;
                    }
                    // Refresh provider list to show updated status
                    return vec![ClientMsg::ListAuthProviders];
                }
                vec![]
            }
            "api_token_result" => {
                if let Some(data) = raw.data
                    && let Ok(result) = serde_json::from_value::<ApiTokenResultData>(data)
                {
                    let level = if result.success {
                        LogLevel::Info
                    } else {
                        LogLevel::Warn
                    };
                    self.push_log(level, "auth", &result.message);
                    self.auth_result_message = Some((result.success, result.message));
                    if result.success {
                        self.auth_api_key_input.clear();
                        self.auth_api_key_cursor = 0;
                    }
                    // Refresh provider list to show updated status
                    return vec![ClientMsg::ListAuthProviders];
                }
                vec![]
            }
            "error" => {
                if let Some(data) = raw.data
                    && let Ok(e) = serde_json::from_value::<ErrorData>(data)
                {
                    self.messages.push(ChatEntry::Error(e.message.clone()));
                    self.set_status(LogLevel::Error, "server", format!("error: {}", e.message));
                }
                vec![]
            }
            _ => vec![],
        }
    }

    fn handle_event(&mut self, envelope: &EventEnvelope) {
        self.apply_event_stats(envelope.kind(), envelope.timestamp());
        self.handle_event_kind(envelope.kind(), false);
    }

    pub(crate) fn handle_event_kind(&mut self, kind: &EventKind, is_replay: bool) {
        match kind {
            EventKind::PromptReceived {
                content,
                message_id,
            } => {
                let text = content_to_string(content);
                if !text.is_empty() {
                    let frontier_message_id = self
                        .undo_state
                        .as_ref()
                        .and_then(|state| state.frontier_message_id.clone());

                    if let Some(message_id) = message_id.as_deref()
                        && self.messages.iter().any(|entry| {
                            matches!(entry, ChatEntry::User { message_id: Some(mid), .. } if mid == message_id)
                        })
                    {
                        self.recent_prompt_text = Some(text.clone());
                        self.suppress_turn_output = false;
                        return;
                    }

                    if !is_replay && let Some(frontier_message_id) = frontier_message_id {
                        if let Some(frontier_idx) = self
                            .messages
                            .iter()
                            .position(|entry| matches!(entry, ChatEntry::User { message_id: Some(mid), .. } if mid == &frontier_message_id))
                        {
                            self.messages.truncate(frontier_idx);
                        }
                        if let Some(turn_idx) = self
                            .undoable_turns
                            .iter()
                            .position(|turn| turn.message_id == frontier_message_id)
                        {
                            self.undoable_turns.truncate(turn_idx);
                        }
                        // A replayed frontier prompt arrives after the backend has already
                        // moved the branch point; prune stale UI state but do not re-add it.
                        if message_id.as_deref() == Some(frontier_message_id.as_str()) {
                            self.suppress_turn_output = true;
                            return;
                        }
                        self.undo_state = None;
                    }

                    self.suppress_turn_output = false;
                    self.messages.push(ChatEntry::User {
                        text: text.clone(),
                        message_id: message_id.clone(),
                    });
                    self.recent_prompt_text = Some(text.clone());
                    if let Some(message_id) = message_id.clone()
                        && !self
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
            }
            EventKind::UserMessageStored { content } => {
                let text = content_to_string(content);
                if !text.is_empty() {
                    // Suppress the noisy batch-result message that immediately
                    // follows DelegationCompleted / DelegationFailed.
                    if self.suppress_delegation_result {
                        self.suppress_delegation_result = false;
                        return;
                    }
                    // If undo_state is still active during a live event, the matching
                    // PromptReceived was suppressed as the reverted frontier turn.
                    if !is_replay && (self.undo_state.is_some() || self.suppress_turn_output) {
                        return;
                    }
                    let dup = self.recent_prompt_text.as_deref() == Some(text.as_str())
                        || matches!(
                            self.messages.last(),
                            Some(ChatEntry::User { text: last, .. }) if last == &text
                        )
                        || self
                            .undoable_turns
                            .last()
                            .map(|turn| turn.text == text)
                            .unwrap_or(false);
                    if !dup {
                        self.messages.push(ChatEntry::User {
                            text,
                            message_id: None,
                        });
                    }
                }
            }
            EventKind::TurnStarted => {
                self.clear_cancel_confirm();
                self.activity = ActivityState::Thinking;
                self.streaming_content.clear();
                self.invalidate_streaming_caches();
                self.set_status(LogLevel::Debug, "activity", "thinking...");
            }
            EventKind::AssistantThinkingDelta { content, .. } => {
                self.streaming_thinking.push_str(content);
            }
            EventKind::AssistantContentDelta { content, .. } => {
                if self.is_turn_active() {
                    self.activity = ActivityState::Streaming;
                }
                self.streaming_content.push_str(content);
            }
            EventKind::CompactionStart { token_estimate } => {
                self.activity = ActivityState::Compacting {
                    token_estimate: *token_estimate,
                };
                self.last_compaction_token_estimate = Some(*token_estimate);
                self.messages.push(ChatEntry::CompactionStart {
                    token_estimate: *token_estimate,
                });
                self.set_status(
                    LogLevel::Debug,
                    "activity",
                    format!("compacting context (~{token_estimate} tokens)"),
                );
            }
            EventKind::CompactionEnd {
                summary,
                summary_len,
            } => {
                self.activity = if self.streaming_content.is_empty() {
                    ActivityState::Thinking
                } else {
                    ActivityState::Streaming
                };
                self.messages
                    .retain(|entry| !matches!(entry, ChatEntry::CompactionStart { .. }));
                self.messages.push(ChatEntry::CompactionEnd {
                    token_estimate: self.last_compaction_token_estimate,
                    summary: summary.clone(),
                    summary_len: *summary_len,
                });
                self.set_status(LogLevel::Info, "activity", "context compacted");
            }
            EventKind::AssistantMessageStored {
                content,
                thinking,
                message_id,
            } => {
                let thinking_text = thinking.clone().or_else(|| {
                    if self.streaming_thinking.is_empty() {
                        None
                    } else {
                        Some(std::mem::take(&mut self.streaming_thinking))
                    }
                });
                self.streaming_content.clear();
                self.invalidate_streaming_caches();
                if self.is_turn_active() {
                    self.activity = ActivityState::Thinking;
                }
                if !content.is_empty() {
                    self.recent_prompt_text = None;
                    if self.suppress_turn_output {
                        return;
                    }
                    if let Some(message_id) = message_id.as_deref()
                        && self.messages.iter().any(|entry| {
                            matches!(entry, ChatEntry::Assistant { message_id: Some(mid), .. } if mid == message_id)
                        })
                    {
                        return;
                    }
                    self.messages.push(ChatEntry::Assistant {
                        content: content.clone(),
                        thinking: thinking_text,
                        message_id: message_id.clone(),
                    });
                }
            }
            EventKind::ToolCallStart {
                tool_call_id,
                tool_name,
                arguments,
            } => {
                if self.suppress_turn_output {
                    return;
                }
                self.activity = ActivityState::RunningTool {
                    name: tool_name.clone(),
                };
                self.set_status(LogLevel::Debug, "tool", format!("tool: {tool_name}"));
                // The question tool renders as an ElicitationCard — skip the
                // redundant "> question …" tool call entry in the chat.
                if tool_name != "question" {
                    let detail = parse_tool_detail(tool_name, arguments.as_ref());
                    self.messages.push(ChatEntry::ToolCall {
                        tool_call_id: tool_call_id.clone(),
                        name: tool_name.clone(),
                        is_error: false,
                        detail,
                    });
                }
            }
            EventKind::ToolCallEnd {
                tool_call_id,
                tool_name,
                is_error,
                result,
            } => {
                if tool_name == "question" {
                    if is_replay && let Some(result_str) = result {
                        backfill_elicitation_outcomes(&mut self.messages, result_str);
                    }
                } else {
                    if let Some(result_str) = result {
                        update_tool_detail(&mut self.messages, tool_call_id.as_deref(), result_str);
                    }
                    if is_error.unwrap_or(false) {
                        self.messages.push(ChatEntry::ToolCall {
                            tool_call_id: tool_call_id.clone(),
                            name: format!("{tool_name} (failed)"),
                            is_error: true,
                            detail: ToolDetail::None,
                        });
                    }
                }
            }
            EventKind::ProviderChanged {
                provider,
                model,
                context_limit,
                ..
            } => {
                self.current_provider = Some(provider.clone());
                self.current_model = Some(model.clone());
                if let Some(limit) = context_limit {
                    self.context_limit = *limit;
                }
                // Keep the session cache in sync with live model changes.
                if !is_replay {
                    self.cache_session_mode_state();
                }
            }
            EventKind::LlmRequestEnd {
                cumulative_cost_usd,
                ..
            } => {
                self.clear_cancel_confirm();
                self.activity = ActivityState::Idle;
                self.cumulative_cost = *cumulative_cost_usd;
                self.set_status(LogLevel::Debug, "activity", "ready");
            }
            EventKind::Error { message } => {
                self.activity = ActivityState::Idle;
                self.clear_cancel_confirm();
                self.messages.push(ChatEntry::Error(message.clone()));
                self.set_status(LogLevel::Error, "server", format!("error: {message}"));
            }
            EventKind::ElicitationRequested {
                elicitation_id,
                message,
                source,
                requested_schema,
                ..
            } => {
                if is_replay {
                    // During replay the elicitation was already answered —
                    // show the card as resolved without reopening the popup.
                    self.messages.push(ChatEntry::Elicitation {
                        elicitation_id: elicitation_id.clone(),
                        message: message.clone(),
                        source: source.clone(),
                        outcome: Some("responded".into()),
                    });
                    return;
                }
                let fields = ElicitationState::parse_schema(requested_schema);
                self.elicitation = Some(ElicitationState {
                    elicitation_id: elicitation_id.clone(),
                    message: message.clone(),
                    source: source.clone(),
                    fields,
                    field_cursor: 0,
                    option_cursor: 0,
                    selected: HashMap::new(),
                    text_input: String::new(),
                    text_cursor: 0,
                });
                self.messages.push(ChatEntry::Elicitation {
                    elicitation_id: elicitation_id.clone(),
                    message: message.clone(),
                    source: source.clone(),
                    outcome: None,
                });
                self.scroll_offset = 0;
                self.set_status(
                    LogLevel::Info,
                    "elicitation",
                    "question — answer in the panel above input",
                );
            }
            EventKind::SessionModeChanged { mode } => {
                self.agent_mode = mode.clone();
            }
            EventKind::Cancelled => {
                self.activity = ActivityState::Idle;
                self.clear_cancel_confirm();
                if self.suppress_turn_output {
                    self.streaming_content.clear();
                    self.streaming_thinking.clear();
                    self.invalidate_streaming_caches();
                    self.set_status(LogLevel::Warn, "activity", "cancelled");
                    return;
                }
                if !self.streaming_content.is_empty() {
                    let partial = std::mem::take(&mut self.streaming_content);
                    self.streaming_cache.invalidate();
                    let thinking = if self.streaming_thinking.is_empty() {
                        None
                    } else {
                        Some(std::mem::take(&mut self.streaming_thinking))
                    };
                    self.messages.push(ChatEntry::Assistant {
                        content: format!("{partial} [cancelled]"),
                        thinking,
                        message_id: None,
                    });
                }
                self.set_status(LogLevel::Warn, "activity", "cancelled");
            }
            // ── Delegation lifecycle events ─────────────────────────────────────
            EventKind::DelegationRequested { delegation } => {
                // Idempotent upsert: the parent session can be replayed more than
                // once (session_loaded audit + parent session_events replay). Keep
                // exactly one DelegateEntry per delegation.public_id.
                if let Some(entry) = self
                    .delegate_entries
                    .iter_mut()
                    .find(|e| e.delegation_id == delegation.public_id)
                {
                    // Refresh metadata in case a later replay has more fields.
                    if entry.target_agent_id.is_none() {
                        entry.target_agent_id = delegation.target_agent_id.clone();
                    }
                    if entry.objective.is_empty() {
                        entry.objective = delegation.objective.clone().unwrap_or_default();
                    }
                } else {
                    self.delegate_entries.push(DelegateEntry {
                        delegation_id: delegation.public_id.clone(),
                        child_session_id: None,
                        target_agent_id: delegation.target_agent_id.clone(),
                        objective: delegation.objective.clone().unwrap_or_default(),
                        status: DelegateStatus::InProgress,
                        stats: DelegateStats::default(),
                    });
                }
            }
            EventKind::SessionForked {
                child_session_id,
                origin,
                fork_point_ref,
                target_agent_id,
            } => {
                if origin.as_deref() == Some("delegation") {
                    if let Some(sid) = child_session_id {
                        // Update the matching delegation entry.
                        if let Some(delegation_id) = fork_point_ref {
                            if let Some(entry) = self
                                .delegate_entries
                                .iter_mut()
                                .find(|e| e.delegation_id == *delegation_id)
                            {
                                entry.child_session_id = Some(sid.clone());
                            }
                        }
                        // Subscribe to the child session using the delegation's
                        // target agent_id — matching the web UI behaviour.
                        // Fall back to parent agent_id if not present.
                        let agent_id = target_agent_id.clone().or_else(|| self.agent_id.clone());
                        self.pending_commands.push(ClientMsg::SubscribeSession {
                            session_id: sid.clone(),
                            agent_id,
                        });
                    }
                }
            }
            EventKind::DelegationCompleted { delegation_id, .. } => {
                if let Some(entry) = self
                    .delegate_entries
                    .iter_mut()
                    .find(|e| e.delegation_id == *delegation_id)
                {
                    entry.status = DelegateStatus::Completed;
                }
                self.suppress_delegation_result = true;
            }
            EventKind::DelegationFailed { delegation_id, .. } => {
                if let Some(entry) = self
                    .delegate_entries
                    .iter_mut()
                    .find(|e| e.delegation_id == *delegation_id)
                {
                    entry.status = DelegateStatus::Failed;
                }
                self.suppress_delegation_result = true;
            }
            _ => {}
        }
    }

    /// Drain any commands queued by event handlers (e.g. SubscribeSession
    /// for delegation child sessions) and return them to the caller.
    fn drain_pending(&mut self) -> Vec<ClientMsg> {
        std::mem::take(&mut self.pending_commands)
    }

    pub(crate) fn replay_audit(&mut self, audit: &serde_json::Value) {
        if let Some(events) = audit.get("events").and_then(|e| e.as_array()) {
            let frontier_message_id = self
                .undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref());
            let mut replay_cutoff = events.len();

            if let Some(frontier_message_id) = frontier_message_id
                && let Some(idx) = events.iter().position(|event_val| {
                    serde_json::from_value::<AgentEvent>(event_val.clone())
                        .ok()
                        .and_then(|event| match event.kind {
                            EventKind::PromptReceived {
                                message_id: Some(message_id),
                                ..
                            } => Some(message_id == frontier_message_id),
                            _ => None,
                        })
                        .unwrap_or(false)
                })
            {
                replay_cutoff = idx;
            }

            for event_val in events.iter().take(replay_cutoff) {
                if let Ok(agent_event) = serde_json::from_value::<AgentEvent>(event_val.clone()) {
                    self.apply_event_stats(&agent_event.kind, agent_event.timestamp);
                    self.handle_event_kind(&agent_event.kind, true);
                }
            }
        }
    }
}

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
        EventKind::ProviderChanged { context_limit, .. } => {
            if let Some(limit) = context_limit {
                stats.context_limit = *limit;
            }
        }
        _ => {}
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
        let labels: Vec<String> = answer_entry
            .get("answers")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| format!("{OUTCOME_BULLET}{s}"))
                    .collect()
            })
            .unwrap_or_default();
        *outcome = Some(labels.join("\n"));
    }
}

fn parse_tool_detail(tool_name: &str, arguments: Option<&serde_json::Value>) -> ToolDetail {
    let Some(args) = arguments else {
        return ToolDetail::None;
    };
    // arguments can be a JSON string or an object
    let obj = if let Some(s) = args.as_str() {
        serde_json::from_str::<serde_json::Value>(s).unwrap_or_default()
    } else {
        args.clone()
    };

    let str_field = |key: &str| -> String {
        obj.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let short = |path: &str| -> String {
        let mut count = 0;
        for (i, c) in path.char_indices().rev() {
            if c == '/' {
                count += 1;
                if count == 2 {
                    return path[i + 1..].to_string();
                }
            }
        }
        path.to_string()
    };

    match tool_name {
        "edit" => {
            let file = obj
                .get("filePath")
                .or_else(|| obj.get("file_path"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let old = obj
                .get("oldString")
                .or_else(|| obj.get("old_string"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let new = obj
                .get("newString")
                .or_else(|| obj.get("new_string"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let cached_lines = build_diff_lines(&old, &new, None);
            ToolDetail::Edit {
                file,
                old,
                new,
                start_line: None,
                cached_lines,
            }
        }
        "multiedit" => {
            let file = obj
                .get("filePath")
                .or_else(|| obj.get("file_path"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let count = obj
                .get("edits")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            ToolDetail::Summary(format!("{} ({} edits)", short(file), count))
        }
        "write_file" => {
            let path = str_field("path");
            let content = str_field("content");
            let cached_lines = build_write_lines(&content);
            ToolDetail::WriteFile {
                path,
                content,
                cached_lines,
            }
        }
        "read_tool" => {
            let path = str_field("path");
            let offset = obj.get("offset").and_then(|v| v.as_u64());
            let limit = obj.get("limit").and_then(|v| v.as_u64());
            let range = match (offset, limit) {
                (Some(o), Some(l)) => format!(":{}-{}", o, o + l),
                (Some(o), None) => format!(":{}", o),
                _ => String::new(),
            };
            ToolDetail::Summary(format!("{}{range}", short(&path)))
        }
        "shell" => {
            let cmd = str_field("command");
            let display = if cmd.len() > 60 {
                format!("{}{ELLIPSIS}", &cmd[..60])
            } else {
                cmd
            };
            ToolDetail::Summary(display)
        }
        "search_text" => {
            let pattern = str_field("pattern");
            let path = str_field("path");
            let include = str_field("include");
            let location = if !include.is_empty() {
                include
            } else if !path.is_empty() {
                short(&path).to_string()
            } else {
                ".".into()
            };
            ToolDetail::Summary(format!("\"{}\" {}", pattern, location))
        }
        "glob" => {
            let pattern = str_field("pattern");
            let path = str_field("path");
            if path.is_empty() {
                ToolDetail::Summary(pattern)
            } else {
                ToolDetail::Summary(format!("{} in {}", pattern, short(&path)))
            }
        }
        "ls" => {
            let path = str_field("path");
            ToolDetail::Summary(if path.is_empty() {
                ".".into()
            } else {
                short(&path).to_string()
            })
        }
        "delete_file" => {
            let path = str_field("path");
            ToolDetail::Summary(short(&path).to_string())
        }
        "browse" | "web_fetch" => {
            let url = str_field("url");
            let display = if url.len() > 60 {
                format!("{}{ELLIPSIS}", &url[..60])
            } else {
                url
            };
            ToolDetail::Summary(display)
        }
        "apply_patch" => ToolDetail::Summary("patch".into()),
        "delegate" => {
            let objective = str_field("objective");
            let display = if objective.len() > 50 {
                format!("{}{ELLIPSIS}", &objective[..50])
            } else {
                objective
            };
            ToolDetail::Summary(display)
        }
        "language_query" => {
            let action = str_field("action");
            let uri = str_field("uri");
            ToolDetail::Summary(format!("{} {}", action, short(&uri)))
        }
        "question" => ToolDetail::Summary("asking...".into()),
        "todowrite" => {
            if let Some(todos) = obj.get("todos").and_then(|v| v.as_array()) {
                let lines: Vec<String> = todos
                    .iter()
                    .filter_map(|t| {
                        let content = t.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let status = t
                            .get("status")
                            .and_then(|v| v.as_str())
                            .unwrap_or("pending");
                        if content.is_empty() {
                            return None;
                        }
                        let check = if status == "completed" { "x" } else { " " };
                        Some(format!("[{check}] {content}"))
                    })
                    .collect();
                if lines.is_empty() {
                    ToolDetail::None
                } else {
                    ToolDetail::Summary(lines.join("\n"))
                }
            } else {
                ToolDetail::None
            }
        }
        _ => ToolDetail::None,
    }
}

fn update_tool_detail(messages: &mut [ChatEntry], tool_call_id: Option<&str>, result: &str) {
    let Some(id) = tool_call_id else { return };
    // parse result JSON
    let obj: serde_json::Value = match serde_json::from_str(result) {
        Ok(v) => v,
        Err(_) => return,
    };

    // walk backwards to find matching ToolCall
    for entry in messages.iter_mut().rev() {
        if let ChatEntry::ToolCall {
            tool_call_id: Some(tid),
            name,
            detail,
            ..
        } = entry
        {
            if tid != id {
                continue;
            }
            // edit tool: update start_line
            if let ToolDetail::Edit { start_line: sl, .. } = detail {
                *sl = obj
                    .get("startLineOld")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
            }
            // shell tool: show last 3 lines of stdout below command
            if name.starts_with("shell")
                && let Some(stdout) = obj.get("stdout").and_then(|v| v.as_str())
            {
                let tail: Vec<&str> = stdout
                    .lines()
                    .rev()
                    .filter(|l| !l.trim().is_empty())
                    .take(3)
                    .collect();
                if !tail.is_empty() {
                    let tail_str = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
                    if let ToolDetail::Summary(header) = detail {
                        *detail = ToolDetail::SummaryWithOutput {
                            header: std::mem::take(header),
                            output: tail_str,
                        };
                    }
                }
            }
            break;
        }
    }
}

fn content_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    block.get("text").and_then(|t| t.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

// ── scroll_tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod scroll_tests {
    use super::*;
    use crate::protocol::EventKind;

    #[test]
    fn content_delta_preserves_scroll_when_scrolled_up() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false);
        app.scroll_offset = 20;

        app.handle_event_kind(
            &EventKind::AssistantContentDelta {
                content: "hello".into(),
                message_id: None,
            },
            false,
        );

        assert_eq!(
            app.scroll_offset, 20,
            "scroll_offset should be preserved when user is scrolled up"
        );
    }

    #[test]
    fn scroll_compensation_bumps_offset_by_growth() {
        let mut app = App::new();
        app.scroll_offset = 30;
        app.prev_total_height = 100;

        // Content grew by 5 rows
        app.compensate_scroll_for_growth(105);

        assert_eq!(
            app.scroll_offset, 35,
            "scroll_offset should increase by growth to keep viewport stable"
        );
        assert_eq!(app.prev_total_height, 105);
    }

    #[test]
    fn scroll_compensation_noop_when_at_bottom() {
        let mut app = App::new();
        app.scroll_offset = 0; // following
        app.prev_total_height = 100;

        app.compensate_scroll_for_growth(110);

        assert_eq!(
            app.scroll_offset, 0,
            "scroll_offset should stay 0 when auto-following"
        );
        assert_eq!(app.prev_total_height, 110);
    }

    #[test]
    fn scroll_compensation_noop_when_no_growth() {
        let mut app = App::new();
        app.scroll_offset = 20;
        app.prev_total_height = 100;

        app.compensate_scroll_for_growth(100);

        assert_eq!(app.scroll_offset, 20);
        assert_eq!(app.prev_total_height, 100);
    }

    #[test]
    fn content_delta_stays_at_bottom_when_following() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false);
        app.scroll_offset = 0; // at bottom

        app.handle_event_kind(
            &EventKind::AssistantContentDelta {
                content: "hello".into(),
                message_id: None,
            },
            false,
        );

        assert_eq!(
            app.scroll_offset, 0,
            "scroll_offset should remain 0 when user is at the bottom"
        );
    }
}
