//! Server message handling for the TUI application.
//!
//! Contains `handle_server_msg`, `handle_event_kind`, `replay_audit`, and
//! helper functions for parsing tool details, updating tool results, and
//! building diff/write content lines.

use std::collections::{HashMap, HashSet};

use crate::app::*;
use crate::protocol::*;
use crate::ui::{ELLIPSIS, OUTCOME_BULLET, build_diff_lines, build_write_lines};

#[derive(Debug, Clone)]
pub(crate) struct PendingElicitationSnapshot {
    elicitation_id: String,
    message: String,
    requested_schema: serde_json::Value,
    source: String,
}

impl App {
    fn parse_delegate_tool_call(
        tool_call_id: &Option<String>,
        arguments: Option<&serde_json::Value>,
    ) -> Option<PendingDelegateToolCall> {
        let tool_call_id = tool_call_id.clone()?;
        let args = arguments?;
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
        Some(PendingDelegateToolCall {
            tool_call_id,
            target_agent_id: {
                let agent = str_field("target_agent_id");
                (!agent.is_empty()).then_some(agent)
            },
            objective: str_field("objective"),
        })
    }

    fn find_pending_delegate_tool_call(
        &self,
        target_agent_id: Option<&str>,
        objective: Option<&str>,
    ) -> Option<usize> {
        self.pending_delegate_tool_calls.iter().position(|pending| {
            target_agent_id.is_none_or(|agent| pending.target_agent_id.as_deref() == Some(agent))
                && objective.is_none_or(|obj| pending.objective == obj)
        })
    }

    fn child_delegate_entry_index(&self, session_id: &str) -> Option<usize> {
        self.delegate_entries
            .iter()
            .position(|e| e.child_session_id.as_deref() == Some(session_id))
    }

    fn unlinked_delegate_entry_index_for_agent(&self, agent_id: Option<&str>) -> Option<usize> {
        let agent_id = agent_id?;
        let mut matches = self
            .delegate_entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.status == DelegateStatus::InProgress
                    && e.child_session_id.is_none()
                    && e.target_agent_id.as_deref() == Some(agent_id)
            })
            .map(|(idx, _)| idx);
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }

    fn apply_pending_child_state(&mut self, session_id: &str) {
        let Some(state) = self.pending_delegate_child_states.remove(session_id) else {
            return;
        };
        if let Some(idx) = self.child_delegate_entry_index(session_id) {
            if self.delegate_entries[idx].child_state != state {
                self.delegate_entries[idx].child_state = state;
                self.invalidate_delegate_render_cache();
            }
        }
    }

    fn current_delegate_pending_elicitation(
        &self,
    ) -> Option<(&str, &str, &serde_json::Value, &str)> {
        let session_id = self.session_id.as_deref()?;
        self.delegate_entries.iter().find_map(|entry| {
            if entry.child_session_id.as_deref() != Some(session_id) {
                return None;
            }
            match &entry.child_state {
                DelegateChildState::PendingElicitation {
                    elicitation_id,
                    message,
                    requested_schema,
                    source,
                } => Some((
                    elicitation_id.as_str(),
                    message.as_str(),
                    requested_schema,
                    source.as_str(),
                )),
                _ => None,
            }
        })
    }

    fn current_delegate_pending_elicitation_snapshot(&self) -> Option<PendingElicitationSnapshot> {
        self.current_delegate_pending_elicitation().map(
            |(elicitation_id, message, requested_schema, source)| PendingElicitationSnapshot {
                elicitation_id: elicitation_id.to_string(),
                message: message.to_string(),
                requested_schema: requested_schema.clone(),
                source: source.to_string(),
            },
        )
    }

    fn remove_elicitation_cards(&mut self, elicitation_id: &str) {
        let before = self.messages.len();
        self.messages.retain(|entry| {
            !matches!(
                entry,
                ChatEntry::Elicitation {
                    elicitation_id: existing_id,
                    ..
                } if existing_id == elicitation_id
            )
        });
        if self.messages.len() != before {
            self.card_cache.invalidate();
        }
    }

    fn reopen_pending_elicitation(&mut self, pending: &PendingElicitationSnapshot) {
        self.remove_elicitation_cards(&pending.elicitation_id);
        self.handle_elicitation_requested(
            &pending.elicitation_id,
            &pending.message,
            &pending.source,
            &pending.requested_schema,
        );
    }

    fn handle_elicitation_requested(
        &mut self,
        elicitation_id: &str,
        message: &str,
        source: &str,
        requested_schema: &serde_json::Value,
    ) {
        let fields = ElicitationState::parse_schema(requested_schema);
        if fields.is_empty() {
            let outcome = "unsupported schema - cannot answer in TUI";
            self.messages.push(ChatEntry::Elicitation {
                elicitation_id: elicitation_id.to_string(),
                message: message.to_string(),
                source: source.to_string(),
                outcome: Some(outcome.into()),
            });
            self.scroll_offset = 0;
            self.set_status(
                LogLevel::Warn,
                "elicitation",
                "question skipped - unsupported schema",
            );
        } else {
            self.elicitation = Some(ElicitationState {
                elicitation_id: elicitation_id.to_string(),
                message: message.to_string(),
                source: source.to_string(),
                fields,
                field_cursor: 0,
                option_cursor: 0,
                selected: HashMap::new(),
                text_input: String::new(),
                text_cursor: 0,
            });
            self.messages.push(ChatEntry::Elicitation {
                elicitation_id: elicitation_id.to_string(),
                message: message.to_string(),
                source: source.to_string(),
                outcome: None,
            });
            self.scroll_offset = 0;
            self.set_status(
                LogLevel::Info,
                "elicitation",
                "question - answer in the panel above input",
            );
        }
    }

    fn link_delegate_child_session(
        &mut self,
        child_session_id: &str,
        delegation_id: Option<&str>,
        target_agent_id: Option<&str>,
    ) -> bool {
        let idx = delegation_id
            .and_then(|id| {
                self.delegate_entries
                    .iter()
                    .position(|e| e.delegation_id == id)
            })
            .or_else(|| self.unlinked_delegate_entry_index_for_agent(target_agent_id));

        let Some(idx) = idx else {
            return false;
        };

        if self.delegate_entries[idx].child_session_id.as_deref() != Some(child_session_id) {
            self.delegate_entries[idx].child_session_id = Some(child_session_id.to_string());
            self.invalidate_delegate_render_cache();
        }
        self.apply_pending_child_state(child_session_id);
        true
    }

    fn apply_delegate_child_event(
        &mut self,
        session_id: &str,
        agent_id: Option<&str>,
        kind: &EventKind,
    ) -> bool {
        let idx = self
            .child_delegate_entry_index(session_id)
            .or_else(|| self.unlinked_delegate_entry_index_for_agent(agent_id));

        let Some(idx) = idx else {
            let mut state = self
                .pending_delegate_child_states
                .remove(session_id)
                .unwrap_or_default();
            update_delegate_child_state(&mut state, kind);
            if state != DelegateChildState::None {
                self.pending_delegate_child_states
                    .insert(session_id.to_string(), state);
            }
            return false;
        };

        let before_stats = self.delegate_entries[idx].stats.clone();
        let before_state = self.delegate_entries[idx].child_state.clone();
        if self.delegate_entries[idx].child_session_id.is_none() {
            self.delegate_entries[idx].child_session_id = Some(session_id.to_string());
        }
        accumulate_delegate_stats(&mut self.delegate_entries[idx].stats, kind);
        update_delegate_child_state(&mut self.delegate_entries[idx].child_state, kind);
        if self.delegate_entries[idx].stats != before_stats
            || self.delegate_entries[idx].child_state != before_state
        {
            self.invalidate_delegate_render_cache();
        }
        true
    }

    pub fn handle_server_msg(&mut self, raw: RawServerMsg) -> Vec<ClientMsg> {
        match raw.msg_type.as_str() {
            "state" => {
                if let Some(data) = raw.data
                    && let Ok(state) = serde_json::from_value::<StateData>(data)
                {
                    self.agent_id = state.agents.first().map(|a| a.id.clone());
                    self.agents = state.agents;
                    if let Some(mode) = state.agent_mode {
                        self.agent_mode = mode;
                        if self.agent_mode != "review" {
                            self.mode_before_review = None;
                        }
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
                    && let Some(validated) =
                        validate_reasoning_effort(re.reasoning_effort.as_deref())
                {
                    self.reasoning_effort = validated;
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
                    if self.agent_mode != "review" {
                        self.mode_before_review = None;
                    }
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
                    self.card_cache.invalidate();
                    self.scroll_offset = 0;
                    self.undo_state = None;
                    self.undoable_turns.clear();
                    self.recent_prompt_text = None;
                    self.suppress_turn_output = false;
                    self.delegate_entries.clear();
                    self.pending_delegate_child_states.clear();
                    self.pending_delegate_tool_calls.clear();
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
                    self.mode_before_review = None;
                    self.cumulative_cost = None;
                    self.session_stats = SessionStatsLite::default();
                    self.screen = Screen::Chat;
                    self.set_status(LogLevel::Info, "session", "session created");
                    let mut cmds = vec![ClientMsg::SubscribeSession {
                        session_id: sc.session_id,
                        agent_id: self.agent_id.clone(),
                    }];
                    // Auto-apply mode model preference for the initial mode.
                    if let Some((provider, model)) =
                        self.get_mode_model_preference(&self.agent_mode)
                    {
                        let provider = provider.to_string();
                        let model = model.to_string();
                        if let Some(entry) = self
                            .models
                            .iter()
                            .find(|m| {
                                m.provider == provider && m.model == model && m.node_id.is_none()
                            })
                            .cloned()
                        {
                            self.current_provider = Some(entry.provider.clone());
                            self.current_model = Some(entry.model.clone());
                            if let Some(sid) = self.session_id.clone() {
                                cmds.push(ClientMsg::SetSessionModel {
                                    session_id: sid,
                                    model_id: entry.id,
                                    node_id: entry.node_id,
                                });
                            }
                        }
                    }
                    return cmds;
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
                            self.card_cache.invalidate();
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
                                self.pending_delegate_child_states.clear();
                                self.pending_delegate_tool_calls.clear();
                            }
                            self.file_index.clear();
                            self.file_index_generated_at = None;
                            self.file_index_loading = false;
                            self.file_index_error = None;
                            self.mention_state = None;
                            self.last_compaction_token_estimate = None;
                            self.elicitation = None;
                            self.clear_cancel_confirm();
                            self.mode_before_review = None;
                            self.undo_state =
                                self.build_undo_state_from_server_stack(&sl.undo_stack, None, None);
                            self.set_status(LogLevel::Debug, "activity", "ready");
                            let parent_pending =
                                self.current_delegate_pending_elicitation_snapshot();
                            let audit_pending =
                                pending_elicitation_from_audit(&sl.audit, parent_pending.as_ref());
                            let session_loaded_pending_elicitation =
                                parent_pending.or(audit_pending);

                            // Replay audit: sets current_provider/model (ProviderChanged)
                            // and agent_mode (SessionModeChanged).
                            self.replay_audit(
                                &sl.audit,
                                session_loaded_pending_elicitation.as_ref(),
                            );

                            if let Some(pending) = session_loaded_pending_elicitation.as_ref()
                                && !matches!(
                                    self.elicitation.as_ref(),
                                    Some(active) if active.elicitation_id == pending.elicitation_id
                                )
                            {
                                self.reopen_pending_elicitation(pending);
                            }

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
                                self.handle_event_with_replay(&envelope, true);
                            }
                        } else {
                            let mut routed = false;
                            for envelope in &parsed.events {
                                routed |= self.apply_delegate_child_event(
                                    &parsed.session_id,
                                    Some(&parsed.agent_id),
                                    envelope.kind(),
                                );
                            }
                            if !routed {
                                self.note_session_activity(&parsed.session_id);
                            }
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
                                let is_current =
                                    self.session_id.as_deref() == Some(se.session_id.as_str());
                                if is_current {
                                    self.note_session_activity(&se.session_id);
                                    for val in se.events {
                                        self.handle_event_value_with_replay(&val, true, true);
                                    }
                                } else {
                                    let mut unknown_kinds = Vec::new();
                                    let mut routed = false;
                                    for val in se.events {
                                        match serde_json::from_value::<EventEnvelope>(val.clone()) {
                                            Ok(envelope) => {
                                                routed |= self.apply_delegate_child_event(
                                                    &se.session_id,
                                                    Some(&se.agent_id),
                                                    envelope.kind(),
                                                );
                                            }
                                            Err(_) => {
                                                if let Some(kind_type) =
                                                    extract_event_kind_type(&val)
                                                {
                                                    unknown_kinds.push(kind_type.to_string());
                                                }
                                            }
                                        }
                                    }
                                    for kind_type in unknown_kinds {
                                        self.warn_unknown_event_kind(&kind_type, true);
                                    }
                                    if !routed {
                                        self.note_session_activity(&se.session_id);
                                    }
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
                        } else if !self.apply_delegate_child_event(
                            &parsed.session_id,
                            Some(&parsed.agent_id),
                            parsed.event.kind(),
                        ) {
                            self.note_session_activity(&parsed.session_id);
                        }
                    } else if let Ok(ed) = serde_json::from_value::<EventDataRaw>(data) {
                        let is_current = self.session_id.as_deref() == Some(ed.session_id.as_str());
                        if is_current {
                            self.note_session_activity(&ed.session_id);
                            self.handle_event_value(&ed.event, false);
                        } else {
                            match serde_json::from_value::<EventEnvelope>(ed.event.clone()) {
                                Ok(envelope) => {
                                    if !self.apply_delegate_child_event(
                                        &ed.session_id,
                                        Some(&ed.agent_id),
                                        envelope.kind(),
                                    ) {
                                        self.note_session_activity(&ed.session_id);
                                    }
                                }
                                Err(_) => {
                                    if let Some(kind_type) = extract_event_kind_type(&ed.event) {
                                        self.warn_unknown_event_kind(kind_type, true);
                                    }
                                    self.note_session_activity(&ed.session_id);
                                }
                            }
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
                    push_error_message(&mut self.messages, &e.message);
                    self.set_status(LogLevel::Error, "server", format!("error: {}", e.message));
                }
                vec![]
            }
            _ => {
                self.push_log(
                    LogLevel::Warn,
                    "protocol",
                    format!("unknown server message type: {}", raw.msg_type),
                );
                vec![]
            }
        }
    }

    fn handle_event(&mut self, envelope: &EventEnvelope) {
        self.handle_event_with_replay(envelope, false);
    }

    fn handle_event_with_replay(&mut self, envelope: &EventEnvelope, is_replay: bool) {
        self.apply_event_stats(envelope.kind(), envelope.timestamp());
        self.handle_event_kind(envelope.kind(), is_replay, envelope.timestamp());
    }

    fn handle_event_value(&mut self, value: &serde_json::Value, is_batch: bool) {
        self.handle_event_value_with_replay(value, false, is_batch);
    }

    fn handle_event_value_with_replay(
        &mut self,
        value: &serde_json::Value,
        is_replay: bool,
        is_batch: bool,
    ) {
        match serde_json::from_value::<EventEnvelope>(value.clone()) {
            Ok(envelope) => self.handle_event_with_replay(&envelope, is_replay),
            Err(_) => {
                if let Some(kind_type) = extract_event_kind_type(value) {
                    self.warn_unknown_event_kind(kind_type, is_batch);
                }
            }
        }
    }

    fn warn_unknown_event_kind(&mut self, kind_type: &str, is_batch: bool) {
        let source = if is_batch { "session_events" } else { "event" };
        self.push_log(
            LogLevel::Warn,
            "protocol",
            format!("unknown {source} kind: {kind_type}"),
        );
    }

    pub(crate) fn handle_event_kind(
        &mut self,
        kind: &EventKind,
        is_replay: bool,
        timestamp: Option<i64>,
    ) {
        self.handle_event_kind_with_pending(kind, is_replay, timestamp, None);
    }

    fn handle_event_kind_with_pending(
        &mut self,
        kind: &EventKind,
        is_replay: bool,
        timestamp: Option<i64>,
        pending_elicitation: Option<&PendingElicitationSnapshot>,
    ) {
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
            EventKind::LlmRequestStart { .. } => {}
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
                    if is_replay
                        && let Some(tool_call_id) = tool_call_id.as_deref()
                        && self.messages.iter().any(|entry| {
                            matches!(
                                entry,
                                ChatEntry::ToolCall {
                                    tool_call_id: Some(existing_id),
                                    is_error: false,
                                    ..
                                } if existing_id == tool_call_id
                            )
                        })
                    {
                        return;
                    }
                    if tool_name == "delegate"
                        && let Some(pending) =
                            Self::parse_delegate_tool_call(tool_call_id, arguments.as_ref())
                        && !self
                            .pending_delegate_tool_calls
                            .iter()
                            .any(|existing| existing.tool_call_id == pending.tool_call_id)
                    {
                        self.delegate_entries.push(DelegateEntry {
                            delegation_id: pending.tool_call_id.clone(),
                            child_session_id: None,
                            delegate_tool_call_id: Some(pending.tool_call_id.clone()),
                            target_agent_id: pending.target_agent_id.clone(),
                            objective: pending.objective.clone(),
                            status: DelegateStatus::InProgress,
                            stats: DelegateStats::default(),
                            started_at: timestamp,
                            ended_at: None,
                            child_state: DelegateChildState::None,
                        });
                        self.pending_delegate_tool_calls.push(pending);
                        self.invalidate_delegate_render_cache();
                    }
                    let detail = parse_tool_detail(tool_name, arguments.as_ref());
                    if reconcile_tool_call_start(
                        &mut self.messages,
                        tool_call_id.as_deref(),
                        tool_name,
                        detail.clone(),
                    ) {
                        self.card_cache.invalidate();
                        return;
                    }
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
                    let mut updated_existing_tool = false;
                    if let Some(result_str) = result {
                        updated_existing_tool = update_tool_detail(
                            &mut self.messages,
                            tool_call_id.as_deref(),
                            result_str,
                        );
                    }
                    if is_error.unwrap_or(false) {
                        if mark_tool_call_failed(
                            &mut self.messages,
                            tool_call_id.as_deref(),
                            tool_name,
                        ) {
                            updated_existing_tool = true;
                        } else if !failed_tool_call_exists(
                            &self.messages,
                            tool_call_id.as_deref(),
                            tool_name,
                        ) {
                            self.messages.push(ChatEntry::ToolCall {
                                tool_call_id: tool_call_id.clone(),
                                name: format!("{tool_name} (failed)"),
                                is_error: true,
                                detail: ToolDetail::None,
                            });
                        }
                    }
                    if updated_existing_tool {
                        self.card_cache.invalidate();
                    }
                }
            }
            EventKind::SnapshotStart { policy } => {
                if !is_replay {
                    self.set_status(
                        LogLevel::Debug,
                        "snapshot",
                        format!("starting {policy} snapshot"),
                    );
                }
            }
            EventKind::SnapshotEnd { summary } => {
                if !is_replay {
                    let summary = summary.as_deref().unwrap_or("completed");
                    let level = if summary.eq_ignore_ascii_case("no changes") {
                        LogLevel::Debug
                    } else {
                        LogLevel::Info
                    };
                    self.set_status(level, "snapshot", format!("snapshot: {summary}"));
                }
            }
            EventKind::ProgressRecorded { progress_entry } => {
                if !is_replay {
                    match &progress_entry.kind {
                        ProgressKind::ToolCall => {}
                        ProgressKind::Artifact => self.push_log(
                            LogLevel::Debug,
                            "progress",
                            format!("artifact: {}", progress_entry.content),
                        ),
                        ProgressKind::Note => self.push_log(
                            LogLevel::Debug,
                            "progress",
                            format!("progress: {}", progress_entry.content),
                        ),
                        ProgressKind::Checkpoint => self.push_log(
                            LogLevel::Debug,
                            "progress",
                            format!("checkpoint: {}", progress_entry.content),
                        ),
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
                push_error_message(&mut self.messages, message);
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
                    let replay_pending = pending_elicitation
                        .filter(|pending| pending.elicitation_id == *elicitation_id)
                        .cloned()
                        .or_else(|| {
                            self.current_delegate_pending_elicitation_snapshot()
                                .filter(|pending| pending.elicitation_id == *elicitation_id)
                        });
                    if let Some(pending) = replay_pending {
                        self.reopen_pending_elicitation(&pending);
                        return;
                    }

                    // Replay can include the same resolved elicitation more than once
                    // (session_loaded audit + current session_events history).
                    if self.messages.iter().any(|entry| {
                        matches!(
                            entry,
                            ChatEntry::Elicitation {
                                elicitation_id: existing_id,
                                ..
                            } if existing_id == elicitation_id
                        )
                    }) {
                        return;
                    }
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
                self.handle_elicitation_requested(
                    elicitation_id,
                    message,
                    source,
                    requested_schema,
                );
            }
            EventKind::SessionModeChanged { mode } => {
                self.agent_mode = mode.clone();
                if mode != "review" {
                    self.mode_before_review = None;
                }
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
            EventKind::SessionCreated => {}
            // ── Delegation lifecycle events ─────────────────────────────────────
            EventKind::DelegationRequested { delegation } => {
                let objective = delegation.objective.clone().unwrap_or_default();
                let pending_idx = self.find_pending_delegate_tool_call(
                    delegation.target_agent_id.as_deref(),
                    delegation.objective.as_deref(),
                );
                let pending = pending_idx.map(|idx| self.pending_delegate_tool_calls.remove(idx));
                let mut changed = false;

                if let Some(idx) = self
                    .delegate_entries
                    .iter()
                    .position(|e| e.delegation_id == delegation.public_id)
                {
                    let entry = &mut self.delegate_entries[idx];
                    if entry.target_agent_id.is_none() && delegation.target_agent_id.is_some() {
                        entry.target_agent_id = delegation.target_agent_id.clone();
                        changed = true;
                    }
                    if entry.objective.is_empty() && !objective.is_empty() {
                        entry.objective = objective.clone();
                        changed = true;
                    }
                    if let Some(pending) = pending.as_ref()
                        && entry.delegate_tool_call_id.is_none()
                    {
                        entry.delegate_tool_call_id = Some(pending.tool_call_id.clone());
                        changed = true;
                    }
                } else if let Some(pending) = pending {
                    if let Some(idx) = self.delegate_entries.iter().position(|e| {
                        e.delegate_tool_call_id.as_deref() == Some(pending.tool_call_id.as_str())
                    }) {
                        let entry = &mut self.delegate_entries[idx];
                        entry.delegation_id = delegation.public_id.clone();
                        if entry.target_agent_id.is_none() {
                            entry.target_agent_id = delegation.target_agent_id.clone();
                        }
                        if entry.objective.is_empty() {
                            entry.objective = objective.clone();
                        }
                    } else {
                        self.delegate_entries.push(DelegateEntry {
                            delegation_id: delegation.public_id.clone(),
                            child_session_id: None,
                            delegate_tool_call_id: Some(pending.tool_call_id),
                            target_agent_id: delegation.target_agent_id.clone(),
                            objective,
                            status: DelegateStatus::InProgress,
                            stats: DelegateStats::default(),
                            started_at: timestamp,
                            ended_at: None,
                            child_state: DelegateChildState::None,
                        });
                    }
                    changed = true;
                } else {
                    self.delegate_entries.push(DelegateEntry {
                        delegation_id: delegation.public_id.clone(),
                        child_session_id: None,
                        delegate_tool_call_id: None,
                        target_agent_id: delegation.target_agent_id.clone(),
                        objective,
                        status: DelegateStatus::InProgress,
                        stats: DelegateStats::default(),
                        started_at: timestamp,
                        ended_at: None,
                        child_state: DelegateChildState::None,
                    });
                    changed = true;
                }
                if changed {
                    self.invalidate_delegate_render_cache();
                }
            }
            EventKind::SessionForked {
                child_session_id,
                origin,
                fork_point_ref,
                target_agent_id,
            } => {
                if origin.as_deref() == Some("delegation")
                    && let Some(sid) = child_session_id
                {
                    // Prefer the explicit delegation id, but fall back to a
                    // single unlinked in-progress delegate for this target agent.
                    self.link_delegate_child_session(
                        sid,
                        fork_point_ref.as_deref(),
                        target_agent_id.as_deref(),
                    );
                    // Subscribe to the child session using the delegation's
                    // target agent_id — matching the web UI behaviour.
                    // Fall back to parent agent_id if not present.
                    let agent_id = target_agent_id.clone().or_else(|| self.agent_id.clone());
                    self.pending_commands.push(ClientMsg::SubscribeSession {
                        session_id: sid.clone(),
                        agent_id,
                    });
                    // Auto-apply delegate model preference if configured.
                    if let Some(target_id) = target_agent_id.as_deref()
                        && let Some((prov, mdl)) = self.get_delegate_model_preference(target_id)
                    {
                        let prov = prov.to_string();
                        let mdl = mdl.to_string();
                        if let Some(entry) = self
                            .models
                            .iter()
                            .find(|m| m.provider == prov && m.model == mdl && m.node_id.is_none())
                            .cloned()
                        {
                            self.pending_commands.push(ClientMsg::SetSessionModel {
                                session_id: sid.clone(),
                                model_id: entry.id,
                                node_id: entry.node_id,
                            });
                        }
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
                    entry.ended_at = timestamp;
                    entry.child_state = DelegateChildState::None;
                    self.invalidate_delegate_render_cache();
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
                    entry.ended_at = timestamp;
                    entry.child_state = DelegateChildState::None;
                    self.invalidate_delegate_render_cache();
                }
                self.suppress_delegation_result = true;
            }
            EventKind::DelegationCancelled { delegation_id } => {
                if let Some(entry) = self
                    .delegate_entries
                    .iter_mut()
                    .find(|e| e.delegation_id == *delegation_id)
                {
                    entry.status = DelegateStatus::Cancelled;
                    entry.ended_at = timestamp;
                    entry.child_state = DelegateChildState::None;
                    self.invalidate_delegate_render_cache();
                }
                self.suppress_delegation_result = true;
            }
            EventKind::Unknown => {
                self.warn_unknown_event_kind("unknown", false);
            }
        }
    }

    /// Drain any commands queued by event handlers (e.g. SubscribeSession
    /// for delegation child sessions) and return them to the caller.
    fn drain_pending(&mut self) -> Vec<ClientMsg> {
        std::mem::take(&mut self.pending_commands)
    }

    pub(crate) fn replay_audit(
        &mut self,
        audit: &serde_json::Value,
        pending_elicitation: Option<&PendingElicitationSnapshot>,
    ) {
        if let Some(events) = audit.get("events").and_then(|e| e.as_array()) {
            let frontier_message_id = self
                .undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref());
            let mut replay_cutoff = events.len();

            if let Some(frontier_message_id) = frontier_message_id
                && let Some(idx) = events.iter().position(|event_val| {
                    parse_audit_event(event_val)
                        .and_then(|(kind, _)| match kind {
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
                if let Some((kind, timestamp)) = parse_audit_event(event_val) {
                    self.apply_event_stats(&kind, timestamp);
                    self.handle_event_kind_with_pending(
                        &kind,
                        true,
                        timestamp,
                        pending_elicitation,
                    );
                }
            }
        }
    }
}

fn parse_audit_event(value: &serde_json::Value) -> Option<(EventKind, Option<i64>)> {
    if let Ok(agent_event) = serde_json::from_value::<AgentEvent>(value.clone()) {
        return Some((agent_event.kind, agent_event.timestamp));
    }
    serde_json::from_value::<EventEnvelope>(value.clone())
        .ok()
        .map(|envelope| (envelope.kind().clone(), envelope.timestamp()))
}

fn pending_elicitation_from_audit(
    audit: &serde_json::Value,
    parent_pending: Option<&PendingElicitationSnapshot>,
) -> Option<PendingElicitationSnapshot> {
    let events = audit.get("events").and_then(|e| e.as_array())?;
    let parent_pending_id = parent_pending.map(|pending| pending.elicitation_id.as_str());
    let mut latest: Option<PendingElicitationSnapshot> = None;
    let mut answered = HashSet::new();

    for event_val in events {
        let Some((kind, _)) = parse_audit_event(event_val) else {
            continue;
        };
        match kind {
            EventKind::ElicitationRequested {
                elicitation_id,
                message,
                requested_schema,
                source,
                ..
            } => {
                answered.remove(elicitation_id.as_str());
                latest = Some(PendingElicitationSnapshot {
                    elicitation_id,
                    message,
                    requested_schema,
                    source,
                });
            }
            EventKind::ToolCallEnd {
                tool_name, result, ..
            } if tool_name == "question" && result.as_deref().is_some_and(is_answer_result) => {
                if let Some(pending) = latest.as_ref() {
                    answered.insert(pending.elicitation_id.clone());
                }
            }
            _ => {}
        }
    }

    latest.filter(|pending| {
        parent_pending_id == Some(pending.elicitation_id.as_str())
            || !answered.contains(&pending.elicitation_id)
    })
}

fn is_answer_result(result: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(result) else {
        return false;
    };
    value
        .get("answers")
        .and_then(|answers| answers.as_array())
        .is_some_and(|answers| !answers.is_empty())
}

fn extract_event_kind_type(value: &serde_json::Value) -> Option<&str> {
    value
        .get("data")
        .and_then(|data| data.get("kind"))
        .and_then(|kind| kind.get("type"))
        .and_then(|kind_type| kind_type.as_str())
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
        | EventKind::ProviderChanged { .. }
        | EventKind::Error { .. }
        | EventKind::Cancelled => {
            *state = DelegateChildState::OtherProgress;
        }
        EventKind::TurnStarted
        | EventKind::SessionModeChanged { .. }
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
            let agent = str_field("target_agent_id");
            let objective = str_field("objective");
            let obj_display = if objective.len() > 50 {
                format!("{}{ELLIPSIS}", &objective[..50])
            } else {
                objective
            };
            let display = if agent.is_empty() {
                obj_display
            } else {
                format!("({agent}) {obj_display}")
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
        "index" => {
            let path = str_field("path");
            ToolDetail::Summary(short(&path).to_string())
        }
        _ => ToolDetail::None,
    }
}

fn fallback_failed_tool_name(tool_name: &str) -> String {
    format!("{tool_name} (failed)")
}

fn is_same_tool_call(existing_id: &Option<String>, tool_call_id: Option<&str>) -> bool {
    existing_id.as_deref() == tool_call_id
}

fn push_error_message(messages: &mut Vec<ChatEntry>, message: &str) -> bool {
    if messages
        .iter()
        .any(|entry| matches!(entry, ChatEntry::Error(existing) if existing == message))
    {
        return false;
    }

    messages.push(ChatEntry::Error(message.to_string()));
    true
}

fn failed_tool_call_exists(
    messages: &[ChatEntry],
    tool_call_id: Option<&str>,
    tool_name: &str,
) -> bool {
    let fallback_name = fallback_failed_tool_name(tool_name);
    messages.iter().any(|entry| {
        matches!(
            entry,
            ChatEntry::ToolCall {
                tool_call_id: existing_id,
                name,
                is_error: true,
                ..
            } if is_same_tool_call(existing_id, tool_call_id)
                && (name == tool_name || name == &fallback_name)
        )
    })
}

fn reconcile_tool_call_start(
    messages: &mut [ChatEntry],
    tool_call_id: Option<&str>,
    tool_name: &str,
    start_detail: ToolDetail,
) -> bool {
    let Some(id) = tool_call_id else { return false };
    let fallback_name = fallback_failed_tool_name(tool_name);
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
            if name == &fallback_name {
                *name = tool_name.to_string();
            }
            if matches!(detail, ToolDetail::None) && !matches!(start_detail, ToolDetail::None) {
                *detail = start_detail;
            }
            return true;
        }
    }
    false
}

fn mark_tool_call_failed(
    messages: &mut [ChatEntry],
    tool_call_id: Option<&str>,
    tool_name: &str,
) -> bool {
    let Some(id) = tool_call_id else { return false };
    for entry in messages.iter_mut().rev() {
        if let ChatEntry::ToolCall {
            tool_call_id: Some(tid),
            name,
            is_error,
            ..
        } = entry
        {
            if tid == id && name == tool_name && !*is_error {
                *is_error = true;
                return true;
            }
        }
    }
    false
}

fn update_tool_detail(
    messages: &mut [ChatEntry],
    tool_call_id: Option<&str>,
    result: &str,
) -> bool {
    let Some(id) = tool_call_id else { return false };
    let parsed_result = serde_json::from_str::<serde_json::Value>(result).ok();

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
            if let Some(obj) = &parsed_result {
                if let ToolDetail::Edit { start_line: sl, .. } = detail {
                    *sl = obj
                        .get("startLineOld")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize);
                }
            }
            // shell tool: show last 3 lines of stdout below command
            if name.starts_with("shell")
                && let Some(stdout) = parsed_result
                    .as_ref()
                    .and_then(|obj| obj.get("stdout"))
                    .and_then(|v| v.as_str())
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
            // index tool: enrich with language and section counts
            if name == "index" {
                let text = parsed_result
                    .as_ref()
                    .map(content_to_string)
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| result.to_string());
                if let Some(summary_parts) = index_outline_summary_parts(&text) {
                    if let ToolDetail::Summary(path) = detail {
                        let suffix = format!(" ({})", summary_parts.join(", "));
                        if !path.ends_with(&suffix) {
                            path.push_str(&suffix);
                        }
                    }
                }
            }
            // search_text tool: append backend result counts from the compact footer.
            if name == "search_text" {
                let text = parsed_result
                    .as_ref()
                    .map(content_to_string)
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| result.to_string());
                if let Some(footer) = search_text_footer(&text) {
                    if let ToolDetail::Summary(summary) = detail {
                        let suffix = format!(" {footer}");
                        if !summary.ends_with(&suffix) {
                            summary.push_str(&suffix);
                        }
                    }
                }
            }
            return true;
        }
    }
    false
}

fn search_text_footer(text: &str) -> Option<&str> {
    let footer = text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())?
        .trim();
    if !footer.starts_with('(') || !footer.ends_with(')') {
        return None;
    }

    let inner = &footer[1..footer.len() - 1];
    let parts: Vec<&str> = inner.split(", ").collect();
    if !(parts.len() == 2 || parts.len() == 3) {
        return None;
    }
    if parts.len() == 3 && parts[2] != "truncated" {
        return None;
    }

    let file_label = if parse_counted_label(parts[0], "file")? == 1 {
        "file"
    } else {
        "files"
    };
    let match_label = if parse_counted_label(parts[1], "match")? == 1 {
        "match"
    } else {
        "matches"
    };
    if !parts[0].ends_with(file_label) || !parts[1].ends_with(match_label) {
        return None;
    }

    Some(footer)
}

fn parse_counted_label(text: &str, singular: &str) -> Option<usize> {
    let (count, label) = text.split_once(' ')?;
    let count = count.parse::<usize>().ok()?;
    let plural = if singular.ends_with("ch") {
        format!("{singular}es")
    } else {
        format!("{singular}s")
    };
    (label == singular || label == plural).then_some(count)
}

fn index_outline_summary_parts(text: &str) -> Option<Vec<String>> {
    let lang = text
        .lines()
        .find(|l| l.starts_with("language:"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, value)| value.trim())
        .filter(|value| !value.is_empty());
    let mut counts: Vec<(String, usize)> = Vec::new();
    let mut current_section: Option<String> = None;
    let mut current_count = 0usize;
    for line in text.lines() {
        if line.ends_with(':') && !line.starts_with(' ') && !line.starts_with('\t') {
            if let Some(ref section) = current_section {
                if current_count > 0 {
                    counts.push((section.clone(), current_count));
                }
            }
            let section = line.trim_end_matches(':');
            current_section =
                (!matches!(section, "path" | "language")).then(|| section.to_string());
            current_count = 0;
        } else if line.starts_with("  - ") && current_section.is_some() {
            current_count += 1;
        }
    }
    if let Some(ref section) = current_section {
        if current_count > 0 {
            counts.push((section.clone(), current_count));
        }
    }
    let mut summary_parts: Vec<String> = Vec::new();
    if let Some(l) = lang {
        summary_parts.push(l.to_string());
    }
    for (sec, n) in counts {
        summary_parts.push(format!("{n} {sec}"));
    }
    (!summary_parts.is_empty()).then_some(summary_parts)
}

fn content_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(obj) => {
            if obj.get("type").and_then(|t| t.as_str()) == Some("text") {
                obj.get("text")
                    .and_then(|t| t.as_str())
                    .map(String::from)
                    .unwrap_or_default()
            } else {
                String::new()
            }
        }
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
mod tool_detail_tests {
    use super::{
        failed_tool_call_exists, mark_tool_call_failed, parse_tool_detail, push_error_message,
        reconcile_tool_call_start, update_tool_detail,
    };
    use crate::app::{App, ChatEntry, ToolDetail};
    use crate::protocol::{EventKind, RawServerMsg};

    fn summary_text(entry: &ChatEntry) -> &str {
        match entry {
            ChatEntry::ToolCall {
                detail: ToolDetail::Summary(s),
                ..
            } => s,
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn duplicate_error_replay_does_not_append_stale_card() {
        let mut messages = Vec::new();

        assert!(push_error_message(&mut messages, "LLM streaming error"));
        messages.push(ChatEntry::User {
            text: "continue".into(),
            message_id: Some("user-continue".into()),
        });
        messages.push(ChatEntry::Assistant {
            content: "latest assistant summary".into(),
            thinking: None,
            message_id: Some("assistant-latest".into()),
        });

        assert!(!push_error_message(&mut messages, "LLM streaming error"));
        assert_eq!(messages.len(), 3, "replayed error must not append");
        assert_eq!(
            messages
                .iter()
                .filter(
                    |entry| matches!(entry, ChatEntry::Error(text) if text == "LLM streaming error")
                )
                .count(),
            1
        );
        assert!(matches!(
            messages.last(),
            Some(ChatEntry::Assistant { content, .. }) if content == "latest assistant summary"
        ));
    }

    #[test]
    fn distinct_error_messages_are_preserved() {
        let mut messages = Vec::new();

        assert!(push_error_message(&mut messages, "first error"));
        assert!(push_error_message(&mut messages, "second error"));

        assert_eq!(messages.len(), 2);
        assert!(matches!(&messages[0], ChatEntry::Error(text) if text == "first error"));
        assert!(matches!(&messages[1], ChatEntry::Error(text) if text == "second error"));
    }

    #[test]
    fn event_error_replay_does_not_append_stale_card() {
        let mut app = App::new();
        let error = EventKind::Error {
            message: "LLM streaming error".into(),
        };

        app.handle_event_kind(&error, false, None);
        app.messages.push(ChatEntry::User {
            text: "continue".into(),
            message_id: Some("user-continue".into()),
        });
        app.messages.push(ChatEntry::Assistant {
            content: "latest assistant summary".into(),
            thinking: None,
            message_id: Some("assistant-latest".into()),
        });
        app.handle_event_kind(&error, true, None);

        assert_eq!(app.messages.len(), 3, "replayed error must not append");
        assert_eq!(
            app.messages
                .iter()
                .filter(
                    |entry| matches!(entry, ChatEntry::Error(text) if text == "LLM streaming error")
                )
                .count(),
            1
        );
        assert!(matches!(
            app.messages.last(),
            Some(ChatEntry::Assistant { content, .. }) if content == "latest assistant summary"
        ));
    }

    #[test]
    fn raw_server_error_duplicate_does_not_append() {
        let mut app = App::new();
        let raw = || RawServerMsg {
            msg_type: "error".into(),
            data: Some(serde_json::json!({ "message": "raw server error" })),
        };

        app.handle_server_msg(raw());
        app.messages.push(ChatEntry::Assistant {
            content: "still latest".into(),
            thinking: None,
            message_id: None,
        });
        app.handle_server_msg(raw());

        assert_eq!(app.messages.len(), 2, "duplicate raw error must not append");
        assert!(matches!(
            app.messages.last(),
            Some(ChatEntry::Assistant { content, .. }) if content == "still latest"
        ));
    }

    #[test]
    fn delegate_tool_shows_agent_and_objective() {
        let args = serde_json::json!({
            "target_agent_id": "coder",
            "objective": "List the contents of /tmp"
        });
        let detail = parse_tool_detail("delegate", Some(&args));
        match detail {
            ToolDetail::Summary(s) => {
                assert!(s.contains("coder"), "must contain agent name, got: {s}");
                assert!(
                    s.contains("List the contents"),
                    "must contain objective, got: {s}"
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn delegate_tool_without_agent_shows_objective_only() {
        let args = serde_json::json!({
            "objective": "Do something"
        });
        let detail = parse_tool_detail("delegate", Some(&args));
        match detail {
            ToolDetail::Summary(s) => {
                assert!(
                    s.contains("Do something"),
                    "must contain objective, got: {s}"
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn index_tool_shows_short_path_from_arguments() {
        let args = serde_json::json!({"path": "/home/user/project/src/main.rs"});
        let detail = parse_tool_detail("index", Some(&args));
        match detail {
            ToolDetail::Summary(s) => {
                assert_eq!(s, "src/main.rs");
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn failed_tool_end_marks_existing_tool_in_place() {
        let mut messages = vec![
            ChatEntry::ToolCall {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                is_error: false,
                detail: ToolDetail::Summary("echo ok".into()),
            },
            ChatEntry::Assistant {
                content: "done".into(),
                thinking: None,
                message_id: None,
            },
        ];

        assert!(mark_tool_call_failed(
            &mut messages,
            Some("tool-1"),
            "shell"
        ));

        assert_eq!(messages.len(), 2, "must not append a stale failed badge");
        match &messages[0] {
            ChatEntry::ToolCall { name, is_error, .. } => {
                assert_eq!(name, "shell");
                assert!(*is_error);
            }
            other => panic!("expected ToolCall, got: {other:?}"),
        }
        assert!(matches!(messages[1], ChatEntry::Assistant { .. }));
    }

    #[test]
    fn failed_tool_fallback_duplicate_check_is_idempotent() {
        let messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("missing-start".into()),
            name: "ls (failed)".into(),
            is_error: true,
            detail: ToolDetail::None,
        }];

        assert!(!mark_tool_call_failed(
            &mut messages.clone(),
            Some("missing-start"),
            "ls"
        ));
        assert!(failed_tool_call_exists(
            &messages,
            Some("missing-start"),
            "ls"
        ));
    }

    #[test]
    fn tool_start_reconciles_failed_fallback_in_place() {
        let mut messages = vec![
            ChatEntry::Assistant {
                content: "before".into(),
                thinking: None,
                message_id: None,
            },
            ChatEntry::ToolCall {
                tool_call_id: Some("missing-start".into()),
                name: "shell (failed)".into(),
                is_error: true,
                detail: ToolDetail::None,
            },
        ];
        let detail = parse_tool_detail(
            "shell",
            Some(&serde_json::json!({
                "command": "cargo test tool_detail_tests"
            })),
        );

        assert!(reconcile_tool_call_start(
            &mut messages,
            Some("missing-start"),
            "shell",
            detail
        ));
        assert_eq!(
            messages.len(),
            2,
            "start must not append a second tool entry"
        );
        match &messages[1] {
            ChatEntry::ToolCall {
                name,
                is_error,
                detail,
                ..
            } => {
                assert_eq!(name, "shell");
                assert!(*is_error);
                assert!(
                    matches!(detail, ToolDetail::Summary(cmd) if cmd == "cargo test tool_detail_tests")
                );
            }
            other => panic!("expected ToolCall, got: {other:?}"),
        }
        assert!(failed_tool_call_exists(
            &messages,
            Some("missing-start"),
            "shell"
        ));
    }

    #[test]
    fn index_tool_enriched_with_language_and_counts() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("tc-1".into()),
            name: "index".into(),
            is_error: false,
            detail: ToolDetail::Summary("src/main.rs".into()),
        }];
        let result = "path: /home/user/project/src/main.rs\nlanguage: rust\n\nimports:\n  - use std::io [1]\n  - use std::fs [2]\ntypes:\n  - struct Foo [10-20]\n    - field: i32 [11]\nfunctions:\n  - fn main [30-40]\n  - fn helper [50-60]\n";
        update_tool_detail(&mut messages, Some("tc-1"), result);
        match &messages[0] {
            ChatEntry::ToolCall {
                detail: ToolDetail::Summary(s),
                ..
            } => {
                assert_eq!(
                    s, "src/main.rs (rust, 2 imports, 1 types, 2 functions)",
                    "must summarize raw outline result"
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn index_tool_enrichment_is_idempotent() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("tc-dup".into()),
            name: "index".into(),
            is_error: false,
            detail: ToolDetail::Summary("src/main.rs".into()),
        }];
        let result = "path: /home/user/project/src/main.rs\nlanguage: rust\n\nimports:\n  - use std::io [1]\ntypes:\n  - struct Foo [10-20]\nfunctions:\n  - fn main [30-40]\n";
        update_tool_detail(&mut messages, Some("tc-dup"), result);
        update_tool_detail(&mut messages, Some("tc-dup"), result);
        match &messages[0] {
            ChatEntry::ToolCall {
                detail: ToolDetail::Summary(s),
                ..
            } => {
                assert_eq!(
                    s, "src/main.rs (rust, 1 imports, 1 types, 1 functions)",
                    "must not append duplicate index summary suffixes"
                );
                assert_eq!(
                    s.matches("(rust, 1 imports, 1 types, 1 functions)").count(),
                    1
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn index_tool_handles_json_text_blocks_and_skips_empty_sections() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("tc-2".into()),
            name: "index".into(),
            is_error: false,
            detail: ToolDetail::Summary("src/lib.rs".into()),
        }];
        let result = r#"[{"type":"text","text":"path: /home/user/project/src/lib.rs\nlanguage: python\n\nimports:\n  - os [1]\ntypes:\nfunctions:\n  - fn foo [10-20]\n"}]"#;
        update_tool_detail(&mut messages, Some("tc-2"), result);
        match &messages[0] {
            ChatEntry::ToolCall {
                detail: ToolDetail::Summary(s),
                ..
            } => {
                assert_eq!(
                    s, "src/lib.rs (python, 1 imports, 1 functions)",
                    "must summarize JSON text blocks and skip empty sections"
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn index_tool_graceful_on_malformed_result() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("tc-3".into()),
            name: "index".into(),
            is_error: false,
            detail: ToolDetail::Summary("src/lib.rs".into()),
        }];
        let result = "totally wrong";
        update_tool_detail(&mut messages, Some("tc-3"), result);
        match &messages[0] {
            ChatEntry::ToolCall {
                detail: ToolDetail::Summary(s),
                ..
            } => {
                assert_eq!(
                    s, "src/lib.rs",
                    "must preserve original path on malformed result"
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn search_text_tool_enriched_with_raw_backend_footer() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("search-1".into()),
            name: "search_text".into(),
            is_error: false,
            detail: ToolDetail::Summary("\"needle\" *.rs".into()),
        }];
        let result = "config.rs\n13:needle\n\nmain.rs\n7:needle\n(5 files, 28 matches)";

        update_tool_detail(&mut messages, Some("search-1"), result);

        assert_eq!(
            summary_text(&messages[0]),
            "\"needle\" *.rs (5 files, 28 matches)"
        );
    }

    #[test]
    fn search_text_tool_enriched_with_truncated_footer() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("search-truncated".into()),
            name: "search_text".into(),
            is_error: false,
            detail: ToolDetail::Summary("\"needle\" src".into()),
        }];
        let result = "main.rs\n7:needle\n(1 file, 100 matches, truncated)";

        update_tool_detail(&mut messages, Some("search-truncated"), result);

        assert_eq!(
            summary_text(&messages[0]),
            "\"needle\" src (1 file, 100 matches, truncated)"
        );
    }

    #[test]
    fn search_text_tool_enrichment_is_idempotent() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("search-dup".into()),
            name: "search_text".into(),
            is_error: false,
            detail: ToolDetail::Summary("\"needle\" .".into()),
        }];
        let result = "(0 files, 0 matches)";

        update_tool_detail(&mut messages, Some("search-dup"), result);
        update_tool_detail(&mut messages, Some("search-dup"), result);

        let summary = summary_text(&messages[0]);
        assert_eq!(summary, "\"needle\" . (0 files, 0 matches)");
        assert_eq!(summary.matches("(0 files, 0 matches)").count(), 1);
    }

    #[test]
    fn search_text_tool_enriched_from_json_text_block_footer() {
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("search-json".into()),
            name: "search_text".into(),
            is_error: false,
            detail: ToolDetail::Summary("\"needle\" .".into()),
        }];
        let result = r#"[{"type":"text","text":"main.rs\n7:needle\n(1 file, 1 match)"}]"#;

        update_tool_detail(&mut messages, Some("search-json"), result);

        assert_eq!(summary_text(&messages[0]), "\"needle\" . (1 file, 1 match)");
    }
}

#[cfg(test)]
mod scroll_tests {
    use super::*;
    use crate::protocol::EventKind;

    #[test]
    fn content_delta_preserves_scroll_when_scrolled_up() {
        let mut app = App::new();
        app.handle_event_kind(&EventKind::TurnStarted, false, None);
        app.scroll_offset = 20;

        app.handle_event_kind(
            &EventKind::AssistantContentDelta {
                content: "hello".into(),
                message_id: None,
            },
            false,
            None,
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
        app.handle_event_kind(&EventKind::TurnStarted, false, None);
        app.scroll_offset = 0; // at bottom

        app.handle_event_kind(
            &EventKind::AssistantContentDelta {
                content: "hello".into(),
                message_id: None,
            },
            false,
            None,
        );

        assert_eq!(
            app.scroll_offset, 0,
            "scroll_offset should remain 0 when user is at the bottom"
        );
    }
}
