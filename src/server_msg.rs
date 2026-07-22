//! Server message handling for the TUI application.
//!
//! Legacy QueryMT websocket/server-message reducers.
//!
//! The normal ACP runtime no longer routes through this module. It remains for
//! compatibility tests and as a reference while old event-shaped behavior is
//! ported to native ACP state.

use crate::app::{
    accumulate_delegate_stats, backfill_elicitation_outcomes, update_delegate_child_state, *,
};
use crate::protocol::*;
use crate::tool_detail;
use std::collections::{HashMap, HashSet};

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
        if let Some(idx) = self.child_delegate_entry_index(session_id)
            && self.delegate_entries[idx].child_state != state
        {
            self.delegate_entries[idx].child_state = state;
            self.invalidate_delegate_render_cache();
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

    fn discard_pending_thinking(&mut self) {
        self.streaming_thinking.clear();
        self.streaming_thinking_message_id = None;
        self.streaming_thinking_cache.invalidate();
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
                allow_custom: source == "builtin:question",
                custom_active: false,
                custom_input: String::new(),
                custom_cursor: 0,
                custom_line_width: 1,
                custom_scroll: 0,
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
                    self.profiles = state.profiles;
                    if let Some(profile_id) = state.active_profile_id {
                        self.active_profile_id = Some(profile_id);
                    }
                    if self.profile_cursor >= self.profiles.len() {
                        self.profile_cursor = self.profiles.len().saturating_sub(1);
                    }
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
                                cwd: self.current_session_cwd(),
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
                                cwd: self.current_session_cwd(),
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
            "fork_result" => {
                self.pending_fork_message_id = None;
                if let Some(data) = raw.data
                    && let Ok(fr) = serde_json::from_value::<ForkResultData>(data)
                {
                    if fr.success {
                        if let Some(forked_session_id) = fr.forked_session_id {
                            self.popup = Popup::None;
                            self.set_status(LogLevel::Info, "fork", "forked - loading session");
                            return vec![
                                ClientMsg::LoadSession {
                                    session_id: forked_session_id.clone(),
                                    cwd: self.current_session_cwd(),
                                },
                                ClientMsg::SubscribeSession {
                                    session_id: forked_session_id,
                                    agent_id: self.agent_id.clone(),
                                },
                            ];
                        }
                        self.set_status(
                            LogLevel::Warn,
                            "fork",
                            fr.message
                                .unwrap_or_else(|| "fork succeeded without session id".into()),
                        );
                    } else {
                        self.set_status(
                            LogLevel::Warn,
                            "fork",
                            fr.message.unwrap_or_else(|| "fork failed".into()),
                        );
                    }
                }
                vec![]
            }
            "session_list" => {
                if let Some(data) = raw.data
                    && let Ok(list) = serde_json::from_value::<SessionListData>(data)
                {
                    let response_cwd = if list.groups.len() == 1 {
                        list.groups.first().and_then(|group| group.cwd.clone())
                    } else {
                        None
                    };
                    let is_group_page = response_cwd
                        .as_ref()
                        .map(|cwd| self.pending_session_group_loads.remove(&Some(cwd.clone())))
                        .unwrap_or_else(|| self.pending_session_group_loads.remove(&None));

                    let mut groups: Vec<SessionGroup> = list
                        .groups
                        .into_iter()
                        .filter(|group| is_group_page || !group.sessions.is_empty())
                        .collect();

                    // Sort sessions within each group by updated_at descending.
                    for group in &mut groups {
                        for session in &group.sessions {
                            if let Some(node_id) = session.node_id.as_deref() {
                                self.remember_remote_session_node(&session.session_id, node_id);
                            }
                        }
                        group
                            .sessions
                            .sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                    }

                    if is_group_page {
                        for group in groups {
                            if let Some(existing) = self
                                .session_groups
                                .iter_mut()
                                .find(|existing| existing.cwd == group.cwd)
                            {
                                let mut seen: HashSet<String> = existing
                                    .sessions
                                    .iter()
                                    .map(|session| session.session_id.clone())
                                    .collect();
                                existing.sessions.extend(
                                    group
                                        .sessions
                                        .into_iter()
                                        .filter(|session| seen.insert(session.session_id.clone())),
                                );
                                existing
                                    .sessions
                                    .sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                                existing.latest_activity = existing
                                    .sessions
                                    .first()
                                    .and_then(|session| session.updated_at.clone())
                                    .or(group.latest_activity);
                                existing.total_count = group.total_count;
                                existing.next_cursor = group.next_cursor;
                            } else if !group.sessions.is_empty() {
                                self.session_groups.push(group);
                            }
                        }
                    } else {
                        // Sort groups by their most-recent visible session activity descending.
                        groups.sort_by(|a, b| {
                            let a_latest = a.sessions.first().and_then(|s| s.updated_at.as_deref());
                            let b_latest = b.sessions.first().and_then(|s| s.updated_at.as_deref());
                            b_latest.cmp(&a_latest)
                        });
                        self.session_groups = groups;
                    }

                    self.session_groups.sort_by(|a, b| {
                        let a_latest = a.sessions.first().and_then(|s| s.updated_at.as_deref());
                        let b_latest = b.sessions.first().and_then(|s| s.updated_at.as_deref());
                        b_latest.cmp(&a_latest)
                    });

                    let total: usize = self.session_groups.iter().map(|g| g.sessions.len()).sum();
                    if !is_group_page && let Some(root_total) = list.total_count {
                        self.push_log(
                            LogLevel::Info,
                            "session",
                            format!(
                                "session list: total root sessions={root_total}, loaded={total}"
                            ),
                        );
                    }

                    // Clamp cursor to the new visible item count.
                    let visible_len = self.visible_start_items().len();
                    if self.session_cursor >= visible_len && visible_len > 0 {
                        self.session_cursor = visible_len - 1;
                    }
                }
                vec![]
            }
            "session_children" => {
                if let Some(data) = raw.data
                    && let Ok(children) = serde_json::from_value::<SessionChildrenData>(data)
                {
                    self.merge_session_children(children);
                }
                vec![]
            }
            "session_created" => {
                if let Some(data) = raw.data
                    && let Ok(sc) = serde_json::from_value::<SessionCreatedData>(data)
                {
                    self.session_id = Some(sc.session_id.clone());
                    self.apply_session_profile_binding(&sc.session_id, sc.profile_id.clone());
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
                    let cmds = vec![ClientMsg::SubscribeSession {
                        session_id: sc.session_id,
                        agent_id: self.agent_id.clone(),
                    }];
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
                            self.apply_session_profile_binding(
                                &sl.session_id,
                                sl.profile_id.clone(),
                            );
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
                        self.apply_session_profile_binding(
                            &parsed.session_id,
                            parsed.profile_id.clone(),
                        );
                        let unknown_kinds =
                            unknown_event_kind_types_from_session_events_data(&data);
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
                        for kind_type in unknown_kinds {
                            self.warn_unknown_event_kind(&kind_type, true);
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
                                self.apply_session_profile_binding(
                                    &se.session_id,
                                    se.profile_id.clone(),
                                );
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
                        self.apply_session_profile_binding(
                            &parsed.session_id,
                            parsed.profile_id.clone(),
                        );
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
                        self.apply_session_profile_binding(&ed.session_id, ed.profile_id.clone());
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
            "control_capabilities" => {
                if let Some(data) = raw.data {
                    self.apply_control_capabilities_log(data);
                }
                vec![]
            }
            "control_capabilities_error" => {
                if let Some(data) = raw.data {
                    let message = data
                        .get("message")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown error");
                    self.push_log(
                        LogLevel::Warn,
                        "capabilities",
                        format!("capabilities unavailable: {message}"),
                    );
                }
                vec![]
            }
            "mesh_nodes" => {
                if let Some(data) = raw.data {
                    let count = data
                        .get("nodes")
                        .and_then(serde_json::Value::as_array)
                        .map(|nodes| nodes.len() as u32)
                        .unwrap_or(0);
                    self.mesh_node_count = Some(count);
                    self.push_log(LogLevel::Info, "mesh", format!("mesh nodes: {count}"));
                }
                vec![]
            }
            "acp_set_session_model" => {
                if let Some(data) = &raw.data {
                    let message = data
                        .get("message")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("ACP SetSessionModel");
                    self.push_log(LogLevel::Info, "acp", message);
                }
                vec![]
            }
            "all_models_list" => {
                if let Some(data) = raw.data
                    && let Some(models) = data.get("models").and_then(serde_json::Value::as_array)
                {
                    self.models = models
                        .iter()
                        .filter_map(|model| serde_json::from_value(model.clone()).ok())
                        .collect();
                    let remote_models = self.models.iter().filter(|m| m.node_id.is_some()).count();
                    let meta = data.get("meta");
                    let remote_nodes = meta
                        .and_then(|m| m.get("remote_node_count"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    let timeouts = meta
                        .and_then(|m| m.get("remote_timeout_count"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0);
                    let mut line = format!(
                        "models: {} total, {} remote",
                        self.models.len(),
                        remote_models
                    );
                    if remote_nodes > 0 || timeouts > 0 {
                        line.push_str(&format!(
                            " (inventory nodes={remote_nodes}, timeouts={timeouts})"
                        ));
                    }
                    self.push_log(LogLevel::Info, "models", line);
                }
                vec![]
            }
            "audio_capabilities" => {
                if let Some(data) = raw.data
                    && let Ok(capabilities) = serde_json::from_value::<AudioCapabilitiesData>(data)
                {
                    let providers = audio_provider_summary(&capabilities);
                    let suffix = if providers.is_empty() {
                        String::new()
                    } else {
                        format!(" ({providers})")
                    };
                    self.push_log(
                        LogLevel::Debug,
                        "audio",
                        format!(
                            "audio: {} STT, {} TTS{}",
                            capabilities.stt_models.len(),
                            capabilities.tts_models.len(),
                            suffix
                        ),
                    );
                }
                vec![]
            }
            "provider_capabilities" => {
                if let Some(data) = raw.data
                    && let Ok(capabilities) =
                        serde_json::from_value::<ProviderCapabilitiesData>(data)
                {
                    let custom_count = capabilities
                        .providers
                        .iter()
                        .filter(|provider| provider.supports_custom_models)
                        .count();
                    self.push_log(
                        LogLevel::Debug,
                        "models",
                        format!(
                            "models: {} provider capability entrie(s), {} support custom models",
                            capabilities.providers.len(),
                            custom_count
                        ),
                    );
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
                    if e.message.contains("Failed to list session children")
                        || e.message
                            .contains("Session children list only supports user forks")
                    {
                        self.pending_session_child_loads.clear();
                    }
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
            EventKind::AssistantThinkingDelta {
                content,
                message_id,
            } => {
                if self.streaming_thinking.is_empty()
                    || (self.streaming_thinking_message_id.is_none() && message_id.is_some())
                {
                    self.streaming_thinking_message_id = message_id.clone();
                }
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
                let streaming_thinking_message_id = self.streaming_thinking_message_id.clone();
                let thinking_text = thinking
                    .as_ref()
                    .filter(|text| !text.is_empty())
                    .cloned()
                    .or_else(|| {
                        if self.streaming_thinking.is_empty() {
                            None
                        } else {
                            Some(std::mem::take(&mut self.streaming_thinking))
                        }
                    });
                let thinking_message_id = message_id.clone().or(streaming_thinking_message_id);
                self.streaming_content.clear();
                self.invalidate_streaming_caches();
                if self.is_turn_active() {
                    self.activity = ActivityState::Thinking;
                }
                if !content.is_empty() || thinking_text.is_some() {
                    self.recent_prompt_text = None;
                    if self.suppress_turn_output {
                        return;
                    }
                    if let Some(message_id) = message_id.as_deref()
                        && self.messages.iter().any(|entry| {
                            matches!(entry, ChatEntry::Assistant { message_id: Some(mid), .. } | ChatEntry::Thinking { message_id: Some(mid), .. } if mid == message_id)
                        })
                    {
                        return;
                    }
                    if content.is_empty() {
                        if let Some(thinking) = thinking_text {
                            push_thinking_entry(&mut self.messages, thinking, thinking_message_id);
                        }
                    } else {
                        self.messages.push(ChatEntry::Assistant {
                            content: content.clone(),
                            thinking: thinking_text,
                            message_id: message_id.clone(),
                        });
                    }
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
                        self.discard_pending_thinking();
                        return;
                    }
                    let cwd = self.current_session_cwd();
                    let detail = tool_detail::parse_tool_detail(
                        tool_name,
                        arguments.as_ref(),
                        cwd.as_deref(),
                    );
                    if tool_detail::reconcile_tool_call_start(
                        &mut self.messages,
                        tool_call_id.as_deref(),
                        tool_name,
                        detail.clone(),
                    ) {
                        self.discard_pending_thinking();
                        self.card_cache.invalidate();
                        return;
                    }
                    if !self.streaming_thinking.is_empty() {
                        let thinking = std::mem::take(&mut self.streaming_thinking);
                        let thinking_message_id = self.streaming_thinking_message_id.take();
                        push_thinking_entry(&mut self.messages, thinking, thinking_message_id);
                        self.streaming_thinking_cache.invalidate();
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
                        updated_existing_tool = tool_detail::update_tool_detail(
                            &mut self.messages,
                            tool_call_id.as_deref(),
                            result_str,
                        );
                    }
                    if is_error.unwrap_or(false) {
                        if tool_detail::mark_tool_call_failed(
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
            EventKind::ArtifactRecorded { artifact } => {
                if !is_replay {
                    self.push_log(
                        LogLevel::Debug,
                        "artifact",
                        format!("artifact recorded: {}", format_artifact_info(artifact)),
                    );
                }
            }
            EventKind::SessionQueued { reason } => {
                if !is_replay {
                    self.set_status(
                        LogLevel::Warn,
                        "session",
                        format!("session queued: {reason}"),
                    );
                }
            }
            EventKind::SessionConfigured {
                cwd,
                mcp_servers,
                limits,
            } => {
                if !is_replay {
                    self.push_log(
                        LogLevel::Debug,
                        "session",
                        format!(
                            "session configured: cwd {}, {} MCP server(s), {}",
                            cwd.as_deref().unwrap_or("none"),
                            mcp_servers.len(),
                            format_session_limits(limits.as_ref())
                        ),
                    );
                }
            }
            EventKind::ToolsAvailable { tools, .. } => {
                if !is_replay {
                    self.push_log(
                        LogLevel::Debug,
                        "tools",
                        format!(
                            "tools available: {} tool(s){}",
                            tools.len(),
                            tool_names_suffix(tools)
                        ),
                    );
                }
            }
            EventKind::ProviderChanged {
                provider,
                model,
                context_limit,
                provider_node_id,
                ..
            } => {
                self.current_provider = Some(provider.clone());
                self.current_model = Some(model.clone());
                if is_replay {
                    if provider_node_id.is_some() {
                        self.current_model_node_id = provider_node_id.clone();
                    }
                } else {
                    self.current_model_node_id = provider_node_id.clone();
                }
                if let Some(limit) = context_limit {
                    self.context_limit = *limit;
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
                        self.streaming_thinking_message_id = None;
                        self.streaming_thinking_cache.invalidate();
                        Some(std::mem::take(&mut self.streaming_thinking))
                    };
                    self.messages.push(ChatEntry::Assistant {
                        content: format!("{partial} [cancelled]"),
                        thinking,
                        message_id: None,
                    });
                } else if !self.streaming_thinking.is_empty() {
                    let thinking = std::mem::take(&mut self.streaming_thinking);
                    let thinking_message_id = self.streaming_thinking_message_id.take();
                    push_thinking_entry(&mut self.messages, thinking, thinking_message_id);
                    self.streaming_thinking_cache.invalidate();
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

    fn apply_control_capabilities_log(&mut self, data: serde_json::Value) {
        let version = data
            .get("querymt_control_version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let agent = data.get("agent");
        let kind = agent
            .and_then(|a| a.get("kind"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let display = agent
            .and_then(|a| a.get("display_name"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let version_str = agent
            .and_then(|a| a.get("version"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let transport = data.get("transport");
        let mesh_on = transport
            .and_then(|t| t.get("mesh"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let mesh_transport = transport
            .and_then(|t| t.get("mesh_transport"))
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("none");
        self.push_log(
            LogLevel::Info,
            "capabilities",
            format!(
                "querymt control v{version}: {display} ({kind} {version_str}), mesh={mesh_on} transport={mesh_transport}"
            ),
        );

        if let Some(features) = data.get("features") {
            let mesh = features
                .get("mesh")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let remote_sessions = features
                .get("remote_sessions")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let profiles = features
                .get("profiles")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let auth = features
                .get("auth")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let models = features
                .get("models")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            self.push_log(
                LogLevel::Debug,
                "capabilities",
                format!(
                    "features: mesh={mesh}, remote_sessions={remote_sessions}, profiles={profiles}, auth={auth}, models={models}"
                ),
            );
        }

        let methods = data
            .get("methods")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let querymt_methods: Vec<&str> = methods
            .iter()
            .map(String::as_str)
            .filter(|m| m.starts_with("querymt/"))
            .collect();
        self.push_log(
            LogLevel::Debug,
            "capabilities",
            format!(
                "methods: {} total ({} querymt/*)",
                methods.len(),
                querymt_methods.len()
            ),
        );
        if !querymt_methods.is_empty() {
            let preview: Vec<&str> = querymt_methods.iter().copied().take(12).collect();
            let suffix = if querymt_methods.len() > preview.len() {
                format!(" …+{}", querymt_methods.len() - preview.len())
            } else {
                String::new()
            };
            self.push_log(
                LogLevel::Debug,
                "capabilities",
                format!("querymt methods: {}{suffix}", preview.join(", ")),
            );
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
        .get("kind")
        .or_else(|| value.get("data").and_then(|data| data.get("kind")))
        .and_then(|kind| kind.get("type"))
        .and_then(|kind_type| kind_type.as_str())
}

fn unknown_event_kind_types_from_session_events_data(data: &serde_json::Value) -> Vec<String> {
    data.get("events")
        .and_then(|events| events.as_array())
        .map(|events| {
            events
                .iter()
                .filter_map(|event| {
                    let kind_type = extract_event_kind_type(event)?;
                    let parsed = serde_json::from_value::<EventEnvelope>(event.clone()).ok()?;
                    matches!(parsed.kind(), EventKind::Unknown).then(|| kind_type.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn audio_provider_summary(capabilities: &AudioCapabilitiesData) -> String {
    let mut providers: Vec<&str> = capabilities
        .stt_models
        .iter()
        .chain(capabilities.tts_models.iter())
        .map(|model| model.provider.as_str())
        .filter(|provider| !provider.is_empty())
        .collect();
    providers.sort_unstable();
    providers.dedup();
    providers.join(", ")
}

fn format_session_limits(limits: Option<&SessionLimits>) -> String {
    let Some(limits) = limits else {
        return "limits none".into();
    };
    let cost = limits
        .max_cost_usd
        .map(|cost| cost.to_string())
        .unwrap_or_else(|| "none".into());
    format!(
        "limits steps={} turns={} cost={}",
        limits
            .max_steps
            .map(|steps| steps.to_string())
            .unwrap_or_else(|| "none".into()),
        limits
            .max_turns
            .map(|turns| turns.to_string())
            .unwrap_or_else(|| "none".into()),
        cost
    )
}

fn format_artifact_info(artifact: &ArtifactInfo) -> String {
    let location = artifact
        .path
        .as_deref()
        .or(artifact.uri.as_deref())
        .unwrap_or("artifact");

    match artifact
        .summary
        .as_deref()
        .filter(|summary| !summary.is_empty())
    {
        Some(summary) => format!("{} {} ({summary})", artifact.kind, location),
        None => format!("{} {}", artifact.kind, location),
    }
}

fn tool_names_suffix(tools: &[ToolInfo]) -> String {
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|tool| tool.function.as_ref())
        .map(|function| function.name.as_str())
        .filter(|name| !name.is_empty())
        .collect();
    if names.is_empty() {
        return String::new();
    }
    format!(" ({})", names.join(", "))
}

fn content_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Object(object) => {
            if object.get("type").and_then(serde_json::Value::as_str) == Some("text") {
                object
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_default()
            } else {
                String::new()
            }
        }
        serde_json::Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                (block.get("type").and_then(serde_json::Value::as_str) == Some("text"))
                    .then(|| block.get("text").and_then(serde_json::Value::as_str))
                    .flatten()
                    .map(str::to_string)
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
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

fn message_entry_exists(messages: &[ChatEntry], message_id: Option<&str>) -> bool {
    let Some(message_id) = message_id else {
        return false;
    };
    messages.iter().any(|entry| {
        matches!(entry, ChatEntry::Assistant { message_id: Some(existing_id), .. } | ChatEntry::Thinking { message_id: Some(existing_id), .. } if existing_id == message_id)
    })
}

fn push_thinking_entry(
    messages: &mut Vec<ChatEntry>,
    content: String,
    message_id: Option<String>,
) -> bool {
    if content.is_empty() || message_entry_exists(messages, message_id.as_deref()) {
        return false;
    }
    messages.push(ChatEntry::Thinking {
        content,
        message_id,
    });
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

// ── scroll_tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tool_call_replay_tests {
    use super::*;

    #[test]
    fn session_loaded_replay_audit_preserves_full_utf8_shell_command() {
        let command =
            "cat > check/kimi.md << 'EOF'\n# Review: feat/profiles\n\n## 🔴 Critical / High";
        let mut app = App::new();

        app.handle_server_msg(RawServerMsg {
            msg_type: "session_loaded".into(),
            data: Some(serde_json::json!({
                "session_id": "sess-utf8",
                "agent_id": "coder",
                "audit": {
                    "events": [{
                        "kind": { "type": "tool_call_start", "data": {
                            "tool_call_id": "tool-utf8",
                            "tool_name": "shell",
                            "arguments": { "command": command }
                        }},
                        "timestamp": null
                    }]
                },
                "undo_stack": []
            })),
        });

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::ToolCall {
                detail: ToolDetail::Shell { command: got, .. },
                ..
            }] if got == command
        ));
    }

    #[test]
    fn repeated_failed_tool_end_adds_one_fallback_card() {
        let mut app = App::new();
        let failed_end = EventKind::ToolCallEnd {
            tool_call_id: Some("missing-start".into()),
            tool_name: "ls".into(),
            is_error: Some(true),
            result: None,
        };

        app.handle_event_kind(&failed_end, true, None);
        app.handle_event_kind(&failed_end, true, None);

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::ToolCall {
                tool_call_id: Some(id),
                name,
                is_error: true,
                detail: ToolDetail::None,
            }] if id == "missing-start" && name == "ls (failed)"
        ));
    }

    #[test]
    fn late_tool_start_reconciles_failed_fallback_in_place() {
        let mut app = App::new();
        app.handle_event_kind(
            &EventKind::ToolCallEnd {
                tool_call_id: Some("missing-start".into()),
                tool_name: "shell".into(),
                is_error: Some(true),
                result: None,
            },
            true,
            None,
        );
        app.handle_event_kind(
            &EventKind::ToolCallStart {
                tool_call_id: Some("missing-start".into()),
                tool_name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "cargo test" })),
            },
            true,
            None,
        );

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::ToolCall {
                tool_call_id: Some(id),
                name,
                is_error: true,
                detail: ToolDetail::Shell { command, .. },
            }] if id == "missing-start" && name == "shell" && command == "cargo test"
        ));
    }
}

#[cfg(test)]
mod fork_result_tests {
    use super::*;

    #[test]
    fn fork_success_loads_and_subscribes_to_child_session() {
        let mut app = App::new();
        app.agent_id = Some("agent-1".into());
        app.popup = Popup::ForkTurnSelect;
        app.pending_fork_message_id = Some("msg-1".into());

        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "fork_result".into(),
            data: Some(serde_json::json!({
                "success": true,
                "source_session_id": "source-1",
                "forked_session_id": "fork-1",
                "message": "forked"
            })),
        });

        assert_eq!(app.pending_fork_message_id, None);
        assert_eq!(app.popup, Popup::None);
        assert_eq!(cmds.len(), 2);
        assert!(
            matches!(&cmds[0], ClientMsg::LoadSession { session_id, .. } if session_id == "fork-1")
        );
        assert!(
            matches!(&cmds[1], ClientMsg::SubscribeSession { session_id, agent_id } if session_id == "fork-1" && agent_id.as_deref() == Some("agent-1"))
        );
    }

    #[test]
    fn fork_failure_clears_pending_without_switching() {
        let mut app = App::new();
        app.popup = Popup::ForkTurnSelect;
        app.pending_fork_message_id = Some("msg-1".into());

        let cmds = app.handle_server_msg(RawServerMsg {
            msg_type: "fork_result".into(),
            data: Some(serde_json::json!({
                "success": false,
                "message": "nope"
            })),
        });

        assert!(cmds.is_empty());
        assert_eq!(app.pending_fork_message_id, None);
        assert_eq!(app.popup, Popup::ForkTurnSelect);
        assert!(app.status.contains("nope"));
    }
}

#[cfg(test)]
mod session_list_pagination_tests {
    use super::*;
    use serde_json::json;

    fn session(id: &str, updated_at: &str) -> serde_json::Value {
        json!({
            "session_id": id,
            "title": id,
            "cwd": "/workspace/project",
            "updated_at": updated_at,
            "parent_session_id": null,
            "has_children": false
        })
    }

    fn list_msg(sessions: Vec<serde_json::Value>, next_cursor: Option<&str>) -> RawServerMsg {
        RawServerMsg {
            msg_type: "session_list".to_string(),
            data: Some(json!({
                "groups": [{
                    "cwd": "/workspace/project",
                    "sessions": sessions,
                    "latest_activity": "2025-01-01T00:00:00Z",
                    "total_count": 20,
                    "next_cursor": next_cursor
                }],
                "next_cursor": null,
                "total_count": 20
            })),
        }
    }

    #[test]
    fn initial_list_does_not_autofill_when_backend_returns_root_page() {
        let mut app = App::new();
        let replies = app.handle_server_msg(list_msg(
            vec![session("root-1", "2025-01-01T00:00:00Z")],
            Some("cursor-1"),
        ));

        assert_eq!(app.session_groups[0].sessions.len(), 1);
        assert_eq!(
            app.session_groups[0].next_cursor.as_deref(),
            Some("cursor-1")
        );
        assert!(replies.is_empty());
    }

    #[test]
    fn group_page_merges_sessions_without_autofill() {
        let mut app = App::new();
        app.handle_server_msg(list_msg(
            vec![session("root-1", "2025-01-01T00:00:00Z")],
            Some("cursor-1"),
        ));
        let initial_total_logs = app
            .logs
            .iter()
            .filter(|entry| {
                entry
                    .message
                    .starts_with("session list: total root sessions=")
            })
            .count();
        let request = app.session_group_page_request(0).unwrap();
        assert!(matches!(
            request,
            ClientMsg::ListSessions {
                session_scope: SessionScope::Root,
                ..
            }
        ));

        let replies = app.handle_server_msg(list_msg(
            vec![
                session("root-2", "2025-01-02T00:00:00Z"),
                session("root-3", "2025-01-03T00:00:00Z"),
            ],
            Some("cursor-2"),
        ));

        assert_eq!(app.session_groups[0].sessions.len(), 3);
        assert_eq!(app.session_groups[0].sessions[0].session_id, "root-3");
        assert_eq!(
            app.session_groups[0].next_cursor.as_deref(),
            Some("cursor-2")
        );
        let total_logs = app
            .logs
            .iter()
            .filter(|entry| {
                entry
                    .message
                    .starts_with("session list: total root sessions=")
            })
            .count();
        assert_eq!(total_logs, initial_total_logs);
        assert!(replies.is_empty());
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
