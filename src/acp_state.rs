use std::collections::HashSet;

use serde_json::Value;

use crate::app::{ActivityState, ChatEntry, ElicitationState, LogLevel, Popup, Screen, ToolDetail};
use crate::protocol::{
    AgentInfo, AuthProviderEntry, ClientMsg, ForkResultData, MeshInviteCreatedInfo, MeshNodesInfo,
    MeshStatusInfo, ModelEntry, OAuthFlowData, OAuthResultData, ProfileInfo, RedoResultData,
    RemoteSessionAttachInfo, RemoteSessionListInfo, SessionGroup, UndoResultData, UndoStackFrame,
};
use crate::tool_detail;

#[derive(Debug, Clone, Default)]
pub(crate) struct AcpModelsMetaInfo {
    pub remote_node_count: u32,
    pub remote_timeout_count: u32,
}

#[derive(Debug, Clone)]
pub(crate) enum AcpSessionUpdate {
    TurnStarted,
    UserMessage {
        content: Value,
        message_id: Option<String>,
    },
    AssistantContentDelta {
        content: String,
        message_id: Option<String>,
    },
    AssistantThinkingDelta {
        content: String,
        message_id: Option<String>,
    },
    AssistantMessage {
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    },
    ToolCallStart {
        tool_call_id: Option<String>,
        name: String,
        arguments: Option<Value>,
    },
    ToolCallEnd {
        tool_call_id: Option<String>,
        name: String,
        is_error: bool,
        result: Option<String>,
    },
    UsageUpdate {
        used: u64,
        size: u64,
        cost_usd: Option<f64>,
    },
    TimingUpdate {
        duration_secs: u64,
    },
    ElicitationRequested {
        elicitation_id: String,
        message: String,
        requested_schema: Value,
        source: String,
        allow_custom: bool,
    },
    Cancelled,
    Finished {
        finish_reason: String,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum AcpAppEvent {
    Initialized {
        agent_id: String,
        agent_name: String,
        profiles: Vec<ProfileInfo>,
        active_profile_id: Option<String>,
        agent_mode: Option<String>,
        reasoning_effort: Option<Option<String>>,
    },
    AgentMode {
        mode: String,
    },
    ReasoningEffort {
        reasoning_effort: Option<String>,
    },
    Profiles {
        profiles: Vec<ProfileInfo>,
        active_profile_id: Option<String>,
    },
    ProviderChanged {
        provider: String,
        model: String,
        context_limit: Option<u64>,
        provider_node_id: Option<String>,
    },
    ControlCapabilities(Value),
    ControlCapabilitiesUnavailable(String),
    MeshStatus(MeshStatusInfo),
    MeshNodes(MeshNodesInfo),
    MeshInviteCreated(MeshInviteCreatedInfo),
    RemoteSessions(RemoteSessionListInfo),
    RemoteSessionAttached(RemoteSessionAttachInfo),
    SessionList {
        groups: Vec<SessionGroup>,
        next_cursor: Option<String>,
        total_count: Option<u64>,
    },
    SessionCreated {
        agent_id: String,
        session_id: String,
        profile_id: Option<String>,
    },
    SessionLoaded {
        agent_id: String,
        session_id: String,
        profile_id: Option<String>,
    },
    SessionUpdate {
        session_id: String,
        update: AcpSessionUpdate,
        is_replay: bool,
    },
    SessionReplay {
        session_id: String,
        updates: Vec<AcpSessionUpdate>,
    },
    Models {
        models: Vec<ModelEntry>,
        meta: Option<AcpModelsMetaInfo>,
    },
    UndoStack(Vec<UndoStackFrame>),
    UndoResult(UndoResultData),
    RedoResult(RedoResultData),
    ForkResult(ForkResultData),
    AuthProviders(Vec<AuthProviderEntry>),
    OAuthFlowStarted(OAuthFlowData),
    OAuthResult(OAuthResultData),
    InfoLog {
        target: &'static str,
        message: String,
    },
    Error {
        message: String,
    },
    PromptFailed {
        local_id: String,
        message: String,
    },
}

impl crate::app::App {
    pub(crate) fn handle_acp_event(&mut self, event: AcpAppEvent) -> Vec<ClientMsg> {
        match event {
            AcpAppEvent::Initialized {
                agent_id,
                agent_name,
                profiles,
                active_profile_id,
                agent_mode,
                reasoning_effort,
            } => {
                self.profiles = profiles;
                if let Some(profile_id) = active_profile_id {
                    self.active_profile_id = Some(profile_id);
                }
                if self.profile_cursor >= self.profiles.len() {
                    self.profile_cursor = self.profiles.len().saturating_sub(1);
                }
                self.agent_id = Some(agent_id.clone());
                self.agents = vec![AgentInfo {
                    id: agent_id,
                    name: agent_name,
                }];
                if let Some(mode) = agent_mode {
                    self.agent_mode = mode;
                    if self.agent_mode != "review" {
                        self.mode_before_review = None;
                    }
                }
                if let Some(effort) = reasoning_effort {
                    self.reasoning_effort = effort;
                }
                self.set_status(LogLevel::Info, "connection", "connected");
                vec![]
            }
            AcpAppEvent::AgentMode { mode } => {
                self.agent_mode = mode;
                if self.agent_mode != "review" {
                    self.mode_before_review = None;
                }
                vec![]
            }
            AcpAppEvent::ReasoningEffort { reasoning_effort } => {
                if let Some(validated) =
                    crate::app::validate_reasoning_effort(reasoning_effort.as_deref())
                {
                    self.reasoning_effort = validated;
                }
                vec![]
            }
            AcpAppEvent::Profiles {
                profiles,
                active_profile_id,
            } => {
                self.profiles = profiles;
                if let Some(profile_id) = active_profile_id {
                    self.active_profile_id = Some(profile_id);
                }
                if self.profile_cursor >= self.profiles.len() {
                    self.profile_cursor = self.profiles.len().saturating_sub(1);
                }
                vec![]
            }
            AcpAppEvent::ProviderChanged {
                provider,
                model,
                context_limit,
                provider_node_id,
            } => {
                self.current_provider = Some(provider);
                self.current_model = Some(model);
                self.current_model_node_id = provider_node_id;
                if let Some(limit) = context_limit {
                    self.context_limit = limit;
                }
                vec![]
            }
            AcpAppEvent::ControlCapabilities(data) => {
                self.apply_acp_control_capabilities_log(data);
                vec![]
            }
            AcpAppEvent::ControlCapabilitiesUnavailable(message) => {
                self.push_log(
                    LogLevel::Warn,
                    "capabilities",
                    format!("capabilities unavailable: {message}"),
                );
                vec![]
            }
            AcpAppEvent::MeshStatus(status) => {
                self.apply_mesh_status(status);
                vec![]
            }
            AcpAppEvent::MeshNodes(nodes) => self.apply_mesh_nodes(nodes),
            AcpAppEvent::MeshInviteCreated(invite) => {
                self.apply_mesh_invite_created(invite);
                vec![]
            }
            AcpAppEvent::RemoteSessions(list) => {
                self.apply_remote_sessions(list);
                vec![]
            }
            AcpAppEvent::RemoteSessionAttached(attached) => self.apply_remote_session_attached(
                &attached.session_id,
                &attached.node_id,
                attached.attached,
            ),
            AcpAppEvent::SessionList {
                groups,
                next_cursor,
                total_count,
            } => {
                self.apply_acp_session_list(groups, next_cursor, total_count);
                vec![]
            }
            AcpAppEvent::SessionCreated {
                agent_id,
                session_id,
                profile_id,
            } => self.apply_acp_session_created(agent_id, session_id, profile_id),
            AcpAppEvent::SessionLoaded {
                agent_id,
                session_id,
                profile_id,
            } => self.apply_acp_session_loaded(agent_id, session_id, profile_id),
            AcpAppEvent::SessionUpdate {
                session_id,
                update,
                is_replay,
            } => {
                self.apply_acp_session_update(&session_id, update, is_replay);
                std::mem::take(&mut self.pending_commands)
            }
            AcpAppEvent::SessionReplay {
                session_id,
                updates,
            } => {
                let updates = normalize_replay_updates(updates);
                self.push_log(
                    LogLevel::Info,
                    "session",
                    format!("session replay: {} update(s)", updates.len()),
                );
                for update in updates {
                    self.apply_acp_session_update(&session_id, update, true);
                }
                std::mem::take(&mut self.pending_commands)
            }
            AcpAppEvent::UndoStack(undo_stack) => {
                self.undo_state = self.build_undo_state_from_server_stack(&undo_stack, None, None);
                vec![]
            }
            AcpAppEvent::UndoResult(ur) => {
                self.activity = ActivityState::Idle;
                let message_id_for_files = ur
                    .message_id
                    .clone()
                    .or_else(|| ur.undo_stack.last().map(|frame| frame.message_id.clone()));
                self.undo_state = self.build_undo_state_from_server_stack(
                    &ur.undo_stack,
                    message_id_for_files.as_deref(),
                    if ur.success {
                        Some(ur.reverted_files.as_slice())
                    } else {
                        None
                    },
                );

                if ur.success {
                    self.recent_prompt_text = None;
                    self.streaming_content.clear();
                    self.streaming_content_message_id = None;
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
                vec![]
            }
            AcpAppEvent::RedoResult(rr) => {
                self.activity = ActivityState::Idle;
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
                vec![]
            }
            AcpAppEvent::ForkResult(fr) => {
                self.pending_fork_message_id = None;
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
                vec![]
            }
            AcpAppEvent::Models { models, meta } => {
                self.models = models;
                let remote_models = self.models.iter().filter(|m| m.node_id.is_some()).count();
                let remote_nodes = meta.as_ref().map(|m| m.remote_node_count).unwrap_or(0);
                let timeouts = meta.as_ref().map(|m| m.remote_timeout_count).unwrap_or(0);
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
                vec![]
            }
            AcpAppEvent::AuthProviders(providers) => {
                self.auth_providers = providers;
                self.push_log(
                    LogLevel::Debug,
                    "auth",
                    format!("{} auth provider(s)", self.auth_providers.len()),
                );
                vec![]
            }
            AcpAppEvent::OAuthFlowStarted(flow) => {
                self.push_log(
                    LogLevel::Info,
                    "auth",
                    format!("OAuth flow started for {}", flow.provider),
                );
                self.auth_oauth_flow = Some(flow);
                self.auth_panel = crate::app::AuthPanel::OAuthFlow;
                self.auth_oauth_response.clear();
                self.auth_oauth_response_cursor = 0;
                self.auth_result_message = None;
                vec![]
            }
            AcpAppEvent::OAuthResult(result) => {
                let level = if result.success {
                    LogLevel::Info
                } else {
                    LogLevel::Warn
                };
                self.push_log(level, "auth", &result.message);
                self.auth_result_message = Some((result.success, result.message));
                if result.success {
                    self.auth_oauth_flow = None;
                    self.auth_panel = crate::app::AuthPanel::List;
                }
                vec![ClientMsg::ListAuthProviders]
            }
            AcpAppEvent::InfoLog { target, message } => {
                self.push_log(LogLevel::Info, target, message);
                vec![]
            }
            AcpAppEvent::Error { message } => {
                self.end_llm_request_span(None);
                self.push_acp_error(&message);
                self.set_status(LogLevel::Error, "acp", format!("error: {message}"));
                vec![]
            }
            AcpAppEvent::PromptFailed { local_id, message } => {
                self.end_llm_request_span(None);
                self.messages.retain(|entry| {
                    !matches!(
                        entry,
                        ChatEntry::User {
                            message_id: Some(message_id),
                            ..
                        } if message_id == &local_id
                    )
                });
                self.card_cache.invalidate();
                self.push_acp_error(&message);
                self.set_status(LogLevel::Error, "acp", format!("error: {message}"));
                vec![]
            }
        }
    }

    fn apply_acp_session_list(
        &mut self,
        mut groups: Vec<SessionGroup>,
        _next_cursor: Option<String>,
        total_count: Option<u64>,
    ) {
        let response_cwd = if groups.len() == 1 {
            groups.first().and_then(|group| group.cwd.clone())
        } else {
            None
        };
        let is_group_page = response_cwd
            .as_ref()
            .map(|cwd| self.pending_session_group_loads.remove(&Some(cwd.clone())))
            .unwrap_or_else(|| self.pending_session_group_loads.remove(&None));

        groups.retain(|group| is_group_page || !group.sessions.is_empty());
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
                    let mut seen = existing
                        .sessions
                        .iter()
                        .map(|session| session.session_id.clone())
                        .collect::<std::collections::HashSet<_>>();
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
        if !is_group_page && let Some(root_total) = total_count {
            self.push_log(
                LogLevel::Info,
                "session",
                format!("session list: total root sessions={root_total}, loaded={total}"),
            );
        }

        let visible_len = self.visible_start_items().len();
        if self.session_cursor >= visible_len && visible_len > 0 {
            self.session_cursor = visible_len - 1;
        }
    }

    fn apply_acp_session_created(
        &mut self,
        agent_id: String,
        session_id: String,
        profile_id: Option<String>,
    ) -> Vec<ClientMsg> {
        self.session_id = Some(session_id.clone());
        self.apply_session_profile_binding(&session_id, profile_id);
        self.agent_id = Some(agent_id);
        self.reset_active_session_view();
        self.screen = Screen::Chat;
        self.set_status(LogLevel::Info, "session", "session created");
        vec![ClientMsg::SubscribeSession {
            session_id,
            agent_id: self.agent_id.clone(),
        }]
    }

    fn apply_acp_session_loaded(
        &mut self,
        agent_id: String,
        session_id: String,
        profile_id: Option<String>,
    ) -> Vec<ClientMsg> {
        self.activity = ActivityState::Idle;
        self.parent_session_id = self.pending_parent_session_id.take().or_else(|| {
            self.session_groups
                .iter()
                .flat_map(|g| &g.sessions)
                .find(|s| s.session_id == session_id)
                .and_then(|s| s.parent_session_id.clone())
        });
        self.apply_session_profile_binding(&session_id, profile_id);
        self.session_id = Some(session_id.clone());
        self.agent_id = Some(agent_id);
        self.reset_active_session_view();
        self.screen = if self.parent_session_id.is_some() {
            Screen::Delegate
        } else {
            Screen::Chat
        };
        self.set_status(LogLevel::Debug, "activity", "ready");
        vec![ClientMsg::SetAgentMode {
            mode: self.agent_mode.clone(),
        }]
    }

    fn reset_active_session_view(&mut self) {
        self.messages.clear();
        self.pending_prompt_seq = 0;
        self.streaming_content.clear();
        self.streaming_content_message_id = None;
        self.streaming_thinking.clear();
        self.streaming_thinking_message_id = None;
        self.invalidate_streaming_caches();
        self.card_cache.invalidate();
        self.scroll_offset = 0;
        self.undo_state = None;
        self.undoable_turns.clear();
        self.recent_prompt_text = None;
        self.suppress_turn_output = false;
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
        self.cumulative_cost = None;
        self.session_stats = crate::app::SessionStatsLite::default();
    }

    fn apply_acp_session_update(
        &mut self,
        session_id: &str,
        update: AcpSessionUpdate,
        is_replay: bool,
    ) {
        if self.session_id.as_deref() != Some(session_id) {
            self.note_session_activity(session_id);
            return;
        }
        self.note_session_activity(session_id);

        match update {
            AcpSessionUpdate::TurnStarted => {
                self.clear_cancel_confirm();
                if !is_replay {
                    self.begin_llm_request_span(None);
                }
                self.activity = ActivityState::Thinking;
                self.streaming_content.clear();
                self.streaming_content_message_id = None;
                self.invalidate_streaming_caches();
                self.set_status(LogLevel::Debug, "activity", "thinking...");
            }
            AcpSessionUpdate::UserMessage {
                content,
                message_id,
            } => self.push_acp_user_message(content, message_id, is_replay),
            AcpSessionUpdate::AssistantContentDelta {
                content,
                message_id,
            } => {
                if self.is_turn_active() {
                    self.activity = ActivityState::Streaming;
                }
                let active_message_id = self
                    .streaming_content_message_id
                    .as_deref()
                    .or(self.streaming_thinking_message_id.as_deref());
                if message_id.as_deref().is_some_and(|incoming| {
                    active_message_id.is_some_and(|active| active != incoming)
                }) {
                    self.finalize_streaming_segment();
                }
                if self.streaming_content.is_empty()
                    || (self.streaming_content_message_id.is_none() && message_id.is_some())
                {
                    self.streaming_content_message_id = message_id;
                }
                self.streaming_content.push_str(&content);
            }
            AcpSessionUpdate::AssistantThinkingDelta {
                content,
                message_id,
            } => {
                // `Finished` clears non-durable streaming thinking, so only an unterminated
                // replay can be present here for a second identical application.
                if is_replay
                    && message_id.as_deref().is_some_and(|incoming| {
                        self.messages.iter().any(|entry| {
                            matches!(entry, ChatEntry::Thinking { message_id: Some(existing), .. } if existing == incoming)
                        }) || (self.streaming_thinking_message_id.as_deref() == Some(incoming)
                            && self.streaming_thinking == content)
                    })
                {
                    return;
                }
                let active_message_id = self
                    .streaming_content_message_id
                    .as_deref()
                    .or(self.streaming_thinking_message_id.as_deref());
                if message_id.as_deref().is_some_and(|incoming| {
                    active_message_id.is_some_and(|active| active != incoming)
                }) {
                    self.finalize_streaming_segment();
                }
                if self.streaming_thinking.is_empty()
                    || (self.streaming_thinking_message_id.is_none() && message_id.is_some())
                {
                    self.streaming_thinking_message_id = message_id;
                }
                self.streaming_thinking.push_str(&content);
            }
            AcpSessionUpdate::AssistantMessage {
                content,
                thinking,
                message_id,
            } => self.push_acp_assistant_message(content, thinking, message_id),
            AcpSessionUpdate::ToolCallStart {
                tool_call_id,
                name,
                arguments,
            } => {
                if self.suppress_turn_output {
                    return;
                }
                self.activity = ActivityState::RunningTool { name: name.clone() };
                self.set_status(LogLevel::Debug, "tool", format!("tool: {name}"));
                self.finalize_streaming_segment();
                if name != "question" {
                    let cwd = self.current_session_cwd();
                    let detail =
                        tool_detail::parse_tool_detail(&name, arguments.as_ref(), cwd.as_deref());
                    if tool_detail::reconcile_tool_call_start(
                        &mut self.messages,
                        tool_call_id.as_deref(),
                        &name,
                        detail.clone(),
                    ) {
                        self.streaming_thinking.clear();
                        self.streaming_thinking_message_id = None;
                        self.streaming_thinking_cache.invalidate();
                        self.card_cache.invalidate();
                        return;
                    }
                    self.session_stats.total_tool_calls =
                        self.session_stats.total_tool_calls.saturating_add(1);
                    if !self.streaming_thinking.is_empty() {
                        let thinking = std::mem::take(&mut self.streaming_thinking);
                        let thinking_message_id = self.streaming_thinking_message_id.take();
                        self.messages.push(ChatEntry::Thinking {
                            content: thinking,
                            message_id: thinking_message_id,
                        });
                        self.streaming_thinking_cache.invalidate();
                    }
                    self.messages.push(ChatEntry::ToolCall {
                        tool_call_id,
                        name,
                        is_error: false,
                        detail,
                    });
                }
            }
            AcpSessionUpdate::ToolCallEnd {
                tool_call_id,
                name,
                is_error,
                result,
            } => {
                let mut updated = false;
                if let Some(result) = result.as_deref() {
                    updated = tool_detail::update_tool_detail(
                        &mut self.messages,
                        tool_call_id.as_deref(),
                        result,
                    );
                }
                if is_error {
                    if tool_detail::mark_tool_call_failed(
                        &mut self.messages,
                        tool_call_id.as_deref(),
                        &name,
                    ) {
                        updated = true;
                    } else {
                        self.messages.push(ChatEntry::ToolCall {
                            tool_call_id,
                            name: format!("{name} (failed)"),
                            is_error: true,
                            detail: result.map(ToolDetail::Summary).unwrap_or(ToolDetail::None),
                        });
                        updated = true;
                    }
                }
                if updated {
                    self.card_cache.invalidate();
                }
            }
            AcpSessionUpdate::UsageUpdate {
                used,
                size,
                cost_usd,
            } => {
                if used > 0 {
                    self.session_stats.latest_context_tokens = Some(used);
                }
                if size > 0 {
                    self.context_limit = size;
                }
                if let Some(cost_usd) = cost_usd {
                    self.cumulative_cost = Some(cost_usd);
                }
                let pct = if used > 0 && size > 0 {
                    format!(" ({}%)", (used as f64 / size as f64 * 100.0) as u32)
                } else {
                    String::new()
                };
                let cost = cost_usd
                    .map(|amount| format!(", cost ${amount:.4}"))
                    .unwrap_or_default();
                self.push_log(
                    LogLevel::Info,
                    "usage",
                    format!("usage: context {used}/{size} tokens{pct}{cost}"),
                );
            }
            AcpSessionUpdate::TimingUpdate { duration_secs } => {
                if duration_secs > 0 {
                    self.session_stats.active_llm_duration +=
                        std::time::Duration::from_secs(duration_secs);
                    self.push_log(
                        LogLevel::Info,
                        "usage",
                        format!("usage: active time {duration_secs}s"),
                    );
                }
            }
            AcpSessionUpdate::ElicitationRequested {
                elicitation_id,
                message,
                requested_schema,
                source,
                allow_custom,
            } => {
                self.handle_acp_elicitation_requested(
                    &elicitation_id,
                    &message,
                    &source,
                    &requested_schema,
                    allow_custom,
                    is_replay,
                );
            }
            AcpSessionUpdate::Cancelled => {
                if !is_replay {
                    self.end_llm_request_span(None);
                }
                self.activity = ActivityState::Idle;
                self.streaming_content.clear();
                self.streaming_content_message_id = None;
                self.invalidate_streaming_caches();
                self.set_status(LogLevel::Warn, "activity", "cancelled");
            }
            AcpSessionUpdate::Finished { finish_reason } => {
                if !is_replay {
                    self.end_llm_request_span(None);
                }
                self.activity = ActivityState::Idle;
                self.streaming_content.clear();
                self.streaming_content_message_id = None;
                self.streaming_thinking.clear();
                self.streaming_thinking_message_id = None;
                self.invalidate_streaming_caches();
                self.set_status(
                    LogLevel::Debug,
                    "activity",
                    format!("finished: {finish_reason}"),
                );
            }
        }
    }

    fn push_acp_user_message(
        &mut self,
        content: Value,
        message_id: Option<String>,
        is_replay: bool,
    ) {
        let text = acp_content_to_string(&content);
        if text.is_empty() {
            return;
        }
        if let Some(message_id) = message_id.as_deref()
            && self.messages.iter().any(|entry| {
                matches!(entry, ChatEntry::User { message_id: Some(mid), .. } if mid == message_id)
            })
        {
            self.recent_prompt_text = Some(text);
            self.suppress_turn_output = false;
            return;
        }
        // QueryMT currently echoes prompts without the client's local ID. Correlating
        // PromptRequest._meta with UserMessageChunk._meta would make this identity-based;
        // until then, normalized text is the interoperable fallback for generic ACP agents.
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
            self.card_cache.invalidate();
            return;
        }
        if !is_replay && (self.undo_state.is_some() || self.suppress_turn_output) {
            return;
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
    }

    fn finalize_streaming_segment(&mut self) {
        if self.streaming_content.is_empty() && self.streaming_thinking.is_empty() {
            return;
        }
        let content = std::mem::take(&mut self.streaming_content);
        let content_message_id = self.streaming_content_message_id.take();
        let thinking = (!self.streaming_thinking.is_empty())
            .then(|| std::mem::take(&mut self.streaming_thinking));
        let thinking_message_id = self.streaming_thinking_message_id.take();
        self.invalidate_streaming_caches();

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
        self.card_cache.invalidate();
    }

    fn push_undoable_user_turn(&mut self, message_id: String, text: String) {
        if !self
            .undoable_turns
            .iter()
            .any(|turn| turn.message_id == message_id)
        {
            self.undoable_turns.push(crate::app::UndoableTurn {
                turn_id: message_id.clone(),
                message_id,
                text,
            });
        }
    }

    fn push_acp_assistant_message(
        &mut self,
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    ) {
        let explicit_thinking = thinking.filter(|text| !text.is_empty());
        let streaming_thinking_message_id = self.streaming_thinking_message_id.clone();
        let thinking_message_id = message_id.clone().or(streaming_thinking_message_id);
        self.streaming_content.clear();
        self.streaming_content_message_id = None;
        self.streaming_cache.invalidate();
        if self.is_turn_active() {
            self.activity = ActivityState::Thinking;
        }
        if content.is_empty() && explicit_thinking.is_none() && self.streaming_thinking.is_empty() {
            self.streaming_thinking_message_id = None;
            self.streaming_thinking_cache.invalidate();
            return;
        }
        self.recent_prompt_text = None;
        if self.suppress_turn_output {
            self.streaming_thinking.clear();
            self.streaming_thinking_message_id = None;
            self.streaming_thinking_cache.invalidate();
            return;
        }

        if let Some(message_id) = message_id.as_deref() {
            if self.messages.iter().any(|entry| {
                matches!(entry, ChatEntry::Assistant { message_id: Some(mid), .. } if mid == message_id)
            }) {
                self.streaming_thinking.clear();
                self.streaming_thinking_message_id = None;
                self.streaming_thinking_cache.invalidate();
                return;
            }

            if let Some(idx) = self.messages.iter().position(|entry| {
                matches!(entry, ChatEntry::Thinking { message_id: Some(mid), .. } if mid == message_id)
            }) {
                if content.is_empty() {
                    self.streaming_thinking.clear();
                    self.streaming_thinking_message_id = None;
                    self.streaming_thinking_cache.invalidate();
                    return;
                }
                let existing_thinking = match &self.messages[idx] {
                    ChatEntry::Thinking { content, .. } => content.clone(),
                    _ => String::new(),
                };
                let streaming_thinking = (!self.streaming_thinking.is_empty())
                    .then(|| std::mem::take(&mut self.streaming_thinking));
                let thinking_text = explicit_thinking
                    .or_else(|| (!existing_thinking.is_empty()).then_some(existing_thinking))
                    .or(streaming_thinking);
                self.streaming_thinking_message_id = None;
                self.streaming_thinking_cache.invalidate();
                self.messages[idx] = ChatEntry::Assistant {
                    content,
                    thinking: thinking_text,
                    message_id: Some(message_id.to_string()),
                };
                self.card_cache.invalidate();
                return;
            }
        }

        let streaming_thinking = (!self.streaming_thinking.is_empty())
            .then(|| std::mem::take(&mut self.streaming_thinking));
        let thinking_text = explicit_thinking.or(streaming_thinking);
        self.streaming_thinking_message_id = None;
        self.streaming_thinking_cache.invalidate();
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
    }

    fn push_acp_error(&mut self, message: &str) {
        if !self
            .messages
            .iter()
            .any(|entry| matches!(entry, ChatEntry::Error(existing) if existing == message))
        {
            self.messages.push(ChatEntry::Error(message.to_string()));
        }
    }

    fn handle_acp_elicitation_requested(
        &mut self,
        elicitation_id: &str,
        message: &str,
        source: &str,
        requested_schema: &Value,
        allow_custom: bool,
        is_replay: bool,
    ) {
        // An elicitation ID is the durable identity across live delivery and replay.
        // Keep an existing card (and any recorded outcome) rather than reopening it.
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

        self.finalize_streaming_segment();
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
            if !is_replay {
                self.elicitation = Some(ElicitationState {
                    elicitation_id: elicitation_id.to_string(),
                    message: message.to_string(),
                    source: source.to_string(),
                    fields,
                    field_cursor: 0,
                    option_cursor: 0,
                    selected: std::collections::HashMap::new(),
                    text_input: String::new(),
                    text_cursor: 0,
                    allow_custom,
                    custom_active: false,
                    custom_input: String::new(),
                    custom_cursor: 0,
                    custom_line_width: 1,
                    custom_scroll: 0,
                });
            }
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

    fn apply_acp_control_capabilities_log(&mut self, data: Value) {
        let version = data
            .get("querymt_control_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let agent = data.get("agent");
        let kind = agent
            .and_then(|a| a.get("kind"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        let display = agent
            .and_then(|a| a.get("display_name"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        let version_str = agent
            .and_then(|a| a.get("version"))
            .and_then(Value::as_str)
            .unwrap_or("?");
        let transport = data.get("transport");
        let mesh_on = transport
            .and_then(|t| t.get("mesh"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mesh_transport = transport
            .and_then(|t| t.get("mesh_transport"))
            .and_then(Value::as_str)
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
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let remote_sessions = features
                .get("remote_sessions")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let profiles = features
                .get("profiles")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let auth = features
                .get("auth")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let models = features
                .get("models")
                .and_then(Value::as_bool)
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
            .and_then(Value::as_array)
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
                format!(" ...+{}", querymt_methods.len() - preview.len())
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

fn normalize_replay_updates(updates: Vec<AcpSessionUpdate>) -> Vec<AcpSessionUpdate> {
    let finalized_messages = finalized_assistant_messages(&updates);
    let mut normalized = Vec::with_capacity(updates.len());
    let mut pending_delta: Option<PendingReplayDelta> = None;
    let mut pending_assistant: Option<PendingReplayAssistant> = None;
    let mut emitted_assistant_ids = HashSet::new();

    for update in updates {
        match update {
            AcpSessionUpdate::AssistantContentDelta {
                content,
                message_id,
            } => {
                flush_pending_replay_assistant(
                    &mut normalized,
                    &mut pending_assistant,
                    &mut emitted_assistant_ids,
                );
                if message_id
                    .as_ref()
                    .is_some_and(|id| finalized_messages.contains(id))
                {
                    continue;
                }
                push_replay_delta(
                    &mut normalized,
                    &mut pending_delta,
                    PendingReplayDelta::Content {
                        content,
                        message_id,
                    },
                );
            }
            AcpSessionUpdate::AssistantThinkingDelta {
                content,
                message_id,
            } => {
                flush_pending_replay_assistant(
                    &mut normalized,
                    &mut pending_assistant,
                    &mut emitted_assistant_ids,
                );
                if message_id
                    .as_ref()
                    .is_some_and(|id| finalized_messages.contains(id))
                {
                    continue;
                }
                push_replay_delta(
                    &mut normalized,
                    &mut pending_delta,
                    PendingReplayDelta::Thinking {
                        content,
                        message_id,
                    },
                );
            }
            AcpSessionUpdate::AssistantMessage {
                content,
                thinking,
                message_id,
            } => {
                flush_pending_replay_delta(&mut normalized, &mut pending_delta);
                push_replay_assistant(
                    &mut normalized,
                    &mut pending_assistant,
                    &mut emitted_assistant_ids,
                    PendingReplayAssistant {
                        content,
                        thinking,
                        message_id,
                    },
                );
            }
            other => {
                flush_pending_replay_delta(&mut normalized, &mut pending_delta);
                flush_pending_replay_assistant(
                    &mut normalized,
                    &mut pending_assistant,
                    &mut emitted_assistant_ids,
                );
                normalized.push(other);
            }
        }
    }

    flush_pending_replay_delta(&mut normalized, &mut pending_delta);
    flush_pending_replay_assistant(
        &mut normalized,
        &mut pending_assistant,
        &mut emitted_assistant_ids,
    );
    normalized
}

#[derive(Debug)]
struct PendingReplayAssistant {
    content: String,
    thinking: Option<String>,
    message_id: Option<String>,
}

impl PendingReplayAssistant {
    fn can_merge(&self, other: &Self) -> bool {
        message_id_matches(self.message_id.as_ref(), other.message_id.as_ref())
    }

    fn merge(&mut self, other: Self) {
        self.content.push_str(&other.content);
        if let Some(thinking) = other.thinking {
            match &mut self.thinking {
                Some(existing) => existing.push_str(&thinking),
                None => self.thinking = Some(thinking),
            }
        }
        if self.message_id.is_none() && other.message_id.is_some() {
            self.message_id = other.message_id;
        }
    }

    fn into_update(self) -> AcpSessionUpdate {
        AcpSessionUpdate::AssistantMessage {
            content: self.content,
            thinking: self.thinking,
            message_id: self.message_id,
        }
    }
}

#[derive(Debug)]
enum PendingReplayDelta {
    Content {
        content: String,
        message_id: Option<String>,
    },
    Thinking {
        content: String,
        message_id: Option<String>,
    },
}

impl PendingReplayDelta {
    fn can_merge(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Content {
                    message_id: left, ..
                },
                Self::Content {
                    message_id: right, ..
                },
            )
            | (
                Self::Thinking {
                    message_id: left, ..
                },
                Self::Thinking {
                    message_id: right, ..
                },
            ) => message_id_matches(left.as_ref(), right.as_ref()),
            _ => false,
        }
    }

    fn merge(&mut self, other: Self) {
        match (self, other) {
            (
                Self::Content {
                    content,
                    message_id,
                },
                Self::Content {
                    content: next,
                    message_id: next_id,
                },
            )
            | (
                Self::Thinking {
                    content,
                    message_id,
                },
                Self::Thinking {
                    content: next,
                    message_id: next_id,
                },
            ) => {
                content.push_str(&next);
                if message_id.is_none() && next_id.is_some() {
                    *message_id = next_id;
                }
            }
            _ => {}
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Content { content, .. } | Self::Thinking { content, .. } => content.is_empty(),
        }
    }

    fn into_update(self) -> Option<AcpSessionUpdate> {
        match self {
            Self::Content { content, .. } => {
                (!content.is_empty()).then_some(AcpSessionUpdate::AssistantMessage {
                    content,
                    thinking: None,
                    // Synthesized historical chunks may be split by tools, so avoid
                    // message-id dedupe from hiding later ordered segments.
                    message_id: None,
                })
            }
            Self::Thinking {
                content,
                message_id,
            } => (!content.is_empty()).then_some(AcpSessionUpdate::AssistantThinkingDelta {
                content,
                message_id,
            }),
        }
    }
}

fn finalized_assistant_messages(updates: &[AcpSessionUpdate]) -> HashSet<String> {
    updates
        .iter()
        .filter_map(|update| match update {
            AcpSessionUpdate::AssistantMessage {
                message_id: Some(message_id),
                ..
            } => Some(message_id.clone()),
            _ => None,
        })
        .collect()
}

fn push_replay_assistant(
    normalized: &mut Vec<AcpSessionUpdate>,
    pending: &mut Option<PendingReplayAssistant>,
    emitted_ids: &mut HashSet<String>,
    assistant: PendingReplayAssistant,
) {
    if assistant
        .message_id
        .as_ref()
        .is_some_and(|id| emitted_ids.contains(id))
    {
        return;
    }

    match pending {
        Some(existing) if existing.can_merge(&assistant) => existing.merge(assistant),
        Some(_) => {
            flush_pending_replay_assistant(normalized, pending, emitted_ids);
            *pending = Some(assistant);
        }
        None => *pending = Some(assistant),
    }
}

fn flush_pending_replay_assistant(
    normalized: &mut Vec<AcpSessionUpdate>,
    pending: &mut Option<PendingReplayAssistant>,
    emitted_ids: &mut HashSet<String>,
) {
    if let Some(assistant) = pending.take() {
        if let Some(message_id) = assistant.message_id.as_ref() {
            emitted_ids.insert(message_id.clone());
        }
        normalized.push(assistant.into_update());
    }
}

fn push_replay_delta(
    normalized: &mut Vec<AcpSessionUpdate>,
    pending: &mut Option<PendingReplayDelta>,
    delta: PendingReplayDelta,
) {
    if delta.is_empty() {
        return;
    }

    match pending {
        Some(existing) if existing.can_merge(&delta) => existing.merge(delta),
        Some(_) => {
            flush_pending_replay_delta(normalized, pending);
            *pending = Some(delta);
        }
        None => *pending = Some(delta),
    }
}

fn flush_pending_replay_delta(
    normalized: &mut Vec<AcpSessionUpdate>,
    pending: &mut Option<PendingReplayDelta>,
) {
    if let Some(delta) = pending.take()
        && let Some(update) = delta.into_update()
    {
        normalized.push(update);
    }
}

fn message_id_matches(left: Option<&String>, right: Option<&String>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        (None, _) | (_, None) => true,
    }
}

fn acp_content_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(obj) => obj
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| value.to_string()),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{App, ChatEntry, Screen};
    use crate::protocol::SessionSummary;

    const TEST_SESSION_ID: &str = "session-1";
    const TEST_ASSISTANT_ID: &str = "a1";

    fn app_with_active_session() -> App {
        let mut app = App::new();
        app.session_id = Some(TEST_SESSION_ID.into());
        app
    }

    fn session_group(cwd: &str, ids: &[&str]) -> SessionGroup {
        SessionGroup {
            cwd: Some(cwd.to_string()),
            latest_activity: None,
            total_count: Some(ids.len() as u64),
            next_cursor: None,
            sessions: ids
                .iter()
                .map(|id| SessionSummary {
                    session_id: id.to_string(),
                    title: Some(format!("Session {id}")),
                    cwd: Some(cwd.to_string()),
                    ..Default::default()
                })
                .collect(),
        }
    }

    fn apply_live_update(app: &mut App, update: AcpSessionUpdate) {
        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: TEST_SESSION_ID.into(),
            is_replay: false,
            update,
        });
    }

    fn failed_tool_end(tool_call_id: &str, name: &str) -> AcpSessionUpdate {
        AcpSessionUpdate::ToolCallEnd {
            tool_call_id: Some(tool_call_id.into()),
            name: name.into(),
            is_error: true,
            result: Some("failed".into()),
        }
    }

    fn assert_single_assistant(
        app: &App,
        expected_content: &str,
        expected_thinking: Option<&str>,
        expected_message_id: &str,
    ) {
        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking, message_id: Some(message_id) }]
                if content == expected_content
                    && thinking.as_deref() == expected_thinking
                    && message_id == expected_message_id
        ));
    }

    fn push_thinking_entry(app: &mut App, content: &str) {
        app.messages.push(ChatEntry::Thinking {
            content: content.into(),
            message_id: Some(TEST_ASSISTANT_ID.into()),
        });
    }

    fn final_assistant_update(content: &str, thinking: Option<&str>) -> AcpSessionUpdate {
        AcpSessionUpdate::AssistantMessage {
            content: content.into(),
            thinking: thinking.map(str::to_string),
            message_id: Some(TEST_ASSISTANT_ID.into()),
        }
    }

    #[test]
    fn native_mesh_nodes_store_state_and_requests_first_node_sessions() {
        let mut app = App::new();

        let replies = app.handle_acp_event(AcpAppEvent::MeshNodes(MeshNodesInfo {
            nodes: vec![crate::protocol::RemoteNodeInfo {
                id: "node-1".into(),
                label: "framework".into(),
                active_sessions: 2,
                ..Default::default()
            }],
        }));

        assert_eq!(app.mesh_node_count, Some(1));
        assert_eq!(app.selected_mesh_node_id(), Some("node-1"));
        assert!(matches!(
            replies.as_slice(),
            [ClientMsg::ListRemoteSessions { node_id, offset: Some(0), limit: Some(50) }]
                if node_id == "node-1"
        ));
    }

    #[test]
    fn native_mesh_invite_created_stores_invite_and_opens_invite_view() {
        let mut app = App::new();

        app.handle_acp_event(AcpAppEvent::MeshInviteCreated(MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: Some("QR".into()),
            expires_at: 1,
            max_uses: 1,
            mesh_name: Some("Team Mesh".into()),
        }));

        assert!(matches!(app.popup, Popup::MeshInviteQr));
        assert_eq!(app.mesh_invite_url(), Some("qmt://mesh/join/token"));
    }

    #[test]
    fn native_remote_attach_loads_attached_session() {
        let mut app = App::new();

        let replies = app.handle_acp_event(AcpAppEvent::RemoteSessionAttached(
            RemoteSessionAttachInfo {
                session_id: "remote-1".into(),
                node_id: "node-1".into(),
                attached: true,
                config_options: Vec::new(),
                snapshot: Value::Null,
            },
        ));

        assert_eq!(app.session_remote_node_id("remote-1"), Some("node-1"));
        assert!(matches!(
            replies.as_slice(),
            [ClientMsg::LoadSession { session_id, .. }] if session_id == "remote-1"
        ));
    }

    #[test]
    fn native_initial_session_list_does_not_autofill() {
        let mut app = App::new();
        let mut group = session_group("/repo", &["s1"]);
        group.next_cursor = Some("cursor-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::SessionList {
            groups: vec![group],
            next_cursor: None,
            total_count: Some(20),
        });

        assert!(replies.is_empty());
        assert_eq!(app.session_groups.len(), 1);
        assert_eq!(app.session_groups[0].sessions.len(), 1);
        assert_eq!(
            app.session_groups[0].next_cursor.as_deref(),
            Some("cursor-1")
        );
    }

    #[test]
    fn acp_group_session_page_merges_group_and_dedupes() {
        let mut app = App::new();
        app.session_groups = vec![session_group("/repo", &["s1"])];
        app.pending_session_group_loads.insert(Some("/repo".into()));

        let mut group = session_group("/repo", &["s1", "s2"]);
        group.next_cursor = Some("cursor-2".into());
        app.handle_acp_event(AcpAppEvent::SessionList {
            groups: vec![group],
            next_cursor: None,
            total_count: None,
        });

        assert!(app.pending_session_group_loads.is_empty());
        assert_eq!(app.session_groups.len(), 1);
        assert_eq!(
            app.session_groups[0].next_cursor.as_deref(),
            Some("cursor-2")
        );
        assert_eq!(
            app.session_groups[0]
                .sessions
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["s1", "s2"]
        );
    }

    #[test]
    fn native_session_created_resets_view_and_subscribes() {
        let mut app = App::new();
        app.messages.push(ChatEntry::Error("stale".into()));
        app.streaming_content = "stale stream".into();
        app.scroll_offset = 3;
        app.undo_state = Some(crate::app::UndoState {
            stack: Vec::new(),
            frontier_message_id: Some("undo-1".into()),
        });
        app.elicitation = Some(ElicitationState::new_for_test(Vec::new()));
        app.session_stats.total_tool_calls = 1;

        let replies = app.handle_acp_event(AcpAppEvent::SessionCreated {
            agent_id: "agent-1".into(),
            session_id: "session-1".into(),
            profile_id: Some("code".into()),
        });

        assert_eq!(app.session_id.as_deref(), Some("session-1"));
        assert_eq!(app.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(app.screen, Screen::Chat);
        assert!(app.messages.is_empty());
        assert!(app.streaming_content.is_empty());
        assert_eq!(app.scroll_offset, 0);
        assert!(app.undo_state.is_none());
        assert!(app.elicitation.is_none());
        assert_eq!(app.session_stats.total_tool_calls, 0);
        assert!(matches!(
            replies.as_slice(),
            [ClientMsg::SubscribeSession { session_id, agent_id }]
                if session_id == "session-1" && agent_id.as_deref() == Some("agent-1")
        ));
    }

    #[test]
    fn native_session_loaded_discovers_parent_preserves_delegate_state_and_resets_view() {
        let mut app = App::new();
        app.agent_mode = "plan".into();
        app.session_groups = vec![session_group("/repo", &["parent", "child"])];
        app.session_groups[0].sessions[1].parent_session_id = Some("parent".into());
        app.messages.push(ChatEntry::Error("stale".into()));
        app.streaming_content = "stale stream".into();
        app.streaming_content_message_id = Some("stream-1".into());
        app.streaming_thinking = "stale thinking".into();
        app.streaming_thinking_message_id = Some("thinking-1".into());
        app.scroll_offset = 4;
        app.undo_state = Some(crate::app::UndoState {
            stack: Vec::new(),
            frontier_message_id: Some("undo-1".into()),
        });
        app.recent_prompt_text = Some("stale prompt".into());
        app.elicitation = Some(ElicitationState::new_for_test(Vec::new()));
        app.session_stats.total_tool_calls = 2;
        app.delegate_entries.push(crate::app::DelegateEntry {
            delegation_id: "delegate-1".into(),
            child_session_id: Some("child".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "keep".into(),
            status: crate::app::DelegateStatus::InProgress,
            stats: crate::app::DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: crate::app::DelegateChildState::None,
        });

        let replies = app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "child".into(),
            profile_id: None,
        });

        assert_eq!(app.parent_session_id.as_deref(), Some("parent"));
        assert_eq!(app.screen, Screen::Delegate);
        assert!(app.messages.is_empty());
        assert!(app.streaming_content.is_empty());
        assert_eq!(app.streaming_content_message_id, None);
        assert!(app.streaming_thinking.is_empty());
        assert_eq!(app.streaming_thinking_message_id, None);
        assert_eq!(app.scroll_offset, 0);
        assert!(app.undo_state.is_none());
        assert!(app.undoable_turns.is_empty());
        assert!(app.recent_prompt_text.is_none());
        assert!(app.elicitation.is_none());
        assert_eq!(app.session_stats.total_tool_calls, 0);
        assert_eq!(app.delegate_entries.len(), 1);
        assert!(matches!(
            replies.as_slice(),
            [ClientMsg::SetAgentMode { mode }] if mode == "plan"
        ));
    }

    #[test]
    fn native_session_loaded_root_clears_delegate_state() {
        let mut app = App::new();
        app.parent_session_id = Some("old-parent".into());
        app.delegate_entries.push(crate::app::DelegateEntry {
            delegation_id: "delegate-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "clear".into(),
            status: crate::app::DelegateStatus::InProgress,
            stats: crate::app::DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: crate::app::DelegateChildState::None,
        });
        app.pending_delegate_child_states.insert(
            "child".into(),
            crate::app::DelegateChildState::OtherProgress,
        );
        app.pending_delegate_tool_calls
            .push(crate::app::PendingDelegateToolCall {
                tool_call_id: "tool-1".into(),
                target_agent_id: None,
                objective: "clear".into(),
            });

        app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "root".into(),
            profile_id: None,
        });

        assert_eq!(app.parent_session_id, None);
        assert_eq!(app.screen, Screen::Chat);
        assert!(app.delegate_entries.is_empty());
        assert!(app.pending_delegate_child_states.is_empty());
        assert!(app.pending_delegate_tool_calls.is_empty());
    }

    #[test]
    fn native_provider_changed_updates_selection_and_preserves_limit_when_absent() {
        let mut app = App::new();
        app.context_limit = 4_096;

        app.handle_acp_event(AcpAppEvent::ProviderChanged {
            provider: "remote-provider".into(),
            model: "model-1".into(),
            context_limit: Some(128_000),
            provider_node_id: Some("node-1".into()),
        });
        assert_eq!(app.current_provider.as_deref(), Some("remote-provider"));
        assert_eq!(app.current_model.as_deref(), Some("model-1"));
        assert_eq!(app.context_limit, 128_000);
        assert_eq!(app.current_model_node_id.as_deref(), Some("node-1"));

        app.handle_acp_event(AcpAppEvent::ProviderChanged {
            provider: "local-provider".into(),
            model: "model-2".into(),
            context_limit: None,
            provider_node_id: None,
        });
        assert_eq!(app.current_provider.as_deref(), Some("local-provider"));
        assert_eq!(app.current_model.as_deref(), Some("model-2"));
        assert_eq!(app.context_limit, 128_000);
        assert_eq!(app.current_model_node_id, None);
    }

    #[test]
    fn native_agent_mode_updates_mode_and_clears_review_return_state() {
        let mut app = App::new();
        app.agent_mode = "review".into();
        app.mode_before_review = Some("plan".into());

        app.handle_acp_event(AcpAppEvent::AgentMode {
            mode: "build".into(),
        });
        assert_eq!(app.agent_mode, "build");
        assert_eq!(app.mode_before_review, None);

        app.mode_before_review = Some("plan".into());
        app.handle_acp_event(AcpAppEvent::AgentMode {
            mode: "review".into(),
        });
        assert_eq!(app.agent_mode, "review");
        assert_eq!(app.mode_before_review.as_deref(), Some("plan"));
    }

    #[test]
    fn native_session_updates_append_user_and_assistant_messages() {
        let mut app = app_with_active_session();

        apply_live_update(
            &mut app,
            AcpSessionUpdate::UserMessage {
                content: serde_json::json!({ "text": "hello" }),
                message_id: Some("u1".into()),
            },
        );
        apply_live_update(&mut app, final_assistant_update("world", None));

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { text, message_id: Some(user_id) },
                ChatEntry::Assistant { content, message_id: Some(assistant_id), .. }
            ] if text == "hello" && user_id == "u1" && content == "world" && assistant_id == "a1"
        ));
    }

    #[test]
    fn authoritative_user_chunk_reconciles_optimistic_prompt_before_elicitation() {
        let mut app = app_with_active_session();
        app.push_pending_prompt("Choose a deployment target".into());
        apply_live_update(
            &mut app,
            AcpSessionUpdate::ElicitationRequested {
                elicitation_id: "question-1".into(),
                message: "Which target?".into(),
                requested_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "selection": { "type": "string", "enum": ["staging"] }
                    },
                    "required": ["selection"]
                }),
                source: "builtin:question".into(),
                allow_custom: true,
            },
        );
        apply_live_update(
            &mut app,
            AcpSessionUpdate::UserMessage {
                content: serde_json::json!({ "text": "Choose a deployment target" }),
                message_id: Some("user-1".into()),
            },
        );

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { message_id: Some(message_id), .. },
                ChatEntry::Elicitation { elicitation_id, .. }
            ] if message_id == "user-1" && elicitation_id == "question-1"
        ));
    }

    #[test]
    fn authoritative_user_chunk_reconciles_prompt_with_surrounding_whitespace() {
        let mut app = app_with_active_session();
        app.push_pending_prompt("  first line\nsecond line\n  ".into());

        apply_live_update(
            &mut app,
            AcpSessionUpdate::UserMessage {
                content: serde_json::json!({ "text": "first line\nsecond line" }),
                message_id: Some("user-1".into()),
            },
        );

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "first line\nsecond line" && message_id == "user-1"
        ));
    }

    #[test]
    fn identical_normalized_optimistic_prompts_reconcile_in_submission_order() {
        let mut app = app_with_active_session();
        app.push_pending_prompt("repeat  \n".into());
        app.push_pending_prompt(" repeat".into());

        for message_id in ["user-1", "user-2"] {
            apply_live_update(
                &mut app,
                AcpSessionUpdate::UserMessage {
                    content: serde_json::json!({ "text": "repeat" }),
                    message_id: Some(message_id.into()),
                },
            );
        }

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { text: first_text, message_id: Some(first), .. },
                ChatEntry::User { text: second_text, message_id: Some(second), .. }
            ] if first_text == "repeat" && second_text == "repeat"
                && first == "user-1" && second == "user-2"
        ));
    }

    #[test]
    fn identical_optimistic_prompts_reconcile_in_submission_order() {
        let mut app = app_with_active_session();
        app.push_pending_prompt("repeat".into());
        app.push_pending_prompt("repeat".into());

        for message_id in ["user-1", "user-2"] {
            apply_live_update(
                &mut app,
                AcpSessionUpdate::UserMessage {
                    content: serde_json::json!({ "text": "repeat" }),
                    message_id: Some(message_id.into()),
                },
            );
        }

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { message_id: Some(first), .. },
                ChatEntry::User { message_id: Some(second), .. }
            ] if first == "user-1" && second == "user-2"
        ));
    }

    #[test]
    fn assistant_message_id_change_finalizes_previous_streaming_segment() {
        let mut app = app_with_active_session();
        apply_live_update(
            &mut app,
            AcpSessionUpdate::AssistantContentDelta {
                content: "first".into(),
                message_id: Some("assistant-1".into()),
            },
        );
        apply_live_update(
            &mut app,
            AcpSessionUpdate::AssistantContentDelta {
                content: "second".into(),
                message_id: Some("assistant-2".into()),
            },
        );

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Assistant { content, message_id: Some(message_id), .. }]
                if content == "first" && message_id == "assistant-1"
        ));
        assert_eq!(app.streaming_content, "second");
        assert_eq!(
            app.streaming_content_message_id.as_deref(),
            Some("assistant-2")
        );
    }

    #[test]
    fn tool_start_finalizes_streaming_assistant_before_tool() {
        let mut app = app_with_active_session();
        apply_live_update(
            &mut app,
            AcpSessionUpdate::AssistantContentDelta {
                content: "Let me check.".into(),
                message_id: Some("assistant-1".into()),
            },
        );
        apply_live_update(
            &mut app,
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "pwd" })),
            },
        );

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::Assistant { content, message_id: Some(message_id), .. },
                ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), .. }
            ] if content == "Let me check." && message_id == "assistant-1" && tool_call_id == "tool-1"
        ));
        assert!(app.streaming_content.is_empty());
    }

    #[test]
    fn repeated_terminal_replay_thinking_delta_is_idempotent() {
        let mut app = app_with_active_session();
        let updates = vec![AcpSessionUpdate::AssistantThinkingDelta {
            content: "inspect files".into(),
            message_id: Some("assistant-1".into()),
        }];

        for _ in 0..2 {
            app.handle_acp_event(AcpAppEvent::SessionReplay {
                session_id: TEST_SESSION_ID.into(),
                updates: updates.clone(),
            });
        }

        assert_eq!(app.streaming_thinking, "inspect files");
        assert_eq!(
            app.streaming_thinking_message_id.as_deref(),
            Some("assistant-1")
        );
        assert!(app.messages.is_empty());
    }

    #[test]
    fn replay_thinking_chunks_with_matching_id_are_not_suppressed() {
        let mut app = app_with_active_session();

        for content in ["hel", "lo"] {
            app.handle_acp_event(AcpAppEvent::SessionReplay {
                session_id: TEST_SESSION_ID.into(),
                updates: vec![AcpSessionUpdate::AssistantThinkingDelta {
                    content: content.into(),
                    message_id: Some("assistant-1".into()),
                }],
            });
        }

        assert_eq!(app.streaming_thinking, "hello");
        assert_eq!(
            app.streaming_thinking_message_id.as_deref(),
            Some("assistant-1")
        );
    }

    #[test]
    fn repeated_replay_thinking_deltas_preserve_tool_boundaries() {
        let mut app = app_with_active_session();
        let updates = vec![
            AcpSessionUpdate::AssistantThinkingDelta {
                content: "inspect ".into(),
                message_id: Some("assistant-1".into()),
            },
            AcpSessionUpdate::AssistantThinkingDelta {
                content: "files".into(),
                message_id: Some("assistant-1".into()),
            },
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "pwd" })),
            },
        ];

        for _ in 0..2 {
            app.handle_acp_event(AcpAppEvent::SessionReplay {
                session_id: TEST_SESSION_ID.into(),
                updates: updates.clone(),
            });
        }

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::Thinking { content, message_id: Some(message_id) },
                ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), .. },
            ] if content == "inspect files" && message_id == "assistant-1" && tool_call_id == "tool-1"
        ));
    }

    #[test]
    fn final_assistant_replaces_matching_thinking_without_crossing_tool() {
        let mut app = app_with_active_session();
        app.messages.push(ChatEntry::Thinking {
            content: "Need to inspect".into(),
            message_id: Some("assistant-1".into()),
        });
        app.messages.push(ChatEntry::ToolCall {
            tool_call_id: Some("tool-1".into()),
            name: "read_tool".into(),
            is_error: false,
            detail: ToolDetail::None,
        });

        apply_live_update(
            &mut app,
            AcpSessionUpdate::AssistantMessage {
                content: "Found it".into(),
                thinking: None,
                message_id: Some("assistant-1".into()),
            },
        );

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::Assistant { content, thinking: Some(thinking), .. },
                ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), .. }
            ] if content == "Found it" && thinking == "Need to inspect" && tool_call_id == "tool-1"
        ));
    }

    #[test]
    fn native_thinking_delta_updates_streaming_thinking() {
        let mut app = app_with_active_session();

        apply_live_update(
            &mut app,
            AcpSessionUpdate::AssistantThinkingDelta {
                content: "thinking".into(),
                message_id: Some(TEST_ASSISTANT_ID.into()),
            },
        );

        assert_eq!(app.streaming_thinking, "thinking");
        assert_eq!(app.streaming_thinking_message_id.as_deref(), Some("a1"));
    }

    #[test]
    fn native_final_message_preserves_streamed_thinking_and_content_after_finished() {
        let mut app = app_with_active_session();

        apply_live_update(&mut app, AcpSessionUpdate::TurnStarted);
        apply_live_update(
            &mut app,
            AcpSessionUpdate::AssistantThinkingDelta {
                content: "streamed thinking".into(),
                message_id: Some(TEST_ASSISTANT_ID.into()),
            },
        );
        apply_live_update(
            &mut app,
            AcpSessionUpdate::AssistantContentDelta {
                content: "visible stream".into(),
                message_id: Some(TEST_ASSISTANT_ID.into()),
            },
        );
        apply_live_update(&mut app, final_assistant_update("final answer", None));
        apply_live_update(
            &mut app,
            AcpSessionUpdate::Finished {
                finish_reason: "EndTurn".into(),
            },
        );

        assert!(app.streaming_content.is_empty());
        assert!(app.streaming_thinking.is_empty());
        assert_single_assistant(
            &app,
            "final answer",
            Some("streamed thinking"),
            TEST_ASSISTANT_ID,
        );
    }

    #[test]
    fn native_final_message_replaces_matching_thinking_entry() {
        let mut app = app_with_active_session();
        push_thinking_entry(&mut app, "existing thinking");

        apply_live_update(&mut app, final_assistant_update("final answer", None));

        assert_single_assistant(
            &app,
            "final answer",
            Some("existing thinking"),
            TEST_ASSISTANT_ID,
        );
    }

    #[test]
    fn native_final_message_prefers_explicit_thinking_when_replacing_thinking_entry() {
        let mut app = app_with_active_session();
        push_thinking_entry(&mut app, "existing thinking");

        apply_live_update(
            &mut app,
            final_assistant_update("final answer", Some("explicit thinking")),
        );

        assert_single_assistant(
            &app,
            "final answer",
            Some("explicit thinking"),
            TEST_ASSISTANT_ID,
        );
    }

    #[test]
    fn native_duplicate_final_assistant_message_is_not_appended() {
        let mut app = app_with_active_session();
        app.messages.push(ChatEntry::Assistant {
            content: "first answer".into(),
            thinking: Some("first thinking".into()),
            message_id: Some(TEST_ASSISTANT_ID.into()),
        });

        apply_live_update(
            &mut app,
            final_assistant_update("duplicate answer", Some("duplicate thinking")),
        );

        assert_single_assistant(
            &app,
            "first answer",
            Some("first thinking"),
            TEST_ASSISTANT_ID,
        );
    }

    #[test]
    fn native_tool_call_start_keeps_shell_details() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({
                    "command": "cargo check --examples",
                    "workdir": "/repo"
                })),
            },
        });

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::ToolCall {
                detail: ToolDetail::Shell { command, workdir, .. },
                ..
            }] if command == "cargo check --examples" && workdir.as_deref() == Some("/repo")
        ));
        assert_eq!(app.session_stats.total_tool_calls, 1);
    }

    #[test]
    fn native_usage_update_updates_status_metrics() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::UsageUpdate {
                used: 2048,
                size: 8192,
                cost_usd: Some(0.0123),
            },
        });

        assert_eq!(app.session_stats.latest_context_tokens, Some(2048));
        assert_eq!(app.context_limit, 8192);
        assert_eq!(app.cumulative_cost, Some(0.0123));
        assert!(matches!(
            app.logs.last(),
            Some(entry)
                if entry.level == LogLevel::Info
                    && entry.target == "usage"
                    && entry.message == "usage: context 2048/8192 tokens (25%), cost $0.0123"
        ));
    }

    #[test]
    fn native_timing_update_adds_active_time() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: true,
            update: AcpSessionUpdate::TimingUpdate { duration_secs: 42 },
        });

        assert_eq!(
            app.llm_request_elapsed(),
            Some(std::time::Duration::from_secs(42))
        );
        assert!(matches!(
            app.logs.last(),
            Some(entry)
                if entry.level == LogLevel::Info
                    && entry.target == "usage"
                    && entry.message == "usage: active time 42s"
        ));
    }

    #[test]
    fn native_live_turn_updates_realtime_elapsed() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::TurnStarted,
        });

        assert!(app.session_stats.open_llm_request_instant.is_some());
        app.session_stats.open_llm_request_instant = app
            .session_stats
            .open_llm_request_instant
            .map(|started| started - std::time::Duration::from_secs(2));
        assert!(app.llm_request_elapsed().is_some_and(|elapsed| {
            elapsed >= std::time::Duration::from_secs(2)
                && elapsed < std::time::Duration::from_secs(3)
        }));

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::Finished {
                finish_reason: "EndTurn".into(),
            },
        });

        assert!(app.session_stats.open_llm_request_instant.is_none());
        assert!(app.llm_request_elapsed().is_some_and(|elapsed| {
            elapsed >= std::time::Duration::from_secs(2)
                && elapsed < std::time::Duration::from_secs(3)
        }));
    }

    #[test]
    fn native_replay_turn_does_not_start_realtime_elapsed() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: true,
            update: AcpSessionUpdate::TurnStarted,
        });

        assert!(app.session_stats.open_llm_request_instant.is_none());
        assert_eq!(app.llm_request_elapsed(), None);
    }

    #[test]
    fn native_prompt_failure_removes_only_matching_optimistic_prompt() {
        let mut app = app_with_active_session();
        let first = app.push_pending_prompt("first".into());
        let second = app.push_pending_prompt("second".into());

        app.handle_acp_event(AcpAppEvent::PromptFailed {
            local_id: first,
            message: "ACP prompt failed".into(),
        });

        assert!(matches!(
            app.messages.first(),
            Some(ChatEntry::User { text, message_id: Some(message_id) })
                if text == "second" && message_id == &second
        ));
        assert_eq!(
            app.messages
                .iter()
                .filter(|entry| matches!(entry, ChatEntry::User { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn native_acp_error_closes_realtime_elapsed() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::TurnStarted,
        });
        app.session_stats.open_llm_request_instant = app
            .session_stats
            .open_llm_request_instant
            .map(|started| started - std::time::Duration::from_secs(3));
        app.handle_acp_event(AcpAppEvent::Error {
            message: "prompt failed".into(),
        });

        assert!(app.session_stats.open_llm_request_instant.is_none());
        assert!(app.llm_request_elapsed().is_some_and(|elapsed| {
            elapsed >= std::time::Duration::from_secs(3)
                && elapsed < std::time::Duration::from_secs(4)
        }));
    }

    #[test]
    fn native_tool_call_end_updates_shell_output_tail() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());
        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({
                    "command": "cargo check",
                    "workdir": "/repo"
                })),
            },
        });

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::ToolCallEnd {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                is_error: false,
                result: Some("Checking qmtui\nFinished dev profile".into()),
            },
        });

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::ToolCall {
                detail: ToolDetail::Shell { output_tail: Some(tail), .. },
                ..
            }] if tail.lines.iter().any(|line| line.contains("Finished dev profile"))
        ));
    }

    #[test]
    fn native_tool_call_start_keeps_read_tool_range() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "read_tool".into(),
                arguments: Some(serde_json::json!({
                    "path": "src/main.rs",
                    "offset": 9,
                    "limit": 5
                })),
            },
        });

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::ToolCall {
                detail: ToolDetail::ReadTool { path, start_line: Some(10), end_line: Some(14) },
                ..
            }] if path == "src/main.rs"
        ));
    }

    #[test]
    fn native_undo_result_success_updates_state_and_reloads_session() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());
        app.activity = ActivityState::SessionOp(crate::app::SessionOp::Undo);
        app.undoable_turns.push(crate::app::UndoableTurn {
            turn_id: "u1".into(),
            message_id: "u1".into(),
            text: "change".into(),
        });

        let replies = app.handle_acp_event(AcpAppEvent::UndoResult(UndoResultData {
            success: true,
            message_id: Some("u1".into()),
            reverted_files: vec!["src/main.rs".into()],
            message: None,
            undo_stack: vec![UndoStackFrame {
                message_id: "u1".into(),
            }],
        }));

        assert!(matches!(app.activity, ActivityState::Idle));
        assert_eq!(app.status, "undone - reloading session");
        assert!(app.can_redo());
        assert!(matches!(
            replies.as_slice(),
            [ClientMsg::LoadSession { session_id, cwd }] if session_id == "session-1" && cwd.is_none()
        ));
    }

    #[test]
    fn native_redo_result_failure_clears_pending_state_and_logs_warning() {
        let mut app = App::new();
        app.activity = ActivityState::SessionOp(crate::app::SessionOp::Redo);

        let replies = app.handle_acp_event(AcpAppEvent::RedoResult(RedoResultData {
            success: false,
            message: Some("Nothing to redo".into()),
            undo_stack: Vec::new(),
        }));

        assert!(matches!(app.activity, ActivityState::Idle));
        assert_eq!(app.status, "Nothing to redo");
        assert!(replies.is_empty());
        assert!(matches!(
            app.logs.last(),
            Some(entry) if entry.level == LogLevel::Warn && entry.target == "session"
        ));
    }

    #[test]
    fn native_fork_result_success_loads_forked_session() {
        let mut app = App::new();
        app.agent_id = Some("agent-1".into());
        app.popup = Popup::ForkTurnSelect;
        app.pending_fork_message_id = Some("msg-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::ForkResult(ForkResultData {
            success: true,
            source_session_id: Some("source-1".into()),
            forked_session_id: Some("fork-1".into()),
            message: None,
        }));

        assert_eq!(app.pending_fork_message_id, None);
        assert_eq!(app.popup, Popup::None);
        assert_eq!(app.status, "forked - loading session");
        assert!(matches!(
            replies.as_slice(),
            [
                ClientMsg::LoadSession { session_id: load_id, .. },
                ClientMsg::SubscribeSession { session_id: subscribe_id, agent_id }
            ] if load_id == "fork-1" && subscribe_id == "fork-1" && agent_id.as_deref() == Some("agent-1")
        ));
    }

    #[test]
    fn native_fork_result_failure_clears_pending_and_keeps_popup() {
        let mut app = App::new();
        app.popup = Popup::ForkTurnSelect;
        app.pending_fork_message_id = Some("msg-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::ForkResult(ForkResultData {
            success: false,
            source_session_id: Some("source-1".into()),
            forked_session_id: None,
            message: Some("fork failed".into()),
        }));

        assert!(replies.is_empty());
        assert_eq!(app.pending_fork_message_id, None);
        assert_eq!(app.popup, Popup::ForkTurnSelect);
        assert_eq!(app.status, "fork failed");
    }

    #[test]
    fn native_undo_stack_hydrates_redo_state() {
        let mut app = App::new();

        app.handle_acp_event(AcpAppEvent::UndoStack(vec![UndoStackFrame {
            message_id: "u1".into(),
        }]));

        assert!(app.can_redo());
    }

    #[test]
    fn native_session_update_for_other_session_marks_activity_only() {
        let mut app = App::new();
        app.session_id = Some("active".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "other".into(),
            is_replay: false,
            update: AcpSessionUpdate::AssistantMessage {
                content: "hidden".into(),
                thinking: None,
                message_id: None,
            },
        });

        assert!(app.messages.is_empty());
        assert!(app.session_activity.contains_key("other"));
    }

    #[test]
    fn native_session_replay_applies_history_as_one_event() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: "session-1".into(),
            updates: vec![
                AcpSessionUpdate::UserMessage {
                    content: serde_json::json!({ "text": "hello" }),
                    message_id: Some("u1".into()),
                },
                AcpSessionUpdate::AssistantMessage {
                    content: "world".into(),
                    thinking: None,
                    message_id: Some("a1".into()),
                },
            ],
        });

        assert!(replies.is_empty());
        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { text, message_id: Some(user_id) },
                ChatEntry::Assistant { content, message_id: Some(assistant_id), .. }
            ] if text == "hello" && user_id == "u1" && content == "world" && assistant_id == "a1"
        ));
    }

    #[test]
    fn native_session_replay_coalesces_deltas_without_crossing_tool_order() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: "session-1".into(),
            updates: vec![
                AcpSessionUpdate::UserMessage {
                    content: serde_json::json!({ "text": "run it" }),
                    message_id: Some("u1".into()),
                },
                AcpSessionUpdate::AssistantContentDelta {
                    content: "before ".into(),
                    message_id: Some("a1".into()),
                },
                AcpSessionUpdate::AssistantContentDelta {
                    content: "tool".into(),
                    message_id: Some("a1".into()),
                },
                AcpSessionUpdate::ToolCallStart {
                    tool_call_id: Some("tool-1".into()),
                    name: "shell".into(),
                    arguments: Some(serde_json::json!({ "command": "echo ok" })),
                },
                AcpSessionUpdate::ToolCallEnd {
                    tool_call_id: Some("tool-1".into()),
                    name: "shell".into(),
                    is_error: false,
                    result: Some("ok".into()),
                },
                AcpSessionUpdate::AssistantContentDelta {
                    content: "after".into(),
                    message_id: Some("a2".into()),
                },
            ],
        });

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { .. },
                ChatEntry::Assistant { content: before, .. },
                ChatEntry::ToolCall { tool_call_id: Some(tool_id), .. },
                ChatEntry::Assistant { content: after, .. },
            ] if before == "before tool" && tool_id == "tool-1" && after == "after"
        ));
    }

    #[test]
    fn native_session_replay_prefers_final_assistant_message() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: "session-1".into(),
            updates: vec![
                AcpSessionUpdate::AssistantContentDelta {
                    content: "partial".into(),
                    message_id: Some("a1".into()),
                },
                AcpSessionUpdate::AssistantThinkingDelta {
                    content: "draft".into(),
                    message_id: Some("a1".into()),
                },
                AcpSessionUpdate::AssistantMessage {
                    content: "final".into(),
                    thinking: Some("final thinking".into()),
                    message_id: Some("a1".into()),
                },
            ],
        });

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(message_id) }]
                if content == "final" && thinking == "final thinking" && message_id == "a1"
        ));
    }

    #[test]
    fn native_session_replay_merges_adjacent_assistant_chunks_from_loading_notifications() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: "session-1".into(),
            updates: vec![
                AcpSessionUpdate::AssistantMessage {
                    content: "hel".into(),
                    thinking: None,
                    message_id: Some("a1".into()),
                },
                AcpSessionUpdate::AssistantMessage {
                    content: "lo".into(),
                    thinking: None,
                    message_id: Some("a1".into()),
                },
                AcpSessionUpdate::ToolCallStart {
                    tool_call_id: Some("tool-1".into()),
                    name: "shell".into(),
                    arguments: Some(serde_json::json!({ "command": "echo ok" })),
                },
                AcpSessionUpdate::AssistantMessage {
                    content: "after".into(),
                    thinking: None,
                    message_id: Some("a2".into()),
                },
            ],
        });

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::Assistant { content: first, message_id: Some(first_id), .. },
                ChatEntry::ToolCall { .. },
                ChatEntry::Assistant { content: second, message_id: Some(second_id), .. },
            ] if first == "hello" && first_id == "a1" && second == "after" && second_id == "a2"
        ));
    }

    #[test]
    fn native_duplicate_user_message_id_is_not_appended() {
        let mut app = app_with_active_session();
        let update = AcpSessionUpdate::UserMessage {
            content: serde_json::json!({ "text": "hello" }),
            message_id: Some("u1".into()),
        };

        apply_live_update(&mut app, update.clone());
        apply_live_update(&mut app, update);

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "hello" && message_id == "u1"
        ));
    }

    #[test]
    fn native_duplicate_thinking_only_assistant_is_not_appended() {
        let mut app = app_with_active_session();
        let update = AcpSessionUpdate::AssistantMessage {
            content: String::new(),
            thinking: Some("checking".into()),
            message_id: Some("a1".into()),
        };

        apply_live_update(&mut app, update.clone());
        apply_live_update(&mut app, update);

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Thinking { content, message_id: Some(message_id) }]
                if content == "checking" && message_id == "a1"
        ));
    }

    #[test]
    fn native_duplicate_acp_error_is_not_appended() {
        let mut app = app_with_active_session();
        let event = AcpAppEvent::Error {
            message: "connection lost".into(),
        };

        app.handle_acp_event(event.clone());
        app.handle_acp_event(event);

        assert!(
            matches!(app.messages.as_slice(), [ChatEntry::Error(message)] if message == "connection lost")
        );
    }

    #[test]
    fn native_failed_tool_end_before_start_reconciles_when_start_arrives() {
        let mut app = app_with_active_session();

        apply_live_update(&mut app, failed_tool_end("tool-1", "shell"));
        apply_live_update(
            &mut app,
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "echo late" })),
            },
        );

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), name, is_error: true, detail: ToolDetail::Shell { command, .. } }]
                if tool_call_id == "tool-1" && name == "shell" && command == "echo late"
        ));
    }

    #[test]
    fn native_repeated_failed_tool_end_dedupes_fallback_by_call_id_and_name() {
        let mut app = app_with_active_session();

        apply_live_update(&mut app, failed_tool_end("tool-1", "shell"));
        apply_live_update(&mut app, failed_tool_end("tool-1", "shell"));
        apply_live_update(&mut app, failed_tool_end("tool-2", "shell"));

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::ToolCall { tool_call_id: Some(first), name: first_name, is_error: true, .. },
                ChatEntry::ToolCall { tool_call_id: Some(second), name: second_name, is_error: true, .. },
            ] if first == "tool-1" && first_name == "shell (failed)"
                && second == "tool-2" && second_name == "shell (failed)"
        ));
    }

    #[test]
    fn native_live_elicitation_creates_card_and_active_state() {
        let mut app = app_with_active_session();

        apply_live_update(
            &mut app,
            AcpSessionUpdate::ElicitationRequested {
                elicitation_id: "elic-1".into(),
                message: "Choose a target".into(),
                requested_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "target": { "type": "string", "enum": ["staging"] } },
                    "required": ["target"]
                }),
                source: "builtin:question".into(),
                allow_custom: true,
            },
        );

        assert_eq!(
            app.elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-1")
        );
        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, message, outcome: None, .. }]
                if elicitation_id == "elic-1" && message == "Choose a target"
        ));
    }

    #[test]
    fn native_replay_elicitation_creates_card_without_active_state() {
        let mut app = app_with_active_session();

        app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: TEST_SESSION_ID.into(),
            updates: vec![AcpSessionUpdate::ElicitationRequested {
                elicitation_id: "elic-1".into(),
                message: "Choose a target".into(),
                requested_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "target": { "type": "string", "enum": ["staging"] } },
                    "required": ["target"]
                }),
                source: "builtin:question".into(),
                allow_custom: true,
            }],
        });

        assert!(app.elicitation.is_none());
        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, message, outcome: None, .. }]
                if elicitation_id == "elic-1" && message == "Choose a target"
        ));
    }

    #[test]
    fn native_repeated_replay_elicitation_preserves_existing_outcome() {
        let mut app = app_with_active_session();
        let updates = vec![AcpSessionUpdate::ElicitationRequested {
            elicitation_id: "elic-1".into(),
            message: "Choose a target".into(),
            requested_schema: serde_json::json!({
                "type": "object",
                "properties": { "target": { "type": "string", "enum": ["staging"] } },
                "required": ["target"]
            }),
            source: "builtin:question".into(),
            allow_custom: true,
        }];

        for _ in 0..2 {
            app.handle_acp_event(AcpAppEvent::SessionReplay {
                session_id: TEST_SESSION_ID.into(),
                updates: updates.clone(),
            });
        }
        app.resolve_elicitation("elic-1", "staging");
        app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: TEST_SESSION_ID.into(),
            updates,
        });

        assert!(app.elicitation.is_none());
        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: Some(outcome), .. }]
                if elicitation_id == "elic-1" && outcome == "staging"
        ));
    }

    #[test]
    fn native_live_elicitation_is_not_reopened_by_replay_and_distinct_ids_append() {
        let mut app = app_with_active_session();
        let elicitation = |id: &str| AcpSessionUpdate::ElicitationRequested {
            elicitation_id: id.into(),
            message: "Choose a target".into(),
            requested_schema: serde_json::json!({
                "type": "object",
                "properties": { "target": { "type": "string", "enum": ["staging"] } },
                "required": ["target"]
            }),
            source: "builtin:question".into(),
            allow_custom: true,
        };

        apply_live_update(&mut app, elicitation("elic-live"));
        app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: TEST_SESSION_ID.into(),
            updates: vec![elicitation("elic-live"), elicitation("elic-replayed")],
        });

        assert_eq!(
            app.elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-live")
        );
        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::Elicitation { elicitation_id: first, .. },
                ChatEntry::Elicitation { elicitation_id: second, .. },
            ] if first == "elic-live" && second == "elic-replayed"
        ));
    }

    #[test]
    fn native_unsupported_replay_elicitation_is_idempotent() {
        let mut app = app_with_active_session();
        let updates = vec![AcpSessionUpdate::ElicitationRequested {
            elicitation_id: "elic-unsupported".into(),
            message: "Upload a file".into(),
            requested_schema: serde_json::json!({ "type": "array" }),
            source: "extension:file-picker".into(),
            allow_custom: false,
        }];

        for _ in 0..2 {
            app.handle_acp_event(AcpAppEvent::SessionReplay {
                session_id: TEST_SESSION_ID.into(),
                updates: updates.clone(),
            });
        }

        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: Some(outcome), .. }]
                if elicitation_id == "elic-unsupported"
                    && outcome == "unsupported schema - cannot answer in TUI"
        ));
    }

    #[test]
    fn native_session_replay_is_idempotent() {
        let mut app = app_with_active_session();
        let updates = vec![
            AcpSessionUpdate::UserMessage {
                content: serde_json::json!({ "text": "hello" }),
                message_id: Some("u1".into()),
            },
            AcpSessionUpdate::AssistantMessage {
                content: "world".into(),
                thinking: None,
                message_id: Some("a1".into()),
            },
        ];

        for _ in 0..2 {
            app.handle_acp_event(AcpAppEvent::SessionReplay {
                session_id: TEST_SESSION_ID.into(),
                updates: updates.clone(),
            });
        }

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { message_id: Some(user_id), .. },
                ChatEntry::Assistant { message_id: Some(assistant_id), .. },
            ] if user_id == "u1" && assistant_id == "a1"
        ));
    }
}
