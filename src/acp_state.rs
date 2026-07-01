use serde_json::Value;

use crate::app::{ActivityState, ChatEntry, ElicitationState, LogLevel, Screen, ToolDetail};
use crate::protocol::{
    AgentInfo, AuthProviderEntry, ClientMsg, ModelEntry, OAuthFlowData, OAuthResultData,
    ProfileInfo, SessionGroup,
};

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
    ElicitationRequested {
        elicitation_id: String,
        message: String,
        requested_schema: Value,
        source: String,
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
    MeshNodes(Value),
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
    Models {
        models: Vec<ModelEntry>,
        meta: Option<AcpModelsMetaInfo>,
    },
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
            AcpAppEvent::MeshNodes(data) => {
                let count = data
                    .get("nodes")
                    .and_then(Value::as_array)
                    .map(|nodes| nodes.len() as u32)
                    .unwrap_or(0);
                self.mesh_node_count = Some(count);
                self.push_log(LogLevel::Info, "mesh", format!("mesh nodes: {count}"));
                vec![]
            }
            AcpAppEvent::SessionList {
                groups,
                total_count,
                ..
            } => {
                self.apply_acp_session_list(groups, total_count);
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
                self.push_acp_error(&message);
                self.set_status(LogLevel::Error, "acp", format!("error: {message}"));
                vec![]
            }
        }
    }

    fn apply_acp_session_list(&mut self, mut groups: Vec<SessionGroup>, total_count: Option<u64>) {
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
        self.streaming_content.clear();
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
                self.activity = ActivityState::Thinking;
                self.streaming_content.clear();
                self.invalidate_streaming_caches();
                self.set_status(LogLevel::Debug, "activity", "thinking...");
            }
            AcpSessionUpdate::UserMessage {
                content,
                message_id,
            } => self.push_acp_user_message(content, message_id, is_replay),
            AcpSessionUpdate::AssistantContentDelta { content, .. } => {
                if self.is_turn_active() {
                    self.activity = ActivityState::Streaming;
                }
                self.streaming_content.push_str(&content);
            }
            AcpSessionUpdate::AssistantThinkingDelta {
                content,
                message_id,
            } => {
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
                tool_call_id, name, ..
            } => {
                if self.suppress_turn_output {
                    return;
                }
                self.activity = ActivityState::RunningTool { name: name.clone() };
                self.set_status(LogLevel::Debug, "tool", format!("tool: {name}"));
                if name != "question" {
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
                        detail: ToolDetail::None,
                    });
                }
            }
            AcpSessionUpdate::ToolCallEnd {
                tool_call_id,
                name,
                is_error,
                result,
            } => {
                if let Some(entry) = self.messages.iter_mut().rev().find(|entry| {
                    matches!(
                        entry,
                        ChatEntry::ToolCall {
                            tool_call_id: existing,
                            name: existing_name,
                            ..
                        } if existing.as_ref() == tool_call_id.as_ref() || existing_name == &name
                    )
                }) {
                    if let ChatEntry::ToolCall { is_error: e, .. } = entry {
                        *e = is_error;
                    }
                } else if is_error {
                    self.messages.push(ChatEntry::ToolCall {
                        tool_call_id,
                        name: format!("{name} (failed)"),
                        is_error: true,
                        detail: result.map(ToolDetail::Summary).unwrap_or(ToolDetail::None),
                    });
                }
                self.card_cache.invalidate();
            }
            AcpSessionUpdate::ElicitationRequested {
                elicitation_id,
                message,
                requested_schema,
                source,
            } => {
                self.handle_acp_elicitation_requested(
                    &elicitation_id,
                    &message,
                    &source,
                    &requested_schema,
                );
            }
            AcpSessionUpdate::Cancelled => {
                self.activity = ActivityState::Idle;
                self.streaming_content.clear();
                self.invalidate_streaming_caches();
                self.set_status(LogLevel::Warn, "activity", "cancelled");
            }
            AcpSessionUpdate::Finished { finish_reason } => {
                self.activity = ActivityState::Idle;
                self.streaming_content.clear();
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
        if !is_replay && (self.undo_state.is_some() || self.suppress_turn_output) {
            return;
        }
        self.suppress_turn_output = false;
        self.messages.push(ChatEntry::User {
            text: text.clone(),
            message_id: message_id.clone(),
        });
        self.recent_prompt_text = Some(text.clone());
        if let Some(message_id) = message_id
            && !self
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
        let streaming_thinking_message_id = self.streaming_thinking_message_id.clone();
        let thinking_text = thinking.filter(|text| !text.is_empty()).or_else(|| {
            (!self.streaming_thinking.is_empty())
                .then(|| std::mem::take(&mut self.streaming_thinking))
        });
        let thinking_message_id = message_id.clone().or(streaming_thinking_message_id);
        self.streaming_content.clear();
        self.invalidate_streaming_caches();
        if self.is_turn_active() {
            self.activity = ActivityState::Thinking;
        }
        if content.is_empty() && thinking_text.is_none() {
            return;
        }
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
                selected: std::collections::HashMap::new(),
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

    #[test]
    fn native_session_created_resets_view_and_subscribes() {
        let mut app = App::new();
        app.messages.push(ChatEntry::Error("stale".into()));

        let replies = app.handle_acp_event(AcpAppEvent::SessionCreated {
            agent_id: "agent-1".into(),
            session_id: "session-1".into(),
            profile_id: Some("code".into()),
        });

        assert_eq!(app.session_id.as_deref(), Some("session-1"));
        assert_eq!(app.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(app.screen, Screen::Chat);
        assert!(app.messages.is_empty());
        assert!(matches!(
            replies.as_slice(),
            [ClientMsg::SubscribeSession { session_id, agent_id }]
                if session_id == "session-1" && agent_id.as_deref() == Some("agent-1")
        ));
    }

    #[test]
    fn native_session_updates_append_user_and_assistant_messages() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::UserMessage {
                content: serde_json::json!({ "text": "hello" }),
                message_id: Some("u1".into()),
            },
        });
        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::AssistantMessage {
                content: "world".into(),
                thinking: None,
                message_id: Some("a1".into()),
            },
        });

        assert!(matches!(
            app.messages.as_slice(),
            [
                ChatEntry::User { text, message_id: Some(user_id) },
                ChatEntry::Assistant { content, message_id: Some(assistant_id), .. }
            ] if text == "hello" && user_id == "u1" && content == "world" && assistant_id == "a1"
        ));
    }

    #[test]
    fn native_thinking_delta_updates_streaming_thinking() {
        let mut app = App::new();
        app.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::AssistantThinkingDelta {
                content: "thinking".into(),
                message_id: Some("a1".into()),
            },
        });

        assert_eq!(app.streaming_thinking, "thinking");
        assert_eq!(app.streaming_thinking_message_id.as_deref(), Some("a1"));
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
}
