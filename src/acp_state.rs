use std::collections::HashSet;

use serde_json::Value;

use crate::application::Effect;
use crate::auth_state::{AuthAction, AuthOutcome};
use crate::chat_state::{
    ChatAction, ChatCoordination, ChatOutcome, ChatToolAction, ChatToolOutcome, ChatToolTransition,
    ChatUsageAction, ChatUsageOutcome, ElicitationAction, ElicitationOutcome, HistoryAction,
    HistoryCoordination, HistoryOutcome, ToolStartPreparationTransition,
};
use crate::command::{Command, SessionListRequest};
use crate::delegates_state::{
    DelegateAction, DelegateChildActivity, DelegateContext, DelegateCoordination, DelegateOutcome,
};
use crate::diagnostics::LogLevel;
use crate::domain::activity::{ActivityState, DelegationUpdate};
use crate::domain::auth::{AuthProviderEntry, OAuthFlow, OAuthResult};
use crate::domain::mesh::{
    MeshInviteCreatedInfo, MeshNodesInfo, MeshStatusInfo, RemoteSessionAttachInfo,
    RemoteSessionListInfo,
};
use crate::domain::model::{DelegateModelOverrideInfo, ModelEntry};
use crate::domain::profile::{AgentInfo, ProfileInfo};
use crate::domain::session::{
    ForkResult, RedoResult, SessionListPage, UndoResult, UndoStackSnapshot,
};
use crate::models_state::{ModelAction, ModelContext, ModelCoordination, ModelOutcome};
use crate::navigation_state::{Popup, Screen};
use crate::profiles_state::{ProfileAction, ProfileOutcome};
use crate::session_state::{SessionAction, SessionCoordination, SessionOutcome};
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
    ProfileAgents {
        profile_id: String,
        agents: Vec<AgentInfo>,
    },
    DelegateModelSet {
        session_id: String,
        agent_id: String,
        model: Option<DelegateModelOverrideInfo>,
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
        request: SessionListRequest,
        page: SessionListPage,
    },
    SessionListFailed {
        request: SessionListRequest,
        message: String,
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
    DelegationUpdate(DelegationUpdate),
    DelegationReplay {
        session_id: String,
        updates: Vec<DelegationUpdate>,
    },
    Models {
        models: Vec<ModelEntry>,
        meta: Option<AcpModelsMetaInfo>,
    },
    UndoStack(UndoStackSnapshot),
    UndoResult(UndoResult),
    RedoResult(RedoResult),
    ForkResult(ForkResult),
    AuthProviders(Vec<AuthProviderEntry>),
    OAuthFlowStarted(OAuthFlow),
    OAuthResult(OAuthResult),
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
    pub(crate) fn handle_acp_event(&mut self, event: AcpAppEvent) -> Vec<Effect> {
        fn effects(commands: Vec<Command>) -> Vec<Effect> {
            commands.into_iter().map(Effect::Command).collect()
        }
        match event {
            AcpAppEvent::Initialized {
                agent_id,
                agent_name,
                profiles,
                active_profile_id,
                agent_mode,
                reasoning_effort,
            } => {
                // ACP initialization sends a placeholder empty catalog before
                // capability-gated profile discovery completes.
                if !profiles.is_empty() {
                    let discarded_effects = self.apply_profile_catalog(profiles, active_profile_id);
                    debug_assert!(!discarded_effects.iter().any(|effect| {
                        !matches!(effect, Effect::Command(Command::ListProfileAgents { .. }))
                    }));
                }
                let effects =
                    self.apply_session_action(SessionAction::InitializeAgentId(agent_id.clone()));
                debug_assert!(effects.is_empty());
                let effects = self.apply_model_action(
                    ModelAction::InitializePrimaryAgent(AgentInfo {
                        id: agent_id,
                        name: agent_name,
                        description: None,
                        capabilities: Vec::new(),
                    }),
                    ModelContext::default(),
                );
                debug_assert!(effects.is_empty());
                if let Some(mode) = agent_mode {
                    let effects = self.apply_session_action(SessionAction::ReplaceAgentMode(mode));
                    debug_assert!(effects.is_empty());
                }
                if let Some(effort) = reasoning_effort {
                    let effects = self.apply_model_action(
                        ModelAction::InitializeReasoningEffort(effort),
                        ModelContext::default(),
                    );
                    debug_assert!(effects.is_empty());
                }
                let effects = self.apply_auth_action(AuthAction::ClearUiNotice);
                debug_assert!(effects.is_empty());
                self.set_status(LogLevel::Info, "connection", "connected");
                vec![]
            }
            AcpAppEvent::AgentMode { mode } => {
                self.apply_session_action(SessionAction::ReplaceAgentMode(mode))
            }
            AcpAppEvent::ReasoningEffort { reasoning_effort } => self.apply_model_action(
                ModelAction::ReasoningEffort(reasoning_effort),
                ModelContext::default(),
            ),
            AcpAppEvent::Profiles {
                profiles,
                active_profile_id,
            } => self.apply_profile_catalog(profiles, active_profile_id),
            AcpAppEvent::ProfileAgents { profile_id, agents } => {
                let context = ModelContext {
                    desired_profile_id: self.desired_agents_profile_id().map(str::to_string),
                    root_session_id: self
                        .delegates
                        .parent_session_id
                        .is_none()
                        .then(|| self.sessions.session_id.clone())
                        .flatten(),
                };
                self.apply_model_action(
                    ModelAction::ReplaceProfileAgents { profile_id, agents },
                    context,
                )
            }
            AcpAppEvent::DelegateModelSet {
                session_id,
                agent_id,
                model,
            } => self.apply_model_action(
                ModelAction::DelegateModelSet {
                    session_id,
                    agent_id,
                    model_id: model.map(|model| model.model_id),
                },
                ModelContext::default(),
            ),
            AcpAppEvent::ProviderChanged {
                provider,
                model,
                context_limit,
                provider_node_id,
            } => self.apply_model_action(
                ModelAction::ProviderChanged {
                    provider,
                    model,
                    node_id: provider_node_id,
                    context_limit,
                },
                ModelContext::default(),
            ),
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
            AcpAppEvent::MeshNodes(nodes) => effects(self.apply_mesh_nodes(nodes)),
            AcpAppEvent::MeshInviteCreated(invite) => {
                self.apply_mesh_invite_created(invite);
                vec![]
            }
            AcpAppEvent::RemoteSessions(list) => {
                self.apply_remote_sessions(list);
                vec![]
            }
            AcpAppEvent::RemoteSessionAttached(attached) => {
                effects(self.apply_remote_session_attached(attached))
            }
            AcpAppEvent::SessionList { request, page } => {
                self.apply_session_action(SessionAction::CatalogPage { request, page })
            }
            AcpAppEvent::SessionListFailed { request, message } => {
                self.apply_session_action(SessionAction::CatalogFailed { request, message })
            }
            AcpAppEvent::SessionCreated {
                agent_id,
                session_id,
                profile_id,
            } => self.apply_session_action(SessionAction::Created {
                agent_id,
                session_id,
                profile_id,
            }),
            AcpAppEvent::SessionLoaded {
                agent_id,
                session_id,
                profile_id,
            } => self.apply_session_action(SessionAction::Loaded {
                agent_id,
                session_id,
                profile_id,
            }),
            AcpAppEvent::SessionUpdate {
                session_id,
                update,
                is_replay,
            } => self.apply_acp_session_update(&session_id, update, is_replay),
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
                let mut effects = Vec::new();
                for update in updates {
                    effects.extend(self.apply_acp_session_update(&session_id, update, true));
                }
                effects
            }
            AcpAppEvent::UndoStack(stack) => {
                self.apply_chat_history_action(HistoryAction::ReplaceStack(stack))
            }
            AcpAppEvent::UndoResult(result) => {
                self.apply_chat_history_action(HistoryAction::UndoCompleted(result))
            }
            AcpAppEvent::RedoResult(result) => {
                self.apply_chat_history_action(HistoryAction::RedoCompleted(result))
            }
            AcpAppEvent::ForkResult(result) => {
                self.apply_chat_history_action(HistoryAction::ForkCompleted(result))
            }
            AcpAppEvent::DelegationUpdate(update) => self.apply_acp_delegation_update(update),
            AcpAppEvent::DelegationReplay {
                session_id,
                updates,
            } => {
                let mut effects = Vec::new();
                if self.sessions.session_id.as_deref() == Some(session_id.as_str()) {
                    for update in updates {
                        effects.extend(self.apply_acp_delegation_update(update));
                    }
                }
                effects
            }
            AcpAppEvent::Models { models, meta } => {
                let meta = meta.unwrap_or_default();
                self.apply_model_action(
                    ModelAction::ReplaceCatalog {
                        models,
                        remote_node_count: meta.remote_node_count,
                        remote_timeout_count: meta.remote_timeout_count,
                    },
                    ModelContext::default(),
                )
            }
            AcpAppEvent::AuthProviders(providers) => {
                self.apply_auth_action(AuthAction::Providers(providers))
            }
            AcpAppEvent::OAuthFlowStarted(flow) => {
                self.apply_auth_action(AuthAction::OAuthFlowStarted(flow))
            }
            AcpAppEvent::OAuthResult(result) => {
                self.apply_auth_action(AuthAction::OAuthResult(result))
            }
            AcpAppEvent::InfoLog { target, message } => {
                self.push_log(LogLevel::Info, target, message);
                vec![]
            }
            AcpAppEvent::Error { message } => {
                self.apply_chat_action(ChatAction::AcpError { message })
            }
            AcpAppEvent::PromptFailed { local_id, message } => {
                self.apply_chat_action(ChatAction::BackendPromptFailed { local_id, message })
            }
        }
    }

    pub(crate) fn apply_auth_action(&mut self, action: AuthAction) -> Vec<Effect> {
        let AuthOutcome {
            diagnostics,
            effects,
        } = self.auth.reduce(action);
        for diagnostic in diagnostics {
            self.push_log(diagnostic.level, "auth", diagnostic.message);
        }
        effects
    }

    fn apply_model_action(&mut self, action: ModelAction, context: ModelContext) -> Vec<Effect> {
        let ModelOutcome {
            coordination,
            effects,
        } = self.models.reduce(action, context);
        for coordination in coordination {
            match coordination {
                ModelCoordination::Log {
                    level,
                    target,
                    message,
                } => self.push_log(level, target, message),
                ModelCoordination::Status {
                    level,
                    target,
                    message,
                } => self.set_status(level, target, message),
                ModelCoordination::SetContextLimit(limit) => self.chat.context_limit = limit,
            }
        }
        effects
    }

    fn apply_profile_catalog(
        &mut self,
        profiles: Vec<ProfileInfo>,
        backend_active_profile_id: Option<String>,
    ) -> Vec<Effect> {
        let ProfileOutcome {
            active_profile_changed,
        } = self.profiles.reduce(ProfileAction::ReplaceCatalog {
            profiles,
            backend_active_profile_id,
        });
        let context = ModelContext {
            desired_profile_id: self.desired_agents_profile_id().map(str::to_string),
            root_session_id: None,
        };
        self.apply_model_action(
            ModelAction::SynchronizeProfileCatalog {
                active_profile_changed,
            },
            context,
        )
    }

    fn apply_session_action(&mut self, action: SessionAction) -> Vec<Effect> {
        let SessionOutcome {
            coordination,
            mut effects,
        } = self.sessions.reduce(action);
        for coordination in coordination {
            match coordination {
                SessionCoordination::PushChatError(message) => {
                    self.chat.push_error(&message);
                }
                SessionCoordination::ClearDelegateParent => {
                    effects.extend(self.apply_delegate_action(DelegateAction::ClearNewRootParent));
                }
                SessionCoordination::SetChatIdle => self.chat.activity = ActivityState::Idle,
                SessionCoordination::ResolveDelegateParent(discovered_parent) => {
                    effects.extend(self.apply_delegate_action(
                        DelegateAction::ResolveLoadedSessionParent(discovered_parent),
                    ));
                }
                SessionCoordination::ApplyProfileBinding {
                    session_id,
                    profile_id,
                    remove_if_missing,
                } => {
                    if let Some(profile_id) = profile_id {
                        self.profiles.bind_session_profile(session_id, profile_id);
                    } else if remove_if_missing {
                        self.profiles.remove_session_profile(&session_id);
                    }
                }
                SessionCoordination::ResetActiveSessionView => self.reset_active_session_view(),
                SessionCoordination::SelectChat => self.navigation.screen = Screen::Chat,
                SessionCoordination::SelectLoadedSessionScreen => {
                    self.navigation.screen = if self.delegates.parent_session_id.is_some() {
                        Screen::Delegate
                    } else {
                        Screen::Chat
                    };
                }
                SessionCoordination::Status {
                    level,
                    target,
                    message,
                } => self.set_status(level, target, message),
                SessionCoordination::SynchronizeProfileCommands {
                    session_id,
                    root_only,
                } => {
                    if (!root_only || self.delegates.parent_session_id.is_none())
                        && let Some(profile_id) =
                            self.current_session_profile_id().map(str::to_string)
                    {
                        if self.models.agents_profile_id.as_deref() == Some(profile_id.as_str()) {
                            effects.extend(
                                self.delegate_model_commands_for_session(&session_id, &profile_id)
                                    .into_iter()
                                    .map(Effect::Command),
                            );
                        } else {
                            effects
                                .push(Effect::Command(Command::ListProfileAgents { profile_id }));
                        }
                    }
                }
            }
        }
        effects
    }

    fn apply_delegate_action(&mut self, action: DelegateAction) -> Vec<Effect> {
        let outcome = self.delegates.reduce(
            action,
            DelegateContext {
                active_session_id: self.sessions.session_id.clone(),
                inactive_activity_session_id: None,
            },
        );
        self.apply_delegate_outcome(outcome)
    }

    fn apply_inactive_delegate_action(
        &mut self,
        session_id: &str,
        activity: DelegateChildActivity,
    ) -> Vec<Effect> {
        self.sessions.note_session_activity(session_id);
        let inactive_activity_session_id = self
            .sessions
            .session_activity
            .contains_key(session_id)
            .then(|| session_id.to_string());
        let outcome = self.delegates.reduce(
            DelegateAction::InactiveChildActivity {
                session_id: session_id.to_string(),
                activity,
            },
            DelegateContext {
                active_session_id: self.sessions.session_id.clone(),
                inactive_activity_session_id,
            },
        );
        self.apply_delegate_outcome(outcome)
    }

    fn apply_delegate_outcome(&mut self, outcome: DelegateOutcome) -> Vec<Effect> {
        let DelegateOutcome {
            changed: _,
            coordination,
            effects,
        } = outcome;
        for coordination in coordination {
            match coordination {
                DelegateCoordination::InvalidateCardCache => {
                    self.render.invalidate_card_cache();
                }
            }
        }
        effects
    }

    pub(crate) fn apply_chat_action(&mut self, action: ChatAction) -> Vec<Effect> {
        let ChatOutcome {
            transition: _,
            coordination,
            effects,
        } = self.chat.reduce(action);
        self.apply_chat_coordination(coordination);
        effects
    }

    fn apply_chat_history_action(&mut self, action: HistoryAction) -> Vec<Effect> {
        let HistoryOutcome {
            transition: _,
            coordination,
            mut effects,
        } = self.chat.reduce_history(action);
        for coordination in coordination {
            match coordination {
                HistoryCoordination::InvalidateContentCache => {
                    self.render.invalidate_content_cache();
                }
                HistoryCoordination::Status {
                    level,
                    target,
                    message,
                } => self.set_status(level, target, message),
                HistoryCoordination::ClosePopup => self.navigation.popup = Popup::None,
                HistoryCoordination::ReloadActiveSession => {
                    if let Some(session_id) = self.sessions.session_id.clone() {
                        effects.extend(
                            Command::load_session_commands(
                                session_id,
                                self.current_session_cwd(),
                                self.sessions.agent_id.clone(),
                            )
                            .into_iter()
                            .map(Effect::Command),
                        );
                    }
                }
                HistoryCoordination::LoadForkedSession { session_id } => {
                    effects.extend(
                        Command::load_session_commands(
                            session_id,
                            self.current_session_cwd(),
                            self.sessions.agent_id.clone(),
                        )
                        .into_iter()
                        .map(Effect::Command),
                    );
                }
            }
        }
        effects
    }

    fn apply_chat_tool_action(
        &mut self,
        action: ChatToolAction,
    ) -> (ChatToolTransition, Vec<Effect>) {
        let ChatToolOutcome {
            transition,
            coordination,
            effects,
        } = self.chat.reduce_tool(action);
        self.apply_chat_coordination(coordination);
        (transition, effects)
    }

    fn apply_chat_usage_action(&mut self, action: ChatUsageAction) -> Vec<Effect> {
        let ChatUsageOutcome {
            transition: _,
            coordination,
            effects,
        } = self.chat.reduce_usage(action);
        self.apply_chat_coordination(coordination);
        effects
    }

    pub(crate) fn apply_chat_elicitation_action(
        &mut self,
        action: ElicitationAction,
    ) -> Vec<Effect> {
        let ElicitationOutcome {
            transition: _,
            coordination,
            effects,
        } = self.chat.reduce_elicitation(action);
        self.apply_chat_coordination(coordination);
        effects
    }

    fn apply_chat_coordination(&mut self, coordination: Vec<ChatCoordination>) {
        for coordination in coordination {
            match coordination {
                ChatCoordination::InvalidateContentCache => {
                    self.render.invalidate_content_cache();
                }
                ChatCoordination::InvalidateThinkingCache => {
                    self.render.invalidate_thinking_cache();
                }
                ChatCoordination::InvalidateCardCache => {
                    self.render.invalidate_card_cache();
                }
                ChatCoordination::Log {
                    level,
                    target,
                    message,
                } => self.push_log(level, target, message),
                ChatCoordination::Status {
                    level,
                    target,
                    message,
                } => self.set_status(level, target, message),
                ChatCoordination::RefreshTransientStatus => self.refresh_transient_status(),
            }
        }
    }

    fn reset_active_session_view(&mut self) {
        self.chat.reset_for_session_switch();
        self.render.invalidate_content_cache();
        self.render.invalidate_thinking_cache();
        self.render.invalidate_card_cache();
        let effects = self.apply_delegate_action(DelegateAction::ClearRootSessionState);
        debug_assert!(effects.is_empty());
        self.composer.reset_for_session_switch();
    }

    fn apply_acp_delegation_update(&mut self, update: DelegationUpdate) -> Vec<Effect> {
        self.apply_delegate_action(DelegateAction::LifecycleUpdate(update))
    }

    fn apply_acp_delegate_child_update(
        &mut self,
        session_id: &str,
        update: &AcpSessionUpdate,
    ) -> Vec<Effect> {
        let activity = match update {
            AcpSessionUpdate::ToolCallStart { .. } => DelegateChildActivity::ToolCallStarted,
            AcpSessionUpdate::AssistantMessage { message_id, .. } => {
                DelegateChildActivity::AssistantMessage {
                    message_id: message_id.clone(),
                }
            }
            AcpSessionUpdate::AssistantContentDelta { message_id, .. } => {
                DelegateChildActivity::AssistantContent {
                    message_id: message_id.clone(),
                }
            }
            AcpSessionUpdate::AssistantThinkingDelta { .. } | AcpSessionUpdate::TurnStarted => {
                DelegateChildActivity::Progress
            }
            AcpSessionUpdate::UserMessage { .. } => DelegateChildActivity::UserMessage,
            AcpSessionUpdate::UsageUpdate {
                used,
                size,
                cost_usd,
            } => DelegateChildActivity::Usage {
                used: *used,
                size: *size,
                cost_usd: *cost_usd,
            },
            AcpSessionUpdate::ElicitationRequested {
                elicitation_id,
                message,
                requested_schema,
                source,
                ..
            } => DelegateChildActivity::PendingElicitation {
                elicitation_id: elicitation_id.clone(),
                message: message.clone(),
                requested_schema: requested_schema.clone(),
                source: source.clone(),
            },
            AcpSessionUpdate::ToolCallEnd { name, .. } if name == "question" => {
                DelegateChildActivity::QuestionToolFinished
            }
            AcpSessionUpdate::ToolCallEnd { .. }
            | AcpSessionUpdate::TimingUpdate { .. }
            | AcpSessionUpdate::Cancelled
            | AcpSessionUpdate::Finished { .. } => DelegateChildActivity::Unchanged,
        };
        self.apply_inactive_delegate_action(session_id, activity)
    }

    fn apply_acp_session_update(
        &mut self,
        session_id: &str,
        update: AcpSessionUpdate,
        is_replay: bool,
    ) -> Vec<Effect> {
        if self.sessions.session_id.as_deref() != Some(session_id) {
            return self.apply_acp_delegate_child_update(session_id, &update);
        }
        self.sessions.note_session_activity(session_id);

        match update {
            AcpSessionUpdate::TurnStarted => {
                self.apply_chat_action(ChatAction::TurnStarted { is_replay })
            }
            AcpSessionUpdate::UserMessage {
                content,
                message_id,
            } => self.apply_chat_action(ChatAction::UserMessage {
                text: acp_content_to_string(&content),
                message_id,
                is_replay,
            }),
            AcpSessionUpdate::AssistantContentDelta {
                content,
                message_id,
            } => self.apply_chat_action(ChatAction::AssistantContentDelta {
                content,
                message_id,
            }),
            AcpSessionUpdate::AssistantThinkingDelta {
                content,
                message_id,
            } => self.apply_chat_action(ChatAction::AssistantThinkingDelta {
                content,
                message_id,
                is_replay,
            }),
            AcpSessionUpdate::AssistantMessage {
                content,
                thinking,
                message_id,
            } => self.apply_chat_action(ChatAction::AssistantMessage {
                content,
                thinking,
                message_id,
            }),
            AcpSessionUpdate::ToolCallStart {
                tool_call_id,
                name,
                arguments,
            } => {
                let (transition, mut effects) =
                    self.apply_chat_tool_action(ChatToolAction::PrepareToolStart {
                        name: name.clone(),
                    });
                match transition {
                    ChatToolTransition::StartPrepared(
                        ToolStartPreparationTransition::Suppressed,
                    ) => return effects,
                    ChatToolTransition::StartPrepared(
                        ToolStartPreparationTransition::QuestionOnly { .. },
                    ) => return effects,
                    ChatToolTransition::StartPrepared(
                        ToolStartPreparationTransition::Prepared { .. },
                    ) => {}
                    _ => unreachable!("prepare tool start returned a non-prepare transition"),
                }

                if name == "delegate" {
                    effects.extend(self.apply_delegate_action(
                        DelegateAction::ProvisionalToolDelegate {
                            tool_call_id: tool_call_id.clone(),
                            arguments: arguments.clone(),
                        },
                    ));
                }

                let cwd = self.current_session_cwd();
                let detail =
                    tool_detail::parse_tool_detail(&name, arguments.as_ref(), cwd.as_deref());
                let (_, insertion_effects) =
                    self.apply_chat_tool_action(ChatToolAction::InsertOrReconcileToolStart {
                        tool_call_id,
                        name,
                        detail,
                    });
                effects.extend(insertion_effects);
                effects
            }
            AcpSessionUpdate::ToolCallEnd {
                tool_call_id,
                name,
                is_error,
                result,
            } => {
                let (_, effects) = self.apply_chat_tool_action(ChatToolAction::ToolCallEnd {
                    tool_call_id,
                    name,
                    is_error,
                    result,
                });
                effects
            }
            AcpSessionUpdate::UsageUpdate {
                used,
                size,
                cost_usd,
            } => self.apply_chat_usage_action(ChatUsageAction::UsageUpdated {
                used,
                size,
                cost_usd,
            }),
            AcpSessionUpdate::TimingUpdate { duration_secs } => {
                self.apply_chat_usage_action(ChatUsageAction::TimingUpdated { duration_secs })
            }
            AcpSessionUpdate::ElicitationRequested {
                elicitation_id,
                message,
                requested_schema,
                source,
                allow_custom,
            } => self.apply_chat_elicitation_action(ElicitationAction::Requested {
                elicitation_id,
                message,
                source,
                requested_schema,
                allow_custom,
                is_replay,
            }),
            AcpSessionUpdate::Cancelled => {
                self.apply_chat_action(ChatAction::Cancelled { is_replay })
            }
            AcpSessionUpdate::Finished { finish_reason } => {
                self.apply_chat_action(ChatAction::Finished {
                    finish_reason,
                    is_replay,
                })
            }
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
    use crate::app::App;

    fn command_refs(effects: &[Effect]) -> Vec<&Command> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Command(command) => Some(command),
                _ => None,
            })
            .collect()
    }
    use crate::chat_state::ElicitationUiState;
    use crate::composer_state::{FileIndexEntryLite, MentionState};
    use crate::domain::activity::{
        DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus, DelegationState,
        PendingDelegateToolCall, SessionOp,
    };
    use crate::domain::chat::ChatEntry;
    use crate::domain::elicitation::ElicitationState;
    use crate::domain::mesh::{RemoteNodeInfo, RemoteSessionInfo};
    use crate::domain::model::DelegateModelPreference;
    use crate::domain::session::{
        SessionGroup, SessionListPage, SessionSummary, UndoFrame, UndoFrameStatus, UndoState,
        UndoableTurn,
    };
    use crate::domain::tool::ToolDetail;

    const TEST_SESSION_ID: &str = "session-1";
    const TEST_ASSISTANT_ID: &str = "a1";

    fn delegation_update(state: DelegationState, updated_at: i64) -> DelegationUpdate {
        DelegationUpdate {
            session_id: TEST_SESSION_ID.into(),
            delegation_id: "delegation-1".into(),
            tool_call_id: Some("call-1".into()),
            state,
            target_agent_id: "coder".into(),
            objective: "Implement the feature".into(),
            child_session_id: (state != DelegationState::Requested).then(|| "child-1".into()),
            requested_at: 100,
            forked_at: (state != DelegationState::Requested).then_some(110),
            finished_at: matches!(
                state,
                DelegationState::Completed | DelegationState::Failed | DelegationState::Cancelled
            )
            .then_some(120),
            updated_at,
            result_summary: (state == DelegationState::Completed).then(|| "done".into()),
            error: (state == DelegationState::Failed).then(|| "boom".into()),
        }
    }

    fn app_with_active_session() -> App {
        let mut app = App::new();
        app.sessions.session_id = Some(TEST_SESSION_ID.into());
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

    fn session_list_page(groups: Vec<SessionGroup>, next_cursor: Option<&str>) -> SessionListPage {
        SessionListPage {
            groups,
            next_cursor: next_cursor.map(str::to_string),
            total_count: None,
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
            app.chat.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking, message_id: Some(message_id) }]
                if content == expected_content
                    && thinking.as_deref() == expected_thinking
                    && message_id == expected_message_id
        ));
    }

    fn push_thinking_entry(app: &mut App, content: &str) {
        app.chat.messages.push(ChatEntry::Thinking {
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

    fn profile(id: &str) -> ProfileInfo {
        ProfileInfo {
            id: id.into(),
            name: id.into(),
            ..Default::default()
        }
    }

    #[test]
    fn initialized_placeholder_empty_catalog_preserves_profiles_and_discards_catalog_commands() {
        let mut app = App::new();
        app.profiles.profiles = vec![profile("fast")];
        app.profiles.active_profile_id = Some("fast".into());
        app.models.agents.clear();
        app.models.model_popup_agent_tab = 3;
        app.models.model_filter = "keep".into();
        app.models.delegate_model_preferences.insert(
            "fast".into(),
            [("coder".into(), DelegateModelPreference::default())]
                .into_iter()
                .collect(),
        );

        let commands = app.handle_acp_event(AcpAppEvent::Initialized {
            agent_id: "agent-1".into(),
            agent_name: "Agent".into(),
            profiles: Vec::new(),
            active_profile_id: None,
            agent_mode: None,
            reasoning_effort: None,
        });

        assert!(commands.is_empty());
        assert_eq!(app.profiles.profiles.len(), 1);
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert_eq!(app.models.agents.len(), 1);
        assert_eq!(app.models.agents_profile_id, None);
        assert_eq!(app.models.model_popup_agent_tab, 3);
        assert_eq!(app.models.model_filter, "keep");
        assert!(
            app.models
                .get_delegate_model_preference("fast", "coder")
                .is_some()
        );
    }

    #[test]
    fn initialized_non_empty_catalog_keeps_discarded_catalog_command_behavior() {
        let mut app = App::new();

        let commands = app.handle_acp_event(AcpAppEvent::Initialized {
            agent_id: "agent-1".into(),
            agent_name: "Agent".into(),
            profiles: vec![profile("fast")],
            active_profile_id: Some("fast".into()),
            agent_mode: None,
            reasoning_effort: None,
        });

        assert!(commands.is_empty());
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert_eq!(app.models.agents.len(), 1);
        assert_eq!(app.models.agents_profile_id, None);
    }

    #[test]
    fn profile_catalog_preserves_valid_local_selection() {
        let mut app = App::new();
        app.profiles.active_profile_id = Some("deep".into());

        app.handle_acp_event(AcpAppEvent::Profiles {
            profiles: vec![profile("fast"), profile("deep")],
            active_profile_id: Some("fast".into()),
        });

        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("deep"));
    }

    #[test]
    fn profile_catalog_falls_back_to_backend_default_then_first_profile() {
        let mut app = App::new();
        app.profiles.active_profile_id = Some("removed".into());

        app.handle_acp_event(AcpAppEvent::Profiles {
            profiles: vec![profile("fast"), profile("deep")],
            active_profile_id: Some("deep".into()),
        });
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("deep"));

        app.profiles.active_profile_id = Some("removed-again".into());
        app.handle_acp_event(AcpAppEvent::Profiles {
            profiles: vec![profile("fast"), profile("deep")],
            active_profile_id: Some("missing".into()),
        });
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
    }

    #[test]
    fn empty_profile_catalog_clears_stale_selection() {
        let mut app = App::new();
        app.profiles.profiles = vec![profile("fast")];
        app.profiles.active_profile_id = Some("fast".into());

        app.handle_acp_event(AcpAppEvent::Profiles {
            profiles: Vec::new(),
            active_profile_id: None,
        });

        assert!(app.profiles.profiles.is_empty());
        assert!(app.profiles.active_profile_id.is_none());
    }

    #[test]
    fn profile_catalog_command_order_remains_list_agents_only() {
        let mut app = App::new();

        let commands = app.handle_acp_event(AcpAppEvent::Profiles {
            profiles: vec![profile("fast")],
            active_profile_id: Some("fast".into()),
        });

        assert!(matches!(
            command_refs(&commands).as_slice(),
            [Command::ListProfileAgents { profile_id }] if profile_id == "fast"
        ));
    }

    #[test]
    fn loading_session_profile_does_not_replace_new_session_selection() {
        let mut app = App::new();
        app.profiles.active_profile_id = Some("deep".into());

        app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "session-1".into(),
            profile_id: Some("fast".into()),
        });

        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("deep"));
        assert_eq!(app.current_session_profile_id(), Some("fast"));
    }

    #[test]
    fn loading_missing_local_profile_preserves_binding_and_command_order() {
        let mut app = App::new();
        app.profiles
            .bind_session_profile("session-1".into(), "fast".into());

        let commands = app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "session-1".into(),
            profile_id: None,
        });

        assert_eq!(app.current_session_profile_id(), Some("fast"));
        assert!(matches!(
            command_refs(&commands).as_slice(),
            [
                Command::SetAgentMode { mode },
                Command::ListProfileAgents { profile_id },
            ] if mode == "build" && profile_id == "fast"
        ));
    }

    #[test]
    fn loading_missing_remote_profile_removes_stale_binding() {
        let mut app = App::new();
        app.sessions
            .remember_remote_session_node("remote-1", "node-1");
        app.profiles
            .bind_session_profile("remote-1".into(), "fast".into());

        let commands = app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "remote-1".into(),
            profile_id: None,
        });

        assert!(app.current_session_profile_id().is_none());
        assert!(matches!(
            command_refs(&commands).as_slice(),
            [Command::SetAgentMode { mode }] if mode == "build"
        ));
    }

    #[test]
    fn native_delegate_tool_start_creates_provisional_entry() {
        let mut app = app_with_active_session();

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: TEST_SESSION_ID.into(),
            update: AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("call-1".into()),
                name: "delegate".into(),
                arguments: Some(serde_json::json!({
                    "target_agent_id": "coder",
                    "objective": "Implement the feature"
                })),
            },
            is_replay: false,
        });

        assert_eq!(app.delegates.delegate_entries.len(), 1);
        assert_eq!(
            app.delegates.delegate_entries[0].delegation_id,
            "tool:call-1"
        );
        assert_eq!(
            app.delegates.delegate_entries[0].target_agent_id.as_deref(),
            Some("coder")
        );
        assert_eq!(
            app.delegates.delegate_entries[0].status,
            DelegateStatus::InProgress
        );
    }

    #[test]
    fn delegation_snapshots_reconcile_by_tool_id_and_ignore_stale_updates() {
        let mut app = app_with_active_session();
        let effects = app.apply_delegate_action(DelegateAction::ProvisionalToolDelegate {
            tool_call_id: Some("call-1".into()),
            arguments: Some(serde_json::json!({
                "target_agent_id": "coder",
                "objective": "Implement the feature"
            })),
        });
        assert!(effects.is_empty());

        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Completed,
            120,
        )));
        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Forked,
            110,
        )));

        assert_eq!(app.delegates.delegate_entries.len(), 1);
        let entry = &app.delegates.delegate_entries[0];
        assert_eq!(entry.delegation_id, "delegation-1");
        assert_eq!(entry.child_session_id.as_deref(), Some("child-1"));
        assert_eq!(entry.status, DelegateStatus::Completed);
        assert_eq!(
            app.delegates.delegation_result_summaries["delegation-1"],
            "done"
        );
    }

    #[test]
    fn delegation_equal_timestamp_does_not_regress_lifecycle_rank() {
        let mut app = app_with_active_session();

        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Completed,
            120,
        )));
        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Forked,
            120,
        )));

        assert_eq!(
            app.delegates.delegate_entries[0].status,
            DelegateStatus::Completed
        );
        assert_eq!(
            app.delegates.delegation_result_summaries["delegation-1"],
            "done"
        );
    }

    #[test]
    fn delegation_updates_insert_and_remove_summary_and_error() {
        let mut app = app_with_active_session();

        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Completed,
            120,
        )));
        assert_eq!(
            app.delegates.delegation_result_summaries["delegation-1"],
            "done"
        );
        assert!(!app.delegates.delegation_errors.contains_key("delegation-1"));

        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Failed,
            130,
        )));
        assert_eq!(
            app.delegates.delegate_entries[0].status,
            DelegateStatus::Failed
        );
        assert!(
            !app.delegates
                .delegation_result_summaries
                .contains_key("delegation-1")
        );
        assert_eq!(app.delegates.delegation_errors["delegation-1"], "boom");

        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Cancelled,
            140,
        )));
        assert_eq!(
            app.delegates.delegate_entries[0].status,
            DelegateStatus::Cancelled
        );
        assert!(!app.delegates.delegation_errors.contains_key("delegation-1"));
    }

    #[test]
    fn delegation_live_and_replay_updates_filter_inactive_parent_sessions() {
        let mut app = app_with_active_session();
        let valid_update = delegation_update(DelegationState::Requested, 100);

        app.handle_acp_event(AcpAppEvent::DelegationReplay {
            session_id: "session-2".into(),
            updates: vec![valid_update.clone()],
        });
        assert!(app.delegates.delegate_entries.is_empty());

        let mut wrong_parent_update = valid_update.clone();
        wrong_parent_update.session_id = "session-2".into();
        app.handle_acp_event(AcpAppEvent::DelegationReplay {
            session_id: TEST_SESSION_ID.into(),
            updates: vec![wrong_parent_update.clone()],
        });
        app.handle_acp_event(AcpAppEvent::DelegationUpdate(wrong_parent_update));
        assert!(app.delegates.delegate_entries.is_empty());

        app.handle_acp_event(AcpAppEvent::DelegationReplay {
            session_id: TEST_SESSION_ID.into(),
            updates: vec![valid_update],
        });
        assert_eq!(app.delegates.delegate_entries.len(), 1);
        assert_eq!(
            app.delegates.delegate_entries[0].status,
            DelegateStatus::InProgress
        );
    }

    #[test]
    fn child_updates_before_fork_are_applied_when_lifecycle_links_child() {
        let mut app = app_with_active_session();
        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "child-1".into(),
            update: AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("child-tool".into()),
                name: "read_tool".into(),
                arguments: None,
            },
            is_replay: false,
        });
        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "child-1".into(),
            update: AcpSessionUpdate::AssistantContentDelta {
                content: "working".into(),
                message_id: Some("child-message".into()),
            },
            is_replay: false,
        });

        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Forked,
            110,
        )));

        let entry = &app.delegates.delegate_entries[0];
        assert_eq!(entry.stats.tool_calls, 1);
        assert_eq!(entry.stats.messages, 1);
        assert_eq!(entry.child_state, DelegateChildState::AssistantMessage);
    }

    #[test]
    fn delegation_completion_does_not_interrupt_parent_streaming() {
        let mut app = app_with_active_session();
        app.chat.streaming_content = "partial".into();
        app.chat.activity = ActivityState::Streaming;

        app.handle_acp_event(AcpAppEvent::DelegationUpdate(delegation_update(
            DelegationState::Completed,
            120,
        )));

        assert_eq!(app.chat.streaming_content, "partial");
        assert_eq!(app.chat.activity, ActivityState::Streaming);
    }

    #[test]
    fn profile_agents_populate_delegate_tabs_and_ignore_stale_responses() {
        let mut app = App::new();
        app.profiles.active_profile_id = Some("quorum".into());

        app.handle_acp_event(AcpAppEvent::ProfileAgents {
            profile_id: "old".into(),
            agents: vec![AgentInfo {
                id: "primary".into(),
                name: "Session".into(),
                description: None,
                capabilities: Vec::new(),
            }],
        });
        assert!(app.models.agents.is_empty());

        app.handle_acp_event(AcpAppEvent::ProfileAgents {
            profile_id: "quorum".into(),
            agents: vec![
                AgentInfo {
                    id: "primary".into(),
                    name: "Session".into(),
                    description: None,
                    capabilities: Vec::new(),
                },
                AgentInfo {
                    id: "coder".into(),
                    name: "Coder".into(),
                    description: Some("Writes code".into()),
                    capabilities: vec!["coding".into()],
                },
            ],
        });
        assert_eq!(app.models.agents_profile_id.as_deref(), Some("quorum"));
        assert_eq!(app.models.model_popup_tab_count(), 2);
        assert_eq!(app.models.model_popup_tab_agent_id(1), Some("coder"));
    }

    #[test]
    fn profile_agents_reapply_profile_scoped_preferences_to_parent_session() {
        let mut app = App::new();
        app.sessions.session_id = Some("parent".into());
        app.profiles
            .bind_session_profile("parent".into(), "quorum".into());
        app.models.delegate_model_preferences.insert(
            "quorum".into(),
            [(
                "coder".into(),
                DelegateModelPreference {
                    model_id: "openai/gpt-5".into(),
                    provider: "openai".into(),
                    model: "gpt-5".into(),
                    node_id: Some("node-1".into()),
                },
            )]
            .into_iter()
            .collect(),
        );

        let commands = app.handle_acp_event(AcpAppEvent::ProfileAgents {
            profile_id: "quorum".into(),
            agents: vec![
                AgentInfo {
                    id: "primary".into(),
                    name: "Session".into(),
                    description: None,
                    capabilities: Vec::new(),
                },
                AgentInfo {
                    id: "coder".into(),
                    name: "Coder".into(),
                    description: None,
                    capabilities: Vec::new(),
                },
            ],
        });

        assert!(matches!(
            command_refs(&commands).as_slice(),
            [Command::SetDelegateModel {
                session_id,
                agent_id,
                model_id: Some(model_id),
                node_id: Some(node_id),
            }] if session_id == "parent"
                && agent_id == "coder"
                && model_id == "openai/gpt-5"
                && node_id == "node-1"
        ));
    }

    #[test]
    fn native_mesh_nodes_store_state_and_requests_first_node_sessions() {
        let mut app = App::new();
        app.mesh.mesh_nodes = vec![
            RemoteNodeInfo {
                id: "node-0".into(),
                label: "local".into(),
                ..Default::default()
            },
            RemoteNodeInfo {
                id: "node-1".into(),
                label: "framework".into(),
                ..Default::default()
            },
        ];
        app.mesh.mesh_node_cursor = 1;
        app.mesh.remote_session_cursor = 4;
        app.mesh.remote_sessions_by_node.insert(
            "stale-node".into(),
            vec![RemoteSessionInfo {
                id: "stale-session".into(),
                node_id: "stale-node".into(),
                ..Default::default()
            }],
        );

        let replies = app.handle_acp_event(AcpAppEvent::MeshNodes(MeshNodesInfo {
            nodes: vec![
                RemoteNodeInfo {
                    id: "node-0".into(),
                    label: "local".into(),
                    ..Default::default()
                },
                RemoteNodeInfo {
                    id: "node-1".into(),
                    label: "framework".into(),
                    active_sessions: 2,
                    ..Default::default()
                },
            ],
        }));

        assert_eq!(app.mesh.mesh_node_count, Some(2));
        assert_eq!(app.mesh.selected_mesh_node_id(), Some("node-1"));
        assert_eq!(app.mesh.mesh_node_cursor, 1);
        assert_eq!(app.mesh.remote_session_cursor, 4);
        assert_eq!(
            app.mesh.remote_sessions_by_node["stale-node"][0].id,
            "stale-session"
        );
        assert!(matches!(
            command_refs(&replies).as_slice(),
            [Command::ListRemoteSessions { node_id, offset: 0, limit: 50 }]
                if node_id == "node-1"
        ));
    }

    #[test]
    fn mesh_status_stores_before_appending_diagnostic_log() {
        let mut app = App::new();
        let before = app.diagnostics.logs.len();

        let replies = app.handle_acp_event(AcpAppEvent::MeshStatus(MeshStatusInfo {
            enabled: true,
            known_peer_count: 2,
            ..Default::default()
        }));

        assert!(replies.is_empty());
        assert!(
            app.mesh
                .mesh_status
                .as_ref()
                .is_some_and(|status| { status.enabled && status.known_peer_count == 2 })
        );
        let logs = &app.diagnostics.logs[before..];
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].level, LogLevel::Debug);
        assert_eq!(logs[0].target, "mesh");
        assert_eq!(logs[0].message, "mesh status: enabled=true, peers=2");
    }

    #[test]
    fn native_remote_sessions_store_values_and_clamp_cursor() {
        let mut app = App::new();
        app.mesh.mesh_nodes = vec![RemoteNodeInfo {
            id: "node-1".into(),
            label: "Framework".into(),
            ..Default::default()
        }];
        app.mesh.remote_session_cursor = 4;
        let before = app.diagnostics.logs.len();

        let replies = app.handle_acp_event(AcpAppEvent::RemoteSessions(RemoteSessionListInfo {
            node_id: "node-1".into(),
            sessions: vec![RemoteSessionInfo {
                id: "session-1".into(),
                node_id: "node-1".into(),
                title: Some("Boundary".into()),
                cwd: Some("/remote/repo".into()),
                ..Default::default()
            }],
            next_offset: Some(50),
            total_count: 51,
        }));

        assert!(replies.is_empty());
        assert_eq!(app.mesh.remote_session_cursor, 0);
        assert_eq!(app.mesh.selected_remote_sessions()[0].id, "session-1");
        assert_eq!(
            app.mesh.selected_remote_sessions()[0].title.as_deref(),
            Some("Boundary")
        );
        assert_eq!(
            app.sessions.session_remote_node_id("session-1"),
            Some("node-1")
        );
        assert_eq!(
            app.sessions.session_remote_cwd("session-1"),
            Some("/remote/repo".into())
        );
        let location = &app.sessions.remote_session_locations["session-1"];
        assert_eq!(location.node_id, "node-1");
        assert_eq!(location.cwd.as_deref(), Some("/remote/repo"));
        let logs = &app.diagnostics.logs[before..];
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].level, LogLevel::Info);
        assert_eq!(logs[0].target, "mesh");
        assert_eq!(logs[0].message, "remote sessions: 1 for node-1");
    }

    #[test]
    fn native_mesh_invite_created_stores_invite_and_opens_invite_view() {
        let mut app = App::new();
        let before = app.diagnostics.logs.len();

        let replies = app.handle_acp_event(AcpAppEvent::MeshInviteCreated(MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: Some("QR".into()),
            expires_at: 1,
            max_uses: 1,
            mesh_name: Some("Team Mesh".into()),
        }));

        assert!(replies.is_empty());
        assert!(matches!(app.navigation.popup, Popup::MeshInviteQr));
        assert_eq!(app.mesh.invite_url(), Some("qmt://mesh/join/token"));
        assert_eq!(app.diagnostics.status, "mesh invite created");
        let logs = &app.diagnostics.logs[before..];
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].level, LogLevel::Info);
        assert_eq!(logs[0].target, "mesh");
        assert_eq!(logs[0].message, "mesh invite created");
        assert_eq!(logs[1].level, LogLevel::Info);
        assert_eq!(logs[1].target, "mesh");
        assert_eq!(logs[1].message, "mesh invite: qmt://mesh/join/token");
    }

    #[test]
    fn native_remote_attach_loads_and_subscribes_once() {
        let mut app = App::new();
        app.sessions.agent_id = Some("agent-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::RemoteSessionAttached(
            RemoteSessionAttachInfo {
                session_id: "remote-1".into(),
                node_id: "node-1".into(),
                attached: true,
                config_options: Vec::new(),
                snapshot: Some(serde_json::json!({
                    "audit": [{ "kind": "message" }],
                    "cursor": { "position": 1 },
                    "delegationUpdates": []
                })),
            },
        ));

        assert_eq!(
            app.sessions.session_remote_node_id("remote-1"),
            Some("node-1")
        );
        assert!(matches!(
            command_refs(&replies).as_slice(),
            [
                Command::LoadSession { session_id: load_id, cwd: None },
                Command::SubscribeSession { session_id: subscribe_id, agent_id },
            ] if load_id == "remote-1"
                && subscribe_id == "remote-1"
                && agent_id.as_deref() == Some("agent-1")
        ));
    }

    #[test]
    fn native_remote_attach_uses_remote_cwd_and_detached_refreshes_sessions() {
        let mut app = App::new();
        app.sessions.agent_id = Some("agent-1".into());
        app.connection.launch_cwd = Some("/launch".into());
        app.sessions.remember_remote_session_location(
            "remote-1",
            "node-1",
            Some("/remote/repo".into()),
        );

        let attached = app.handle_acp_event(AcpAppEvent::RemoteSessionAttached(
            RemoteSessionAttachInfo {
                session_id: "remote-1".into(),
                node_id: "node-1".into(),
                attached: true,
                config_options: vec![Value::String("retained but unused".into())],
                snapshot: Some(serde_json::json!({ "retained": true })),
            },
        ));
        assert!(matches!(
            command_refs(&attached).as_slice(),
            [
                Command::LoadSession { session_id, cwd: Some(cwd) },
                Command::SubscribeSession { session_id: subscribed_id, agent_id },
            ] if session_id == "remote-1"
                && cwd == "/remote/repo"
                && subscribed_id == "remote-1"
                && agent_id.as_deref() == Some("agent-1")
        ));

        let detached = app.handle_acp_event(AcpAppEvent::RemoteSessionAttached(
            RemoteSessionAttachInfo {
                session_id: "remote-2".into(),
                node_id: "node-2".into(),
                attached: false,
                config_options: Vec::new(),
                snapshot: None,
            },
        ));
        assert_eq!(
            app.sessions.session_remote_node_id("remote-2"),
            Some("node-2")
        );
        assert_eq!(app.sessions.session_remote_cwd("remote-2"), None);
        assert!(matches!(
            detached.as_slice(),
            [Effect::Command(Command::ListRemoteSessions {
                node_id,
                offset: 0,
                limit: 50,
            })] if node_id == "node-2"
        ));
    }

    #[test]
    fn acp_node_less_session_page_does_not_overwrite_remote_location() {
        let mut app = App::new();
        app.sessions.remember_remote_session_location(
            "remote-1",
            "node-1",
            Some("/remote/repo".into()),
        );

        app.handle_acp_event(AcpAppEvent::SessionList {
            request: SessionListRequest::Discovery,
            page: session_list_page(
                vec![SessionGroup {
                    cwd: Some("/local/repo".into()),
                    sessions: vec![SessionSummary {
                        session_id: "remote-1".into(),
                        cwd: Some("/local/repo".into()),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                None,
            ),
        });

        let location = &app.sessions.remote_session_locations["remote-1"];
        assert_eq!(location.node_id, "node-1");
        assert_eq!(location.cwd.as_deref(), Some("/remote/repo"));
    }

    #[test]
    fn acp_discovery_hydrates_workspaces_and_replays_root_cursor() {
        let mut app = App::new();
        app.sessions.session_discovery_in_progress = true;

        let replies = app.handle_acp_event(AcpAppEvent::SessionList {
            request: SessionListRequest::Discovery,
            page: session_list_page(
                vec![session_group("/repo", &["s1", "s2"])],
                Some("opaque-root-2"),
            ),
        });

        assert!(app.sessions.session_discovery_in_progress);
        assert!(
            app.sessions
                .pending_session_group_loads
                .contains(&Some("/repo".into()))
        );
        assert_eq!(app.sessions.session_groups[0].next_cursor, None);
        assert_eq!(
            replies,
            vec![
                Effect::Command(Command::list_sessions_workspace("/repo".into())),
                Effect::Command(Command::list_sessions_discovery(Some(
                    "opaque-root-2".into()
                ))),
            ]
        );
    }

    #[test]
    fn acp_workspace_first_page_replaces_discovery_preview() {
        let mut app = App::new();
        app.sessions.session_groups = vec![session_group("/repo", &["preview"])];
        app.sessions
            .pending_session_group_loads
            .insert(Some("/repo".into()));
        let mut page = session_group("/repo", &["new-1", "new-2"]);
        page.next_cursor = Some("opaque-workspace-2".into());

        app.handle_acp_event(AcpAppEvent::SessionList {
            request: SessionListRequest::WorkspaceFirstPage {
                cwd: "/repo".into(),
            },
            page: session_list_page(vec![page], Some("opaque-workspace-2")),
        });

        assert!(app.sessions.hydrated_session_groups.contains("/repo"));
        assert!(app.sessions.pending_session_group_loads.is_empty());
        assert_eq!(app.sessions.session_groups[0].total_count, None);
        assert_eq!(
            app.sessions.session_groups[0]
                .sessions
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["new-2", "new-1"]
        );
        assert_eq!(
            app.sessions.session_groups[0].next_cursor.as_deref(),
            Some("opaque-workspace-2")
        );
    }

    #[test]
    fn acp_discovery_merges_later_workspaces_without_overwriting_hydrated_groups() {
        let mut app = App::new();
        app.sessions.session_discovery_in_progress = true;
        app.sessions.hydrated_session_groups.insert("/repo".into());
        app.sessions.session_groups = vec![session_group("/repo", &["authoritative"])];

        let replies = app.handle_acp_event(AcpAppEvent::SessionList {
            request: SessionListRequest::Discovery,
            page: session_list_page(
                vec![
                    session_group("/repo", &["stale-discovery"]),
                    session_group("/later", &["later-1"]),
                ],
                None,
            ),
        });

        let repo = app
            .sessions
            .session_groups
            .iter()
            .find(|group| group.cwd.as_deref() == Some("/repo"))
            .unwrap();
        assert_eq!(repo.sessions[0].session_id, "authoritative");
        assert!(
            app.sessions
                .session_groups
                .iter()
                .any(|group| group.cwd.as_deref() == Some("/later"))
        );
        assert!(replies.iter().any(|reply| matches!(
            reply,
            Effect::Command(Command::ListSessions {
                request: SessionListRequest::WorkspaceFirstPage { cwd },
                ..
            }) if cwd == "/later"
        )));
    }

    #[test]
    fn acp_group_session_page_merges_group_and_dedupes() {
        let mut app = App::new();
        app.sessions.session_groups = vec![session_group("/repo", &["s1"])];
        app.sessions
            .pending_session_group_loads
            .insert(Some("/repo".into()));

        let mut group = session_group("/repo", &["s1", "s2"]);
        group.next_cursor = Some("cursor-2".into());
        app.handle_acp_event(AcpAppEvent::SessionList {
            request: SessionListRequest::WorkspaceContinuation {
                cwd: "/repo".into(),
            },
            page: session_list_page(vec![group], Some("cursor-2")),
        });

        assert!(app.sessions.pending_session_group_loads.is_empty());
        assert_eq!(app.sessions.session_groups.len(), 1);
        assert_eq!(
            app.sessions.session_groups[0].next_cursor.as_deref(),
            Some("cursor-2")
        );
        assert_eq!(
            app.sessions.session_groups[0]
                .sessions
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["s2", "s1"]
        );
    }

    #[test]
    fn acp_session_list_failure_clears_scoped_pending_state() {
        let mut app = App::new();
        app.sessions
            .pending_session_group_loads
            .insert(Some("/repo".into()));

        let replies = app.handle_acp_event(AcpAppEvent::SessionListFailed {
            request: SessionListRequest::WorkspaceContinuation {
                cwd: "/repo".into(),
            },
            message: "failed".into(),
        });

        assert!(app.sessions.pending_session_group_loads.is_empty());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "failed"
        ));
        assert_eq!(app.diagnostics.status, "error: failed");
        let diagnostic = app.diagnostics.logs.last().expect("failure diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Error);
        assert_eq!(diagnostic.target, "acp");
        assert_eq!(diagnostic.message, "error: failed");
        assert!(replies.is_empty());
    }

    fn seed_model_state(app: &mut App) {
        app.models.current_provider = Some("provider".into());
        app.models.current_model = Some("model".into());
        app.models.reasoning_effort = Some("high".into());
        app.models.models = vec![crate::models_state::ModelsState::test_model_entry(
            "provider/model",
            "provider",
            "model",
            None,
            None,
        )];
        app.models.model_filter = "filter".into();
        app.models.model_cursor = 5;
        app.models.model_popup_agent_tab = 2;
    }

    fn assert_seeded_model_state(app: &App) {
        assert_eq!(app.models.current_provider.as_deref(), Some("provider"));
        assert_eq!(app.models.current_model.as_deref(), Some("model"));
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(app.models.models.len(), 1);
        assert_eq!(app.models.model_filter, "filter");
        assert_eq!(app.models.model_cursor, 5);
        assert_eq!(app.models.model_popup_agent_tab, 2);
    }

    #[test]
    fn native_session_created_resets_view_and_subscribes() {
        let mut app = App::new();
        app.chat.messages.push(ChatEntry::Error("stale".into()));
        app.chat.streaming_content = "stale stream".into();
        app.delegates.parent_session_id = Some("old-parent".into());
        app.delegates.pending_parent_session_id = Some("staged-parent".into());
        app.delegates.delegate_entries.push(DelegateEntry {
            delegation_id: "stale-delegate".into(),
            child_session_id: Some("old-child".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "stale".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.chat.scroll_offset = 3;
        app.composer.input = "/mo".into();
        app.composer.input_cursor = 3;
        app.composer.input_scroll = 2;
        app.composer.input_line_width = 40;
        app.composer.input_preferred_col = Some(4);
        app.composer.refresh_slash_state();
        app.composer.file_index = vec![FileIndexEntryLite {
            path: "src/main.rs".into(),
            is_dir: false,
        }];
        app.composer.file_index_generated_at = Some(7);
        app.composer.file_index_loading = true;
        app.composer.file_index_error = Some("stale".into());
        app.composer.mention_state = Some(MentionState {
            trigger_start: 0,
            query: "src".into(),
            selected_index: 0,
            results: Vec::new(),
        });
        app.chat.undo_state = Some(UndoState {
            stack: Vec::new(),
            frontier_message_id: Some("undo-1".into()),
        });
        app.chat.elicitation = Some(ElicitationState::new_for_test(Vec::new()));
        app.chat.elicitation_ui = Some(ElicitationUiState::default());
        app.chat.session_stats.total_tool_calls = 1;
        app.mesh.mesh_nodes = vec![RemoteNodeInfo {
            id: "node-1".into(),
            label: "framework".into(),
            ..Default::default()
        }];
        app.mesh.mesh_node_count = Some(1);
        app.mesh.mesh_invite_name = "preserved".into();
        seed_model_state(&mut app);

        let replies = app.handle_acp_event(AcpAppEvent::SessionCreated {
            agent_id: "agent-1".into(),
            session_id: "session-1".into(),
            profile_id: Some("code".into()),
        });

        assert_eq!(app.sessions.session_id.as_deref(), Some("session-1"));
        assert_eq!(app.sessions.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(app.profiles.session_profile_id("session-1"), Some("code"));
        assert_eq!(app.navigation.screen, Screen::Chat);
        assert_eq!(app.delegates.parent_session_id, None);
        assert_eq!(app.delegates.pending_parent_session_id, None);
        assert!(app.delegates.delegate_entries.is_empty());
        assert!(app.chat.messages.is_empty());
        assert!(app.chat.streaming_content.is_empty());
        assert_eq!(app.chat.scroll_offset, 0);
        assert_eq!(app.composer.input, "/mo");
        assert_eq!(app.composer.input_cursor, 3);
        assert_eq!(app.composer.input_scroll, 2);
        assert_eq!(app.composer.input_line_width, 40);
        assert_eq!(app.composer.input_preferred_col, Some(4));
        assert!(app.composer.file_index.is_empty());
        assert_eq!(app.composer.file_index_generated_at, None);
        assert!(!app.composer.file_index_loading);
        assert_eq!(app.composer.file_index_error, None);
        assert!(app.composer.mention_state.is_none());
        assert!(app.composer.slash_state.is_some());
        assert!(app.chat.undo_state.is_none());
        assert!(app.chat.elicitation.is_none());
        assert_eq!(app.chat.session_stats.total_tool_calls, 0);
        assert_eq!(app.mesh.mesh_node_count, Some(1));
        assert_eq!(app.mesh.selected_mesh_node_id(), Some("node-1"));
        assert_eq!(app.mesh.mesh_invite_name, "preserved");
        assert_seeded_model_state(&app);
        assert_eq!(app.diagnostics.status, "session created");
        let diagnostic = app.diagnostics.logs.last().expect("creation diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Info);
        assert_eq!(diagnostic.target, "session");
        assert_eq!(diagnostic.message, "session created");
        assert!(matches!(
            command_refs(&replies).as_slice(),
            [
                Command::SubscribeSession { session_id, agent_id },
                Command::ListProfileAgents { profile_id },
            ] if session_id == "session-1"
                && agent_id.as_deref() == Some("agent-1")
                && profile_id == "code"
        ));
    }

    #[test]
    fn native_session_loaded_discovers_parent_preserves_delegate_state_and_resets_view() {
        let mut app = App::new();
        app.sessions.agent_mode = "plan".into();
        let mut child = SessionSummary {
            session_id: "child".into(),
            parent_session_id: Some("parent".into()),
            ..Default::default()
        };
        child.children = vec![SessionSummary {
            session_id: "grandchild".into(),
            parent_session_id: Some("child".into()),
            ..Default::default()
        }];
        let mut parent = SessionSummary {
            session_id: "parent".into(),
            ..Default::default()
        };
        parent.children = vec![child];
        app.sessions.session_groups = vec![SessionGroup {
            cwd: Some("/repo".into()),
            sessions: vec![parent],
            ..Default::default()
        }];
        app.chat.messages.push(ChatEntry::Error("stale".into()));
        app.chat.streaming_content = "stale stream".into();
        app.chat.streaming_content_message_id = Some("stream-1".into());
        app.chat.streaming_thinking = "stale thinking".into();
        app.chat.streaming_thinking_message_id = Some("thinking-1".into());
        app.chat.scroll_offset = 4;
        app.chat.undo_state = Some(UndoState {
            stack: Vec::new(),
            frontier_message_id: Some("undo-1".into()),
        });
        app.chat.recent_prompt_text = Some("stale prompt".into());
        app.chat.elicitation = Some(ElicitationState::new_for_test(Vec::new()));
        app.chat.elicitation_ui = Some(ElicitationUiState::default());
        app.chat.session_stats.total_tool_calls = 2;
        seed_model_state(&mut app);
        app.delegates.delegate_entries.push(DelegateEntry {
            delegation_id: "delegate-1".into(),
            child_session_id: Some("child".into()),
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "keep".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });

        let replies = app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "child".into(),
            profile_id: None,
        });

        assert_eq!(app.sessions.session_id.as_deref(), Some("child"));
        assert_eq!(app.sessions.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert_eq!(app.delegates.parent_session_id.as_deref(), Some("parent"));
        assert_eq!(app.navigation.screen, Screen::Delegate);
        assert!(app.chat.messages.is_empty());
        assert!(app.chat.streaming_content.is_empty());
        assert_eq!(app.chat.streaming_content_message_id, None);
        assert!(app.chat.streaming_thinking.is_empty());
        assert_eq!(app.chat.streaming_thinking_message_id, None);
        assert_eq!(app.chat.scroll_offset, 0);
        assert!(app.chat.undo_state.is_none());
        assert!(app.chat.undoable_turns.is_empty());
        assert!(app.chat.recent_prompt_text.is_none());
        assert!(app.chat.elicitation.is_none());
        assert_eq!(app.chat.session_stats.total_tool_calls, 0);
        assert_eq!(app.delegates.delegate_entries.len(), 1);
        assert_seeded_model_state(&app);
        assert_eq!(app.diagnostics.status, "ready");
        let diagnostic = app.diagnostics.logs.last().expect("load diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "activity");
        assert_eq!(diagnostic.message, "ready");
        assert!(matches!(
            command_refs(&replies).as_slice(),
            [Command::SetAgentMode { mode }] if mode == "plan"
        ));
    }

    #[test]
    fn native_session_loaded_root_clears_delegate_state() {
        let mut app = App::new();
        app.delegates.parent_session_id = Some("old-parent".into());
        app.delegates.delegate_entries.push(DelegateEntry {
            delegation_id: "delegate-1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "clear".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        app.delegates
            .pending_delegate_child_states
            .insert("child".into(), DelegateChildState::OtherProgress);
        app.delegates
            .pending_delegate_tool_calls
            .push(PendingDelegateToolCall {
                tool_call_id: "tool-1".into(),
                target_agent_id: None,
                objective: "clear".into(),
            });
        app.mesh.mesh_node_count = Some(2);
        app.mesh.mesh_invite_ttl = "48h".into();

        app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "root".into(),
            profile_id: None,
        });

        assert_eq!(app.delegates.parent_session_id, None);
        assert_eq!(app.navigation.screen, Screen::Chat);
        assert!(app.delegates.delegate_entries.is_empty());
        assert!(app.delegates.pending_delegate_child_states.is_empty());
        assert!(app.delegates.pending_delegate_tool_calls.is_empty());
        assert_eq!(app.mesh.mesh_node_count, Some(2));
        assert_eq!(app.mesh.mesh_invite_ttl, "48h");
    }

    #[test]
    fn model_catalog_events_preserve_state_and_log_exact_inventory_summary() {
        let mut app = App::new();
        app.models.current_provider = Some("old-provider".into());
        app.models.current_model = Some("old-model".into());
        app.models.current_model_node_id = Some("old-node".into());
        app.models.model_cursor = 9;
        app.models.model_filter = "stale".into();
        app.models.model_popup_agent_tab = 3;
        app.models.reasoning_effort = Some("high".into());

        app.handle_acp_event(AcpAppEvent::Models {
            models: vec![
                crate::models_state::ModelsState::test_model_entry(
                    "local", "provider", "local", None, None,
                ),
                crate::models_state::ModelsState::test_model_entry(
                    "remote",
                    "provider",
                    "remote",
                    Some("node-1"),
                    Some("peer"),
                ),
            ],
            meta: Some(AcpModelsMetaInfo {
                remote_node_count: 2,
                remote_timeout_count: 1,
            }),
        });

        assert_eq!(app.models.models.len(), 2);
        assert_eq!(app.models.current_provider.as_deref(), Some("old-provider"));
        assert_eq!(app.models.current_model.as_deref(), Some("old-model"));
        assert_eq!(
            app.models.current_model_node_id.as_deref(),
            Some("old-node")
        );
        assert_eq!(app.models.model_cursor, 9);
        assert_eq!(app.models.model_filter, "stale");
        assert_eq!(app.models.model_popup_agent_tab, 3);
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(
            app.diagnostics
                .logs
                .last()
                .map(|entry| entry.message.as_str()),
            Some("models: 2 total, 1 remote (inventory nodes=2, timeouts=1)")
        );

        app.handle_acp_event(AcpAppEvent::Models {
            models: Vec::new(),
            meta: Some(AcpModelsMetaInfo::default()),
        });
        assert!(app.models.models.is_empty());
        assert_eq!(
            app.diagnostics
                .logs
                .last()
                .map(|entry| entry.message.as_str()),
            Some("models: 0 total, 0 remote")
        );
        assert_eq!(app.models.model_cursor, 9);
        assert_eq!(app.models.model_filter, "stale");
    }

    #[test]
    fn native_provider_changed_updates_selection_and_preserves_limit_when_absent() {
        let mut app = App::new();
        app.chat.context_limit = 4_096;

        app.handle_acp_event(AcpAppEvent::ProviderChanged {
            provider: "remote-provider".into(),
            model: "model-1".into(),
            context_limit: Some(128_000),
            provider_node_id: Some("node-1".into()),
        });
        assert_eq!(
            app.models.current_provider.as_deref(),
            Some("remote-provider")
        );
        assert_eq!(app.models.current_model.as_deref(), Some("model-1"));
        assert_eq!(app.chat.context_limit, 128_000);
        assert_eq!(app.models.current_model_node_id.as_deref(), Some("node-1"));

        app.handle_acp_event(AcpAppEvent::ProviderChanged {
            provider: "local-provider".into(),
            model: "model-2".into(),
            context_limit: None,
            provider_node_id: None,
        });
        assert_eq!(
            app.models.current_provider.as_deref(),
            Some("local-provider")
        );
        assert_eq!(app.models.current_model.as_deref(), Some("model-2"));
        assert_eq!(app.chat.context_limit, 128_000);
        assert_eq!(app.models.current_model_node_id, None);
    }

    #[test]
    fn native_agent_mode_updates_mode_and_clears_review_return_state() {
        let mut app = App::new();
        app.sessions.agent_mode = "review".into();
        app.sessions.mode_before_review = Some("plan".into());

        let replies = app.handle_acp_event(AcpAppEvent::AgentMode {
            mode: "build".into(),
        });
        assert!(replies.is_empty());
        assert_eq!(app.sessions.agent_mode, "build");
        assert_eq!(app.sessions.mode_before_review, None);

        app.sessions.mode_before_review = Some("plan".into());
        app.handle_acp_event(AcpAppEvent::AgentMode {
            mode: "review".into(),
        });
        assert_eq!(app.sessions.agent_mode, "review");
        assert_eq!(app.sessions.mode_before_review.as_deref(), Some("plan"));
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
            [ChatEntry::Assistant { content, message_id: Some(message_id), .. }]
                if content == "first" && message_id == "assistant-1"
        ));
        assert_eq!(app.chat.streaming_content, "second");
        assert_eq!(
            app.chat.streaming_content_message_id.as_deref(),
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
            app.chat.messages.as_slice(),
            [
                ChatEntry::Assistant { content, message_id: Some(message_id), .. },
                ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), .. }
            ] if content == "Let me check." && message_id == "assistant-1" && tool_call_id == "tool-1"
        ));
        assert!(app.chat.streaming_content.is_empty());
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

        assert_eq!(app.chat.streaming_thinking, "inspect files");
        assert_eq!(
            app.chat.streaming_thinking_message_id.as_deref(),
            Some("assistant-1")
        );
        assert!(app.chat.messages.is_empty());
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

        assert_eq!(app.chat.streaming_thinking, "hello");
        assert_eq!(
            app.chat.streaming_thinking_message_id.as_deref(),
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
            app.chat.messages.as_slice(),
            [
                ChatEntry::Thinking { content, message_id: Some(message_id) },
                ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), .. },
            ] if content == "inspect files" && message_id == "assistant-1" && tool_call_id == "tool-1"
        ));
    }

    #[test]
    fn final_assistant_replaces_matching_thinking_without_crossing_tool() {
        let mut app = app_with_active_session();
        app.chat.messages.push(ChatEntry::Thinking {
            content: "Need to inspect".into(),
            message_id: Some("assistant-1".into()),
        });
        app.chat.messages.push(ChatEntry::ToolCall {
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
            app.chat.messages.as_slice(),
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

        assert_eq!(app.chat.streaming_thinking, "thinking");
        assert_eq!(
            app.chat.streaming_thinking_message_id.as_deref(),
            Some("a1")
        );
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

        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_thinking.is_empty());
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
        app.chat.messages.push(ChatEntry::Assistant {
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
        app.sessions.session_id = Some("session-1".into());

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
            app.chat.messages.as_slice(),
            [ChatEntry::ToolCall {
                detail: ToolDetail::Shell { command, workdir, .. },
                ..
            }] if command == "cargo check --examples" && workdir.as_deref() == Some("/repo")
        ));
        assert_eq!(app.chat.session_stats.total_tool_calls, 1);
    }

    #[test]
    fn native_usage_update_updates_status_metrics() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::UsageUpdate {
                used: 2048,
                size: 8192,
                cost_usd: Some(0.0123),
            },
        });

        assert_eq!(app.chat.session_stats.latest_context_tokens, Some(2048));
        assert_eq!(app.chat.context_limit, 8192);
        assert_eq!(app.chat.cumulative_cost, Some(0.0123));
        assert!(matches!(
            app.diagnostics.logs.last(),
            Some(entry)
                if entry.level == LogLevel::Info
                    && entry.target == "usage"
                    && entry.message == "usage: context 2048/8192 tokens (25%), cost $0.0123"
        ));
    }

    #[test]
    fn native_timing_update_adds_active_time() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: true,
            update: AcpSessionUpdate::TimingUpdate { duration_secs: 42 },
        });

        assert_eq!(
            app.chat.llm_request_elapsed(),
            Some(std::time::Duration::from_secs(42))
        );
        assert!(matches!(
            app.diagnostics.logs.last(),
            Some(entry)
                if entry.level == LogLevel::Info
                    && entry.target == "usage"
                    && entry.message == "usage: active time 42s"
        ));
    }

    #[test]
    fn native_live_turn_updates_realtime_elapsed() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::TurnStarted,
        });

        assert!(app.chat.session_stats.open_llm_request_instant.is_some());
        app.chat.session_stats.open_llm_request_instant = app
            .chat
            .session_stats
            .open_llm_request_instant
            .map(|started| started - std::time::Duration::from_secs(2));
        assert!(app.chat.llm_request_elapsed().is_some_and(|elapsed| {
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

        assert!(app.chat.session_stats.open_llm_request_instant.is_none());
        assert!(app.chat.llm_request_elapsed().is_some_and(|elapsed| {
            elapsed >= std::time::Duration::from_secs(2)
                && elapsed < std::time::Duration::from_secs(3)
        }));
    }

    #[test]
    fn native_replay_turn_does_not_start_realtime_elapsed() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: true,
            update: AcpSessionUpdate::TurnStarted,
        });

        assert!(app.chat.session_stats.open_llm_request_instant.is_none());
        assert_eq!(app.chat.llm_request_elapsed(), None);
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
            app.chat.messages.first(),
            Some(ChatEntry::User { text, message_id: Some(message_id) })
                if text == "second" && message_id == &second
        ));
        assert_eq!(
            app.chat
                .messages
                .iter()
                .filter(|entry| matches!(entry, ChatEntry::User { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn native_acp_error_closes_realtime_elapsed() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "session-1".into(),
            is_replay: false,
            update: AcpSessionUpdate::TurnStarted,
        });
        app.chat.session_stats.open_llm_request_instant = app
            .chat
            .session_stats
            .open_llm_request_instant
            .map(|started| started - std::time::Duration::from_secs(3));
        app.handle_acp_event(AcpAppEvent::Error {
            message: "prompt failed".into(),
        });

        assert!(app.chat.session_stats.open_llm_request_instant.is_none());
        assert!(app.chat.llm_request_elapsed().is_some_and(|elapsed| {
            elapsed >= std::time::Duration::from_secs(3)
                && elapsed < std::time::Duration::from_secs(4)
        }));
    }

    #[test]
    fn native_tool_call_end_updates_shell_output_tail() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());
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
            app.chat.messages.as_slice(),
            [ChatEntry::ToolCall {
                detail: ToolDetail::Shell { output_tail: Some(tail), .. },
                ..
            }] if tail.lines.iter().any(|line| line.contains("Finished dev profile"))
        ));
    }

    #[test]
    fn native_tool_call_start_keeps_read_tool_range() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

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
            app.chat.messages.as_slice(),
            [ChatEntry::ToolCall {
                detail: ToolDetail::ReadTool { path, start_line: Some(10), end_line: Some(14) },
                ..
            }] if path == "src/main.rs"
        ));
    }

    #[test]
    fn native_undo_result_success_updates_state_and_reloads_session() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());
        app.sessions.agent_id = Some("agent-1".into());
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.chat.undoable_turns.push(UndoableTurn {
            turn_id: "u1".into(),
            message_id: "u1".into(),
            text: "change".into(),
        });

        let replies = app.handle_acp_event(AcpAppEvent::UndoResult(UndoResult::Applied {
            target_message_id: Some("u1".into()),
            reverted_files: vec!["src/main.rs".into()],
            message: None,
            stack: UndoStackSnapshot {
                message_ids: vec!["u1".into()],
            },
        }));

        assert!(matches!(app.chat.activity, ActivityState::Idle));
        assert_eq!(app.diagnostics.status, "undone - reloading session");
        assert!(app.chat.can_redo());
        assert_eq!(
            app.chat.undo_state.as_ref().unwrap().stack[0].reverted_files,
            ["src/main.rs"]
        );
        assert!(matches!(
            command_refs(&replies).as_slice(),
            [
                Command::LoadSession { session_id: load_id, cwd },
                Command::SubscribeSession { session_id: subscribe_id, agent_id },
            ] if load_id == "session-1"
                && subscribe_id == "session-1"
                && cwd.is_none()
                && agent_id.as_deref() == Some("agent-1")
        ));
    }

    #[test]
    fn native_undo_rejection_without_target_uses_stack_tail_without_reload() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.chat.streaming_content = "keep streaming content".into();
        app.chat.undoable_turns = vec![
            UndoableTurn {
                turn_id: "turn-1".into(),
                message_id: "msg-1".into(),
                text: "first".into(),
            },
            UndoableTurn {
                turn_id: "turn-2".into(),
                message_id: "msg-2".into(),
                text: "second".into(),
            },
        ];
        app.chat.undo_state = Some(UndoState {
            stack: vec![UndoFrame {
                turn_id: "turn-1".into(),
                message_id: "msg-1".into(),
                status: UndoFrameStatus::Confirmed,
                reverted_files: vec![],
            }],
            frontier_message_id: Some("msg-1".into()),
        });

        let replies = app.handle_acp_event(AcpAppEvent::UndoResult(UndoResult::Rejected {
            target_message_id: None,
            message: Some("Undo rejected".into()),
            stack: UndoStackSnapshot {
                message_ids: vec!["msg-1".into(), "msg-2".into()],
            },
        }));

        assert!(matches!(app.chat.activity, ActivityState::Idle));
        assert_eq!(app.diagnostics.status, "Undo rejected");
        assert_eq!(app.chat.streaming_content, "keep streaming content");
        assert!(replies.is_empty());
        let undo_state = app.chat.undo_state.as_ref().expect("undo state");
        assert_eq!(undo_state.frontier_message_id.as_deref(), Some("msg-2"));
        assert!(
            undo_state
                .stack
                .iter()
                .all(|frame| frame.reverted_files.is_empty())
        );
        assert_eq!(
            app.chat
                .current_undo_target()
                .map(|turn| turn.message_id.as_str()),
            Some("msg-1")
        );
    }

    #[test]
    fn native_undo_rejection_prefers_explicit_target() {
        let mut app = App::new();
        app.chat.undoable_turns = vec![
            UndoableTurn {
                turn_id: "turn-1".into(),
                message_id: "msg-1".into(),
                text: "first".into(),
            },
            UndoableTurn {
                turn_id: "turn-2".into(),
                message_id: "msg-2".into(),
                text: "second".into(),
            },
        ];

        let replies = app.handle_acp_event(AcpAppEvent::UndoResult(UndoResult::Rejected {
            target_message_id: Some("msg-1".into()),
            message: None,
            stack: UndoStackSnapshot {
                message_ids: vec!["msg-1".into(), "msg-2".into()],
            },
        }));

        assert!(replies.is_empty());
        assert_eq!(
            app.chat
                .undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("msg-1")
        );
    }

    #[test]
    fn native_redo_result_success_rebuilds_state_and_reloads_session() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());
        app.sessions.agent_id = Some("agent-1".into());
        app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);

        let replies = app.handle_acp_event(AcpAppEvent::RedoResult(RedoResult::Applied {
            message: Some("redone".into()),
            stack: UndoStackSnapshot {
                message_ids: vec!["u1".into()],
            },
        }));

        assert!(matches!(app.chat.activity, ActivityState::Idle));
        assert_eq!(app.diagnostics.status, "redone - reloading session");
        assert!(app.chat.can_redo());
        assert!(matches!(
            command_refs(&replies).as_slice(),
            [
                Command::LoadSession { session_id: load_id, cwd },
                Command::SubscribeSession { session_id: subscribe_id, agent_id },
            ] if load_id == "session-1"
                && subscribe_id == "session-1"
                && cwd.is_none()
                && agent_id.as_deref() == Some("agent-1")
        ));
    }

    #[test]
    fn native_redo_result_failure_clears_pending_state_and_logs_warning() {
        let mut app = App::new();
        app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);

        let replies = app.handle_acp_event(AcpAppEvent::RedoResult(RedoResult::Rejected {
            message: Some("Nothing to redo".into()),
            stack: UndoStackSnapshot {
                message_ids: vec!["u1".into()],
            },
        }));

        assert!(matches!(app.chat.activity, ActivityState::Idle));
        assert_eq!(app.diagnostics.status, "Nothing to redo");
        assert!(app.chat.can_redo());
        assert!(replies.is_empty());
        assert!(matches!(
            app.diagnostics.logs.last(),
            Some(entry) if entry.level == LogLevel::Warn && entry.target == "session"
        ));
    }

    #[test]
    fn native_fork_result_success_loads_forked_session() {
        let mut app = App::new();
        app.sessions.agent_id = Some("agent-1".into());
        app.navigation.popup = Popup::ForkTurnSelect;
        app.chat.pending_fork_message_id = Some("msg-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::ForkResult(ForkResult::Succeeded {
            source_session_id: Some("source-1".into()),
            forked_session_id: Some("fork-1".into()),
            message: None,
        }));

        assert_eq!(app.chat.pending_fork_message_id, None);
        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(app.diagnostics.status, "forked - loading session");
        assert_eq!(replies.len(), 2);
        assert!(matches!(
            command_refs(&replies).as_slice(),
            [
                Command::LoadSession { session_id: load_id, .. },
                Command::SubscribeSession { session_id: subscribe_id, agent_id }
            ] if load_id == "fork-1" && subscribe_id == "fork-1" && agent_id.as_deref() == Some("agent-1")
        ));
    }

    #[test]
    fn native_fork_result_success_without_id_warns_and_keeps_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::ForkTurnSelect;
        app.chat.pending_fork_message_id = Some("msg-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::ForkResult(ForkResult::Succeeded {
            source_session_id: Some("source-1".into()),
            forked_session_id: None,
            message: None,
        }));

        assert!(replies.is_empty());
        assert_eq!(app.chat.pending_fork_message_id, None);
        assert_eq!(app.navigation.popup, Popup::ForkTurnSelect);
        assert_eq!(app.diagnostics.status, "fork succeeded without session id");
    }

    #[test]
    fn native_fork_result_failure_clears_pending_and_keeps_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::ForkTurnSelect;
        app.chat.pending_fork_message_id = Some("msg-1".into());

        let replies = app.handle_acp_event(AcpAppEvent::ForkResult(ForkResult::Failed {
            source_session_id: Some("source-1".into()),
            message: Some("fork failed".into()),
        }));

        assert!(replies.is_empty());
        assert_eq!(app.chat.pending_fork_message_id, None);
        assert_eq!(app.navigation.popup, Popup::ForkTurnSelect);
        assert_eq!(app.diagnostics.status, "fork failed");
    }

    #[test]
    fn native_undo_stack_hydrates_redo_state() {
        let mut app = App::new();

        app.handle_acp_event(AcpAppEvent::UndoStack(UndoStackSnapshot {
            message_ids: vec!["u1".into()],
        }));

        assert!(app.chat.can_redo());
    }

    #[test]
    fn native_session_update_for_other_session_marks_activity_only() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());

        app.handle_acp_event(AcpAppEvent::SessionUpdate {
            session_id: "other".into(),
            is_replay: false,
            update: AcpSessionUpdate::AssistantMessage {
                content: "hidden".into(),
                thinking: None,
                message_id: None,
            },
        });

        assert!(app.chat.messages.is_empty());
        assert!(app.sessions.session_activity.contains_key("other"));
    }

    #[test]
    fn native_session_replay_applies_history_as_one_event() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

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
            app.chat.messages.as_slice(),
            [
                ChatEntry::User { text, message_id: Some(user_id) },
                ChatEntry::Assistant { content, message_id: Some(assistant_id), .. }
            ] if text == "hello" && user_id == "u1" && content == "world" && assistant_id == "a1"
        ));
    }

    #[test]
    fn native_session_replay_coalesces_deltas_without_crossing_tool_order() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

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
            app.chat.messages.as_slice(),
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
        app.sessions.session_id = Some("session-1".into());

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
            app.chat.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(message_id) }]
                if content == "final" && thinking == "final thinking" && message_id == "a1"
        ));
    }

    #[test]
    fn native_session_replay_merges_adjacent_assistant_chunks_from_loading_notifications() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());

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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            matches!(app.chat.messages.as_slice(), [ChatEntry::Error(message)] if message == "connection lost")
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            app.chat
                .elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-1")
        );
        assert!(matches!(
            app.chat.messages.as_slice(),
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

        assert!(app.chat.elicitation.is_none());
        assert!(matches!(
            app.chat.messages.as_slice(),
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
        app.apply_chat_elicitation_action(ElicitationAction::ResponseAcknowledged {
            elicitation_id: "elic-1".into(),
            outcome: "staging".into(),
        });
        app.handle_acp_event(AcpAppEvent::SessionReplay {
            session_id: TEST_SESSION_ID.into(),
            updates,
        });

        assert!(app.chat.elicitation.is_none());
        assert!(matches!(
            app.chat.messages.as_slice(),
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
            app.chat
                .elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-live")
        );
        assert!(matches!(
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
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
            app.chat.messages.as_slice(),
            [
                ChatEntry::User { message_id: Some(user_id), .. },
                ChatEntry::Assistant { message_id: Some(assistant_id), .. },
            ] if user_id == "u1" && assistant_id == "a1"
        ));
    }
}
