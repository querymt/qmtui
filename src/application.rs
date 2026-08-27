use crossterm::event::{KeyEvent, MouseEvent};

use crate::{
    acp_state::AcpAppEvent,
    app::{App, ConnectionEvent},
    auth_state::AuthAction,
    chat_state::{ChatAction, ElicitationAction},
    command::Command,
    connection_state::ConnState,
    diagnostics::LogLevel,
    handlers,
    server_manager::ServerEvent,
};

pub(crate) enum AppEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Acp(AcpAppEvent),
    Connection(ConnectionEvent),
    Supervisor(ServerEvent),
    Tick,
    Runtime(RuntimeEvent),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RuntimeEvent {
    ClipboardFinished {
        target: ClipboardTarget,
        success: bool,
    },
    ExternalEditorFinished {
        outcome: ExternalEditorOutcome,
    },
    ElicitationResponseSent {
        elicitation_id: String,
        outcome: String,
    },
    CommandFailed {
        command: Command,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ClipboardTarget {
    Auth { provider: String },
    MeshInvite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExternalEditorOutcome {
    Completed(String),
    Cancelled,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Effect {
    Command(Command),
    ElicitationResponse {
        elicitation_id: String,
        action: String,
        content: Option<serde_json::Value>,
        outcome: String,
    },
    PersistConfig,
    CopyToClipboard {
        target: ClipboardTarget,
        text: String,
    },
    OpenExternalEditor {
        initial_text: String,
    },
    Terminal(TerminalAction),
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TerminalAction {
    Redraw,
}

pub(crate) fn update(app: &mut App, event: AppEvent) -> Vec<Effect> {
    match event {
        AppEvent::Key(key) => handlers::handle_key(app, key),
        AppEvent::Mouse(mouse) => handlers::handle_mouse(app, mouse),
        AppEvent::Acp(event) => app.handle_acp_event(event),
        AppEvent::Connection(event) => handle_connection_event(app, event),
        AppEvent::Supervisor(event) => handle_supervisor_event(app, event),
        AppEvent::Tick => {
            app.render
                .replace_tick((app.diagnostics.started_at.elapsed().as_millis() / 80) as u64);
            app.clear_expired_cancel_confirm();
            Vec::new()
        }
        AppEvent::Runtime(event) => handle_runtime_event(app, event),
    }
}
fn handle_connection_event(app: &mut App, event: ConnectionEvent) -> Vec<Effect> {
    let was_connected = app.connection.conn == ConnState::Connected;
    app.handle_connection_event(event);

    if app.connection.conn == ConnState::Connected {
        let mut effects = vec![
            Effect::Command(Command::Init),
            Effect::Command(Command::list_sessions_browse()),
            Effect::Command(Command::ListAllModels { refresh: false }),
        ];
        effects.extend(reconnect_session_effects(app));
        effects
    } else {
        if was_connected && app.connection.conn == ConnState::Disconnected {
            app.set_status(
                LogLevel::Warn,
                "connection",
                "connection lost - reconnecting...",
            );
        }
        Vec::new()
    }
}

fn reconnect_session_effects(app: &mut App) -> Vec<Effect> {
    let Some(session_id) = app.sessions.session_id.clone() else {
        return Vec::new();
    };

    if let Some(node_id) = app.sessions.session_remote_node_id(&session_id) {
        return vec![Effect::Command(Command::AttachRemoteSession {
            node_id: node_id.to_string(),
            session_id,
        })];
    }
    if app.sessions.is_remote_session_id(&session_id) {
        app.set_status(
            LogLevel::Warn,
            "session",
            "remote session is missing node id; reconnect attach skipped",
        );
        return Vec::new();
    }

    Command::load_session_commands(
        session_id,
        app.current_session_cwd(),
        app.sessions.agent_id.clone(),
    )
    .into_iter()
    .map(Effect::Command)
    .collect()
}

fn handle_supervisor_event(app: &mut App, event: ServerEvent) -> Vec<Effect> {
    match event {
        ServerEvent::Starting => {
            app.connection.apply_server_starting();
            if app.connection.conn != ConnState::Connected {
                app.set_status(LogLevel::Info, "acp", "starting qmtcode ACP agent...");
            }
        }
        ServerEvent::Started => {
            app.connection.apply_server_started();
            if app.connection.conn != ConnState::Connected {
                app.set_status(LogLevel::Info, "acp", "qmtcode ACP agent started");
            }
        }
        ServerEvent::BinaryNotFound => {
            app.connection.apply_server_binary_not_found();
            if app.connection.conn != ConnState::Connected {
                app.set_status(
                    LogLevel::Warn,
                    "acp",
                    "qmtcode not found; install it or set acp.binary_path in ~/.qmt/qmtui.toml",
                );
            }
        }
        ServerEvent::StartFailed { error } => {
            app.connection.apply_server_start_failed(error.clone());
            app.set_status(LogLevel::Error, "acp", format!("ACP start failed: {error}"));
        }
        ServerEvent::Stopped { reason } => {
            app.connection.apply_server_stopped(reason.clone());
            app.set_status(
                LogLevel::Warn,
                "acp",
                format!("ACP agent stopped ({reason})"),
            );
        }
    }
    Vec::new()
}

fn handle_runtime_event(app: &mut App, event: RuntimeEvent) -> Vec<Effect> {
    match event {
        RuntimeEvent::ClipboardFinished { target, success } => match target {
            ClipboardTarget::Auth { provider } => {
                app.apply_auth_action(AuthAction::ClipboardFinished { provider, success })
            }
            ClipboardTarget::MeshInvite => app.apply_mesh_clipboard_result(success),
        },
        RuntimeEvent::ExternalEditorFinished { outcome } => {
            app.render.invalidate_card_cache();
            app.render.invalidate_content_cache();
            match outcome {
                ExternalEditorOutcome::Completed(updated_input) => {
                    app.composer.replace_input_from_editor(updated_input);
                    app.set_status(
                        LogLevel::Info,
                        "editor",
                        "loaded prompt from external editor",
                    );
                }
                ExternalEditorOutcome::Cancelled => {
                    app.set_status(LogLevel::Info, "editor", "external editor cancelled");
                }
                ExternalEditorOutcome::Failed(message) => {
                    app.set_status(
                        LogLevel::Error,
                        "editor",
                        format!("external editor failed: {message}"),
                    );
                }
            }
            vec![Effect::Terminal(TerminalAction::Redraw)]
        }
        RuntimeEvent::ElicitationResponseSent {
            elicitation_id,
            outcome,
        } => app.apply_chat_elicitation_action(ElicitationAction::ResponseAcknowledged {
            elicitation_id,
            outcome,
        }),
        RuntimeEvent::CommandFailed { command, message } => {
            let effects = if let Command::Prompt { local_id, .. } = command {
                app.apply_chat_action(ChatAction::RuntimePromptDispatchFailed { local_id })
            } else {
                Vec::new()
            };
            app.set_status(LogLevel::Error, "command", message);
            effects
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEventKind};

    use super::*;
    use crate::auth_state::AuthUiNotice;
    use crate::{
        acp_state::AcpSessionUpdate,
        chat_state::ElicitationUiState,
        command::SessionListRequest,
        composer_state::FileIndexEntryLite,
        connection_state::ServerState,
        domain::tool::ToolDetail,
        domain::{
            activity::{
                ActivityState, DelegateChildState, DelegateStatus, DelegationState,
                DelegationUpdate, SessionOp,
            },
            auth::{OAuthFlow, OAuthFlowKind, OAuthResult, OAuthResultStatus},
            chat::ChatEntry,
            elicitation::ElicitationState,
            mesh::{MeshInviteCreatedInfo, MeshNodesInfo, RemoteNodeInfo},
            model::DelegateModelPreference,
            profile::{AgentInfo, ProfileInfo},
            session::{
                ForkResult, RedoResult, SessionGroup, SessionListPage, SessionSummary, UndoResult,
                UndoStackSnapshot, UndoableTurn,
            },
        },
        navigation_state::{Popup, Screen},
    };

    fn add_elicitation(app: &mut App, elicitation_id: &str, active: bool) {
        app.chat.messages.push(ChatEntry::Elicitation {
            elicitation_id: elicitation_id.into(),
            message: format!("Question {elicitation_id}"),
            source: "builtin:question".into(),
            outcome: None,
        });
        if active {
            let mut state = ElicitationState::new_for_test(Vec::new());
            state.elicitation_id = elicitation_id.into();
            state.message = format!("Question {elicitation_id}");
            app.chat.elicitation = Some(state);
            app.chat.elicitation_ui = Some(ElicitationUiState::default());
        }
    }

    fn profile(id: &str) -> ProfileInfo {
        ProfileInfo {
            id: id.into(),
            name: id.into(),
            ..Default::default()
        }
    }

    fn agent(id: &str) -> AgentInfo {
        AgentInfo {
            id: id.into(),
            name: id.into(),
            description: None,
            capabilities: Vec::new(),
        }
    }

    fn apply_session_update(app: &mut App, update_value: AcpSessionUpdate) -> Vec<Effect> {
        update(
            app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: update_value,
                is_replay: false,
            }),
        )
    }

    fn supported_elicitation(elicitation_id: &str) -> AcpSessionUpdate {
        AcpSessionUpdate::ElicitationRequested {
            elicitation_id: elicitation_id.into(),
            message: format!("Question {elicitation_id}"),
            requested_schema: serde_json::json!({
                "type": "object",
                "properties": { "target": { "type": "string", "enum": ["staging"] } },
                "required": ["target"]
            }),
            source: "builtin:question".into(),
            allow_custom: true,
        }
    }

    fn unsupported_elicitation(elicitation_id: &str) -> AcpSessionUpdate {
        AcpSessionUpdate::ElicitationRequested {
            elicitation_id: elicitation_id.into(),
            message: format!("Question {elicitation_id}"),
            requested_schema: serde_json::json!({ "type": "array" }),
            source: "extension:file-picker".into(),
            allow_custom: false,
        }
    }

    fn delegation_update(
        session_id: &str,
        state: DelegationState,
        updated_at: i64,
    ) -> DelegationUpdate {
        DelegationUpdate {
            session_id: session_id.into(),
            delegation_id: "delegation-1".into(),
            tool_call_id: Some("delegate-call".into()),
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

    #[test]
    fn key_and_mouse_events_dispatch_through_update() {
        let mut app = App::new();
        app.composer.input = "draft".into();

        let effects = update(
            &mut app,
            AppEvent::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        );
        assert!(effects.is_empty());
        assert!(app.composer.input.is_empty());

        app.navigation.screen = Screen::Chat;
        app.chat.scroll_offset = 2;
        let effects = update(
            &mut app,
            AppEvent::Mouse(MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.scroll_offset, 5);

        let effects = update(
            &mut app,
            AppEvent::Mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }),
        );
        assert!(effects.is_empty());
    }

    #[test]
    fn acp_event_dispatches_command_effect_through_update() {
        let mut app = App::new();
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Profiles {
                profiles: vec![ProfileInfo {
                    id: "fast".into(),
                    name: "Fast".into(),
                    ..Default::default()
                }],
                active_profile_id: Some("fast".into()),
            }),
        );

        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert_eq!(
            effects,
            vec![Effect::Command(Command::ListProfileAgents {
                profile_id: "fast".into(),
            })]
        );
    }

    #[test]
    fn delegation_lifecycle_acp_route_is_filtered_ranked_and_release_safe() {
        let mut app = App::new();
        app.sessions.session_id = Some("parent".into());
        app.render.card_cache.processed_messages = 7;

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::DelegationUpdate(delegation_update(
                "parent",
                DelegationState::Completed,
                120,
            ))),
        );
        assert!(effects.is_empty());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.delegates.delegate_entries.len(), 1);
        assert_eq!(
            app.delegates.delegate_entries[0].status,
            DelegateStatus::Completed
        );
        assert_eq!(
            app.delegates.delegation_result_summaries["delegation-1"],
            "done"
        );

        app.render.card_cache.processed_messages = 8;
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::DelegationUpdate(delegation_update(
                "parent",
                DelegationState::Forked,
                120,
            ))),
        );
        assert!(effects.is_empty());
        assert_eq!(app.render.card_cache.processed_messages, 8);
        assert_eq!(
            app.delegates.delegate_entries[0].status,
            DelegateStatus::Completed
        );

        app.render.card_cache.processed_messages = 9;
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::DelegationUpdate(delegation_update(
                "other-parent",
                DelegationState::Failed,
                130,
            ))),
        );
        assert!(effects.is_empty());
        assert_eq!(app.render.card_cache.processed_messages, 9);
        assert_eq!(app.delegates.delegate_entries.len(), 1);
    }

    #[test]
    fn provisional_delegate_acp_route_reconciles_without_reordering_chat_tool_state() {
        let mut app = App::new();
        app.sessions.session_id = Some("parent".into());
        app.chat.streaming_content = "before delegate".into();
        app.chat.streaming_content_message_id = Some("assistant-1".into());
        app.chat.streaming_thinking = "delegate plan".into();
        app.chat.streaming_thinking_message_id = Some("assistant-1".into());
        app.render.streaming_cache.store(15, Vec::new());
        app.render.streaming_thinking_cache.store(13, Vec::new());
        app.render.card_cache.processed_messages = 7;

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "parent".into(),
                update: crate::acp_state::AcpSessionUpdate::ToolCallStart {
                    tool_call_id: Some("delegate-call".into()),
                    name: "delegate".into(),
                    arguments: Some(serde_json::json!({
                        "target_agent_id": "coder",
                        "objective": "Implement the feature"
                    })),
                },
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("parent"));
        assert_eq!(
            app.chat.activity,
            ActivityState::RunningTool {
                name: "delegate".into()
            }
        );
        assert_eq!(app.chat.session_stats.total_tool_calls, 1);
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_thinking.is_empty());
        assert!(app.render.streaming_cache.get(15).is_none());
        assert!(app.render.streaming_thinking_cache.get(13).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "tool: delegate");
        let diagnostic = app.diagnostics.logs.last().expect("delegate tool status");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "tool");
        assert_eq!(diagnostic.message, "tool: delegate");
        assert_eq!(app.delegates.delegate_entries.len(), 1);
        assert_eq!(
            app.delegates.delegate_entries[0].delegation_id,
            "tool:delegate-call"
        );
        assert_eq!(app.delegates.pending_delegate_tool_calls.len(), 1);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [
                ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(message_id) },
                ChatEntry::ToolCall {
                    tool_call_id: Some(tool_call_id),
                    name,
                    is_error: false,
                    detail: ToolDetail::Summary(summary),
                },
            ] if content == "before delegate"
                && thinking == "delegate plan"
                && message_id == "assistant-1"
                && tool_call_id == "delegate-call"
                && name == "delegate"
                && summary == "(coder) Implement the feature"
        ));

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::DelegationUpdate(delegation_update(
                "parent",
                DelegationState::Forked,
                110,
            ))),
        );
        assert!(effects.is_empty());
        assert_eq!(app.delegates.delegate_entries.len(), 1);
        let entry = &app.delegates.delegate_entries[0];
        assert_eq!(entry.delegation_id, "delegation-1");
        assert_eq!(
            entry.delegate_tool_call_id.as_deref(),
            Some("delegate-call")
        );
        assert_eq!(entry.child_session_id.as_deref(), Some("child-1"));
        assert_eq!(entry.target_agent_id.as_deref(), Some("coder"));
        assert_eq!(entry.objective, "Implement the feature");
        assert!(app.delegates.pending_delegate_tool_calls.is_empty());
    }

    #[test]
    fn chat_tool_start_acp_route_is_owner_bounded_and_release_safe() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.chat.suppress_turn_output = true;
        app.chat.activity = ActivityState::Thinking;
        app.chat.streaming_content = "preserved while suppressed".into();
        app.chat.streaming_content_message_id = Some("assistant-suppressed".into());
        app.render.streaming_cache.store(26, Vec::new());
        app.render.streaming_thinking_cache.store(9, Vec::new());
        app.render.card_cache.processed_messages = 8;
        let status_before = app.diagnostics.status.clone();
        let log_count_before = app.diagnostics.logs.len();

        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("shell-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "cargo check" })),
            },
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("active"));
        assert_eq!(app.chat.activity, ActivityState::Thinking);
        assert_eq!(app.chat.streaming_content, "preserved while suppressed");
        assert!(app.chat.messages.is_empty());
        assert_eq!(app.chat.session_stats.total_tool_calls, 0);
        assert_eq!(app.diagnostics.status, status_before);
        assert_eq!(app.diagnostics.logs.len(), log_count_before);
        assert!(app.render.streaming_cache.get(26).is_some());
        assert!(app.render.streaming_thinking_cache.get(9).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 8);

        app.chat.suppress_turn_output = false;
        app.chat.streaming_thinking = "plan".into();
        app.chat.streaming_thinking_message_id = Some("assistant-suppressed".into());
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("shell-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "cargo check" })),
            },
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.chat.activity,
            ActivityState::RunningTool {
                name: "shell".into()
            }
        );
        assert_eq!(app.chat.session_stats.total_tool_calls, 1);
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_thinking.is_empty());
        assert!(app.render.streaming_cache.get(26).is_none());
        assert!(app.render.streaming_thinking_cache.get(9).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "tool: shell");
        let diagnostic = app.diagnostics.logs.last().expect("shell tool status");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "tool");
        assert_eq!(diagnostic.message, "tool: shell");
        assert!(matches!(
            app.chat.messages.as_slice(),
            [
                ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(message_id) },
                ChatEntry::ToolCall { tool_call_id: Some(tool_call_id), name, is_error: false, detail: ToolDetail::Shell { command, .. } },
            ] if content == "preserved while suppressed"
                && thinking == "plan"
                && message_id == "assistant-suppressed"
                && tool_call_id == "shell-1"
                && name == "shell"
                && command == "cargo check"
        ));

        app.render.streaming_cache.store(15, Vec::new());
        app.render.streaming_thinking_cache.store(16, Vec::new());
        app.render.card_cache.processed_messages = 9;
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("question-1".into()),
                name: "question".into(),
                arguments: Some(serde_json::json!({ "prompt": "Continue?" })),
            },
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.chat.activity,
            ActivityState::RunningTool {
                name: "question".into()
            }
        );
        assert_eq!(app.chat.session_stats.total_tool_calls, 1);
        assert_eq!(app.chat.messages.len(), 2);
        assert!(app.render.streaming_cache.get(15).is_some());
        assert!(app.render.streaming_thinking_cache.get(16).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 9);
        assert_eq!(app.diagnostics.status, "tool: question");
        let diagnostic = app.diagnostics.logs.last().expect("question tool status");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "tool");
        assert_eq!(diagnostic.message, "tool: question");

        app.render.streaming_cache.store(17, Vec::new());
        app.render.streaming_thinking_cache.store(18, Vec::new());
        app.render.card_cache.processed_messages = 10;
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("shell-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "ignored duplicate args" })),
            },
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.session_stats.total_tool_calls, 1);
        assert_eq!(app.chat.messages.len(), 2);
        assert!(app.chat.streaming_thinking.is_empty());
        assert!(app.render.streaming_cache.get(17).is_some());
        assert!(app.render.streaming_thinking_cache.get(18).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            &app.chat.messages[1],
            ChatEntry::ToolCall { detail: ToolDetail::Shell { command, .. }, .. }
                if command == "cargo check"
        ));
    }

    #[test]
    fn chat_tool_end_acp_route_is_idempotent_and_release_safe() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.render.streaming_cache.store(11, Vec::new());
        app.render.streaming_thinking_cache.store(12, Vec::new());
        app.render.card_cache.processed_messages = 7;

        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallEnd {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                is_error: false,
                result: Some("successful before start".into()),
            },
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("active"));
        assert!(app.chat.messages.is_empty());
        assert!(app.render.streaming_cache.get(11).is_some());
        assert!(app.render.streaming_thinking_cache.get(12).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 7);

        let failed_end = AcpSessionUpdate::ToolCallEnd {
            tool_call_id: Some("tool-1".into()),
            name: "shell".into(),
            is_error: true,
            result: Some("failed before start".into()),
        };
        let effects = apply_session_update(&mut app, failed_end.clone());
        assert!(effects.is_empty());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::ToolCall { tool_call_id: Some(id), name, is_error: true, detail: ToolDetail::Summary(detail) }]
                if id == "tool-1" && name == "shell (failed)" && detail == "failed before start"
        ));

        app.render.card_cache.processed_messages = 8;
        let effects = apply_session_update(&mut app, failed_end);
        assert!(effects.is_empty());
        assert_eq!(app.chat.messages.len(), 1);
        assert_eq!(app.render.card_cache.processed_messages, 8);

        app.render.streaming_cache.store(13, Vec::new());
        app.render.streaming_thinking_cache.store(14, Vec::new());
        app.render.card_cache.processed_messages = 9;
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                arguments: Some(serde_json::json!({ "command": "echo late" })),
            },
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.session_stats.total_tool_calls, 0);
        assert_eq!(app.chat.messages.len(), 1);
        assert!(app.render.streaming_cache.get(13).is_some());
        assert!(app.render.streaming_thinking_cache.get(14).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "tool: shell");
        let diagnostic = app.diagnostics.logs.last().expect("late shell status");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "tool");
        assert_eq!(diagnostic.message, "tool: shell");
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::ToolCall { tool_call_id: Some(id), name, is_error: true, detail: ToolDetail::Shell { command, output_tail: None, .. } }]
                if id == "tool-1" && name == "shell" && command == "echo late"
        ));

        app.render.streaming_cache.store(15, Vec::new());
        app.render.streaming_thinking_cache.store(16, Vec::new());
        app.render.card_cache.processed_messages = 10;
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallEnd {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                is_error: true,
                result: Some("line one\nline two".into()),
            },
        );
        assert!(effects.is_empty());
        assert!(app.render.streaming_cache.get(15).is_some());
        assert!(app.render.streaming_thinking_cache.get(16).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::ToolCall { is_error: true, detail: ToolDetail::Shell { output_tail: Some(tail), .. }, .. }]
                if tail.lines == ["line one", "line two"]
        ));

        app.render.card_cache.processed_messages = 11;
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::ToolCallEnd {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                is_error: true,
                result: Some("line one\nline two".into()),
            },
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.messages.len(), 1);
        assert_eq!(app.render.card_cache.processed_messages, 11);
    }

    #[test]
    fn inactive_child_acp_route_marks_activity_without_mutating_active_chat() {
        let mut app = App::new();
        app.sessions.session_id = Some("parent".into());
        app.chat.messages.push(ChatEntry::Error("preserved".into()));
        app.chat.streaming_content = "active stream".into();
        app.chat.activity = ActivityState::Streaming;
        app.chat.apply_usage(900, 9_000, Some(9.0));
        app.render.card_cache.processed_messages = 6;

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "child-1".into(),
                update: crate::acp_state::AcpSessionUpdate::AssistantContentDelta {
                    content: "child content".into(),
                    message_id: Some("child-message".into()),
                },
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("child-1"));
        assert_eq!(app.render.card_cache.processed_messages, 6);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "preserved"
        ));
        assert_eq!(app.chat.streaming_content, "active stream");
        assert_eq!(app.chat.activity, ActivityState::Streaming);
        assert_eq!(
            app.delegates.pending_delegate_child_states["child-1"],
            DelegateChildState::AssistantMessage
        );
        assert_eq!(
            app.delegates.pending_delegate_child_stats["child-1"].messages,
            1
        );

        let log_count_before_usage = app.diagnostics.logs.len();
        for (used, size, cost_usd) in [(100, 1_000, Some(1.5)), (250, 2_000, Some(0.25))] {
            let effects = update(
                &mut app,
                AppEvent::Acp(AcpAppEvent::SessionUpdate {
                    session_id: "child-1".into(),
                    update: crate::acp_state::AcpSessionUpdate::UsageUpdate {
                        used,
                        size,
                        cost_usd,
                    },
                    is_replay: false,
                }),
            );
            assert!(effects.is_empty());
        }
        let child_stats = &app.delegates.pending_delegate_child_stats["child-1"];
        assert_eq!(child_stats.messages, 1);
        assert_eq!(child_stats.context_tokens, 250);
        assert_eq!(child_stats.context_limit, 2_000);
        assert_eq!(child_stats.cost_usd, 0.25);
        assert_eq!(app.chat.session_stats.latest_context_tokens, Some(900));
        assert_eq!(app.chat.context_limit, 9_000);
        assert_eq!(app.chat.cumulative_cost, Some(9.0));
        assert_eq!(app.diagnostics.logs.len(), log_count_before_usage);
        assert_eq!(app.render.card_cache.processed_messages, 6);

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::DelegationUpdate(delegation_update(
                "parent",
                DelegationState::Forked,
                110,
            ))),
        );
        assert!(effects.is_empty());
        let entry = &app.delegates.delegate_entries[0];
        assert_eq!(entry.child_session_id.as_deref(), Some("child-1"));
        assert_eq!(entry.stats.messages, 1);
        assert_eq!(entry.stats.context_tokens, 250);
        assert_eq!(entry.stats.context_limit, 2_000);
        assert_eq!(entry.stats.cost_usd, 0.25);
        assert_eq!(entry.child_state, DelegateChildState::AssistantMessage);
        assert!(
            !app.delegates
                .pending_delegate_child_states
                .contains_key("child-1")
        );
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "preserved"
        ));
        assert_eq!(app.chat.streaming_content, "active stream");
        assert_eq!(app.chat.activity, ActivityState::Streaming);
    }

    #[test]
    fn chat_elicitation_acp_route_is_owner_bounded_and_release_safe() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.chat.streaming_thinking = "plan".into();
        app.chat.streaming_thinking_message_id = Some("assistant-1".into());
        app.chat.streaming_content = "answer".into();
        app.chat.streaming_content_message_id = Some("assistant-1".into());
        app.chat.scroll_offset = 8;
        app.render.streaming_cache.store(6, Vec::new());
        app.render.streaming_thinking_cache.store(4, Vec::new());
        app.render.card_cache.processed_messages = 7;
        let log_count_before = app.diagnostics.logs.len();

        let effects = apply_session_update(&mut app, supported_elicitation("elic-live"));
        assert!(effects.is_empty());
        assert_eq!(
            app.chat
                .elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-live")
        );
        assert!(app.chat.elicitation_ui.is_some());
        assert_eq!(app.chat.scroll_offset, 0);
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_thinking.is_empty());
        assert!(app.render.streaming_cache.get(6).is_none());
        assert!(app.render.streaming_thinking_cache.get(4).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [
                ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(message_id) },
                ChatEntry::Elicitation { elicitation_id, outcome: None, .. },
            ] if content == "answer"
                && thinking == "plan"
                && message_id == "assistant-1"
                && elicitation_id == "elic-live"
        ));
        assert_eq!(
            app.diagnostics.status,
            "question - answer in the panel above input"
        );
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Info);
        assert_eq!(diagnostics[0].target, "elicitation");
        assert_eq!(
            diagnostics[0].message,
            "question - answer in the panel above input"
        );

        app.render.streaming_cache.store(9, Vec::new());
        app.render.streaming_thinking_cache.store(10, Vec::new());
        app.render.card_cache.processed_messages = 2;
        let log_count_before_duplicate = app.diagnostics.logs.len();
        let effects = apply_session_update(&mut app, supported_elicitation("elic-live"));
        assert!(effects.is_empty());
        assert_eq!(app.chat.messages.len(), 2);
        assert!(app.render.streaming_cache.get(9).is_some());
        assert!(app.render.streaming_thinking_cache.get(10).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 2);
        assert_eq!(app.diagnostics.logs.len(), log_count_before_duplicate);
    }

    #[test]
    fn chat_elicitation_replay_and_unsupported_routes_preserve_exact_behavior() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.render.streaming_cache.store(5, Vec::new());
        app.render.streaming_thinking_cache.store(6, Vec::new());
        app.render.card_cache.processed_messages = 7;
        let log_count_before_replay = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionReplay {
                session_id: "active".into(),
                updates: vec![supported_elicitation("elic-replay")],
            }),
        );
        assert!(effects.is_empty());
        assert!(app.chat.elicitation.is_none());
        assert!(app.chat.elicitation_ui.is_none());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: None, .. }]
                if elicitation_id == "elic-replay"
        ));
        assert!(app.render.streaming_cache.get(5).is_some());
        assert!(app.render.streaming_thinking_cache.get(6).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 7);
        assert_eq!(
            app.diagnostics.status,
            "question - answer in the panel above input"
        );
        let replay_diagnostics = &app.diagnostics.logs[log_count_before_replay..];
        assert_eq!(replay_diagnostics.len(), 2);
        assert_eq!(replay_diagnostics[0].level, LogLevel::Info);
        assert_eq!(replay_diagnostics[0].target, "session");
        assert_eq!(replay_diagnostics[0].message, "session replay: 1 update(s)");
        assert_eq!(replay_diagnostics[1].level, LogLevel::Info);
        assert_eq!(replay_diagnostics[1].target, "elicitation");
        assert_eq!(
            replay_diagnostics[1].message,
            "question - answer in the panel above input"
        );

        let log_count_before_unsupported = app.diagnostics.logs.len();
        let effects = apply_session_update(&mut app, unsupported_elicitation("elic-unsupported"));
        assert!(effects.is_empty());
        assert!(app.chat.elicitation.is_none());
        assert!(app.chat.elicitation_ui.is_none());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [
                ChatEntry::Elicitation { elicitation_id: replay, outcome: None, .. },
                ChatEntry::Elicitation { elicitation_id: unsupported, outcome: Some(outcome), .. },
            ] if replay == "elic-replay"
                && unsupported == "elic-unsupported"
                && outcome == "unsupported schema - cannot answer in TUI"
        ));
        assert_eq!(app.render.card_cache.processed_messages, 7);
        assert_eq!(
            app.diagnostics.status,
            "question skipped - unsupported schema"
        );
        let unsupported_diagnostic = app
            .diagnostics
            .logs
            .get(log_count_before_unsupported)
            .expect("unsupported diagnostic");
        assert_eq!(unsupported_diagnostic.level, LogLevel::Warn);
        assert_eq!(unsupported_diagnostic.target, "elicitation");
        assert_eq!(
            unsupported_diagnostic.message,
            "question skipped - unsupported schema"
        );
    }

    #[test]
    fn inactive_child_elicitation_stays_in_delegate_state_without_chat_mutation() {
        let mut app = App::new();
        app.sessions.session_id = Some("parent".into());
        add_elicitation(&mut app, "active-elicitation", true);
        app.chat.streaming_content = "active stream".into();
        app.render.card_cache.processed_messages = 6;
        app.diagnostics.status = "preserved status".into();
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "child-1".into(),
                update: supported_elicitation("child-elicitation"),
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.chat
                .elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("active-elicitation")
        );
        assert!(app.chat.elicitation_ui.is_some());
        assert_eq!(app.chat.streaming_content, "active stream");
        assert_eq!(app.chat.messages.len(), 1);
        assert_eq!(app.render.card_cache.processed_messages, 6);
        assert_eq!(app.diagnostics.status, "preserved status");
        assert_eq!(app.diagnostics.logs.len(), log_count_before);
        assert!(matches!(
            &app.delegates.pending_delegate_child_states["child-1"],
            DelegateChildState::PendingElicitation { elicitation_id, message, source, .. }
                if elicitation_id == "child-elicitation"
                    && message == "Question child-elicitation"
                    && source == "builtin:question"
        ));
    }

    #[test]
    fn chat_usage_and_timing_acp_route_is_owner_bounded_and_release_safe() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.chat.apply_usage(128, 4_096, Some(1.5));
        app.chat.add_active_llm_duration(Duration::from_secs(2));
        app.render.streaming_cache.store(5, Vec::new());
        app.render.streaming_thinking_cache.store(6, Vec::new());
        app.render.card_cache.processed_messages = 7;
        app.diagnostics.status = "preserved status".into();
        app.push_log(LogLevel::Debug, "test", "before usage");
        let log_count_before = app.diagnostics.logs.len();

        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::UsageUpdate {
                used: 2_048,
                size: 8_192,
                cost_usd: Some(0.0123),
            },
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("active"));
        assert_eq!(app.chat.session_stats.latest_context_tokens, Some(2_048));
        assert_eq!(app.chat.context_limit, 8_192);
        assert_eq!(app.chat.cumulative_cost, Some(0.0123));

        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::UsageUpdate {
                used: 512,
                size: 0,
                cost_usd: None,
            },
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.session_stats.latest_context_tokens, Some(512));
        assert_eq!(app.chat.context_limit, 8_192);
        assert_eq!(app.chat.cumulative_cost, Some(0.0123));

        app.sessions.session_activity.remove("active");
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::TimingUpdate { duration_secs: 3 },
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("active"));
        assert_eq!(app.chat.llm_request_elapsed(), Some(Duration::from_secs(5)));

        app.sessions.session_activity.remove("active");
        let log_count_before_zero = app.diagnostics.logs.len();
        let effects = apply_session_update(
            &mut app,
            AcpSessionUpdate::TimingUpdate { duration_secs: 0 },
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("active"));
        assert_eq!(app.chat.llm_request_elapsed(), Some(Duration::from_secs(5)));
        assert_eq!(app.diagnostics.logs.len(), log_count_before_zero);

        assert_eq!(app.diagnostics.status, "preserved status");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 3);
        assert_eq!(diagnostics[0].level, LogLevel::Info);
        assert_eq!(diagnostics[0].target, "usage");
        assert_eq!(
            diagnostics[0].message,
            "usage: context 2048/8192 tokens (25%), cost $0.0123"
        );
        assert_eq!(diagnostics[1].level, LogLevel::Info);
        assert_eq!(diagnostics[1].target, "usage");
        assert_eq!(diagnostics[1].message, "usage: context 512/0 tokens");
        assert_eq!(diagnostics[2].level, LogLevel::Info);
        assert_eq!(diagnostics[2].target, "usage");
        assert_eq!(diagnostics[2].message, "usage: active time 3s");
        assert!(app.render.streaming_cache.get(5).is_some());
        assert!(app.render.streaming_thinking_cache.get(6).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 7);
    }

    #[test]
    fn chat_usage_timing_replay_preserves_summary_and_diagnostic_order() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.render.streaming_cache.store(8, Vec::new());
        app.render.streaming_thinking_cache.store(9, Vec::new());
        app.render.card_cache.processed_messages = 10;
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionReplay {
                session_id: "active".into(),
                updates: vec![
                    AcpSessionUpdate::TimingUpdate { duration_secs: 4 },
                    AcpSessionUpdate::UsageUpdate {
                        used: 30,
                        size: 120,
                        cost_usd: Some(0.5),
                    },
                ],
            }),
        );

        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("active"));
        assert_eq!(app.chat.llm_request_elapsed(), Some(Duration::from_secs(4)));
        assert_eq!(app.chat.session_stats.latest_context_tokens, Some(30));
        assert_eq!(app.chat.context_limit, 120);
        assert_eq!(app.chat.cumulative_cost, Some(0.5));
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 3);
        assert_eq!(diagnostics[0].level, LogLevel::Info);
        assert_eq!(diagnostics[0].target, "session");
        assert_eq!(diagnostics[0].message, "session replay: 2 update(s)");
        assert_eq!(diagnostics[1].level, LogLevel::Info);
        assert_eq!(diagnostics[1].target, "usage");
        assert_eq!(diagnostics[1].message, "usage: active time 4s");
        assert_eq!(diagnostics[2].level, LogLevel::Info);
        assert_eq!(diagnostics[2].target, "usage");
        assert_eq!(
            diagnostics[2].message,
            "usage: context 30/120 tokens (25%), cost $0.5000"
        );
        assert!(app.render.streaming_cache.get(8).is_some());
        assert!(app.render.streaming_thinking_cache.get(9).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 10);
    }

    #[test]
    fn chat_turn_and_user_routes_apply_release_sensitive_coordination() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.chat.streaming_content = "discarded".into();
        app.chat.streaming_thinking = "discarded thinking".into();
        app.render.streaming_cache.store(9, Vec::new());
        app.render.streaming_thinking_cache.store(8, Vec::new());
        app.render.card_cache.processed_messages = 7;

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::TurnStarted,
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert!(app.sessions.session_activity.contains_key("active"));
        assert_eq!(app.chat.activity, ActivityState::Thinking);
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_thinking.is_empty());
        assert!(app.chat.session_stats.open_llm_request_instant.is_some());
        assert!(app.render.streaming_cache.get(9).is_none());
        assert!(app.render.streaming_thinking_cache.get(8).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 7);
        assert_eq!(app.diagnostics.status, "thinking...");
        let diagnostic = app.diagnostics.logs.last().expect("turn diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "activity");
        assert_eq!(diagnostic.message, "thinking...");

        let local_id = app.chat.push_pending_prompt("  repeat\n".into());
        app.render.card_cache.processed_messages = 8;
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::UserMessage {
                    content: serde_json::json!({ "text": "repeat" }),
                    message_id: Some("server-user".into()),
                },
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(id) }]
                if text == "repeat" && id == "server-user" && id != &local_id
        ));
        assert_eq!(app.chat.undoable_turns[0].message_id, "server-user");

        app.sessions.session_activity.remove("active");
        app.render.card_cache.processed_messages = 9;
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::UserMessage {
                    content: serde_json::json!({ "text": "repeat" }),
                    message_id: Some("server-user".into()),
                },
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.render.card_cache.processed_messages, 9);
        assert_eq!(app.chat.messages.len(), 1);
    }

    #[test]
    fn chat_stream_and_final_routes_preserve_boundaries_and_invalidation_scope() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.chat.activity = ActivityState::Thinking;
        app.render.streaming_cache.store(5, Vec::new());
        app.render.streaming_thinking_cache.store(6, Vec::new());
        app.render.card_cache.processed_messages = 7;

        for update_value in [
            crate::acp_state::AcpSessionUpdate::AssistantThinkingDelta {
                content: "plan".into(),
                message_id: Some("assistant-1".into()),
            },
            crate::acp_state::AcpSessionUpdate::AssistantContentDelta {
                content: "first".into(),
                message_id: Some("assistant-1".into()),
            },
        ] {
            let effects = update(
                &mut app,
                AppEvent::Acp(AcpAppEvent::SessionUpdate {
                    session_id: "active".into(),
                    update: update_value,
                    is_replay: false,
                }),
            );
            assert!(effects.is_empty());
        }
        assert!(app.render.streaming_cache.get(5).is_some());
        assert!(app.render.streaming_thinking_cache.get(6).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 7);

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::AssistantThinkingDelta {
                    content: "second plan".into(),
                    message_id: Some("assistant-2".into()),
                },
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert!(app.render.streaming_cache.get(5).is_none());
        assert!(app.render.streaming_thinking_cache.get(6).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(id) }]
                if content == "first" && thinking == "plan" && id == "assistant-1"
        ));
        assert_eq!(app.chat.streaming_thinking, "second plan");

        app.render.streaming_cache.store(10, Vec::new());
        app.render.streaming_thinking_cache.store(11, Vec::new());
        app.render.card_cache.processed_messages = 12;
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::AssistantMessage {
                    content: "final".into(),
                    thinking: None,
                    message_id: Some("assistant-2".into()),
                },
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert!(app.render.streaming_cache.get(10).is_none());
        assert!(app.render.streaming_thinking_cache.get(11).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 12);
        assert!(matches!(
            &app.chat.messages[1],
            ChatEntry::Assistant { content, thinking: Some(thinking), message_id: Some(id) }
                if content == "final" && thinking == "second plan" && id == "assistant-2"
        ));
    }

    #[test]
    fn chat_cancel_and_finish_routes_discard_streams_and_emit_exact_diagnostics() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::TurnStarted,
                is_replay: false,
            }),
        );
        app.chat.streaming_content = "discarded".into();
        app.chat.streaming_thinking = "discarded thinking".into();
        app.render.streaming_cache.store(9, Vec::new());
        app.render.streaming_thinking_cache.store(8, Vec::new());
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::Cancelled,
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert!(app.chat.session_stats.open_llm_request_instant.is_none());
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_thinking.is_empty());
        assert!(app.chat.messages.is_empty());
        assert!(app.render.streaming_cache.get(9).is_none());
        assert!(app.render.streaming_thinking_cache.get(8).is_none());
        assert_eq!(app.diagnostics.status, "cancelled");
        let diagnostic = app.diagnostics.logs.last().expect("cancel diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Warn);
        assert_eq!(diagnostic.target, "activity");
        assert_eq!(diagnostic.message, "cancelled");

        update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::TurnStarted,
                is_replay: false,
            }),
        );
        app.chat.streaming_content = "discarded final".into();
        app.chat.streaming_thinking = "discarded final thinking".into();
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionUpdate {
                session_id: "active".into(),
                update: crate::acp_state::AcpSessionUpdate::Finished {
                    finish_reason: "EndTurn".into(),
                },
                is_replay: false,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert!(app.chat.session_stats.open_llm_request_instant.is_none());
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_thinking.is_empty());
        assert!(app.chat.messages.is_empty());
        assert_eq!(app.diagnostics.status, "finished: EndTurn");
        let diagnostic = app.diagnostics.logs.last().expect("finish diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "activity");
        assert_eq!(diagnostic.message, "finished: EndTurn");
    }

    #[test]
    fn chat_acp_error_route_is_unscoped_deduplicated_and_release_safe() {
        let mut app = App::new();
        app.chat.activity = ActivityState::Streaming;
        app.chat.recent_prompt_text = Some("preserved prompt".into());
        app.chat.messages.push(ChatEntry::Error("preserved".into()));
        app.chat.session_stats.open_llm_request_instant =
            Some(Instant::now() - Duration::from_secs(2));
        app.render.streaming_cache.store(5, Vec::new());
        app.render.streaming_thinking_cache.store(6, Vec::new());
        app.render.card_cache.processed_messages = 7;
        app.composer.input = "preserved input".into();
        app.mesh.mesh_invite_name = "preserved invite".into();
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Error {
                message: "connection lost".into(),
            }),
        );

        assert!(effects.is_empty());
        assert!(app.sessions.session_id.is_none());
        assert_eq!(app.chat.activity, ActivityState::Streaming);
        assert!(app.chat.session_stats.open_llm_request_instant.is_none());
        assert!(app.chat.session_stats.active_llm_duration >= Duration::from_secs(2));
        assert_eq!(
            app.chat.recent_prompt_text.as_deref(),
            Some("preserved prompt")
        );
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(preserved), ChatEntry::Error(inserted)]
                if preserved == "preserved" && inserted == "connection lost"
        ));
        assert!(app.render.streaming_cache.get(5).is_some());
        assert!(app.render.streaming_thinking_cache.get(6).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 7);
        assert_eq!(app.diagnostics.status, "error: connection lost");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Error);
        assert_eq!(diagnostics[0].target, "acp");
        assert_eq!(diagnostics[0].message, "error: connection lost");
        assert_eq!(app.composer.input, "preserved input");
        assert_eq!(app.mesh.mesh_invite_name, "preserved invite");

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Error {
                message: "connection lost".into(),
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.messages.len(), 2);
        assert_eq!(app.render.card_cache.processed_messages, 7);
        assert_eq!(app.diagnostics.logs.len(), log_count_before + 1);
    }

    #[test]
    fn chat_backend_prompt_failure_route_is_owner_bounded_and_release_safe() {
        let mut app = App::new();
        let failed_id = app.chat.push_pending_prompt("failed".into());
        let retained_id = app.chat.push_pending_prompt("retained".into());
        app.chat.activity = ActivityState::Thinking;
        app.chat.recent_prompt_text = Some("preserved prompt".into());
        app.chat.session_stats.open_llm_request_instant =
            Some(Instant::now() - Duration::from_secs(2));
        app.render.streaming_cache.store(8, Vec::new());
        app.render.streaming_thinking_cache.store(9, Vec::new());
        app.render.card_cache.processed_messages = 10;
        app.composer.input = "preserved input".into();
        app.mesh.mesh_invite_name = "preserved invite".into();
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::PromptFailed {
                local_id: failed_id.clone(),
                message: "backend rejected prompt".into(),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::Thinking);
        assert!(app.chat.session_stats.open_llm_request_instant.is_none());
        assert!(app.chat.session_stats.active_llm_duration >= Duration::from_secs(2));
        assert_eq!(
            app.chat.recent_prompt_text.as_deref(),
            Some("preserved prompt")
        );
        assert!(matches!(
            app.chat.messages.as_slice(),
            [
                ChatEntry::User { text, message_id: Some(message_id) },
                ChatEntry::Error(message),
            ] if text == "retained"
                && message_id == &retained_id
                && message == "backend rejected prompt"
        ));
        assert!(app.chat.messages.iter().all(|entry| {
            !matches!(entry, ChatEntry::User { message_id: Some(id), .. } if id == &failed_id)
        }));
        assert!(app.render.streaming_cache.get(8).is_some());
        assert!(app.render.streaming_thinking_cache.get(9).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "error: backend rejected prompt");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Error);
        assert_eq!(diagnostics[0].target, "acp");
        assert_eq!(diagnostics[0].message, "error: backend rejected prompt");
        assert_eq!(app.composer.input, "preserved input");
        assert_eq!(app.mesh.mesh_invite_name, "preserved invite");
    }

    #[test]
    fn chat_history_stack_route_replaces_only_undo_state() {
        let mut app = App::new();
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.chat.streaming_content = "content".into();
        app.chat.streaming_content_message_id = Some("content-id".into());
        app.chat.streaming_thinking = "thinking".into();
        app.chat.streaming_thinking_message_id = Some("thinking-id".into());
        app.chat.recent_prompt_text = Some("prompt".into());
        app.chat.pending_fork_message_id = Some("fork-target".into());
        app.render.streaming_cache.store(7, Vec::new());
        app.render.streaming_thinking_cache.store(8, Vec::new());
        app.render.card_cache.processed_messages = 9;
        let log_count = app.diagnostics.logs.len();
        let status = app.diagnostics.status.clone();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::UndoStack(UndoStackSnapshot {
                message_ids: vec!["one".into()],
            })),
        );

        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::SessionOp(SessionOp::Undo));
        assert_eq!(app.chat.streaming_content, "content");
        assert_eq!(
            app.chat.streaming_content_message_id.as_deref(),
            Some("content-id")
        );
        assert_eq!(app.chat.streaming_thinking, "thinking");
        assert_eq!(
            app.chat.streaming_thinking_message_id.as_deref(),
            Some("thinking-id")
        );
        assert_eq!(app.chat.recent_prompt_text.as_deref(), Some("prompt"));
        assert_eq!(
            app.chat.pending_fork_message_id.as_deref(),
            Some("fork-target")
        );
        assert_eq!(
            app.chat
                .undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("one")
        );
        assert!(app.render.streaming_cache.get(7).is_some());
        assert!(app.render.streaming_thinking_cache.get(8).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 9);
        assert_eq!(app.diagnostics.logs.len(), log_count);
        assert_eq!(app.diagnostics.status, status);
    }

    #[test]
    fn chat_history_undo_routes_apply_release_sensitive_asymmetry_and_reload_context() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.sessions.agent_id = Some("agent-current".into());
        app.sessions.session_groups = vec![SessionGroup {
            cwd: Some("/group".into()),
            sessions: vec![SessionSummary {
                session_id: "active".into(),
                cwd: Some("/active-session".into()),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.connection.launch_cwd = Some("/launch".into());
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.chat.undoable_turns = vec![
            UndoableTurn {
                turn_id: "turn-one".into(),
                message_id: "one".into(),
                text: "first".into(),
            },
            UndoableTurn {
                turn_id: "turn-two".into(),
                message_id: "two".into(),
                text: "second".into(),
            },
        ];
        app.chat.streaming_content = "discard content".into();
        app.chat.streaming_content_message_id = Some("content-id".into());
        app.chat.streaming_thinking = "keep thinking".into();
        app.chat.streaming_thinking_message_id = Some("thinking-id".into());
        app.chat.recent_prompt_text = Some("discard prompt".into());
        app.render.streaming_cache.store(15, Vec::new());
        app.render.streaming_thinking_cache.store(13, Vec::new());
        app.render.card_cache.processed_messages = 7;
        let log_count = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::UndoResult(UndoResult::Applied {
                target_message_id: None,
                reverted_files: vec!["src/main.rs".into()],
                message: Some("ignored success".into()),
                stack: UndoStackSnapshot {
                    message_ids: vec!["one".into(), "two".into()],
                },
            })),
        );

        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_content_message_id.is_none());
        assert_eq!(app.chat.streaming_thinking, "keep thinking");
        assert_eq!(
            app.chat.streaming_thinking_message_id.as_deref(),
            Some("thinking-id")
        );
        assert!(app.chat.recent_prompt_text.is_none());
        let state = app.chat.undo_state.as_ref().expect("undo state");
        assert_eq!(state.frontier_message_id.as_deref(), Some("two"));
        assert!(state.stack[0].reverted_files.is_empty());
        assert_eq!(state.stack[1].reverted_files, ["src/main.rs"]);
        assert!(app.render.streaming_cache.get(15).is_none());
        assert!(app.render.streaming_thinking_cache.get(13).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 7);
        assert_eq!(app.diagnostics.status, "undone - reloading session");
        let diagnostics = &app.diagnostics.logs[log_count..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Info);
        assert_eq!(diagnostics[0].target, "session");
        assert_eq!(diagnostics[0].message, "undone - reloading session");
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::LoadSession {
                    session_id: "active".into(),
                    cwd: Some("/active-session".into()),
                }),
                Effect::Command(Command::SubscribeSession {
                    session_id: "active".into(),
                    agent_id: Some("agent-current".into()),
                }),
            ]
        );

        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.chat.streaming_content = "preserve content".into();
        app.chat.streaming_content_message_id = Some("preserve-content-id".into());
        app.chat.streaming_thinking = "preserve thinking".into();
        app.chat.streaming_thinking_message_id = Some("preserve-thinking-id".into());
        app.chat.recent_prompt_text = Some("preserve prompt".into());
        app.render.streaming_cache.store(16, Vec::new());
        app.render.streaming_thinking_cache.store(17, Vec::new());
        app.render.card_cache.processed_messages = 8;
        let log_count = app.diagnostics.logs.len();
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::UndoResult(UndoResult::Rejected {
                target_message_id: Some("one".into()),
                message: None,
                stack: UndoStackSnapshot {
                    message_ids: vec!["one".into(), "two".into()],
                },
            })),
        );

        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert_eq!(app.chat.streaming_content, "preserve content");
        assert_eq!(
            app.chat.streaming_content_message_id.as_deref(),
            Some("preserve-content-id")
        );
        assert_eq!(app.chat.streaming_thinking, "preserve thinking");
        assert_eq!(
            app.chat.streaming_thinking_message_id.as_deref(),
            Some("preserve-thinking-id")
        );
        assert_eq!(
            app.chat.recent_prompt_text.as_deref(),
            Some("preserve prompt")
        );
        assert_eq!(
            app.chat
                .undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("one")
        );
        assert!(app.render.streaming_cache.get(16).is_some());
        assert!(app.render.streaming_thinking_cache.get(17).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 8);
        assert_eq!(app.diagnostics.status, "undo failed");
        let diagnostics = &app.diagnostics.logs[log_count..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Warn);
        assert_eq!(diagnostics[0].target, "session");
        assert_eq!(diagnostics[0].message, "undo failed");
    }

    #[test]
    fn chat_history_redo_routes_preserve_streams_and_reload_only_when_applied() {
        let mut app = App::new();
        app.sessions.session_id = Some("active".into());
        app.sessions.agent_id = Some("agent-current".into());
        app.sessions.session_groups = vec![SessionGroup {
            cwd: Some("/group-cwd".into()),
            sessions: vec![SessionSummary {
                session_id: "active".into(),
                cwd: None,
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.connection.launch_cwd = Some("/launch".into());
        app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
        app.chat.streaming_content = "keep content".into();
        app.chat.streaming_content_message_id = Some("content-id".into());
        app.chat.streaming_thinking = "keep thinking".into();
        app.chat.streaming_thinking_message_id = Some("thinking-id".into());
        app.chat.recent_prompt_text = Some("keep prompt".into());
        app.render.streaming_cache.store(12, Vec::new());
        app.render.streaming_thinking_cache.store(13, Vec::new());
        app.render.card_cache.processed_messages = 14;
        let log_count = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::RedoResult(RedoResult::Applied {
                message: Some("ignored success".into()),
                stack: UndoStackSnapshot {
                    message_ids: vec!["one".into()],
                },
            })),
        );

        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert_eq!(app.chat.streaming_content, "keep content");
        assert_eq!(
            app.chat.streaming_content_message_id.as_deref(),
            Some("content-id")
        );
        assert_eq!(app.chat.streaming_thinking, "keep thinking");
        assert_eq!(
            app.chat.streaming_thinking_message_id.as_deref(),
            Some("thinking-id")
        );
        assert_eq!(app.chat.recent_prompt_text.as_deref(), Some("keep prompt"));
        assert!(app.render.streaming_cache.get(12).is_some());
        assert!(app.render.streaming_thinking_cache.get(13).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 14);
        assert_eq!(app.diagnostics.status, "redone - reloading session");
        let diagnostics = &app.diagnostics.logs[log_count..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Info);
        assert_eq!(diagnostics[0].target, "session");
        assert_eq!(diagnostics[0].message, "redone - reloading session");
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::LoadSession {
                    session_id: "active".into(),
                    cwd: Some("/group-cwd".into()),
                }),
                Effect::Command(Command::SubscribeSession {
                    session_id: "active".into(),
                    agent_id: Some("agent-current".into()),
                }),
            ]
        );

        app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
        let log_count = app.diagnostics.logs.len();
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::RedoResult(RedoResult::Rejected {
                message: None,
                stack: UndoStackSnapshot {
                    message_ids: vec!["one".into(), "two".into()],
                },
            })),
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert_eq!(app.diagnostics.status, "redo failed");
        assert_eq!(
            app.chat
                .undo_state
                .as_ref()
                .and_then(|state| state.frontier_message_id.as_deref()),
            Some("one")
        );
        let diagnostics = &app.diagnostics.logs[log_count..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Warn);
        assert_eq!(diagnostics[0].target, "session");
        assert_eq!(diagnostics[0].message, "redo failed");
    }

    #[test]
    fn chat_history_fork_routes_apply_popup_diagnostics_and_load_context_exactly() {
        let mut app = App::new();
        app.sessions.agent_id = Some("agent-current".into());
        app.connection.launch_cwd = Some("/launch-cwd".into());
        app.navigation.popup = Popup::ForkTurnSelect;
        app.chat.pending_fork_message_id = Some("fork-target".into());
        let log_count = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ForkResult(ForkResult::Succeeded {
                source_session_id: Some("ignored-source".into()),
                forked_session_id: Some("forked".into()),
                message: Some("ignored success".into()),
            })),
        );

        assert!(app.chat.pending_fork_message_id.is_none());
        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(app.diagnostics.status, "forked - loading session");
        let diagnostics = &app.diagnostics.logs[log_count..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Info);
        assert_eq!(diagnostics[0].target, "fork");
        assert_eq!(diagnostics[0].message, "forked - loading session");
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::LoadSession {
                    session_id: "forked".into(),
                    cwd: Some("/launch-cwd".into()),
                }),
                Effect::Command(Command::SubscribeSession {
                    session_id: "forked".into(),
                    agent_id: Some("agent-current".into()),
                }),
            ]
        );

        app.navigation.popup = Popup::ForkTurnSelect;
        app.chat.pending_fork_message_id = Some("fork-target".into());
        let log_count = app.diagnostics.logs.len();
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ForkResult(ForkResult::Succeeded {
                source_session_id: Some("ignored-source".into()),
                forked_session_id: None,
                message: None,
            })),
        );
        assert!(effects.is_empty());
        assert!(app.chat.pending_fork_message_id.is_none());
        assert_eq!(app.navigation.popup, Popup::ForkTurnSelect);
        assert_eq!(app.diagnostics.status, "fork succeeded without session id");
        let diagnostics = &app.diagnostics.logs[log_count..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Warn);
        assert_eq!(diagnostics[0].target, "fork");
        assert_eq!(diagnostics[0].message, "fork succeeded without session id");

        app.chat.pending_fork_message_id = Some("fork-target".into());
        let log_count = app.diagnostics.logs.len();
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ForkResult(ForkResult::Failed {
                source_session_id: Some("ignored-source".into()),
                message: None,
            })),
        );
        assert!(effects.is_empty());
        assert!(app.chat.pending_fork_message_id.is_none());
        assert_eq!(app.navigation.popup, Popup::ForkTurnSelect);
        assert_eq!(app.diagnostics.status, "fork failed");
        let diagnostics = &app.diagnostics.logs[log_count..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Warn);
        assert_eq!(diagnostics[0].target, "fork");
        assert_eq!(diagnostics[0].message, "fork failed");
    }

    #[test]
    fn chat_history_applied_results_without_active_session_still_coordinate_in_release() {
        let mut app = App::new();
        app.sessions.agent_id = Some("unused-agent".into());
        app.connection.launch_cwd = Some("/unused-cwd".into());
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.chat.streaming_content = "discard".into();
        app.chat.streaming_content_message_id = Some("content-id".into());
        app.chat.streaming_thinking = "keep thinking".into();
        app.chat.streaming_thinking_message_id = Some("thinking-id".into());
        app.chat.recent_prompt_text = Some("discard prompt".into());
        app.render.streaming_cache.store(7, Vec::new());
        app.render.streaming_thinking_cache.store(8, Vec::new());
        let log_count = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::UndoResult(UndoResult::Applied {
                target_message_id: None,
                reverted_files: Vec::new(),
                message: None,
                stack: UndoStackSnapshot {
                    message_ids: vec!["one".into()],
                },
            })),
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert!(app.chat.streaming_content.is_empty());
        assert!(app.chat.streaming_content_message_id.is_none());
        assert_eq!(app.chat.streaming_thinking, "keep thinking");
        assert!(app.chat.recent_prompt_text.is_none());
        assert!(app.render.streaming_cache.get(7).is_none());
        assert!(app.render.streaming_thinking_cache.get(8).is_some());
        assert_eq!(app.diagnostics.status, "undone - reloading session");
        assert_eq!(app.diagnostics.logs.len(), log_count + 1);

        app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
        app.chat.streaming_content = "keep redo content".into();
        app.chat.streaming_content_message_id = Some("redo-content-id".into());
        app.chat.recent_prompt_text = Some("keep redo prompt".into());
        let log_count = app.diagnostics.logs.len();
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::RedoResult(RedoResult::Applied {
                message: None,
                stack: UndoStackSnapshot {
                    message_ids: vec!["one".into()],
                },
            })),
        );
        assert!(effects.is_empty());
        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert_eq!(app.chat.streaming_content, "keep redo content");
        assert_eq!(
            app.chat.streaming_content_message_id.as_deref(),
            Some("redo-content-id")
        );
        assert_eq!(
            app.chat.recent_prompt_text.as_deref(),
            Some("keep redo prompt")
        );
        assert_eq!(app.diagnostics.status, "redone - reloading session");
        assert_eq!(app.diagnostics.logs.len(), log_count + 1);
    }

    #[test]
    fn profile_acp_route_preserves_selection_and_applies_refresh_conditions() {
        let mut app = App::new();
        app.profiles.active_profile_id = Some("deep".into());
        app.models.agents_profile_id = Some("deep".into());
        app.models.agents = vec![agent("primary"), agent("coder")];
        app.auth.filter = "preserved auth".into();
        app.mesh.mesh_invite_name = "preserved mesh".into();
        app.chat
            .messages
            .push(ChatEntry::Error("preserved chat".into()));

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Profiles {
                profiles: vec![profile("fast"), profile("deep")],
                active_profile_id: Some("fast".into()),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("deep"));
        assert_eq!(app.models.agents_profile_id.as_deref(), Some("deep"));
        assert_eq!(app.models.agents.len(), 2);

        app.sessions.session_id = Some("session-1".into());
        app.profiles
            .bind_session_profile("session-1".into(), "fast".into());
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Profiles {
                profiles: vec![profile("fast"), profile("deep")],
                active_profile_id: Some("deep".into()),
            }),
        );
        assert!(app.models.agents.is_empty());
        assert_eq!(app.models.agents_profile_id, None);
        assert_eq!(
            effects,
            vec![Effect::Command(Command::ListProfileAgents {
                profile_id: "fast".into(),
            })]
        );

        app.sessions.session_id = None;
        app.profiles.active_profile_id = Some("removed".into());
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Profiles {
                profiles: vec![profile("fast"), profile("deep")],
                active_profile_id: Some("fast".into()),
            }),
        );

        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert!(app.models.agents.is_empty());
        assert_eq!(app.models.agents_profile_id, None);
        assert_eq!(
            effects,
            vec![Effect::Command(Command::ListProfileAgents {
                profile_id: "fast".into(),
            })]
        );

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Profiles {
                profiles: Vec::new(),
                active_profile_id: None,
            }),
        );
        assert!(effects.is_empty());
        assert!(app.profiles.active_profile_id.is_none());
        assert!(app.profiles.profiles.is_empty());
        assert_eq!(app.auth.filter, "preserved auth");
        assert_eq!(app.mesh.mesh_invite_name, "preserved mesh");
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "preserved chat"
        ));
    }

    #[test]
    fn profile_agents_acp_route_ignores_stale_and_reapplies_root_preferences() {
        let mut app = App::new();
        app.sessions.session_id = Some("parent".into());
        app.profiles
            .bind_session_profile("parent".into(), "quorum".into());
        app.models.agents_profile_id = Some("quorum".into());
        app.models.agents = vec![agent("sentinel"), agent("old")];
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
        app.auth.filter = "preserved auth".into();
        app.chat.context_limit = 4_096;

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ProfileAgents {
                profile_id: "stale".into(),
                agents: vec![agent("stale")],
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.models.agents[0].id, "sentinel");
        assert_eq!(app.models.agents_profile_id.as_deref(), Some("quorum"));

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ProfileAgents {
                profile_id: "quorum".into(),
                agents: vec![agent("primary"), agent("coder")],
            }),
        );
        assert_eq!(app.models.agents[1].id, "coder");
        assert_eq!(app.models.agents_profile_id.as_deref(), Some("quorum"));
        assert_eq!(
            effects,
            vec![Effect::Command(Command::SetDelegateModel {
                session_id: "parent".into(),
                agent_id: "coder".into(),
                model_id: Some("openai/gpt-5".into()),
                node_id: Some("node-1".into()),
            })]
        );

        app.delegates.parent_session_id = Some("root".into());
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ProfileAgents {
                profile_id: "quorum".into(),
                agents: vec![agent("primary"), agent("reviewer")],
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.models.agents[1].id, "reviewer");
        assert_eq!(app.auth.filter, "preserved auth");
        assert_eq!(app.chat.context_limit, 4_096);
    }

    #[test]
    fn model_acp_route_preserves_ui_state_and_logs_exact_inventory_diagnostics() {
        let mut app = App::new();
        app.models.current_provider = Some("old-provider".into());
        app.models.current_model = Some("old-model".into());
        app.models.current_model_node_id = Some("old-node".into());
        app.models.model_filter = "keep".into();
        app.models.model_cursor = 7;
        app.models.model_popup_agent_tab = 2;
        app.models.reasoning_effort = Some("high".into());
        app.auth.filter = "preserved auth".into();
        app.sessions.session_id = Some("preserved session".into());
        app.chat
            .messages
            .push(ChatEntry::Error("preserved chat".into()));

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Models {
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
                meta: Some(crate::acp_state::AcpModelsMetaInfo {
                    remote_node_count: 2,
                    remote_timeout_count: 1,
                }),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(app.models.models.len(), 2);
        assert_eq!(app.models.current_provider.as_deref(), Some("old-provider"));
        assert_eq!(app.models.current_model.as_deref(), Some("old-model"));
        assert_eq!(
            app.models.current_model_node_id.as_deref(),
            Some("old-node")
        );
        assert_eq!(app.models.model_filter, "keep");
        assert_eq!(app.models.model_cursor, 7);
        assert_eq!(app.models.model_popup_agent_tab, 2);
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("high"));
        let diagnostic = app.diagnostics.logs.last().expect("model diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Info);
        assert_eq!(diagnostic.target, "models");
        assert_eq!(
            diagnostic.message,
            "models: 2 total, 1 remote (inventory nodes=2, timeouts=1)"
        );
        assert_eq!(app.auth.filter, "preserved auth");
        assert_eq!(
            app.sessions.session_id.as_deref(),
            Some("preserved session")
        );
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "preserved chat"
        ));

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::DelegateModelSet {
                session_id: "preserved session".into(),
                agent_id: "coder".into(),
                model: None,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.diagnostics.status,
            "delegate model reset for coder in preserved session"
        );
        let diagnostic = app.diagnostics.logs.last().expect("delegate diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Info);
        assert_eq!(diagnostic.target, "model");
        assert_eq!(
            diagnostic.message,
            "delegate model reset for coder in preserved session"
        );
    }

    #[test]
    fn provider_and_effort_acp_routes_apply_release_sensitive_coordination() {
        let mut app = App::new();
        app.chat.context_limit = 4_096;
        app.auth.filter = "preserved auth".into();
        app.mesh.mesh_invite_name = "preserved mesh".into();
        app.sessions.session_id = Some("preserved session".into());

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ProviderChanged {
                provider: "remote-provider".into(),
                model: "model-1".into(),
                context_limit: Some(128_000),
                provider_node_id: Some("node-1".into()),
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.models.current_provider.as_deref(),
            Some("remote-provider")
        );
        assert_eq!(app.models.current_model.as_deref(), Some("model-1"));
        assert_eq!(app.models.current_model_node_id.as_deref(), Some("node-1"));
        assert_eq!(app.chat.context_limit, 128_000);

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::ProviderChanged {
                provider: "local-provider".into(),
                model: "model-2".into(),
                context_limit: None,
                provider_node_id: None,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.models.current_provider.as_deref(),
            Some("local-provider")
        );
        assert_eq!(app.models.current_model.as_deref(), Some("model-2"));
        assert_eq!(app.models.current_model_node_id, None);
        assert_eq!(app.chat.context_limit, 128_000);

        assert!(
            update(
                &mut app,
                AppEvent::Acp(AcpAppEvent::ReasoningEffort {
                    reasoning_effort: Some("med".into()),
                }),
            )
            .is_empty()
        );
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("medium"));
        assert!(
            update(
                &mut app,
                AppEvent::Acp(AcpAppEvent::ReasoningEffort {
                    reasoning_effort: Some("invalid".into()),
                }),
            )
            .is_empty()
        );
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("medium"));
        assert_eq!(app.auth.filter, "preserved auth");
        assert_eq!(app.mesh.mesh_invite_name, "preserved mesh");
        assert_eq!(
            app.sessions.session_id.as_deref(),
            Some("preserved session")
        );
    }

    #[test]
    fn session_catalog_acp_route_preserves_order_and_remote_location() {
        let mut app = App::new();
        app.sessions.session_discovery_in_progress = true;
        app.sessions.remember_remote_session_location(
            "remote-1",
            "node-1",
            Some("/remote/repo".into()),
        );
        app.auth.filter = "preserved auth".into();

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionList {
                request: SessionListRequest::Discovery,
                page: SessionListPage {
                    groups: vec![
                        SessionGroup {
                            cwd: Some("/first".into()),
                            sessions: vec![SessionSummary {
                                session_id: "remote-1".into(),
                                cwd: Some("/catalog/repo".into()),
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                        SessionGroup {
                            cwd: Some("/second".into()),
                            sessions: vec![SessionSummary {
                                session_id: "local-1".into(),
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                    ],
                    next_cursor: Some("discovery-2".into()),
                    total_count: Some(2),
                },
            }),
        );

        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::list_sessions_workspace("/first".into())),
                Effect::Command(Command::list_sessions_workspace("/second".into())),
                Effect::Command(Command::list_sessions_discovery(Some("discovery-2".into()))),
            ]
        );
        let location = &app.sessions.remote_session_locations["remote-1"];
        assert_eq!(location.node_id, "node-1");
        assert_eq!(location.cwd.as_deref(), Some("/remote/repo"));
        assert_eq!(app.auth.filter, "preserved auth");
    }

    #[test]
    fn session_catalog_failure_acp_route_applies_ordered_release_coordination() {
        let mut app = App::new();
        app.sessions
            .pending_session_group_loads
            .insert(Some("/repo".into()));
        app.chat
            .messages
            .push(ChatEntry::Error("preserved first".into()));
        app.models.reasoning_effort = Some("high".into());

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionListFailed {
                request: SessionListRequest::WorkspaceContinuation {
                    cwd: "/repo".into(),
                },
                message: "catalog unavailable".into(),
            }),
        );

        assert!(effects.is_empty());
        assert!(app.sessions.pending_session_group_loads.is_empty());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(first), ChatEntry::Error(second)]
                if first == "preserved first" && second == "catalog unavailable"
        ));
        assert_eq!(app.diagnostics.status, "error: catalog unavailable");
        let diagnostic = app.diagnostics.logs.last().expect("failure diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Error);
        assert_eq!(diagnostic.target, "acp");
        assert_eq!(diagnostic.message, "error: catalog unavailable");
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn session_created_acp_route_applies_resets_diagnostic_and_effect_order() {
        let mut app = App::new();
        app.delegates.parent_session_id = Some("old-parent".into());
        app.delegates.pending_parent_session_id = Some("staged-parent".into());
        app.delegates
            .pending_delegate_child_states
            .insert("old-child".into(), DelegateChildState::OtherProgress);
        app.chat
            .messages
            .push(ChatEntry::Error("stale chat".into()));
        app.chat.streaming_content = "stale stream".into();
        app.composer.input = "preserved draft".into();
        app.composer.input_cursor = 4;
        app.composer.file_index = vec![FileIndexEntryLite {
            path: "src/main.rs".into(),
            is_dir: false,
        }];
        app.render.streaming_cache.store(7, Vec::new());
        app.render.streaming_thinking_cache.store(8, Vec::new());
        app.render.card_cache.processed_messages = 2;
        app.sessions.mode_before_review = Some("plan".into());
        app.models.agents_profile_id = Some("code".into());
        app.models.agents = vec![agent("primary"), agent("coder")];
        app.models.delegate_model_preferences.insert(
            "code".into(),
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

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionCreated {
                agent_id: "agent-1".into(),
                session_id: "session-1".into(),
                profile_id: Some("code".into()),
            }),
        );

        assert_eq!(app.sessions.session_id.as_deref(), Some("session-1"));
        assert_eq!(app.sessions.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(app.sessions.mode_before_review, None);
        assert_eq!(app.navigation.screen, Screen::Chat);
        assert_eq!(app.delegates.parent_session_id, None);
        assert_eq!(app.delegates.pending_parent_session_id, None);
        assert!(app.delegates.pending_delegate_child_states.is_empty());
        assert_eq!(app.profiles.session_profile_id("session-1"), Some("code"));
        assert!(app.chat.messages.is_empty());
        assert!(app.chat.streaming_content.is_empty());
        assert_eq!(app.composer.input, "preserved draft");
        assert_eq!(app.composer.input_cursor, 4);
        assert!(app.composer.file_index.is_empty());
        assert!(app.render.streaming_cache.get(7).is_none());
        assert!(app.render.streaming_thinking_cache.get(8).is_none());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "session created");
        let diagnostic = app.diagnostics.logs.last().expect("creation diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Info);
        assert_eq!(diagnostic.target, "session");
        assert_eq!(diagnostic.message, "session created");
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::SubscribeSession {
                    session_id: "session-1".into(),
                    agent_id: Some("agent-1".into()),
                }),
                Effect::Command(Command::SetDelegateModel {
                    session_id: "session-1".into(),
                    agent_id: "coder".into(),
                    model_id: Some("openai/gpt-5".into()),
                    node_id: Some("node-1".into()),
                }),
            ]
        );
    }

    #[test]
    fn session_loaded_acp_route_preserves_local_and_removes_remote_profile_binding() {
        let mut app = App::new();
        app.sessions.agent_mode = "plan".into();
        app.profiles
            .bind_session_profile("local".into(), "code".into());
        app.models.agents_profile_id = Some("code".into());
        app.models.agents = vec![agent("primary"), agent("coder")];
        app.models.delegate_model_preferences.insert(
            "code".into(),
            [(
                "coder".into(),
                DelegateModelPreference {
                    model_id: "openai/gpt-5".into(),
                    provider: "openai".into(),
                    model: "gpt-5".into(),
                    node_id: None,
                },
            )]
            .into_iter()
            .collect(),
        );
        app.delegates.parent_session_id = Some("old-parent".into());
        app.delegates
            .pending_delegate_child_states
            .insert("old-child".into(), DelegateChildState::OtherProgress);
        app.chat.activity = ActivityState::Streaming;
        app.chat
            .messages
            .push(ChatEntry::Error("stale chat".into()));
        app.composer.file_index = vec![FileIndexEntryLite {
            path: "src/lib.rs".into(),
            is_dir: false,
        }];
        app.sessions.mode_before_review = Some("build".into());

        let local_effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionLoaded {
                agent_id: "agent-local".into(),
                session_id: "local".into(),
                profile_id: None,
            }),
        );

        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert_eq!(app.delegates.parent_session_id, None);
        assert!(app.delegates.pending_delegate_child_states.is_empty());
        assert_eq!(app.navigation.screen, Screen::Chat);
        assert_eq!(app.profiles.session_profile_id("local"), Some("code"));
        assert_eq!(app.sessions.session_id.as_deref(), Some("local"));
        assert_eq!(app.sessions.agent_id.as_deref(), Some("agent-local"));
        assert_eq!(app.sessions.mode_before_review, None);
        assert!(app.chat.messages.is_empty());
        assert!(app.composer.file_index.is_empty());
        assert_eq!(app.diagnostics.status, "ready");
        let diagnostic = app.diagnostics.logs.last().expect("load diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Debug);
        assert_eq!(diagnostic.target, "activity");
        assert_eq!(diagnostic.message, "ready");
        assert_eq!(
            local_effects,
            vec![
                Effect::Command(Command::SetAgentMode {
                    mode: "plan".into(),
                }),
                Effect::Command(Command::SetDelegateModel {
                    session_id: "local".into(),
                    agent_id: "coder".into(),
                    model_id: Some("openai/gpt-5".into()),
                    node_id: None,
                }),
            ]
        );

        let mut app = App::new();
        app.sessions.remember_remote_session_location(
            "remote-child",
            "node-1",
            Some("/remote".into()),
        );
        app.profiles
            .bind_session_profile("remote-child".into(), "stale".into());
        app.sessions.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-child".into(),
                parent_session_id: Some("catalog-parent".into()),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.delegates.pending_parent_session_id = Some("staged-parent".into());
        app.delegates
            .pending_delegate_child_states
            .insert("keep-child".into(), DelegateChildState::OtherProgress);
        app.chat.activity = ActivityState::Thinking;

        let remote_effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::SessionLoaded {
                agent_id: "agent-remote".into(),
                session_id: "remote-child".into(),
                profile_id: None,
            }),
        );

        assert_eq!(app.chat.activity, ActivityState::Idle);
        assert_eq!(
            app.delegates.parent_session_id.as_deref(),
            Some("staged-parent")
        );
        assert_eq!(app.delegates.pending_parent_session_id, None);
        assert!(
            app.delegates
                .pending_delegate_child_states
                .contains_key("keep-child")
        );
        assert_eq!(app.navigation.screen, Screen::Delegate);
        assert!(app.profiles.session_profile_id("remote-child").is_none());
        assert_eq!(
            remote_effects,
            vec![Effect::Command(Command::SetAgentMode {
                mode: "build".into(),
            })]
        );
    }

    #[test]
    fn session_agent_mode_acp_route_is_owner_bounded_and_release_safe() {
        let mut app = App::new();
        app.sessions.agent_mode = "review".into();
        app.sessions.mode_before_review = Some("plan".into());
        app.auth.filter = "preserved auth".into();
        app.mesh.mesh_invite_name = "preserved mesh".into();
        app.chat
            .messages
            .push(ChatEntry::Error("preserved chat".into()));

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::AgentMode {
                mode: "build".into(),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(app.sessions.agent_mode, "build");
        assert_eq!(app.sessions.mode_before_review, None);
        assert_eq!(app.auth.filter, "preserved auth");
        assert_eq!(app.mesh.mesh_invite_name, "preserved mesh");
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "preserved chat"
        ));
    }

    #[test]
    fn initialized_acp_route_preserves_placeholder_and_discards_profile_effects() {
        let mut app = App::new();
        app.profiles.profiles = vec![profile("deep")];
        app.profiles.active_profile_id = Some("deep".into());
        app.models.model_popup_agent_tab = 3;
        app.models.model_filter = "keep".into();
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: false,
            message: "stale".into(),
        });

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Initialized {
                agent_id: "agent-1".into(),
                agent_name: "Agent".into(),
                profiles: Vec::new(),
                active_profile_id: None,
                agent_mode: Some("plan".into()),
                reasoning_effort: Some(Some("high".into())),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(app.profiles.profiles[0].id, "deep");
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("deep"));
        assert_eq!(app.sessions.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(app.sessions.agent_mode, "plan");
        assert_eq!(app.models.agents[0].id, "agent-1");
        assert_eq!(app.models.agents_profile_id, None);
        assert_eq!(app.models.model_popup_agent_tab, 3);
        assert_eq!(app.models.model_filter, "keep");
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("high"));
        assert!(app.auth.ui_notice.is_none());
        assert_eq!(app.diagnostics.status, "connected");
        let diagnostic = app.diagnostics.logs.last().expect("connection diagnostic");
        assert_eq!(diagnostic.level, LogLevel::Info);
        assert_eq!(diagnostic.target, "connection");
        assert_eq!(diagnostic.message, "connected");

        let mut app = App::new();
        app.models.model_popup_agent_tab = 3;
        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::Initialized {
                agent_id: "agent-2".into(),
                agent_name: "Agent Two".into(),
                profiles: vec![profile("fast")],
                active_profile_id: Some("fast".into()),
                agent_mode: None,
                reasoning_effort: None,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert_eq!(app.models.agents[0].id, "agent-2");
        assert_eq!(app.models.agents_profile_id, None);
        assert_eq!(app.models.model_popup_agent_tab, 0);
    }

    #[test]
    fn mesh_acp_route_mutates_only_mesh_and_explicit_coordination_targets() {
        let mut app = App::new();
        app.auth.filter = "preserved auth".into();
        app.chat
            .messages
            .push(ChatEntry::Error("preserved chat".into()));
        app.models.reasoning_effort = Some("high".into());

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::MeshNodes(MeshNodesInfo {
                nodes: vec![RemoteNodeInfo {
                    id: "node-1".into(),
                    label: "Remote".into(),
                    ..Default::default()
                }],
            })),
        );

        assert_eq!(app.mesh.selected_mesh_node_id(), Some("node-1"));
        assert_eq!(
            effects,
            vec![Effect::Command(Command::ListRemoteSessions {
                node_id: "node-1".into(),
                offset: 0,
                limit: 50,
            })]
        );
        assert_eq!(app.auth.filter, "preserved auth");
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "preserved chat"
        ));
        assert_eq!(app.models.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn auth_acp_route_mutates_only_auth_and_explicit_diagnostic_effect_targets() {
        let mut app = App::new();
        app.mesh.mesh_nodes = vec![RemoteNodeInfo {
            id: "node-1".into(),
            label: "Remote".into(),
            ..Default::default()
        }];
        app.chat
            .messages
            .push(ChatEntry::Error("preserved chat".into()));
        app.sessions.session_id = Some("session-1".into());
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "flow-1".into(),
            provider: "openai".into(),
            authorization_url: "https://example.com/authorize".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });

        let effects = update(
            &mut app,
            AppEvent::Acp(AcpAppEvent::OAuthResult(OAuthResult {
                provider: "openai".into(),
                status: OAuthResultStatus::Success,
                message: "connected".into(),
            })),
        );

        assert!(app.auth.oauth_flow.is_none());
        assert_eq!(effects, vec![Effect::Command(Command::ListAuthProviders)]);
        assert_eq!(app.mesh.selected_mesh_node_id(), Some("node-1"));
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Error(message)] if message == "preserved chat"
        ));
        assert_eq!(app.sessions.session_id.as_deref(), Some("session-1"));
        assert_eq!(
            app.diagnostics
                .logs
                .last()
                .map(|entry| entry.message.as_str()),
            Some("connected")
        );
    }

    #[test]
    fn connection_initialization_and_local_reconnect_effects_are_ordered() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());
        app.sessions.agent_id = Some("agent-1".into());
        app.connection.launch_cwd = Some("/repo".into());

        let effects = update(&mut app, AppEvent::Connection(ConnectionEvent::Connected));

        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::Init),
                Effect::Command(Command::list_sessions_browse()),
                Effect::Command(Command::ListAllModels { refresh: false }),
                Effect::Command(Command::LoadSession {
                    session_id: "session-1".into(),
                    cwd: Some("/repo".into()),
                }),
                Effect::Command(Command::SubscribeSession {
                    session_id: "session-1".into(),
                    agent_id: Some("agent-1".into()),
                }),
            ]
        );
    }

    #[test]
    fn connection_reconnect_attaches_remembered_remote_session() {
        let mut app = App::new();
        app.sessions.session_id = Some("remote-1".into());
        app.sessions.remember_remote_session_location(
            "remote-1",
            "node-1",
            Some("/remote/repo".into()),
        );

        let effects = update(&mut app, AppEvent::Connection(ConnectionEvent::Connected));

        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::Init),
                Effect::Command(Command::list_sessions_browse()),
                Effect::Command(Command::ListAllModels { refresh: false }),
                Effect::Command(Command::AttachRemoteSession {
                    node_id: "node-1".into(),
                    session_id: "remote-1".into(),
                }),
            ]
        );
    }

    #[test]
    fn connection_reconnect_skips_remote_session_without_node_id() {
        let mut app = App::new();
        app.sessions.session_id = Some("remote-1".into());
        app.sessions.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-1".into(),
                node: Some("remote".into()),
                ..Default::default()
            }],
            ..Default::default()
        }];

        let effects = update(&mut app, AppEvent::Connection(ConnectionEvent::Connected));

        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::Init),
                Effect::Command(Command::list_sessions_browse()),
                Effect::Command(Command::ListAllModels { refresh: false }),
            ]
        );
        assert_eq!(
            app.diagnostics.status,
            "remote session is missing node id; reconnect attach skipped"
        );
    }

    #[test]
    fn connected_to_disconnected_overrides_status_for_reconnect() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;

        let effects = update(
            &mut app,
            AppEvent::Connection(ConnectionEvent::Disconnected {
                reason: "closed".into(),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(app.diagnostics.status, "connection lost - reconnecting...");
    }

    #[test]
    fn supervisor_events_update_state_and_preserve_connected_status() {
        let mut app = App::new();
        let effects = update(&mut app, AppEvent::Supervisor(ServerEvent::Starting));
        assert!(effects.is_empty());
        assert_eq!(app.connection.server_state, ServerState::Starting);
        assert_eq!(app.diagnostics.status, "starting qmtcode ACP agent...");

        update(&mut app, AppEvent::Supervisor(ServerEvent::Started));
        assert_eq!(app.connection.server_state, ServerState::Running);
        assert_eq!(app.diagnostics.status, "qmtcode ACP agent started");

        update(&mut app, AppEvent::Supervisor(ServerEvent::BinaryNotFound));
        assert_eq!(app.connection.server_state, ServerState::BinaryNotFound);
        assert_eq!(
            app.diagnostics.status,
            "qmtcode not found; install it or set acp.binary_path in ~/.qmt/qmtui.toml"
        );

        app.connection.conn = ConnState::Connected;
        app.set_status(LogLevel::Debug, "test", "retained");
        update(&mut app, AppEvent::Supervisor(ServerEvent::Starting));
        assert_eq!(app.connection.server_state, ServerState::Starting);
        assert_eq!(app.diagnostics.status, "retained");

        update(&mut app, AppEvent::Supervisor(ServerEvent::Started));
        update(&mut app, AppEvent::Supervisor(ServerEvent::BinaryNotFound));
        assert_eq!(app.connection.server_state, ServerState::BinaryNotFound);
        assert_eq!(app.diagnostics.status, "retained");

        update(
            &mut app,
            AppEvent::Supervisor(ServerEvent::StartFailed {
                error: "invalid command".into(),
            }),
        );
        assert_eq!(app.diagnostics.status, "ACP start failed: invalid command");

        update(
            &mut app,
            AppEvent::Supervisor(ServerEvent::Stopped {
                reason: "process exited".into(),
            }),
        );
        assert_eq!(app.diagnostics.status, "ACP agent stopped (process exited)");
    }

    #[test]
    fn tick_updates_render_clock_and_clears_expired_cancel_confirmation() {
        let mut app = App::new();
        app.chat.pending_cancel_confirm_until = Some(Instant::now() - Duration::from_millis(1));
        app.render.tick = u64::MAX;

        app.diagnostics.started_at = Instant::now() - Duration::from_millis(240);
        let effects = update(&mut app, AppEvent::Tick);

        assert!(effects.is_empty());
        assert!(app.render.tick >= 3);
        assert_ne!(app.render.tick, u64::MAX);
        assert!(app.chat.pending_cancel_confirm_until.is_none());
    }

    #[test]
    fn empty_ctrl_c_produces_quit_without_mutating_control_state() {
        let mut app = App::new();
        let effects = update(
            &mut app,
            AppEvent::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        );

        assert_eq!(effects, vec![Effect::Quit]);
        assert!(!app.should_quit);
    }

    #[test]
    fn profile_selection_produces_persistence_after_local_mutation() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = crate::navigation_state::Popup::ProfileSelect;
        app.profiles.profiles = vec![crate::domain::profile::ProfileInfo {
            id: "fast".into(),
            name: "Fast".into(),
            ..Default::default()
        }];

        let effects = update(
            &mut app,
            AppEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        );

        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert!(matches!(
            effects.as_slice(),
            [
                Effect::Command(Command::ListProfileAgents { profile_id }),
                Effect::PersistConfig,
            ] if profile_id == "fast"
        ));
    }

    #[test]
    fn auth_clipboard_failure_uses_current_oauth_url() {
        let mut app = App::new();
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "flow-1".into(),
            provider: "codex".into(),
            authorization_url: "https://auth.example.com/authorize".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: None,
            success: true,
            message: "old notice".into(),
        });

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ClipboardFinished {
                target: ClipboardTarget::Auth {
                    provider: "codex".into(),
                },
                success: false,
            }),
        );

        assert!(effects.is_empty());
        assert!(app.auth.ui_notice.is_none());
        assert_eq!(
            app.auth.clipboard_fallback.as_deref(),
            Some("https://auth.example.com/authorize")
        );
    }

    #[test]
    fn mesh_clipboard_success_sets_exact_status() {
        let mut app = App::new();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ClipboardFinished {
                target: ClipboardTarget::MeshInvite,
                success: true,
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(app.diagnostics.status, "invite URL copied");
        assert!(app.mesh.mesh_clipboard_fallback.is_none());
    }

    #[test]
    fn clipboard_success_and_failure_results_preserve_exact_feedback() {
        let mut app = App::new();
        app.auth.clipboard_fallback = Some("old fallback".into());

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ClipboardFinished {
                target: ClipboardTarget::Auth {
                    provider: "codex".into(),
                },
                success: true,
            }),
        );
        assert!(effects.is_empty());
        assert!(matches!(
            app.auth.ui_notice.as_ref(),
            Some(notice)
                if notice.provider.as_deref() == Some("codex")
                    && notice.success
                    && notice.message == "Copied to clipboard"
        ));
        assert!(app.auth.clipboard_fallback.is_none());

        app.apply_mesh_invite_created(MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: None,
            expires_at: 1,
            max_uses: 1,
            mesh_name: None,
        });
        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ClipboardFinished {
                target: ClipboardTarget::MeshInvite,
                success: false,
            }),
        );
        assert!(effects.is_empty());
        assert_eq!(
            app.mesh.mesh_clipboard_fallback.as_deref(),
            Some("qmt://mesh/join/token")
        );
    }

    #[test]
    fn editor_results_apply_state_then_request_redraw() {
        let mut app = App::new();
        app.composer.input = "draft".into();
        app.render.card_cache.processed_messages = 2;

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ExternalEditorFinished {
                outcome: ExternalEditorOutcome::Completed("revised".into()),
            }),
        );

        assert_eq!(app.composer.input, "revised");
        assert_eq!(app.composer.input_cursor, "revised".len());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "loaded prompt from external editor");
        assert_eq!(effects, vec![Effect::Terminal(TerminalAction::Redraw)]);
    }

    #[test]
    fn editor_cancel_and_failure_preserve_input_and_request_redraw() {
        let mut app = App::new();
        app.composer.input = "draft".into();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ExternalEditorFinished {
                outcome: ExternalEditorOutcome::Cancelled,
            }),
        );
        assert_eq!(app.composer.input, "draft");
        assert_eq!(app.diagnostics.status, "external editor cancelled");
        assert_eq!(effects, vec![Effect::Terminal(TerminalAction::Redraw)]);

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ExternalEditorFinished {
                outcome: ExternalEditorOutcome::Failed("editor exited".into()),
            }),
        );
        assert_eq!(app.composer.input, "draft");
        assert_eq!(
            app.diagnostics.status,
            "external editor failed: editor exited"
        );
        assert_eq!(effects, vec![Effect::Terminal(TerminalAction::Redraw)]);
    }

    #[test]
    fn elicitation_response_ack_resolves_the_matching_active_card() {
        let mut app = App::new();
        add_elicitation(&mut app, "elic-1", true);
        app.connection.conn = ConnState::Connected;
        app.render.streaming_cache.store(3, Vec::new());
        app.render.streaming_thinking_cache.store(4, Vec::new());
        app.render.card_cache.processed_messages = 5;
        app.diagnostics.status = "question pending".into();
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ElicitationResponseSent {
                elicitation_id: "elic-1".into(),
                outcome: "accepted".into(),
            }),
        );

        assert!(effects.is_empty());
        assert!(app.chat.elicitation.is_none());
        assert!(app.chat.elicitation_ui.is_none());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: Some(outcome), .. }]
                if elicitation_id == "elic-1" && outcome == "accepted"
        ));
        assert!(app.render.streaming_cache.get(3).is_some());
        assert!(app.render.streaming_thinking_cache.get(4).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "ready");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Debug);
        assert_eq!(diagnostics[0].target, "activity");
        assert_eq!(diagnostics[0].message, "ready");
    }

    #[test]
    fn elicitation_command_failure_leaves_active_card_pending() {
        let mut app = App::new();
        add_elicitation(&mut app, "elic-1", true);
        app.render.streaming_cache.store(3, Vec::new());
        app.render.streaming_thinking_cache.store(4, Vec::new());
        app.render.card_cache.processed_messages = 5;
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::CommandFailed {
                command: Command::ElicitationResponse {
                    elicitation_id: "elic-1".into(),
                    action: "decline".into(),
                    content: None,
                },
                message: "channel closed".into(),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(
            app.chat
                .elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-1")
        );
        assert!(app.chat.elicitation_ui.is_some());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: None, .. }]
                if elicitation_id == "elic-1"
        ));
        assert!(app.render.streaming_cache.get(3).is_some());
        assert!(app.render.streaming_thinking_cache.get(4).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 5);
        assert_eq!(app.diagnostics.status, "channel closed");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Error);
        assert_eq!(diagnostics[0].target, "command");
        assert_eq!(diagnostics[0].message, "channel closed");
    }

    #[test]
    fn stale_elicitation_ack_backfills_old_card_without_clearing_newer_active() {
        let mut app = App::new();
        add_elicitation(&mut app, "elic-old", false);
        add_elicitation(&mut app, "elic-new", true);
        app.render.streaming_cache.store(3, Vec::new());
        app.render.streaming_thinking_cache.store(4, Vec::new());
        app.render.card_cache.processed_messages = 2;
        app.diagnostics.status = "newer question pending".into();
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ElicitationResponseSent {
                elicitation_id: "elic-old".into(),
                outcome: "accepted".into(),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(
            app.chat
                .elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-new")
        );
        assert!(app.chat.elicitation_ui.is_some());
        assert!(app.render.streaming_cache.get(3).is_some());
        assert!(app.render.streaming_thinking_cache.get(4).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "newer question pending");
        assert_eq!(app.diagnostics.logs.len(), log_count_before);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [
                ChatEntry::Elicitation { elicitation_id: old_id, outcome: Some(outcome), .. },
                ChatEntry::Elicitation { elicitation_id: new_id, outcome: None, .. },
            ] if old_id == "elic-old" && outcome == "accepted" && new_id == "elic-new"
        ));
    }

    #[test]
    fn unknown_elicitation_ack_is_a_complete_noop() {
        let mut app = App::new();
        add_elicitation(&mut app, "elic-active", true);
        app.render.streaming_cache.store(3, Vec::new());
        app.render.streaming_thinking_cache.store(4, Vec::new());
        app.render.card_cache.processed_messages = 5;
        app.diagnostics.status = "question pending".into();
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ElicitationResponseSent {
                elicitation_id: "missing".into(),
                outcome: "accepted".into(),
            }),
        );

        assert!(effects.is_empty());
        assert_eq!(
            app.chat
                .elicitation
                .as_ref()
                .map(|state| state.elicitation_id.as_str()),
            Some("elic-active")
        );
        assert!(app.chat.elicitation_ui.is_some());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: None, .. }]
                if elicitation_id == "elic-active"
        ));
        assert!(app.render.streaming_cache.get(3).is_some());
        assert!(app.render.streaming_thinking_cache.get(4).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 5);
        assert_eq!(app.diagnostics.status, "question pending");
        assert_eq!(app.diagnostics.logs.len(), log_count_before);
    }

    #[test]
    fn duplicate_elicitation_ack_rewrites_and_invalidates_again_without_status_refresh() {
        let mut app = App::new();
        add_elicitation(&mut app, "elic-1", false);
        app.render.card_cache.processed_messages = 5;
        app.diagnostics.status = "preserved status".into();
        let log_count_before = app.diagnostics.logs.len();

        let first_effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ElicitationResponseSent {
                elicitation_id: "elic-1".into(),
                outcome: "accepted".into(),
            }),
        );
        assert!(first_effects.is_empty());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        app.render.card_cache.processed_messages = 6;

        let duplicate_effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::ElicitationResponseSent {
                elicitation_id: "elic-1".into(),
                outcome: "declined".into(),
            }),
        );
        assert!(duplicate_effects.is_empty());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { elicitation_id, outcome: Some(outcome), .. }]
                if elicitation_id == "elic-1" && outcome == "declined"
        ));
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "preserved status");
        assert_eq!(app.diagnostics.logs.len(), log_count_before);
    }

    #[test]
    fn command_failure_rolls_back_only_matching_optimistic_prompt() {
        let mut app = App::new();
        let failed_id = app.chat.push_pending_prompt("failed".into());
        let retained_id = app.chat.push_pending_prompt("retained".into());
        app.chat.recent_prompt_text = Some("preserved prompt".into());
        app.chat.session_stats.open_llm_request_instant = Some(Instant::now());
        let open_timing = app.chat.session_stats.open_llm_request_instant;
        app.render.streaming_cache.store(3, Vec::new());
        app.render.streaming_thinking_cache.store(4, Vec::new());
        app.render.card_cache.processed_messages = 2;
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::CommandFailed {
                command: Command::Prompt {
                    prompt: vec![],
                    local_id: failed_id.clone(),
                },
                message: "channel closed".into(),
            }),
        );

        assert!(effects.is_empty());
        assert!(app.chat.messages.iter().all(|entry| {
            !matches!(entry, ChatEntry::User { message_id: Some(id), .. } if id == &failed_id)
        }));
        assert!(app.chat.messages.iter().any(|entry| {
            matches!(entry, ChatEntry::User { message_id: Some(id), .. } if id == &retained_id)
        }));
        assert!(
            app.chat
                .messages
                .iter()
                .all(|entry| !matches!(entry, ChatEntry::Error(_)))
        );
        assert_eq!(
            app.chat.recent_prompt_text.as_deref(),
            Some("preserved prompt")
        );
        assert_eq!(app.chat.session_stats.open_llm_request_instant, open_timing);
        assert_eq!(app.chat.session_stats.active_llm_duration, Duration::ZERO);
        assert!(app.render.streaming_cache.get(3).is_some());
        assert!(app.render.streaming_thinking_cache.get(4).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "channel closed");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Error);
        assert_eq!(diagnostics[0].target, "command");
        assert_eq!(diagnostics[0].message, "channel closed");
    }

    #[test]
    fn unmatched_prompt_command_failure_is_diagnostic_only_and_release_safe() {
        let mut app = App::new();
        let retained_id = app.chat.push_pending_prompt("retained".into());
        app.chat.recent_prompt_text = Some("preserved prompt".into());
        app.chat.session_stats.open_llm_request_instant = Some(Instant::now());
        let open_timing = app.chat.session_stats.open_llm_request_instant;
        app.render.streaming_cache.store(5, Vec::new());
        app.render.streaming_thinking_cache.store(6, Vec::new());
        app.render.card_cache.processed_messages = 7;
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::CommandFailed {
                command: Command::Prompt {
                    prompt: vec![],
                    local_id: "missing".into(),
                },
                message: "unknown prompt failed".into(),
            }),
        );

        assert!(effects.is_empty());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "retained" && message_id == &retained_id
        ));
        assert_eq!(
            app.chat.recent_prompt_text.as_deref(),
            Some("preserved prompt")
        );
        assert_eq!(app.chat.session_stats.open_llm_request_instant, open_timing);
        assert_eq!(app.chat.session_stats.active_llm_duration, Duration::ZERO);
        assert!(app.render.streaming_cache.get(5).is_some());
        assert!(app.render.streaming_thinking_cache.get(6).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 7);
        assert_eq!(app.diagnostics.status, "unknown prompt failed");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Error);
        assert_eq!(diagnostics[0].target, "command");
        assert_eq!(diagnostics[0].message, "unknown prompt failed");
    }

    #[test]
    fn non_prompt_command_failure_is_diagnostic_only_and_release_safe() {
        let mut app = App::new();
        let retained_id = app.chat.push_pending_prompt("retained".into());
        app.chat.recent_prompt_text = Some("preserved prompt".into());
        app.chat.session_stats.open_llm_request_instant = Some(Instant::now());
        let open_timing = app.chat.session_stats.open_llm_request_instant;
        app.render.streaming_cache.store(8, Vec::new());
        app.render.streaming_thinking_cache.store(9, Vec::new());
        app.render.card_cache.processed_messages = 10;
        let log_count_before = app.diagnostics.logs.len();

        let effects = update(
            &mut app,
            AppEvent::Runtime(RuntimeEvent::CommandFailed {
                command: Command::CancelSession,
                message: "cancel channel closed".into(),
            }),
        );

        assert!(effects.is_empty());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "retained" && message_id == &retained_id
        ));
        assert_eq!(
            app.chat.recent_prompt_text.as_deref(),
            Some("preserved prompt")
        );
        assert_eq!(app.chat.session_stats.open_llm_request_instant, open_timing);
        assert_eq!(app.chat.session_stats.active_llm_duration, Duration::ZERO);
        assert!(app.render.streaming_cache.get(8).is_some());
        assert!(app.render.streaming_thinking_cache.get(9).is_some());
        assert_eq!(app.render.card_cache.processed_messages, 10);
        assert_eq!(app.diagnostics.status, "cancel channel closed");
        let diagnostics = &app.diagnostics.logs[log_count_before..];
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, LogLevel::Error);
        assert_eq!(diagnostics[0].target, "command");
        assert_eq!(diagnostics[0].message, "cancel channel closed");
    }
}
