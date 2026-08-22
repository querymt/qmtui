use crossterm::event::{KeyEvent, MouseEvent};

use crate::{
    acp_state::AcpAppEvent,
    app::{App, ConnectionEvent},
    auth_state::AuthUiNotice,
    command::Command,
    connection_state::ConnState,
    diagnostics::LogLevel,
    domain::chat::ChatEntry,
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
    CommandThen {
        command: Command,
        on_sent: RuntimeEvent,
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
        RuntimeEvent::ClipboardFinished { target, success } => {
            match target {
                ClipboardTarget::Auth { provider } => {
                    if success {
                        app.auth.ui_notice = Some(AuthUiNotice {
                            provider: Some(provider),
                            success: true,
                            message: "Copied to clipboard".into(),
                        });
                        app.auth.clipboard_fallback = None;
                    } else {
                        app.auth.ui_notice = None;
                        app.auth.clipboard_fallback = app
                            .auth
                            .oauth_flow
                            .as_ref()
                            .map(|flow| flow.authorization_url.clone());
                    }
                }
                ClipboardTarget::MeshInvite => {
                    if success {
                        app.set_status(LogLevel::Info, "mesh", "invite URL copied");
                    } else if let Some(url) = app.mesh.invite_url().map(str::to_string) {
                        app.mesh.set_clipboard_fallback(url);
                    }
                }
            }
            Vec::new()
        }
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
        } => {
            let is_active = app
                .chat
                .elicitation
                .as_ref()
                .is_some_and(|state| state.elicitation_id == elicitation_id);
            if is_active {
                app.resolve_elicitation(&elicitation_id, &outcome);
            } else if let Some(ChatEntry::Elicitation {
                outcome: card_outcome,
                ..
            }) = app.chat.messages.iter_mut().find(|entry| {
                matches!(
                    entry,
                    ChatEntry::Elicitation {
                        elicitation_id: existing_id,
                        ..
                    } if existing_id == &elicitation_id
                )
            }) {
                *card_outcome = Some(outcome);
                app.render.invalidate_card_cache();
            }
            Vec::new()
        }
        RuntimeEvent::CommandFailed { command, message } => {
            if let Command::Prompt { local_id, .. } = command {
                app.chat.rollback_pending_prompt(&local_id);
                app.render.invalidate_card_cache();
            }
            app.set_status(LogLevel::Error, "command", message);
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEventKind};

    use super::*;
    use crate::{
        chat_state::ElicitationUiState,
        connection_state::ServerState,
        domain::{
            auth::{OAuthFlow, OAuthFlowKind},
            chat::ChatEntry,
            elicitation::ElicitationState,
            mesh::MeshInviteCreatedInfo,
            profile::ProfileInfo,
            session::{SessionGroup, SessionSummary},
        },
        navigation_state::Screen,
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
    }

    #[test]
    fn elicitation_command_failure_leaves_active_card_pending() {
        let mut app = App::new();
        add_elicitation(&mut app, "elic-1", true);

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
        assert_eq!(app.diagnostics.status, "channel closed");
    }

    #[test]
    fn stale_elicitation_ack_backfills_old_card_without_clearing_newer_active() {
        let mut app = App::new();
        add_elicitation(&mut app, "elic-old", false);
        add_elicitation(&mut app, "elic-new", true);
        app.render.card_cache.processed_messages = 2;

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
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [
                ChatEntry::Elicitation { elicitation_id: old_id, outcome: Some(outcome), .. },
                ChatEntry::Elicitation { elicitation_id: new_id, outcome: None, .. },
            ] if old_id == "elic-old" && outcome == "accepted" && new_id == "elic-new"
        ));
    }

    #[test]
    fn command_failure_rolls_back_only_matching_optimistic_prompt() {
        let mut app = App::new();
        let failed_id = app.chat.push_pending_prompt("failed".into());
        let retained_id = app.chat.push_pending_prompt("retained".into());
        app.render.card_cache.processed_messages = 2;

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
        assert_eq!(app.render.card_cache.processed_messages, 0);
        assert_eq!(app.diagnostics.status, "channel closed");
    }
}
