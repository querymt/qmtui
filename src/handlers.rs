use crate::app::App;
use crate::application::{ClipboardTarget, Effect};
use crate::diagnostics::LogLevel;
use crate::domain::activity::{ActivityState, SessionOp};
use crate::features::auth::input::{AuthInputResult, handle_key as handle_auth_input_key};
use crate::features::chat::input::{
    CancelIntent, ChatCommandIntent, ChatInputContext, ChatInputResult, ChatViewportIntent,
    CompletionResult, ComposerKeyResult, ElicitationResponseEffect, PromptSubmission,
    build_prompt_submission, chat_command_intent, handle_completion_key, handle_composer_key,
    handle_coordination_key, handle_elicitation_key,
};
use crate::features::delegates::input::{
    DelegateInputContext, DelegateInputResult, DelegateViewportIntent,
    handle_popup_key as handle_delegate_input_key,
    handle_view_key as handle_delegate_view_input_key,
};
use crate::features::diagnostics::input::{
    DiagnosticsInputResult, handle_key as handle_diagnostics_input_key,
};
use crate::features::mesh::input::{
    MeshInputResult, handle_invite_key as handle_mesh_invite_input_key,
    handle_invite_qr_key as handle_mesh_invite_qr_input_key,
    handle_popup_key as handle_mesh_input_key,
};
use crate::features::models::input::{
    ModelInputResult, ModelTab, handle_key as handle_model_input_key,
};
use crate::features::navigation::input::{
    HelpInputResult, PaletteInputResult, ThemeInputResult, handle_help_key, handle_palette_key,
    handle_theme_key,
};
use crate::features::profiles::input::{
    ProfileInputResult, handle_key as handle_profile_input_key,
};
use crate::features::sessions::input::{
    NewSessionInputResult, SessionPopupInputResult, SessionsInputResult,
    handle_new_session_key as handle_new_session_input_key,
    handle_session_popup_key as handle_session_popup_input_key,
    handle_sessions_key as handle_sessions_input_key, switch_session_popup_tab,
    toggle_popup_session_children, toggle_start_session_children,
};
use crate::navigation_state::{CommandPaletteAction, Popup, Screen};
use crate::render_state::RenderChange;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use crate::command::{Command, PromptBlock};
use crate::connection_state::ConnState;
use crate::theme;

pub(crate) fn can_send_server_commands(app: &mut App) -> bool {
    if app.connection.conn == ConnState::Connected {
        true
    } else {
        app.diagnostics.set_status(
            LogLevel::Warn,
            "connection",
            "not connected - waiting to reconnect",
        );
        false
    }
}

fn send_load_session_commands(
    session_id: String,
    cwd: Option<String>,
    agent_id: Option<String>,
) -> Vec<Effect> {
    Command::load_session_commands(session_id, cwd, agent_id)
        .into_iter()
        .map(Effect::Command)
        .collect()
}

fn open_model_popup(app: &mut App) -> Vec<Effect> {
    if app.navigation.screen != Screen::Chat {
        app.diagnostics.set_status(
            LogLevel::Warn,
            "model",
            "model select is only available in chat",
        );
        return Vec::new();
    }
    if !can_send_server_commands(app) {
        return Vec::new();
    }
    app.navigation.popup = Popup::ModelSelect;
    app.models.reset_for_open();
    vec![Effect::Command(Command::ListAllModels { refresh: true })]
}

fn open_session_popup(app: &mut App) -> Vec<Effect> {
    if !can_send_server_commands(app) {
        return Vec::new();
    }
    app.navigation.popup = Popup::SessionSelect;
    app.sessions.reset_browser_for_open();
    app.begin_session_discovery()
        .map(|command| vec![Effect::Command(command)])
        .unwrap_or_default()
}

fn open_log_popup(app: &mut App) {
    app.navigation.popup = Popup::Log;
    app.diagnostics.log_cursor = app.diagnostics.filtered_logs().len().saturating_sub(1);
    app.diagnostics.log_filter.clear();
}

fn execute_command_palette_action(app: &mut App, action: CommandPaletteAction) -> Vec<Effect> {
    let mut effects = Vec::new();
    match action {
        CommandPaletteAction::OpenMesh | CommandPaletteAction::AttachRemoteSession => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_mesh_popup();
            effects.push(Effect::Command(Command::ListRemoteNodes));
        }
        CommandPaletteAction::CreateRemoteSession => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_mesh_popup();
            effects.push(Effect::Command(Command::ListRemoteNodes));
        }
        CommandPaletteAction::CreateMeshInvite => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_mesh_invite_form();
        }
        CommandPaletteAction::ModelSelect => effects.extend(open_model_popup(app)),
        CommandPaletteAction::SessionSelect => effects.extend(open_session_popup(app)),
        CommandPaletteAction::DelegateSessions => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_delegate_popup();
        }
        CommandPaletteAction::NewSession => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_new_session_popup();
        }
        CommandPaletteAction::ThemeSelect => {
            app.navigation
                .open_theme_selector(theme::Theme::current_index());
        }
        CommandPaletteAction::Help => app.navigation.open_help(),
        CommandPaletteAction::Log => open_log_popup(app),
        CommandPaletteAction::ProviderAuth => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_auth_popup();
            effects.push(Effect::Command(Command::ListAuthProviders));
        }
        CommandPaletteAction::ForkTurnSelect => app.open_fork_turn_popup(),
        CommandPaletteAction::ProfileSelect => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_profile_popup();
            effects.push(Effect::Command(Command::ListProfiles));
        }
    }
    effects
}

pub(crate) fn handle_mesh_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match handle_mesh_input_key(&mut app.mesh, key) {
        MeshInputResult::Close => {
            app.navigation.popup = Popup::None;
            Vec::new()
        }
        MeshInputResult::MoveNode {
            refresh_node_id: Some(node_id),
        }
        | MeshInputResult::FocusSessions {
            node_id: Some(node_id),
        } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::ListRemoteSessions {
                node_id,
                offset: 0,
                limit: 50,
            })]
        }
        MeshInputResult::AttachRemoteSession {
            node_id,
            session_id,
        } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::AttachRemoteSession {
                node_id,
                session_id,
            })]
        }
        MeshInputResult::CreateRemoteSession { node_id } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::CreateRemoteSession {
                node_id,
                cwd: None,
            })]
        }
        MeshInputResult::RefreshNodes => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::ListRemoteNodes)]
        }
        MeshInputResult::NotHandled
        | MeshInputResult::ToggleFocus
        | MeshInputResult::MoveNode {
            refresh_node_id: None,
        }
        | MeshInputResult::MoveRemoteSession
        | MeshInputResult::FocusSessions { node_id: None }
        | MeshInputResult::InviteEdited
        | MeshInputResult::SubmitInvite
        | MeshInputResult::BackToInviteForm
        | MeshInputResult::ShowInviteUrl
        | MeshInputResult::CopyInviteUrl { .. } => Vec::new(),
    }
}

pub(crate) fn handle_mesh_invite_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if app.mesh.consume_clipboard_fallback() {
        return Vec::new();
    }

    match handle_mesh_invite_input_key(&mut app.mesh, key) {
        MeshInputResult::Close => app.navigation.popup = Popup::None,
        MeshInputResult::SubmitInvite => {
            if let Some(command) = app.mesh_invite_form_command() {
                if !can_send_server_commands(app) {
                    return Vec::new();
                }
                app.diagnostics
                    .set_status(LogLevel::Info, "mesh", "creating invite...");
                return vec![Effect::Command(command)];
            }
        }
        MeshInputResult::NotHandled
        | MeshInputResult::ToggleFocus
        | MeshInputResult::MoveNode { .. }
        | MeshInputResult::MoveRemoteSession
        | MeshInputResult::FocusSessions { .. }
        | MeshInputResult::AttachRemoteSession { .. }
        | MeshInputResult::CreateRemoteSession { .. }
        | MeshInputResult::RefreshNodes
        | MeshInputResult::InviteEdited
        | MeshInputResult::BackToInviteForm
        | MeshInputResult::ShowInviteUrl
        | MeshInputResult::CopyInviteUrl { .. } => {}
    }
    Vec::new()
}

pub(crate) fn handle_mesh_invite_qr_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if app.mesh.consume_clipboard_fallback() {
        return Vec::new();
    }

    match handle_mesh_invite_qr_input_key(&mut app.mesh, key) {
        MeshInputResult::BackToInviteForm => app.navigation.popup = Popup::MeshInvite,
        MeshInputResult::ShowInviteUrl => {
            app.mesh.show_invite_url_fallback();
        }
        MeshInputResult::CopyInviteUrl { url } => {
            return vec![Effect::CopyToClipboard {
                target: ClipboardTarget::MeshInvite,
                text: url,
            }];
        }
        MeshInputResult::NotHandled
        | MeshInputResult::Close
        | MeshInputResult::ToggleFocus
        | MeshInputResult::MoveNode { .. }
        | MeshInputResult::MoveRemoteSession
        | MeshInputResult::FocusSessions { .. }
        | MeshInputResult::AttachRemoteSession { .. }
        | MeshInputResult::CreateRemoteSession { .. }
        | MeshInputResult::RefreshNodes
        | MeshInputResult::InviteEdited
        | MeshInputResult::SubmitInvite => {}
    }
    Vec::new()
}

pub(crate) fn handle_command_palette_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match handle_palette_key(&mut app.navigation, key) {
        PaletteInputResult::Close => app.navigation.popup = Popup::None,
        PaletteInputResult::Execute(action) => return execute_command_palette_action(app, action),
        PaletteInputResult::NotHandled
        | PaletteInputResult::Moved
        | PaletteInputResult::Filtered => {}
    }
    Vec::new()
}

pub(crate) fn handle_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if key.code != KeyCode::Esc && app.chat.pending_cancel_confirm_until.is_some() {
        app.chat.clear_cancel_confirm();
        app.refresh_transient_status();
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if !app.composer.input.is_empty() {
            app.composer.clear_input();
            app.render.reset_composer_input_geometry();
            return Vec::new();
        }
        return vec![Effect::Quit];
    }

    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('p') | KeyCode::Char('P'))
    {
        app.navigation.chord = false;
        app.navigation.open_command_palette();
        return Vec::new();
    }

    if app.navigation.chord {
        app.navigation.chord = false;
        app.diagnostics
            .set_status(LogLevel::Debug, "input", "ready");
        if app.navigation.screen == Screen::Chat {
            if let Some(ChatCommandIntent::OpenExternalEditor { initial_text }) =
                chat_command_intent(&app.composer, key)
            {
                return vec![Effect::OpenExternalEditor { initial_text }];
            }
        } else if key.code == KeyCode::Char('e') {
            app.diagnostics.set_status(
                LogLevel::Warn,
                "editor",
                "external editor is only available in chat",
            );
            return Vec::new();
        }
        return handle_chord(app, key);
    }

    if app.chat.elicitation.is_some() {
        return handle_elicitation_key(&mut app.chat, &mut app.render, key)
            .into_iter()
            .map(
                |ElicitationResponseEffect {
                     elicitation_id,
                     action,
                     content,
                     outcome,
                 }| Effect::ElicitationResponse {
                    elicitation_id,
                    action,
                    content,
                    outcome,
                },
            )
            .collect();
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
        if !can_send_server_commands(app) {
            return Vec::new();
        }
        return match app.cycle_reasoning_effort() {
            Some(command) => {
                app.diagnostics.set_status(
                    LogLevel::Info,
                    "model",
                    format!("thinking: {}", app.models.reasoning_effort_label()),
                );
                vec![Effect::Command(command)]
            }
            None => {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "model",
                    format!(
                        "unknown reasoning effort {:?}; cannot cycle",
                        app.models.reasoning_effort
                    ),
                );
                Vec::new()
            }
        };
    }

    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('l') | KeyCode::Char('L'))
    {
        open_log_popup(app);
        return Vec::new();
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('x') {
        app.navigation.chord = true;
        app.diagnostics
            .set_status(LogLevel::Debug, "input", "C-x ...");
        return Vec::new();
    }

    match app.navigation.popup {
        Popup::CommandPalette => return handle_command_palette_key(app, key),
        Popup::Mesh => return handle_mesh_popup_key(app, key),
        Popup::MeshInvite => return handle_mesh_invite_popup_key(app, key),
        Popup::MeshInviteQr => return handle_mesh_invite_qr_popup_key(app, key),
        Popup::ModelSelect => return handle_model_popup_key(app, key),
        Popup::SessionSelect => return handle_session_popup_key(app, key),
        Popup::NewSession => return handle_new_session_popup_key(app, key),
        Popup::ThemeSelect => return handle_theme_popup_key(app, key),
        Popup::Help => {
            match handle_help_key(&mut app.navigation, key) {
                HelpInputResult::Close => app.navigation.popup = Popup::None,
                HelpInputResult::NotHandled | HelpInputResult::Scroll { .. } => {}
            }
            return Vec::new();
        }
        Popup::Log => {
            handle_log_popup_key(app, key);
            return Vec::new();
        }
        Popup::ProviderAuth => return handle_auth_popup_key(app, key),
        Popup::ForkTurnSelect => return handle_fork_turn_popup_key(app, key),
        Popup::ProfileSelect => return handle_profile_popup_key(app, key),
        Popup::None => {}
    }

    if key.code == KeyCode::Tab {
        if !can_send_server_commands(app) {
            return Vec::new();
        }
        return switch_mode(app, &app.sessions.next_mode());
    }

    match app.navigation.screen {
        Screen::Sessions => handle_sessions_key(app, key),
        Screen::Chat => handle_chat_key(app, key),
        Screen::Delegate => handle_delegate_view_key(app, key),
    }
}

pub(crate) fn handle_mouse(app: &mut App, mouse: MouseEvent) -> Vec<Effect> {
    match (mouse.kind, &app.navigation.screen, &app.navigation.popup) {
        (MouseEventKind::ScrollUp, Screen::Chat | Screen::Delegate, Popup::None) => {
            app.render.scroll_chat_up(3);
        }
        (MouseEventKind::ScrollDown, Screen::Chat | Screen::Delegate, Popup::None) => {
            app.render.scroll_chat_down(3);
        }
        _ => {}
    }
    Vec::new()
}

/// Handle second key of a ctrl+x chord. Works in any screen.
pub(crate) fn handle_chord(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let mut effects = Vec::new();
    match key.code {
        KeyCode::Char('m') => effects.extend(open_model_popup(app)),
        KeyCode::Char('n') => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_new_session_popup();
        }
        KeyCode::Char('q') => effects.push(Effect::Quit),
        KeyCode::Char('e') => {
            app.diagnostics.set_status(
                LogLevel::Warn,
                "editor",
                "external editor unavailable here",
            );
        }
        KeyCode::Char('t') => app
            .navigation
            .open_theme_selector(theme::Theme::current_index()),
        KeyCode::Char('l') => effects.extend(open_session_popup(app)),
        KeyCode::Char('a') => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_auth_popup();
            effects.push(Effect::Command(Command::ListAuthProviders));
        }
        KeyCode::Char('p') => {
            if !can_send_server_commands(app) {
                return effects;
            }
            app.open_profile_popup();
            effects.push(Effect::Command(Command::ListProfiles));
        }
        KeyCode::Char('j') => {
            if !matches!(app.navigation.screen, Screen::Chat | Screen::Delegate) {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "session",
                    "parent jump only available in chat",
                );
                return effects;
            }
            if !can_send_server_commands(app) {
                return effects;
            }
            if let Some(parent_sid) = app.delegates.parent_session_id.clone() {
                effects.extend(send_load_session_commands(
                    parent_sid,
                    app.current_session_cwd(),
                    app.sessions.agent_id.clone(),
                ));
            } else {
                app.diagnostics
                    .set_status(LogLevel::Info, "session", "no parent session");
            }
        }
        KeyCode::Char('?') => app.navigation.open_help(),
        KeyCode::Char('f') => {
            if app.navigation.screen != Screen::Chat {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "fork",
                    "fork selector is only available in chat",
                );
                return effects;
            }
            app.open_fork_turn_popup();
        }
        KeyCode::Char('u') => {
            if !can_send_server_commands(app) {
                return effects;
            }
            if app.chat.is_turn_active() {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot undo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "undo already pending");
            } else if let Some(turn) = app.chat.current_undo_target().cloned() {
                if app.composer.input.trim().is_empty() && !turn.text.is_empty() {
                    app.composer.replace_input(turn.text.clone());
                    app.render.reset_composer_input_geometry();
                }
                app.chat.push_pending_undo(&turn);
                app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
                app.diagnostics
                    .set_status(LogLevel::Info, "session", "undoing...");
                effects.push(Effect::Command(Command::Undo {
                    message_id: turn.message_id,
                }));
            } else {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "nothing to undo");
            }
        }
        KeyCode::Char('r') => {
            if !can_send_server_commands(app) {
                return effects;
            }
            if app.chat.is_turn_active() {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot redo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "undo already pending");
            } else if app.chat.can_redo() {
                app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
                app.diagnostics
                    .set_status(LogLevel::Info, "session", "redoing...");
                effects.push(Effect::Command(Command::Redo));
            } else {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "nothing to redo");
            }
        }
        _ => app
            .diagnostics
            .set_status(LogLevel::Debug, "input", "unknown chord"),
    }
    effects
}

fn session_action_effects(app: &mut App, action: SessionKeyAction) -> Vec<Effect> {
    match action {
        SessionKeyAction::LoadSession {
            session_id,
            agent_id,
            cwd,
        } => send_load_session_commands(
            session_id,
            cwd,
            agent_id.or_else(|| app.sessions.agent_id.clone()),
        ),
        SessionKeyAction::AttachRemoteSession {
            node_id,
            session_id,
        } => vec![Effect::Command(Command::AttachRemoteSession {
            node_id,
            session_id,
        })],
        SessionKeyAction::DeleteSession { session_id } => {
            vec![Effect::Command(Command::DeleteSession { session_id })]
        }
        SessionKeyAction::DismissRemoteSession { session_id } => {
            vec![Effect::Command(Command::DismissRemoteSession {
                session_id,
            })]
        }
        SessionKeyAction::NewSession => {
            app.open_new_session_popup();
            Vec::new()
        }
        SessionKeyAction::LoadMoreSessions {
            group_idx,
            parent_path,
        } => {
            let request = if parent_path.is_empty() {
                app.session_group_page_request(group_idx)
            } else {
                app.session_child_page_request(group_idx, &parent_path)
            };
            request
                .map(|command| vec![Effect::Command(command)])
                .unwrap_or_default()
        }
        SessionKeyAction::None => Vec::new(),
    }
}

fn resolve_session_selection(
    app: &mut App,
    session_id: String,
    cwd: Option<String>,
) -> SessionKeyAction {
    if let Some(node_id) = app
        .sessions
        .session_remote_node_id(&session_id)
        .map(str::to_string)
    {
        app.sessions
            .remember_remote_session_node(&session_id, &node_id);
        return SessionKeyAction::AttachRemoteSession {
            node_id,
            session_id,
        };
    }
    if app.sessions.is_remote_session_id(&session_id) {
        app.diagnostics.set_status(
            LogLevel::Warn,
            "session",
            "remote session is missing node id; refresh sessions and try again",
        );
        return SessionKeyAction::None;
    }
    SessionKeyAction::LoadSession {
        session_id,
        agent_id: None,
        cwd,
    }
}

fn apply_sessions_input_result(app: &mut App, result: SessionsInputResult) -> SessionKeyAction {
    match result {
        SessionsInputResult::Moved {
            keep_visible_from_above,
        } => {
            if keep_visible_from_above {
                app.render
                    .keep_start_page_cursor_visible_from_above(app.sessions.session_cursor);
            }
            SessionKeyAction::None
        }
        SessionsInputResult::Filtered => {
            app.render.reset_start_page_scroll();
            SessionKeyAction::None
        }
        SessionsInputResult::OpenSession { session_id, cwd } => {
            resolve_session_selection(app, session_id, cwd)
        }
        SessionsInputResult::DeleteSession { session_id } => {
            SessionKeyAction::DeleteSession { session_id }
        }
        SessionsInputResult::DismissRemoteSession { session_id } => {
            SessionKeyAction::DismissRemoteSession { session_id }
        }
        SessionsInputResult::OpenSessionPopup => {
            app.navigation.popup = Popup::SessionSelect;
            SessionKeyAction::None
        }
        SessionsInputResult::OpenNewSession => SessionKeyAction::NewSession,
        SessionsInputResult::LoadMore {
            group_idx,
            parent_path,
        } => SessionKeyAction::LoadMoreSessions {
            group_idx,
            parent_path,
        },
        SessionsInputResult::NotHandled | SessionsInputResult::ToggledGroup => {
            SessionKeyAction::None
        }
    }
}

fn apply_session_popup_input_result(
    app: &mut App,
    result: SessionPopupInputResult,
) -> SessionKeyAction {
    match result {
        SessionPopupInputResult::OpenSession { session_id, cwd } => {
            app.navigation.popup = Popup::None;
            resolve_session_selection(app, session_id, cwd)
        }
        SessionPopupInputResult::DeleteSession { session_id } => {
            SessionKeyAction::DeleteSession { session_id }
        }
        SessionPopupInputResult::DismissRemoteSession { session_id } => {
            SessionKeyAction::DismissRemoteSession { session_id }
        }
        SessionPopupInputResult::LoadMore {
            group_idx,
            parent_path,
        } => SessionKeyAction::LoadMoreSessions {
            group_idx,
            parent_path,
        },
        SessionPopupInputResult::ClosePopup => {
            app.navigation.popup = Popup::None;
            SessionKeyAction::None
        }
        SessionPopupInputResult::NotHandled
        | SessionPopupInputResult::Moved
        | SessionPopupInputResult::Filtered
        | SessionPopupInputResult::ToggledGroup => SessionKeyAction::None,
    }
}

pub(crate) fn handle_sessions_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
        return vec![Effect::Quit];
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        let result = toggle_start_session_children(&mut app.sessions);
        let action = apply_sessions_input_result(app, result);
        return session_action_effects(app, action);
    }

    let key = if key.modifiers.contains(KeyModifiers::CONTROL) {
        KeyCode::Null
    } else {
        key.code
    };
    let result = handle_sessions_input_key(&mut app.sessions, key);
    let action = apply_sessions_input_result(app, result);
    session_action_effects(app, action)
}

pub(crate) fn handle_session_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
        switch_session_popup_tab(&mut app.sessions);
        return Vec::new();
    }

    if app.sessions.session_popup_tab == 1 {
        return handle_delegate_popup_key(app, key);
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('n') {
        if can_send_server_commands(app) {
            app.open_new_session_popup();
        }
        return Vec::new();
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        let result = toggle_popup_session_children(&mut app.sessions);
        let action = apply_session_popup_input_result(app, result);
        return session_action_effects(app, action);
    }

    let key = if key.modifiers.contains(KeyModifiers::CONTROL) {
        KeyCode::Null
    } else {
        key.code
    };
    let page_step = app.render.session_popup_page_step();
    let result = handle_session_popup_input_key(&mut app.sessions, key, page_step);
    let action = apply_session_popup_input_result(app, result);
    session_action_effects(app, action)
}

/// Adapt session-popup input into the existing root-owned session action.
#[cfg(test)]
pub(crate) fn apply_popup_session_key(
    app: &mut App,
    key: crossterm::event::KeyCode,
) -> SessionKeyAction {
    let page_step = app.render.session_popup_page_step();
    let result = handle_session_popup_input_key(&mut app.sessions, key, page_step);
    apply_session_popup_input_result(app, result)
}

#[cfg(test)]
pub(crate) fn apply_session_fork_toggle_key(app: &mut App, popup_items: bool) -> SessionKeyAction {
    if popup_items {
        let result = toggle_popup_session_children(&mut app.sessions);
        apply_session_popup_input_result(app, result)
    } else {
        let result = toggle_start_session_children(&mut app.sessions);
        apply_sessions_input_result(app, result)
    }
}

// ── Delegate view key handler (read-only child session) ──────────────────────

fn apply_delegate_viewport_intent(app: &mut App, intent: DelegateViewportIntent) {
    match intent {
        DelegateViewportIntent::UpOne => app.render.scroll_chat_up(1),
        DelegateViewportIntent::DownOne => app.render.scroll_chat_down(1),
        DelegateViewportIntent::UpTen => app.render.scroll_chat_up(10),
        DelegateViewportIntent::DownTen => app.render.scroll_chat_down(10),
        DelegateViewportIntent::Top => app.render.scroll_chat_to_top(),
        DelegateViewportIntent::Bottom => app.render.scroll_chat_to_bottom(),
    }
}

fn apply_delegate_popup_input_result(
    app: &mut App,
    result: DelegateInputResult,
) -> SessionKeyAction {
    match result {
        DelegateInputResult::ClosePopup => {
            app.navigation.popup = Popup::None;
            SessionKeyAction::None
        }
        DelegateInputResult::LoadChild {
            session_id,
            target_agent_id,
            current_session_id,
            parent_session_id,
            cwd,
        } => {
            debug_assert_eq!(current_session_id, app.sessions.session_id);
            debug_assert_eq!(parent_session_id, app.delegates.parent_session_id);
            app.navigation.popup = Popup::None;
            SessionKeyAction::LoadSession {
                session_id,
                agent_id: target_agent_id,
                cwd,
            }
        }
        DelegateInputResult::PendingChild => {
            app.diagnostics.set_status(
                LogLevel::Warn,
                "delegates",
                "delegation still pending — no session to load",
            );
            SessionKeyAction::None
        }
        DelegateInputResult::NotHandled
        | DelegateInputResult::Moved
        | DelegateInputResult::Filtered
        | DelegateInputResult::Viewport(_)
        | DelegateInputResult::NavigateParent { .. } => SessionKeyAction::None,
    }
}

fn handle_delegate_view_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match handle_delegate_view_input_key(&app.delegates, key.code) {
        DelegateInputResult::Viewport(intent) => {
            apply_delegate_viewport_intent(app, intent);
            Vec::new()
        }
        DelegateInputResult::NavigateParent { session_id } => send_load_session_commands(
            session_id,
            app.current_session_cwd(),
            app.sessions.agent_id.clone(),
        ),
        _ => Vec::new(),
    }
}

// ── Delegate popup key handler ────────────────────────────────────────────────

pub(crate) fn handle_delegate_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let key = if key.modifiers.contains(KeyModifiers::CONTROL) {
        KeyCode::Null
    } else {
        key.code
    };
    let action = apply_delegate_popup_key(app, key);
    session_action_effects(app, action)
}

/// Adapt delegate-popup input into the existing root-owned session action.
pub(crate) fn apply_delegate_popup_key(
    app: &mut App,
    key: crossterm::event::KeyCode,
) -> SessionKeyAction {
    let context = DelegateInputContext {
        page_step: app.render.delegate_popup_page_step(),
        current_session_id: app.sessions.session_id.clone(),
        cwd: app.current_session_cwd(),
    };
    let result = handle_delegate_input_key(&mut app.delegates, key, context);
    apply_delegate_popup_input_result(app, result)
}

fn begin_fork_session(app: &mut App, message_id: String) -> Vec<Effect> {
    if app.chat.pending_fork_message_id.is_some() {
        app.diagnostics
            .set_status(LogLevel::Warn, "fork", "fork already pending");
        return Vec::new();
    }
    if app.chat.is_turn_active() {
        app.diagnostics
            .set_status(LogLevel::Warn, "fork", "cannot fork while agent is active");
        return Vec::new();
    }
    if app.chat.has_pending_session_op() {
        app.diagnostics
            .set_status(LogLevel::Warn, "fork", "session operation already pending");
        return Vec::new();
    }
    if message_id.is_empty() {
        app.diagnostics
            .set_status(LogLevel::Warn, "fork", "selected turn has no message id");
        return Vec::new();
    }

    app.chat.pending_fork_message_id = Some(message_id.clone());
    app.diagnostics
        .set_status(LogLevel::Info, "fork", "forking session...");
    vec![Effect::Command(Command::ForkSession { message_id })]
}

pub(crate) fn handle_fork_turn_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Up => app.chat.move_fork_cursor(-1),
        KeyCode::Down => app.chat.move_fork_cursor(1),
        KeyCode::Backspace => app.chat.fork_filter_backspace(),
        KeyCode::Enter => {
            if let Some(turn) = app.chat.selected_fork_turn() {
                return begin_fork_session(app, turn.message_id);
            }
            app.diagnostics
                .set_status(LogLevel::Warn, "fork", "no forkable turns");
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.chat.fork_filter_insert(c);
        }
        _ => {}
    }
    Vec::new()
}

pub(crate) fn handle_log_popup_key(app: &mut App, key: KeyEvent) {
    match handle_diagnostics_input_key(&mut app.diagnostics, key) {
        DiagnosticsInputResult::Close => app.navigation.popup = Popup::None,
        DiagnosticsInputResult::NotHandled
        | DiagnosticsInputResult::Moved
        | DiagnosticsInputResult::Paged
        | DiagnosticsInputResult::ToStart
        | DiagnosticsInputResult::ToEnd
        | DiagnosticsInputResult::Filtered
        | DiagnosticsInputResult::CycledLevel => {}
    }
}

pub(crate) fn handle_profile_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if key.code == KeyCode::Enter && !can_send_server_commands(app) {
        return Vec::new();
    }

    match handle_profile_input_key(&mut app.profiles, key) {
        ProfileInputResult::Close => app.navigation.popup = Popup::None,
        ProfileInputResult::SelectProfile { profile_id } => {
            app.profiles.active_profile_id = Some(profile_id.clone());
            let mut effects = Vec::new();
            if app.current_session_profile_id().is_none() {
                app.models.clear_profile_agents();
                effects.push(Effect::Command(Command::ListProfileAgents {
                    profile_id: profile_id.clone(),
                }));
            }
            app.navigation.popup = Popup::None;
            app.diagnostics.set_status(
                LogLevel::Info,
                "profile",
                format!("new sessions will use {profile_id}"),
            );
            effects.push(Effect::PersistConfig);
            return effects;
        }
        ProfileInputResult::NoMatchingProfile => {
            app.diagnostics
                .set_status(LogLevel::Warn, "profile", "no matching profile");
        }
        ProfileInputResult::NotHandled
        | ProfileInputResult::Moved
        | ProfileInputResult::Filtered => {}
    }
    Vec::new()
}

pub(crate) fn handle_new_session_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match handle_new_session_input_key(&mut app.sessions, key) {
        NewSessionInputResult::Edited => app.refresh_new_session_completion(),
        NewSessionInputResult::AcceptCompletion => {
            app.accept_selected_new_session_completion();
        }
        NewSessionInputResult::Cancel => app.navigation.popup = Popup::None,
        NewSessionInputResult::Submit { raw_path } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            let cwd = app.normalize_new_session_path(&raw_path);
            app.navigation.popup = Popup::None;
            return vec![Effect::Command(Command::NewSession {
                cwd,
                profile_id: app.profiles.active_profile_id.clone(),
            })];
        }
        NewSessionInputResult::NotHandled | NewSessionInputResult::MovedCompletion => {}
    }
    Vec::new()
}

pub(crate) fn handle_theme_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match handle_theme_key(&mut app.navigation, theme::Theme::available_themes(), key) {
        ThemeInputResult::Close => app.navigation.popup = Popup::None,
        ThemeInputResult::Apply { index } => {
            theme::Theme::set_by_index(index);
            theme::Theme::begin_frame();
            app.render.apply_change(RenderChange::ThemeChanged);
            app.navigation.popup = Popup::None;
            return vec![Effect::PersistConfig];
        }
        ThemeInputResult::NotHandled | ThemeInputResult::Moved | ThemeInputResult::Filtered => {}
    }
    Vec::new()
}

fn apply_chat_input_result(app: &mut App, result: ChatInputResult) -> Option<Vec<Effect>> {
    match result {
        ChatInputResult::NotHandled => None,
        ChatInputResult::CompletionDismissed => {
            app.chat.clear_cancel_confirm();
            Some(Vec::new())
        }
        ChatInputResult::Cancel(CancelIntent::ArmConfirmation) => {
            app.arm_cancel_confirm();
            Some(Vec::new())
        }
        ChatInputResult::Cancel(CancelIntent::ConfirmCancellation) => {
            app.chat.clear_cancel_confirm();
            app.diagnostics
                .set_status(LogLevel::Warn, "activity", "stopping...");
            Some(vec![Effect::Command(Command::CancelSession)])
        }
        ChatInputResult::Cancel(CancelIntent::ClearConfirmation) => {
            app.chat.clear_cancel_confirm();
            Some(Vec::new())
        }
        ChatInputResult::Viewport(ChatViewportIntent::ScrollUp { rows }) => {
            app.render.scroll_chat_up(rows);
            Some(Vec::new())
        }
        ChatInputResult::Viewport(ChatViewportIntent::ScrollDown { rows }) => {
            app.render.scroll_chat_down(rows);
            Some(Vec::new())
        }
        ChatInputResult::Viewport(ChatViewportIntent::ToBottom) => {
            app.render.scroll_chat_to_bottom();
            Some(Vec::new())
        }
    }
}

pub(crate) fn handle_chat_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let input_blocked = app.chat.input_blocked_by_activity();
    let context = ChatInputContext {
        editable: !input_blocked,
        has_cancellable_activity: app.chat.has_cancellable_activity(),
        cancel_confirmation_active: app.chat.cancel_confirm_active(),
    };
    let result = handle_coordination_key(&mut app.composer, &mut app.render, key, context);
    if let Some(effects) = apply_chat_input_result(app, result) {
        return effects;
    }

    let mut effects = Vec::new();
    match key.code {
        KeyCode::Enter => {
            let completion = handle_completion_key(&mut app.composer, key, !input_blocked);
            if completion == CompletionResult::SlashAccepted
                || (!app.composer.input.is_empty()
                    && app.composer.input.trim_start().starts_with('/'))
            {
                let (result, slash_effects) = try_execute_slash_command(app);
                effects.extend(slash_effects);
                match result {
                    SlashResult::OpenEditor => {
                        effects.push(Effect::OpenExternalEditor {
                            initial_text: app.composer.input.clone(),
                        });
                        return effects;
                    }
                    SlashResult::Handled => return effects,
                    SlashResult::NotACommand => {}
                }
            }
            if let CompletionResult::MentionAccepted { request_file_index } = completion {
                if request_file_index {
                    effects.push(Effect::Command(Command::GetFileIndex));
                }
                return effects;
            }
            if !app.composer.input.is_empty() {
                if input_blocked || !can_send_server_commands(app) {
                    return effects;
                }
                let Some(PromptSubmission { text, links }) = build_prompt_submission(&app.composer)
                else {
                    return effects;
                };
                let _ = app.take_input();
                if text.is_empty() {
                    return effects;
                }
                let mut prompt = vec![PromptBlock::Text { text: text.clone() }];
                for path in links {
                    prompt.push(PromptBlock::ResourceLink {
                        name: path.clone(),
                        uri: path,
                    });
                }
                let local_id = app.push_pending_prompt(text);
                effects.push(Effect::Command(Command::Prompt { prompt, local_id }));
            }
        }
        KeyCode::Tab if !input_blocked => {
            if let CompletionResult::MentionAccepted { request_file_index } =
                handle_completion_key(&mut app.composer, key, true)
                && request_file_index
            {
                effects.push(Effect::Command(Command::GetFileIndex));
            }
        }
        KeyCode::Char(_)
            if handle_composer_key(&mut app.composer, &mut app.render, key, !input_blocked)
                == ComposerKeyResult::Edited
                && app.composer.prepare_file_index_request() =>
        {
            effects.push(Effect::Command(Command::GetFileIndex));
        }
        KeyCode::Up | KeyCode::Down
            if !input_blocked
                && handle_completion_key(&mut app.composer, key, true)
                    == CompletionResult::NoOp =>
        {
            handle_composer_key(&mut app.composer, &mut app.render, key, true);
        }
        KeyCode::Backspace | KeyCode::Delete | KeyCode::Left | KeyCode::Right | KeyCode::Home => {
            handle_composer_key(&mut app.composer, &mut app.render, key, !input_blocked);
        }
        _ => {}
    }
    effects
}

// ── Mode switching ─────────────────────────────────────────────────────────────

/// Switch the agent mode to `target` (e.g. "build", "plan").
/// Caches the outgoing mode state, sends `SetAgentMode`, restores cached state
/// for the target mode, and persists config/cache.
fn switch_mode(app: &mut App, target: &str) -> Vec<Effect> {
    let command = Effect::Command(Command::SetAgentMode {
        mode: target.to_string(),
    });
    app.sessions.apply_mode_transition(target);
    vec![command, Effect::PersistConfig]
}

// ── Slash command execution ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlashResult {
    /// Command executed; input has been consumed.
    Handled,
    /// User invoked `/editor`; caller should open the external editor.
    OpenEditor,
    /// Input starts with `/` but the command name is not recognised;
    /// treat as a normal chat message.
    NotACommand,
}

/// Parse and execute a slash command from `app.composer.input`.
///
/// Expects `app.composer.input` to begin with `/`.  Always calls `app.take_input()`
/// before performing any side-effects so the command text is cleared first
/// (this allows `/undo` to optionally restore the previous turn text).
fn try_execute_slash_command(app: &mut App) -> (SlashResult, Vec<Effect>) {
    let after_slash = app.composer.input.trim_start_matches('/');
    let cmd = after_slash
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase();
    let arg = after_slash
        .strip_prefix(cmd.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let mut effects = Vec::new();

    if cmd.is_empty() {
        return (SlashResult::NotACommand, effects);
    }

    match cmd.as_str() {
        "model" => {
            app.take_input();
            if app.navigation.screen != Screen::Chat {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "model",
                    "model select is only available in chat",
                );
                return (SlashResult::Handled, effects);
            }
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            app.navigation.popup = Popup::ModelSelect;
            if arg.is_empty() {
                app.models.reset_for_open();
            } else {
                app.models.model_popup_agent_tab = 0;
                app.models.replace_filter(arg);
            }
        }
        "mode" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if arg.is_empty() {
                effects.extend(switch_mode(app, &app.sessions.next_mode()));
            } else {
                match arg.as_str() {
                    "build" | "plan" => {
                        if app.sessions.agent_mode == arg {
                            app.diagnostics.set_status(
                                LogLevel::Info,
                                "mode",
                                format!("already in {} mode", arg),
                            );
                        } else {
                            effects.extend(switch_mode(app, &arg));
                        }
                    }
                    _ => app.diagnostics.set_status(
                        LogLevel::Warn,
                        "mode",
                        format!("unknown mode: {} (try build or plan)", arg),
                    ),
                }
            }
        }
        "review" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if app.sessions.agent_mode == "review" {
                app.diagnostics
                    .set_status(LogLevel::Info, "mode", "already in review mode");
            } else {
                effects.extend(switch_mode(app, "review"));
            }
        }
        "thinking" => {
            app.take_input();
            if arg.is_empty() {
                app.diagnostics.set_status(
                    LogLevel::Info,
                    "model",
                    format!("thinking: {}", app.models.reasoning_effort_label()),
                );
            } else {
                let level = arg.to_lowercase();
                if crate::models_state::validate_reasoning_effort(Some(&level)).is_none() {
                    app.diagnostics.set_status(
                        LogLevel::Warn,
                        "model",
                        format!(
                            "unknown level: {} (try auto, low, medium, high, max)",
                            level
                        ),
                    );
                } else {
                    if !can_send_server_commands(app) {
                        return (SlashResult::Handled, effects);
                    }
                    effects.push(Effect::Command(
                        app.set_reasoning_effort(Some(&level)).unwrap(),
                    ));
                    app.diagnostics.set_status(
                        LogLevel::Info,
                        "model",
                        format!("thinking: {}", app.models.reasoning_effort_label()),
                    );
                }
            }
        }
        "theme" => {
            app.take_input();
            app.navigation
                .open_theme_selector(crate::theme::Theme::current_index());
        }
        "profile" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if arg.is_empty() {
                app.open_profile_popup();
                effects.push(Effect::Command(Command::ListProfiles));
            } else if let Some(profile_id) = app.profiles.find_profile_id(&arg) {
                app.profiles.active_profile_id = Some(profile_id.clone());
                if app.current_session_profile_id().is_none() {
                    app.models.clear_profile_agents();
                    effects.push(Effect::Command(Command::ListProfileAgents {
                        profile_id: profile_id.clone(),
                    }));
                }
                app.diagnostics.set_status(
                    LogLevel::Info,
                    "profile",
                    format!("new sessions will use {profile_id}"),
                );
                effects.push(Effect::PersistConfig);
            } else {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "profile",
                    format!("unknown profile: {}", arg),
                );
            }
        }
        "sessions" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            app.navigation.popup = Popup::SessionSelect;
            app.sessions.reset_browser_for_open();
            if let Some(request) = app.begin_session_discovery() {
                effects.push(Effect::Command(request));
            }
        }
        "delegates" => {
            app.take_input();
            if app.navigation.screen != Screen::Chat {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "delegates",
                    "delegates only available in chat",
                );
                return (SlashResult::Handled, effects);
            }
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            app.open_delegate_popup();
        }
        "new" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            app.open_new_session_popup();
        }
        "help" => {
            app.take_input();
            app.navigation.open_help();
        }
        "logs" => {
            app.take_input();
            open_log_popup(app);
        }
        "auth" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            app.open_auth_popup();
            effects.push(Effect::Command(Command::ListAuthProviders));
        }
        "fork" => {
            app.take_input();
            if app.navigation.screen != Screen::Chat {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "fork",
                    "forking is only available in chat",
                );
                return (SlashResult::Handled, effects);
            }
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if app.chat.pending_fork_message_id.is_some() {
                app.diagnostics
                    .set_status(LogLevel::Warn, "fork", "fork already pending");
            } else if app.chat.has_pending_session_op() {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "fork",
                    "session operation already pending",
                );
            } else if app.chat.is_turn_active() {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "fork",
                    "cannot fork while agent is active",
                );
            } else if let Some(turn) = app.chat.latest_fork_boundary() {
                effects.extend(begin_fork_session(app, turn.message_id));
            } else {
                app.diagnostics
                    .set_status(LogLevel::Warn, "fork", "no forkable turns");
            }
        }
        "undo" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if app.chat.is_turn_active() {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot undo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "undo already pending");
            } else if let Some(turn) = app.chat.current_undo_target().cloned() {
                if app.composer.input.trim().is_empty() && !turn.text.is_empty() {
                    app.composer.replace_input(turn.text.clone());
                    app.render.reset_composer_input_geometry();
                }
                app.chat.push_pending_undo(&turn);
                app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
                app.diagnostics
                    .set_status(LogLevel::Info, "session", "undoing...");
                effects.push(Effect::Command(Command::Undo {
                    message_id: turn.message_id,
                }));
            } else {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "nothing to undo");
            }
        }
        "redo" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if app.chat.is_turn_active() {
                app.diagnostics.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot redo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "redo already pending");
            } else if app.chat.can_redo() {
                app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
                app.diagnostics
                    .set_status(LogLevel::Info, "session", "redoing...");
                effects.push(Effect::Command(Command::Redo));
            } else {
                app.diagnostics
                    .set_status(LogLevel::Warn, "session", "nothing to redo");
            }
        }
        "editor" => {
            app.take_input();
            return (SlashResult::OpenEditor, effects);
        }
        "cancel" => {
            app.take_input();
            if app.chat.has_cancellable_activity() {
                app.chat.clear_cancel_confirm();
                app.diagnostics
                    .set_status(LogLevel::Warn, "activity", "stopping...");
                effects.push(Effect::Command(Command::CancelSession));
            } else {
                app.diagnostics
                    .set_status(LogLevel::Warn, "activity", "nothing to cancel");
            }
        }
        "quit" => {
            app.take_input();
            effects.push(Effect::Quit);
        }
        _ => return (SlashResult::NotACommand, effects),
    }

    (SlashResult::Handled, effects)
}

// ── Auth popup key handler ─────────────────────────────────────────────────────

pub(crate) fn handle_auth_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if app.auth.clipboard_fallback.is_some() {
        app.auth.clipboard_fallback = None;
        return Vec::new();
    }

    match handle_auth_input_key(&mut app.auth, key) {
        AuthInputResult::ClosePopup => {
            app.navigation.popup = Popup::None;
            Vec::new()
        }
        AuthInputResult::StartOAuth { provider } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::StartOAuthLogin { provider })]
        }
        AuthInputResult::DisconnectOAuth { provider } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::DisconnectOAuth { provider })]
        }
        AuthInputResult::ClearApiToken { provider } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::ClearApiToken { provider })]
        }
        AuthInputResult::SetApiToken { provider, api_key } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::SetApiToken { provider, api_key })]
        }
        AuthInputResult::CopyOAuthUrl { provider, url } => {
            app.auth.ui_notice = None;
            app.auth.clipboard_fallback = None;
            vec![Effect::CopyToClipboard {
                target: ClipboardTarget::Auth { provider },
                text: url,
            }]
        }
        AuthInputResult::CompleteOAuth { flow_id, response } => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            vec![Effect::Command(Command::CompleteOAuthLogin {
                flow_id,
                response,
            })]
        }
        AuthInputResult::NotHandled
        | AuthInputResult::Moved
        | AuthInputResult::Filtered
        | AuthInputResult::Edited
        | AuthInputResult::SelectedProvider
        | AuthInputResult::ToggledApiKeyMask
        | AuthInputResult::CloseDetail
        | AuthInputResult::BackToList
        | AuthInputResult::EnterApiKeyPanel => Vec::new(),
    }
}

fn apply_model_input_result(app: &mut App, result: ModelInputResult) -> Vec<Effect> {
    match result {
        ModelInputResult::Close => {
            app.navigation.popup = Popup::None;
            Vec::new()
        }
        ModelInputResult::SelectModel {
            model,
            tab: ModelTab::Session,
        } => {
            let mut effects = Vec::new();
            if !app.sessions.current_session_is_remote() {
                let sends_server_command =
                    app.sessions.session_id.is_some() || app.models.reasoning_effort.is_some();
                if sends_server_command && !can_send_server_commands(app) {
                    return Vec::new();
                }
                if let Some(session_id) = app.sessions.session_id.clone() {
                    effects.push(Effect::Command(Command::SetSessionModel {
                        session_id,
                        model_id: model.id.clone(),
                        node_id: model.node_id.clone(),
                    }));
                }
                app.models.apply_model_selection_from_entry(&model);
                if app.models.reasoning_effort.is_some() {
                    app.models.reasoning_effort = None;
                    effects.push(Effect::Command(Command::SetReasoningEffort {
                        reasoning_effort: "auto".into(),
                    }));
                }
            }
            app.diagnostics.set_status(
                LogLevel::Info,
                "model",
                format!("session: {}", model.label),
            );
            effects.push(Effect::PersistConfig);
            effects
        }
        ModelInputResult::SelectModel {
            model,
            tab: ModelTab::Delegate { agent_id, label },
        } => {
            let profile_id = app.delegate_preference_profile_id().map(str::to_string);
            let root_session_id = app
                .delegates
                .parent_session_id
                .is_none()
                .then(|| app.sessions.session_id.clone())
                .flatten();
            if profile_id.is_some() && root_session_id.is_some() && !can_send_server_commands(app) {
                return Vec::new();
            }
            let mut effects = Vec::new();
            if let Some(profile_id) = profile_id {
                app.models
                    .set_delegate_model_preference(&profile_id, &agent_id, &model);
                if let Some(session_id) = root_session_id {
                    effects.push(Effect::Command(Command::SetDelegateModel {
                        session_id,
                        agent_id,
                        model_id: Some(model.id.clone()),
                        node_id: model.node_id.clone(),
                    }));
                }
                app.diagnostics.set_status(
                    LogLevel::Info,
                    "model",
                    format!("{label}: {}", model.label),
                );
            }
            effects.push(Effect::PersistConfig);
            effects
        }
        ModelInputResult::ClearDelegatePreference { agent_id } => {
            let Some(profile_id) = app.delegate_preference_profile_id().map(str::to_string) else {
                return Vec::new();
            };
            let root_session_id = app
                .delegates
                .parent_session_id
                .is_none()
                .then(|| app.sessions.session_id.clone())
                .flatten();
            if root_session_id.is_some() && !can_send_server_commands(app) {
                return Vec::new();
            }
            app.models
                .clear_delegate_model_preference(&profile_id, &agent_id);
            let mut effects = Vec::new();
            if let Some(session_id) = root_session_id {
                effects.push(Effect::Command(Command::SetDelegateModel {
                    session_id,
                    agent_id,
                    model_id: None,
                    node_id: None,
                }));
            }
            app.diagnostics.set_status(
                LogLevel::Info,
                "model",
                "delegate model uses profile default",
            );
            effects.push(Effect::PersistConfig);
            effects
        }
        ModelInputResult::NotHandled
        | ModelInputResult::Moved
        | ModelInputResult::SwitchedTab
        | ModelInputResult::Filtered => Vec::new(),
    }
}

pub(crate) fn handle_model_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let delegate_profile_id = app.delegate_preference_profile_id().map(str::to_string);
    let result = handle_model_input_key(&mut app.models, key, delegate_profile_id.as_deref());
    apply_model_input_result(app, result)
}

// ── Sessions screen result adapter ───────────────────────────────────────────

/// Result of handling a sessions-screen key.
#[derive(Debug, PartialEq)]
pub(crate) enum SessionKeyAction {
    /// Nothing to send to the server.
    None,
    /// Load the local session with the given id and subscribe to it.
    LoadSession {
        session_id: String,
        agent_id: Option<String>,
        cwd: Option<String>,
    },
    /// Attach a remote session through its node.
    AttachRemoteSession { node_id: String, session_id: String },
    /// Delete the local session with the given id.
    DeleteSession { session_id: String },
    /// Dismiss the remote session from this TUI.
    DismissRemoteSession { session_id: String },
    /// Create a new session.
    NewSession,
    /// Load the next session page for a cwd group or fork children for a parent path.
    LoadMoreSessions {
        group_idx: usize,
        parent_path: Vec<usize>,
    },
}

/// Adapt sessions-screen input into the existing root-owned session action.
#[cfg(test)]
pub(crate) fn apply_sessions_key(
    app: &mut App,
    key: crossterm::event::KeyCode,
) -> SessionKeyAction {
    let result = handle_sessions_input_key(&mut app.sessions, key);
    apply_sessions_input_result(app, result)
}

// ── model popup tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod model_popup_tests {
    use super::*;
    use crate::app::App;
    use crate::command::PromptBlock;
    use crate::config::TestPersistenceGuard;
    use crate::domain::chat::ChatEntry;
    use crate::domain::model::ModelEntry;
    use crate::domain::profile::{AgentInfo, ProfileInfo};
    use crate::domain::session::{SessionGroup, SessionSummary, UndoableTurn};
    use crate::runtime::TestEffects;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    #[test]
    fn mention_typing_prepares_and_sends_file_index_request_once() {
        let mut effects = TestEffects::default();
        let mut app = App::new();

        effects.extend(handle_chat_key(&mut app, key(KeyCode::Char('@'))));
        effects.extend(handle_chat_key(&mut app, key(KeyCode::Char('s'))));

        assert!(matches!(
            effects.next_command(),
            Some(Command::GetFileIndex)
        ));
        assert!(effects.next_command().is_none());
        assert!(app.composer.file_index_loading);
        assert_eq!(app.composer.input, "@s");
    }

    #[test]
    fn prompt_send_is_optimistic_and_clears_input_before_dispatch() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.composer.replace_input(" hello ".into());
        app.render.test_seed_composer_input_geometry(24, 2);
        app.render.set_chat_scroll_offset(9);

        effects.extend(handle_chat_key(&mut app, key(KeyCode::Enter)));

        assert!(app.composer.input.is_empty());
        assert_eq!(app.render.test_composer_input_geometry(), (1, 0, false));
        assert_eq!(app.render.chat_scroll_offset(), 0);
        let local_id = match app.chat.messages.as_slice() {
            [
                ChatEntry::User {
                    text,
                    message_id: Some(local_id),
                },
            ] if text == "hello" => local_id.clone(),
            messages => panic!("unexpected optimistic messages: {messages:?}"),
        };
        assert!(matches!(
            effects.next_command(),
            Some(Command::Prompt { prompt, local_id: sent_id })
                if sent_id == local_id
                    && matches!(prompt.as_slice(), [PromptBlock::Text { text }] if text == "hello")
        ));
    }

    #[test]
    fn whitespace_prompt_early_return_still_snaps_chat_to_bottom() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.composer.replace_input("   ".into());
        app.render.test_seed_composer_input_geometry(18, 1);
        app.render.set_chat_scroll_offset(9);

        let effects = handle_chat_key(&mut app, key(KeyCode::Enter));

        assert!(effects.is_empty());
        assert!(app.composer.input.is_empty());
        assert!(app.chat.messages.is_empty());
        assert_eq!(app.render.test_composer_input_geometry(), (1, 0, false));
        assert_eq!(app.render.chat_scroll_offset(), 0);
    }

    #[test]
    fn optimistic_prompt_insertion_snaps_chat_to_bottom() {
        let mut app = App::new();
        app.render.set_chat_scroll_offset(9);

        let local_id = app.push_pending_prompt("hello".into());

        assert_eq!(app.render.chat_scroll_offset(), 0);
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "hello" && message_id == &local_id
        ));
    }

    #[test]
    fn failed_prompt_dispatch_rolls_back_only_optimistic_message() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.chat.messages.push(ChatEntry::Assistant {
            content: "keep".into(),
            thinking: None,
            message_id: Some("assistant-1".into()),
        });
        app.composer.replace_input("send me".into());

        effects.extend(handle_chat_key(&mut app, key(KeyCode::Enter)));
        let command = effects.next_command().expect("prompt effect");
        assert!(matches!(command, Command::Prompt { .. }));
        crate::application::update(
            &mut app,
            crate::application::AppEvent::Runtime(
                crate::application::RuntimeEvent::CommandFailed {
                    command,
                    message: "command channel closed".into(),
                },
            ),
        );

        assert!(app.composer.input.is_empty());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Assistant { content, .. }] if content == "keep"
        ));
    }

    #[test]
    fn take_input_resets_composer_then_chat_scroll_boundary() {
        let mut app = App::new();
        app.composer.input = "draft".into();
        app.composer.input_cursor = 3;
        app.render.test_seed_composer_input_geometry(20, 2);
        app.composer.input_preferred_col = Some(4);
        app.composer.refresh_mention_state();
        app.composer.input = "/mo".into();
        app.composer.input_cursor = 3;
        app.composer.refresh_slash_state();
        app.render.set_chat_scroll_offset(7);

        assert_eq!(app.take_input(), "/mo");
        assert_eq!(app.composer.input_cursor, 0);
        assert_eq!(app.render.test_composer_input_geometry(), (1, 0, false));
        assert_eq!(app.composer.input_preferred_col, None);
        assert!(app.composer.mention_state.is_none());
        assert!(app.composer.slash_state.is_none());
        assert_eq!(app.render.chat_scroll_offset(), 0);
    }

    #[test]
    fn chord_and_slash_undo_replacements_reset_render_geometry() {
        let turn = UndoableTurn {
            turn_id: "turn-1".into(),
            message_id: "message-1".into(),
            text: "restored prompt".into(),
        };

        let mut chord_app = App::new();
        chord_app.connection.conn = ConnState::Connected;
        chord_app.chat.undoable_turns.push(turn.clone());
        chord_app.render.test_seed_composer_input_geometry(25, 2);
        let effects = handle_chord(&mut chord_app, key(KeyCode::Char('u')));
        assert!(matches!(
            effects.as_slice(),
            [Effect::Command(Command::Undo { message_id })] if message_id == "message-1"
        ));
        assert_eq!(chord_app.composer.input, "restored prompt");
        assert_eq!(
            chord_app.render.test_composer_input_geometry(),
            (1, 0, false)
        );

        let mut slash_app = App::new();
        slash_app.connection.conn = ConnState::Connected;
        slash_app.chat.undoable_turns.push(turn);
        slash_app.composer.replace_input("/undo".into());
        slash_app.render.test_seed_composer_input_geometry(31, 3);
        let effects = handle_chat_key(&mut slash_app, key(KeyCode::Enter));
        assert!(matches!(
            effects.as_slice(),
            [Effect::Command(Command::Undo { message_id })] if message_id == "message-1"
        ));
        assert_eq!(slash_app.composer.input, "restored prompt");
        assert_eq!(
            slash_app.render.test_composer_input_geometry(),
            (1, 0, false)
        );
    }

    #[test]
    fn can_send_server_commands_accepts_only_connected_and_preserves_exact_warning() {
        let mut app = App::new();

        assert!(!can_send_server_commands(&mut app));
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );

        app.connection.conn = ConnState::Disconnected;
        app.diagnostics
            .set_status(LogLevel::Debug, "test", "before disconnected guard");
        assert!(!can_send_server_commands(&mut app));
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );

        app.connection.conn = ConnState::Connected;
        app.diagnostics
            .set_status(LogLevel::Debug, "test", "retained");
        assert!(can_send_server_commands(&mut app));
        assert_eq!(app.diagnostics.status, "retained");
    }

    fn make_model(provider: &str, model: &str) -> ModelEntry {
        ModelEntry {
            id: format!("{provider}/{model}"),
            label: model.into(),
            provider: provider.into(),
            model: model.into(),
            node_id: None,
            node_label: None,
            family: None,
            quant: None,
        }
    }

    fn make_agent(id: &str, name: &str) -> AgentInfo {
        AgentInfo {
            id: id.into(),
            name: name.into(),
            description: None,
            capabilities: Vec::new(),
        }
    }

    fn make_profile(id: &str, name: &str) -> ProfileInfo {
        ProfileInfo {
            id: id.into(),
            name: name.into(),
            ..Default::default()
        }
    }

    #[test]
    fn start_page_enter_on_remote_session_attaches_instead_of_loading() {
        let mut app = App::new();
        app.sessions.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-1".into(),
                node_id: Some("node-1".into()),
                attached: Some(false),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.sessions.session_cursor = 1;

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "remote-1".into(),
            }
        );
    }

    #[test]
    fn popup_enter_on_remote_session_attaches_instead_of_loading() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-1".into(),
                node_id: Some("node-1".into()),
                attached: Some(true),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.sessions.session_cursor = 1;

        let action = apply_popup_session_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "remote-1".into(),
            }
        );
        assert!(matches!(app.navigation.popup, Popup::None));
    }

    #[test]
    fn ctrl_p_opens_command_palette_over_existing_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::ThemeSelect;
        app.navigation.chord = true;
        app.navigation.command_palette_cursor = 4;
        app.navigation.command_palette_filter = "old".into();
        app.navigation.theme_filter = "keep".into();
        app.navigation.help_scroll = 9;
        let mut effects = TestEffects::default();
        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
        ));

        assert!(matches!(app.navigation.popup, Popup::CommandPalette));
        assert!(!app.navigation.chord);
        assert_eq!(app.navigation.command_palette_cursor, 0);
        assert!(app.navigation.command_palette_filter.is_empty());
        assert_eq!(app.navigation.theme_filter, "keep");
        assert_eq!(app.navigation.help_scroll, 9);
    }

    #[test]
    fn help_popup_routes_scroll_and_escape_before_screen_keys() {
        let mut app = App::new();
        app.navigation.open_help();
        app.sessions.session_filter = "keep".into();
        let mut effects = TestEffects::default();

        effects.extend(handle_key(&mut app, key(KeyCode::Char('x'))));
        assert_eq!(app.sessions.session_filter, "keep");
        assert_eq!(app.navigation.popup, Popup::Help);

        effects.extend(handle_key(&mut app, key(KeyCode::Down)));
        assert_eq!(app.navigation.help_scroll, 1);
        effects.extend(handle_key(&mut app, key(KeyCode::Up)));
        effects.extend(handle_key(&mut app, key(KeyCode::Up)));
        assert_eq!(app.navigation.help_scroll, 0);

        effects.extend(handle_key(&mut app, key(KeyCode::Esc)));
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn theme_popup_clamps_cursor_and_keeps_no_match_open() {
        let mut app = App::new();
        app.navigation.popup = Popup::ThemeSelect;
        app.navigation.theme_filter = "no theme matches this".into();
        app.navigation.theme_cursor = 8;

        assert!(handle_theme_popup_key(&mut app, key(KeyCode::Down)).is_empty());
        assert_eq!(app.navigation.theme_cursor, 0);
        assert!(handle_theme_popup_key(&mut app, key(KeyCode::Enter)).is_empty());
        assert_eq!(app.navigation.popup, Popup::ThemeSelect);
    }

    #[test]
    fn empty_theme_filter_out_of_range_enter_still_closes() {
        let _guard = TestPersistenceGuard::new("theme-out-of-range");
        let mut app = App::new();
        app.navigation.popup = Popup::ThemeSelect;
        app.navigation.theme_cursor = theme::Theme::available_themes().len() + 10;

        let effects = handle_theme_popup_key(&mut app, key(KeyCode::Enter));

        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(effects, vec![Effect::PersistConfig]);
    }

    #[test]
    fn command_palette_enter_opens_theme_picker() {
        let mut app = App::new();
        app.navigation.popup = Popup::CommandPalette;
        app.navigation.command_palette_filter = "theme".into();
        let mut effects = TestEffects::default();
        effects.extend(handle_command_palette_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(app.navigation.popup, Popup::ThemeSelect));
    }

    #[test]
    fn command_palette_open_mesh_refreshes_remote_nodes() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::CommandPalette;
        app.navigation.command_palette_filter = "open mesh".into();
        let mut effects = TestEffects::default();
        effects.extend(handle_command_palette_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(app.navigation.popup, Popup::Mesh));
        assert!(matches!(
            effects.next_command(),
            Some(Command::ListRemoteNodes)
        ));
    }

    #[test]
    fn disconnected_command_palette_action_keeps_popup_and_emits_no_command() {
        let mut app = App::new();
        app.connection.conn = ConnState::Disconnected;
        app.navigation.popup = Popup::CommandPalette;
        app.navigation.command_palette_filter = "open mesh".into();

        assert!(handle_command_palette_key(&mut app, key(KeyCode::Enter)).is_empty());
        assert_eq!(app.navigation.popup, Popup::CommandPalette);
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );
    }

    #[test]
    fn mesh_popup_enter_on_session_attaches_remote_session() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::Mesh;
        app.mesh.mesh_focus = crate::mesh_state::MeshFocus::Sessions;
        app.mesh.mesh_nodes = vec![crate::domain::mesh::RemoteNodeInfo {
            id: "node-1".into(),
            label: "framework".into(),
            ..Default::default()
        }];
        app.mesh.remote_sessions_by_node.insert(
            "node-1".into(),
            vec![crate::domain::mesh::RemoteSessionInfo {
                id: "remote-1".into(),
                node_id: "node-1".into(),
                title: Some("Fix bug".into()),
                ..Default::default()
            }],
        );
        let mut effects = TestEffects::default();
        effects.extend(handle_mesh_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(
            effects.next_command(),
            Some(Command::AttachRemoteSession { node_id, session_id })
                if node_id == "node-1" && session_id == "remote-1"
        ));
    }

    #[test]
    fn mesh_node_movement_refreshes_exact_selected_node_page() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::Mesh;
        app.mesh.mesh_nodes = vec![
            crate::domain::mesh::RemoteNodeInfo {
                id: "node-1".into(),
                ..Default::default()
            },
            crate::domain::mesh::RemoteNodeInfo {
                id: "node-2".into(),
                ..Default::default()
            },
        ];

        assert_eq!(
            handle_mesh_popup_key(&mut app, key(KeyCode::Down)),
            vec![Effect::Command(Command::ListRemoteSessions {
                node_id: "node-2".into(),
                offset: 0,
                limit: 50,
            })]
        );
    }

    #[test]
    fn mesh_popup_create_remote_session_does_not_forward_local_cwd() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::Mesh;
        app.connection.launch_cwd = Some("/local/launch".into());
        app.mesh.mesh_nodes = vec![crate::domain::mesh::RemoteNodeInfo {
            id: "node-1".into(),
            label: "framework".into(),
            ..Default::default()
        }];
        let mut effects = TestEffects::default();
        effects.extend(handle_mesh_popup_key(&mut app, key(KeyCode::Char('n'))));

        assert_eq!(
            effects.next_command().unwrap(),
            Command::CreateRemoteSession {
                node_id: "node-1".into(),
                cwd: None,
            }
        );
    }

    #[test]
    fn disconnected_mesh_command_intents_are_gated() {
        let mut app = App::new();
        app.navigation.popup = Popup::Mesh;
        app.mesh.mesh_nodes = vec![crate::domain::mesh::RemoteNodeInfo {
            id: "node-1".into(),
            ..Default::default()
        }];

        assert!(handle_mesh_popup_key(&mut app, key(KeyCode::Char('r'))).is_empty());
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );

        app.open_mesh_invite_form();
        assert!(handle_mesh_invite_popup_key(&mut app, key(KeyCode::Enter)).is_empty());
        assert_eq!(app.navigation.popup, Popup::MeshInvite);
    }

    #[test]
    fn command_palette_create_mesh_invite_opens_prefilled_invite_popup() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::CommandPalette;
        app.navigation.command_palette_filter = "mesh invite".into();
        let mut effects = TestEffects::default();
        effects.extend(handle_command_palette_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(app.navigation.popup, Popup::MeshInvite));
        assert_eq!(app.mesh.mesh_invite_ttl, "24h");
        assert_eq!(app.mesh.mesh_invite_max_uses, "1");
    }

    #[test]
    fn mesh_popup_i_no_longer_opens_invite_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::Mesh;
        let mut effects = TestEffects::default();
        effects.extend(handle_mesh_popup_key(&mut app, key(KeyCode::Char('i'))));

        assert!(matches!(app.navigation.popup, Popup::Mesh));
    }

    #[test]
    fn mesh_invite_u_shows_manual_url_overlay() {
        let mut app = App::new();
        app.open_mesh_invite_form();
        app.apply_mesh_invite_created(crate::domain::mesh::MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: Some("QR".into()),
            expires_at: 1,
            max_uses: 1,
            mesh_name: None,
        });

        handle_mesh_invite_qr_popup_key(&mut app, key(KeyCode::Char('u')));

        assert_eq!(
            app.mesh.mesh_clipboard_fallback.as_deref(),
            Some("qmt://mesh/join/token")
        );

        handle_mesh_invite_qr_popup_key(&mut app, key(KeyCode::Esc));

        assert!(matches!(app.navigation.popup, Popup::MeshInviteQr));
        assert!(app.mesh.mesh_clipboard_fallback.is_none());
    }

    #[test]
    fn mesh_invite_form_rejects_invalid_ttl_and_max_uses() {
        let mut app = App::new();
        app.open_mesh_invite_form();
        app.mesh.mesh_invite_ttl = "lol".into();
        let mut effects = TestEffects::default();
        effects.extend(handle_mesh_invite_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(effects.next_command().is_none());
        assert_eq!(
            app.mesh.mesh_error.as_deref(),
            Some("ttl must be like 30m, 1d3h, or 1d3h5m")
        );

        app.mesh.mesh_invite_ttl = "24h".into();
        app.mesh.mesh_invite_max_uses = "0".into();
        effects.extend(handle_mesh_invite_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(effects.next_command().is_none());
        assert_eq!(
            app.mesh.mesh_error.as_deref(),
            Some("max uses must be at least 1")
        );
    }

    #[test]
    fn mesh_invite_form_enter_sends_create_invite() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::Mesh;
        app.open_mesh_invite_form();
        app.mesh.mesh_invite_name = "Team Mesh".into();
        app.mesh.mesh_invite_ttl = "1d3h5m".into();
        let mut effects = TestEffects::default();
        effects.extend(handle_mesh_invite_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(
            effects.next_command(),
            Some(Command::CreateMeshInvite { mesh_name, ttl, max_uses })
                if mesh_name.as_deref() == Some("Team Mesh")
                    && ttl.as_deref() == Some("1d3h5m")
                    && max_uses == Some(1)
        ));
    }

    #[test]
    fn mesh_invite_qr_esc_returns_to_create_invite_popup() {
        let mut app = App::new();
        app.apply_mesh_invite_created(crate::domain::mesh::MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: Some("QR".into()),
            expires_at: 1,
            max_uses: 1,
            mesh_name: None,
        });

        handle_mesh_invite_qr_popup_key(&mut app, key(KeyCode::Esc));

        assert!(matches!(app.navigation.popup, Popup::MeshInvite));
    }

    #[test]
    fn command_palette_profile_lists_profiles() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::CommandPalette;
        app.navigation.command_palette_filter = "profile".into();
        app.profiles.profiles = vec![make_profile("fast", "Fast"), make_profile("deep", "Deep")];
        app.profiles.active_profile_id = Some("deep".into());
        app.profiles.profile_filter = "stale".into();
        app.profiles.profile_cursor = 0;
        let mut effects = TestEffects::default();
        effects.extend(handle_command_palette_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(app.navigation.popup, Popup::ProfileSelect));
        assert!(app.profiles.profile_filter.is_empty());
        assert_eq!(app.profiles.profile_cursor, 1);
        assert!(matches!(
            effects.next_command(),
            Some(Command::ListProfiles)
        ));
    }

    #[test]
    fn slash_profile_without_arg_opens_selector_and_lists_profiles() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/profile".into();
        app.composer.input_cursor = app.composer.input.len();
        let mut effects = TestEffects::default();
        effects.extend(handle_chat_key(&mut app, key(KeyCode::Enter)));

        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, Effect::OpenExternalEditor { .. }))
        );
        assert!(matches!(app.navigation.popup, Popup::ProfileSelect));
        assert!(matches!(
            effects.next_command(),
            Some(Command::ListProfiles)
        ));
    }

    #[test]
    fn slash_profile_with_arg_updates_local_new_session_profile() {
        let _guard = TestPersistenceGuard::new("slash-profile");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.screen = Screen::Chat;
        app.profiles.profiles = vec![make_profile("fast", "Fast")];
        app.profiles.active_profile_id = Some("old".into());
        app.composer.input = "/profile Fast".into();
        app.composer.input_cursor = app.composer.input.len();
        let mut effects = TestEffects::default();
        effects.extend(handle_chat_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(
            effects.next_command(),
            Some(Command::ListProfileAgents { profile_id }) if profile_id == "fast"
        ));
        assert!(effects.next_command().is_none());
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert_eq!(app.current_session_profile_id(), None);
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::PersistConfig))
        );
    }

    #[test]
    fn profile_popup_enter_updates_local_new_session_profile() {
        let _guard = TestPersistenceGuard::new("profile-popup-enter");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ProfileSelect;
        app.profiles.profiles = vec![make_profile("fast", "Fast"), make_profile("deep", "Deep")];
        app.profiles.profile_cursor = 1;
        let effects = handle_profile_popup_key(&mut app, key(KeyCode::Enter));

        assert!(matches!(app.navigation.popup, Popup::None));
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::ListProfileAgents {
                    profile_id: "deep".into(),
                }),
                Effect::PersistConfig,
            ]
        );
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("deep"));
    }

    #[test]
    fn disconnected_profile_submit_preserves_selection_and_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::ProfileSelect;
        app.profiles.profiles = vec![make_profile("fast", "Fast")];
        app.profiles.active_profile_id = Some("old".into());

        assert!(handle_profile_popup_key(&mut app, key(KeyCode::Enter)).is_empty());
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("old"));
        assert_eq!(app.navigation.popup, Popup::ProfileSelect);
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );
    }

    #[test]
    fn profile_selection_with_bound_session_skips_agent_refresh() {
        let _guard = TestPersistenceGuard::new("profile-bound-session");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ProfileSelect;
        app.sessions.session_id = Some("session".into());
        app.profiles
            .bind_session_profile("session".into(), "current".into());
        app.profiles.profiles = vec![make_profile("fast", "Fast")];
        let mut effects = TestEffects::default();
        effects.extend(handle_profile_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(effects.next_command().is_none());
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("fast"));
        assert_eq!(app.current_session_profile_id(), Some("current"));
    }

    #[test]
    fn new_session_uses_locally_selected_profile() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::NewSession;
        app.sessions.new_session_path = "/repo/./project/..".into();
        app.sessions.new_session_cursor = app.sessions.new_session_path.len();
        app.profiles.active_profile_id = Some("coder-delegate".into());

        let effects = handle_new_session_popup_key(&mut app, key(KeyCode::Enter));

        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(
            effects,
            vec![Effect::Command(Command::NewSession {
                cwd: Some("/repo".into()),
                profile_id: Some("coder-delegate".into()),
            })]
        );
    }

    #[test]
    fn select_model_on_delegate_tab_saves_pref_not_session_model() {
        let _guard = TestPersistenceGuard::new("delegate-tab");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.session_id = Some("s1".into());
        app.profiles.active_profile_id = Some("profile".into());
        app.profiles
            .bind_session_profile("s1".into(), "profile".into());
        app.sessions.agent_mode = "plan".into();
        app.models.current_provider = Some("openai".into());
        app.models.current_model = Some("gpt-4o".into());
        app.models.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        app.models.models = vec![
            make_model("openai", "gpt-4o"),
            make_model("anthropic", "claude-sonnet"),
        ];
        app.models.model_popup_agent_tab = 1; // delegate tab
        let items = app.models.visible_model_popup_items();
        app.models.model_cursor = items
            .iter()
            .position(|i| {
                matches!(
                    i,
                    crate::models_state::ModelPopupItem::Model { model_idx } if app.models.models[*model_idx].model == "claude-sonnet"
                )
            }).unwrap();
        let mut effects = TestEffects::default();
        effects.extend(handle_model_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(
            effects.next_command(),
            Some(Command::SetDelegateModel {
                ref session_id,
                ref agent_id,
                ref model_id,
                node_id: None,
            }) if session_id == "s1"
                && agent_id == "coder"
                && model_id.as_deref() == Some("anthropic/claude-sonnet")
        ));
        assert_eq!(app.models.current_provider.as_deref(), Some("openai"));
        assert_eq!(app.models.current_model.as_deref(), Some("gpt-4o"));
        assert_eq!(
            app.models
                .get_delegate_model_preference("profile", "coder")
                .map(|preference| preference.model_id.as_str()),
            Some("anthropic/claude-sonnet")
        );
    }

    #[test]
    fn delete_on_delegate_tab_resets_parent_override() {
        let _guard = TestPersistenceGuard::new("delegate-reset");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.session_id = Some("s1".into());
        app.profiles.active_profile_id = Some("profile".into());
        app.profiles
            .bind_session_profile("s1".into(), "profile".into());
        app.models.agents = vec![
            make_agent("primary", "Session"),
            make_agent("coder", "Coder"),
        ];
        app.models.model_popup_agent_tab = 1;
        let model = make_model("anthropic", "claude-sonnet");
        app.models
            .set_delegate_model_preference("profile", "coder", &model);
        let mut effects = TestEffects::default();
        effects.extend(handle_model_popup_key(&mut app, key(KeyCode::Delete)));

        assert!(
            app.models
                .get_delegate_model_preference("profile", "coder")
                .is_none()
        );
        assert!(matches!(
            effects.next_command(),
            Some(Command::SetDelegateModel {
                ref session_id,
                ref agent_id,
                model_id: None,
                node_id: None,
            }) if session_id == "s1" && agent_id == "coder"
        ));
    }

    #[test]
    fn delegate_child_model_selection_is_saved_without_mutating_child_session() {
        let _guard = TestPersistenceGuard::new("delegate-child-model");
        let mut app = App::new();
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.session_id = Some("child".into());
        app.delegates.parent_session_id = Some("parent".into());
        app.profiles.active_profile_id = Some("profile".into());
        app.profiles
            .bind_session_profile("child".into(), "profile".into());
        app.models.agents = vec![
            make_agent("primary", "Session"),
            make_agent("coder", "Coder"),
        ];
        app.models.models = vec![make_model("anthropic", "claude-sonnet")];
        app.models.model_popup_agent_tab = 1;
        app.models.model_cursor = 1;
        let mut effects = TestEffects::default();
        effects.extend(handle_model_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(effects.next_command().is_none());
        assert!(
            app.models
                .get_delegate_model_preference("profile", "coder")
                .is_some()
        );
    }

    #[test]
    fn select_model_on_session_tab_applies_set_session_model() {
        let _guard = TestPersistenceGuard::new("active-mode");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.session_id = Some("s1".into());
        app.sessions.agent_mode = "build".into();
        app.models.current_provider = Some("openai".into());
        app.models.current_model = Some("gpt-4o".into());
        app.models.reasoning_effort = Some("high".into());

        app.models.models = vec![
            make_model("openai", "gpt-4o"),
            make_model("anthropic", "claude-sonnet"),
        ];
        app.models.model_popup_agent_tab = 0;
        let items = app.models.visible_model_popup_items();
        app.models.model_cursor = items
            .iter()
            .position(|i| {
                matches!(
                    i,
                    crate::models_state::ModelPopupItem::Model { model_idx } if app.models.models[*model_idx].model == "claude-sonnet"
                )
            }).unwrap();
        let mut effects = TestEffects::default();
        effects.extend(handle_model_popup_key(&mut app, key(KeyCode::Enter)));

        let msg1 = effects.next_command().expect("expected SetSessionModel");
        assert!(matches!(msg1, Command::SetSessionModel { .. }));
        let msg2 = effects
            .next_command()
            .expect("expected SetReasoningEffort auto");
        assert!(
            matches!(msg2, Command::SetReasoningEffort { reasoning_effort } if reasoning_effort == "auto")
        );
        assert!(effects.next_command().is_none());

        assert_eq!(app.models.current_provider.as_deref(), Some("anthropic"));
        assert_eq!(app.models.current_model.as_deref(), Some("claude-sonnet"));
        assert_eq!(app.models.reasoning_effort, None);
    }

    #[test]
    fn select_remote_model_on_local_session_sends_node_id() {
        let _guard = TestPersistenceGuard::new("remote-mesh-model");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.session_id = Some("s1".into());
        app.sessions.agent_mode = "build".into();
        app.models.current_provider = Some("openai".into());
        app.models.current_model = Some("gpt-4o".into());
        app.models.models = vec![
            make_model("openai", "gpt-4o"),
            ModelEntry {
                id: "mesh/anthropic/claude".into(),
                label: "Claude".into(),
                provider: "anthropic".into(),
                model: "claude".into(),
                node_id: Some("node-a".into()),
                node_label: Some("peer-a".into()),
                family: None,
                quant: None,
            },
        ];
        app.models.model_popup_agent_tab = 0;
        let items = app.models.visible_model_popup_items();
        app.models.model_cursor = items
            .iter()
            .position(|i| {
                matches!(
                    i,
                    crate::models_state::ModelPopupItem::Model { model_idx } if app.models.models[*model_idx].node_id.is_some()
                )
            }).unwrap();
        let mut effects = TestEffects::default();
        effects.extend(handle_model_popup_key(&mut app, key(KeyCode::Enter)));

        match effects.next_command().expect("SetSessionModel") {
            Command::SetSessionModel {
                node_id, model_id, ..
            } => {
                assert_eq!(node_id.as_deref(), Some("node-a"));
                assert_eq!(model_id, "mesh/anthropic/claude");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(effects.next_command().is_none());
        assert_eq!(app.models.current_provider.as_deref(), Some("anthropic"));
        assert_eq!(app.models.current_model.as_deref(), Some("claude"));
        assert_eq!(app.models.current_model_node_id.as_deref(), Some("node-a"));
    }

    #[test]
    fn select_model_on_attached_remote_session_does_not_apply() {
        let _guard = TestPersistenceGuard::new("active-remote-mode");
        let mut app = App::new();
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.session_id = Some("remote-1".into());
        app.sessions.agent_mode = "build".into();
        app.sessions.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-1".into(),
                node_id: Some("node-1".into()),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.models.models = vec![make_model("anthropic", "claude-sonnet")];
        app.models.model_popup_agent_tab = 0;
        app.models.model_cursor = app
            .models
            .visible_model_popup_items()
            .iter()
            .position(|i| matches!(i, crate::models_state::ModelPopupItem::Model { .. }))
            .unwrap();
        let effects = handle_model_popup_key(&mut app, key(KeyCode::Enter));
        assert_eq!(effects, vec![Effect::PersistConfig]);
        assert_eq!(app.diagnostics.status, "session: claude-sonnet");
    }

    #[test]
    fn model_header_enter_is_noop_without_persistence() {
        let mut app = App::new();
        app.navigation.popup = Popup::ModelSelect;
        app.models.models = vec![make_model("anthropic", "claude-sonnet")];
        app.models.model_cursor = 0;

        assert!(handle_model_popup_key(&mut app, key(KeyCode::Enter)).is_empty());
        assert_eq!(app.models.current_model, None);
    }

    #[test]
    fn disconnected_local_model_selection_is_gated_before_mutation() {
        let mut app = App::new();
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.session_id = Some("session-1".into());
        app.models.current_provider = Some("old-provider".into());
        app.models.current_model = Some("old-model".into());
        app.models.models = vec![make_model("anthropic", "claude-sonnet")];
        app.models.model_cursor = 1;

        assert!(handle_model_popup_key(&mut app, key(KeyCode::Enter)).is_empty());
        assert_eq!(app.models.current_provider.as_deref(), Some("old-provider"));
        assert_eq!(app.models.current_model.as_deref(), Some("old-model"));
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );
    }

    #[test]
    fn opening_model_popup_starts_on_session_tab() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.screen = Screen::Chat;
        app.sessions.agent_mode = "review".into();
        app.models.current_provider = Some("openai".into());
        app.models.current_model = Some("gpt-4o".into());
        app.models.models = vec![make_model("openai", "gpt-4o")];
        let mut effects = TestEffects::default();
        effects.extend(handle_chord(&mut app, key(KeyCode::Char('m'))));

        assert!(matches!(app.navigation.popup, Popup::ModelSelect));
        assert_eq!(app.models.model_popup_agent_tab, 0);
        assert_eq!(
            app.models.model_cursor,
            app.models.model_popup_open_cursor()
        );
    }

    #[test]
    fn slash_model_starts_on_session_tab() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.screen = Screen::Chat;
        app.sessions.agent_mode = "review".into();
        app.models.current_provider = Some("openai".into());
        app.models.current_model = Some("gpt-4o".into());
        app.models.models = vec![make_model("openai", "gpt-4o")];
        app.composer.input = "/model".into();
        app.composer.input_cursor = app.composer.input.len();
        let mut effects = TestEffects::default();
        effects.extend(handle_chat_key(&mut app, key(KeyCode::Enter)));

        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, Effect::OpenExternalEditor { .. }))
        );
        assert!(matches!(app.navigation.popup, Popup::ModelSelect));
        assert_eq!(app.models.model_popup_agent_tab, 0);
        assert_eq!(
            app.models.model_cursor,
            app.models.model_popup_open_cursor()
        );
    }

    #[test]
    fn review_slash_command_enters_review_and_tab_returns_to_previous_mode() {
        let _guard = TestPersistenceGuard::new("review-cycle");
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.screen = Screen::Chat;
        app.sessions.agent_mode = "plan".into();
        app.composer.input = "/review".into();
        app.composer.input_cursor = app.composer.input.len();
        let effects = handle_chat_key(&mut app, key(KeyCode::Enter));
        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, Effect::OpenExternalEditor { .. }))
        );
        assert_eq!(app.sessions.agent_mode, "review");
        assert_eq!(app.sessions.mode_before_review.as_deref(), Some("plan"));
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::SetAgentMode {
                    mode: "review".into(),
                }),
                Effect::PersistConfig,
            ]
        );

        let effects = handle_key(&mut app, key(KeyCode::Tab));
        assert_eq!(app.sessions.agent_mode, "plan");
        assert_eq!(app.sessions.mode_before_review, None);
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::SetAgentMode {
                    mode: "plan".into(),
                }),
                Effect::PersistConfig,
            ]
        );
    }

    #[test]
    fn delegate_popup_subscribes_with_child_target_agent() {
        use crate::domain::activity::{
            DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus,
        };

        let mut app = App::new();
        app.sessions.agent_id = Some("planner".into());
        app.navigation.popup = Popup::SessionSelect;
        app.delegates.delegate_entries.push(DelegateEntry {
            delegation_id: "del-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "Fix bug".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_delegate_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(
            effects.next_command().expect("expected LoadSession"),
            Command::LoadSession { session_id, .. } if session_id == "child-1"
        ));
        assert!(matches!(
            effects.next_command().expect("expected SubscribeSession"),
            Command::SubscribeSession { session_id, agent_id }
                if session_id == "child-1" && agent_id.as_deref() == Some("coder")
        ));
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn review_slash_command_is_noop_when_already_in_review() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.screen = Screen::Chat;
        app.sessions.agent_mode = "review".into();
        app.sessions.mode_before_review = Some("plan".into());
        app.composer.input = "/review".into();
        app.composer.input_cursor = app.composer.input.len();
        let mut effects = TestEffects::default();
        effects.extend(handle_chat_key(&mut app, key(KeyCode::Enter)));
        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, Effect::OpenExternalEditor { .. }))
        );
        assert_eq!(app.sessions.agent_mode, "review");
        assert_eq!(app.sessions.mode_before_review.as_deref(), Some("plan"));
        assert!(effects.next_command().is_none());
        assert_eq!(app.diagnostics.status, "already in review mode");
    }
}
