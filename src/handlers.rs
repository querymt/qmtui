use crate::app::App;
use crate::application::{ClipboardTarget, Effect};
use crate::auth_state::AuthPanel;
use crate::diagnostics::LogLevel;
use crate::domain::activity::{ActivityState, SessionOp};
use crate::domain::auth::{OAuthFlowKind, OAuthStatus};
use crate::domain::model::ModelEntry;
use crate::features::chat::input::{
    CompletionResult, ComposerKeyResult, ElicitationResponseEffect, handle_completion_key,
    handle_composer_key, handle_elicitation_key,
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
        app.set_status(
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
        app.set_status(
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
    app.diagnostics.log_cursor = app.filtered_logs().len().saturating_sub(1);
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
    let mut effects = Vec::new();
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Tab | KeyCode::Right | KeyCode::Left => app.mesh.toggle_focus(),
        KeyCode::Up => match app.mesh.mesh_focus {
            crate::mesh_state::MeshFocus::Nodes => {
                if let Some(node_id) = app.mesh.move_mesh_node_cursor(-1) {
                    effects.push(Effect::Command(Command::ListRemoteSessions {
                        node_id,
                        offset: 0,
                        limit: 50,
                    }));
                }
            }
            crate::mesh_state::MeshFocus::Sessions => app.mesh.move_remote_session_cursor(-1),
        },
        KeyCode::Down => match app.mesh.mesh_focus {
            crate::mesh_state::MeshFocus::Nodes => {
                if let Some(node_id) = app.mesh.move_mesh_node_cursor(1) {
                    effects.push(Effect::Command(Command::ListRemoteSessions {
                        node_id,
                        offset: 0,
                        limit: 50,
                    }));
                }
            }
            crate::mesh_state::MeshFocus::Sessions => app.mesh.move_remote_session_cursor(1),
        },
        KeyCode::Char('r') => effects.push(Effect::Command(Command::ListRemoteNodes)),
        KeyCode::Enter => match app.mesh.mesh_focus {
            crate::mesh_state::MeshFocus::Nodes => {
                app.mesh.focus_sessions();
                if let Some(node_id) = app.mesh.selected_mesh_node_id() {
                    effects.push(Effect::Command(Command::ListRemoteSessions {
                        node_id: node_id.to_string(),
                        offset: 0,
                        limit: 50,
                    }));
                }
            }
            crate::mesh_state::MeshFocus::Sessions => {
                if let Some(session) = app.mesh.selected_remote_session() {
                    effects.push(Effect::Command(Command::AttachRemoteSession {
                        node_id: session.node_id.clone(),
                        session_id: session.id.clone(),
                    }));
                }
            }
        },
        KeyCode::Char('n') => {
            if let Some(node_id) = app.mesh.selected_mesh_node_id() {
                effects.push(Effect::Command(Command::CreateRemoteSession {
                    node_id: node_id.to_string(),
                    cwd: None,
                }));
            }
        }
        _ => {}
    }
    effects
}

pub(crate) fn handle_mesh_invite_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if app.mesh.consume_clipboard_fallback() {
        return Vec::new();
    }

    let mut effects = Vec::new();
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Up => app.mesh.move_invite_form_field(-1),
        KeyCode::Down | KeyCode::Tab => app.mesh.move_invite_form_field(1),
        KeyCode::Backspace => app.mesh.invite_form_backspace(),
        KeyCode::Enter => {
            if let Some(command) = app.mesh_invite_form_command() {
                effects.push(Effect::Command(command));
                app.set_status(LogLevel::Info, "mesh", "creating invite...");
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.mesh.invite_form_insert(c);
        }
        _ => {}
    }
    effects
}

pub(crate) fn handle_mesh_invite_qr_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if app.mesh.consume_clipboard_fallback() {
        return Vec::new();
    }

    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::MeshInvite,
        KeyCode::Char('u') => {
            app.mesh.show_invite_url_fallback();
        }
        KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(text) = app.mesh.invite_url().map(str::to_string) {
                return vec![Effect::CopyToClipboard {
                    target: ClipboardTarget::MeshInvite,
                    text,
                }];
            }
        }
        _ => {}
    }
    Vec::new()
}

pub(crate) fn handle_command_palette_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Up => app.navigation.move_command_palette_cursor(-1),
        KeyCode::Down => app.navigation.move_command_palette_cursor(1),
        KeyCode::Backspace => app.navigation.command_palette_filter_backspace(),
        KeyCode::Enter => {
            if let Some(action) = app.navigation.selected_command_palette_action() {
                return execute_command_palette_action(app, action);
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.navigation.command_palette_filter_insert(c);
        }
        _ => {}
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
        app.set_status(LogLevel::Debug, "input", "ready");
        if key.code == KeyCode::Char('e') {
            if app.navigation.screen != Screen::Chat {
                app.set_status(
                    LogLevel::Warn,
                    "editor",
                    "external editor is only available in chat",
                );
                return Vec::new();
            }
            return vec![Effect::OpenExternalEditor {
                initial_text: app.composer.input.clone(),
            }];
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
                app.set_status(
                    LogLevel::Info,
                    "model",
                    format!("thinking: {}", app.models.reasoning_effort_label()),
                );
                vec![Effect::Command(command)]
            }
            None => {
                app.set_status(
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
        app.set_status(LogLevel::Debug, "input", "C-x ...");
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
            match key.code {
                KeyCode::Esc => app.navigation.popup = Popup::None,
                KeyCode::Up => app.navigation.scroll_help_up(),
                KeyCode::Down => app.navigation.scroll_help_down(),
                _ => {}
            }
            return Vec::new();
        }
        Popup::Log => {
            let _ = handle_log_popup_key(app, key);
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
            app.set_status(LogLevel::Warn, "editor", "external editor unavailable here");
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
                app.set_status(
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
                app.set_status(LogLevel::Info, "session", "no parent session");
            }
        }
        KeyCode::Char('?') => app.navigation.open_help(),
        KeyCode::Char('f') => {
            if app.navigation.screen != Screen::Chat {
                app.set_status(
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
                app.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot undo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.set_status(LogLevel::Warn, "session", "undo already pending");
            } else if let Some(turn) = app.chat.current_undo_target().cloned() {
                if app.composer.input.trim().is_empty() && !turn.text.is_empty() {
                    app.composer.replace_input(turn.text.clone());
                    app.render.reset_composer_input_geometry();
                }
                app.chat.push_pending_undo(&turn);
                app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
                app.set_status(LogLevel::Info, "session", "undoing...");
                effects.push(Effect::Command(Command::Undo {
                    message_id: turn.message_id,
                }));
            } else {
                app.set_status(LogLevel::Warn, "session", "nothing to undo");
            }
        }
        KeyCode::Char('r') => {
            if !can_send_server_commands(app) {
                return effects;
            }
            if app.chat.is_turn_active() {
                app.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot redo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.set_status(LogLevel::Warn, "session", "undo already pending");
            } else if app.chat.can_redo() {
                app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
                app.set_status(LogLevel::Info, "session", "redoing...");
                effects.push(Effect::Command(Command::Redo));
            } else {
                app.set_status(LogLevel::Warn, "session", "nothing to redo");
            }
        }
        _ => app.set_status(LogLevel::Debug, "input", "unknown chord"),
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

pub(crate) fn handle_sessions_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
        return vec![Effect::Quit];
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        if let SessionKeyAction::LoadMoreSessions {
            group_idx,
            parent_path,
        } = apply_session_fork_toggle_key(app, false)
            && let Some(request) = app.session_child_page_request(group_idx, &parent_path)
        {
            return vec![Effect::Command(request)];
        }
        return Vec::new();
    }

    let action = apply_sessions_key(
        app,
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            KeyCode::Null
        } else {
            key.code
        },
    );
    session_action_effects(app, action)
}

pub(crate) fn handle_session_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
        app.sessions.switch_session_popup_tab();
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
        if let SessionKeyAction::LoadMoreSessions {
            group_idx,
            parent_path,
        } = apply_session_fork_toggle_key(app, true)
            && let Some(request) = app.session_child_page_request(group_idx, &parent_path)
        {
            return vec![Effect::Command(request)];
        }
        return Vec::new();
    }

    let action = apply_popup_session_key(
        app,
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            KeyCode::Null
        } else {
            key.code
        },
    );
    session_action_effects(app, action)
}

/// Pure key-handler for the session popup. Returns a [`SessionKeyAction`] that
/// the caller should forward to the server.
///
/// Uses [`App::visible_popup_items`] (grouped, no caps) and
/// [`App::popup_collapsed_groups`] so its collapse state is fully independent
/// of the start-page.
pub(crate) fn apply_popup_session_key(
    app: &mut App,
    key: crossterm::event::KeyCode,
) -> SessionKeyAction {
    use crate::session_state::PopupItem;

    match key {
        KeyCode::Esc => {
            app.navigation.popup = Popup::None;
        }
        KeyCode::Up => app.sessions.move_popup_cursor_up(),
        KeyCode::Down => app.sessions.move_popup_cursor_down(),
        KeyCode::PageUp => {
            let step = app.render.session_popup_page_step();
            app.sessions.move_popup_cursor_page(step, false);
        }
        KeyCode::PageDown => {
            let step = app.render.session_popup_page_step();
            app.sessions.move_popup_cursor_page(step, true);
        }
        KeyCode::Enter => {
            let items = app.sessions.visible_popup_items();
            if let Some(item) = items.get(app.sessions.session_cursor).cloned() {
                match item {
                    PopupItem::GroupHeader { cwd, .. } => {
                        app.sessions.toggle_popup_group_collapse(cwd.as_deref());
                        app.sessions.clamp_popup_cursor();
                    }
                    PopupItem::Session {
                        group_idx, path, ..
                    } => {
                        let session = app.sessions.session_by_path(group_idx, &path).cloned();
                        if let Some(session) = session {
                            app.navigation.popup = Popup::None;
                            let session_id = session.session_id;
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
                                app.set_status(
                                    LogLevel::Warn,
                                    "session",
                                    "remote session is missing node id; refresh sessions and try again",
                                );
                                return SessionKeyAction::None;
                            }
                            return SessionKeyAction::LoadSession {
                                session_id,
                                agent_id: None,
                                cwd: session
                                    .cwd
                                    .or_else(|| app.sessions.session_groups[group_idx].cwd.clone()),
                            };
                        }
                    }
                    PopupItem::LoadMore {
                        group_idx,
                        parent_path,
                    } => {
                        return SessionKeyAction::LoadMoreSessions {
                            group_idx,
                            parent_path,
                        };
                    }
                }
            }
        }
        KeyCode::Delete => {
            let items = app.sessions.visible_popup_items();
            if let Some(PopupItem::Session {
                group_idx, path, ..
            }) = items.get(app.sessions.session_cursor).cloned()
            {
                let Some((session, is_remote)) =
                    app.sessions.remove_session_at(group_idx, &path, true)
                else {
                    return SessionKeyAction::None;
                };
                let sid = session.session_id;
                return if is_remote {
                    SessionKeyAction::DismissRemoteSession { session_id: sid }
                } else {
                    SessionKeyAction::DeleteSession { session_id: sid }
                };
            }
            // Delete on a GroupHeader: no-op
        }
        KeyCode::Backspace => app.sessions.popup_filter_backspace(),
        KeyCode::Char(c) => app.sessions.popup_filter_insert(c),
        _ => {}
    }
    SessionKeyAction::None
}

pub(crate) fn apply_session_fork_toggle_key(app: &mut App, popup_items: bool) -> SessionKeyAction {
    use crate::session_state::{PopupItem, StartPageItem};

    let selected = if popup_items {
        app.sessions
            .visible_popup_items()
            .get(app.sessions.session_cursor)
            .cloned()
            .and_then(|item| match item {
                PopupItem::Session {
                    group_idx, path, ..
                } => Some((group_idx, path)),
                _ => None,
            })
    } else {
        app.sessions
            .visible_start_items()
            .get(app.sessions.session_cursor)
            .cloned()
            .and_then(|item| match item {
                StartPageItem::Session {
                    group_idx, path, ..
                } => Some((group_idx, path)),
                _ => None,
            })
    };

    let Some((group_idx, path)) = selected else {
        return SessionKeyAction::None;
    };

    let should_load = app.sessions.toggle_session_children(group_idx, &path);
    if should_load {
        SessionKeyAction::LoadMoreSessions {
            group_idx,
            parent_path: path,
        }
    } else {
        SessionKeyAction::None
    }
}

// ── Delegate view key handler (read-only child session) ──────────────────────

fn handle_delegate_view_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match key.code {
        KeyCode::Up => app.render.scroll_chat_up(1),
        KeyCode::Down => app.render.scroll_chat_down(1),
        KeyCode::PageUp => app.render.scroll_chat_up(10),
        KeyCode::PageDown => app.render.scroll_chat_down(10),
        KeyCode::Home => app.render.scroll_chat_to_top(),
        KeyCode::End => app.render.scroll_chat_to_bottom(),
        KeyCode::Esc => {
            if let Some(parent_sid) = app.delegates.parent_session_id.clone() {
                return send_load_session_commands(
                    parent_sid,
                    app.current_session_cwd(),
                    app.sessions.agent_id.clone(),
                );
            }
        }
        _ => {}
    }
    Vec::new()
}

// ── Delegate popup key handler ────────────────────────────────────────────────

pub(crate) fn handle_delegate_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let action = apply_delegate_popup_key(
        app,
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            KeyCode::Null
        } else {
            key.code
        },
    );
    session_action_effects(app, action)
}

/// Pure key-handler for the delegate popup. Returns a [`SessionKeyAction`]
/// that the caller should forward to the server.
///
/// Delegation entries are built from the parent session's event stream, not
/// from the session list. Enter loads the child session if its ID is known.
pub(crate) fn apply_delegate_popup_key(
    app: &mut App,
    key: crossterm::event::KeyCode,
) -> SessionKeyAction {
    match key {
        KeyCode::Esc => {
            app.navigation.popup = Popup::None;
        }
        KeyCode::Up => app.delegates.move_cursor_up(),
        KeyCode::Down => app.delegates.move_cursor_down(),
        KeyCode::PageUp => {
            let step = app.render.delegate_popup_page_step();
            app.delegates.move_cursor_page(step, false);
        }
        KeyCode::PageDown => {
            let step = app.render.delegate_popup_page_step();
            app.delegates.move_cursor_page(step, true);
        }
        KeyCode::Enter => {
            let selected = app.delegates.selected_entry().map(|entry| {
                (
                    entry.child_session_id.clone(),
                    entry.target_agent_id.clone(),
                )
            });
            if let Some((child_session_id, target_agent_id)) = selected {
                if let Some(sid) = child_session_id {
                    // Use the real parent when navigating between siblings.
                    app.delegates.stage_parent_for_child_navigation(
                        app.delegates.parent_session_id.clone(),
                        app.sessions.session_id.clone(),
                    );
                    app.navigation.popup = Popup::None;
                    return SessionKeyAction::LoadSession {
                        session_id: sid,
                        agent_id: target_agent_id,
                        cwd: app.current_session_cwd(),
                    };
                } else {
                    app.set_status(
                        LogLevel::Warn,
                        "delegates",
                        "delegation still pending — no session to load",
                    );
                }
            }
        }
        KeyCode::Backspace => app.delegates.filter_backspace(),
        KeyCode::Char(c) => app.delegates.filter_insert(c),
        _ => {}
    }
    SessionKeyAction::None
}

fn begin_fork_session(app: &mut App, message_id: String) -> Vec<Effect> {
    if app.chat.pending_fork_message_id.is_some() {
        app.set_status(LogLevel::Warn, "fork", "fork already pending");
        return Vec::new();
    }
    if app.chat.is_turn_active() {
        app.set_status(LogLevel::Warn, "fork", "cannot fork while agent is active");
        return Vec::new();
    }
    if app.chat.has_pending_session_op() {
        app.set_status(LogLevel::Warn, "fork", "session operation already pending");
        return Vec::new();
    }
    if message_id.is_empty() {
        app.set_status(LogLevel::Warn, "fork", "selected turn has no message id");
        return Vec::new();
    }

    app.chat.pending_fork_message_id = Some(message_id.clone());
    app.set_status(LogLevel::Info, "fork", "forking session...");
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
            app.set_status(LogLevel::Warn, "fork", "no forkable turns");
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.chat.fork_filter_insert(c);
        }
        _ => {}
    }
    Vec::new()
}

pub(crate) fn handle_log_popup_key(app: &mut App, key: KeyEvent) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => {
            app.navigation.popup = Popup::None;
        }
        KeyCode::Up => {
            app.diagnostics.log_cursor = app.diagnostics.log_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            let max = app.filtered_logs().len().saturating_sub(1);
            app.diagnostics.log_cursor = (app.diagnostics.log_cursor + 1).min(max);
        }
        KeyCode::PageUp => {
            app.diagnostics.log_cursor = app.diagnostics.log_cursor.saturating_sub(10);
        }
        KeyCode::PageDown => {
            let max = app.filtered_logs().len().saturating_sub(1);
            app.diagnostics.log_cursor = (app.diagnostics.log_cursor + 10).min(max);
        }
        KeyCode::Home => {
            app.diagnostics.log_cursor = 0;
        }
        KeyCode::End => {
            app.diagnostics.log_cursor = app.filtered_logs().len().saturating_sub(1);
        }
        KeyCode::Backspace => {
            app.diagnostics.log_filter.pop();
            app.diagnostics.log_cursor = app.filtered_logs().len().saturating_sub(1);
        }
        KeyCode::Tab => {
            app.cycle_log_level_filter();
            app.diagnostics.log_cursor = app.filtered_logs().len().saturating_sub(1);
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.diagnostics.log_filter.push(c);
            app.diagnostics.log_cursor = 0;
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_profile_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let mut effects = Vec::new();
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Up => app.profiles.move_profile_cursor(-1),
        KeyCode::Down => app.profiles.move_profile_cursor(1),
        KeyCode::Backspace => {
            app.profiles.profile_filter.pop();
            app.profiles.profile_cursor = 0;
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.profiles.profile_filter.push(c);
            app.profiles.profile_cursor = 0;
        }
        KeyCode::Enter => {
            if !can_send_server_commands(app) {
                return effects;
            }
            if let Some(profile_id) = app
                .profiles
                .selected_profile()
                .map(|profile| profile.id.clone())
            {
                app.profiles.active_profile_id = Some(profile_id.clone());
                if app.current_session_profile_id().is_none() {
                    app.models.clear_profile_agents();
                    effects.push(Effect::Command(Command::ListProfileAgents {
                        profile_id: profile_id.clone(),
                    }));
                }
                app.navigation.popup = Popup::None;
                app.set_status(
                    LogLevel::Info,
                    "profile",
                    format!("new sessions will use {profile_id}"),
                );
                effects.push(Effect::PersistConfig);
            } else {
                app.set_status(LogLevel::Warn, "profile", "no matching profile");
            }
        }
        _ => {}
    }
    effects
}

pub(crate) fn handle_new_session_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Up => app.sessions.move_new_session_completion_selection(-1),
        KeyCode::Down => app.sessions.move_new_session_completion_selection(1),
        KeyCode::Tab => {
            app.accept_selected_new_session_completion();
        }
        KeyCode::Left => {
            app.sessions.move_new_session_cursor_left();
            app.refresh_new_session_completion();
        }
        KeyCode::Right => {
            app.sessions.move_new_session_cursor_right();
            app.refresh_new_session_completion();
        }
        KeyCode::Home => {
            app.sessions.move_new_session_cursor_home();
            app.refresh_new_session_completion();
        }
        KeyCode::End => {
            app.sessions.move_new_session_cursor_end();
            app.refresh_new_session_completion();
        }
        KeyCode::Backspace => {
            app.sessions.new_session_backspace();
            app.refresh_new_session_completion();
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.sessions.new_session_insert(c);
            app.refresh_new_session_completion();
        }
        KeyCode::Enter => {
            if !can_send_server_commands(app) {
                return Vec::new();
            }
            let cwd = app.normalize_new_session_path(&app.sessions.new_session_path);
            app.navigation.popup = Popup::None;
            return vec![Effect::Command(Command::NewSession {
                cwd,
                profile_id: app.profiles.active_profile_id.clone(),
            })];
        }
        _ => {}
    }
    Vec::new()
}

pub(crate) fn handle_theme_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Up => app.navigation.move_theme_cursor_up(),
        KeyCode::Down => {
            let filtered_len = app
                .navigation
                .filtered_themes(theme::Theme::available_themes())
                .len();
            app.navigation.move_theme_cursor_down(filtered_len);
        }
        KeyCode::Enter => {
            if let Some(idx) = app
                .navigation
                .selected_theme_index(theme::Theme::available_themes())
            {
                theme::Theme::set_by_index(idx);
                theme::Theme::begin_frame();
                app.render.apply_change(RenderChange::ThemeChanged);
                app.navigation.popup = Popup::None;
                return vec![Effect::PersistConfig];
            }
        }
        KeyCode::Backspace => app.navigation.theme_filter_backspace(),
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.navigation.theme_filter_insert(c);
        }
        _ => {}
    }
    Vec::new()
}

pub(crate) fn handle_chat_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let input_blocked = app.chat.input_blocked_by_activity();
    let mut effects = Vec::new();
    match key.code {
        KeyCode::Esc => {
            if handle_completion_key(&mut app.composer, key, !input_blocked)
                == CompletionResult::Dismissed
            {
                app.chat.clear_cancel_confirm();
                return effects;
            }
            if app.chat.has_cancellable_activity() {
                if app.chat.cancel_confirm_active() {
                    app.chat.clear_cancel_confirm();
                    app.set_status(LogLevel::Warn, "activity", "stopping...");
                    effects.push(Effect::Command(Command::CancelSession));
                } else {
                    app.arm_cancel_confirm();
                }
            } else {
                app.chat.clear_cancel_confirm();
            }
        }
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
                let (text, links) = app
                    .composer
                    .build_prompt_text_and_links(&app.composer.input);
                let text = text.trim().to_string();
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
        KeyCode::PageUp => app.render.scroll_chat_up(10),
        KeyCode::PageDown => app.render.scroll_chat_down(10),
        KeyCode::Backspace | KeyCode::Delete | KeyCode::Left | KeyCode::Right | KeyCode::Home => {
            handle_composer_key(&mut app.composer, &mut app.render, key, !input_blocked);
        }
        KeyCode::End
            if handle_composer_key(&mut app.composer, &mut app.render, key, !input_blocked)
                == ComposerKeyResult::NotHandled =>
        {
            app.render.scroll_chat_to_bottom();
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
                app.set_status(
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
                            app.set_status(
                                LogLevel::Info,
                                "mode",
                                format!("already in {} mode", arg),
                            );
                        } else {
                            effects.extend(switch_mode(app, &arg));
                        }
                    }
                    _ => app.set_status(
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
                app.set_status(LogLevel::Info, "mode", "already in review mode");
            } else {
                effects.extend(switch_mode(app, "review"));
            }
        }
        "thinking" => {
            app.take_input();
            if arg.is_empty() {
                app.set_status(
                    LogLevel::Info,
                    "model",
                    format!("thinking: {}", app.models.reasoning_effort_label()),
                );
            } else {
                let level = arg.to_lowercase();
                if crate::models_state::validate_reasoning_effort(Some(&level)).is_none() {
                    app.set_status(
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
                    app.set_status(
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
                app.set_status(
                    LogLevel::Info,
                    "profile",
                    format!("new sessions will use {profile_id}"),
                );
                effects.push(Effect::PersistConfig);
            } else {
                app.set_status(
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
                app.set_status(
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
                app.set_status(LogLevel::Warn, "fork", "forking is only available in chat");
                return (SlashResult::Handled, effects);
            }
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if app.chat.pending_fork_message_id.is_some() {
                app.set_status(LogLevel::Warn, "fork", "fork already pending");
            } else if app.chat.has_pending_session_op() {
                app.set_status(LogLevel::Warn, "fork", "session operation already pending");
            } else if app.chat.is_turn_active() {
                app.set_status(LogLevel::Warn, "fork", "cannot fork while agent is active");
            } else if let Some(turn) = app.chat.latest_fork_boundary() {
                effects.extend(begin_fork_session(app, turn.message_id));
            } else {
                app.set_status(LogLevel::Warn, "fork", "no forkable turns");
            }
        }
        "undo" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if app.chat.is_turn_active() {
                app.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot undo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.set_status(LogLevel::Warn, "session", "undo already pending");
            } else if let Some(turn) = app.chat.current_undo_target().cloned() {
                if app.composer.input.trim().is_empty() && !turn.text.is_empty() {
                    app.composer.replace_input(turn.text.clone());
                    app.render.reset_composer_input_geometry();
                }
                app.chat.push_pending_undo(&turn);
                app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
                app.set_status(LogLevel::Info, "session", "undoing...");
                effects.push(Effect::Command(Command::Undo {
                    message_id: turn.message_id,
                }));
            } else {
                app.set_status(LogLevel::Warn, "session", "nothing to undo");
            }
        }
        "redo" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return (SlashResult::Handled, effects);
            }
            if app.chat.is_turn_active() {
                app.set_status(
                    LogLevel::Warn,
                    "session",
                    "cannot redo while agent is active",
                );
            } else if app.chat.has_pending_session_op() || app.chat.has_pending_undo() {
                app.set_status(LogLevel::Warn, "session", "redo already pending");
            } else if app.chat.can_redo() {
                app.chat.activity = ActivityState::SessionOp(SessionOp::Redo);
                app.set_status(LogLevel::Info, "session", "redoing...");
                effects.push(Effect::Command(Command::Redo));
            } else {
                app.set_status(LogLevel::Warn, "session", "nothing to redo");
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
                app.set_status(LogLevel::Warn, "activity", "stopping...");
                effects.push(Effect::Command(Command::CancelSession));
            } else {
                app.set_status(LogLevel::Warn, "activity", "nothing to cancel");
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

    let mut effects = Vec::new();
    match app.auth.panel {
        AuthPanel::List => match key.code {
            KeyCode::Esc => {
                if app.auth.selected.is_some() {
                    app.auth.close_detail();
                } else {
                    app.navigation.popup = Popup::None;
                }
            }
            KeyCode::Up => {
                let max = app.auth.filtered_providers().len().saturating_sub(1);
                app.auth.cursor = app.auth.cursor.saturating_sub(1).min(max);
            }
            KeyCode::Down => {
                let max = app.auth.filtered_providers().len().saturating_sub(1);
                app.auth.cursor = (app.auth.cursor + 1).min(max);
            }
            KeyCode::Enter => {
                let filtered = app.auth.filtered_providers();
                if let Some(&(real_idx, _)) = filtered.get(app.auth.cursor) {
                    let provider = &app.auth.providers[real_idx];
                    app.auth.last_result = None;
                    app.auth.ui_notice = None;
                    if provider.is_unconfigurable() {
                        app.auth.selected = Some(real_idx);
                    } else if provider.is_api_key_only() {
                        app.auth.selected = Some(real_idx);
                        app.auth.panel = AuthPanel::ApiKeyInput;
                        app.auth.api_key_input.clear();
                        app.auth.api_key_cursor = 0;
                    } else if provider.is_oauth_only()
                        && provider.oauth_status != Some(OAuthStatus::Connected)
                    {
                        app.auth.selected = Some(real_idx);
                        effects.push(Effect::Command(Command::StartOAuthLogin {
                            provider: provider.provider.clone(),
                        }));
                    } else {
                        app.auth.selected = Some(real_idx);
                    }
                }
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(idx) = app.auth.selected {
                    let provider = &app.auth.providers[idx];
                    if provider.oauth_status == Some(OAuthStatus::Connected) {
                        effects.push(Effect::Command(Command::DisconnectOAuth {
                            provider: provider.provider.clone(),
                        }));
                    } else if provider.has_stored_api_key {
                        effects.push(Effect::Command(Command::ClearApiToken {
                            provider: provider.provider.clone(),
                        }));
                    }
                }
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let filtered = app.auth.filtered_providers();
                if let Some(&(real_idx, _)) = filtered.get(app.auth.cursor) {
                    let provider = &app.auth.providers[real_idx];
                    if provider.env_var_name.is_some() || provider.has_stored_api_key {
                        app.auth.ui_notice = None;
                        app.auth.selected = Some(real_idx);
                        app.auth.panel = AuthPanel::ApiKeyInput;
                        app.auth.api_key_input.clear();
                        app.auth.api_key_cursor = 0;
                    }
                }
            }
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                let filtered = app.auth.filtered_providers();
                if let Some(&(real_idx, _)) = filtered.get(app.auth.cursor) {
                    let provider = &app.auth.providers[real_idx];
                    if provider.supports_oauth {
                        app.auth.ui_notice = None;
                        app.auth.selected = Some(real_idx);
                        effects.push(Effect::Command(Command::StartOAuthLogin {
                            provider: provider.provider.clone(),
                        }));
                    }
                }
            }
            KeyCode::Backspace => {
                app.auth.filter.pop();
                app.auth.cursor = 0;
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.auth.filter.push(c);
                app.auth.cursor = 0;
            }
            _ => {}
        },
        AuthPanel::ApiKeyInput => match key.code {
            KeyCode::Esc => {
                app.auth.panel = AuthPanel::List;
                app.auth.api_key_input.clear();
                app.auth.api_key_cursor = 0;
            }
            KeyCode::Enter => {
                if let Some(idx) = app.auth.selected {
                    let trimmed = app.auth.api_key_input.trim().to_string();
                    if !trimmed.is_empty() {
                        effects.push(Effect::Command(Command::SetApiToken {
                            provider: app.auth.providers[idx].provider.clone(),
                            api_key: trimmed,
                        }));
                    }
                }
            }
            KeyCode::Tab => app.auth.api_key_masked = !app.auth.api_key_masked,
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(idx) = app.auth.selected {
                    effects.push(Effect::Command(Command::ClearApiToken {
                        provider: app.auth.providers[idx].provider.clone(),
                    }));
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.auth.api_key_input.insert(app.auth.api_key_cursor, c);
                app.auth.api_key_cursor += c.len_utf8();
            }
            KeyCode::Backspace if app.auth.api_key_cursor > 0 => {
                let cursor = app.auth.api_key_cursor;
                let ch = app.auth.api_key_input[..cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth.api_key_input.remove(cursor - ch.len_utf8());
                app.auth.api_key_cursor -= ch.len_utf8();
            }
            KeyCode::Left if app.auth.api_key_cursor > 0 => {
                let ch = app.auth.api_key_input[..app.auth.api_key_cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth.api_key_cursor -= ch.len_utf8();
            }
            KeyCode::Right if app.auth.api_key_cursor < app.auth.api_key_input.len() => {
                let ch = app.auth.api_key_input[app.auth.api_key_cursor..]
                    .chars()
                    .next()
                    .unwrap();
                app.auth.api_key_cursor += ch.len_utf8();
            }
            _ => {}
        },
        AuthPanel::OAuthFlow => match key.code {
            KeyCode::Esc => {
                app.auth.oauth_flow = None;
                app.auth.panel = AuthPanel::List;
                app.auth.oauth_response.clear();
                app.auth.oauth_response_cursor = 0;
            }
            KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(ref flow) = app.auth.oauth_flow {
                    app.auth.ui_notice = None;
                    app.auth.clipboard_fallback = None;
                    effects.push(Effect::CopyToClipboard {
                        target: ClipboardTarget::Auth {
                            provider: flow.provider.clone(),
                        },
                        text: flow.authorization_url.clone(),
                    });
                }
            }
            KeyCode::Enter => {
                if let Some(ref flow) = app.auth.oauth_flow {
                    let is_device_poll = flow.flow_kind == OAuthFlowKind::DevicePoll;
                    let response = if is_device_poll {
                        String::new()
                    } else {
                        app.auth.oauth_response.trim().to_string()
                    };
                    if is_device_poll || !response.is_empty() {
                        effects.push(Effect::Command(Command::CompleteOAuthLogin {
                            flow_id: flow.flow_id.clone(),
                            response,
                        }));
                    }
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.auth
                    .oauth_response
                    .insert(app.auth.oauth_response_cursor, c);
                app.auth.oauth_response_cursor += c.len_utf8();
            }
            KeyCode::Backspace if app.auth.oauth_response_cursor > 0 => {
                let cursor = app.auth.oauth_response_cursor;
                let ch = app.auth.oauth_response[..cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth.oauth_response.remove(cursor - ch.len_utf8());
                app.auth.oauth_response_cursor -= ch.len_utf8();
            }
            KeyCode::Left if app.auth.oauth_response_cursor > 0 => {
                let ch = app.auth.oauth_response[..app.auth.oauth_response_cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth.oauth_response_cursor -= ch.len_utf8();
            }
            KeyCode::Right if app.auth.oauth_response_cursor < app.auth.oauth_response.len() => {
                let ch = app.auth.oauth_response[app.auth.oauth_response_cursor..]
                    .chars()
                    .next()
                    .unwrap();
                app.auth.oauth_response_cursor += ch.len_utf8();
            }
            _ => {}
        },
    }
    effects
}

pub(crate) fn handle_model_popup_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    let mut effects = Vec::new();
    match key.code {
        KeyCode::Esc => app.navigation.popup = Popup::None,
        KeyCode::Tab | KeyCode::BackTab => {
            if !app.models.model_popup_has_tabs() {
                return effects;
            }
            let agent_id = app
                .models
                .switch_model_popup_tab(key.code == KeyCode::BackTab);
            app.models.model_cursor = if app
                .models
                .model_popup_is_session_tab(app.models.model_popup_agent_tab)
            {
                app.models.model_popup_open_cursor()
            } else if let Some(agent_id) = agent_id {
                app.delegate_model_cursor(&agent_id)
            } else {
                0
            };
        }
        KeyCode::Up => app.models.move_model_cursor_up(),
        KeyCode::Down => app.models.move_model_cursor_down(),
        KeyCode::Enter => {
            let selected: Option<ModelEntry> = app
                .models
                .visible_model_popup_items()
                .get(app.models.model_cursor)
                .and_then(|item| match item {
                    crate::models_state::ModelPopupItem::Model { model_idx } => {
                        app.models.models.get(*model_idx)
                    }
                    crate::models_state::ModelPopupItem::ProviderHeader { .. } => None,
                })
                .cloned();
            if let Some(model) = selected {
                let tab_label = app
                    .models
                    .model_popup_tab_label(app.models.model_popup_agent_tab)
                    .to_string();
                if app
                    .models
                    .model_popup_is_session_tab(app.models.model_popup_agent_tab)
                {
                    if !app.sessions.current_session_is_remote() {
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
                    app.set_status(LogLevel::Info, "model", format!("session: {}", model.label));
                } else if let Some(agent_id) = app
                    .models
                    .model_popup_tab_agent_id(app.models.model_popup_agent_tab)
                    .map(str::to_string)
                    && let Some(profile_id) =
                        app.delegate_preference_profile_id().map(str::to_string)
                {
                    app.models
                        .set_delegate_model_preference(&profile_id, &agent_id, &model);
                    if app.delegates.parent_session_id.is_none()
                        && let Some(session_id) = app.sessions.session_id.clone()
                    {
                        effects.push(Effect::Command(Command::SetDelegateModel {
                            session_id,
                            agent_id,
                            model_id: Some(model.id.clone()),
                            node_id: model.node_id.clone(),
                        }));
                    }
                    app.set_status(
                        LogLevel::Info,
                        "model",
                        format!("{tab_label}: {}", model.label),
                    );
                }
                effects.push(Effect::PersistConfig);
            }
        }
        KeyCode::Delete
            if !app
                .models
                .model_popup_is_session_tab(app.models.model_popup_agent_tab) =>
        {
            if let Some(agent_id) = app
                .models
                .model_popup_tab_agent_id(app.models.model_popup_agent_tab)
                .map(str::to_string)
                && let Some(profile_id) = app.delegate_preference_profile_id().map(str::to_string)
            {
                app.models
                    .clear_delegate_model_preference(&profile_id, &agent_id);
                if app.delegates.parent_session_id.is_none()
                    && let Some(session_id) = app.sessions.session_id.clone()
                {
                    effects.push(Effect::Command(Command::SetDelegateModel {
                        session_id,
                        agent_id,
                        model_id: None,
                        node_id: None,
                    }));
                }
                app.set_status(
                    LogLevel::Info,
                    "model",
                    "delegate model uses profile default",
                );
                effects.push(Effect::PersistConfig);
            }
        }
        KeyCode::Backspace => app.models.model_filter_backspace(),
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.models.model_filter_insert(c);
        }
        _ => {}
    }
    effects
}

// ── Pure key logic for the sessions screen ────────────────────────────────────
//
// `apply_sessions_key` returns the `Command`(s) that should be sent to the
// server (if any).  Keeping the mutation separate from the channel send makes
// it fully unit-testable without a real channel.

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

/// Apply a key event on the sessions screen, mutate `app`, and return the
/// action (if any) that the caller should forward to the server.
pub(crate) fn apply_sessions_key(
    app: &mut App,
    key: crossterm::event::KeyCode,
) -> SessionKeyAction {
    use crate::session_state::StartPageItem;

    match key {
        KeyCode::Up => {
            app.sessions.move_start_cursor_up();
            app.render
                .keep_start_page_cursor_visible_from_above(app.sessions.session_cursor);
        }
        KeyCode::Down => app.sessions.move_start_cursor_down(),
        KeyCode::Enter => {
            let items = app.sessions.visible_start_items();
            // Cursor on the button slot (one past the last item)?
            if app.sessions.session_cursor == items.len() {
                return SessionKeyAction::NewSession;
            }
            if let Some(item) = items.get(app.sessions.session_cursor).cloned() {
                match item {
                    StartPageItem::GroupHeader { cwd, .. } => {
                        app.sessions.toggle_group_collapse(cwd.as_deref());
                        app.sessions.clamp_start_cursor();
                    }
                    StartPageItem::Session {
                        group_idx, path, ..
                    } => {
                        if let Some(session) =
                            app.sessions.session_by_path(group_idx, &path).cloned()
                        {
                            let session_id = session.session_id;
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
                                app.set_status(
                                    LogLevel::Warn,
                                    "session",
                                    "remote session is missing node id; refresh sessions and try again",
                                );
                                return SessionKeyAction::None;
                            }
                            return SessionKeyAction::LoadSession {
                                session_id,
                                agent_id: None,
                                cwd: session
                                    .cwd
                                    .or_else(|| app.sessions.session_groups[group_idx].cwd.clone()),
                            };
                        }
                    }
                    StartPageItem::ShowMore { .. } => {
                        app.navigation.popup = Popup::SessionSelect;
                        app.sessions.reset_browser_for_open();
                    }
                }
            }
        }
        KeyCode::Delete => {
            let items = app.sessions.visible_start_items();
            if let Some(StartPageItem::Session {
                group_idx, path, ..
            }) = items.get(app.sessions.session_cursor).cloned()
            {
                let Some((session, is_remote)) =
                    app.sessions.remove_session_at(group_idx, &path, false)
                else {
                    return SessionKeyAction::None;
                };
                let sid = session.session_id;
                return if is_remote {
                    SessionKeyAction::DismissRemoteSession { session_id: sid }
                } else {
                    SessionKeyAction::DeleteSession { session_id: sid }
                };
            }
            // Delete on a GroupHeader: no-op
        }
        KeyCode::Backspace => {
            app.sessions.start_filter_backspace();
            app.render.reset_start_page_scroll();
        }
        KeyCode::Char(c) => {
            app.sessions.start_filter_insert(c);
            app.render.reset_start_page_scroll();
        }
        _ => {}
    }
    SessionKeyAction::None
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
        app.set_status(LogLevel::Debug, "test", "before disconnected guard");
        assert!(!can_send_server_commands(&mut app));
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );

        app.connection.conn = ConnState::Connected;
        app.set_status(LogLevel::Debug, "test", "retained");
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

        handle_theme_popup_key(&mut app, key(KeyCode::Down));
        assert_eq!(app.navigation.theme_cursor, 0);
        handle_theme_popup_key(&mut app, key(KeyCode::Enter));
        assert_eq!(app.navigation.popup, Popup::ThemeSelect);
    }

    #[test]
    fn empty_theme_filter_out_of_range_enter_still_closes() {
        let _guard = TestPersistenceGuard::new("theme-out-of-range");
        let mut app = App::new();
        app.navigation.popup = Popup::ThemeSelect;
        app.navigation.theme_cursor = theme::Theme::available_themes().len() + 10;

        handle_theme_popup_key(&mut app, key(KeyCode::Enter));

        assert_eq!(app.navigation.popup, Popup::None);
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
    fn mesh_popup_enter_on_session_attaches_remote_session() {
        let mut app = App::new();
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
    fn mesh_popup_create_remote_session_does_not_forward_local_cwd() {
        let mut app = App::new();
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
        let mut effects = TestEffects::default();
        effects.extend(handle_profile_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(app.navigation.popup, Popup::None));
        assert!(matches!(
            effects.next_command(),
            Some(Command::ListProfileAgents { profile_id }) if profile_id == "deep"
        ));
        assert!(effects.next_command().is_none());
        assert_eq!(app.profiles.active_profile_id.as_deref(), Some("deep"));
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::PersistConfig))
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
        app.sessions.new_session_path = "/repo".into();
        app.sessions.new_session_cursor = app.sessions.new_session_path.len();
        app.profiles.active_profile_id = Some("coder-delegate".into());
        let mut effects = TestEffects::default();
        effects.extend(handle_new_session_popup_key(&mut app, key(KeyCode::Enter)));

        assert!(matches!(
            effects.next_command(),
            Some(Command::NewSession { profile_id: Some(profile_id), .. })
                if profile_id == "coder-delegate"
        ));
    }

    #[test]
    fn select_model_on_delegate_tab_saves_pref_not_session_model() {
        let _guard = TestPersistenceGuard::new("delegate-tab");
        let mut app = App::new();
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
        let mut effects = TestEffects::default();
        effects.extend(handle_model_popup_key(&mut app, key(KeyCode::Enter)));
        assert!(effects.next_command().is_none());
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
