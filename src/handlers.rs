use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use tokio::sync::mpsc;

use crate::app::{self, App, AuthUiNotice, CommandPaletteAction, LogLevel, Popup, Screen};
use crate::domain::activity::{ActivityState, SessionOp};
use crate::domain::auth::{OAuthFlowKind, OAuthStatus};
use crate::domain::chat::{ChatEntry, format_outcome_labels};
use crate::domain::model::ModelEntry;

fn popup_page_step(visible_rows: usize) -> usize {
    visible_rows.saturating_sub(1).max(1)
}
use crate::command::{Command, PromptBlock};
use crate::config;
use crate::theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppAction {
    None,
    OpenExternalEditor,
}

pub(crate) fn can_send_server_commands(app: &mut App) -> bool {
    if app.conn == app::ConnState::Connected {
        true
    } else {
        app.set_status(
            app::LogLevel::Warn,
            "connection",
            "not connected - waiting to reconnect",
        );
        false
    }
}

fn send_load_session_commands(
    cmd_tx: &mpsc::UnboundedSender<Command>,
    session_id: String,
    cwd: Option<String>,
    agent_id: Option<String>,
) -> anyhow::Result<()> {
    for command in Command::load_session_commands(session_id, cwd, agent_id) {
        cmd_tx.send(command)?;
    }
    Ok(())
}

/// Handle all keyboard input while an elicitation popup is active.
///
/// Returns `Ok(())` in all cases; the caller should return immediately after
/// this to avoid routing the key to the normal chat handler.
pub(crate) fn handle_elicitation_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    use crate::domain::elicitation::ElicitationFieldKind;

    let (Some(state), Some(ui)) = (app.elicitation.as_mut(), app.elicitation_ui.as_mut()) else {
        return Ok(());
    };
    let Some(field_index) = ui.current_field_index(state.fields.len()) else {
        return Ok(());
    };

    let selected_display = |state: &crate::domain::elicitation::ElicitationState,
                            custom_active: bool| {
        let field = &state.fields[field_index];
        if custom_active {
            return format_outcome_labels([state.custom_input.trim()]);
        }
        match &field.kind {
            ElicitationFieldKind::SingleSelect { options } => options
                .iter()
                .find(|option| state.selected.get(&field.name) == Some(&option.value))
                .map(|option| format_outcome_labels([option.label.as_str()]))
                .unwrap_or_default(),
            ElicitationFieldKind::MultiSelect { options } => state
                .selected
                .get(&field.name)
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    format_outcome_labels(
                        options
                            .iter()
                            .filter(|option| values.contains(&option.value))
                            .map(|option| option.label.as_str()),
                    )
                })
                .unwrap_or_default(),
            ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
                state.text_input.clone()
            }
            ElicitationFieldKind::BooleanToggle => match state
                .selected
                .get(&field.name)
                .and_then(serde_json::Value::as_bool)
            {
                Some(true) => "Yes".into(),
                Some(false) => "No".into(),
                None => String::new(),
            },
        }
    };

    if ui.custom_active {
        match key.code {
            KeyCode::Esc => {
                ui.custom_active = false;
                ui.custom_scroll = 0;
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                ui.custom_insert(&mut state.custom_input, '\n');
            }
            KeyCode::Enter if state.is_valid(Some(&state.custom_input)) => {
                let elicitation_id = state.elicitation_id.clone();
                let content = state.build_accept_content(Some(&state.custom_input));
                let display = selected_display(state, true);
                cmd_tx.send(Command::ElicitationResponse {
                    elicitation_id: elicitation_id.clone(),
                    action: "accept".into(),
                    content: Some(content),
                })?;
                app.resolve_elicitation(&elicitation_id, &display);
            }
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                ui.custom_insert(&mut state.custom_input, character);
            }
            KeyCode::Backspace => ui.custom_backspace(&mut state.custom_input),
            KeyCode::Delete => ui.custom_delete(&mut state.custom_input),
            KeyCode::Left => ui.custom_left(&state.custom_input),
            KeyCode::Right => ui.custom_right(&state.custom_input),
            KeyCode::Home => ui.custom_home(&state.custom_input),
            KeyCode::End => ui.custom_end(&state.custom_input),
            KeyCode::Up => ui.custom_move_visual(&state.custom_input, -1),
            KeyCode::Down => ui.custom_move_visual(&state.custom_input, 1),
            _ => {}
        }
        return Ok(());
    }

    let field = &state.fields[field_index];
    let option_count = match &field.kind {
        ElicitationFieldKind::SingleSelect { options }
        | ElicitationFieldKind::MultiSelect { options } => options.len(),
        _ => 0,
    };
    let custom_available = state.allow_custom && option_count > 0;
    let custom_option_selected = custom_available && ui.option_cursor == option_count;

    match key.code {
        KeyCode::Esc => {
            let elicitation_id = state.elicitation_id.clone();
            cmd_tx.send(Command::ElicitationResponse {
                elicitation_id: elicitation_id.clone(),
                action: "decline".into(),
                content: None,
            })?;
            app.resolve_elicitation(&elicitation_id, "declined");
        }
        KeyCode::Down => {
            let max = option_count + usize::from(custom_available);
            ui.option_cursor = (ui.option_cursor + 1).min(max.saturating_sub(1));
        }
        KeyCode::Up => ui.option_cursor = ui.option_cursor.saturating_sub(1),
        KeyCode::Char(' ') => {
            if matches!(
                field.kind,
                ElicitationFieldKind::MultiSelect { .. } | ElicitationFieldKind::BooleanToggle
            ) {
                state.toggle_option(field_index, ui.option_cursor);
            }
        }
        KeyCode::Enter => {
            match field.kind {
                ElicitationFieldKind::SingleSelect { .. } if custom_option_selected => {
                    ui.custom_active = true;
                    ui.custom_cursor = state.custom_input.len();
                    state.clear_selection(field_index);
                    return Ok(());
                }
                ElicitationFieldKind::SingleSelect { .. } => {
                    state.select_option(field_index, ui.option_cursor);
                }
                ElicitationFieldKind::MultiSelect { .. } if custom_option_selected => {
                    ui.custom_active = true;
                    ui.custom_cursor = state.custom_input.len();
                    state.clear_selection(field_index);
                    return Ok(());
                }
                ElicitationFieldKind::MultiSelect { .. }
                | ElicitationFieldKind::TextInput
                | ElicitationFieldKind::NumberInput { .. }
                | ElicitationFieldKind::BooleanToggle => {}
            }

            if state.is_valid(None) {
                let elicitation_id = state.elicitation_id.clone();
                let content = state.build_accept_content(None);
                let display = selected_display(state, false);
                cmd_tx.send(Command::ElicitationResponse {
                    elicitation_id: elicitation_id.clone(),
                    action: "accept".into(),
                    content: Some(content),
                })?;
                app.resolve_elicitation(&elicitation_id, &display);
            }
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            if matches!(
                field.kind,
                ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. }
            ) {
                state.text_input.insert(ui.text_cursor, character);
                ui.text_cursor += character.len_utf8();
            }
        }
        KeyCode::Backspace
            if matches!(
                field.kind,
                ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. }
            ) && ui.text_cursor > 0 =>
        {
            let previous = state.text_input[..ui.text_cursor]
                .char_indices()
                .last()
                .map(|(index, _)| index)
                .unwrap_or(0);
            state.text_input.drain(previous..ui.text_cursor);
            ui.text_cursor = previous;
        }
        _ => {}
    }
    Ok(())
}

fn open_model_popup(app: &mut App, cmd_tx: &mpsc::UnboundedSender<Command>) -> anyhow::Result<()> {
    if app.screen != Screen::Chat {
        app.set_status(
            app::LogLevel::Warn,
            "model",
            "model select is only available in chat",
        );
        return Ok(());
    }
    if !can_send_server_commands(app) {
        return Ok(());
    }
    app.popup = Popup::ModelSelect;
    app.model_filter.clear();
    app.model_popup_agent_tab = 0;
    app.model_cursor = app.model_popup_open_cursor();
    cmd_tx.send(Command::ListAllModels { refresh: true })?;
    Ok(())
}

fn open_session_popup(
    app: &mut App,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    if !can_send_server_commands(app) {
        return Ok(());
    }
    app.popup = Popup::SessionSelect;
    app.session_popup_tab = 0;
    app.session_cursor = 0;
    app.session_filter.clear();
    if let Some(request) = app.begin_session_discovery() {
        cmd_tx.send(request)?;
    }
    Ok(())
}

fn open_log_popup(app: &mut App) {
    app.popup = Popup::Log;
    app.log_cursor = app.filtered_logs().len().saturating_sub(1);
    app.log_filter.clear();
}

fn execute_command_palette_action(
    app: &mut App,
    action: CommandPaletteAction,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match action {
        CommandPaletteAction::OpenMesh | CommandPaletteAction::AttachRemoteSession => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_mesh_popup();
            cmd_tx.send(Command::ListRemoteNodes)?;
        }
        CommandPaletteAction::CreateRemoteSession => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_mesh_popup();
            cmd_tx.send(Command::ListRemoteNodes)?;
        }
        CommandPaletteAction::CreateMeshInvite => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_mesh_invite_form();
        }
        CommandPaletteAction::ModelSelect => open_model_popup(app, cmd_tx)?,
        CommandPaletteAction::SessionSelect => open_session_popup(app, cmd_tx)?,
        CommandPaletteAction::DelegateSessions => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_delegate_popup();
        }
        CommandPaletteAction::NewSession => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_new_session_popup();
        }
        CommandPaletteAction::ThemeSelect => {
            app.popup = Popup::ThemeSelect;
            app.theme_filter.clear();
            app.theme_cursor = theme::Theme::current_index();
        }
        CommandPaletteAction::Help => {
            app.popup = Popup::Help;
            app.help_scroll = 0;
        }
        CommandPaletteAction::Log => open_log_popup(app),
        CommandPaletteAction::ProviderAuth => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_auth_popup();
            cmd_tx.send(Command::ListAuthProviders)?;
        }
        CommandPaletteAction::ForkTurnSelect => {
            app.open_fork_turn_popup();
        }
        CommandPaletteAction::ProfileSelect => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_profile_popup();
            cmd_tx.send(Command::ListProfiles)?;
        }
    }
    Ok(())
}

pub(crate) fn handle_mesh_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => app.popup = Popup::None,
        KeyCode::Tab | KeyCode::Right | KeyCode::Left => {
            app.mesh_focus = match app.mesh_focus {
                crate::mesh::MeshFocus::Nodes => crate::mesh::MeshFocus::Sessions,
                crate::mesh::MeshFocus::Sessions => crate::mesh::MeshFocus::Nodes,
            };
        }
        KeyCode::Up => match app.mesh_focus {
            crate::mesh::MeshFocus::Nodes => {
                if let Some(msg) = app.move_mesh_node_cursor(-1) {
                    cmd_tx.send(msg)?;
                }
            }
            crate::mesh::MeshFocus::Sessions => app.move_remote_session_cursor(-1),
        },
        KeyCode::Down => match app.mesh_focus {
            crate::mesh::MeshFocus::Nodes => {
                if let Some(msg) = app.move_mesh_node_cursor(1) {
                    cmd_tx.send(msg)?;
                }
            }
            crate::mesh::MeshFocus::Sessions => app.move_remote_session_cursor(1),
        },
        KeyCode::Char('r') => cmd_tx.send(Command::ListRemoteNodes)?,
        KeyCode::Enter => match app.mesh_focus {
            crate::mesh::MeshFocus::Nodes => {
                app.mesh_focus = crate::mesh::MeshFocus::Sessions;
                if let Some(node_id) = app.selected_mesh_node_id() {
                    cmd_tx.send(Command::ListRemoteSessions {
                        node_id: node_id.to_string(),
                        offset: 0,
                        limit: 50,
                    })?;
                }
            }
            crate::mesh::MeshFocus::Sessions => {
                if let Some(session) = app.selected_remote_session() {
                    cmd_tx.send(Command::AttachRemoteSession {
                        node_id: session.node_id.clone(),
                        session_id: session.id.clone(),
                    })?;
                }
            }
        },
        KeyCode::Char('n') => {
            if let Some(node_id) = app.selected_mesh_node_id() {
                cmd_tx.send(Command::CreateRemoteSession {
                    node_id: node_id.to_string(),
                    cwd: app.current_session_cwd(),
                })?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_mesh_invite_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    if app.mesh_clipboard_fallback.is_some() {
        app.mesh_clipboard_fallback = None;
        return Ok(());
    }

    match key.code {
        KeyCode::Esc => app.popup = Popup::None,
        KeyCode::Up => {
            app.mesh_invite_form_field = match app.mesh_invite_form_field {
                crate::mesh::MeshInviteFormField::MeshName => {
                    crate::mesh::MeshInviteFormField::MaxUses
                }
                crate::mesh::MeshInviteFormField::Ttl => crate::mesh::MeshInviteFormField::MeshName,
                crate::mesh::MeshInviteFormField::MaxUses => crate::mesh::MeshInviteFormField::Ttl,
            };
        }
        KeyCode::Down | KeyCode::Tab => {
            app.mesh_invite_form_field = match app.mesh_invite_form_field {
                crate::mesh::MeshInviteFormField::MeshName => crate::mesh::MeshInviteFormField::Ttl,
                crate::mesh::MeshInviteFormField::Ttl => crate::mesh::MeshInviteFormField::MaxUses,
                crate::mesh::MeshInviteFormField::MaxUses => {
                    crate::mesh::MeshInviteFormField::MeshName
                }
            };
        }
        KeyCode::Backspace => match app.mesh_invite_form_field {
            crate::mesh::MeshInviteFormField::MeshName => {
                app.mesh_invite_name.pop();
            }
            crate::mesh::MeshInviteFormField::Ttl => {
                app.mesh_invite_ttl.pop();
            }
            crate::mesh::MeshInviteFormField::MaxUses => {
                app.mesh_invite_max_uses.pop();
            }
        },
        KeyCode::Enter => {
            if let Some(msg) = app.mesh_invite_form_command() {
                cmd_tx.send(msg)?;
                app.set_status(app::LogLevel::Info, "mesh", "creating invite...");
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            match app.mesh_invite_form_field {
                crate::mesh::MeshInviteFormField::MeshName => app.mesh_invite_name.push(c),
                crate::mesh::MeshInviteFormField::Ttl => app.mesh_invite_ttl.push(c),
                crate::mesh::MeshInviteFormField::MaxUses if c.is_ascii_digit() => {
                    app.mesh_invite_max_uses.push(c)
                }
                crate::mesh::MeshInviteFormField::MaxUses => {}
            }
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_mesh_invite_qr_popup_key(app: &mut App, key: KeyEvent) -> anyhow::Result<()> {
    if app.mesh_clipboard_fallback.is_some() {
        app.mesh_clipboard_fallback = None;
        return Ok(());
    }

    match key.code {
        KeyCode::Esc => app.popup = Popup::MeshInvite,
        KeyCode::Char('u') => {
            if let Some(url) = app.mesh_invite_url().map(str::to_string) {
                app.mesh_clipboard_fallback = Some(url);
            }
        }
        KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if let Some(url) = app.mesh_invite_url().map(str::to_string) {
                if copy_text_to_clipboard(&url) {
                    app.set_status(app::LogLevel::Info, "mesh", "invite URL copied");
                } else {
                    app.mesh_clipboard_fallback = Some(url);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_command_palette_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => app.popup = Popup::None,
        KeyCode::Up => app.move_command_palette_cursor(-1),
        KeyCode::Down => app.move_command_palette_cursor(1),
        KeyCode::Backspace => app.command_palette_filter_backspace(),
        KeyCode::Enter => {
            if let Some(action) = app.selected_command_palette_action() {
                execute_command_palette_action(app, action, cmd_tx)?;
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.command_palette_filter_insert(c);
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<AppAction> {
    if key.code != KeyCode::Esc && app.pending_cancel_confirm_until.is_some() {
        app.clear_cancel_confirm();
        app.refresh_transient_status();
    }

    // ctrl-c: clear input first, quit on second press
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if !app.input.is_empty() {
            app.input.clear();
            app.input_cursor = 0;
            app.input_scroll = 0;
        } else {
            app.should_quit = true;
        }
        return Ok(AppAction::None);
    }

    // direct: ctrl+p opens the command palette, replacing any active popup.
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('p') | KeyCode::Char('P'))
    {
        app.chord = false;
        app.open_command_palette();
        return Ok(AppAction::None);
    }

    // chord second key: ctrl+x was pressed, now handle the follow-up
    if app.chord {
        app.chord = false;
        app.set_status(app::LogLevel::Debug, "input", "ready");
        if key.code == KeyCode::Char('e') {
            if app.screen != Screen::Chat {
                app.set_status(
                    app::LogLevel::Warn,
                    "editor",
                    "external editor is only available in chat",
                );
                return Ok(AppAction::None);
            }
            return Ok(AppAction::OpenExternalEditor);
        }
        handle_chord(app, key, cmd_tx)?;
        return Ok(AppAction::None);
    }

    // elicitation popup takes full control of input when active
    if app.elicitation.is_some() {
        handle_elicitation_key(app, key, cmd_tx)?;
        return Ok(AppAction::None);
    }

    // direct: ctrl+t cycles thinking level
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
        if !can_send_server_commands(app) {
            return Ok(AppAction::None);
        }
        match app.cycle_reasoning_effort() {
            Some(msg) => {
                cmd_tx.send(msg)?;
                app.set_status(
                    app::LogLevel::Info,
                    "model",
                    format!("thinking: {}", app.reasoning_effort_label()),
                );
            }
            None => {
                app.set_status(
                    app::LogLevel::Warn,
                    "model",
                    format!(
                        "unknown reasoning effort {:?}; cannot cycle",
                        app.reasoning_effort
                    ),
                );
            }
        }
        return Ok(AppAction::None);
    }

    // direct: ctrl+l opens log popup
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('l') | KeyCode::Char('L'))
    {
        open_log_popup(app);
        return Ok(AppAction::None);
    }

    // chord start: ctrl+x
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('x') {
        app.chord = true;
        app.set_status(app::LogLevel::Debug, "input", "C-x ...");
        return Ok(AppAction::None);
    }

    // popup handling
    match app.popup {
        Popup::CommandPalette => {
            handle_command_palette_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::Mesh => {
            handle_mesh_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::MeshInvite => {
            handle_mesh_invite_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::MeshInviteQr => {
            handle_mesh_invite_qr_popup_key(app, key)?;
            return Ok(AppAction::None);
        }
        Popup::ModelSelect => {
            handle_model_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::SessionSelect => {
            handle_session_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::NewSession => {
            handle_new_session_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::ThemeSelect => {
            handle_theme_popup_key(app, key)?;
            return Ok(AppAction::None);
        }
        Popup::Help => {
            match key.code {
                KeyCode::Esc => {
                    app.popup = Popup::None;
                }
                KeyCode::Up => {
                    app.help_scroll = app.help_scroll.saturating_sub(1);
                }
                KeyCode::Down => {
                    app.help_scroll = app.help_scroll.saturating_add(1);
                }
                _ => {}
            }
            return Ok(AppAction::None);
        }
        Popup::Log => {
            handle_log_popup_key(app, key)?;
            return Ok(AppAction::None);
        }
        Popup::ProviderAuth => {
            handle_auth_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::ForkTurnSelect => {
            handle_fork_turn_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }
        Popup::ProfileSelect => {
            handle_profile_popup_key(app, key, cmd_tx)?;
            return Ok(AppAction::None);
        }

        Popup::None => {}
    }

    // global: tab toggles mode when no popup is active
    if key.code == KeyCode::Tab {
        if !can_send_server_commands(app) {
            return Ok(AppAction::None);
        }
        switch_mode(app, cmd_tx, &app.next_mode())?;
        return Ok(AppAction::None);
    }

    match app.screen {
        Screen::Sessions => handle_sessions_key(app, key, cmd_tx)?,
        Screen::Chat => return handle_chat_key(app, key, cmd_tx),
        Screen::Delegate => return handle_delegate_view_key(app, key, cmd_tx),
    }
    Ok(AppAction::None)
}

pub(crate) fn handle_mouse(app: &mut App, mouse: MouseEvent) {
    match (mouse.kind, &app.screen, &app.popup) {
        (MouseEventKind::ScrollUp, Screen::Chat | Screen::Delegate, Popup::None) => {
            app.scroll_offset = app.scroll_offset.saturating_add(3);
        }
        (MouseEventKind::ScrollDown, Screen::Chat | Screen::Delegate, Popup::None) => {
            app.scroll_offset = app.scroll_offset.saturating_sub(3);
        }
        _ => {}
    }
}

/// Persist current app state to `~/.qmt/qmtui.toml`.  Called at every
/// user-initiated change that should survive a restart.
pub(crate) fn save_config(app: &App) {
    let merged = config::TuiConfig::load().with_app_settings(app);
    merged.save();
}

/// Handle second key of a ctrl+x chord. Works in any screen.
pub(crate) fn handle_chord(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Char('m') => {
            open_model_popup(app, cmd_tx)?;
        }
        KeyCode::Char('n') => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_new_session_popup();
        }
        KeyCode::Char('q') => {
            app.should_quit = true;
        }
        KeyCode::Char('e') => {
            app.set_status(
                app::LogLevel::Warn,
                "editor",
                "external editor unavailable here",
            );
        }

        KeyCode::Char('t') => {
            app.popup = Popup::ThemeSelect;
            app.theme_filter.clear();
            app.theme_cursor = theme::Theme::current_index();
        }
        KeyCode::Char('l') => {
            open_session_popup(app, cmd_tx)?;
        }
        KeyCode::Char('a') => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_auth_popup();
            cmd_tx.send(Command::ListAuthProviders)?;
        }

        KeyCode::Char('p') => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            app.open_profile_popup();
            cmd_tx.send(Command::ListProfiles)?;
        }
        KeyCode::Char('j') => {
            if !matches!(app.screen, Screen::Chat | Screen::Delegate) {
                app.set_status(
                    app::LogLevel::Warn,
                    "session",
                    "parent jump only available in chat",
                );
                return Ok(());
            }
            if !can_send_server_commands(app) {
                return Ok(());
            }
            if let Some(parent_sid) = app.parent_session_id.clone() {
                send_load_session_commands(
                    cmd_tx,
                    parent_sid,
                    app.current_session_cwd(),
                    app.agent_id.clone(),
                )?;
            } else {
                app.set_status(app::LogLevel::Info, "session", "no parent session");
            }
        }
        KeyCode::Char('?') => {
            app.popup = Popup::Help;
            app.help_scroll = 0;
        }
        KeyCode::Char('f') => {
            if app.screen != Screen::Chat {
                app.set_status(
                    app::LogLevel::Warn,
                    "fork",
                    "fork selector is only available in chat",
                );
                return Ok(());
            }
            app.open_fork_turn_popup();
        }
        KeyCode::Char('u') => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            if app.is_turn_active() {
                app.set_status(
                    app::LogLevel::Warn,
                    "session",
                    "cannot undo while agent is active",
                );
            } else if app.has_pending_session_op() || app.has_pending_undo() {
                app.set_status(app::LogLevel::Warn, "session", "undo already pending");
            } else if let Some(turn) = app.current_undo_target().cloned() {
                if app.input.trim().is_empty() && !turn.text.is_empty() {
                    app.input = turn.text.clone();
                    app.input_cursor = app.input.len();
                    app.input_scroll = 0;
                }
                app.push_pending_undo(&turn);
                app.activity = ActivityState::SessionOp(SessionOp::Undo);
                app.set_status(app::LogLevel::Info, "session", "undoing...");
                cmd_tx.send(Command::Undo {
                    message_id: turn.message_id,
                })?;
            } else {
                app.set_status(app::LogLevel::Warn, "session", "nothing to undo");
            }
        }
        KeyCode::Char('r') => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            if app.is_turn_active() {
                app.set_status(
                    app::LogLevel::Warn,
                    "session",
                    "cannot redo while agent is active",
                );
            } else if app.has_pending_session_op() || app.has_pending_undo() {
                app.set_status(app::LogLevel::Warn, "session", "undo already pending");
            } else if app.can_redo() {
                app.activity = ActivityState::SessionOp(SessionOp::Redo);
                app.set_status(app::LogLevel::Info, "session", "redoing...");
                cmd_tx.send(Command::Redo)?;
            } else {
                app.set_status(app::LogLevel::Warn, "session", "nothing to redo");
            }
        }
        _ => {
            app.set_status(app::LogLevel::Debug, "input", "unknown chord");
        }
    }
    Ok(())
}

pub(crate) fn handle_sessions_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => {
            app.should_quit = true;
            return Ok(());
        }
        _ => {}
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        match apply_session_fork_toggle_key(app, false) {
            SessionKeyAction::LoadMoreSessions {
                group_idx,
                parent_path,
            } => {
                if let Some(request) = app.session_child_page_request(group_idx, &parent_path) {
                    cmd_tx.send(request)?;
                }
            }
            SessionKeyAction::None => {}
            _ => {}
        }
        return Ok(());
    }

    match apply_sessions_key(
        app,
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            KeyCode::Null
        } else {
            key.code
        },
    ) {
        SessionKeyAction::LoadSession {
            session_id,
            agent_id,
            cwd,
        } => {
            send_load_session_commands(
                cmd_tx,
                session_id,
                cwd,
                agent_id.or_else(|| app.agent_id.clone()),
            )?;
        }
        SessionKeyAction::AttachRemoteSession {
            node_id,
            session_id,
        } => {
            cmd_tx.send(Command::AttachRemoteSession {
                node_id,
                session_id,
            })?;
        }
        SessionKeyAction::DeleteSession { session_id } => {
            cmd_tx.send(Command::DeleteSession { session_id })?;
        }
        SessionKeyAction::DismissRemoteSession { session_id } => {
            cmd_tx.send(Command::DismissRemoteSession { session_id })?;
        }
        SessionKeyAction::NewSession => {
            app.open_new_session_popup();
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
            if let Some(request) = request {
                cmd_tx.send(request)?;
            }
        }
        SessionKeyAction::None => {}
    }
    Ok(())
}

pub(crate) fn handle_session_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    // Tab / BackTab: switch between sessions and delegates tabs
    if matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
        app.session_popup_tab = 1 - app.session_popup_tab;
        return Ok(());
    }

    if app.session_popup_tab == 1 {
        // Delegates tab
        return handle_delegate_popup_key(app, key, cmd_tx);
    }

    // Sessions tab
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('n') {
        if !can_send_server_commands(app) {
            return Ok(());
        }
        app.open_new_session_popup();
        return Ok(());
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        match apply_session_fork_toggle_key(app, true) {
            SessionKeyAction::LoadMoreSessions {
                group_idx,
                parent_path,
            } => {
                if let Some(request) = app.session_child_page_request(group_idx, &parent_path) {
                    cmd_tx.send(request)?;
                }
            }
            SessionKeyAction::None => {}
            _ => {}
        }
        return Ok(());
    }

    match apply_popup_session_key(
        app,
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            KeyCode::Null
        } else {
            key.code
        },
    ) {
        SessionKeyAction::LoadSession {
            session_id,
            agent_id,
            cwd,
        } => {
            send_load_session_commands(
                cmd_tx,
                session_id,
                cwd,
                agent_id.or_else(|| app.agent_id.clone()),
            )?;
        }
        SessionKeyAction::AttachRemoteSession {
            node_id,
            session_id,
        } => {
            cmd_tx.send(Command::AttachRemoteSession {
                node_id,
                session_id,
            })?;
        }
        SessionKeyAction::DeleteSession { session_id } => {
            cmd_tx.send(Command::DeleteSession { session_id })?;
        }
        SessionKeyAction::DismissRemoteSession { session_id } => {
            cmd_tx.send(Command::DismissRemoteSession { session_id })?;
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
            if let Some(request) = request {
                cmd_tx.send(request)?;
            }
        }
        SessionKeyAction::NewSession | SessionKeyAction::None => {}
    }
    Ok(())
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
    use crate::app::PopupItem;

    match key {
        KeyCode::Esc => {
            app.popup = Popup::None;
        }
        KeyCode::Up => {
            app.session_cursor = app.session_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            let max = app.visible_popup_items().len().saturating_sub(1);
            app.session_cursor = (app.session_cursor + 1).min(max);
        }
        KeyCode::PageUp => {
            let step = popup_page_step(app.session_popup_visible_rows);
            app.session_cursor = app.session_cursor.saturating_sub(step);
        }
        KeyCode::PageDown => {
            let max = app.visible_popup_items().len().saturating_sub(1);
            let step = popup_page_step(app.session_popup_visible_rows);
            app.session_cursor = app.session_cursor.saturating_add(step).min(max);
        }
        KeyCode::Enter => {
            let items = app.visible_popup_items();
            if let Some(item) = items.get(app.session_cursor).cloned() {
                match item {
                    PopupItem::GroupHeader { cwd, .. } => {
                        app.toggle_popup_group_collapse(cwd.as_deref());
                        // Clamp cursor: collapsing may hide rows the cursor pointed at.
                        let new_len = app.visible_popup_items().len();
                        if new_len > 0 && app.session_cursor >= new_len {
                            app.session_cursor = new_len - 1;
                        }
                    }
                    PopupItem::Session {
                        group_idx, path, ..
                    } => {
                        let session = app.session_by_path(group_idx, &path).cloned();
                        if let Some(session) = session {
                            app.popup = Popup::None;
                            let session_id = session.session_id;
                            if let Some(node_id) =
                                app.session_remote_node_id(&session_id).map(str::to_string)
                            {
                                app.remember_remote_session_node(&session_id, &node_id);
                                return SessionKeyAction::AttachRemoteSession {
                                    node_id,
                                    session_id,
                                };
                            }
                            if app.is_remote_session_id(&session_id) {
                                app.set_status(
                                    app::LogLevel::Warn,
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
                                    .or_else(|| app.session_groups[group_idx].cwd.clone()),
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
            let items = app.visible_popup_items();
            if let Some(PopupItem::Session {
                group_idx, path, ..
            }) = items.get(app.session_cursor).cloned()
            {
                let Some(session) = app.session_by_path(group_idx, &path).cloned() else {
                    return SessionKeyAction::None;
                };
                let sid = session.session_id;
                let is_remote = app.is_remote_session_id(&sid);
                // Optimistic remove
                if path.len() == 1 {
                    app.session_groups[group_idx].sessions.remove(path[0]);
                } else if let Some(parent_path) = path.get(..path.len() - 1).map(Vec::from)
                    && let Some(parent) = app.session_by_path_mut(group_idx, &parent_path)
                    && let Some(child_idx) = path.last()
                    && *child_idx < parent.children.len()
                {
                    parent.children.remove(*child_idx);
                }
                app.session_groups.retain(|g| !g.sessions.is_empty());
                let new_len = app.visible_popup_items().len();
                if new_len > 0 && app.session_cursor >= new_len {
                    app.session_cursor = new_len - 1;
                }
                return if is_remote {
                    SessionKeyAction::DismissRemoteSession { session_id: sid }
                } else {
                    SessionKeyAction::DeleteSession { session_id: sid }
                };
            }
            // Delete on a GroupHeader: no-op
        }
        KeyCode::Backspace => {
            app.session_filter.pop();
            app.session_cursor = 0;
        }
        KeyCode::Char(c) => {
            app.session_filter.push(c);
            app.session_cursor = 0;
        }
        _ => {}
    }
    SessionKeyAction::None
}

pub(crate) fn apply_session_fork_toggle_key(app: &mut App, popup_items: bool) -> SessionKeyAction {
    use crate::app::{PopupItem, StartPageItem};

    let selected = if popup_items {
        app.visible_popup_items()
            .get(app.session_cursor)
            .cloned()
            .and_then(|item| match item {
                PopupItem::Session {
                    group_idx, path, ..
                } => Some((group_idx, path)),
                _ => None,
            })
    } else {
        app.visible_start_items()
            .get(app.session_cursor)
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

    let should_load = app.toggle_session_children(group_idx, &path);
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

fn handle_delegate_view_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<AppAction> {
    match key.code {
        KeyCode::Up => {
            app.scroll_offset = app.scroll_offset.saturating_add(1);
        }
        KeyCode::Down => {
            app.scroll_offset = app.scroll_offset.saturating_sub(1);
        }
        KeyCode::PageUp => {
            app.scroll_offset = app.scroll_offset.saturating_add(10);
        }
        KeyCode::PageDown => {
            app.scroll_offset = app.scroll_offset.saturating_sub(10);
        }
        KeyCode::Home => {
            // Scroll to top: set a large offset (draw_messages clamps it).
            app.scroll_offset = u16::MAX;
        }
        KeyCode::End => {
            app.scroll_offset = 0;
        }
        KeyCode::Esc => {
            // Go back to parent session.
            if let Some(parent_sid) = app.parent_session_id.clone() {
                send_load_session_commands(
                    cmd_tx,
                    parent_sid,
                    app.current_session_cwd(),
                    app.agent_id.clone(),
                )?;
            }
        }
        _ => {}
    }
    Ok(AppAction::None)
}

// ── Delegate popup key handler ────────────────────────────────────────────────

pub(crate) fn handle_delegate_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match apply_delegate_popup_key(
        app,
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            KeyCode::Null
        } else {
            key.code
        },
    ) {
        SessionKeyAction::LoadSession {
            session_id,
            agent_id,
            cwd,
        } => {
            send_load_session_commands(
                cmd_tx,
                session_id,
                cwd,
                agent_id.or_else(|| app.agent_id.clone()),
            )?;
        }
        SessionKeyAction::NewSession
        | SessionKeyAction::AttachRemoteSession { .. }
        | SessionKeyAction::DeleteSession { .. }
        | SessionKeyAction::DismissRemoteSession { .. }
        | SessionKeyAction::LoadMoreSessions { .. }
        | SessionKeyAction::None => {}
    }
    Ok(())
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
            app.popup = Popup::None;
        }
        KeyCode::Up => {
            app.delegate_cursor = app.delegate_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            let max = app.visible_delegate_entries().len().saturating_sub(1);
            app.delegate_cursor = (app.delegate_cursor + 1).min(max);
        }
        KeyCode::PageUp => {
            let step = popup_page_step(app.delegate_popup_visible_rows);
            app.delegate_cursor = app.delegate_cursor.saturating_sub(step);
        }
        KeyCode::PageDown => {
            let max = app.visible_delegate_entries().len().saturating_sub(1);
            let step = popup_page_step(app.delegate_popup_visible_rows);
            app.delegate_cursor = app.delegate_cursor.saturating_add(step).min(max);
        }
        KeyCode::Enter => {
            let selected = app
                .visible_delegate_entries()
                .get(app.delegate_cursor)
                .map(|entry| {
                    (
                        entry.child_session_id.clone(),
                        entry.target_agent_id.clone(),
                    )
                });
            if let Some((child_session_id, target_agent_id)) = selected {
                if let Some(sid) = child_session_id {
                    // Use the real parent when navigating between siblings.
                    app.pending_parent_session_id = app
                        .parent_session_id
                        .clone()
                        .or_else(|| app.session_id.clone());
                    app.popup = Popup::None;
                    return SessionKeyAction::LoadSession {
                        session_id: sid,
                        agent_id: target_agent_id,
                        cwd: app.current_session_cwd(),
                    };
                } else {
                    app.set_status(
                        app::LogLevel::Warn,
                        "delegates",
                        "delegation still pending — no session to load",
                    );
                }
            }
        }
        KeyCode::Backspace => {
            app.delegate_filter.pop();
            app.delegate_cursor = 0;
        }
        KeyCode::Char(c) => {
            app.delegate_filter.push(c);
            app.delegate_cursor = 0;
        }
        _ => {}
    }
    SessionKeyAction::None
}

fn begin_fork_session(
    app: &mut App,
    message_id: String,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    if app.pending_fork_message_id.is_some() {
        app.set_status(LogLevel::Warn, "fork", "fork already pending");
        return Ok(());
    }
    if app.is_turn_active() {
        app.set_status(LogLevel::Warn, "fork", "cannot fork while agent is active");
        return Ok(());
    }
    if app.has_pending_session_op() {
        app.set_status(LogLevel::Warn, "fork", "session operation already pending");
        return Ok(());
    }
    if message_id.is_empty() {
        app.set_status(LogLevel::Warn, "fork", "selected turn has no message id");
        return Ok(());
    }

    app.pending_fork_message_id = Some(message_id.clone());
    app.set_status(LogLevel::Info, "fork", "forking session...");
    cmd_tx.send(Command::ForkSession { message_id })?;
    Ok(())
}

pub(crate) fn handle_fork_turn_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => app.popup = Popup::None,
        KeyCode::Up => app.move_fork_cursor(-1),
        KeyCode::Down => app.move_fork_cursor(1),
        KeyCode::Backspace => app.fork_filter_backspace(),
        KeyCode::Enter => {
            if let Some(turn) = app.selected_fork_turn() {
                begin_fork_session(app, turn.message_id, cmd_tx)?;
            } else {
                app.set_status(LogLevel::Warn, "fork", "no forkable turns");
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.fork_filter_insert(c);
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_log_popup_key(app: &mut App, key: KeyEvent) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => {
            app.popup = Popup::None;
        }
        KeyCode::Up => {
            app.log_cursor = app.log_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            let max = app.filtered_logs().len().saturating_sub(1);
            app.log_cursor = (app.log_cursor + 1).min(max);
        }
        KeyCode::PageUp => {
            app.log_cursor = app.log_cursor.saturating_sub(10);
        }
        KeyCode::PageDown => {
            let max = app.filtered_logs().len().saturating_sub(1);
            app.log_cursor = (app.log_cursor + 10).min(max);
        }
        KeyCode::Home => {
            app.log_cursor = 0;
        }
        KeyCode::End => {
            app.log_cursor = app.filtered_logs().len().saturating_sub(1);
        }
        KeyCode::Backspace => {
            app.log_filter.pop();
            app.log_cursor = app.filtered_logs().len().saturating_sub(1);
        }
        KeyCode::Tab => {
            app.cycle_log_level_filter();
            app.log_cursor = app.filtered_logs().len().saturating_sub(1);
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.log_filter.push(c);
            app.log_cursor = 0;
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_profile_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => {
            app.popup = Popup::None;
        }
        KeyCode::Up => app.move_profile_cursor(-1),
        KeyCode::Down => app.move_profile_cursor(1),
        KeyCode::Backspace => {
            app.profile_filter.pop();
            app.profile_cursor = 0;
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.profile_filter.push(c);
            app.profile_cursor = 0;
        }
        KeyCode::Enter => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            if let Some(profile_id) = app.selected_profile().map(|profile| profile.id.clone()) {
                app.active_profile_id = Some(profile_id.clone());
                if app.current_session_profile_id().is_none() {
                    app.agents.clear();
                    app.agents_profile_id = None;
                    app.model_popup_agent_tab = 0;
                    cmd_tx.send(Command::ListProfileAgents {
                        profile_id: profile_id.clone(),
                    })?;
                }
                app.popup = Popup::None;
                app.set_status(
                    app::LogLevel::Info,
                    "profile",
                    format!("new sessions will use {profile_id}"),
                );
                save_config(app);
            } else {
                app.set_status(app::LogLevel::Warn, "profile", "no matching profile");
            }
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_new_session_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => {
            app.popup = Popup::None;
        }
        KeyCode::Up => {
            app.move_new_session_completion_selection(-1);
        }
        KeyCode::Down => {
            app.move_new_session_completion_selection(1);
        }
        KeyCode::Tab => {
            app.accept_selected_new_session_completion();
        }
        KeyCode::Left => {
            app.new_session_cursor = app.new_session_cursor.saturating_sub(1);
            app.refresh_new_session_completion();
        }
        KeyCode::Right => {
            app.new_session_cursor = (app.new_session_cursor + 1).min(app.new_session_path.len());
            app.refresh_new_session_completion();
        }
        KeyCode::Home => {
            app.new_session_cursor = 0;
            app.refresh_new_session_completion();
        }
        KeyCode::End => {
            app.new_session_cursor = app.new_session_path.len();
            app.refresh_new_session_completion();
        }
        KeyCode::Backspace => {
            if app.new_session_cursor > 0 && !app.new_session_path.is_empty() {
                let idx = app.new_session_cursor - 1;
                app.new_session_path.remove(idx);
                app.new_session_cursor = idx;
            }
            app.refresh_new_session_completion();
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.new_session_path.insert(app.new_session_cursor, c);
            app.new_session_cursor += 1;
            app.refresh_new_session_completion();
        }
        KeyCode::Enter => {
            if !can_send_server_commands(app) {
                return Ok(());
            }
            let cwd = app.normalize_new_session_path(&app.new_session_path);
            app.popup = Popup::None;
            cmd_tx.send(Command::NewSession {
                cwd,
                profile_id: app.active_profile_id.clone(),
            })?;
        }
        _ => {}
    }
    Ok(())
}

/// Invalidate every theme-dependent cache in `app` so that the next render
/// frame rebuilds styled lines with the current palette.
pub(crate) fn invalidate_theme_caches(app: &mut App) {
    app.card_cache.invalidate();
    app.streaming_cache.invalidate();
    app.streaming_thinking_cache.invalidate();
}

pub(crate) fn handle_theme_popup_key(app: &mut App, key: KeyEvent) -> anyhow::Result<()> {
    let filtered_len = || -> usize {
        let q = app.theme_filter.to_lowercase();
        if q.is_empty() {
            theme::Theme::available_themes().len()
        } else {
            theme::Theme::available_themes()
                .iter()
                .filter(|t| t.label.to_lowercase().contains(&q) || t.id.to_lowercase().contains(&q))
                .count()
        }
    };

    let filtered_index = |cursor: usize| -> Option<usize> {
        let q = app.theme_filter.to_lowercase();
        let iter = theme::Theme::available_themes().iter().enumerate();
        if q.is_empty() {
            Some(cursor)
        } else {
            iter.filter(|(_, t)| {
                t.label.to_lowercase().contains(&q) || t.id.to_lowercase().contains(&q)
            })
            .nth(cursor)
            .map(|(i, _)| i)
        }
    };

    match key.code {
        KeyCode::Esc => {
            app.popup = Popup::None;
        }
        KeyCode::Up => {
            app.theme_cursor = app.theme_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            let max = filtered_len().saturating_sub(1);
            app.theme_cursor = (app.theme_cursor + 1).min(max);
        }
        KeyCode::Enter => {
            if let Some(idx) = filtered_index(app.theme_cursor) {
                theme::Theme::set_by_index(idx);
                theme::Theme::begin_frame();
                invalidate_theme_caches(app);
                app.popup = Popup::None;
                save_config(app);
            }
        }
        KeyCode::Backspace => {
            app.theme_filter.pop();
            app.theme_cursor = 0;
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.theme_filter.push(c);
            app.theme_cursor = 0;
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn handle_chat_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<AppAction> {
    if app.input_line_width == 0 {
        app.input_line_width = 1;
    }
    let input_blocked = app.input_blocked_by_activity();
    match key.code {
        KeyCode::Esc => {
            // Dismiss whichever completion popup is open first.
            if app.mention_state.is_some() || app.slash_state.is_some() {
                app.mention_state = None;
                app.slash_state = None;
                app.clear_cancel_confirm();
            } else if app.has_cancellable_activity() {
                if app.cancel_confirm_active() {
                    app.clear_cancel_confirm();
                    app.set_status(app::LogLevel::Warn, "activity", "stopping...");
                    cmd_tx.send(Command::CancelSession)?;
                } else {
                    app.arm_cancel_confirm();
                }
            } else {
                app.clear_cancel_confirm();
            }
        }
        KeyCode::Enter => {
            // 1. Complete slash completion, then fall through to execute.
            if app.slash_state.is_some() {
                app.accept_selected_slash_completion();
            }
            // 2. Try slash command execution (input starts with '/').
            if !app.input.is_empty() && app.input.trim_start().starts_with('/') {
                match try_execute_slash_command(app, cmd_tx)? {
                    SlashResult::OpenEditor => return Ok(AppAction::OpenExternalEditor),
                    SlashResult::Handled => return Ok(AppAction::None),
                    SlashResult::NotACommand => {}
                }
            }
            // 3. Accept mention completion.
            if app.mention_state.is_some() && app.accept_selected_mention() {
                if let Some(msg) = app.request_file_index_if_needed() {
                    cmd_tx.send(msg)?;
                }
                return Ok(AppAction::None);
            }
            // 4. Normal prompt send.
            if !app.input.is_empty() {
                if input_blocked || !can_send_server_commands(app) {
                    return Ok(AppAction::None);
                }
                let (text, links) = app.build_prompt_text_and_links(&app.input);
                let text = text.trim().to_string();
                let _ = app.take_input();
                if text.is_empty() {
                    return Ok(AppAction::None);
                }
                let mut prompt = vec![PromptBlock::Text { text: text.clone() }];
                for path in links {
                    prompt.push(PromptBlock::ResourceLink {
                        name: path.clone(),
                        uri: path,
                    });
                }
                let local_id = app.push_pending_prompt(text);
                if let Err(error) = cmd_tx.send(Command::Prompt {
                    prompt,
                    local_id: local_id.clone(),
                }) {
                    app.messages.retain(|entry| {
                        !matches!(
                            entry,
                            ChatEntry::User {
                                message_id: Some(message_id),
                                ..
                            } if message_id == &local_id
                        )
                    });
                    app.card_cache.invalidate();
                    return Err(error.into());
                }
            }
        }
        KeyCode::Tab if !input_blocked => {
            if app.slash_state.is_some() {
                app.accept_selected_slash_completion();
            } else if app.mention_state.is_some()
                && app.accept_selected_mention()
                && let Some(msg) = app.request_file_index_if_needed()
            {
                cmd_tx.send(msg)?;
            }
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) && !input_blocked => {
            app.input_insert(c);
            if let Some(msg) = app.request_file_index_if_needed() {
                cmd_tx.send(msg)?;
            }
        }
        KeyCode::Up => {
            if input_blocked {
                return Ok(AppAction::None);
            }
            if app.slash_state.is_some() {
                app.move_slash_selection(-1);
            } else if app.mention_state.is_some() {
                app.move_mention_selection(-1);
            } else {
                app.input_up_visual(2);
            }
        }
        KeyCode::Down => {
            if input_blocked {
                return Ok(AppAction::None);
            }
            if app.slash_state.is_some() {
                app.move_slash_selection(1);
            } else if app.mention_state.is_some() {
                app.move_mention_selection(1);
            } else {
                app.input_down_visual(2);
            }
        }
        KeyCode::PageUp => {
            app.scroll_offset = app.scroll_offset.saturating_add(10);
        }
        KeyCode::PageDown => {
            app.scroll_offset = app.scroll_offset.saturating_sub(10);
        }
        KeyCode::Backspace if !input_blocked => {
            app.input_backspace();
        }
        KeyCode::Delete if !input_blocked => {
            app.input_delete();
        }
        KeyCode::Left if !input_blocked => {
            app.input_left();
        }
        KeyCode::Right if !input_blocked => {
            app.input_right();
        }
        KeyCode::Home if !input_blocked => {
            app.input_home();
        }
        KeyCode::End => {
            if input_blocked {
                app.scroll_offset = 0;
            } else if app.input.is_empty() {
                app.scroll_offset = 0; // snap to bottom
            } else {
                app.input_end();
            }
        }
        _ => {}
    }
    Ok(AppAction::None)
}

// ── Mode switching ─────────────────────────────────────────────────────────────

/// Switch the agent mode to `target` (e.g. "build", "plan").
/// Caches the outgoing mode state, sends `SetAgentMode`, restores cached state
/// for the target mode, and persists config/cache.
fn switch_mode(
    app: &mut App,
    cmd_tx: &mpsc::UnboundedSender<Command>,
    target: &str,
) -> anyhow::Result<()> {
    cmd_tx.send(Command::SetAgentMode {
        mode: target.to_string(),
    })?;

    if target == "review" {
        if app.agent_mode != "review" {
            app.mode_before_review = Some(app.agent_mode.clone());
        }
    } else {
        app.mode_before_review = None;
    }

    app.agent_mode = target.to_string();

    save_config(app);
    Ok(())
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

/// Parse and execute a slash command from `app.input`.
///
/// Expects `app.input` to begin with `/`.  Always calls `app.take_input()`
/// before performing any side-effects so the command text is cleared first
/// (this allows `/undo` to optionally restore the previous turn text).
fn try_execute_slash_command(
    app: &mut App,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<SlashResult> {
    // Extract the command name (first word after '/') and optional argument.
    let after_slash = app.input.trim_start_matches('/');
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

    if cmd.is_empty() {
        return Ok(SlashResult::NotACommand);
    }

    match cmd.as_str() {
        "model" => {
            app.take_input();
            if app.screen != crate::app::Screen::Chat {
                app.set_status(
                    app::LogLevel::Warn,
                    "model",
                    "model select is only available in chat",
                );
                return Ok(SlashResult::Handled);
            }
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            app.popup = crate::app::Popup::ModelSelect;
            app.model_popup_agent_tab = 0;
            if arg.is_empty() {
                app.model_filter.clear();
                app.model_cursor = app.model_popup_open_cursor();
            } else {
                app.model_filter = arg;
                app.model_cursor = 0;
            }
        }
        "mode" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            if arg.is_empty() {
                // No arg: cycle to next mode (same as Tab).
                switch_mode(app, cmd_tx, &app.next_mode())?;
            } else {
                match arg.as_str() {
                    "build" | "plan" => {
                        if app.agent_mode == arg {
                            app.set_status(
                                app::LogLevel::Info,
                                "mode",
                                format!("already in {} mode", arg),
                            );
                        } else {
                            switch_mode(app, cmd_tx, &arg)?;
                        }
                    }
                    _ => {
                        app.set_status(
                            app::LogLevel::Warn,
                            "mode",
                            format!("unknown mode: {} (try build or plan)", arg),
                        );
                    }
                }
            }
        }
        "review" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            if app.agent_mode == "review" {
                app.set_status(app::LogLevel::Info, "mode", "already in review mode");
            } else {
                switch_mode(app, cmd_tx, "review")?;
            }
        }
        "thinking" => {
            app.take_input();
            if arg.is_empty() {
                app.set_status(
                    app::LogLevel::Info,
                    "model",
                    format!("thinking: {}", app.reasoning_effort_label()),
                );
            } else {
                let level = arg.to_lowercase();
                if app::validate_reasoning_effort(Some(&level)).is_none() {
                    app.set_status(
                        app::LogLevel::Warn,
                        "model",
                        format!(
                            "unknown level: {} (try auto, low, medium, high, max)",
                            level
                        ),
                    );
                } else {
                    if !can_send_server_commands(app) {
                        return Ok(SlashResult::Handled);
                    }
                    let msg = app.set_reasoning_effort(Some(&level)).unwrap();
                    cmd_tx.send(msg)?;
                    app.set_status(
                        app::LogLevel::Info,
                        "model",
                        format!("thinking: {}", app.reasoning_effort_label()),
                    );
                }
            }
        }
        "theme" => {
            app.take_input();
            app.popup = crate::app::Popup::ThemeSelect;
            app.theme_filter.clear();
            app.theme_cursor = crate::theme::Theme::current_index();
        }
        "profile" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            if arg.is_empty() {
                app.open_profile_popup();
                cmd_tx.send(Command::ListProfiles)?;
            } else if let Some(profile_id) = app.find_profile_id(&arg) {
                app.active_profile_id = Some(profile_id.clone());
                if app.current_session_profile_id().is_none() {
                    app.agents.clear();
                    app.agents_profile_id = None;
                    app.model_popup_agent_tab = 0;
                    cmd_tx.send(Command::ListProfileAgents {
                        profile_id: profile_id.clone(),
                    })?;
                }
                app.set_status(
                    app::LogLevel::Info,
                    "profile",
                    format!("new sessions will use {profile_id}"),
                );
                save_config(app);
            } else {
                app.set_status(
                    app::LogLevel::Warn,
                    "profile",
                    format!("unknown profile: {}", arg),
                );
            }
        }
        "sessions" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            app.popup = crate::app::Popup::SessionSelect;
            app.session_popup_tab = 0;
            app.session_cursor = 0;
            app.session_filter.clear();
            if let Some(request) = app.begin_session_discovery() {
                cmd_tx.send(request)?;
            }
        }
        "delegates" => {
            app.take_input();
            if app.screen != crate::app::Screen::Chat {
                app.set_status(
                    app::LogLevel::Warn,
                    "delegates",
                    "delegates only available in chat",
                );
                return Ok(SlashResult::Handled);
            }
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            app.open_delegate_popup();
        }
        "new" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            app.open_new_session_popup();
        }
        "help" => {
            app.take_input();
            app.popup = crate::app::Popup::Help;
            app.help_scroll = 0;
        }
        "logs" => {
            app.take_input();
            app.popup = crate::app::Popup::Log;
            app.log_cursor = app.filtered_logs().len().saturating_sub(1);
            app.log_filter.clear();
        }
        "auth" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            app.open_auth_popup();
            cmd_tx.send(Command::ListAuthProviders)?;
        }
        "fork" => {
            app.take_input();
            if app.screen != crate::app::Screen::Chat {
                app.set_status(
                    app::LogLevel::Warn,
                    "fork",
                    "forking is only available in chat",
                );
                return Ok(SlashResult::Handled);
            }
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            if app.pending_fork_message_id.is_some() {
                app.set_status(app::LogLevel::Warn, "fork", "fork already pending");
            } else if app.has_pending_session_op() {
                app.set_status(
                    app::LogLevel::Warn,
                    "fork",
                    "session operation already pending",
                );
            } else if app.is_turn_active() {
                app.set_status(
                    app::LogLevel::Warn,
                    "fork",
                    "cannot fork while agent is active",
                );
            } else if let Some(turn) = app.latest_fork_boundary() {
                begin_fork_session(app, turn.message_id, cmd_tx)?;
            } else {
                app.set_status(app::LogLevel::Warn, "fork", "no forkable turns");
            }
        }
        "undo" => {
            // Clear '/undo' first so the undo logic can optionally restore
            // the previous turn's text into the now-empty input.
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            if app.is_turn_active() {
                app.set_status(
                    app::LogLevel::Warn,
                    "session",
                    "cannot undo while agent is active",
                );
            } else if app.has_pending_session_op() || app.has_pending_undo() {
                app.set_status(app::LogLevel::Warn, "session", "undo already pending");
            } else if let Some(turn) = app.current_undo_target().cloned() {
                if app.input.trim().is_empty() && !turn.text.is_empty() {
                    app.input = turn.text.clone();
                    app.input_cursor = app.input.len();
                    app.input_scroll = 0;
                }
                app.push_pending_undo(&turn);
                app.activity = ActivityState::SessionOp(SessionOp::Undo);
                app.set_status(app::LogLevel::Info, "session", "undoing...");
                cmd_tx.send(Command::Undo {
                    message_id: turn.message_id,
                })?;
            } else {
                app.set_status(app::LogLevel::Warn, "session", "nothing to undo");
            }
        }
        "redo" => {
            app.take_input();
            if !can_send_server_commands(app) {
                return Ok(SlashResult::Handled);
            }
            if app.is_turn_active() {
                app.set_status(
                    app::LogLevel::Warn,
                    "session",
                    "cannot redo while agent is active",
                );
            } else if app.has_pending_session_op() || app.has_pending_undo() {
                app.set_status(app::LogLevel::Warn, "session", "redo already pending");
            } else if app.can_redo() {
                app.activity = ActivityState::SessionOp(SessionOp::Redo);
                app.set_status(app::LogLevel::Info, "session", "redoing...");
                cmd_tx.send(Command::Redo)?;
            } else {
                app.set_status(app::LogLevel::Warn, "session", "nothing to redo");
            }
        }
        "editor" => {
            app.take_input();
            return Ok(SlashResult::OpenEditor);
        }
        "cancel" => {
            app.take_input();
            if app.has_cancellable_activity() {
                app.clear_cancel_confirm();
                app.set_status(app::LogLevel::Warn, "activity", "stopping...");
                cmd_tx.send(Command::CancelSession)?;
            } else {
                app.set_status(app::LogLevel::Warn, "activity", "nothing to cancel");
            }
        }
        "quit" => {
            app.take_input();
            app.should_quit = true;
        }
        _ => return Ok(SlashResult::NotACommand),
    }

    Ok(SlashResult::Handled)
}

// ── Auth popup key handler ─────────────────────────────────────────────────────

pub(crate) fn handle_auth_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    use crate::app::AuthPanel;

    // Clipboard fallback popup: any key dismisses it
    if app.auth_clipboard_fallback.is_some() {
        app.auth_clipboard_fallback = None;
        return Ok(());
    }

    match app.auth_panel {
        AuthPanel::List => match key.code {
            KeyCode::Esc => {
                if app.auth_selected.is_some() {
                    app.auth_close_detail();
                } else {
                    app.popup = Popup::None;
                }
            }
            KeyCode::Up => {
                let max = app.filtered_auth_providers().len().saturating_sub(1);
                app.auth_cursor = app.auth_cursor.saturating_sub(1).min(max);
            }
            KeyCode::Down => {
                let max = app.filtered_auth_providers().len().saturating_sub(1);
                app.auth_cursor = (app.auth_cursor + 1).min(max);
            }
            KeyCode::Enter => {
                let filtered = app.filtered_auth_providers();
                if let Some(&(real_idx, _)) = filtered.get(app.auth_cursor) {
                    let provider = &app.auth_providers[real_idx];
                    app.auth_last_result = None;
                    app.auth_ui_notice = None;
                    if provider.is_unconfigurable() {
                        app.auth_selected = Some(real_idx);
                        // Stay in list — the draw fn shows the info message
                    } else if provider.is_api_key_only() {
                        app.auth_selected = Some(real_idx);
                        app.auth_panel = AuthPanel::ApiKeyInput;
                        app.auth_api_key_input.clear();
                        app.auth_api_key_cursor = 0;
                    } else if provider.is_oauth_only()
                        && provider.oauth_status != Some(OAuthStatus::Connected)
                    {
                        app.auth_selected = Some(real_idx);
                        let provider_id = provider.provider.clone();
                        cmd_tx.send(Command::StartOAuthLogin {
                            provider: provider_id,
                        })?;
                    } else {
                        // Multi-method or connected OAuth-only: select for detail
                        app.auth_selected = Some(real_idx);
                    }
                }
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+D: disconnect/clear credential for selected provider
                if let Some(idx) = app.auth_selected {
                    let provider = &app.auth_providers[idx];
                    if provider.oauth_status == Some(OAuthStatus::Connected) {
                        let provider_id = provider.provider.clone();
                        cmd_tx.send(Command::DisconnectOAuth {
                            provider: provider_id,
                        })?;
                    } else if provider.has_stored_api_key {
                        let provider_id = provider.provider.clone();
                        cmd_tx.send(Command::ClearApiToken {
                            provider: provider_id,
                        })?;
                    }
                }
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+K: open API key panel for selected provider
                let filtered = app.filtered_auth_providers();
                if let Some(&(real_idx, _)) = filtered.get(app.auth_cursor) {
                    let provider = &app.auth_providers[real_idx];
                    if provider.env_var_name.is_some() || provider.has_stored_api_key {
                        app.auth_ui_notice = None;
                        app.auth_selected = Some(real_idx);
                        app.auth_panel = AuthPanel::ApiKeyInput;
                        app.auth_api_key_input.clear();
                        app.auth_api_key_cursor = 0;
                    }
                }
            }
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+O: start OAuth for selected provider
                let filtered = app.filtered_auth_providers();
                if let Some(&(real_idx, _)) = filtered.get(app.auth_cursor) {
                    let provider = &app.auth_providers[real_idx];
                    if provider.supports_oauth {
                        app.auth_ui_notice = None;
                        app.auth_selected = Some(real_idx);
                        let provider_id = provider.provider.clone();
                        cmd_tx.send(Command::StartOAuthLogin {
                            provider: provider_id,
                        })?;
                    }
                }
            }
            KeyCode::Backspace => {
                app.auth_filter.pop();
                app.auth_cursor = 0;
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.auth_filter.push(c);
                app.auth_cursor = 0;
            }
            _ => {}
        },
        AuthPanel::ApiKeyInput => match key.code {
            KeyCode::Esc => {
                app.auth_panel = AuthPanel::List;
                app.auth_api_key_input.clear();
                app.auth_api_key_cursor = 0;
            }
            KeyCode::Enter => {
                if let Some(idx) = app.auth_selected {
                    let trimmed = app.auth_api_key_input.trim().to_string();
                    if !trimmed.is_empty() {
                        let provider = app.auth_providers[idx].provider.clone();
                        cmd_tx.send(Command::SetApiToken {
                            provider,
                            api_key: trimmed,
                        })?;
                    }
                }
            }
            KeyCode::Tab => {
                app.auth_api_key_masked = !app.auth_api_key_masked;
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Clear stored key
                if let Some(idx) = app.auth_selected {
                    let provider = app.auth_providers[idx].provider.clone();
                    cmd_tx.send(Command::ClearApiToken { provider })?;
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.auth_api_key_input.insert(app.auth_api_key_cursor, c);
                app.auth_api_key_cursor += c.len_utf8();
            }
            KeyCode::Backspace if app.auth_api_key_cursor > 0 => {
                let cursor = app.auth_api_key_cursor;
                let ch = app.auth_api_key_input[..cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth_api_key_input.remove(cursor - ch.len_utf8());
                app.auth_api_key_cursor -= ch.len_utf8();
            }
            KeyCode::Left if app.auth_api_key_cursor > 0 => {
                let ch = app.auth_api_key_input[..app.auth_api_key_cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth_api_key_cursor -= ch.len_utf8();
            }
            KeyCode::Right if app.auth_api_key_cursor < app.auth_api_key_input.len() => {
                let ch = app.auth_api_key_input[app.auth_api_key_cursor..]
                    .chars()
                    .next()
                    .unwrap();
                app.auth_api_key_cursor += ch.len_utf8();
            }
            _ => {}
        },
        AuthPanel::OAuthFlow => match key.code {
            KeyCode::Esc => {
                app.auth_oauth_flow = None;
                app.auth_panel = AuthPanel::List;
                app.auth_oauth_response.clear();
                app.auth_oauth_response_cursor = 0;
            }
            KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Copy authorization URL to clipboard (C-y to avoid global C-c quit)
                if let Some(ref flow) = app.auth_oauth_flow {
                    let provider = flow.provider.clone();
                    let url = flow.authorization_url.clone();
                    try_copy_to_clipboard(app, &provider, &url);
                }
            }
            KeyCode::Enter => {
                if let Some(ref flow) = app.auth_oauth_flow {
                    let flow_id = flow.flow_id.clone();
                    let is_device_poll = flow.flow_kind == OAuthFlowKind::DevicePoll;
                    let response = if is_device_poll {
                        String::new()
                    } else {
                        app.auth_oauth_response.trim().to_string()
                    };
                    if is_device_poll || !response.is_empty() {
                        cmd_tx.send(Command::CompleteOAuthLogin { flow_id, response })?;
                    }
                }
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.auth_oauth_response
                    .insert(app.auth_oauth_response_cursor, c);
                app.auth_oauth_response_cursor += c.len_utf8();
            }
            KeyCode::Backspace if app.auth_oauth_response_cursor > 0 => {
                let cursor = app.auth_oauth_response_cursor;
                let ch = app.auth_oauth_response[..cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth_oauth_response.remove(cursor - ch.len_utf8());
                app.auth_oauth_response_cursor -= ch.len_utf8();
            }
            KeyCode::Left if app.auth_oauth_response_cursor > 0 => {
                let ch = app.auth_oauth_response[..app.auth_oauth_response_cursor]
                    .chars()
                    .next_back()
                    .unwrap();
                app.auth_oauth_response_cursor -= ch.len_utf8();
            }
            KeyCode::Right if app.auth_oauth_response_cursor < app.auth_oauth_response.len() => {
                let ch = app.auth_oauth_response[app.auth_oauth_response_cursor..]
                    .chars()
                    .next()
                    .unwrap();
                app.auth_oauth_response_cursor += ch.len_utf8();
            }
            _ => {}
        },
    }
    Ok(())
}

/// Try to copy text to the system clipboard. On failure, store in the fallback
/// field so the draw function can show a popup with the URL for manual copy.
fn copy_text_to_clipboard(text: &str) -> bool {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let commands = [
        ("xclip", &["-selection", "clipboard"] as &[&str]),
        ("xsel", &["--clipboard", "--input"]),
        ("wl-copy", &[]),
        ("pbcopy", &[]),
    ];

    for (cmd, args) in &commands {
        if let Ok(mut child) = Command::new(cmd)
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            if let Some(ref mut stdin) = child.stdin {
                let _ = stdin.write_all(text.as_bytes());
            }
            if child.wait().is_ok_and(|s| s.success()) {
                return true;
            }
        }
    }
    false
}

fn try_copy_to_clipboard(app: &mut App, provider: &str, text: &str) {
    app.auth_ui_notice = None;
    app.auth_clipboard_fallback = None;
    if copy_text_to_clipboard(text) {
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some(provider.to_string()),
            success: true,
            message: "Copied to clipboard".into(),
        });
    } else {
        app.auth_ui_notice = None;
        app.auth_clipboard_fallback = Some(text.to_string());
    }
}

pub(crate) fn handle_model_popup_key(
    app: &mut App,
    key: KeyEvent,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    match key.code {
        KeyCode::Esc => {
            app.popup = Popup::None;
        }
        KeyCode::Tab | KeyCode::BackTab => {
            if !app.model_popup_has_tabs() {
                return Ok(());
            }
            let n = app.model_popup_tab_count();
            if key.code == KeyCode::BackTab {
                app.model_popup_agent_tab = if app.model_popup_agent_tab == 0 {
                    n - 1
                } else {
                    app.model_popup_agent_tab - 1
                };
            } else {
                app.model_popup_agent_tab = (app.model_popup_agent_tab + 1) % n;
            }
            app.model_filter.clear();
            app.model_cursor = if app.model_popup_is_session_tab(app.model_popup_agent_tab) {
                app.model_popup_open_cursor()
            } else if let Some(aid) = app
                .model_popup_tab_agent_id(app.model_popup_agent_tab)
                .map(str::to_string)
            {
                app.delegate_model_cursor(&aid)
            } else {
                0
            };
        }
        KeyCode::Up => {
            app.model_cursor = app.model_cursor.saturating_sub(1);
        }
        KeyCode::Down => {
            let max = app.visible_model_popup_items().len().saturating_sub(1);
            app.model_cursor = (app.model_cursor + 1).min(max);
        }
        KeyCode::Enter => {
            let selected: Option<ModelEntry> = app
                .visible_model_popup_items()
                .get(app.model_cursor)
                .and_then(|item| match item {
                    crate::app::ModelPopupItem::Model { model_idx } => app.models.get(*model_idx),
                    crate::app::ModelPopupItem::ProviderHeader { .. } => None,
                })
                .cloned();
            if let Some(model) = selected {
                let tab_label = app
                    .model_popup_tab_label(app.model_popup_agent_tab)
                    .to_string();
                if app.model_popup_is_session_tab(app.model_popup_agent_tab) {
                    if !app.current_session_is_remote() {
                        if let Some(sid) = app.session_id.clone() {
                            cmd_tx.send(Command::SetSessionModel {
                                session_id: sid,
                                model_id: model.id.clone(),
                                node_id: model.node_id.clone(),
                            })?;
                        }
                        app.apply_model_selection_from_entry(&model);
                        if app.reasoning_effort.is_some() {
                            app.reasoning_effort = None;
                            cmd_tx.send(Command::SetReasoningEffort {
                                reasoning_effort: "auto".into(),
                            })?;
                        }
                    }
                    app.set_status(
                        app::LogLevel::Info,
                        "model",
                        format!("session: {}", model.label),
                    );
                } else if let Some(agent_id) = app
                    .model_popup_tab_agent_id(app.model_popup_agent_tab)
                    .map(str::to_string)
                    && let Some(profile_id) =
                        app.delegate_preference_profile_id().map(str::to_string)
                {
                    app.set_delegate_model_preference(&profile_id, &agent_id, &model);
                    if app.parent_session_id.is_none()
                        && let Some(session_id) = app.session_id.clone()
                    {
                        cmd_tx.send(Command::SetDelegateModel {
                            session_id,
                            agent_id,
                            model_id: Some(model.id.clone()),
                            node_id: model.node_id.clone(),
                        })?;
                    }
                    app.set_status(
                        app::LogLevel::Info,
                        "model",
                        format!("{tab_label}: {}", model.label),
                    );
                }
                save_config(app);
            }
        }
        KeyCode::Delete if !app.model_popup_is_session_tab(app.model_popup_agent_tab) => {
            if let Some(agent_id) = app
                .model_popup_tab_agent_id(app.model_popup_agent_tab)
                .map(str::to_string)
                && let Some(profile_id) = app.delegate_preference_profile_id().map(str::to_string)
            {
                app.clear_delegate_model_preference(&profile_id, &agent_id);
                if app.parent_session_id.is_none()
                    && let Some(session_id) = app.session_id.clone()
                {
                    cmd_tx.send(Command::SetDelegateModel {
                        session_id,
                        agent_id,
                        model_id: None,
                        node_id: None,
                    })?;
                }
                app.set_status(
                    app::LogLevel::Info,
                    "model",
                    "delegate model uses profile default",
                );
                save_config(app);
            }
        }
        KeyCode::Backspace => {
            app.model_filter.pop();
            app.model_cursor = 0;
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.model_filter.push(c);
            app.model_cursor = 0;
        }
        _ => {}
    }
    Ok(())
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
    use crate::app::StartPageItem;

    match key {
        KeyCode::Up => {
            app.session_cursor = app.session_cursor.saturating_sub(1);
            // Adjust scroll if needed (draw_start also does this, but keeping
            // state consistent here means tests don't need a frame).
            if app.session_cursor < app.start_page_scroll {
                app.start_page_scroll = app.session_cursor;
            }
        }
        KeyCode::Down => {
            // Button slot is one past the last item (items.len()).
            let max = app.visible_start_items().len(); // inclusive: button slot
            if app.session_cursor < max {
                app.session_cursor += 1;
            }
        }
        KeyCode::Enter => {
            let items = app.visible_start_items();
            // Cursor on the button slot (one past the last item)?
            if app.session_cursor == items.len() {
                return SessionKeyAction::NewSession;
            }
            if let Some(item) = items.get(app.session_cursor).cloned() {
                match item {
                    StartPageItem::GroupHeader { cwd, .. } => {
                        app.toggle_group_collapse(cwd.as_deref());
                        // Clamp cursor after collapse may hide rows.
                        let new_len = app.visible_start_items().len();
                        if new_len > 0 && app.session_cursor >= new_len {
                            app.session_cursor = new_len - 1;
                        }
                    }
                    StartPageItem::Session {
                        group_idx, path, ..
                    } => {
                        if let Some(session) = app.session_by_path(group_idx, &path).cloned() {
                            let session_id = session.session_id;
                            if let Some(node_id) =
                                app.session_remote_node_id(&session_id).map(str::to_string)
                            {
                                app.remember_remote_session_node(&session_id, &node_id);
                                return SessionKeyAction::AttachRemoteSession {
                                    node_id,
                                    session_id,
                                };
                            }
                            if app.is_remote_session_id(&session_id) {
                                app.set_status(
                                    app::LogLevel::Warn,
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
                                    .or_else(|| app.session_groups[group_idx].cwd.clone()),
                            };
                        }
                    }
                    StartPageItem::ShowMore { .. } => {
                        app.popup = Popup::SessionSelect;
                        app.session_popup_tab = 0;
                        app.session_cursor = 0;
                        app.session_filter.clear();
                    }
                }
            }
        }
        KeyCode::Delete => {
            let items = app.visible_start_items();
            if let Some(StartPageItem::Session {
                group_idx, path, ..
            }) = items.get(app.session_cursor).cloned()
            {
                let Some(session) = app.session_by_path(group_idx, &path).cloned() else {
                    return SessionKeyAction::None;
                };
                let sid = session.session_id;
                let is_remote = app.is_remote_session_id(&sid);
                // Optimistic remove
                if path.len() == 1 {
                    app.session_groups[group_idx].sessions.remove(path[0]);
                } else if let Some(parent_path) = path.get(..path.len() - 1).map(Vec::from)
                    && let Some(parent) = app.session_by_path_mut(group_idx, &parent_path)
                    && let Some(child_idx) = path.last()
                    && *child_idx < parent.children.len()
                {
                    parent.children.remove(*child_idx);
                }
                app.session_groups.retain(|g| !g.sessions.is_empty());
                let new_len = app.visible_start_items().len();
                if new_len > 0 && app.session_cursor >= new_len {
                    app.session_cursor = new_len - 1;
                }
                return if is_remote {
                    SessionKeyAction::DismissRemoteSession { session_id: sid }
                } else {
                    SessionKeyAction::DeleteSession { session_id: sid }
                };
            }
            // Delete on a GroupHeader: no-op
        }
        KeyCode::Backspace => {
            app.session_filter.pop();
            app.session_cursor = 0;
            app.start_page_scroll = 0;
        }
        KeyCode::Char(c) => {
            app.session_filter.push(c);
            app.session_cursor = 0;
            app.start_page_scroll = 0;
        }
        _ => {}
    }
    SessionKeyAction::None
}

// ── model popup tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod model_popup_tests {
    use super::*;
    use crate::app::{App, Popup};
    use crate::config::TestPersistenceGuard;
    use crate::domain::model::ModelEntry;
    use crate::domain::profile::{AgentInfo, ProfileInfo};
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use tokio::sync::mpsc;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
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
        app.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-1".into(),
                node_id: Some("node-1".into()),
                attached: Some(false),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.session_cursor = 1;

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
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-1".into(),
                node_id: Some("node-1".into()),
                attached: Some(true),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.session_cursor = 1;

        let action = apply_popup_session_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "remote-1".into(),
            }
        );
        assert!(matches!(app.popup, Popup::None));
    }

    #[test]
    fn ctrl_p_opens_command_palette_over_existing_popup() {
        let mut app = App::new();
        app.popup = Popup::ThemeSelect;

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
            &tx,
        )
        .unwrap();

        assert!(matches!(app.popup, Popup::CommandPalette));
        assert_eq!(app.command_palette_cursor, 0);
        assert!(app.command_palette_filter.is_empty());
    }

    #[test]
    fn command_palette_enter_opens_theme_picker() {
        let mut app = App::new();
        app.popup = Popup::CommandPalette;
        app.command_palette_filter = "theme".into();

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_command_palette_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(app.popup, Popup::ThemeSelect));
    }

    #[test]
    fn command_palette_open_mesh_refreshes_remote_nodes() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.popup = Popup::CommandPalette;
        app.command_palette_filter = "open mesh".into();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_command_palette_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(app.popup, Popup::Mesh));
        assert!(matches!(rx.try_recv(), Ok(Command::ListRemoteNodes)));
    }

    #[test]
    fn mesh_popup_enter_on_session_attaches_remote_session() {
        let mut app = App::new();
        app.popup = Popup::Mesh;
        app.mesh_focus = crate::mesh::MeshFocus::Sessions;
        app.mesh_nodes = vec![crate::protocol::RemoteNodeInfo {
            id: "node-1".into(),
            label: "framework".into(),
            ..Default::default()
        }];
        app.remote_sessions_by_node.insert(
            "node-1".into(),
            vec![crate::protocol::RemoteSessionInfo {
                id: "remote-1".into(),
                node_id: "node-1".into(),
                title: Some("Fix bug".into()),
                ..Default::default()
            }],
        );

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_mesh_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(
            rx.try_recv(),
            Ok(Command::AttachRemoteSession { node_id, session_id })
                if node_id == "node-1" && session_id == "remote-1"
        ));
    }

    #[test]
    fn command_palette_create_mesh_invite_opens_prefilled_invite_popup() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.popup = Popup::CommandPalette;
        app.command_palette_filter = "mesh invite".into();

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_command_palette_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(app.popup, Popup::MeshInvite));
        assert_eq!(app.mesh_invite_ttl, "24h");
        assert_eq!(app.mesh_invite_max_uses, "1");
    }

    #[test]
    fn mesh_popup_i_no_longer_opens_invite_popup() {
        let mut app = App::new();
        app.popup = Popup::Mesh;

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_mesh_popup_key(&mut app, key(KeyCode::Char('i')), &tx).unwrap();

        assert!(matches!(app.popup, Popup::Mesh));
    }

    #[test]
    fn mesh_invite_u_shows_manual_url_overlay() {
        let mut app = App::new();
        app.open_mesh_invite_form();
        app.apply_mesh_invite_created(crate::protocol::MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: Some("QR".into()),
            expires_at: 1,
            max_uses: 1,
            mesh_name: None,
        });

        handle_mesh_invite_qr_popup_key(&mut app, key(KeyCode::Char('u'))).unwrap();

        assert_eq!(
            app.mesh_clipboard_fallback.as_deref(),
            Some("qmt://mesh/join/token")
        );
    }

    #[test]
    fn mesh_invite_form_rejects_invalid_ttl_and_max_uses() {
        let mut app = App::new();
        app.open_mesh_invite_form();
        app.mesh_invite_ttl = "lol".into();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_mesh_invite_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(rx.try_recv().is_err());
        assert_eq!(
            app.mesh_error.as_deref(),
            Some("ttl must be like 30m, 1d3h, or 1d3h5m")
        );

        app.mesh_invite_ttl = "24h".into();
        app.mesh_invite_max_uses = "0".into();
        handle_mesh_invite_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(rx.try_recv().is_err());
        assert_eq!(
            app.mesh_error.as_deref(),
            Some("max uses must be at least 1")
        );
    }

    #[test]
    fn mesh_invite_form_enter_sends_create_invite() {
        let mut app = App::new();
        app.popup = Popup::Mesh;
        app.open_mesh_invite_form();
        app.mesh_invite_name = "Team Mesh".into();
        app.mesh_invite_ttl = "1d3h5m".into();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_mesh_invite_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(
            rx.try_recv(),
            Ok(Command::CreateMeshInvite { mesh_name, ttl, max_uses })
                if mesh_name.as_deref() == Some("Team Mesh")
                    && ttl.as_deref() == Some("1d3h5m")
                    && max_uses == Some(1)
        ));
    }

    #[test]
    fn mesh_invite_qr_esc_returns_to_create_invite_popup() {
        let mut app = App::new();
        app.apply_mesh_invite_created(crate::protocol::MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: Some("QR".into()),
            expires_at: 1,
            max_uses: 1,
            mesh_name: None,
        });

        handle_mesh_invite_qr_popup_key(&mut app, key(KeyCode::Esc)).unwrap();

        assert!(matches!(app.popup, Popup::MeshInvite));
    }

    #[test]
    fn command_palette_profile_lists_profiles() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.popup = Popup::CommandPalette;
        app.command_palette_filter = "profile".into();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_command_palette_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(app.popup, Popup::ProfileSelect));
        assert!(matches!(rx.try_recv(), Ok(Command::ListProfiles)));
    }

    #[test]
    fn slash_profile_without_arg_opens_selector_and_lists_profiles() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.screen = Screen::Chat;
        app.input = "/profile".into();
        app.input_cursor = app.input.len();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let action = handle_chat_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(action, AppAction::None));
        assert!(matches!(app.popup, Popup::ProfileSelect));
        assert!(matches!(rx.try_recv(), Ok(Command::ListProfiles)));
    }

    #[test]
    fn slash_profile_with_arg_updates_local_new_session_profile() {
        let _guard = TestPersistenceGuard::new("slash-profile");
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.screen = Screen::Chat;
        app.profiles = vec![make_profile("fast", "Fast")];
        app.active_profile_id = Some("old".into());
        app.input = "/profile Fast".into();
        app.input_cursor = app.input.len();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_chat_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(
            rx.try_recv(),
            Ok(Command::ListProfileAgents { profile_id }) if profile_id == "fast"
        ));
        assert_eq!(app.active_profile_id.as_deref(), Some("fast"));
    }

    #[test]
    fn profile_popup_enter_updates_local_new_session_profile() {
        let _guard = TestPersistenceGuard::new("profile-popup-enter");
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.popup = Popup::ProfileSelect;
        app.profiles = vec![make_profile("fast", "Fast"), make_profile("deep", "Deep")];
        app.profile_cursor = 1;

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_profile_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(app.popup, Popup::None));
        assert!(matches!(
            rx.try_recv(),
            Ok(Command::ListProfileAgents { profile_id }) if profile_id == "deep"
        ));
        assert_eq!(app.active_profile_id.as_deref(), Some("deep"));
    }

    #[test]
    fn new_session_uses_locally_selected_profile() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.popup = Popup::NewSession;
        app.new_session_path = "/repo".into();
        app.new_session_cursor = app.new_session_path.len();
        app.active_profile_id = Some("coder-delegate".into());

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_new_session_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(
            rx.try_recv(),
            Ok(Command::NewSession { profile_id: Some(profile_id), .. })
                if profile_id == "coder-delegate"
        ));
    }

    #[test]
    fn select_model_on_delegate_tab_saves_pref_not_session_model() {
        let _guard = TestPersistenceGuard::new("delegate-tab");
        let mut app = App::new();
        app.popup = Popup::ModelSelect;
        app.session_id = Some("s1".into());
        app.active_profile_id = Some("profile".into());
        app.session_profiles.insert("s1".into(), "profile".into());
        app.agent_mode = "plan".into();
        app.current_provider = Some("openai".into());
        app.current_model = Some("gpt-4o".into());
        app.agents = vec![make_agent("main", "Main"), make_agent("coder", "Coder")];
        app.models = vec![
            make_model("openai", "gpt-4o"),
            make_model("anthropic", "claude-sonnet"),
        ];
        app.model_popup_agent_tab = 1; // delegate tab
        let items = app.visible_model_popup_items();
        app.model_cursor = items
            .iter()
            .position(|i| {
                matches!(
                    i,
                    crate::app::ModelPopupItem::Model { model_idx } if app.models[*model_idx].model == "claude-sonnet"
                )
            })
            .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_model_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(
            rx.try_recv(),
            Ok(Command::SetDelegateModel {
                ref session_id,
                ref agent_id,
                ref model_id,
                node_id: None,
            }) if session_id == "s1"
                && agent_id == "coder"
                && model_id.as_deref() == Some("anthropic/claude-sonnet")
        ));
        assert_eq!(app.current_provider.as_deref(), Some("openai"));
        assert_eq!(app.current_model.as_deref(), Some("gpt-4o"));
        assert_eq!(
            app.get_delegate_model_preference("profile", "coder")
                .map(|preference| preference.model_id.as_str()),
            Some("anthropic/claude-sonnet")
        );
    }

    #[test]
    fn delete_on_delegate_tab_resets_parent_override() {
        let _guard = TestPersistenceGuard::new("delegate-reset");
        let mut app = App::new();
        app.popup = Popup::ModelSelect;
        app.session_id = Some("s1".into());
        app.active_profile_id = Some("profile".into());
        app.session_profiles.insert("s1".into(), "profile".into());
        app.agents = vec![
            make_agent("primary", "Session"),
            make_agent("coder", "Coder"),
        ];
        app.model_popup_agent_tab = 1;
        let model = make_model("anthropic", "claude-sonnet");
        app.set_delegate_model_preference("profile", "coder", &model);

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_model_popup_key(&mut app, key(KeyCode::Delete), &tx).unwrap();

        assert!(
            app.get_delegate_model_preference("profile", "coder")
                .is_none()
        );
        assert!(matches!(
            rx.try_recv(),
            Ok(Command::SetDelegateModel {
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
        app.popup = Popup::ModelSelect;
        app.session_id = Some("child".into());
        app.parent_session_id = Some("parent".into());
        app.active_profile_id = Some("profile".into());
        app.session_profiles
            .insert("child".into(), "profile".into());
        app.agents = vec![
            make_agent("primary", "Session"),
            make_agent("coder", "Coder"),
        ];
        app.models = vec![make_model("anthropic", "claude-sonnet")];
        app.model_popup_agent_tab = 1;
        app.model_cursor = 1;

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_model_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(rx.try_recv().is_err());
        assert!(
            app.get_delegate_model_preference("profile", "coder")
                .is_some()
        );
    }

    #[test]
    fn select_model_on_session_tab_applies_set_session_model() {
        let _guard = TestPersistenceGuard::new("active-mode");
        let mut app = App::new();
        app.popup = Popup::ModelSelect;
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();
        app.current_provider = Some("openai".into());
        app.current_model = Some("gpt-4o".into());
        app.reasoning_effort = Some("high".into());

        app.models = vec![
            make_model("openai", "gpt-4o"),
            make_model("anthropic", "claude-sonnet"),
        ];
        app.model_popup_agent_tab = 0;
        let items = app.visible_model_popup_items();
        app.model_cursor = items
            .iter()
            .position(|i| {
                matches!(
                    i,
                    crate::app::ModelPopupItem::Model { model_idx } if app.models[*model_idx].model == "claude-sonnet"
                )
            })
            .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_model_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        let msg1 = rx.try_recv().expect("expected SetSessionModel");
        assert!(matches!(msg1, Command::SetSessionModel { .. }));
        let msg2 = rx.try_recv().expect("expected SetReasoningEffort auto");
        assert!(
            matches!(msg2, Command::SetReasoningEffort { reasoning_effort } if reasoning_effort == "auto")
        );
        assert!(rx.try_recv().is_err());

        assert_eq!(app.current_provider.as_deref(), Some("anthropic"));
        assert_eq!(app.current_model.as_deref(), Some("claude-sonnet"));
        assert_eq!(app.reasoning_effort, None);
    }

    #[test]
    fn select_remote_model_on_local_session_sends_node_id() {
        let _guard = TestPersistenceGuard::new("remote-mesh-model");
        let mut app = App::new();
        app.popup = Popup::ModelSelect;
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();
        app.current_provider = Some("openai".into());
        app.current_model = Some("gpt-4o".into());
        app.models = vec![
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
        app.model_popup_agent_tab = 0;
        let items = app.visible_model_popup_items();
        app.model_cursor = items
            .iter()
            .position(|i| {
                matches!(
                    i,
                    crate::app::ModelPopupItem::Model { model_idx } if app.models[*model_idx].node_id.is_some()
                )
            })
            .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_model_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        match rx.try_recv().expect("SetSessionModel") {
            Command::SetSessionModel {
                node_id, model_id, ..
            } => {
                assert_eq!(node_id.as_deref(), Some("node-a"));
                assert_eq!(model_id, "mesh/anthropic/claude");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(rx.try_recv().is_err());
        assert_eq!(app.current_provider.as_deref(), Some("anthropic"));
        assert_eq!(app.current_model.as_deref(), Some("claude"));
        assert_eq!(app.current_model_node_id.as_deref(), Some("node-a"));
    }

    #[test]
    fn select_model_on_attached_remote_session_does_not_apply() {
        let _guard = TestPersistenceGuard::new("active-remote-mode");
        let mut app = App::new();
        app.popup = Popup::ModelSelect;
        app.session_id = Some("remote-1".into());
        app.agent_mode = "build".into();
        app.session_groups = vec![SessionGroup {
            sessions: vec![SessionSummary {
                session_id: "remote-1".into(),
                node_id: Some("node-1".into()),
                ..Default::default()
            }],
            ..Default::default()
        }];
        app.models = vec![make_model("anthropic", "claude-sonnet")];
        app.model_popup_agent_tab = 0;
        app.model_cursor = app
            .visible_model_popup_items()
            .iter()
            .position(|i| matches!(i, crate::app::ModelPopupItem::Model { .. }))
            .unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_model_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn opening_model_popup_starts_on_session_tab() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.screen = Screen::Chat;
        app.agent_mode = "review".into();
        app.current_provider = Some("openai".into());
        app.current_model = Some("gpt-4o".into());
        app.models = vec![make_model("openai", "gpt-4o")];

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_chord(&mut app, key(KeyCode::Char('m')), &tx).unwrap();

        assert!(matches!(app.popup, Popup::ModelSelect));
        assert_eq!(app.model_popup_agent_tab, 0);
        assert_eq!(app.model_cursor, app.model_popup_open_cursor());
    }

    #[test]
    fn slash_model_starts_on_session_tab() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.screen = Screen::Chat;
        app.agent_mode = "review".into();
        app.current_provider = Some("openai".into());
        app.current_model = Some("gpt-4o".into());
        app.models = vec![make_model("openai", "gpt-4o")];
        app.input = "/model".into();
        app.input_cursor = app.input.len();

        let (tx, _rx) = mpsc::unbounded_channel();
        let action = handle_chat_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(action, AppAction::None));
        assert!(matches!(app.popup, Popup::ModelSelect));
        assert_eq!(app.model_popup_agent_tab, 0);
        assert_eq!(app.model_cursor, app.model_popup_open_cursor());
    }

    #[test]
    fn review_slash_command_enters_review_and_tab_returns_to_previous_mode() {
        let _guard = TestPersistenceGuard::new("review-cycle");
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.screen = Screen::Chat;
        app.agent_mode = "plan".into();
        app.input = "/review".into();
        app.input_cursor = app.input.len();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let action = handle_chat_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert!(matches!(action, AppAction::None));
        assert_eq!(app.agent_mode, "review");
        assert_eq!(app.mode_before_review.as_deref(), Some("plan"));
        assert!(matches!(
            rx.try_recv().expect("expected SetAgentMode(review)"),
            Command::SetAgentMode { mode } if mode == "review"
        ));
        assert!(rx.try_recv().is_err());

        handle_key(&mut app, key(KeyCode::Tab), &tx).unwrap();
        assert_eq!(app.agent_mode, "plan");
        assert_eq!(app.mode_before_review, None);
        assert!(matches!(
            rx.try_recv().expect("expected SetAgentMode(plan)"),
            Command::SetAgentMode { mode } if mode == "plan"
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn delegate_popup_subscribes_with_child_target_agent() {
        use crate::app::Popup;
        use crate::domain::activity::{
            DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus,
        };

        let mut app = App::new();
        app.agent_id = Some("planner".into());
        app.popup = Popup::SessionSelect;
        app.delegate_entries.push(DelegateEntry {
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

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_delegate_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(matches!(
            rx.try_recv().expect("expected LoadSession"),
            Command::LoadSession { session_id, .. } if session_id == "child-1"
        ));
        assert!(matches!(
            rx.try_recv().expect("expected SubscribeSession"),
            Command::SubscribeSession { session_id, agent_id }
                if session_id == "child-1" && agent_id.as_deref() == Some("coder")
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn review_slash_command_is_noop_when_already_in_review() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.screen = Screen::Chat;
        app.agent_mode = "review".into();
        app.mode_before_review = Some("plan".into());
        app.input = "/review".into();
        app.input_cursor = app.input.len();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let action = handle_chat_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert!(matches!(action, AppAction::None));
        assert_eq!(app.agent_mode, "review");
        assert_eq!(app.mode_before_review.as_deref(), Some("plan"));
        assert!(rx.try_recv().is_err());
        assert_eq!(app.status, "already in review mode");
    }
}
