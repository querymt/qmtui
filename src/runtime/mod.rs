mod connection;
mod editor;
mod endpoint;
mod event_loop;
mod terminal;

use std::{
    io::Write,
    process::{Command as ProcessCommand, Stdio},
    time::Duration,
};

use crate::{
    acp,
    app::App,
    application::{self, AppEvent, Effect, RuntimeEvent, TerminalAction},
    command::Command,
    config,
    connection_state::ServerState,
    diagnostics::LogLevel,
    domain::model::DelegateModelPreference,
    navigation_state::Screen,
    runtime_events::{ConnectionManagerEvent, ServerChannelMsg},
    server_manager, theme,
};
use clap::Parser;
use connection::connection_manager;
use endpoint::{
    Cli, EndpointSelection, default_acp_ws_url, detect_launch_cwd, select_acp_endpoint,
};
use event_loop::run_loop;
use terminal::AppTerminal;

use tokio::sync::mpsc;

#[cfg(test)]
use crate::{acp_state::AcpAppEvent, connection_state::ConnState, navigation_state::Popup};

pub(super) struct EffectExecutor<'a> {
    cmd_tx: &'a mpsc::UnboundedSender<Command>,
}

impl<'a> EffectExecutor<'a> {
    fn new(cmd_tx: &'a mpsc::UnboundedSender<Command>) -> Self {
        Self { cmd_tx }
    }

    pub(super) fn execute(
        &mut self,
        terminal: &mut AppTerminal,
        app: &mut App,
        effects: Vec<Effect>,
    ) -> anyhow::Result<()> {
        let mut effects = effects.into_iter();
        while let Some(effect) = effects.next() {
            let runtime_event = match effect {
                Effect::Command(command) => self.send_command(command),
                Effect::ElicitationResponse {
                    elicitation_id,
                    action,
                    content,
                    outcome,
                } => {
                    let command = Command::ElicitationResponse {
                        elicitation_id: elicitation_id.clone(),
                        action,
                        content,
                    };
                    match self.send_command(command) {
                        Some(failed) => Some(failed),
                        None => Some(RuntimeEvent::ElicitationResponseSent {
                            elicitation_id,
                            outcome,
                        }),
                    }
                }
                Effect::PersistConfig => {
                    config::TuiConfig::load().with_app_settings(app).save();
                    None
                }
                Effect::CopyToClipboard { target, text } => Some(RuntimeEvent::ClipboardFinished {
                    target,
                    success: copy_text_to_clipboard(&text),
                }),
                Effect::OpenExternalEditor { initial_text } => {
                    Some(RuntimeEvent::ExternalEditorFinished {
                        outcome: terminal::open_external_editor_with_terminal(
                            terminal,
                            &initial_text,
                        )?,
                    })
                }
                Effect::Terminal(TerminalAction::Redraw) => {
                    terminal::redraw(terminal)?;
                    None
                }
                Effect::Quit => {
                    app.should_quit = true;
                    return Ok(());
                }
            };

            if let Some(event) = runtime_event {
                effects = application::update(app, AppEvent::Runtime(event))
                    .into_iter()
                    .chain(effects)
                    .collect::<Vec<_>>()
                    .into_iter();
            }
        }
        Ok(())
    }

    fn send_command(&self, command: Command) -> Option<RuntimeEvent> {
        match self.cmd_tx.send(command) {
            Ok(()) => None,
            Err(error) => {
                let message = error.to_string();
                Some(RuntimeEvent::CommandFailed {
                    command: error.0,
                    message,
                })
            }
        }
    }
}

fn copy_text_to_clipboard(text: &str) -> bool {
    let commands = [
        ("xclip", &["-selection", "clipboard"] as &[&str]),
        ("xsel", &["--clipboard", "--input"]),
        ("wl-copy", &[]),
        ("pbcopy", &[]),
    ];

    for (command, args) in commands {
        if let Ok(mut child) = ProcessCommand::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(text.as_bytes());
            }
            if child.wait().is_ok_and(|status| status.success()) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct TestEffects {
    effects: Vec<Effect>,
}

#[cfg(test)]
impl TestEffects {
    pub(crate) fn extend(&mut self, effects: Vec<Effect>) {
        self.effects.extend(effects);
    }

    pub(crate) fn next_command(&mut self) -> Option<Command> {
        if !matches!(
            self.effects.first(),
            Some(Effect::Command(_) | Effect::ElicitationResponse { .. })
        ) {
            return None;
        }
        match self.effects.remove(0) {
            Effect::Command(command) => Some(command),
            Effect::ElicitationResponse {
                elicitation_id,
                action,
                content,
                ..
            } => Some(Command::ElicitationResponse {
                elicitation_id,
                action,
                content,
            }),
            _ => unreachable!("first effect must contain a command"),
        }
    }

    pub(crate) fn as_slice(&self) -> &[Effect] {
        &self.effects
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::command::PromptBlock;
    use crate::domain::chat::{ChatEntry, ElicitationResponseOutcome};
    use crate::domain::elicitation::{
        ElicitationField, ElicitationFieldKind, ElicitationOption, ElicitationState,
    };
    use crate::domain::tool::ToolDetail;
    use crate::features::chat::input::handle_elicitation_key as handle_feature_elicitation_key;
    use crate::handlers::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn handle_elicitation_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
        handle_feature_elicitation_key(&mut app.chat, &mut app.render, key)
            .into_iter()
            .map(|response| Effect::ElicitationResponse {
                elicitation_id: response.elicitation_id,
                action: response.action,
                content: response.content,
                outcome: response.outcome,
            })
            .collect()
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn modified_key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn test_terminal() -> AppTerminal {
        ratatui::Terminal::with_options(
            ratatui::backend::CrosstermBackend::new(std::io::stdout()),
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Fixed(ratatui::layout::Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap()
    }

    #[test]
    fn test_effects_next_command_does_not_skip_non_command_effects() {
        let mut effects = TestEffects::default();
        effects.extend(vec![
            Effect::PersistConfig,
            Effect::Command(Command::Init),
            Effect::Quit,
        ]);

        assert_eq!(effects.next_command(), None);
        assert_eq!(
            effects.as_slice(),
            &[
                Effect::PersistConfig,
                Effect::Command(Command::Init),
                Effect::Quit,
            ]
        );
    }

    #[test]
    fn effect_executor_sends_commands_once_in_order_and_applies_quit() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = App::new();

        executor
            .execute(
                &mut terminal,
                &mut app,
                vec![
                    Effect::Command(Command::Init),
                    Effect::Quit,
                    Effect::Command(Command::ListAllModels { refresh: false }),
                ],
            )
            .unwrap();

        assert_eq!(rx.try_recv(), Ok(Command::Init));
        assert!(rx.try_recv().is_err());
        assert!(app.should_quit);
    }

    #[test]
    fn effect_executor_runs_runtime_effects_before_remaining_effects() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let command = Command::ElicitationResponse {
            elicitation_id: "test-id".into(),
            action: "accept".into(),
            content: Some(serde_json::json!({ "choice": "a" })),
        };

        executor
            .execute(
                &mut terminal,
                &mut app,
                vec![
                    Effect::ElicitationResponse {
                        elicitation_id: "test-id".into(),
                        action: "accept".into(),
                        content: Some(serde_json::json!({ "choice": "a" })),
                        outcome: ElicitationResponseOutcome::Selected(vec!["Alpha".into()]),
                    },
                    Effect::Command(Command::Init),
                ],
            )
            .unwrap();

        assert_eq!(rx.try_recv(), Ok(command));
        assert_eq!(rx.try_recv(), Ok(Command::Init));
        assert!(rx.try_recv().is_err());
        assert!(app.chat.elicitation.is_none());
        assert!(app.chat.messages.iter().any(|entry| matches!(
            entry,
            ChatEntry::Elicitation { outcome: Some(outcome), .. }
                if outcome == &ElicitationResponseOutcome::Selected(vec!["Alpha".into()])
        )));
    }

    #[test]
    fn effect_executor_elicitation_send_failure_keeps_elicitation_open() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        executor
            .execute(
                &mut terminal,
                &mut app,
                vec![Effect::ElicitationResponse {
                    elicitation_id: "test-id".into(),
                    action: "decline".into(),
                    content: None,
                    outcome: ElicitationResponseOutcome::Declined,
                }],
            )
            .unwrap();

        assert!(app.chat.elicitation.is_some());
        assert!(
            app.chat
                .messages
                .iter()
                .any(|entry| matches!(entry, ChatEntry::Elicitation { outcome: None, .. }))
        );
        assert!(app.diagnostics.status.contains("channel closed"));
    }

    #[test]
    #[serial]
    fn effect_executor_persists_handler_settings_only_to_guarded_path() {
        let guard = crate::config::TestPersistenceGuard::new("effect-executor-persist");
        let guarded_path = guard.path().to_path_buf();
        assert_eq!(config::TuiConfig::config_path(), guarded_path);
        assert_ne!(
            guarded_path,
            crate::config::TestPersistenceGuard::user_config_path()
        );
        assert!(!guarded_path.exists());

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ProfileSelect;
        app.profiles.profiles = vec![crate::domain::profile::ProfileInfo {
            id: "fast".into(),
            name: "Fast".into(),
            ..Default::default()
        }];
        app.chat.show_thinking = false;

        let effects = handle_profile_popup_key(&mut app, key(KeyCode::Enter));
        assert!(matches!(
            effects.as_slice(),
            [
                Effect::Command(Command::ListProfileAgents { profile_id }),
                Effect::PersistConfig,
            ] if profile_id == "fast"
        ));
        executor.execute(&mut terminal, &mut app, effects).unwrap();

        assert_eq!(
            rx.try_recv(),
            Ok(Command::ListProfileAgents {
                profile_id: "fast".into()
            })
        );
        assert!(rx.try_recv().is_err());
        assert!(guarded_path.exists());
        let persisted = config::TuiConfig::load_from_path(&guarded_path);
        assert_eq!(persisted.profile.id.as_deref(), Some("fast"));
        assert_eq!(persisted.show_thinking, Some(false));
    }

    #[test]
    fn effect_executor_failed_send_preserves_owned_command() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let executor = EffectExecutor::new(&tx);
        let command = Command::Prompt {
            prompt: vec![PromptBlock::Text {
                text: "owned payload".into(),
            }],
            local_id: "local-owned".into(),
        };

        let event = executor
            .send_command(command.clone())
            .expect("closed receiver should reject command");

        assert!(matches!(
            event,
            RuntimeEvent::CommandFailed {
                command: failed,
                message,
            } if failed == command && message.contains("channel closed")
        ));
    }

    #[test]
    fn effect_executor_routes_command_failure_back_through_update() {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = App::new();
        let local_id = app.chat.push_pending_prompt("pending".into());

        executor
            .execute(
                &mut terminal,
                &mut app,
                vec![Effect::Command(Command::Prompt {
                    prompt: vec![],
                    local_id: local_id.clone(),
                })],
            )
            .unwrap();

        assert!(app.chat.messages.iter().all(|entry| {
            !matches!(entry, ChatEntry::User { message_id: Some(id), .. } if id == &local_id)
        }));
        assert!(app.diagnostics.status.contains("channel closed"));
    }

    fn make_elicitation_single_select() -> ElicitationState {
        ElicitationState::new_for_test(vec![ElicitationField {
            name: "choice".into(),
            title: "Pick one".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![
                    ElicitationOption {
                        value: serde_json::json!("a"),
                        label: "Alpha".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("b"),
                        label: "Beta".into(),
                        description: None,
                    },
                ],
            },
        }])
    }

    fn make_app_with_elicitation(state: ElicitationState) -> App {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("sess-1".into());
        app.chat.messages.push(ChatEntry::Elicitation {
            elicitation_id: state.elicitation_id.clone(),
            message: state.message.clone(),
            source: state.source.clone(),
            outcome: None,
        });
        app.chat.elicitation = Some(state);
        app.chat.elicitation_ui = Some(crate::chat_state::ElicitationUiState::default());
        app
    }

    fn make_boolean_elicitation(required: bool) -> ElicitationState {
        ElicitationState::new_for_test(vec![ElicitationField {
            name: "confirm".into(),
            title: "Confirm".into(),
            description: None,
            required,
            kind: ElicitationFieldKind::BooleanToggle,
        }])
    }

    fn make_multi_select_elicitation() -> ElicitationState {
        ElicitationState::new_for_test(vec![ElicitationField {
            name: "choices".into(),
            title: "Pick several".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::MultiSelect {
                options: vec![
                    ElicitationOption {
                        value: serde_json::json!("a"),
                        label: "Alpha".into(),
                        description: None,
                    },
                    ElicitationOption {
                        value: serde_json::json!("b"),
                        label: "Beta".into(),
                        description: None,
                    },
                ],
            },
        }])
    }

    // ── Elicitation key handling ──────────────────────────────────────────────

    #[test]
    fn elicitation_down_moves_option_cursor() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let mut effects = TestEffects::default();
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Down)));
        assert_eq!(app.chat.elicitation_ui.as_ref().unwrap().option_cursor, 1);
    }

    #[test]
    fn elicitation_up_does_not_go_below_zero() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let mut effects = TestEffects::default();
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Up)));
        assert_eq!(app.chat.elicitation_ui.as_ref().unwrap().option_cursor, 0);
    }

    #[test]
    fn elicitation_enter_on_single_select_defers_resolution_until_send_ack() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        assert!(handle_elicitation_key(&mut app, key(KeyCode::Down)).is_empty());

        let effects = handle_elicitation_key(&mut app, key(KeyCode::Enter));

        assert_eq!(
            effects,
            vec![Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "choice": "b" })),
                outcome: ElicitationResponseOutcome::Selected(vec!["Beta".into()]),
            }]
        );
        assert!(app.chat.elicitation.is_some());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { outcome: None, .. }]
        ));
    }

    #[test]
    fn elicitation_multi_select_preserves_wire_and_label_order() {
        let mut app = make_app_with_elicitation(make_multi_select_elicitation());
        let mut effects = TestEffects::default();

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Down)));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(' '))));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Up)));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(' '))));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));

        assert_eq!(
            effects.as_slice(),
            &[Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "choices": ["b", "a"] })),
                outcome: ElicitationResponseOutcome::Selected(vec!["Alpha".into(), "Beta".into(),]),
            }]
        );
    }

    #[test]
    fn elicitation_other_opens_multiline_editor_and_submits_custom_answer() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let mut effects = TestEffects::default();

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Down)));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Down)));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));
        assert!(
            app.chat
                .elicitation_ui
                .as_ref()
                .is_some_and(|ui| ui.custom_active)
        );

        for c in "custom".chars() {
            effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(c))));
        }
        effects.extend(handle_elicitation_key(
            &mut app,
            modified_key(KeyCode::Enter, KeyModifiers::SHIFT),
        ));
        for c in "answer".chars() {
            effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(c))));
        }
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));

        assert!(app.chat.elicitation.is_some());
        assert_eq!(
            effects.as_slice(),
            &[Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "choice": "custom\nanswer" })),
                outcome: ElicitationResponseOutcome::Selected(vec!["custom\nanswer".into()]),
            }]
        );
    }

    #[test]
    fn elicitation_custom_esc_returns_to_choices_before_declining() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let mut effects = TestEffects::default();
        for _ in 0..2 {
            effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Down)));
        }
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));
        app.render.test_seed_elicitation_custom_geometry(19, 3);
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Esc)));

        assert!(
            app.chat
                .elicitation_ui
                .as_ref()
                .is_some_and(|ui| !ui.custom_active)
        );
        assert_eq!(app.render.test_elicitation_custom_geometry(), (19, 0, true));
        assert!(effects.next_command().is_none());

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Esc)));
        assert!(app.chat.elicitation.is_some());
        assert_eq!(
            effects.as_slice(),
            &[Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "decline".into(),
                content: None,
                outcome: ElicitationResponseOutcome::Declined,
            }]
        );
    }

    #[test]
    fn elicitation_empty_custom_answer_stays_open() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let mut effects = TestEffects::default();
        for _ in 0..2 {
            effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Down)));
        }
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(' '))));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));

        assert!(app.chat.elicitation.is_some());
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn elicitation_esc_defers_decline_resolution_until_send_ack() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());

        let effects = handle_elicitation_key(&mut app, key(KeyCode::Esc));

        assert_eq!(
            effects,
            vec![Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "decline".into(),
                content: None,
                outcome: ElicitationResponseOutcome::Declined,
            }]
        );
        assert!(app.chat.elicitation.is_some());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { outcome: None, .. }]
        ));
    }

    #[test]
    fn elicitation_enter_on_text_field_sends_accept_with_text() {
        let mut app =
            make_app_with_elicitation(ElicitationState::new_for_test(vec![ElicitationField {
                name: "name".into(),
                title: "Name".into(),
                description: None,
                required: true,
                kind: ElicitationFieldKind::TextInput,
            }]));
        app.chat.elicitation.as_mut().unwrap().text_input = "Alice".into();
        let effects = handle_elicitation_key(&mut app, key(KeyCode::Enter));

        assert_eq!(
            effects,
            vec![Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "name": "Alice" })),
                outcome: ElicitationResponseOutcome::Text("Alice".into()),
            }]
        );
        assert!(app.chat.elicitation.is_some());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::Elicitation { outcome: None, .. }]
        ));
    }

    #[test]
    fn elicitation_enter_on_number_field_preserves_text_outcome_and_numeric_wire_value() {
        let mut app =
            make_app_with_elicitation(ElicitationState::new_for_test(vec![ElicitationField {
                name: "count".into(),
                title: "Count".into(),
                description: None,
                required: true,
                kind: ElicitationFieldKind::NumberInput { integer: true },
            }]));
        app.chat.elicitation.as_mut().unwrap().text_input = "42".into();

        let effects = handle_elicitation_key(&mut app, key(KeyCode::Enter));

        assert_eq!(
            effects,
            vec![Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "count": 42 })),
                outcome: ElicitationResponseOutcome::Text("42".into()),
            }]
        );
    }

    #[test]
    fn elicitation_char_input_appends_to_text_buffer() {
        let mut app =
            make_app_with_elicitation(ElicitationState::new_for_test(vec![ElicitationField {
                name: "msg".into(),
                title: "Message".into(),
                description: None,
                required: false,
                kind: ElicitationFieldKind::TextInput,
            }]));
        let mut effects = TestEffects::default();
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char('H'))));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char('i'))));
        assert_eq!(app.chat.elicitation.as_ref().unwrap().text_input, "Hi");
    }

    #[test]
    fn elicitation_backspace_removes_last_char_from_text_buffer() {
        let mut app =
            make_app_with_elicitation(ElicitationState::new_for_test(vec![ElicitationField {
                name: "msg".into(),
                title: "Message".into(),
                description: None,
                required: false,
                kind: ElicitationFieldKind::TextInput,
            }]));
        app.chat.elicitation.as_mut().unwrap().text_input = "Hi".into();
        app.chat.elicitation_ui.as_mut().unwrap().text_cursor = 2;
        let mut effects = TestEffects::default();
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Backspace)));
        assert_eq!(app.chat.elicitation.as_ref().unwrap().text_input, "H");
    }

    #[test]
    fn elicitation_enter_on_required_boolean_without_toggle_does_not_submit() {
        let mut app = make_app_with_elicitation(make_boolean_elicitation(true));
        let mut effects = TestEffects::default();

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));

        assert!(app.chat.elicitation.is_some(), "popup should remain open");
        assert!(
            effects.next_command().is_none(),
            "no response should be sent"
        );
        assert!(
            app.chat
                .messages
                .iter()
                .any(|m| matches!(m, ChatEntry::Elicitation { outcome: None, .. }))
        );
    }

    #[test]
    fn elicitation_boolean_space_toggles_true_then_enter_submits() {
        let mut app = make_app_with_elicitation(make_boolean_elicitation(true));
        let mut effects = TestEffects::default();

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(' '))));
        assert_eq!(
            app.chat
                .elicitation
                .as_ref()
                .and_then(|state| state.selected.get("confirm")),
            Some(&serde_json::json!(true))
        );

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));

        assert!(app.chat.elicitation.is_some());
        assert_eq!(
            effects.as_slice(),
            &[Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "confirm": true })),
                outcome: ElicitationResponseOutcome::Boolean(true),
            }]
        );
    }

    #[test]
    fn elicitation_boolean_second_space_toggles_false_and_still_submits() {
        let mut app = make_app_with_elicitation(make_boolean_elicitation(true));
        let mut effects = TestEffects::default();

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(' '))));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(' '))));
        assert_eq!(
            app.chat
                .elicitation
                .as_ref()
                .and_then(|state| state.selected.get("confirm")),
            Some(&serde_json::json!(false))
        );

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));

        assert!(app.chat.elicitation.is_some());
        assert_eq!(
            effects.as_slice(),
            &[Effect::ElicitationResponse {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "confirm": false })),
                outcome: ElicitationResponseOutcome::Boolean(false),
            }]
        );
    }

    #[test]
    fn elicitation_key_handler_ignores_empty_field_list() {
        let mut app = make_app_with_elicitation(ElicitationState::new_for_test(vec![]));
        let mut effects = TestEffects::default();

        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Down)));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char(' '))));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Char('x'))));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Backspace)));
        effects.extend(handle_elicitation_key(&mut app, key(KeyCode::Enter)));

        assert!(app.chat.elicitation.is_some());
        assert!(
            effects.next_command().is_none(),
            "no response should be sent"
        );
    }

    #[test]
    #[serial]
    fn theme_selection_clears_all_render_caches() {
        use crate::theme::Theme;

        Theme::set_by_index(0);
        Theme::begin_frame();

        let mut app = App::new();
        app.chat.messages.push(ChatEntry::User {
            text: "hello".into(),
            message_id: None,
        });
        app.chat.messages.push(ChatEntry::ToolCall {
            tool_call_id: None,
            name: "edit".into(),
            is_error: false,
            detail: ToolDetail::Edit {
                file: "f.rs".into(),
                old: "aaa".into(),
                new: "bbb".into(),
                replace_all: false,
                start_line: None,
            },
        });

        let old_preview_fg = crate::ui::build_message_cards(&mut app)
            .iter()
            .find_map(|card| {
                let lines = card.lines_for(120);
                lines
                    .iter()
                    .flat_map(|line| line.spans.iter())
                    .find(|span| span.content.as_ref() == "f.rs")
                    .and_then(|span| span.style.fg)
            })
            .expect("edit preview should contain a styled file span");

        app.render
            .test_seed_streaming_cache(crate::render_state::StreamKind::Content);
        app.render
            .test_seed_streaming_cache(crate::render_state::StreamKind::Thinking);

        assert!(
            app.render
                .test_streaming_cache_populated(crate::render_state::StreamKind::Content)
        );
        assert!(
            app.render
                .test_streaming_cache_populated(crate::render_state::StreamKind::Thinking)
        );
        assert_eq!(
            app.render.test_card_source_entry_count(),
            app.chat.messages.len(),
            "card_cache should be populated"
        );

        app.navigation.popup = crate::navigation_state::Popup::ThemeSelect;
        app.navigation.theme_cursor = 2;
        let effects = handle_theme_popup_key(&mut app, key(KeyCode::Enter));
        assert_eq!(effects, vec![Effect::PersistConfig]);
        assert_eq!(app.navigation.popup, crate::navigation_state::Popup::None);
        let current_preview_fg = Theme::diff_file().fg.expect("diff_file should define fg");
        assert_ne!(old_preview_fg, current_preview_fg);

        assert_eq!(
            app.render.test_card_source_entry_count(),
            0,
            "card_cache should be invalidated"
        );
        assert!(
            !app.render
                .test_streaming_cache_populated(crate::render_state::StreamKind::Content),
            "streaming_cache should be invalidated"
        );
        assert!(
            !app.render
                .test_streaming_cache_populated(crate::render_state::StreamKind::Thinking),
            "streaming_thinking_cache should be invalidated"
        );
        assert!(matches!(
            &app.chat.messages[1],
            ChatEntry::ToolCall {
                detail: ToolDetail::Edit { old, new, .. },
                ..
            } if old == "aaa" && new == "bbb"
        ));

        let rebuilt_preview_fg = crate::ui::build_message_cards(&mut app)
            .iter()
            .find_map(|card| {
                let lines = card.lines_for(120);
                lines
                    .iter()
                    .flat_map(|line| line.spans.iter())
                    .find(|span| span.content.as_ref() == "f.rs")
                    .and_then(|span| span.style.fg)
            })
            .expect("rebuilt edit preview should contain a styled file span");
        assert_eq!(rebuilt_preview_fg, current_preview_fg);
        assert_ne!(rebuilt_preview_fg, old_preview_fg);

        Theme::set_by_index(0);
        Theme::begin_frame();
    }
}

#[cfg(test)]
mod external_editor_tests {
    use super::*;
    use crate::app::App;
    use crate::command::PromptBlock;
    use crate::config::{AcpConfig, TestPersistenceGuard, TuiConfig};
    use crate::domain::activity::{ActivityState, SessionOp};
    use crate::domain::chat::ChatEntry;
    use crate::handlers::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use serial_test::serial;

    fn ctrl_x() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL)
    }

    fn plain_key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::empty())
    }

    #[test]
    fn composer_visual_movement_uses_default_published_and_resized_render_widths() {
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "abc".into();
        app.composer.input_cursor = app.composer.input.len();

        handle_chat_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::empty()));
        assert_eq!(app.render.composer_input_line_width(), 1);
        assert_eq!(app.composer.input_cursor, 2);

        app.composer.input_cursor = app.composer.input.len();
        app.composer.input_preferred_col = None;
        app.render.prepare_composer_input_layout("abc", 3, 4, 2);
        handle_chat_key(&mut app, KeyEvent::new(KeyCode::Up, KeyModifiers::empty()));
        assert_eq!(app.composer.input_cursor, 1);

        app.composer.input_preferred_col = None;
        app.render.prepare_composer_input_layout("abc", 1, 20, 2);
        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::empty()),
        );
        assert_eq!(app.composer.input_cursor, 1);
        assert_eq!(app.composer.input_preferred_col, Some(1));
    }

    #[test]
    fn chat_up_down_navigate_wrapped_input_without_scrolling_history() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "abcdef".into();
        app.composer.input_cursor = 4;
        app.render.prepare_composer_input_layout("abcdef", 4, 4, 2);
        app.render.set_chat_scroll_offset(7);

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Up, KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input_cursor, 2);
        assert_eq!(app.render.chat_scroll_offset(), 7);

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input_cursor, 4);
        assert_eq!(app.render.chat_scroll_offset(), 7);
    }

    #[test]
    fn chat_pageup_pagedown_still_scroll_history_while_input_is_blocked() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.render.set_chat_scroll_offset(3);

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::PageUp, KeyModifiers::empty()),
        ));
        assert_eq!(app.render.chat_scroll_offset(), 13);

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::PageDown, KeyModifiers::empty()),
        ));
        assert_eq!(app.render.chat_scroll_offset(), 3);
    }

    #[test]
    fn delegate_keys_preserve_exact_shared_viewport_steps() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Delegate;
        app.render.set_chat_scroll_offset(10);

        for (key, expected) in [
            (KeyCode::Up, 11),
            (KeyCode::Down, 10),
            (KeyCode::PageUp, 20),
            (KeyCode::PageDown, 10),
            (KeyCode::Home, u16::MAX),
            (KeyCode::End, 0),
        ] {
            effects.extend(handle_key(
                &mut app,
                KeyEvent::new(key, KeyModifiers::empty()),
            ));
            assert_eq!(app.render.chat_scroll_offset(), expected);
        }
    }

    #[test]
    fn chat_end_routes_between_composer_and_viewport_contextually() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "draft".into();
        app.composer.input_cursor = 1;
        app.render.set_chat_scroll_offset(7);

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::End, KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input_cursor, app.composer.input.len());
        assert_eq!(app.render.chat_scroll_offset(), 7);

        app.composer.input.clear();
        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::End, KeyModifiers::empty()),
        ));
        assert_eq!(app.render.chat_scroll_offset(), 0);

        app.composer.input = "blocked".into();
        app.composer.input_cursor = 1;
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.render.set_chat_scroll_offset(5);
        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::End, KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input_cursor, 1);
        assert_eq!(app.render.chat_scroll_offset(), 0);
    }

    #[test]
    fn ctrl_x_e_returns_open_editor_action_in_chat() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "draft".into();
        assert!(handle_key(&mut app, ctrl_x()).is_empty());

        effects.extend(handle_key(&mut app, plain_key('e')));

        assert!(matches!(
            effects.as_slice(),
            [Effect::OpenExternalEditor { initial_text }] if initial_text == "draft"
        ));
        assert!(!app.navigation.chord);
        assert_eq!(app.composer.input, "draft");
    }

    #[test]
    fn ctrl_x_e_outside_chat_stays_in_tui() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;
        assert!(handle_key(&mut app, ctrl_x()).is_empty());

        effects.extend(handle_key(&mut app, plain_key('e')));

        assert!(effects.is_empty());
        assert!(app.diagnostics.status.contains("only available in chat"));
        assert!(matches!(app.diagnostics.logs.last(), Some(entry) if entry.target == "editor"));
    }

    #[test]
    fn ctrl_x_m_outside_chat_does_not_open_model_popup() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;
        assert!(handle_key(&mut app, ctrl_x()).is_empty());

        effects.extend(handle_key(&mut app, plain_key('m')));

        assert!(effects.is_empty());
        assert_ne!(app.navigation.popup, Popup::ModelSelect);
        assert!(app.diagnostics.status.contains("only available in chat"));
        assert!(matches!(app.diagnostics.logs.last(), Some(entry) if entry.target == "model"));
    }

    #[test]
    fn log_server_binary_discovery_records_path_lookup_when_binary_path_unset() {
        let mut app = App::new();
        let cfg = TuiConfig {
            acp: AcpConfig {
                binary_path: None,
                ..AcpConfig::default()
            },
            ..TuiConfig::default()
        };

        log_server_binary_discovery(
            &mut app,
            &cfg,
            &server_manager::BinaryDiscovery {
                binary: None,
                configured_path: None,
                configured_exists: false,
                used_path_lookup: true,
            },
        );

        assert!(
            app.diagnostics
                .logs
                .iter()
                .any(|entry| entry.target == "acp"
                    && entry.level == LogLevel::Info
                    && entry.message == "acp.binary_path not set; checking qmtcode on PATH")
        );
        assert!(
            app.diagnostics
                .logs
                .iter()
                .any(|entry| entry.target == "acp"
                    && entry.level == LogLevel::Info
                    && entry.message == "qmtcode not found on PATH")
        );
    }

    #[test]
    fn chat_input_accepts_typing_and_submit_while_turn_active() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.chat.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::empty()),
        ));
        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(matches!(
            effects.next_command().expect("prompt sent"),
            Command::Prompt { prompt, local_id }
                if local_id.starts_with("local:pending:")
                    && matches!(prompt.as_slice(), [PromptBlock::Text { text }] if text == "n")
        ));
        assert!(app.composer.input.is_empty());
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "n" && message_id.starts_with("local:pending:")
        ));
    }

    #[test]
    fn chat_submit_normalizes_prompt_before_sending_and_rendering() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.composer.input = "  first line\nsecond line\n  ".into();
        app.composer.input_cursor = app.composer.input.len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(matches!(
            effects.next_command().expect("prompt sent"),
            Command::Prompt { prompt, .. }
                if matches!(prompt.as_slice(), [PromptBlock::Text { text }]
                    if text == "first line\nsecond line")
        ));
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::User { text, .. }] if text == "first line\nsecond line"
        ));
    }

    #[test]
    fn disconnected_chat_submit_retains_prompt() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "retained @src/main.rs".into();
        app.composer.input_cursor = app.composer.input.len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(effects.next_command().is_none());
        assert_eq!(app.composer.input, "retained @src/main.rs");
        assert!(app.chat.messages.is_empty());
    }

    #[test]
    fn chat_submit_preserves_resource_link_order_after_text() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.composer.input = "  inspect @src/main.rs then @src/lib.rs and @src/main.rs  ".into();
        app.composer.input_cursor = app.composer.input.len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(matches!(
            effects.next_command().expect("prompt sent"),
            Command::Prompt { prompt, .. }
                if matches!(prompt.as_slice(), [
                    PromptBlock::Text { text },
                    PromptBlock::ResourceLink { name: first_name, uri: first_uri },
                    PromptBlock::ResourceLink { name: second_name, uri: second_uri },
                ] if text == "inspect @src/main.rs then @src/lib.rs and @src/main.rs"
                    && first_name == "src/main.rs"
                    && first_uri == "src/main.rs"
                    && second_name == "src/lib.rs"
                    && second_uri == "src/lib.rs")
        ));
        assert!(matches!(
            app.chat.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "inspect @src/main.rs then @src/lib.rs and @src/main.rs"
                    && message_id.starts_with("local:pending:")
        ));
    }

    #[test]
    fn whitespace_only_chat_submit_does_not_send_or_render_prompt() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.composer.input = " \n  ".into();
        app.composer.input_cursor = app.composer.input.len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(effects.next_command().is_none());
        assert!(app.chat.messages.is_empty());
        assert!(app.composer.input.is_empty());
    }

    // ── slash command handler tests ────────────────────────────────────────────

    #[test]
    fn left_arrow_with_slash_input_does_not_crash() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/model".into();
        app.composer.input_cursor = "/model".len();
        app.composer.refresh_slash_state();
        assert!(app.composer.slash_state.is_some());

        // hold left until cursor reaches 0 — must not panic at any step
        for _ in 0..=app.composer.input.len() {
            effects.extend(handle_chat_key(
                &mut app,
                KeyEvent::new(KeyCode::Left, KeyModifiers::empty()),
            ));
        }
        assert_eq!(app.composer.input_cursor, 0);
        assert!(app.composer.slash_state.is_none());
    }

    #[test]
    fn slash_esc_clears_slash_state() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/mo".into();
        app.composer.input_cursor = 3;
        app.composer.refresh_slash_state();
        assert!(app.composer.slash_state.is_some());

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
        ));

        assert!(app.composer.slash_state.is_none());
    }

    #[test]
    fn slash_enter_opens_help_popup() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/help".into();
        app.composer.input_cursor = "/help".len();
        app.composer.refresh_slash_state();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.navigation.popup, Popup::Help);
        assert!(app.composer.input.is_empty());
    }

    #[test]
    fn slash_enter_with_partial_completion_executes_command() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/hel".into();
        app.composer.input_cursor = "/hel".len();
        app.composer.refresh_slash_state();
        assert!(app.composer.slash_state.is_some());

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.navigation.popup, Popup::Help);
        assert!(app.composer.input.is_empty());
        assert!(app.composer.slash_state.is_none());
    }

    #[test]
    fn slash_tab_completes_command_name_without_executing() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/hel".into();
        app.composer.input_cursor = "/hel".len();
        app.composer.refresh_slash_state();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::empty()),
        ));

        // Completed but not executed — no popup opened
        assert_eq!(app.composer.input, "/help ");
        assert_eq!(app.navigation.popup, Popup::None);
        assert!(app.composer.slash_state.is_none());
    }

    #[test]
    fn slash_down_up_navigates_selection() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/".into();
        app.composer.input_cursor = 1;
        app.composer.refresh_slash_state();
        let initial = app.composer.slash_state.as_ref().unwrap().selected_index;

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::empty()),
        ));
        assert_eq!(
            app.composer.slash_state.as_ref().unwrap().selected_index,
            initial + 1
        );

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Up, KeyModifiers::empty()),
        ));
        assert_eq!(
            app.composer.slash_state.as_ref().unwrap().selected_index,
            initial
        );
    }

    #[test]
    #[serial]
    fn slash_mode_no_arg_cycles_mode() {
        let _guard = TestPersistenceGuard::new("slash-mode-cycle");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.sessions.agent_mode = "build".into();
        app.composer.input = "/mode".into();
        app.composer.input_cursor = "/mode".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.sessions.agent_mode, "plan");
        assert!(app.composer.input.is_empty());
        // SetAgentMode should have been sent
        assert!(matches!(
            effects.next_command().expect("SetAgentMode sent"),
            Command::SetAgentMode { mode } if mode == "plan"
        ));
    }

    #[test]
    #[serial]
    fn slash_mode_plan_switches_to_plan() {
        let _guard = TestPersistenceGuard::new("slash-mode-plan");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.sessions.agent_mode = "build".into();
        app.composer.input = "/mode plan".into();
        app.composer.input_cursor = "/mode plan".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.sessions.agent_mode, "plan");
        assert!(matches!(
            effects.next_command().expect("SetAgentMode sent"),
            Command::SetAgentMode { mode } if mode == "plan"
        ));
    }

    #[test]
    fn slash_mode_same_is_idempotent() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.sessions.agent_mode = "build".into();
        app.composer.input = "/mode build".into();
        app.composer.input_cursor = "/mode build".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.sessions.agent_mode, "build");
        assert!(app.diagnostics.status.contains("already in build"));
    }

    #[test]
    fn slash_mode_unknown_shows_error() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.composer.input = "/mode xyz".into();
        app.composer.input_cursor = "/mode xyz".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(app.diagnostics.status.contains("unknown mode"));
    }

    #[test]
    #[serial]
    fn slash_thinking_high_sets_level() {
        let _guard = TestPersistenceGuard::new("slash-thinking-high");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.composer.input = "/thinking high".into();
        app.composer.input_cursor = "/thinking high".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.models.reasoning_effort, Some("high".into()));
        assert!(matches!(
            effects.next_command().expect("SetReasoningEffort sent"),
            Command::SetReasoningEffort { reasoning_effort } if reasoning_effort == "high"
        ));
    }

    #[test]
    #[serial]
    fn slash_thinking_auto_clears_level() {
        let _guard = TestPersistenceGuard::new("slash-thinking-auto");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.models.reasoning_effort = Some("max".into());
        app.composer.input = "/thinking auto".into();
        app.composer.input_cursor = "/thinking auto".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.models.reasoning_effort, None);
        assert!(matches!(
            effects.next_command().expect("SetReasoningEffort sent"),
            Command::SetReasoningEffort { reasoning_effort } if reasoning_effort == "auto"
        ));
    }

    #[test]
    #[serial]
    fn slash_thinking_med_alias_sets_medium() {
        let _guard = TestPersistenceGuard::new("slash-thinking-med");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.composer.input = "/thinking med".into();
        app.composer.input_cursor = "/thinking med".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.models.reasoning_effort, Some("medium".into()));
        assert!(matches!(
            effects.next_command().expect("SetReasoningEffort sent"),
            Command::SetReasoningEffort { reasoning_effort } if reasoning_effort == "medium"
        ));
    }

    #[test]
    fn slash_thinking_no_arg_shows_current() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.models.reasoning_effort = Some("high".into());
        app.composer.input = "/thinking".into();
        app.composer.input_cursor = "/thinking".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(app.diagnostics.status.contains("thinking: high"));
    }

    #[test]
    fn slash_thinking_unknown_shows_error() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer.input = "/thinking xyz".into();
        app.composer.input_cursor = "/thinking xyz".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(app.diagnostics.status.contains("unknown level"));
    }

    #[test]
    fn slash_thinking_when_disconnected_does_not_change_state() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.models.reasoning_effort = Some("high".into());
        app.composer.input = "/thinking max".into();
        app.composer.input_cursor = "/thinking max".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        // state must not change when disconnected
        assert_eq!(app.models.reasoning_effort, Some("high".into()));
        assert!(app.diagnostics.status.contains("not connected"));
    }

    fn app_with_forkable_messages() -> App {
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.chat.messages = vec![
            ChatEntry::User {
                text: "alpha prompt".into(),
                message_id: Some("user-1".into()),
            },
            ChatEntry::Assistant {
                content: "alpha reply".into(),
                thinking: None,
                message_id: Some("asst-1".into()),
            },
            ChatEntry::User {
                text: "beta prompt".into(),
                message_id: Some("user-2".into()),
            },
        ];
        app
    }

    #[test]
    fn slash_fork_sends_latest_boundary() {
        let mut effects = TestEffects::default();
        let mut app = app_with_forkable_messages();
        app.composer.input = "/fork".into();
        app.composer.input_cursor = "/fork".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(matches!(
            effects.next_command().expect("ForkSession sent"),
            Command::ForkSession { message_id } if message_id == "user-2"
        ));
        assert_eq!(app.chat.pending_fork_message_id.as_deref(), Some("user-2"));
    }

    #[test]
    fn ctrl_x_f_opens_fork_popup_and_filter_captures_text() {
        let mut effects = TestEffects::default();
        let mut app = app_with_forkable_messages();

        effects.extend(handle_key(&mut app, ctrl_x()));
        effects.extend(handle_key(&mut app, plain_key('f')));
        effects.extend(handle_key(&mut app, plain_key('b')));

        assert_eq!(app.navigation.popup, Popup::ForkTurnSelect);
        assert_eq!(app.chat.fork_filter, "b");
        assert!(app.composer.input.is_empty());
        assert_eq!(app.chat.filtered_fork_turns().len(), 1);
    }

    #[test]
    fn ctrl_x_f_in_delegate_view_does_not_open_fork_popup() {
        let mut effects = TestEffects::default();
        let mut app = app_with_forkable_messages();
        app.navigation.screen = Screen::Delegate;

        effects.extend(handle_key(&mut app, ctrl_x()));
        effects.extend(handle_key(&mut app, plain_key('f')));

        assert_ne!(app.navigation.popup, Popup::ForkTurnSelect);
        assert!(effects.next_command().is_none());
        assert!(app.diagnostics.status.contains("only available in chat"));
        assert!(matches!(app.diagnostics.logs.last(), Some(entry) if entry.target == "fork"));
    }

    #[test]
    fn slash_fork_in_delegate_view_sends_nothing() {
        let mut effects = TestEffects::default();
        let mut app = app_with_forkable_messages();
        app.navigation.screen = Screen::Delegate;
        app.composer.input = "/fork".into();
        app.composer.input_cursor = "/fork".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(effects.next_command().is_none());
        assert!(app.chat.pending_fork_message_id.is_none());
        assert!(app.diagnostics.status.contains("only available in chat"));
        assert!(matches!(app.diagnostics.logs.last(), Some(entry) if entry.target == "fork"));
    }

    #[test]
    fn fork_popup_enter_sends_selected_turn() {
        let mut effects = TestEffects::default();
        let mut app = app_with_forkable_messages();
        app.open_fork_turn_popup();
        app.chat.fork_filter = "alpha".into();

        effects.extend(handle_fork_turn_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(matches!(
            effects.next_command().expect("ForkSession sent"),
            Command::ForkSession { message_id } if message_id == "asst-1"
        ));
        assert_eq!(app.chat.pending_fork_message_id.as_deref(), Some("asst-1"));
    }

    #[test]
    fn fork_popup_enter_with_default_cursor_sends_latest_visible_turn() {
        let mut effects = TestEffects::default();
        let mut app = app_with_forkable_messages();
        app.open_fork_turn_popup();

        effects.extend(handle_fork_turn_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(matches!(
            effects.next_command().expect("ForkSession sent"),
            Command::ForkSession { message_id } if message_id == "user-2"
        ));
        assert_eq!(app.chat.pending_fork_message_id.as_deref(), Some("user-2"));
    }

    #[test]
    fn fork_popup_enter_with_no_eligible_turns_sends_nothing() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.open_fork_turn_popup();

        effects.extend(handle_fork_turn_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert!(effects.next_command().is_none());
        assert!(app.diagnostics.status.contains("no forkable turns"));
    }

    #[test]
    fn slash_model_with_arg_prefilters_popup() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.composer.input = "/model sonnet".into();
        app.composer.input_cursor = "/model sonnet".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.navigation.popup, Popup::ModelSelect);
        assert_eq!(app.models.model_filter, "sonnet");
        assert_eq!(app.models.model_cursor, 0);
    }

    #[test]
    fn slash_model_no_arg_opens_popup_unfiltered() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.composer.input = "/model".into();
        app.composer.input_cursor = "/model".len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));

        assert_eq!(app.navigation.popup, Popup::ModelSelect);
        assert!(app.models.model_filter.is_empty());
    }

    #[test]
    fn chat_completion_dismissal_precedes_double_esc_cancellation() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.chat.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };
        app.composer.input = "/mo".into();
        app.composer.input_cursor = app.composer.input.len();
        app.composer.refresh_slash_state();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
        ));
        assert!(app.composer.slash_state.is_none());
        assert!(!app.chat.cancel_confirm_active());

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
        ));
        assert!(app.chat.cancel_confirm_active());

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
        ));
        assert!(matches!(
            effects.next_command().expect("cancel sent"),
            Command::CancelSession
        ));
        assert_eq!(app.diagnostics.status, "stopping...");
        assert!(matches!(
            app.diagnostics.logs.last(),
            Some(entry) if entry.target == "activity" && entry.message == "stopping..."
        ));
    }

    #[test]
    fn chat_input_is_blocked_while_undo_is_pending() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.composer.input = "draft".into();
        app.composer.input_cursor = app.composer.input.len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input, "draft");

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input, "draft");

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Left, KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input_cursor, "draft".len());

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input, "draft");
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn chat_input_is_blocked_while_cancel_confirm_is_active() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.chat.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };
        app.composer.input = "draft".into();
        app.composer.input_cursor = app.composer.input.len();

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
        ));
        assert!(app.chat.cancel_confirm_active());
        assert!(app.chat.input_blocked_by_activity());

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::empty()),
        ));
        assert_eq!(app.composer.input, "draft");

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
        ));
        assert_eq!(app.diagnostics.status, "press Esc again to stop");
        assert!(effects.next_command().is_none());

        effects.extend(handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
        ));
        assert_eq!(app.diagnostics.status, "stopping...");
        assert!(matches!(
            effects.next_command().expect("cancel sent"),
            Command::CancelSession
        ));
    }
}

fn log_server_binary_discovery(
    app: &mut App,
    cfg: &config::TuiConfig,
    discovery: &server_manager::BinaryDiscovery,
) {
    if !discovery.used_path_lookup {
        return;
    }
    if let Some(path) = discovery.configured_path.as_deref() {
        app.push_log(
            LogLevel::Info,
            "acp",
            format!("configured qmtcode path not found: {path}; checking PATH"),
        );
    } else if cfg.acp.binary_path.is_none() {
        app.push_log(
            LogLevel::Info,
            "acp",
            "acp.binary_path not set; checking qmtcode on PATH",
        );
    }
    if discovery.binary.is_none() {
        app.push_log(LogLevel::Info, "acp", "qmtcode not found on PATH");
    }
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load persistent config; CLI args override config defaults.
    let cfg = config::TuiConfig::load();

    // Apply saved theme (falls back to built-in default if absent or unknown).
    let theme_id = cfg.theme.as_deref().unwrap_or("base16-querymate");
    theme::Theme::init(theme_id);

    // channels for the event loop
    let (srv_tx, mut srv_rx) = mpsc::unbounded_channel::<ServerChannelMsg>();
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<Command>();
    let (conn_tx, mut conn_rx) = mpsc::unbounded_channel::<ConnectionManagerEvent>();

    let mut app = App::new();
    app.connection.launch_cwd = detect_launch_cwd();
    app.profiles.active_profile_id = cfg.profile.id.clone();
    app.chat.show_thinking = cfg.show_thinking.unwrap_or(true);
    app.models.delegate_model_preferences = cfg.profile_delegate_models.clone();
    if let Some(profile_id) = cfg.profile.id.as_deref() {
        let legacy_preferences = app
            .models
            .delegate_model_preferences
            .entry(profile_id.to_string())
            .or_default();
        for (agent_id, model_key) in &cfg.delegate_models {
            if legacy_preferences.contains_key(agent_id) {
                continue;
            }
            let Some((provider, model)) = model_key.split_once('/') else {
                continue;
            };
            legacy_preferences.insert(
                agent_id.clone(),
                DelegateModelPreference {
                    model_id: model_key.clone(),
                    provider: provider.to_string(),
                    model: model.to_string(),
                    node_id: None,
                },
            );
        }
    }
    if let Some(session_id) = cli.session.clone() {
        app.sessions.session_id = Some(session_id);
        app.navigation.screen = Screen::Chat;
    }
    // -- ACP auto-start ---------------------------------------------------------
    let (sup_event_tx, mut sup_event_rx) = mpsc::unbounded_channel::<server_manager::ServerEvent>();
    let default_ws_available = cli.ws.is_none()
        && cli.acp_websocket.is_none()
        && cli.acp_binary.is_none()
        && cfg.acp.websocket_url.is_none()
        && cfg.acp.transport.unwrap_or_default() != config::AcpTransportMode::WebSocket
        && cfg.acp.auto_start.unwrap_or(true)
        && acp::probe_websocket(&default_acp_ws_url(), Duration::from_millis(250)).await;

    let selection = select_acp_endpoint(&cli, &cfg, default_ws_available);
    if let EndpointSelection::Endpoint {
        discovered_ws: Some(url),
        ..
    } = &selection
    {
        app.push_log(
            LogLevel::Info,
            "acp",
            format!("found ACP WebSocket server at {url}"),
        );
    }
    if cli.acp_binary.is_none()
        && cfg.acp.transport.unwrap_or_default() == config::AcpTransportMode::Stdio
    {
        let discovery = server_manager::find_binary_info(cfg.acp.binary_path.as_deref());
        if matches!(
            selection,
            EndpointSelection::Endpoint {
                endpoint: acp::AcpEndpoint::Stdio { .. },
                ..
            } | EndpointSelection::Endpoint {
                missing_binary_fallback: true,
                ..
            } | EndpointSelection::BinaryNotFound
        ) {
            log_server_binary_discovery(&mut app, &cfg, &discovery);
        }
    }
    if let EndpointSelection::Endpoint {
        missing_binary_fallback: true,
        endpoint: acp::AcpEndpoint::WebSocket { url },
        ..
    } = &selection
    {
        app.push_log(
            LogLevel::Warn,
            "acp",
            format!("qmtcode unavailable; waiting for ACP WebSocket at {url}"),
        );
    }

    let (endpoint, initial_server_state) = match selection {
        EndpointSelection::Endpoint {
            endpoint,
            state,
            discovered_ws: _,
            missing_binary_fallback: _,
        } => (Some(endpoint), state),
        EndpointSelection::BinaryNotFound => {
            let _ = sup_event_tx.send(server_manager::ServerEvent::BinaryNotFound);
            (None, ServerState::BinaryNotFound)
        }
        EndpointSelection::Disabled => (None, ServerState::Disabled),
    };

    if let Some(endpoint) = endpoint {
        tokio::spawn(connection_manager(
            endpoint,
            srv_tx,
            cmd_rx,
            conn_tx,
            sup_event_tx.clone(),
            app.connection.launch_cwd.clone(),
        ));
    }

    let mut terminal = terminal::enter()?;

    app.connection.server_state = initial_server_state;
    let mut executor = EffectExecutor::new(&cmd_tx);
    let result = run_loop(
        &mut terminal,
        &mut app,
        &mut srv_rx,
        &mut conn_rx,
        &mut sup_event_rx,
        &mut executor,
    )
    .await;

    terminal::leave(&mut terminal)?;

    if let Some(session_id) = &app.sessions.session_id {
        eprintln!("{}", restore_hint(session_id));
    }

    result
}

fn restore_hint(session_id: &str) -> String {
    use clap::CommandFactory;
    let bin = Cli::command().get_name().to_string();
    format!("{bin} -s {session_id}")
}

#[cfg(test)]
struct PersistenceGuard(config::TestPersistenceGuard);

#[cfg(test)]
impl PersistenceGuard {
    fn new(label: &str) -> Self {
        Self(config::TestPersistenceGuard::new(label))
    }
}

#[cfg(test)]
mod sessions_key_tests {
    use super::*;
    use crate::command::Command;
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::handlers::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn make_group(cwd: Option<&str>, ids: &[&str]) -> SessionGroup {
        SessionGroup {
            cwd: cwd.map(String::from),
            latest_activity: None,
            sessions: ids
                .iter()
                .map(|id| SessionSummary {
                    session_id: id.to_string(),
                    title: None,
                    cwd: None,
                    created_at: None,
                    updated_at: None,
                    parent_session_id: None,
                    has_children: false,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn make_group_with_cursor(cwd: Option<&str>, ids: &[&str], next_cursor: &str) -> SessionGroup {
        let mut group = make_group(cwd, ids);
        group.next_cursor = Some(next_cursor.to_string());
        group
    }

    // ── Enter on Session loads it ─────────────────────────────────────────────

    #[test]
    fn enter_on_session_emits_one_load_and_one_subscribe() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.sessions.agent_id = Some("agent-1".into());
        app.sessions.session_groups = vec![make_group(Some("/a"), &["abc12345"])];
        app.sessions.session_cursor = 1;
        let mut effects = TestEffects::default();

        effects.extend(handle_sessions_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        ));

        assert!(matches!(
            effects.next_command(),
            Some(Command::LoadSession { session_id, cwd: Some(cwd) })
                if session_id == "abc12345" && cwd == "/a"
        ));
        assert!(matches!(
            effects.next_command(),
            Some(Command::SubscribeSession { session_id, agent_id })
                if session_id == "abc12345" && agent_id.as_deref() == Some("agent-1")
        ));
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn enter_on_known_remote_emits_only_exact_attach_command() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(Some("/a"), &["remote-1"])];
        app.sessions.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.sessions.session_cursor = 1;
        let mut effects = TestEffects::default();

        effects.extend(handle_sessions_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        ));

        assert!(matches!(
            effects.next_command(),
            Some(Command::AttachRemoteSession { node_id, session_id })
                if node_id == "node-1" && session_id == "remote-1"
        ));
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn enter_on_remote_label_without_node_id_is_noop() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(Some("/a"), &["remote-1"])];
        app.sessions.session_groups[0].sessions[0].node = Some("remote node".into());
        app.sessions.session_cursor = 1;

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(
            app.diagnostics.status,
            "remote session is missing node id; refresh sessions and try again"
        );
    }

    #[test]
    fn enter_on_remote_session_remembers_node_id_for_reconnect() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group(Some("/a"), &["remote-1"])];
        app.sessions.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.sessions.session_cursor = 1;

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "remote-1".into(),
            }
        );
        assert_eq!(
            app.sessions.session_remote_node_id("remote-1"),
            Some("node-1")
        );
    }

    #[test]
    fn ctrl_o_on_expandable_root_emits_exact_child_list_command() {
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.sessions.session_groups[0].sessions[0].fork_count = 1;
        app.sessions.session_cursor = 1;

        let effects = handle_sessions_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
        );

        assert!(app.sessions.expanded_session_children.contains("root"));
        assert_eq!(
            effects,
            vec![Effect::Command(Command::ListSessionChildren {
                parent_session_id: "root".into(),
                cursor: None,
                limit: 10,
            })]
        );
    }

    #[test]
    fn ctrl_o_on_expanded_root_collapses_without_effects() {
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.sessions.session_groups[0].sessions[0].fork_count = 1;
        app.sessions.expanded_session_children.insert("root".into());
        app.sessions.session_cursor = 1;

        let effects = handle_sessions_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
        );

        assert!(!app.sessions.expanded_session_children.contains("root"));
        assert!(effects.is_empty());
    }

    #[test]
    fn delete_on_remote_session_emits_exact_dismiss_and_updates_projection() {
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["remote-1", "local-1"])];
        app.sessions.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.sessions.session_cursor = 1;

        let effects =
            handle_sessions_key(&mut app, KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));

        assert_eq!(
            effects,
            vec![Effect::Command(Command::DismissRemoteSession {
                session_id: "remote-1".into(),
            })]
        );
        assert_eq!(app.sessions.session_groups[0].sessions.len(), 1);
        assert_eq!(
            app.sessions.session_groups[0].sessions[0].session_id,
            "local-1"
        );
    }

    #[test]
    fn delete_on_local_session_emits_exact_delete_and_updates_projection() {
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["local-1", "local-2"])];
        app.sessions.session_cursor = 1;

        let effects =
            handle_sessions_key(&mut app, KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));

        assert_eq!(
            effects,
            vec![Effect::Command(Command::DeleteSession {
                session_id: "local-1".into(),
            })]
        );
        assert_eq!(app.sessions.session_groups[0].sessions.len(), 1);
        assert_eq!(
            app.sessions.session_groups[0].sessions[0].session_id,
            "local-2"
        );
    }

    #[test]
    fn enter_on_empty_start_page_button_opens_new_session_without_effects() {
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;

        let effects =
            handle_sessions_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert!(effects.is_empty());
    }

    // ── ShowMore Enter opens session popup ────────────────────────────────────

    #[test]
    fn enter_on_show_more_opens_session_popup() {
        let mut app = App::new();
        // 4 sessions -> header(0) + 3 sessions + ShowMore(4)
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3", "s4"])];
        app.sessions.session_cursor = 4; // ShowMore row
        let action = apply_sessions_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(app.sessions.session_cursor, 0);
        assert!(app.sessions.session_filter.is_empty());
    }

    #[test]
    fn enter_on_show_more_with_backend_cursor_still_only_opens_popup() {
        let mut app = App::new();
        app.sessions.session_groups = vec![make_group_with_cursor(
            Some("/workspace/project"),
            &["s1", "s2", "s3"],
            "cursor-1",
        )];
        app.sessions.session_cursor = 4; // ShowMore row created by next_cursor

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(app.sessions.session_popup_tab, 0);
        assert_eq!(app.sessions.session_cursor, 0);
    }

    // ── q quits ───────────────────────────────────────────────────────────────
    // (q is handled in handle_sessions_key, not apply_sessions_key — tested
    //  via the existing integration path)
}

#[cfg(test)]
mod session_popup_key_tests {
    use super::*;
    use crate::command::{Command, SessionListRequest};
    use crate::domain::activity::{
        DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus,
    };
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::handlers::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn make_group(cwd: Option<&str>, ids: &[&str]) -> SessionGroup {
        SessionGroup {
            cwd: cwd.map(String::from),
            latest_activity: None,
            sessions: ids
                .iter()
                .map(|id| SessionSummary {
                    session_id: id.to_string(),
                    title: Some(format!("Session {id}")),
                    cwd: cwd.map(String::from),
                    created_at: None,
                    updated_at: None,
                    parent_session_id: None,
                    has_children: false,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    fn make_group_with_cursor(cwd: Option<&str>, ids: &[&str], next_cursor: &str) -> SessionGroup {
        let mut group = make_group(cwd, ids);
        group.next_cursor = Some(next_cursor.to_string());
        group
    }

    struct NewSessionTempDir(std::path::PathBuf);

    impl NewSessionTempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};

            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "qmt-runtime-new-session-{label}-{}-{nanos}-{id}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for NewSessionTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // ── Down / Up navigation ──────────────────────────────────────────────────

    #[test]
    fn popup_page_down_uses_visible_rows_with_overlap() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(
            Some("/a"),
            &["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        )];
        app.render.publish_session_popup_visible_rows(4);

        assert!(
            handle_session_popup_key(
                &mut app,
                KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE),
            )
            .is_empty()
        );
        assert_eq!(app.sessions.session_cursor, 3);

        assert!(
            handle_session_popup_key(
                &mut app,
                KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE),
            )
            .is_empty()
        );
        assert_eq!(app.sessions.session_cursor, 6);
    }

    #[test]
    fn popup_page_up_uses_visible_rows_with_overlap() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(
            Some("/a"),
            &["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        )];
        app.render.publish_session_popup_visible_rows(4);
        app.sessions.session_cursor = 6;

        assert!(
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE),)
                .is_empty()
        );
        assert_eq!(app.sessions.session_cursor, 3);

        assert!(
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE),)
                .is_empty()
        );
        assert_eq!(app.sessions.session_cursor, 0);
    }

    #[test]
    fn popup_page_keys_fallback_to_single_row_when_visible_rows_unknown() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3"])];

        assert!(
            handle_session_popup_key(
                &mut app,
                KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE),
            )
            .is_empty()
        );
        assert_eq!(app.sessions.session_cursor, 1);

        assert!(
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE),)
                .is_empty()
        );
        assert_eq!(app.sessions.session_cursor, 0);
    }

    // ── Enter on Session loads it ─────────────────────────────────────────────

    #[test]
    fn popup_header_toggle_routes_through_root_adapter() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Sessions;
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_popup_tab = 0;
        app.sessions.session_groups = vec![
            make_group(Some("/work"), &["one"]),
            make_group(Some("/work"), &["two"]),
        ];
        app.sessions.collapsed_groups.insert("/start-only".into());
        let start_page_collapsed_groups = app.sessions.collapsed_groups.clone();
        app.sessions.session_cursor = 2;
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::empty());

        effects.extend(handle_session_popup_key(&mut app, enter));

        assert!(effects.effects.is_empty());
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert!(app.sessions.popup_collapsed_groups.contains("/work"));
        assert_eq!(app.sessions.collapsed_groups, start_page_collapsed_groups);
        assert_eq!(app.sessions.session_cursor, 1);

        effects.extend(handle_session_popup_key(&mut app, enter));

        assert!(effects.effects.is_empty());
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert!(!app.sessions.popup_collapsed_groups.contains("/work"));
        assert_eq!(app.sessions.collapsed_groups, start_page_collapsed_groups);
        assert_eq!(app.sessions.session_cursor, 1);
        assert!(app.sessions.session_cursor < app.sessions.visible_popup_items().len());
    }

    #[test]
    fn popup_enter_on_local_session_emits_load_and_subscribe() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.agent_id = Some("agent-1".into());
        app.sessions.session_groups = vec![make_group(Some("/a"), &["abc12345"])];
        app.sessions.session_groups[0].sessions[0].cwd = None;
        app.sessions.session_cursor = 1;

        let effects =
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(
            effects,
            vec![
                Effect::Command(Command::LoadSession {
                    session_id: "abc12345".into(),
                    cwd: Some("/a".into()),
                }),
                Effect::Command(Command::SubscribeSession {
                    session_id: "abc12345".into(),
                    agent_id: Some("agent-1".into()),
                }),
            ]
        );
    }

    #[test]
    fn popup_enter_on_known_remote_session_emits_exact_attach() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["remote-1"])];
        app.sessions.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.sessions.session_cursor = 1;

        let effects =
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(
            effects,
            vec![Effect::Command(Command::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "remote-1".into(),
            })]
        );
        assert_eq!(
            app.sessions.session_remote_node_id("remote-1"),
            Some("node-1")
        );
    }

    #[test]
    fn popup_enter_on_missing_remote_session_warns_without_command() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["remote-1"])];
        app.sessions.session_groups[0].sessions[0].node = Some("remote node".into());
        app.sessions.session_cursor = 1;

        let effects =
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(
            app.diagnostics.status,
            "remote session is missing node id; refresh sessions and try again"
        );
        assert!(effects.is_empty());
    }

    #[test]
    fn popup_delete_on_remote_session_emits_dismiss_and_removes() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["remote-1", "s2"])];
        app.sessions.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.sessions.session_cursor = 1;

        let effects =
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));

        assert_eq!(
            effects,
            vec![Effect::Command(Command::DismissRemoteSession {
                session_id: "remote-1".into(),
            })]
        );
        assert_eq!(app.sessions.session_groups[0].sessions.len(), 1);
        assert_eq!(app.sessions.session_groups[0].sessions[0].session_id, "s2");
    }

    // ── Enter with all groups shows all sessions (no cap) ─────────────────────

    #[test]
    fn popup_enter_can_reach_session_beyond_start_page_cap() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        // 5 sessions — start page would cap at 3; popup shows all
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3", "s4", "s5"])];
        // visible: [Header(0), s1(1), s2(2), s3(3), s4(4), s5(5)]
        app.sessions.session_cursor = 5;
        let action = apply_popup_session_key(&mut app, KeyCode::Enter);
        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "s5".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
    }

    #[test]
    fn popup_enter_on_expandable_root_loads_session() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.sessions.session_groups[0].sessions[0].fork_count = 1;
        app.sessions.session_cursor = 1;

        let action = apply_popup_session_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "root".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
        assert!(!app.sessions.expanded_session_children.contains("root"));
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn popup_ctrl_o_on_expandable_root_requests_children() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.sessions.session_groups[0].sessions[0].fork_count = 1;
        app.sessions.session_cursor = 1;
        let effects = handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
        );

        assert!(app.sessions.expanded_session_children.contains("root"));
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(
            effects,
            vec![Effect::Command(Command::ListSessionChildren {
                parent_session_id: "root".into(),
                cursor: None,
                limit: 10,
            })]
        );
    }

    #[test]
    fn popup_ctrl_o_on_expanded_root_collapses_without_loading_session() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.sessions.session_groups[0].sessions[0].fork_count = 1;
        app.sessions
            .expanded_session_children
            .insert("root".to_string());
        app.sessions.session_cursor = 1;
        let effects = handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
        );

        assert!(!app.sessions.expanded_session_children.contains("root"));
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert!(effects.is_empty());
    }

    #[test]
    fn popup_enter_on_expanded_child_loads_session() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.sessions.session_groups[0].sessions[0].fork_count = 1;
        app.sessions.session_groups[0].sessions[0].children = vec![SessionSummary {
            session_id: "child".to_string(),
            title: Some("child".to_string()),
            parent_session_id: Some("root".to_string()),
            ..Default::default()
        }];
        app.sessions
            .expanded_session_children
            .insert("root".to_string());
        app.sessions.session_cursor = 2;

        let action = apply_popup_session_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "child".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn popup_enter_on_load_more_emits_exact_continuation_and_keeps_popup_open() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group_with_cursor(
            Some("/workspace/project"),
            &["s1"],
            "cursor-1",
        )];
        app.sessions.session_cursor = 2;

        let effects =
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(
            effects,
            vec![Effect::Command(Command::ListSessions {
                request: SessionListRequest::WorkspaceContinuation {
                    cwd: "/workspace/project".into(),
                },
                cursor: Some("cursor-1".into()),
            })]
        );
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
    }

    // ── Delete on Session removes it ─────────────────────────────────────────

    #[test]
    fn popup_delete_on_local_session_emits_delete_and_removes() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.sessions.session_cursor = 1;

        let effects =
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));

        assert_eq!(
            effects,
            vec![Effect::Command(Command::DeleteSession {
                session_id: "s1".into(),
            })]
        );
        assert_eq!(app.sessions.session_groups[0].sessions.len(), 1);
        assert_eq!(app.sessions.session_groups[0].sessions[0].session_id, "s2");
    }

    // ── Esc closes popup ──────────────────────────────────────────────────────

    #[test]
    fn popup_esc_closes_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;

        let effects =
            handle_session_popup_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        assert_eq!(app.navigation.popup, Popup::None);
        assert!(effects.is_empty());
    }

    #[test]
    fn popup_ctrl_n_when_disconnected_keeps_session_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;

        let effects = handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL),
        );

        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );
        assert!(effects.is_empty());
    }

    #[test]
    fn popup_ctrl_n_opens_new_session_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.connection.conn = ConnState::Connected;
        app.connection.launch_cwd = Some("/launch".into());
        let mut effects = TestEffects::default();
        effects.extend(handle_session_popup_key(
            &mut app,
            KeyEvent {
                code: KeyCode::Char('n'),
                modifiers: KeyModifiers::CONTROL,
                kind: crossterm::event::KeyEventKind::Press,
                state: crossterm::event::KeyEventState::NONE,
            },
        ));

        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert_eq!(app.sessions.new_session_path, "/launch");
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn global_ctrl_x_n_opens_new_session_popup() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.connection.launch_cwd = Some("/launch".into());
        let mut effects = TestEffects::default();

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
        ));
        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
        ));

        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert_eq!(app.sessions.new_session_path, "/launch");
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn global_ctrl_x_l_opens_session_popup() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        let mut effects = TestEffects::default();

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
        ));
        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE),
        ));

        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(app.sessions.session_popup_tab, 0);
    }

    #[test]
    fn global_ctrl_l_opens_log_popup() {
        let mut app = App::new();
        let mut effects = TestEffects::default();

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL),
        ));

        assert_eq!(app.navigation.popup, Popup::Log);
        assert_eq!(app.diagnostics.log_cursor, 0);
        assert!(app.diagnostics.log_filter.is_empty());
    }

    #[test]
    fn log_popup_filters_cycles_level_and_closes() {
        let mut app = App::new();
        app.navigation.popup = Popup::Log;
        app.diagnostics.log_cursor = 2;
        app.diagnostics.log_level_filter = LogLevel::Info;
        let mut effects = TestEffects::default();
        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
        ));
        assert_eq!(app.diagnostics.log_filter, "x");
        assert_eq!(app.diagnostics.log_cursor, 0);

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));
        assert!(app.diagnostics.log_filter.is_empty());

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        ));
        assert_eq!(app.diagnostics.log_level_filter, LogLevel::Warn);

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
        ));
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn new_session_popup_edited_input_refreshes_completion() {
        let temp_dir = NewSessionTempDir::new("edited-refresh");
        let directory = temp_dir.path().join("project-dir");
        let file = temp_dir.path().join("project-file.txt");
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(&file, "not a directory").unwrap();

        let mut app = App::new();
        app.navigation.popup = Popup::NewSession;
        app.connection.launch_cwd = Some(temp_dir.path().to_string_lossy().into_owned());
        app.sessions.new_session_path = "project-".into();
        app.sessions.new_session_cursor = app.sessions.new_session_path.len();

        let effects = handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE),
        );

        assert_eq!(effects, Vec::<Effect>::new());
        assert_eq!(app.navigation.popup, Popup::NewSession);
        let completion = app.sessions.new_session_completion.as_ref().unwrap();
        assert_eq!(completion.query, "project-d");
        assert!(
            completion
                .results
                .iter()
                .any(|entry| entry.path == directory.to_string_lossy() && entry.is_dir)
        );
        assert!(
            !completion
                .results
                .iter()
                .any(|entry| entry.path == file.to_string_lossy())
        );
    }

    #[test]
    fn new_session_popup_navigation_and_unsupported_keys_are_noops_at_root() {
        let mut app = App::new();
        app.navigation.popup = Popup::NewSession;
        app.sessions.new_session_path = "pro".into();
        app.sessions.new_session_cursor = app.sessions.new_session_path.len();
        app.sessions.new_session_completion = Some(crate::session_state::PathCompletionState {
            query: "pro".into(),
            selected_index: 0,
            results: vec![
                crate::composer_state::FileIndexEntryLite {
                    path: "/launch/project-one".into(),
                    is_dir: true,
                },
                crate::composer_state::FileIndexEntryLite {
                    path: "/launch/project-two".into(),
                    is_dir: true,
                },
            ],
        });

        let effects = handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
        );

        assert_eq!(effects, Vec::<Effect>::new());
        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert_eq!(
            app.sessions
                .new_session_completion
                .as_ref()
                .unwrap()
                .selected_index,
            1
        );

        let path = app.sessions.new_session_path.clone();
        let cursor = app.sessions.new_session_cursor;
        let completion = app.sessions.new_session_completion.clone();
        let effects = handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE),
        );

        assert_eq!(effects, Vec::<Effect>::new());
        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert_eq!(app.sessions.new_session_path, path);
        assert_eq!(app.sessions.new_session_cursor, cursor);
        assert_eq!(app.sessions.new_session_completion, completion);
    }

    #[test]
    fn new_session_popup_escape_closes_popup() {
        let mut app = App::new();
        app.navigation.popup = Popup::NewSession;

        let effects =
            handle_new_session_popup_key(&mut app, KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        assert_eq!(effects, Vec::<Effect>::new());
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn new_session_popup_enter_with_empty_path_uses_launch_cwd() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::NewSession;
        app.connection.launch_cwd = Some("/launch".into());
        app.sessions.new_session_path.clear();
        app.sessions.new_session_cursor = 0;

        let effects = handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );

        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(
            effects,
            vec![Effect::Command(Command::NewSession {
                cwd: Some("/launch".into()),
                profile_id: None,
            })]
        );
    }

    #[test]
    fn disconnected_new_session_submit_keeps_popup_and_sends_no_command() {
        let mut app = App::new();
        app.navigation.popup = Popup::NewSession;
        app.sessions.new_session_path = "project".into();
        app.sessions.new_session_cursor = app.sessions.new_session_path.len();

        let effects = handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );

        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert_eq!(app.sessions.new_session_path, "project");
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );
        assert_eq!(effects, Vec::<Effect>::new());
    }

    #[test]
    fn new_session_popup_enter_normalizes_relative_path_to_absolute() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::NewSession;
        app.connection.launch_cwd = Some("/launch".into());
        app.sessions.new_session_path = "proj/subdir".into();
        app.sessions.new_session_cursor = app.sessions.new_session_path.len();

        let effects = handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );

        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(
            effects,
            vec![Effect::Command(Command::NewSession {
                cwd: Some("/launch/proj/subdir".into()),
                profile_id: None,
            })]
        );
    }

    #[test]
    fn new_session_popup_tab_accepts_selected_completion() {
        let mut app = App::new();
        app.navigation.popup = Popup::NewSession;
        app.sessions.new_session_completion = Some(crate::session_state::PathCompletionState {
            query: "pro".into(),
            selected_index: 0,
            results: vec![crate::composer_state::FileIndexEntryLite {
                path: "/launch/project/../project-two".into(),
                is_dir: true,
            }],
        });

        let effects =
            handle_new_session_popup_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));

        assert_eq!(effects, Vec::<Effect>::new());
        assert_eq!(app.navigation.popup, Popup::NewSession);
        assert_eq!(app.sessions.new_session_path, "/launch/project-two/");
        assert_eq!(
            app.sessions.new_session_cursor,
            "/launch/project-two/".len()
        );
        assert!(app.sessions.new_session_completion.is_none());
    }

    #[test]
    fn handle_key_routes_tab_to_new_session_popup_before_global_mode_switch() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::NewSession;
        app.sessions.agent_mode = "build".into();
        app.sessions.new_session_completion = Some(crate::session_state::PathCompletionState {
            query: "pro".into(),
            selected_index: 0,
            results: vec![crate::composer_state::FileIndexEntryLite {
                path: "/launch/project".into(),
                is_dir: true,
            }],
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        ));

        assert_eq!(app.sessions.new_session_path, "/launch/project/");
        assert!(app.sessions.new_session_completion.is_none());
        assert_eq!(app.sessions.agent_mode, "build");
        assert!(effects.next_command().is_none());
    }

    #[test]
    fn session_popup_delegate_tab_routes_through_root_handler() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_popup_tab = 1;
        app.sessions.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.sessions.session_filter = "session".into();
        let session_projection = app.sessions.visible_popup_items();
        app.delegates.delegate_cursor = 3;
        app.delegates.delegate_entries = vec![DelegateEntry {
            delegation_id: "delegate-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "Write docs".into(),
            status: DelegateStatus::Completed,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        }];

        let effects = handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE),
        );

        assert!(effects.is_empty());
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(app.sessions.session_popup_tab, 1);
        assert_eq!(app.delegates.delegate_filter, "d");
        assert_eq!(app.delegates.delegate_cursor, 0);
        assert_eq!(app.sessions.session_filter, "session");
        assert_eq!(app.sessions.visible_popup_items(), session_projection);
    }

    #[test]
    fn root_router_keeps_session_popup_tab_and_backtab_ahead_of_tab_handlers() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.agent_mode = "build".into();
        app.sessions.session_filter = "sessions-owner".into();
        app.sessions.session_cursor = 3;
        app.delegates.delegate_filter = "delegates-owner".into();
        app.delegates.delegate_cursor = 4;

        assert!(handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),).is_empty());
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(app.sessions.session_popup_tab, 1);
        assert_eq!(app.sessions.agent_mode, "build");
        assert_eq!(app.sessions.session_filter, "sessions-owner");
        assert_eq!(app.sessions.session_cursor, 3);
        assert_eq!(app.delegates.delegate_filter, "delegates-owner");
        assert_eq!(app.delegates.delegate_cursor, 4);

        assert!(
            handle_key(
                &mut app,
                KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            )
            .is_empty()
        );
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(app.sessions.session_popup_tab, 0);
        assert_eq!(app.sessions.agent_mode, "build");
        assert_eq!(app.sessions.session_filter, "sessions-owner");
        assert_eq!(app.sessions.session_cursor, 3);
        assert_eq!(app.delegates.delegate_filter, "delegates-owner");
        assert_eq!(app.delegates.delegate_cursor, 4);
    }
}

#[cfg(test)]
mod delegate_popup_key_tests {
    use super::*;
    use crate::domain::activity::{
        DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus,
    };
    use crate::handlers::*;
    use crossterm::event::KeyCode;

    fn make_entry(id: &str, objective: &str, child_sid: Option<&str>) -> DelegateEntry {
        DelegateEntry {
            delegation_id: id.into(),
            child_session_id: child_sid.map(String::from),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: objective.into(),
            status: DelegateStatus::Completed,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        }
    }

    fn setup_delegate_app() -> App {
        let mut app = App::new();
        app.sessions.session_id = Some("parent-1".into());
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_popup_tab = 1;
        app.delegates.delegate_entries = vec![
            make_entry("d1", "Build feature", Some("child-1")),
            make_entry("d2", "Fix tests", Some("child-2")),
            make_entry("d3", "Write docs", Some("child-3")),
        ];
        app
    }

    #[test]
    fn delegate_navigation_clamps_cursor_within_bounds() {
        let mut app = setup_delegate_app();
        apply_delegate_popup_key(&mut app, KeyCode::Up);
        assert_eq!(app.delegates.delegate_cursor, 0);

        apply_delegate_popup_key(&mut app, KeyCode::Down);
        apply_delegate_popup_key(&mut app, KeyCode::Down);
        apply_delegate_popup_key(&mut app, KeyCode::Down);
        assert_eq!(app.delegates.delegate_cursor, 2);

        apply_delegate_popup_key(&mut app, KeyCode::Up);
        assert_eq!(app.delegates.delegate_cursor, 1);
    }

    #[test]
    fn delegate_page_down_uses_visible_rows_with_overlap() {
        let mut app = setup_delegate_app();
        app.delegates.delegate_entries.extend([
            make_entry("d4", "Check logs", Some("child-4")),
            make_entry("d5", "Refactor code", Some("child-5")),
            make_entry("d6", "Polish UI", Some("child-6")),
            make_entry("d7", "Ship release", Some("child-7")),
        ]);
        app.render.publish_delegate_popup_visible_rows(4);

        apply_delegate_popup_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.delegates.delegate_cursor, 3);

        apply_delegate_popup_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.delegates.delegate_cursor, 6);
    }

    #[test]
    fn delegate_page_up_uses_visible_rows_with_overlap() {
        let mut app = setup_delegate_app();
        app.delegates.delegate_entries.extend([
            make_entry("d4", "Check logs", Some("child-4")),
            make_entry("d5", "Refactor code", Some("child-5")),
            make_entry("d6", "Polish UI", Some("child-6")),
            make_entry("d7", "Ship release", Some("child-7")),
        ]);
        app.render.publish_delegate_popup_visible_rows(4);
        app.delegates.delegate_cursor = 6;

        apply_delegate_popup_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.delegates.delegate_cursor, 3);

        apply_delegate_popup_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.delegates.delegate_cursor, 0);
    }

    #[test]
    fn delegate_page_keys_fallback_to_single_row_when_visible_rows_unknown() {
        let mut app = setup_delegate_app();

        apply_delegate_popup_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.delegates.delegate_cursor, 1);

        apply_delegate_popup_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.delegates.delegate_cursor, 0);
    }

    #[test]
    fn delegate_enter_loads_selected_child_session() {
        let mut app = setup_delegate_app();
        app.delegates.delegate_cursor = 1;
        let action = apply_delegate_popup_key(&mut app, KeyCode::Enter);
        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "child-2".into(),
                agent_id: Some("coder".into()),
                cwd: None,
            }
        );
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn delegate_enter_noop_when_child_session_is_unavailable() {
        let mut app = App::new();
        app.navigation.popup = Popup::SessionSelect;
        app.sessions.session_popup_tab = 1;
        app.delegates.delegate_entries = vec![DelegateEntry {
            delegation_id: "d1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "pending task".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        }];
        let action = apply_delegate_popup_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(
            app.diagnostics.status,
            "delegation still pending — no session to load"
        );
    }

    #[test]
    fn delegate_filter_updates_cursor_and_loads_filtered_result() {
        let mut app = setup_delegate_app();
        app.delegates.delegate_cursor = 2;
        for c in "docs".chars() {
            apply_delegate_popup_key(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.delegates.delegate_filter, "docs");
        assert_eq!(app.delegates.delegate_cursor, 0);
        assert_eq!(app.delegates.visible_entries().len(), 1);
        assert_eq!(app.delegates.visible_entries()[0].delegation_id, "d3");

        apply_delegate_popup_key(&mut app, KeyCode::Backspace);
        assert_eq!(app.delegates.delegate_filter, "doc");
        assert_eq!(app.delegates.delegate_cursor, 0);

        let action = apply_delegate_popup_key(&mut app, KeyCode::Enter);
        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "child-3".into(),
                agent_id: Some("coder".into()),
                cwd: None,
            }
        );
    }

    #[test]
    fn delegate_enter_loads_awaiting_input_child_session() {
        let mut app = setup_delegate_app();
        app.delegates.delegate_entries[0].status = DelegateStatus::InProgress;
        app.delegates.delegate_entries[0].child_state = DelegateChildState::PendingElicitation {
            elicitation_id: "elic-1".into(),
            message: "Need approval".into(),
            requested_schema: serde_json::json!({ "properties": {} }),
            source: "builtin:question".into(),
        };

        let action = apply_delegate_popup_key(&mut app, KeyCode::Enter);
        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "child-1".into(),
                agent_id: Some("coder".into()),
                cwd: None,
            }
        );
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn delegate_esc_closes_popup() {
        let mut app = setup_delegate_app();
        apply_delegate_popup_key(&mut app, KeyCode::Esc);
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn delegate_popup_enter_sets_parent_for_sibling_navigation() {
        let mut app = setup_delegate_app();
        // Simulate being in a child session (parent_session_id is set).
        app.delegates.parent_session_id = Some("parent-1".into());
        app.sessions.session_id = Some("child-old".into());

        let action = apply_delegate_popup_key(&mut app, KeyCode::Enter);
        assert!(
            matches!(action, SessionKeyAction::LoadSession { .. }),
            "enter must trigger LoadSession"
        );
        assert_eq!(
            app.delegates.pending_parent_session_id.as_deref(),
            Some("parent-1"),
            "pending_parent must be the real parent, not the child session_id"
        );
    }
}

#[cfg(test)]
mod chord_reasoning_effort_tests {
    use super::*;
    use crate::handlers::handle_key;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use serial_test::serial;

    fn ctrl_t() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL)
    }

    // ── Ctrl+t cycles reasoning effort and sends message ─────────────────────

    #[test]
    #[serial]
    fn ctrl_t_cycles_effort_and_sends_msg() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        assert_eq!(app.models.reasoning_effort, None);

        effects.extend(handle_key(&mut app, ctrl_t()));

        assert_eq!(app.models.reasoning_effort, Some("low".into()));
        let msg = effects
            .next_command()
            .expect("expected SetReasoningEffort message");
        match msg {
            Command::SetReasoningEffort { reasoning_effort } => {
                assert_eq!(reasoning_effort, "low");
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn ctrl_t_full_cycle_sends_auto_on_wrap() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.models.reasoning_effort = Some("max".into());

        effects.extend(handle_key(&mut app, ctrl_t()));

        assert_eq!(app.models.reasoning_effort, None);
        let msg = effects
            .next_command()
            .expect("expected SetReasoningEffort message");
        match msg {
            Command::SetReasoningEffort { reasoning_effort } => {
                assert_eq!(reasoning_effort, "auto");
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn ctrl_t_status_updated() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        effects.extend(handle_key(&mut app, ctrl_t()));
        // status should reflect the new level
        assert!(
            app.diagnostics.status.contains("low"),
            "expected status to mention 'low', got: {}",
            app.diagnostics.status
        );
    }

    #[test]
    fn ctrl_t_when_disconnected_does_not_change_state() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.models.reasoning_effort = Some("high".into());

        effects.extend(handle_key(&mut app, ctrl_t()));

        // state must not change when disconnected
        assert_eq!(app.models.reasoning_effort, Some("high".into()));
        assert!(app.diagnostics.status.contains("not connected"));
    }
}

#[cfg(test)]
mod reasoning_effort_integration_tests {
    use super::*;
    use crate::domain::{model::ModelEntry, profile::AgentInfo};
    use crate::handlers::handle_key;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use serial_test::serial;

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

    fn chord_key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn tab_key() -> KeyEvent {
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)
    }

    #[test]
    #[serial]
    fn ctrl_t_cycles_reasoning_effort() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
        ));

        assert_eq!(app.models.reasoning_effort, Some("low".into()));
        assert!(matches!(
            effects.next_command(),
            Some(Command::SetReasoningEffort { reasoning_effort }) if reasoning_effort == "low"
        ));
    }

    #[test]
    #[serial]
    fn tab_switches_mode_without_changing_model_or_effort() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.sessions.agent_mode = "build".into();
        app.models.current_provider = Some("anthropic".into());
        app.models.current_model = Some("claude-sonnet".into());
        app.models.reasoning_effort = Some("high".into());

        effects.extend(handle_key(&mut app, tab_key()));

        assert_eq!(app.sessions.agent_mode, "plan");
        assert_eq!(app.models.current_model.as_deref(), Some("claude-sonnet"));
        assert_eq!(app.models.reasoning_effort, Some("high".into()));

        let msgs: Vec<_> = std::iter::from_fn(|| effects.next_command()).collect();
        assert!(
            msgs.iter()
                .any(|m| matches!(m, Command::SetAgentMode { mode } if mode == "plan")),
            "expected SetAgentMode(plan): {msgs:?}"
        );
        assert!(
            !msgs
                .iter()
                .any(|m| matches!(m, Command::SetReasoningEffort { .. })),
            "no effort restore on mode switch: {msgs:?}"
        );
    }

    #[test]
    fn root_router_cycles_model_backtab_but_has_no_global_backtab_mode_switch() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.agent_mode = "build".into();
        app.models.models = vec![make_model("anthropic", "claude-sonnet")];
        app.models.agents = vec![
            AgentInfo {
                id: "primary".into(),
                name: "Primary".into(),
                description: None,
                capabilities: Vec::new(),
            },
            AgentInfo {
                id: "coder".into(),
                name: "Coder".into(),
                description: None,
                capabilities: Vec::new(),
            },
        ];
        app.models.model_popup_agent_tab = 0;
        let expected_delegate_cursor = app.delegate_model_cursor("coder");
        app.models.model_filter = "clear-on-tab-change".into();
        app.models.model_cursor = 9;

        assert!(
            handle_key(
                &mut app,
                KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            )
            .is_empty()
        );
        assert_eq!(app.navigation.popup, Popup::ModelSelect);
        assert_eq!(app.models.model_popup_agent_tab, 1);
        assert!(app.models.model_filter.is_empty());
        assert_eq!(app.models.model_cursor, expected_delegate_cursor);
        assert_eq!(app.sessions.agent_mode, "build");

        app.navigation.popup = Popup::None;
        app.models.model_filter = "unchanged-without-popup".into();
        assert!(
            handle_key(
                &mut app,
                KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            )
            .is_empty()
        );
        assert_eq!(app.navigation.popup, Popup::None);
        assert_eq!(app.sessions.agent_mode, "build");
        assert_eq!(app.models.model_popup_agent_tab, 1);
        assert_eq!(app.models.model_filter, "unchanged-without-popup");
    }

    #[test]
    #[serial]
    fn tab_no_cache_entry_leaves_model_and_effort_unchanged() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.sessions.agent_mode = "build".into();
        app.models.current_provider = Some("anthropic".into());
        app.models.current_model = Some("claude-sonnet".into());
        app.models.reasoning_effort = Some("high".into());
        // No plan cache entry

        effects.extend(handle_key(&mut app, tab_key()));

        // Mode switched but model/effort unchanged (no cache to restore from)
        assert_eq!(app.sessions.agent_mode, "plan");
        assert_eq!(app.models.reasoning_effort, Some("high".into()));
        assert_eq!(app.models.current_model.as_deref(), Some("claude-sonnet"));
        let msgs: Vec<_> = std::iter::from_fn(|| effects.next_command()).collect();
        assert!(
            !msgs
                .iter()
                .any(|m| matches!(m, Command::SetReasoningEffort { .. })),
            "no SetReasoningEffort expected: {msgs:?}"
        );
    }

    // ── Model select: drops effort to auto ────────────────────────────────────

    #[test]
    #[serial]
    fn ctrl_x_m_opens_model_popup_at_current_mode_model() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.connection.conn = ConnState::Connected;
        app.sessions.agent_mode = "plan".into();
        app.models.model_popup_agent_tab = 0;
        app.models.current_provider = Some("anthropic".into());
        app.models.current_model = Some("claude-sonnet".into());
        app.models.models = vec![
            make_model("anthropic", "claude-sonnet"),
            make_model("openai", "gpt-4o"),
            make_model("openai", "o3-mini"),
        ];

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
        ));
        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE),
        ));

        assert_eq!(app.navigation.popup, Popup::ModelSelect);
        assert_eq!(app.models.model_filter, "");
        let expected = app.models.model_popup_open_cursor();
        assert_eq!(app.models.model_cursor, expected);
    }

    #[test]
    #[serial]
    fn model_select_drops_effort_to_auto() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.agent_mode = "build".into();
        app.models.model_popup_agent_tab = 0;
        app.models.current_provider = Some("anthropic".into());
        app.models.current_model = Some("claude-sonnet".into());
        app.models.reasoning_effort = Some("high".into());
        app.models.models = vec![make_model("anthropic", "claude-opus")];
        app.models.model_cursor = app
            .models
            .visible_model_popup_items()
            .iter()
            .position(|i| matches!(i, crate::models_state::ModelPopupItem::Model { .. }))
            .unwrap();

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        ));

        assert_eq!(app.models.reasoning_effort, None);

        let msgs: Vec<_> = std::iter::from_fn(|| effects.next_command()).collect();
        assert!(
            msgs.iter().any(|m| matches!(
                m,
                Command::SetReasoningEffort { reasoning_effort }
                if reasoning_effort == "auto"
            )),
            "expected SetReasoningEffort(auto): {msgs:?}"
        );
    }

    #[test]
    #[serial]
    fn model_select_no_effort_msg_when_already_auto() {
        let _guard = PersistenceGuard::new("main-test");
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.sessions.session_id = Some("s1".into());
        app.navigation.popup = Popup::ModelSelect;
        app.sessions.agent_mode = "build".into();
        app.models.model_popup_agent_tab = 0;
        app.models.current_provider = Some("anthropic".into());
        app.models.current_model = Some("claude-sonnet".into());
        app.models.reasoning_effort = None; // already auto
        app.models.models = vec![make_model("anthropic", "claude-opus")];
        app.models.model_cursor = app
            .models
            .visible_model_popup_items()
            .iter()
            .position(|i| matches!(i, crate::models_state::ModelPopupItem::Model { .. }))
            .unwrap();

        effects.extend(handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        ));

        let msgs: Vec<_> = std::iter::from_fn(|| effects.next_command()).collect();
        assert!(
            !msgs
                .iter()
                .any(|m| matches!(m, Command::SetReasoningEffort { .. })),
            "no SetReasoningEffort when already auto: {msgs:?}"
        );
    }

    #[test]
    fn native_reasoning_effort_event_updates_state() {
        let mut app = App::new();
        app.handle_acp_event(AcpAppEvent::ReasoningEffort {
            reasoning_effort: Some("medium".into()),
        });
        assert_eq!(app.models.reasoning_effort, Some("medium".into()));
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::{Cli, restore_hint};
    use clap::CommandFactory;

    #[test]
    fn restore_hint_formats_correctly() {
        let bin = Cli::command().get_name().to_string();
        assert_eq!(restore_hint("abc-123-def"), format!("{bin} -s abc-123-def"));
    }
}

#[cfg(test)]
mod auth_tests {
    use super::*;
    use crate::auth_state::{AuthPanel, AuthUiNotice};
    use crate::command::Command;
    use crate::domain::auth::{
        AuthProviderEntry, OAuthFlow, OAuthFlowKind, OAuthResult, OAuthResultStatus, OAuthStatus,
    };
    use crate::handlers::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn make_provider(name: &str) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: name.to_lowercase(),
            display_name: name.to_string(),
            oauth_status: Some(OAuthStatus::NotAuthenticated),
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: Some(format!("{}_API_KEY", name.to_uppercase())),
            supports_oauth: true,
            preferred_method: None,
        }
    }

    fn make_oauth_only(name: &str) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: name.to_lowercase(),
            display_name: name.to_string(),
            oauth_status: Some(OAuthStatus::NotAuthenticated),
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: None,
            supports_oauth: true,
            preferred_method: None,
        }
    }

    fn make_api_key_only(name: &str) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: name.to_lowercase(),
            display_name: name.to_string(),
            oauth_status: None,
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: Some(format!("{}_API_KEY", name.to_uppercase())),
            supports_oauth: false,
            preferred_method: None,
        }
    }

    fn make_app_with_providers(providers: Vec<AuthProviderEntry>) -> App {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.auth.providers = providers;
        app.navigation.popup = Popup::ProviderAuth;
        app
    }

    // ── App state tests ───────────────────────────────────────────────────────

    #[test]
    fn open_auth_popup_resets_state() {
        let mut app = App::new();
        app.auth.cursor = 5;
        app.auth.filter = "test".into();
        app.auth.selected = Some(2);
        app.auth.api_key_input = "secret".into();
        app.auth.last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        app.open_auth_popup();
        assert_eq!(app.navigation.popup, Popup::ProviderAuth);
        assert_eq!(app.auth.cursor, 0);
        assert!(app.auth.filter.is_empty());
        assert!(app.auth.selected.is_none());
        assert!(app.auth.api_key_input.is_empty());
        assert!(app.auth.last_result.is_none());
        assert!(app.auth.ui_notice.is_none());
        assert!(app.auth.api_key_masked);
        assert_eq!(app.auth.panel, AuthPanel::List);
    }

    // ── Key handler tests: List panel ─────────────────────────────────────────

    #[test]
    fn auth_list_esc_closes_popup_when_no_selection() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Esc)));
        assert_eq!(app.navigation.popup, Popup::None);
    }

    #[test]
    fn auth_list_esc_clears_selection_when_selected() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth.selected = Some(0);
        app.auth.last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Esc)));
        assert_eq!(app.navigation.popup, Popup::ProviderAuth);
        assert!(app.auth.selected.is_none());
        assert!(app.auth.last_result.is_none());
        assert!(app.auth.ui_notice.is_none());
    }

    #[test]
    fn auth_list_down_up_navigates() {
        let mut app = make_app_with_providers(vec![
            make_provider("OpenAI"),
            make_provider("Groq"),
            make_provider("DeepSeek"),
        ]);
        let mut effects = TestEffects::default();
        assert_eq!(app.auth.cursor, 0);
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Down)));
        assert_eq!(app.auth.cursor, 1);
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Down)));
        assert_eq!(app.auth.cursor, 2);
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Down)));
        assert_eq!(app.auth.cursor, 2); // clamped
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Up)));
        assert_eq!(app.auth.cursor, 1);
    }

    #[test]
    fn auth_list_enter_on_api_key_only_opens_api_key_panel() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Enter)));
        assert_eq!(app.auth.panel, AuthPanel::ApiKeyInput);
        assert_eq!(app.auth.selected, Some(0));
        assert!(app.auth.last_result.is_none());
        assert!(app.auth.ui_notice.is_none());
    }

    #[test]
    fn auth_list_enter_on_oauth_only_starts_flow() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("codex".into()),
            success: true,
            message: "old notice".into(),
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Enter)));
        assert_eq!(app.auth.selected, Some(0));
        let msg = effects.next_command().expect("message sent");
        assert!(matches!(msg, Command::StartOAuthLogin { provider } if provider == "codex"));
        assert!(app.auth.ui_notice.is_none());
    }

    #[test]
    fn auth_list_enter_on_multi_method_selects_provider() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Enter)));
        assert_eq!(app.auth.selected, Some(0));
        assert_eq!(app.auth.panel, AuthPanel::List);
        assert!(app.auth.ui_notice.is_none());
    }

    #[test]
    fn auth_list_char_input_filters() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI"), make_provider("Groq")]);
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('g'))));
        assert_eq!(app.auth.filter, "g");
        assert_eq!(app.auth.cursor, 0);
    }

    #[test]
    fn auth_list_backspace_removes_filter() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth.filter = "op".into();
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Backspace)));
        assert_eq!(app.auth.filter, "o");
    }

    #[test]
    fn auth_list_ctrl_k_opens_api_key_panel() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, ctrl('k')));
        assert_eq!(app.auth.panel, AuthPanel::ApiKeyInput);
        assert_eq!(app.auth.selected, Some(0));
        assert!(app.auth.ui_notice.is_none());
    }

    #[test]
    fn auth_list_ctrl_o_starts_oauth() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let mut effects = TestEffects::default();
        effects.extend(handle_auth_popup_key(&mut app, ctrl('o')));
        let msg = effects.next_command().expect("message sent");
        assert!(matches!(msg, Command::StartOAuthLogin { provider } if provider == "openai"));
        assert!(app.auth.ui_notice.is_none());
    }

    // ── Key handler tests: API Key panel ──────────────────────────────────────

    #[test]
    fn auth_api_key_cursor_uses_utf8_byte_offsets() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('é'))));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('ß'))));
        assert_eq!(app.auth.api_key_input, "éß");
        assert_eq!(app.auth.api_key_cursor, 4);

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Left)));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('x'))));
        assert_eq!(app.auth.api_key_input, "éxß");
        assert_eq!(app.auth.api_key_cursor, 3);

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Backspace)));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Right)));
        assert_eq!(app.auth.api_key_input, "éß");
        assert_eq!(app.auth.api_key_cursor, 4);
    }

    #[test]
    fn auth_api_key_typing_and_submit() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('s'))));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('k'))));
        assert_eq!(app.auth.api_key_input, "sk");
        assert_eq!(app.auth.api_key_cursor, 2);

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Enter)));
        assert_eq!(
            effects.as_slice(),
            [Effect::Command(Command::SetApiToken {
                provider: "groq".into(),
                api_key: "sk".into(),
            })]
        );
    }

    #[test]
    fn auth_api_key_submit_trims_payload_without_mutating_input() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        app.auth.api_key_input = "  secret  ".into();
        app.auth.api_key_cursor = app.auth.api_key_input.len();

        assert_eq!(
            handle_auth_popup_key(&mut app, key(KeyCode::Enter)),
            vec![Effect::Command(Command::SetApiToken {
                provider: "groq".into(),
                api_key: "secret".into(),
            })]
        );
        assert_eq!(app.auth.api_key_input, "  secret  ");
    }

    #[test]
    fn disconnected_auth_submit_is_gated_before_dispatch() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.connection.conn = ConnState::Disconnected;
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        app.auth.api_key_input = "secret".into();
        app.auth.api_key_cursor = app.auth.api_key_input.len();

        assert!(handle_auth_popup_key(&mut app, key(KeyCode::Enter)).is_empty());
        assert_eq!(app.auth.api_key_input, "secret");
        assert_eq!(
            app.diagnostics.status,
            "not connected - waiting to reconnect"
        );
    }

    #[test]
    fn auth_api_key_backspace() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        app.auth.api_key_input = "abc".into();
        app.auth.api_key_cursor = 3;
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Backspace)));
        assert_eq!(app.auth.api_key_input, "ab");
        assert_eq!(app.auth.api_key_cursor, 2);
    }

    #[test]
    fn auth_api_key_esc_returns_to_list() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        app.auth.api_key_input = "draft".into();
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Esc)));
        assert_eq!(app.auth.panel, AuthPanel::List);
        assert!(app.auth.api_key_input.is_empty());
    }

    #[test]
    fn auth_api_key_tab_toggles_mask() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        assert!(app.auth.api_key_masked);
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Tab)));
        assert!(!app.auth.api_key_masked);
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Tab)));
        assert!(app.auth.api_key_masked);
    }

    #[test]
    fn auth_api_key_ctrl_d_sends_clear() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, ctrl('d')));
        let msg = effects.next_command().expect("message sent");
        assert!(matches!(msg, Command::ClearApiToken { provider } if provider == "groq"));
    }

    #[test]
    fn auth_api_key_empty_submit_does_nothing() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Enter)));
        assert!(effects.next_command().is_none()); // nothing sent
    }

    // ── Key handler tests: OAuth flow panel ───────────────────────────────────

    #[test]
    fn auth_oauth_esc_returns_to_list() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::OAuthFlow;
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Esc)));
        assert_eq!(app.auth.panel, AuthPanel::List);
        assert!(app.auth.oauth_flow.is_none());
    }

    #[test]
    fn auth_oauth_redirect_code_typing_and_submit() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::OAuthFlow;
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('c'))));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('o'))));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('d'))));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('e'))));
        assert_eq!(app.auth.oauth_response, "code");

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Enter)));
        let msg = effects.next_command().expect("message sent");
        assert!(matches!(
            msg,
            Command::CompleteOAuthLogin { flow_id, response }
            if flow_id == "f1" && response == "code"
        ));
    }

    #[test]
    fn auth_oauth_device_poll_enter_sends_empty_response() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::OAuthFlow;
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://example.com/device".into(),
            flow_kind: OAuthFlowKind::DevicePoll,
        });
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Enter)));
        let msg = effects.next_command().expect("message sent");
        assert!(matches!(
            msg,
            Command::CompleteOAuthLogin { flow_id, response }
            if flow_id == "f1" && response.is_empty()
        ));
    }

    #[test]
    fn auth_oauth_cursor_uses_utf8_byte_offsets() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::OAuthFlow;
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('é'))));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('ß'))));
        assert_eq!(app.auth.oauth_response, "éß");
        assert_eq!(app.auth.oauth_response_cursor, 4);

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Left)));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('x'))));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Backspace)));
        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Right)));
        assert_eq!(app.auth.oauth_response, "éß");
        assert_eq!(app.auth.oauth_response_cursor, 4);
    }

    // ── Native ACP event handling tests ───────────────────────────────────────

    #[test]
    fn native_initialized_event_clears_auth_ui_notice() {
        let mut app = App::new();
        app.auth.filter = "keep".into();
        app.auth.selected = Some(3);
        app.auth.panel = AuthPanel::ApiKeyInput;
        app.auth.last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth.clipboard_fallback = Some("https://example.com".into());
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });

        let cmds = app.handle_acp_event(AcpAppEvent::Initialized {
            agent_id: "agent-1".into(),
            agent_name: "Agent".into(),
            profiles: Vec::new(),
            active_profile_id: None,
            agent_mode: None,
            reasoning_effort: None,
        });

        assert!(cmds.is_empty());
        assert!(app.auth.ui_notice.is_none());
        assert_eq!(app.auth.filter, "keep");
        assert_eq!(app.auth.selected, Some(3));
        assert_eq!(app.auth.panel, AuthPanel::ApiKeyInput);
        assert!(app.auth.last_result.is_some());
        assert_eq!(
            app.auth.clipboard_fallback.as_deref(),
            Some("https://example.com")
        );
    }

    #[test]
    fn native_auth_providers_event_populates_list() {
        let mut app = App::new();
        app.auth.cursor = 7;
        app.auth.selected = Some(9);
        let mut openai = make_provider("OpenAI");
        openai.has_env_api_key = true;
        let mut groq = make_api_key_only("Groq");
        groq.has_stored_api_key = true;

        let cmds = app.handle_acp_event(AcpAppEvent::AuthProviders(vec![openai, groq]));

        assert!(cmds.is_empty());
        assert_eq!(app.auth.providers.len(), 2);
        assert_eq!(app.auth.providers[0].provider, "openai");
        assert!(app.auth.providers[0].has_env_api_key);
        assert_eq!(app.auth.providers[1].provider, "groq");
        assert!(app.auth.providers[1].has_stored_api_key);
        assert_eq!(app.auth.cursor, 7);
        assert_eq!(app.auth.selected, Some(9));
    }

    #[test]
    fn auth_state_survives_session_load() {
        let mut app = App::new();
        app.auth.providers = vec![make_provider("OpenAI")];
        app.auth.cursor = 4;
        app.auth.filter = "keep".into();
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::ApiKeyInput;
        app.auth.api_key_input = "secret".into();

        app.handle_acp_event(AcpAppEvent::SessionLoaded {
            agent_id: "agent-1".into(),
            session_id: "session-1".into(),
            profile_id: None,
        });

        assert_eq!(app.auth.providers.len(), 1);
        assert_eq!(app.auth.cursor, 4);
        assert_eq!(app.auth.filter, "keep");
        assert_eq!(app.auth.selected, Some(0));
        assert_eq!(app.auth.panel, AuthPanel::ApiKeyInput);
        assert_eq!(app.auth.api_key_input, "secret");
    }

    #[test]
    fn native_oauth_flow_started_event_sets_flow_state() {
        let mut app = App::new();
        app.navigation.popup = Popup::ProviderAuth;

        app.auth.last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });

        let cmds = app.handle_acp_event(AcpAppEvent::OAuthFlowStarted(OAuthFlow {
            flow_id: "flow-123".into(),
            provider: "openai".into(),
            authorization_url: "https://auth.example.com/authorize".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        }));

        assert!(cmds.is_empty());
        assert!(app.auth.oauth_flow.is_some());
        let flow = app.auth.oauth_flow.unwrap();
        assert_eq!(flow.flow_id, "flow-123");
        assert_eq!(flow.provider, "openai");
        assert_eq!(flow.flow_kind, OAuthFlowKind::RedirectCode);
        assert_eq!(app.auth.panel, AuthPanel::OAuthFlow);
        assert!(app.auth.last_result.is_none());
        assert!(app.auth.ui_notice.is_none());
    }

    #[test]
    fn native_oauth_result_success_clears_flow() {
        let mut app = App::new();
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "openai".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        app.auth.panel = AuthPanel::OAuthFlow;

        let cmds = app.handle_acp_event(AcpAppEvent::OAuthResult(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Success,
            message: "Connected successfully".into(),
        }));

        assert_eq!(cmds.len(), 1);
        assert!(matches!(
            cmds[0],
            Effect::Command(Command::ListAuthProviders)
        ));
        assert!(app.auth.oauth_flow.is_none());
        assert_eq!(app.auth.panel, AuthPanel::List);
        assert_eq!(
            app.auth.last_result,
            Some(OAuthResult {
                provider: "openai".into(),
                status: OAuthResultStatus::Success,
                message: "Connected successfully".into(),
            })
        );
    }

    #[test]
    fn native_oauth_result_failure_preserves_flow_and_refreshes_providers() {
        let mut app = App::new();
        let flow = OAuthFlow {
            flow_id: "f1".into(),
            provider: "anthropic".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        };
        app.auth.oauth_flow = Some(flow.clone());
        app.auth.panel = AuthPanel::OAuthFlow;

        let result = OAuthResult {
            provider: "anthropic".into(),
            status: OAuthResultStatus::Failure,
            message: "Authorization denied".into(),
        };
        let cmds = app.handle_acp_event(AcpAppEvent::OAuthResult(result.clone()));

        assert_eq!(cmds.len(), 1);
        assert!(matches!(
            cmds[0],
            Effect::Command(Command::ListAuthProviders)
        ));
        assert_eq!(app.auth.oauth_flow, Some(flow));
        assert_eq!(app.auth.panel, AuthPanel::OAuthFlow);
        assert_eq!(app.auth.last_result, Some(result));
    }

    #[test]
    fn native_oauth_result_clears_auth_ui_notice() {
        let mut app = App::new();
        app.auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "Copied to clipboard".into(),
        });

        app.handle_acp_event(AcpAppEvent::OAuthResult(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "Authorization denied".into(),
        }));

        assert!(app.auth.ui_notice.is_none());
    }

    // ── Disconnect / clear credential tests (C-d in List panel) ─────────────

    #[test]
    fn auth_list_ctrl_d_disconnects_oauth_when_connected() {
        let mut provider = make_provider("OpenAI");
        provider.oauth_status = Some(OAuthStatus::Connected);
        let mut app = make_app_with_providers(vec![provider]);
        app.auth.selected = Some(0);
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, ctrl('d')));
        let msg = effects.next_command().expect("message sent");
        assert!(matches!(
            msg,
            Command::DisconnectOAuth { provider } if provider == "openai"
        ));
    }

    #[test]
    fn auth_list_ctrl_d_clears_api_key_when_stored() {
        let mut provider = make_api_key_only("Groq");
        provider.has_stored_api_key = true;
        let mut app = make_app_with_providers(vec![provider]);
        app.auth.selected = Some(0);
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, ctrl('d')));
        let msg = effects.next_command().expect("message sent");
        assert!(matches!(
            msg,
            Command::ClearApiToken { provider } if provider == "groq"
        ));
    }

    #[test]
    fn auth_list_ctrl_d_noop_when_no_credential() {
        let app_provider = make_provider("OpenAI"); // not connected, no stored key
        let mut app = make_app_with_providers(vec![app_provider]);
        app.auth.selected = Some(0);
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, ctrl('d')));
        assert!(effects.next_command().is_none()); // nothing sent
    }

    #[test]
    fn auth_list_ctrl_d_noop_when_no_selection() {
        let mut provider = make_provider("OpenAI");
        provider.oauth_status = Some(OAuthStatus::Connected);
        let mut app = make_app_with_providers(vec![provider]);
        // auth_selected is None
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, ctrl('d')));
        assert!(effects.next_command().is_none()); // nothing sent
    }

    #[test]
    fn auth_list_ctrl_d_prefers_oauth_disconnect_over_api_key_clear() {
        // Provider has both OAuth connected AND a stored API key
        let mut provider = make_provider("OpenAI");
        provider.oauth_status = Some(OAuthStatus::Connected);
        provider.has_stored_api_key = true;
        let mut app = make_app_with_providers(vec![provider]);
        app.auth.selected = Some(0);
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, ctrl('d')));
        let msg = effects.next_command().expect("message sent");
        // Should disconnect OAuth first, not clear API key
        assert!(matches!(msg, Command::DisconnectOAuth { .. }));
    }

    // ── Clipboard copy tests ────────────────────────────────────────────────

    #[test]
    fn auth_oauth_ctrl_y_triggers_clipboard_copy() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth.selected = Some(0);
        app.auth.panel = AuthPanel::OAuthFlow;
        app.auth.oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://auth.example.com/authorize".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, ctrl('y')));

        assert!(matches!(
            effects.as_slice(),
            [Effect::CopyToClipboard {
                target: crate::application::ClipboardTarget::Auth { provider },
                text,
            }] if provider == "codex" && text == "https://auth.example.com/authorize"
        ));
        assert!(app.auth.ui_notice.is_none());
        assert!(app.auth.clipboard_fallback.is_none());
        assert!(app.auth.last_result.is_none());
    }

    #[test]
    fn auth_clipboard_fallback_dismisses_on_any_key() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth.clipboard_fallback = Some("https://example.com".into());
        let mut effects = TestEffects::default();

        effects.extend(handle_auth_popup_key(&mut app, key(KeyCode::Char('x'))));
        assert!(app.auth.clipboard_fallback.is_none());
        assert!(app.auth.filter.is_empty());
        assert!(effects.is_empty());
    }

    // ── Chord binding test ────────────────────────────────────────────────────

    #[test]
    fn chord_a_opens_auth_popup_and_sends_list() {
        let mut effects = TestEffects::default();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;

        // Activate chord mode
        let ctrl_x = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL);
        effects.extend(handle_key(&mut app, ctrl_x));
        assert!(app.navigation.chord);

        // Press 'a'
        effects.extend(handle_key(&mut app, key(KeyCode::Char('a'))));
        assert_eq!(app.navigation.popup, Popup::ProviderAuth);
        assert!(!app.navigation.chord);

        let msg = effects.next_command().expect("message sent");
        assert!(matches!(msg, Command::ListAuthProviders));
    }
}
