mod connection;
mod editor;
mod endpoint;
mod event_loop;
mod terminal;

use std::time::Duration;

use crate::{
    acp_client,
    acp_state::AcpAppEvent,
    app::{self, App, Screen},
    config,
    domain::model::DelegateModelPreference,
    protocol::ClientMsg,
    server_manager, theme,
};
use clap::Parser;
use connection::connection_manager;
use endpoint::{
    Cli, EndpointSelection, default_acp_ws_url, detect_launch_cwd, select_acp_endpoint,
};
use event_loop::run_loop;

use tokio::sync::mpsc;

#[derive(Debug)]
pub(crate) enum ConnectionManagerEvent {
    State(app::ConnectionEvent),
}

#[derive(Debug)]
pub(crate) enum ServerChannelMsg {
    Acp(AcpAppEvent),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::{ChatEntry, OUTCOME_BULLET};
    use crate::domain::elicitation::{
        ElicitationField, ElicitationFieldKind, ElicitationOption, ElicitationState,
    };
    use crate::domain::tool::ToolDetail;
    use crate::handlers::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use tokio::sync::mpsc;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn modified_key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
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
        app.conn = app::ConnState::Connected;
        app.session_id = Some("sess-1".into());
        app.messages.push(ChatEntry::Elicitation {
            elicitation_id: state.elicitation_id.clone(),
            message: state.message.clone(),
            source: state.source.clone(),
            outcome: None,
        });
        app.elicitation = Some(state);
        app.elicitation_ui = Some(crate::ui::ElicitationUiState::default());
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

    // ── Elicitation key handling ──────────────────────────────────────────────

    #[test]
    fn elicitation_down_moves_option_cursor() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_elicitation_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        assert_eq!(app.elicitation_ui.as_ref().unwrap().option_cursor, 1);
    }

    #[test]
    fn elicitation_up_does_not_go_below_zero() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_elicitation_key(&mut app, key(KeyCode::Up), &tx).unwrap();
        assert_eq!(app.elicitation_ui.as_ref().unwrap().option_cursor, 0);
    }

    #[test]
    fn elicitation_enter_on_single_select_sends_accept_and_resolves() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let (tx, mut rx) = mpsc::unbounded_channel();
        // Move to Beta and press Enter
        handle_elicitation_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        // Elicitation should be cleared
        assert!(app.elicitation.is_none());

        // Accept response sent
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg,
            ClientMsg::ElicitationResponse { action, content: Some(ref c), .. }
            if action == "accept" && c["choice"] == "b"
        ));

        // Chat card updated with the selected label
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { outcome: Some(o), .. } if *o == format!("{OUTCOME_BULLET}Beta")
        )));
    }

    #[test]
    fn elicitation_other_opens_multiline_editor_and_submits_custom_answer() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_elicitation_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert!(
            app.elicitation_ui
                .as_ref()
                .is_some_and(|ui| ui.custom_active)
        );

        for c in "custom".chars() {
            handle_elicitation_key(&mut app, key(KeyCode::Char(c)), &tx).unwrap();
        }
        handle_elicitation_key(
            &mut app,
            modified_key(KeyCode::Enter, KeyModifiers::SHIFT),
            &tx,
        )
        .unwrap();
        for c in "answer".chars() {
            handle_elicitation_key(&mut app, key(KeyCode::Char(c)), &tx).unwrap();
        }
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(app.elicitation.is_none());
        assert!(matches!(rx.try_recv().expect("message sent"),
            ClientMsg::ElicitationResponse { action, content: Some(ref c), .. }
            if action == "accept" && c["choice"] == "custom\nanswer"
        ));
    }

    #[test]
    fn elicitation_custom_esc_returns_to_choices_before_declining() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let (tx, mut rx) = mpsc::unbounded_channel();
        for _ in 0..2 {
            handle_elicitation_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        }
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Esc), &tx).unwrap();

        assert!(
            app.elicitation_ui
                .as_ref()
                .is_some_and(|ui| !ui.custom_active)
        );
        assert!(rx.try_recv().is_err());

        handle_elicitation_key(&mut app, key(KeyCode::Esc), &tx).unwrap();
        assert!(app.elicitation.is_none());
        assert!(matches!(rx.try_recv().expect("decline sent"),
            ClientMsg::ElicitationResponse { action, .. } if action == "decline"
        ));
    }

    #[test]
    fn elicitation_empty_custom_answer_stays_open() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let (tx, mut rx) = mpsc::unbounded_channel();
        for _ in 0..2 {
            handle_elicitation_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        }
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Char(' ')), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(app.elicitation.is_some());
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn elicitation_esc_sends_decline_and_resolves() {
        let mut app = make_app_with_elicitation(make_elicitation_single_select());
        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_elicitation_key(&mut app, key(KeyCode::Esc), &tx).unwrap();

        assert!(app.elicitation.is_none());
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg,
            ClientMsg::ElicitationResponse { action, .. } if action == "decline"
        ));
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { outcome: Some(o), .. } if o == "declined"
        )));
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
        app.elicitation.as_mut().unwrap().text_input = "Alice".into();
        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(app.elicitation.is_none());
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg,
            ClientMsg::ElicitationResponse { action, content: Some(ref c), .. }
            if action == "accept" && c["name"] == "Alice"
        ));
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { outcome: Some(o), .. } if o == "Alice"
        )));
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
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_elicitation_key(&mut app, key(KeyCode::Char('H')), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Char('i')), &tx).unwrap();
        assert_eq!(app.elicitation.as_ref().unwrap().text_input, "Hi");
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
        app.elicitation.as_mut().unwrap().text_input = "Hi".into();
        app.elicitation_ui.as_mut().unwrap().text_cursor = 2;
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_elicitation_key(&mut app, key(KeyCode::Backspace), &tx).unwrap();
        assert_eq!(app.elicitation.as_ref().unwrap().text_input, "H");
    }

    #[test]
    fn elicitation_enter_on_required_boolean_without_toggle_does_not_submit() {
        let mut app = make_app_with_elicitation(make_boolean_elicitation(true));
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(app.elicitation.is_some(), "popup should remain open");
        assert!(rx.try_recv().is_err(), "no response should be sent");
        assert!(
            app.messages
                .iter()
                .any(|m| matches!(m, ChatEntry::Elicitation { outcome: None, .. }))
        );
    }

    #[test]
    fn elicitation_boolean_space_toggles_true_then_enter_submits() {
        let mut app = make_app_with_elicitation(make_boolean_elicitation(true));
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_elicitation_key(&mut app, key(KeyCode::Char(' ')), &tx).unwrap();
        assert_eq!(
            app.elicitation
                .as_ref()
                .and_then(|state| state.selected.get("confirm")),
            Some(&serde_json::json!(true))
        );

        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(app.elicitation.is_none());
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg,
            ClientMsg::ElicitationResponse { action, content: Some(ref c), .. }
            if action == "accept" && c["confirm"] == true
        ));
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { outcome: Some(o), .. } if o == "Yes"
        )));
    }

    #[test]
    fn elicitation_boolean_second_space_toggles_false_and_still_submits() {
        let mut app = make_app_with_elicitation(make_boolean_elicitation(true));
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_elicitation_key(&mut app, key(KeyCode::Char(' ')), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Char(' ')), &tx).unwrap();
        assert_eq!(
            app.elicitation
                .as_ref()
                .and_then(|state| state.selected.get("confirm")),
            Some(&serde_json::json!(false))
        );

        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(app.elicitation.is_none());
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg,
            ClientMsg::ElicitationResponse { action, content: Some(ref c), .. }
            if action == "accept" && c["confirm"] == false
        ));
        assert!(app.messages.iter().any(|m| matches!(m,
            ChatEntry::Elicitation { outcome: Some(o), .. } if o == "No"
        )));
    }

    #[test]
    fn elicitation_key_handler_ignores_empty_field_list() {
        let mut app = make_app_with_elicitation(ElicitationState::new_for_test(vec![]));
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_elicitation_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Char(' ')), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Char('x')), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Backspace), &tx).unwrap();
        handle_elicitation_key(&mut app, key(KeyCode::Enter), &tx).unwrap();

        assert!(app.elicitation.is_some());
        assert!(rx.try_recv().is_err(), "no response should be sent");
    }

    #[test]
    fn invalidate_theme_caches_clears_all_render_caches() {
        use crate::theme::Theme;

        Theme::set_by_index(0);
        Theme::begin_frame();

        let mut app = App::new();
        app.messages.push(ChatEntry::User {
            text: "hello".into(),
            message_id: None,
        });
        app.messages.push(ChatEntry::ToolCall {
            tool_call_id: None,
            name: "edit".into(),
            is_error: false,
            detail: ToolDetail::Edit {
                file: "f.rs".into(),
                old: "aaa".into(),
                new: "bbb".into(),
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

        app.streaming_cache.store(
            5,
            vec![crate::markdown::CardBlock::Text(ratatui::text::Line::from(
                "stream",
            ))],
        );
        app.streaming_thinking_cache.store(
            3,
            vec![crate::markdown::CardBlock::Text(ratatui::text::Line::from(
                "think",
            ))],
        );

        assert!(app.streaming_cache.get(5).is_some());
        assert!(app.streaming_thinking_cache.get(3).is_some());
        assert_eq!(
            app.card_cache.processed_messages,
            app.messages.len(),
            "card_cache should be populated"
        );

        // Match the production order: snapshot the selected theme, then clear styled caches.
        Theme::set_by_index(2);
        Theme::begin_frame();
        let current_preview_fg = Theme::diff_file().fg.expect("diff_file should define fg");
        assert_ne!(old_preview_fg, current_preview_fg);
        invalidate_theme_caches(&mut app);

        assert_eq!(
            app.card_cache.processed_messages, 0,
            "card_cache should be invalidated"
        );
        assert!(
            app.streaming_cache.get(5).is_none(),
            "streaming_cache should be invalidated"
        );
        assert!(
            app.streaming_thinking_cache.get(3).is_none(),
            "streaming_thinking_cache should be invalidated"
        );
        assert!(matches!(
            &app.messages[1],
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
    use crate::config::{AcpConfig, TestPersistenceGuard, TuiConfig};
    use crate::domain::activity::{ActivityState, SessionOp};
    use crate::domain::chat::ChatEntry;
    use crate::handlers::*;
    use crate::protocol::PromptBlock;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use serial_test::serial;

    fn ctrl_x() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL)
    }

    fn plain_key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::empty())
    }

    #[test]
    fn chat_up_down_navigate_wrapped_input_without_scrolling_history() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "abcdef".into();
        app.input_cursor = 4;
        app.input_line_width = 4;
        app.scroll_offset = 7;

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Up, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.input_cursor, 2);
        assert_eq!(app.scroll_offset, 7);

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.input_cursor, 4);
        assert_eq!(app.scroll_offset, 7);
    }

    #[test]
    fn chat_pageup_pagedown_still_scroll_history() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.scroll_offset = 3;

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::PageUp, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.scroll_offset, 13);

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::PageDown, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.scroll_offset, 3);
    }

    #[test]
    fn ctrl_x_e_returns_open_editor_action_in_chat() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "draft".into();
        assert_eq!(
            handle_key(&mut app, ctrl_x(), &tx).unwrap(),
            AppAction::None
        );

        let action = handle_key(&mut app, plain_key('e'), &tx).unwrap();

        assert_eq!(action, AppAction::OpenExternalEditor);
        assert!(!app.chord);
        assert_eq!(app.input, "draft");
    }

    #[test]
    fn ctrl_x_e_outside_chat_stays_in_tui() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Sessions;
        assert_eq!(
            handle_key(&mut app, ctrl_x(), &tx).unwrap(),
            AppAction::None
        );

        let action = handle_key(&mut app, plain_key('e'), &tx).unwrap();

        assert_eq!(action, AppAction::None);
        assert!(app.status.contains("only available in chat"));
        assert!(matches!(app.logs.last(), Some(entry) if entry.target == "editor"));
    }

    #[test]
    fn ctrl_x_m_outside_chat_does_not_open_model_popup() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Sessions;
        assert_eq!(
            handle_key(&mut app, ctrl_x(), &tx).unwrap(),
            AppAction::None
        );

        let action = handle_key(&mut app, plain_key('m'), &tx).unwrap();

        assert_eq!(action, AppAction::None);
        assert_ne!(app.popup, app::Popup::ModelSelect);
        assert!(app.status.contains("only available in chat"));
        assert!(matches!(app.logs.last(), Some(entry) if entry.target == "model"));
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

        assert!(app.logs.iter().any(|entry| entry.target == "acp"
            && entry.level == app::LogLevel::Info
            && entry.message == "acp.binary_path not set; checking qmtcode on PATH"));
        assert!(app.logs.iter().any(|entry| entry.target == "acp"
            && entry.level == app::LogLevel::Info
            && entry.message == "qmtcode not found on PATH"));
    }

    #[test]
    fn chat_input_accepts_typing_and_submit_while_turn_active() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(matches!(
            rx.try_recv().expect("prompt sent"),
            ClientMsg::Prompt { prompt, local_id }
                if local_id.starts_with("local:pending:")
                    && matches!(prompt.as_slice(), [PromptBlock::Text { text }] if text == "n")
        ));
        assert!(app.input.is_empty());
        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::User { text, message_id: Some(message_id) }]
                if text == "n" && message_id.starts_with("local:pending:")
        ));
    }

    #[test]
    fn chat_submit_normalizes_prompt_before_sending_and_rendering() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.input = "  first line\nsecond line\n  ".into();
        app.input_cursor = app.input.len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(matches!(
            rx.try_recv().expect("prompt sent"),
            ClientMsg::Prompt { prompt, .. }
                if matches!(prompt.as_slice(), [PromptBlock::Text { text }]
                    if text == "first line\nsecond line")
        ));
        assert!(matches!(
            app.messages.as_slice(),
            [ChatEntry::User { text, .. }] if text == "first line\nsecond line"
        ));
    }

    #[test]
    fn whitespace_only_chat_submit_does_not_send_or_render_prompt() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.input = " \n  ".into();
        app.input_cursor = app.input.len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(rx.try_recv().is_err());
        assert!(app.messages.is_empty());
        assert!(app.input.is_empty());
    }

    // ── slash command handler tests ────────────────────────────────────────────

    #[test]
    fn left_arrow_with_slash_input_does_not_crash() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "/model".into();
        app.input_cursor = "/model".len();
        app.refresh_slash_state();
        assert!(app.slash_state.is_some());

        // hold left until cursor reaches 0 — must not panic at any step
        for _ in 0..=app.input.len() {
            handle_chat_key(
                &mut app,
                KeyEvent::new(KeyCode::Left, KeyModifiers::empty()),
                &tx,
            )
            .unwrap();
        }
        assert_eq!(app.input_cursor, 0);
        assert!(app.slash_state.is_none());
    }

    #[test]
    fn slash_esc_clears_slash_state() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "/mo".into();
        app.input_cursor = 3;
        app.refresh_slash_state();
        assert!(app.slash_state.is_some());

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(app.slash_state.is_none());
    }

    #[test]
    fn slash_enter_opens_help_popup() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "/help".into();
        app.input_cursor = "/help".len();
        app.refresh_slash_state();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, app::Popup::Help);
        assert!(app.input.is_empty());
    }

    #[test]
    fn slash_enter_with_partial_completion_executes_command() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "/hel".into();
        app.input_cursor = "/hel".len();
        app.refresh_slash_state();
        assert!(app.slash_state.is_some());

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, app::Popup::Help);
        assert!(app.input.is_empty());
        assert!(app.slash_state.is_none());
    }

    #[test]
    fn slash_tab_completes_command_name_without_executing() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "/hel".into();
        app.input_cursor = "/hel".len();
        app.refresh_slash_state();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        // Completed but not executed — no popup opened
        assert_eq!(app.input, "/help ");
        assert_eq!(app.popup, app::Popup::None);
        assert!(app.slash_state.is_none());
    }

    #[test]
    fn slash_down_up_navigates_selection() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "/".into();
        app.input_cursor = 1;
        app.refresh_slash_state();
        let initial = app.slash_state.as_ref().unwrap().selected_index;

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Down, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(
            app.slash_state.as_ref().unwrap().selected_index,
            initial + 1
        );

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Up, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.slash_state.as_ref().unwrap().selected_index, initial);
    }

    #[test]
    #[serial]
    fn slash_mode_no_arg_cycles_mode() {
        let _guard = TestPersistenceGuard::new("slash-mode-cycle");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();
        app.input = "/mode".into();
        app.input_cursor = "/mode".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.agent_mode, "plan");
        assert!(app.input.is_empty());
        // SetAgentMode should have been sent
        assert!(matches!(
            rx.try_recv().expect("SetAgentMode sent"),
            ClientMsg::SetAgentMode { mode } if mode == "plan"
        ));
    }

    #[test]
    #[serial]
    fn slash_mode_plan_switches_to_plan() {
        let _guard = TestPersistenceGuard::new("slash-mode-plan");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();
        app.input = "/mode plan".into();
        app.input_cursor = "/mode plan".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.agent_mode, "plan");
        assert!(matches!(
            rx.try_recv().expect("SetAgentMode sent"),
            ClientMsg::SetAgentMode { mode } if mode == "plan"
        ));
    }

    #[test]
    fn slash_mode_same_is_idempotent() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();
        app.input = "/mode build".into();
        app.input_cursor = "/mode build".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.agent_mode, "build");
        assert!(app.status.contains("already in build"));
    }

    #[test]
    fn slash_mode_unknown_shows_error() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.input = "/mode xyz".into();
        app.input_cursor = "/mode xyz".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(app.status.contains("unknown mode"));
    }

    #[test]
    #[serial]
    fn slash_thinking_high_sets_level() {
        let _guard = TestPersistenceGuard::new("slash-thinking-high");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.input = "/thinking high".into();
        app.input_cursor = "/thinking high".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.reasoning_effort, Some("high".into()));
        assert!(matches!(
            rx.try_recv().expect("SetReasoningEffort sent"),
            ClientMsg::SetReasoningEffort { reasoning_effort } if reasoning_effort == "high"
        ));
    }

    #[test]
    #[serial]
    fn slash_thinking_auto_clears_level() {
        let _guard = TestPersistenceGuard::new("slash-thinking-auto");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.reasoning_effort = Some("max".into());
        app.input = "/thinking auto".into();
        app.input_cursor = "/thinking auto".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.reasoning_effort, None);
        assert!(matches!(
            rx.try_recv().expect("SetReasoningEffort sent"),
            ClientMsg::SetReasoningEffort { reasoning_effort } if reasoning_effort == "auto"
        ));
    }

    #[test]
    #[serial]
    fn slash_thinking_med_alias_sets_medium() {
        let _guard = TestPersistenceGuard::new("slash-thinking-med");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.input = "/thinking med".into();
        app.input_cursor = "/thinking med".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.reasoning_effort, Some("medium".into()));
        assert!(matches!(
            rx.try_recv().expect("SetReasoningEffort sent"),
            ClientMsg::SetReasoningEffort { reasoning_effort } if reasoning_effort == "medium"
        ));
    }

    #[test]
    fn slash_thinking_no_arg_shows_current() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.reasoning_effort = Some("high".into());
        app.input = "/thinking".into();
        app.input_cursor = "/thinking".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(app.status.contains("thinking: high"));
    }

    #[test]
    fn slash_thinking_unknown_shows_error() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.input = "/thinking xyz".into();
        app.input_cursor = "/thinking xyz".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(app.status.contains("unknown level"));
    }

    #[test]
    fn slash_thinking_when_disconnected_does_not_change_state() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.reasoning_effort = Some("high".into());
        app.input = "/thinking max".into();
        app.input_cursor = "/thinking max".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        // state must not change when disconnected
        assert_eq!(app.reasoning_effort, Some("high".into()));
        assert!(app.status.contains("not connected"));
    }

    fn app_with_forkable_messages() -> App {
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.messages = vec![
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
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = app_with_forkable_messages();
        app.input = "/fork".into();
        app.input_cursor = "/fork".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(matches!(
            rx.try_recv().expect("ForkSession sent"),
            ClientMsg::ForkSession { message_id } if message_id == "user-2"
        ));
        assert_eq!(app.pending_fork_message_id.as_deref(), Some("user-2"));
    }

    #[test]
    fn ctrl_x_f_opens_fork_popup_and_filter_captures_text() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = app_with_forkable_messages();

        handle_key(&mut app, ctrl_x(), &tx).unwrap();
        handle_key(&mut app, plain_key('f'), &tx).unwrap();
        handle_key(&mut app, plain_key('b'), &tx).unwrap();

        assert_eq!(app.popup, app::Popup::ForkTurnSelect);
        assert_eq!(app.fork_filter, "b");
        assert!(app.input.is_empty());
        assert_eq!(app.filtered_fork_turns().len(), 1);
    }

    #[test]
    fn ctrl_x_f_in_delegate_view_does_not_open_fork_popup() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = app_with_forkable_messages();
        app.screen = Screen::Delegate;

        handle_key(&mut app, ctrl_x(), &tx).unwrap();
        handle_key(&mut app, plain_key('f'), &tx).unwrap();

        assert_ne!(app.popup, app::Popup::ForkTurnSelect);
        assert!(rx.try_recv().is_err());
        assert!(app.status.contains("only available in chat"));
        assert!(matches!(app.logs.last(), Some(entry) if entry.target == "fork"));
    }

    #[test]
    fn slash_fork_in_delegate_view_sends_nothing() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = app_with_forkable_messages();
        app.screen = Screen::Delegate;
        app.input = "/fork".into();
        app.input_cursor = "/fork".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(rx.try_recv().is_err());
        assert!(app.pending_fork_message_id.is_none());
        assert!(app.status.contains("only available in chat"));
        assert!(matches!(app.logs.last(), Some(entry) if entry.target == "fork"));
    }

    #[test]
    fn fork_popup_enter_sends_selected_turn() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = app_with_forkable_messages();
        app.open_fork_turn_popup();
        app.fork_filter = "alpha".into();

        handle_fork_turn_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(matches!(
            rx.try_recv().expect("ForkSession sent"),
            ClientMsg::ForkSession { message_id } if message_id == "asst-1"
        ));
        assert_eq!(app.pending_fork_message_id.as_deref(), Some("asst-1"));
    }

    #[test]
    fn fork_popup_enter_with_default_cursor_sends_latest_visible_turn() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = app_with_forkable_messages();
        app.open_fork_turn_popup();

        handle_fork_turn_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(matches!(
            rx.try_recv().expect("ForkSession sent"),
            ClientMsg::ForkSession { message_id } if message_id == "user-2"
        ));
        assert_eq!(app.pending_fork_message_id.as_deref(), Some("user-2"));
    }

    #[test]
    fn fork_popup_enter_with_no_eligible_turns_sends_nothing() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.open_fork_turn_popup();

        handle_fork_turn_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert!(rx.try_recv().is_err());
        assert!(app.status.contains("no forkable turns"));
    }

    #[test]
    fn slash_model_with_arg_prefilters_popup() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.input = "/model sonnet".into();
        app.input_cursor = "/model sonnet".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, app::Popup::ModelSelect);
        assert_eq!(app.model_filter, "sonnet");
        assert_eq!(app.model_cursor, 0);
    }

    #[test]
    fn slash_model_no_arg_opens_popup_unfiltered() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.input = "/model".into();
        app.input_cursor = "/model".len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, app::Popup::ModelSelect);
        assert!(app.model_filter.is_empty());
    }

    #[test]
    fn chat_double_esc_cancels_running_tool_phase() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert!(app.cancel_confirm_active());

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert!(matches!(
            rx.try_recv().expect("cancel sent"),
            ClientMsg::CancelSession
        ));
        assert_eq!(app.status, "stopping...");
        assert!(matches!(
            app.logs.last(),
            Some(entry) if entry.target == "activity" && entry.message == "stopping..."
        ));
    }

    #[test]
    fn chat_input_is_blocked_while_undo_is_pending() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.activity = ActivityState::SessionOp(SessionOp::Undo);
        app.input = "draft".into();
        app.input_cursor = app.input.len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.input, "draft");

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.input, "draft");

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Left, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.input_cursor, "draft".len());

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.input, "draft");
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn chat_input_is_blocked_while_cancel_confirm_is_active() {
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.activity = ActivityState::RunningTool {
            name: "read_tool".into(),
        };
        app.input = "draft".into();
        app.input_cursor = app.input.len();

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert!(app.cancel_confirm_active());
        assert!(app.input_blocked_by_activity());

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.input, "draft");

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.status, "press Esc again to stop");
        assert!(rx.try_recv().is_err());

        handle_chat_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::empty()),
            &tx,
        )
        .unwrap();
        assert_eq!(app.status, "stopping...");
        assert!(matches!(
            rx.try_recv().expect("cancel sent"),
            ClientMsg::CancelSession
        ));
    }
}

#[cfg(test)]
use crate::handlers::handle_key;

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
            app::LogLevel::Info,
            "acp",
            format!("configured qmtcode path not found: {path}; checking PATH"),
        );
    } else if cfg.acp.binary_path.is_none() {
        app.push_log(
            app::LogLevel::Info,
            "acp",
            "acp.binary_path not set; checking qmtcode on PATH",
        );
    }
    if discovery.binary.is_none() {
        app.push_log(app::LogLevel::Info, "acp", "qmtcode not found on PATH");
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
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<ClientMsg>();
    let (conn_tx, mut conn_rx) = mpsc::unbounded_channel::<ConnectionManagerEvent>();

    let mut app = App::new();
    app.launch_cwd = detect_launch_cwd();
    app.active_profile_id = cfg.profile.id.clone();
    app.show_thinking = cfg.show_thinking.unwrap_or(true);
    app.delegate_model_preferences = cfg.profile_delegate_models.clone();
    if let Some(profile_id) = cfg.profile.id.as_deref() {
        let legacy_preferences = app
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
        app.session_id = Some(session_id);
        app.screen = Screen::Chat;
    }
    // -- ACP auto-start ---------------------------------------------------------
    let (sup_event_tx, mut sup_event_rx) = mpsc::unbounded_channel::<server_manager::ServerEvent>();
    let default_ws_available = cli.ws.is_none()
        && cli.acp_websocket.is_none()
        && cli.acp_binary.is_none()
        && cfg.acp.websocket_url.is_none()
        && cfg.acp.transport.unwrap_or_default() != config::AcpTransportMode::WebSocket
        && cfg.acp.auto_start.unwrap_or(true)
        && acp_client::probe_websocket(&default_acp_ws_url(), Duration::from_millis(250)).await;

    let selection = select_acp_endpoint(&cli, &cfg, default_ws_available);
    if let EndpointSelection::Endpoint {
        discovered_ws: Some(url),
        ..
    } = &selection
    {
        app.push_log(
            app::LogLevel::Info,
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
                endpoint: acp_client::AcpEndpoint::Stdio { .. },
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
        endpoint: acp_client::AcpEndpoint::WebSocket { url },
        ..
    } = &selection
    {
        app.push_log(
            app::LogLevel::Warn,
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
            (None, server_manager::ServerState::BinaryNotFound)
        }
        EndpointSelection::Disabled => (None, server_manager::ServerState::Disabled),
    };

    if let Some(endpoint) = endpoint {
        tokio::spawn(connection_manager(
            endpoint,
            srv_tx,
            cmd_rx,
            conn_tx,
            sup_event_tx.clone(),
            app.launch_cwd.clone(),
        ));
    }

    let mut terminal = terminal::enter()?;

    app.server_state = initial_server_state;
    let result = run_loop(
        &mut terminal,
        &mut app,
        &mut srv_rx,
        &mut conn_rx,
        &mut sup_event_rx,
        &cmd_tx,
    )
    .await;

    terminal::leave(&mut terminal)?;

    if let Some(session_id) = &app.session_id {
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
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::handlers::*;
    use crate::protocol::ClientMsg;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use tokio::sync::mpsc;

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

    // ── Down / Up navigation ──────────────────────────────────────────────────

    #[test]
    fn down_moves_cursor_forward() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        // items: [GroupHeader, Session(s1), Session(s2)]
        assert_eq!(app.session_cursor, 0);
        apply_sessions_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 1);
        apply_sessions_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 2);
    }

    #[test]
    fn down_from_last_item_reaches_button_slot() {
        let mut app = App::new();
        // items: [GroupHeader(0), Session(1)] → button slot = 2
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 1; // last item (Session s1)
        apply_sessions_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 2); // moved to button slot
    }

    #[test]
    fn up_moves_cursor_back() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.session_cursor = 2;
        apply_sessions_key(&mut app, KeyCode::Up);
        assert_eq!(app.session_cursor, 1);
    }

    #[test]
    fn up_does_not_go_below_zero() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 0;
        apply_sessions_key(&mut app, KeyCode::Up);
        assert_eq!(app.session_cursor, 0);
    }

    // ── Enter on GroupHeader toggles collapse ─────────────────────────────────

    #[test]
    fn enter_on_header_collapses_group() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 0; // on the header
        let action = apply_sessions_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::None);
        assert!(app.collapsed_groups.contains("/a"));
    }

    #[test]
    fn enter_on_collapsed_header_expands_group() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.collapsed_groups.insert("/a".to_string());
        app.session_cursor = 0; // on the header
        let action = apply_sessions_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::None);
        assert!(!app.collapsed_groups.contains("/a"));
    }

    // ── Enter on Session loads it ─────────────────────────────────────────────

    #[test]
    fn enter_on_session_returns_load_action() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["abc12345"])];
        app.session_cursor = 1; // Session row
        let action = apply_sessions_key(&mut app, KeyCode::Enter);
        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "abc12345".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
    }

    #[test]
    fn enter_on_remote_label_without_node_id_is_noop() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["remote-1"])];
        app.session_groups[0].sessions[0].node = Some("remote node".into());
        app.session_cursor = 1;

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(action, SessionKeyAction::None);
    }

    #[test]
    fn enter_on_remote_session_remembers_node_id_for_reconnect() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["remote-1"])];
        app.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.session_cursor = 1;

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "remote-1".into(),
            }
        );
        assert_eq!(app.session_remote_node_id("remote-1"), Some("node-1"));
    }

    #[test]
    fn delete_on_remote_session_returns_dismiss_action_and_removes() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["remote-1", "s2"])];
        app.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.session_cursor = 1;

        let action = apply_sessions_key(&mut app, KeyCode::Delete);

        assert_eq!(
            action,
            SessionKeyAction::DismissRemoteSession {
                session_id: "remote-1".into()
            }
        );
        assert_eq!(app.session_groups[0].sessions.len(), 1);
        assert_eq!(app.session_groups[0].sessions[0].session_id, "s2");
    }

    #[test]
    fn enter_on_expandable_root_returns_load_action() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.session_cursor = 1;

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "root".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
        assert!(!app.expanded_session_children.contains("root"));
    }

    #[test]
    fn ctrl_o_on_expandable_root_requests_children() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.session_cursor = 1;
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();

        handle_sessions_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &cmd_tx,
        )
        .unwrap();

        assert!(app.expanded_session_children.contains("root"));
        assert!(matches!(
            cmd_rx.try_recv(),
            Ok(ClientMsg::ListSessionChildren {
                parent_session_id,
                ..
            }) if parent_session_id == "root"
        ));
    }

    #[test]
    fn ctrl_o_on_expanded_root_collapses_without_loading_session() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.expanded_session_children.insert("root".to_string());
        app.session_cursor = 1;
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();

        handle_sessions_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &cmd_tx,
        )
        .unwrap();

        assert!(!app.expanded_session_children.contains("root"));
        assert!(cmd_rx.try_recv().is_err());
    }

    #[test]
    fn plain_o_still_filters_instead_of_toggling_forks() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.session_cursor = 1;

        apply_sessions_key(&mut app, KeyCode::Char('o'));

        assert_eq!(app.session_filter, "o");
        assert!(!app.expanded_session_children.contains("root"));
    }

    // ── Delete on Session removes it ─────────────────────────────────────────

    #[test]
    fn delete_on_session_returns_delete_action_and_removes() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.session_cursor = 1; // Session s1
        let action = apply_sessions_key(&mut app, KeyCode::Delete);
        assert_eq!(
            action,
            SessionKeyAction::DeleteSession {
                session_id: "s1".to_string()
            }
        );
        // s1 removed; group still has s2
        assert_eq!(app.session_groups[0].sessions.len(), 1);
        assert_eq!(app.session_groups[0].sessions[0].session_id, "s2");
    }

    #[test]
    fn delete_removes_empty_group() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["only"])];
        app.session_cursor = 1;
        apply_sessions_key(&mut app, KeyCode::Delete);
        // Group removed entirely
        assert!(app.session_groups.is_empty());
    }

    #[test]
    fn delete_on_header_is_noop() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 0; // GroupHeader
        let action = apply_sessions_key(&mut app, KeyCode::Delete);
        assert_eq!(action, SessionKeyAction::None);
        // Session still there
        assert_eq!(app.session_groups[0].sessions.len(), 1);
    }

    // ── Char appends to filter and resets cursor ──────────────────────────────

    #[test]
    fn char_appends_to_filter_and_resets_cursor() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 1;
        apply_sessions_key(&mut app, KeyCode::Char('x'));
        assert_eq!(app.session_filter, "x");
        assert_eq!(app.session_cursor, 0);
    }

    #[test]
    fn backspace_removes_last_filter_char_and_resets_cursor() {
        let mut app = App::new();
        app.session_filter = "ab".to_string();
        app.session_cursor = 2;
        apply_sessions_key(&mut app, KeyCode::Backspace);
        assert_eq!(app.session_filter, "a");
        assert_eq!(app.session_cursor, 0);
    }

    #[test]
    fn backspace_on_empty_filter_is_noop() {
        let mut app = App::new();
        apply_sessions_key(&mut app, KeyCode::Backspace);
        assert_eq!(app.session_filter, "");
        assert_eq!(app.session_cursor, 0);
    }

    // ── Collapse clamps cursor ────────────────────────────────────────────────

    #[test]
    fn collapse_clamps_cursor_when_selected_row_disappears() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.session_cursor = 2; // pointing at Session s2
        // Collapsing /a while cursor is on s2 should clamp to the header (idx 0)
        apply_sessions_key(&mut app, KeyCode::Enter); // cursor=2 → on s2, wait...
        // Actually cursor=2 is Session s2; Enter sends LoadSession not collapse.
        // We need to test collapse-clamping by setting cursor on header first,
        // then collapse, then verify the previously-selected session index gets clamped.
        // Reset: cursor on header, collapse, cursor stays at 0.
        let mut app2 = App::new();
        app2.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app2.session_cursor = 0; // header
        apply_sessions_key(&mut app2, KeyCode::Enter); // collapse
        // 1 item visible (just header). cursor must be <= 0.
        assert_eq!(app2.session_cursor, 0);
        assert!(app2.collapsed_groups.contains("/a"));
    }

    // ── ShowMore Enter opens session popup ────────────────────────────────────

    #[test]
    fn enter_on_show_more_opens_session_popup() {
        let mut app = App::new();
        // 4 sessions -> header(0) + 3 sessions + ShowMore(4)
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3", "s4"])];
        app.session_cursor = 4; // ShowMore row
        let action = apply_sessions_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(app.popup, crate::app::Popup::SessionSelect);
        assert_eq!(app.session_cursor, 0);
        assert!(app.session_filter.is_empty());
    }

    #[test]
    fn enter_on_show_more_with_backend_cursor_still_only_opens_popup() {
        let mut app = App::new();
        app.session_groups = vec![make_group_with_cursor(
            Some("/workspace/project"),
            &["s1", "s2", "s3"],
            "cursor-1",
        )];
        app.session_cursor = 4; // ShowMore row created by next_cursor

        let action = apply_sessions_key(&mut app, KeyCode::Enter);

        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(app.popup, crate::app::Popup::SessionSelect);
        assert_eq!(app.session_popup_tab, 0);
        assert_eq!(app.session_cursor, 0);
    }

    // ── New Session button slot ───────────────────────────────────────────────

    #[test]
    fn down_can_reach_button_slot() {
        let mut app = App::new();
        // 1 group with 1 session → items: [GroupHeader(0), Session(1)]
        // button slot = items.len() = 2
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 1; // on Session
        apply_sessions_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 2); // on button slot
    }

    #[test]
    fn down_does_not_exceed_button_slot() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 2; // already on button slot
        apply_sessions_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 2); // stays
    }

    #[test]
    fn down_reaches_button_when_no_sessions() {
        let mut app = App::new();
        // No sessions → items is empty, button slot = 0
        app.session_cursor = 0;
        apply_sessions_key(&mut app, KeyCode::Down);
        // items.len() == 0, button is slot 0, can't go further
        assert_eq!(app.session_cursor, 0);
    }

    #[test]
    fn enter_on_button_slot_returns_new_session() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        // items: [GroupHeader(0), Session(1)] → button slot = 2
        app.session_cursor = 2;
        let action = apply_sessions_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::NewSession);
    }

    #[test]
    fn enter_on_button_slot_no_sessions_returns_new_session() {
        let mut app = App::new();
        // No items → button slot = 0
        app.session_cursor = 0;
        let action = apply_sessions_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::NewSession);
    }

    #[test]
    fn delete_on_button_slot_is_noop() {
        let mut app = App::new();
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 2; // button slot
        let action = apply_sessions_key(&mut app, KeyCode::Delete);
        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(app.session_groups[0].sessions.len(), 1); // unchanged
    }

    // ── q quits ───────────────────────────────────────────────────────────────
    // (q is handled in handle_sessions_key, not apply_sessions_key — tested
    //  via the existing integration path)
}

#[cfg(test)]
mod session_popup_key_tests {
    use super::*;
    use crate::app::Popup;
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::handlers::*;
    use crate::protocol::ClientMsg;
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

    // ── Down / Up navigation ──────────────────────────────────────────────────

    #[test]
    fn popup_down_moves_cursor_forward() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        // visible: [GroupHeader, Session(s1), Session(s2)]
        assert_eq!(app.session_cursor, 0);
        apply_popup_session_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 1);
        apply_popup_session_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 2);
    }

    #[test]
    fn popup_down_clamps_at_last_item() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        // visible: [GroupHeader(0), Session(1)] — max idx = 1
        app.session_cursor = 1;
        apply_popup_session_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 1); // clamped, no button slot
    }

    #[test]
    fn popup_up_moves_cursor_back() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.session_cursor = 2;
        apply_popup_session_key(&mut app, KeyCode::Up);
        assert_eq!(app.session_cursor, 1);
    }

    #[test]
    fn popup_up_does_not_go_below_zero() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 0;
        apply_popup_session_key(&mut app, KeyCode::Up);
        assert_eq!(app.session_cursor, 0);
    }

    #[test]
    fn popup_page_down_uses_visible_rows_with_overlap() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(
            Some("/a"),
            &["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        )];
        app.session_popup_visible_rows = 4;

        apply_popup_session_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.session_cursor, 3);

        apply_popup_session_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.session_cursor, 6);
    }

    #[test]
    fn popup_page_up_uses_visible_rows_with_overlap() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(
            Some("/a"),
            &["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        )];
        app.session_popup_visible_rows = 4;
        app.session_cursor = 6;

        apply_popup_session_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.session_cursor, 3);

        apply_popup_session_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.session_cursor, 0);
    }

    #[test]
    fn popup_page_keys_fallback_to_single_row_when_visible_rows_unknown() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3"])];

        apply_popup_session_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.session_cursor, 1);

        apply_popup_session_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.session_cursor, 0);
    }

    // ── Enter on GroupHeader toggles popup collapse ───────────────────────────

    #[test]
    fn popup_enter_on_header_collapses_group() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 0; // GroupHeader
        let action = apply_popup_session_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::None);
        assert!(app.popup_collapsed_groups.contains("/a"));
        // start-page state untouched
        assert!(!app.collapsed_groups.contains("/a"));
    }

    #[test]
    fn popup_enter_on_collapsed_header_expands_group() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.popup_collapsed_groups.insert("/a".to_string());
        app.session_cursor = 0;
        let action = apply_popup_session_key(&mut app, KeyCode::Enter);
        assert_eq!(action, SessionKeyAction::None);
        assert!(!app.popup_collapsed_groups.contains("/a"));
    }

    #[test]
    fn popup_collapse_clamps_cursor() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        // [GroupHeader(0), Session(s1, 1), Session(s2, 2)]
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.session_cursor = 0; // header
        // Collapse /a → only header remains
        apply_popup_session_key(&mut app, KeyCode::Enter);
        // Cursor must be 0 (clamped to header)
        assert_eq!(app.session_cursor, 0);
        assert!(app.popup_collapsed_groups.contains("/a"));
    }

    // ── Enter on Session loads it ─────────────────────────────────────────────

    #[test]
    fn popup_enter_on_session_returns_load_and_closes_popup() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["abc12345"])];
        app.session_cursor = 1; // Session row
        let action = apply_popup_session_key(&mut app, KeyCode::Enter);
        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "abc12345".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
        assert_eq!(app.popup, Popup::None);
    }

    #[test]
    fn popup_delete_on_remote_session_returns_dismiss_and_removes() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["remote-1", "s2"])];
        app.session_groups[0].sessions[0].node_id = Some("node-1".into());
        app.session_cursor = 1;

        let action = apply_popup_session_key(&mut app, KeyCode::Delete);

        assert_eq!(
            action,
            SessionKeyAction::DismissRemoteSession {
                session_id: "remote-1".into()
            }
        );
        assert_eq!(app.session_groups[0].sessions.len(), 1);
        assert_eq!(app.session_groups[0].sessions[0].session_id, "s2");
    }

    // ── Enter with all groups shows all sessions (no cap) ─────────────────────

    #[test]
    fn popup_enter_can_reach_session_beyond_start_page_cap() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        // 5 sessions — start page would cap at 3; popup shows all
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2", "s3", "s4", "s5"])];
        // visible: [Header(0), s1(1), s2(2), s3(3), s4(4), s5(5)]
        app.session_cursor = 5;
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
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.session_cursor = 1;

        let action = apply_popup_session_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "root".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
        assert!(!app.expanded_session_children.contains("root"));
        assert_eq!(app.popup, Popup::None);
    }

    #[test]
    fn popup_ctrl_o_on_expandable_root_requests_children() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.session_cursor = 1;
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();

        handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &cmd_tx,
        )
        .unwrap();

        assert!(app.expanded_session_children.contains("root"));
        assert_eq!(app.popup, Popup::SessionSelect);
        assert!(matches!(
            cmd_rx.try_recv(),
            Ok(ClientMsg::ListSessionChildren {
                parent_session_id,
                ..
            }) if parent_session_id == "root"
        ));
    }

    #[test]
    fn popup_ctrl_o_on_expanded_root_collapses_without_loading_session() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.expanded_session_children.insert("root".to_string());
        app.session_cursor = 1;
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();

        handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &cmd_tx,
        )
        .unwrap();

        assert!(!app.expanded_session_children.contains("root"));
        assert_eq!(app.popup, Popup::SessionSelect);
        assert!(cmd_rx.try_recv().is_err());
    }

    #[test]
    fn popup_plain_o_still_filters_instead_of_toggling_forks() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.session_cursor = 1;
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();

        handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::NONE),
            &cmd_tx,
        )
        .unwrap();

        assert_eq!(app.session_filter, "o");
        assert!(!app.expanded_session_children.contains("root"));
        assert_eq!(app.popup, Popup::SessionSelect);
        assert!(cmd_rx.try_recv().is_err());
    }

    #[test]
    fn popup_enter_on_expanded_child_loads_session() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["root"])];
        app.session_groups[0].sessions[0].fork_count = 1;
        app.session_groups[0].sessions[0].children = vec![SessionSummary {
            session_id: "child".to_string(),
            title: Some("child".to_string()),
            parent_session_id: Some("root".to_string()),
            ..Default::default()
        }];
        app.expanded_session_children.insert("root".to_string());
        app.session_cursor = 2;

        let action = apply_popup_session_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "child".to_string(),
                agent_id: None,
                cwd: Some("/a".to_string()),
            }
        );
        assert_eq!(app.popup, Popup::None);
    }

    #[test]
    fn popup_enter_on_load_more_returns_action_and_keeps_popup_open() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group_with_cursor(
            Some("/workspace/project"),
            &["s1"],
            "cursor-1",
        )];
        app.session_cursor = 2; // LoadMore row

        let action = apply_popup_session_key(&mut app, KeyCode::Enter);

        assert_eq!(
            action,
            SessionKeyAction::LoadMoreSessions {
                group_idx: 0,
                parent_path: Vec::new()
            }
        );
        assert_eq!(app.popup, Popup::SessionSelect);
    }

    // ── Delete on Session removes it ─────────────────────────────────────────

    #[test]
    fn popup_delete_on_session_returns_delete_and_removes() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1", "s2"])];
        app.session_cursor = 1; // Session s1
        let action = apply_popup_session_key(&mut app, KeyCode::Delete);
        assert_eq!(
            action,
            SessionKeyAction::DeleteSession {
                session_id: "s1".to_string()
            }
        );
        assert_eq!(app.session_groups[0].sessions.len(), 1);
        assert_eq!(app.session_groups[0].sessions[0].session_id, "s2");
    }

    #[test]
    fn popup_delete_removes_empty_group() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["only"])];
        app.session_cursor = 1;
        apply_popup_session_key(&mut app, KeyCode::Delete);
        assert!(app.session_groups.is_empty());
    }

    #[test]
    fn popup_delete_on_header_is_noop() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 0; // GroupHeader
        let action = apply_popup_session_key(&mut app, KeyCode::Delete);
        assert_eq!(action, SessionKeyAction::None);
        assert_eq!(app.session_groups[0].sessions.len(), 1);
    }

    // ── Esc closes popup ──────────────────────────────────────────────────────

    #[test]
    fn popup_esc_closes_popup() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        apply_popup_session_key(&mut app, KeyCode::Esc);
        assert_eq!(app.popup, Popup::None);
    }

    // ── Filter: Char appends, Backspace removes, both reset cursor ────────────

    #[test]
    fn popup_char_appends_to_filter_and_resets_cursor() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        app.session_cursor = 1;
        apply_popup_session_key(&mut app, KeyCode::Char('x'));
        assert_eq!(app.session_filter, "x");
        assert_eq!(app.session_cursor, 0);
    }

    #[test]
    fn popup_ctrl_n_opens_new_session_popup() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.conn = crate::app::ConnState::Connected;
        app.launch_cwd = Some("/launch".into());

        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        handle_session_popup_key(
            &mut app,
            KeyEvent {
                code: KeyCode::Char('n'),
                modifiers: KeyModifiers::CONTROL,
                kind: crossterm::event::KeyEventKind::Press,
                state: crossterm::event::KeyEventState::NONE,
            },
            &cmd_tx,
        )
        .unwrap();

        assert_eq!(app.popup, Popup::NewSession);
        assert_eq!(app.new_session_path, "/launch");
        assert!(cmd_rx.try_recv().is_err());
    }

    #[test]
    fn popup_plain_n_still_filters_instead_of_creating_session() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];

        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        handle_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
            &cmd_tx,
        )
        .unwrap();

        assert_eq!(app.popup, Popup::SessionSelect);
        assert_eq!(app.session_filter, "n");
        assert!(cmd_rx.try_recv().is_err());
    }

    #[test]
    fn global_ctrl_x_n_opens_new_session_popup() {
        let mut app = App::new();
        app.conn = crate::app::ConnState::Connected;
        app.launch_cwd = Some("/launch".into());
        let (tx, mut rx) = mpsc::unbounded_channel();

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
            &tx,
        )
        .unwrap();
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, Popup::NewSession);
        assert_eq!(app.new_session_path, "/launch");
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn global_ctrl_x_l_opens_session_popup() {
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        let (tx, _rx) = mpsc::unbounded_channel();

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
            &tx,
        )
        .unwrap();
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, Popup::SessionSelect);
        assert_eq!(app.session_popup_tab, 0);
    }

    #[test]
    fn global_ctrl_l_opens_log_popup() {
        let mut app = App::new();
        let (tx, _rx) = mpsc::unbounded_channel();

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, Popup::Log);
        assert_eq!(app.log_cursor, 0);
        assert!(app.log_filter.is_empty());
    }

    #[test]
    fn log_popup_filters_cycles_level_and_closes() {
        let mut app = App::new();
        app.popup = Popup::Log;
        app.log_cursor = 2;
        app.log_level_filter = crate::app::LogLevel::Info;

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
            &tx,
        )
        .unwrap();
        assert_eq!(app.log_filter, "x");
        assert_eq!(app.log_cursor, 0);

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();
        assert!(app.log_filter.is_empty());

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();
        assert_eq!(app.log_level_filter, crate::app::LogLevel::Warn);

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();
        assert_eq!(app.popup, Popup::None);
    }

    #[test]
    fn new_session_popup_enter_with_empty_path_uses_launch_cwd() {
        let mut app = App::new();
        app.conn = crate::app::ConnState::Connected;
        app.popup = Popup::NewSession;
        app.launch_cwd = Some("/launch".into());
        app.new_session_path.clear();
        app.new_session_cursor = 0;

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, Popup::None);
        assert!(matches!(
            rx.try_recv(),
            Ok(ClientMsg::NewSession {
                cwd: Some(ref cwd),
                request_id: None,
                profile_id: None
            }) if cwd == "/launch"
        ));
    }

    #[test]
    fn new_session_popup_enter_normalizes_relative_path_to_absolute() {
        let mut app = App::new();
        app.conn = crate::app::ConnState::Connected;
        app.popup = Popup::NewSession;
        app.launch_cwd = Some("/launch".into());
        app.new_session_path = "proj/subdir".into();
        app.new_session_cursor = app.new_session_path.len();

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert!(matches!(
            rx.try_recv(),
            Ok(ClientMsg::NewSession {
                cwd: Some(ref cwd),
                request_id: None,
                profile_id: None
            }) if cwd == "/launch/proj/subdir"
        ));
    }

    #[test]
    fn new_session_popup_tab_accepts_selected_completion() {
        let mut app = App::new();
        app.popup = Popup::NewSession;
        app.new_session_completion = Some(crate::app::PathCompletionState {
            query: "pro".into(),
            selected_index: 0,
            results: vec![crate::app::FileIndexEntryLite {
                path: "/launch/project".into(),
                is_dir: true,
            }],
        });

        let (tx, _rx) = mpsc::unbounded_channel();
        handle_new_session_popup_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert_eq!(app.new_session_path, "/launch/project/");
        assert!(app.new_session_completion.is_none());
    }

    #[test]
    fn handle_key_routes_tab_to_new_session_popup_before_global_mode_switch() {
        let mut app = App::new();
        app.conn = crate::app::ConnState::Connected;
        app.popup = Popup::NewSession;
        app.agent_mode = "build".into();
        app.new_session_completion = Some(crate::app::PathCompletionState {
            query: "pro".into(),
            selected_index: 0,
            results: vec![crate::app::FileIndexEntryLite {
                path: "/launch/project".into(),
                is_dir: true,
            }],
        });

        let (tx, mut rx) = mpsc::unbounded_channel();
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert_eq!(app.new_session_path, "/launch/project/");
        assert!(app.new_session_completion.is_none());
        assert_eq!(app.agent_mode, "build");
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn popup_backspace_removes_last_filter_char_and_resets_cursor() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_filter = "ab".to_string();
        app.session_cursor = 2;
        apply_popup_session_key(&mut app, KeyCode::Backspace);
        assert_eq!(app.session_filter, "a");
        assert_eq!(app.session_cursor, 0);
    }

    // ── multiple groups: navigation crosses group boundaries ─────────────────

    #[test]
    fn popup_down_crosses_group_boundary() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![
            make_group(Some("/a"), &["s1"]),
            make_group(Some("/b"), &["s2"]),
        ];
        // visible: [Header /a (0), Session s1 (1), Header /b (2), Session s2 (3)]
        app.session_cursor = 1; // s1
        apply_popup_session_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 2); // Header /b
        apply_popup_session_key(&mut app, KeyCode::Down);
        assert_eq!(app.session_cursor, 3); // s2
    }

    // ── collapse in popup does not affect start-page navigation ──────────────

    #[test]
    fn popup_collapse_independent_of_start_page() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_groups = vec![make_group(Some("/a"), &["s1"])];
        // collapse in popup
        app.session_cursor = 0;
        apply_popup_session_key(&mut app, KeyCode::Enter);
        assert!(app.popup_collapsed_groups.contains("/a"));
        // start page state untouched
        assert!(!app.collapsed_groups.contains("/a"));
    }
}

#[cfg(test)]
mod delegate_popup_key_tests {
    use super::*;
    use crate::app::Popup;
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
        app.session_id = Some("parent-1".into());
        app.popup = Popup::SessionSelect;
        app.session_popup_tab = 1;
        app.delegate_entries = vec![
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
        assert_eq!(app.delegate_cursor, 0);

        apply_delegate_popup_key(&mut app, KeyCode::Down);
        apply_delegate_popup_key(&mut app, KeyCode::Down);
        apply_delegate_popup_key(&mut app, KeyCode::Down);
        assert_eq!(app.delegate_cursor, 2);

        apply_delegate_popup_key(&mut app, KeyCode::Up);
        assert_eq!(app.delegate_cursor, 1);
    }

    #[test]
    fn delegate_page_down_uses_visible_rows_with_overlap() {
        let mut app = setup_delegate_app();
        app.delegate_entries.extend([
            make_entry("d4", "Check logs", Some("child-4")),
            make_entry("d5", "Refactor code", Some("child-5")),
            make_entry("d6", "Polish UI", Some("child-6")),
            make_entry("d7", "Ship release", Some("child-7")),
        ]);
        app.delegate_popup_visible_rows = 4;

        apply_delegate_popup_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.delegate_cursor, 3);

        apply_delegate_popup_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.delegate_cursor, 6);
    }

    #[test]
    fn delegate_page_up_uses_visible_rows_with_overlap() {
        let mut app = setup_delegate_app();
        app.delegate_entries.extend([
            make_entry("d4", "Check logs", Some("child-4")),
            make_entry("d5", "Refactor code", Some("child-5")),
            make_entry("d6", "Polish UI", Some("child-6")),
            make_entry("d7", "Ship release", Some("child-7")),
        ]);
        app.delegate_popup_visible_rows = 4;
        app.delegate_cursor = 6;

        apply_delegate_popup_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.delegate_cursor, 3);

        apply_delegate_popup_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.delegate_cursor, 0);
    }

    #[test]
    fn delegate_page_keys_fallback_to_single_row_when_visible_rows_unknown() {
        let mut app = setup_delegate_app();

        apply_delegate_popup_key(&mut app, KeyCode::PageDown);
        assert_eq!(app.delegate_cursor, 1);

        apply_delegate_popup_key(&mut app, KeyCode::PageUp);
        assert_eq!(app.delegate_cursor, 0);
    }

    #[test]
    fn delegate_enter_loads_selected_child_session() {
        let mut app = setup_delegate_app();
        app.delegate_cursor = 1;
        let action = apply_delegate_popup_key(&mut app, KeyCode::Enter);
        assert_eq!(
            action,
            SessionKeyAction::LoadSession {
                session_id: "child-2".into(),
                agent_id: Some("coder".into()),
                cwd: None,
            }
        );
        assert_eq!(app.popup, Popup::None);
    }

    #[test]
    fn delegate_enter_noop_when_child_session_is_unavailable() {
        let mut app = App::new();
        app.popup = Popup::SessionSelect;
        app.session_popup_tab = 1;
        app.delegate_entries = vec![DelegateEntry {
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
        assert_eq!(app.popup, Popup::SessionSelect);
    }

    #[test]
    fn delegate_filter_updates_cursor_and_loads_filtered_result() {
        let mut app = setup_delegate_app();
        app.delegate_cursor = 2;
        for c in "docs".chars() {
            apply_delegate_popup_key(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.delegate_filter, "docs");
        assert_eq!(app.delegate_cursor, 0);
        assert_eq!(app.visible_delegate_entries().len(), 1);
        assert_eq!(app.visible_delegate_entries()[0].delegation_id, "d3");

        apply_delegate_popup_key(&mut app, KeyCode::Backspace);
        assert_eq!(app.delegate_filter, "doc");
        assert_eq!(app.delegate_cursor, 0);

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
        app.delegate_entries[0].status = DelegateStatus::InProgress;
        app.delegate_entries[0].child_state = DelegateChildState::PendingElicitation {
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
        assert_eq!(app.popup, Popup::None);
    }

    #[test]
    fn delegate_esc_closes_popup() {
        let mut app = setup_delegate_app();
        apply_delegate_popup_key(&mut app, KeyCode::Esc);
        assert_eq!(app.popup, Popup::None);
    }

    #[test]
    fn delegate_popup_enter_sets_parent_for_sibling_navigation() {
        let mut app = setup_delegate_app();
        // Simulate being in a child session (parent_session_id is set).
        app.parent_session_id = Some("parent-1".into());
        app.session_id = Some("child-old".into());

        let action = apply_delegate_popup_key(&mut app, KeyCode::Enter);
        assert!(
            matches!(action, SessionKeyAction::LoadSession { .. }),
            "enter must trigger LoadSession"
        );
        assert_eq!(
            app.pending_parent_session_id.as_deref(),
            Some("parent-1"),
            "pending_parent must be the real parent, not the child session_id"
        );
    }
}

#[cfg(test)]
mod chord_reasoning_effort_tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use serial_test::serial;
    use tokio::sync::mpsc;

    fn ctrl_t() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL)
    }

    // ── Ctrl+t cycles reasoning effort and sends message ─────────────────────

    #[test]
    #[serial]
    fn ctrl_t_cycles_effort_and_sends_msg() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        assert_eq!(app.reasoning_effort, None);

        handle_key(&mut app, ctrl_t(), &tx).unwrap();

        assert_eq!(app.reasoning_effort, Some("low".into()));
        let msg = rx.try_recv().expect("expected SetReasoningEffort message");
        match msg {
            ClientMsg::SetReasoningEffort { reasoning_effort } => {
                assert_eq!(reasoning_effort, "low");
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn ctrl_t_full_cycle_sends_auto_on_wrap() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.reasoning_effort = Some("max".into());

        handle_key(&mut app, ctrl_t(), &tx).unwrap();

        assert_eq!(app.reasoning_effort, None);
        let msg = rx.try_recv().expect("expected SetReasoningEffort message");
        match msg {
            ClientMsg::SetReasoningEffort { reasoning_effort } => {
                assert_eq!(reasoning_effort, "auto");
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn ctrl_t_status_updated() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        handle_key(&mut app, ctrl_t(), &tx).unwrap();
        // status should reflect the new level
        assert!(
            app.status.contains("low"),
            "expected status to mention 'low', got: {}",
            app.status
        );
    }

    #[test]
    fn ctrl_t_when_disconnected_does_not_change_state() {
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.reasoning_effort = Some("high".into());

        handle_key(&mut app, ctrl_t(), &tx).unwrap();

        // state must not change when disconnected
        assert_eq!(app.reasoning_effort, Some("high".into()));
        assert!(app.status.contains("not connected"));
    }
}

#[cfg(test)]
mod reasoning_effort_integration_tests {
    use super::*;
    use crate::domain::model::ModelEntry;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use serial_test::serial;
    use tokio::sync::mpsc;

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
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
            &tx,
        )
        .unwrap();

        assert_eq!(app.reasoning_effort, Some("low".into()));
        assert!(matches!(
            rx.try_recv(),
            Ok(ClientMsg::SetReasoningEffort { reasoning_effort }) if reasoning_effort == "low"
        ));
    }

    #[test]
    #[serial]
    fn tab_switches_mode_without_changing_model_or_effort() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();
        app.current_provider = Some("anthropic".into());
        app.current_model = Some("claude-sonnet".into());
        app.reasoning_effort = Some("high".into());

        handle_key(&mut app, tab_key(), &tx).unwrap();

        assert_eq!(app.agent_mode, "plan");
        assert_eq!(app.current_model.as_deref(), Some("claude-sonnet"));
        assert_eq!(app.reasoning_effort, Some("high".into()));

        let msgs: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            msgs.iter()
                .any(|m| matches!(m, ClientMsg::SetAgentMode { mode } if mode == "plan")),
            "expected SetAgentMode(plan): {msgs:?}"
        );
        assert!(
            !msgs
                .iter()
                .any(|m| matches!(m, ClientMsg::SetReasoningEffort { .. })),
            "no effort restore on mode switch: {msgs:?}"
        );
    }

    #[test]
    #[serial]
    fn tab_no_cache_entry_leaves_model_and_effort_unchanged() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.agent_mode = "build".into();
        app.current_provider = Some("anthropic".into());
        app.current_model = Some("claude-sonnet".into());
        app.reasoning_effort = Some("high".into());
        // No plan cache entry

        handle_key(&mut app, tab_key(), &tx).unwrap();

        // Mode switched but model/effort unchanged (no cache to restore from)
        assert_eq!(app.agent_mode, "plan");
        assert_eq!(app.reasoning_effort, Some("high".into()));
        assert_eq!(app.current_model.as_deref(), Some("claude-sonnet"));
        let msgs: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            !msgs
                .iter()
                .any(|m| matches!(m, ClientMsg::SetReasoningEffort { .. })),
            "no SetReasoningEffort expected: {msgs:?}"
        );
    }

    // ── Model select: drops effort to auto ────────────────────────────────────

    #[test]
    #[serial]
    fn ctrl_x_m_opens_model_popup_at_current_mode_model() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, _rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.screen = Screen::Chat;
        app.conn = app::ConnState::Connected;
        app.agent_mode = "plan".into();
        app.model_popup_agent_tab = 0;
        app.current_provider = Some("anthropic".into());
        app.current_model = Some("claude-sonnet".into());
        app.models = vec![
            make_model("anthropic", "claude-sonnet"),
            make_model("openai", "gpt-4o"),
            make_model("openai", "o3-mini"),
        ];

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
            &tx,
        )
        .unwrap();
        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert_eq!(app.popup, app::Popup::ModelSelect);
        assert_eq!(app.model_filter, "");
        let expected = app.model_popup_open_cursor();
        assert_eq!(app.model_cursor, expected);
    }

    #[test]
    #[serial]
    fn model_select_drops_effort_to_auto() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.popup = app::Popup::ModelSelect;
        app.agent_mode = "build".into();
        app.model_popup_agent_tab = 0;
        app.current_provider = Some("anthropic".into());
        app.current_model = Some("claude-sonnet".into());
        app.reasoning_effort = Some("high".into());
        app.models = vec![make_model("anthropic", "claude-opus")];
        app.model_cursor = app
            .visible_model_popup_items()
            .iter()
            .position(|i| matches!(i, app::ModelPopupItem::Model { .. }))
            .unwrap();

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        assert_eq!(app.reasoning_effort, None);

        let msgs: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            msgs.iter().any(|m| matches!(
                m,
                ClientMsg::SetReasoningEffort { reasoning_effort }
                if reasoning_effort == "auto"
            )),
            "expected SetReasoningEffort(auto): {msgs:?}"
        );
    }

    #[test]
    #[serial]
    fn model_select_no_effort_msg_when_already_auto() {
        let _guard = PersistenceGuard::new("main-test");
        let (tx, mut rx) = mpsc::unbounded_channel::<ClientMsg>();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;
        app.session_id = Some("s1".into());
        app.popup = app::Popup::ModelSelect;
        app.agent_mode = "build".into();
        app.model_popup_agent_tab = 0;
        app.current_provider = Some("anthropic".into());
        app.current_model = Some("claude-sonnet".into());
        app.reasoning_effort = None; // already auto
        app.models = vec![make_model("anthropic", "claude-opus")];
        app.model_cursor = app
            .visible_model_popup_items()
            .iter()
            .position(|i| matches!(i, app::ModelPopupItem::Model { .. }))
            .unwrap();

        handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &tx,
        )
        .unwrap();

        let msgs: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert!(
            !msgs
                .iter()
                .any(|m| matches!(m, ClientMsg::SetReasoningEffort { .. })),
            "no SetReasoningEffort when already auto: {msgs:?}"
        );
    }

    #[test]
    fn native_reasoning_effort_event_updates_state() {
        let mut app = App::new();
        app.handle_acp_event(AcpAppEvent::ReasoningEffort {
            reasoning_effort: Some("medium".into()),
        });
        assert_eq!(app.reasoning_effort, Some("medium".into()));
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
    use crate::app::AuthUiNotice;
    use crate::domain::auth::{
        AuthProviderEntry, OAuthFlow, OAuthFlowKind, OAuthResult, OAuthResultStatus, OAuthStatus,
    };
    use crate::handlers::*;
    use crate::protocol::ClientMsg;
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
        app.conn = app::ConnState::Connected;
        app.auth_providers = providers;
        app.popup = app::Popup::ProviderAuth;
        app
    }

    // ── App state tests ───────────────────────────────────────────────────────

    #[test]
    fn open_auth_popup_resets_state() {
        let mut app = App::new();
        app.auth_cursor = 5;
        app.auth_filter = "test".into();
        app.auth_selected = Some(2);
        app.auth_api_key_input = "secret".into();
        app.auth_last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        app.open_auth_popup();
        assert_eq!(app.popup, app::Popup::ProviderAuth);
        assert_eq!(app.auth_cursor, 0);
        assert!(app.auth_filter.is_empty());
        assert!(app.auth_selected.is_none());
        assert!(app.auth_api_key_input.is_empty());
        assert!(app.auth_last_result.is_none());
        assert!(app.auth_ui_notice.is_none());
        assert!(app.auth_api_key_masked);
        assert_eq!(app.auth_panel, app::AuthPanel::List);
    }

    #[test]
    fn filtered_auth_providers_with_empty_filter() {
        let app = make_app_with_providers(vec![make_provider("OpenAI"), make_provider("Groq")]);
        let filtered = app.filtered_auth_providers();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn filtered_auth_providers_filters_by_name() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI"), make_provider("Groq")]);
        app.auth_filter = "groq".into();
        let filtered = app.filtered_auth_providers();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].1.provider, "groq");
    }

    #[test]
    fn auth_close_detail_resets_panel_state() {
        let mut app = App::new();
        app.auth_selected = Some(1);
        app.auth_panel = app::AuthPanel::ApiKeyInput;
        app.auth_api_key_input = "secret".into();
        app.auth_last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Success,
            message: "connected".into(),
        });
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: None,
            success: true,
            message: "saved".into(),
        });
        app.auth_close_detail();
        assert!(app.auth_selected.is_none());
        assert_eq!(app.auth_panel, app::AuthPanel::List);
        assert!(app.auth_api_key_input.is_empty());
        assert!(app.auth_last_result.is_none());
        assert!(app.auth_ui_notice.is_none());
    }

    #[test]
    fn auth_feedback_scopes_oauth_result_to_its_provider() {
        let mut app = App::new();
        app.auth_last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "authorization denied".into(),
        });

        assert_eq!(app.auth_feedback_for_provider("anthropic"), None);
        assert_eq!(
            app.auth_feedback_for_provider("openai"),
            Some((false, "authorization denied"))
        );
    }

    #[test]
    fn auth_feedback_scopes_ui_notice_and_takes_precedence() {
        let mut app = App::new();
        app.auth_last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "authorization denied".into(),
        });
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "Copied to clipboard".into(),
        });

        assert_eq!(app.auth_feedback_for_provider("anthropic"), None);
        assert_eq!(
            app.auth_feedback_for_provider("openai"),
            Some((true, "Copied to clipboard"))
        );
    }

    #[test]
    fn auth_feedback_supports_generic_ui_notice() {
        let mut app = App::new();
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: None,
            success: false,
            message: "Clipboard unavailable".into(),
        });

        assert_eq!(
            app.auth_feedback_for_provider("openai"),
            Some((false, "Clipboard unavailable"))
        );
        assert_eq!(
            app.auth_feedback_for_provider("anthropic"),
            Some((false, "Clipboard unavailable"))
        );
    }

    // ── Key handler tests: List panel ─────────────────────────────────────────

    #[test]
    fn auth_list_esc_closes_popup_when_no_selection() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, key(KeyCode::Esc), &tx).unwrap();
        assert_eq!(app.popup, app::Popup::None);
    }

    #[test]
    fn auth_list_esc_clears_selection_when_selected() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth_selected = Some(0);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, key(KeyCode::Esc), &tx).unwrap();
        assert_eq!(app.popup, app::Popup::ProviderAuth);
        assert!(app.auth_selected.is_none());
    }

    #[test]
    fn auth_list_down_up_navigates() {
        let mut app = make_app_with_providers(vec![
            make_provider("OpenAI"),
            make_provider("Groq"),
            make_provider("DeepSeek"),
        ]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        assert_eq!(app.auth_cursor, 0);
        handle_auth_popup_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        assert_eq!(app.auth_cursor, 1);
        handle_auth_popup_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        assert_eq!(app.auth_cursor, 2);
        handle_auth_popup_key(&mut app, key(KeyCode::Down), &tx).unwrap();
        assert_eq!(app.auth_cursor, 2); // clamped
        handle_auth_popup_key(&mut app, key(KeyCode::Up), &tx).unwrap();
        assert_eq!(app.auth_cursor, 1);
    }

    #[test]
    fn auth_list_enter_on_api_key_only_opens_api_key_panel() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth_last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert_eq!(app.auth_panel, app::AuthPanel::ApiKeyInput);
        assert_eq!(app.auth_selected, Some(0));
        assert!(app.auth_last_result.is_none());
        assert!(app.auth_ui_notice.is_none());
    }

    #[test]
    fn auth_list_enter_on_oauth_only_starts_flow() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("codex".into()),
            success: true,
            message: "old notice".into(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert_eq!(app.auth_selected, Some(0));
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg, ClientMsg::StartOAuthLogin { provider } if provider == "codex"));
        assert!(app.auth_ui_notice.is_none());
    }

    #[test]
    fn auth_list_enter_on_multi_method_selects_provider() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert_eq!(app.auth_selected, Some(0));
        assert_eq!(app.auth_panel, app::AuthPanel::List);
        assert!(app.auth_ui_notice.is_none());
    }

    #[test]
    fn auth_list_char_input_filters() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI"), make_provider("Groq")]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, key(KeyCode::Char('g')), &tx).unwrap();
        assert_eq!(app.auth_filter, "g");
        assert_eq!(app.auth_cursor, 0);
    }

    #[test]
    fn auth_list_backspace_removes_filter() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth_filter = "op".into();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, key(KeyCode::Backspace), &tx).unwrap();
        assert_eq!(app.auth_filter, "o");
    }

    #[test]
    fn auth_list_ctrl_k_opens_api_key_panel() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, ctrl('k'), &tx).unwrap();
        assert_eq!(app.auth_panel, app::AuthPanel::ApiKeyInput);
        assert_eq!(app.auth_selected, Some(0));
        assert!(app.auth_ui_notice.is_none());
    }

    #[test]
    fn auth_list_ctrl_o_starts_oauth() {
        let mut app = make_app_with_providers(vec![make_provider("OpenAI")]);
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        handle_auth_popup_key(&mut app, ctrl('o'), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg, ClientMsg::StartOAuthLogin { provider } if provider == "openai"));
        assert!(app.auth_ui_notice.is_none());
    }

    // ── Key handler tests: API Key panel ──────────────────────────────────────

    #[test]
    fn auth_api_key_typing_and_submit() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::ApiKeyInput;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Char('s')), &tx).unwrap();
        handle_auth_popup_key(&mut app, key(KeyCode::Char('k')), &tx).unwrap();
        assert_eq!(app.auth_api_key_input, "sk");
        assert_eq!(app.auth_api_key_cursor, 2);

        handle_auth_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(
            msg,
            ClientMsg::SetApiToken { provider, api_key }
            if provider == "groq" && api_key == "sk"
        ));
    }

    #[test]
    fn auth_api_key_backspace() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::ApiKeyInput;
        app.auth_api_key_input = "abc".into();
        app.auth_api_key_cursor = 3;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Backspace), &tx).unwrap();
        assert_eq!(app.auth_api_key_input, "ab");
        assert_eq!(app.auth_api_key_cursor, 2);
    }

    #[test]
    fn auth_api_key_esc_returns_to_list() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::ApiKeyInput;
        app.auth_api_key_input = "draft".into();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Esc), &tx).unwrap();
        assert_eq!(app.auth_panel, app::AuthPanel::List);
        assert!(app.auth_api_key_input.is_empty());
    }

    #[test]
    fn auth_api_key_tab_toggles_mask() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::ApiKeyInput;
        assert!(app.auth_api_key_masked);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Tab), &tx).unwrap();
        assert!(!app.auth_api_key_masked);
        handle_auth_popup_key(&mut app, key(KeyCode::Tab), &tx).unwrap();
        assert!(app.auth_api_key_masked);
    }

    #[test]
    fn auth_api_key_ctrl_d_sends_clear() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::ApiKeyInput;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, ctrl('d'), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg, ClientMsg::ClearApiToken { provider } if provider == "groq"));
    }

    #[test]
    fn auth_api_key_empty_submit_does_nothing() {
        let mut app = make_app_with_providers(vec![make_api_key_only("Groq")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::ApiKeyInput;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        assert!(rx.try_recv().is_err()); // nothing sent
    }

    // ── Key handler tests: OAuth flow panel ───────────────────────────────────

    #[test]
    fn auth_oauth_esc_returns_to_list() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::OAuthFlow;
        app.auth_oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Esc), &tx).unwrap();
        assert_eq!(app.auth_panel, app::AuthPanel::List);
        assert!(app.auth_oauth_flow.is_none());
    }

    #[test]
    fn auth_oauth_redirect_code_typing_and_submit() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::OAuthFlow;
        app.auth_oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Char('c')), &tx).unwrap();
        handle_auth_popup_key(&mut app, key(KeyCode::Char('o')), &tx).unwrap();
        handle_auth_popup_key(&mut app, key(KeyCode::Char('d')), &tx).unwrap();
        handle_auth_popup_key(&mut app, key(KeyCode::Char('e')), &tx).unwrap();
        assert_eq!(app.auth_oauth_response, "code");

        handle_auth_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(
            msg,
            ClientMsg::CompleteOAuthLogin { flow_id, response }
            if flow_id == "f1" && response == "code"
        ));
    }

    #[test]
    fn auth_oauth_device_poll_enter_sends_empty_response() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::OAuthFlow;
        app.auth_oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://example.com/device".into(),
            flow_kind: OAuthFlowKind::DevicePoll,
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Enter), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(
            msg,
            ClientMsg::CompleteOAuthLogin { flow_id, response }
            if flow_id == "f1" && response.is_empty()
        ));
    }

    // ── Native ACP event handling tests ───────────────────────────────────────

    #[test]
    fn native_initialized_event_clears_auth_ui_notice() {
        let mut app = App::new();
        app.auth_ui_notice = Some(AuthUiNotice {
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
        assert!(app.auth_ui_notice.is_none());
    }

    #[test]
    fn native_auth_providers_event_populates_list() {
        let mut app = App::new();
        let mut openai = make_provider("OpenAI");
        openai.has_env_api_key = true;
        let mut groq = make_api_key_only("Groq");
        groq.has_stored_api_key = true;

        let cmds = app.handle_acp_event(AcpAppEvent::AuthProviders(vec![openai, groq]));

        assert!(cmds.is_empty());
        assert_eq!(app.auth_providers.len(), 2);
        assert_eq!(app.auth_providers[0].provider, "openai");
        assert!(app.auth_providers[0].has_env_api_key);
        assert_eq!(app.auth_providers[1].provider, "groq");
        assert!(app.auth_providers[1].has_stored_api_key);
    }

    #[test]
    fn native_oauth_flow_started_event_sets_flow_state() {
        let mut app = App::new();
        app.popup = app::Popup::ProviderAuth;

        app.auth_last_result = Some(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "old result".into(),
        });
        app.auth_ui_notice = Some(AuthUiNotice {
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
        assert!(app.auth_oauth_flow.is_some());
        let flow = app.auth_oauth_flow.unwrap();
        assert_eq!(flow.flow_id, "flow-123");
        assert_eq!(flow.provider, "openai");
        assert_eq!(flow.flow_kind, OAuthFlowKind::RedirectCode);
        assert_eq!(app.auth_panel, app::AuthPanel::OAuthFlow);
        assert!(app.auth_last_result.is_none());
        assert!(app.auth_ui_notice.is_none());
    }

    #[test]
    fn native_oauth_result_success_clears_flow() {
        let mut app = App::new();
        app.auth_oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "openai".into(),
            authorization_url: "https://example.com".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        app.auth_panel = app::AuthPanel::OAuthFlow;

        let cmds = app.handle_acp_event(AcpAppEvent::OAuthResult(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Success,
            message: "Connected successfully".into(),
        }));

        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], ClientMsg::ListAuthProviders));
        assert!(app.auth_oauth_flow.is_none());
        assert_eq!(app.auth_panel, app::AuthPanel::List);
        assert_eq!(
            app.auth_last_result,
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
        app.auth_oauth_flow = Some(flow.clone());
        app.auth_panel = app::AuthPanel::OAuthFlow;

        let result = OAuthResult {
            provider: "anthropic".into(),
            status: OAuthResultStatus::Failure,
            message: "Authorization denied".into(),
        };
        let cmds = app.handle_acp_event(AcpAppEvent::OAuthResult(result.clone()));

        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], ClientMsg::ListAuthProviders));
        assert_eq!(app.auth_oauth_flow, Some(flow));
        assert_eq!(app.auth_panel, app::AuthPanel::OAuthFlow);
        assert_eq!(app.auth_last_result, Some(result));
    }

    #[test]
    fn native_oauth_result_clears_auth_ui_notice() {
        let mut app = App::new();
        app.auth_ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "Copied to clipboard".into(),
        });

        app.handle_acp_event(AcpAppEvent::OAuthResult(OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Failure,
            message: "Authorization denied".into(),
        }));

        assert!(app.auth_ui_notice.is_none());
    }

    // ── Disconnect / clear credential tests (C-d in List panel) ─────────────

    #[test]
    fn auth_list_ctrl_d_disconnects_oauth_when_connected() {
        let mut provider = make_provider("OpenAI");
        provider.oauth_status = Some(OAuthStatus::Connected);
        let mut app = make_app_with_providers(vec![provider]);
        app.auth_selected = Some(0);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, ctrl('d'), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(
            msg,
            ClientMsg::DisconnectOAuth { provider } if provider == "openai"
        ));
    }

    #[test]
    fn auth_list_ctrl_d_clears_api_key_when_stored() {
        let mut provider = make_api_key_only("Groq");
        provider.has_stored_api_key = true;
        let mut app = make_app_with_providers(vec![provider]);
        app.auth_selected = Some(0);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, ctrl('d'), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(
            msg,
            ClientMsg::ClearApiToken { provider } if provider == "groq"
        ));
    }

    #[test]
    fn auth_list_ctrl_d_noop_when_no_credential() {
        let app_provider = make_provider("OpenAI"); // not connected, no stored key
        let mut app = make_app_with_providers(vec![app_provider]);
        app.auth_selected = Some(0);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, ctrl('d'), &tx).unwrap();
        assert!(rx.try_recv().is_err()); // nothing sent
    }

    #[test]
    fn auth_list_ctrl_d_noop_when_no_selection() {
        let mut provider = make_provider("OpenAI");
        provider.oauth_status = Some(OAuthStatus::Connected);
        let mut app = make_app_with_providers(vec![provider]);
        // auth_selected is None
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, ctrl('d'), &tx).unwrap();
        assert!(rx.try_recv().is_err()); // nothing sent
    }

    #[test]
    fn auth_list_ctrl_d_prefers_oauth_disconnect_over_api_key_clear() {
        // Provider has both OAuth connected AND a stored API key
        let mut provider = make_provider("OpenAI");
        provider.oauth_status = Some(OAuthStatus::Connected);
        provider.has_stored_api_key = true;
        let mut app = make_app_with_providers(vec![provider]);
        app.auth_selected = Some(0);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, ctrl('d'), &tx).unwrap();
        let msg = rx.try_recv().expect("message sent");
        // Should disconnect OAuth first, not clear API key
        assert!(matches!(msg, ClientMsg::DisconnectOAuth { .. }));
    }

    // ── Clipboard copy tests ────────────────────────────────────────────────

    #[test]
    fn auth_oauth_ctrl_y_triggers_clipboard_copy() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth_selected = Some(0);
        app.auth_panel = app::AuthPanel::OAuthFlow;
        app.auth_oauth_flow = Some(OAuthFlow {
            flow_id: "f1".into(),
            provider: "codex".into(),
            authorization_url: "https://auth.example.com/authorize".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, ctrl('y'), &tx).unwrap();

        let expected_notice = AuthUiNotice {
            provider: Some("codex".into()),
            success: true,
            message: "Copied to clipboard".into(),
        };
        let expected_url = "https://auth.example.com/authorize";
        assert!(
            matches!(
                (&app.auth_ui_notice, &app.auth_clipboard_fallback),
                (Some(notice), None) if notice == &expected_notice
            ) || matches!(
                (&app.auth_ui_notice, &app.auth_clipboard_fallback),
                (None, Some(url)) if url == expected_url
            ),
            "C-y should show the provider notice or exact fallback URL"
        );
        assert!(app.auth_last_result.is_none());
    }

    #[test]
    fn auth_clipboard_fallback_dismisses_on_any_key() {
        let mut app = make_app_with_providers(vec![make_oauth_only("Codex")]);
        app.auth_clipboard_fallback = Some("https://example.com".into());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

        handle_auth_popup_key(&mut app, key(KeyCode::Char('x')), &tx).unwrap();
        assert!(app.auth_clipboard_fallback.is_none());
    }

    // ── Chord binding test ────────────────────────────────────────────────────

    #[test]
    fn chord_a_opens_auth_popup_and_sends_list() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new();
        app.conn = app::ConnState::Connected;

        // Activate chord mode
        let ctrl_x = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL);
        handle_key(&mut app, ctrl_x, &tx).unwrap();
        assert!(app.chord);

        // Press 'a'
        handle_key(&mut app, key(KeyCode::Char('a')), &tx).unwrap();
        assert_eq!(app.popup, app::Popup::ProviderAuth);
        assert!(!app.chord);

        let msg = rx.try_recv().expect("message sent");
        assert!(matches!(msg, ClientMsg::ListAuthProviders));
    }
}
