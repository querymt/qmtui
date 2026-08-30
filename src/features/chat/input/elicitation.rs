use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::{
    chat_state::ChatState,
    domain::{
        chat::ElicitationResponseOutcome,
        elicitation::{ElicitationFieldKind, ElicitationState},
    },
    render_state::RenderState,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElicitationResponseEffect {
    pub(crate) elicitation_id: String,
    pub(crate) action: String,
    pub(crate) content: Option<serde_json::Value>,
    pub(crate) outcome: ElicitationResponseOutcome,
}

fn response_effect(
    elicitation_id: String,
    action: &str,
    content: Option<serde_json::Value>,
    outcome: ElicitationResponseOutcome,
) -> ElicitationResponseEffect {
    ElicitationResponseEffect {
        elicitation_id,
        action: action.into(),
        content,
        outcome,
    }
}

fn selected_outcome(
    state: &ElicitationState,
    field_index: usize,
    custom_active: bool,
) -> ElicitationResponseOutcome {
    let field = &state.fields[field_index];
    if custom_active {
        return ElicitationResponseOutcome::Selected(vec![state.custom_input.trim().to_string()]);
    }
    match &field.kind {
        ElicitationFieldKind::SingleSelect { options } => ElicitationResponseOutcome::Selected(
            options
                .iter()
                .find(|option| state.selected.get(&field.name) == Some(&option.value))
                .map(|option| vec![option.label.clone()])
                .unwrap_or_default(),
        ),
        ElicitationFieldKind::MultiSelect { options } => {
            let labels = state
                .selected
                .get(&field.name)
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    options
                        .iter()
                        .filter(|option| values.contains(&option.value))
                        .map(|option| option.label.clone())
                        .collect()
                })
                .unwrap_or_default();
            ElicitationResponseOutcome::Selected(labels)
        }
        ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
            ElicitationResponseOutcome::Text(state.text_input.clone())
        }
        ElicitationFieldKind::BooleanToggle => state
            .selected
            .get(&field.name)
            .and_then(serde_json::Value::as_bool)
            .map(ElicitationResponseOutcome::Boolean)
            .unwrap_or_else(|| ElicitationResponseOutcome::Text(String::new())),
    }
}

/// Handle keyboard input owned by an active chat elicitation.
pub(crate) fn handle_key(
    chat: &mut ChatState,
    render: &mut RenderState,
    key: KeyEvent,
) -> Vec<ElicitationResponseEffect> {
    let mut effects = Vec::new();
    let custom_line_width = render.elicitation_custom_line_width();
    let (Some(state), Some(ui)) = (chat.elicitation.as_mut(), chat.elicitation_ui.as_mut()) else {
        return effects;
    };
    let Some(field_index) = ui.current_field_index(state.fields.len()) else {
        return effects;
    };

    if ui.custom_active {
        match key.code {
            KeyCode::Esc => {
                ui.custom_active = false;
                render.reset_elicitation_custom_scroll();
            }
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                ui.custom_insert(&mut state.custom_input, '\n');
            }
            KeyCode::Enter if state.is_valid(Some(&state.custom_input)) => {
                let elicitation_id = state.elicitation_id.clone();
                let content = state.build_accept_content(Some(&state.custom_input));
                let outcome = selected_outcome(state, field_index, true);
                effects.push(response_effect(
                    elicitation_id,
                    "accept",
                    Some(content),
                    outcome,
                ));
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
            KeyCode::Up => ui.custom_move_visual(&state.custom_input, custom_line_width, -1),
            KeyCode::Down => ui.custom_move_visual(&state.custom_input, custom_line_width, 1),
            _ => {}
        }
        return effects;
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
            effects.push(response_effect(
                elicitation_id,
                "decline",
                None,
                ElicitationResponseOutcome::Declined,
            ));
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
                    return effects;
                }
                ElicitationFieldKind::SingleSelect { .. } => {
                    state.select_option(field_index, ui.option_cursor);
                }
                ElicitationFieldKind::MultiSelect { .. } if custom_option_selected => {
                    ui.custom_active = true;
                    ui.custom_cursor = state.custom_input.len();
                    state.clear_selection(field_index);
                    return effects;
                }
                ElicitationFieldKind::MultiSelect { .. }
                | ElicitationFieldKind::TextInput
                | ElicitationFieldKind::NumberInput { .. }
                | ElicitationFieldKind::BooleanToggle => {}
            }

            if state.is_valid(None) {
                let elicitation_id = state.elicitation_id.clone();
                let content = state.build_accept_content(None);
                let outcome = selected_outcome(state, field_index, false);
                effects.push(response_effect(
                    elicitation_id,
                    "accept",
                    Some(content),
                    outcome,
                ));
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
    effects
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        chat_state::ElicitationUiState,
        domain::elicitation::{ElicitationField, ElicitationOption},
    };

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn modified_key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn field(name: &str, kind: ElicitationFieldKind) -> ElicitationField {
        ElicitationField {
            name: name.into(),
            title: name.into(),
            description: None,
            required: true,
            kind,
        }
    }

    fn option(value: &str, label: &str) -> ElicitationOption {
        ElicitationOption {
            value: serde_json::json!(value),
            label: label.into(),
            description: None,
        }
    }

    fn select_state(kind: ElicitationFieldKind) -> ElicitationState {
        ElicitationState::new_for_test(vec![field("choice", kind)])
    }

    fn chat_with(state: ElicitationState) -> ChatState {
        let mut chat = ChatState::new();
        chat.elicitation = Some(state);
        chat.elicitation_ui = Some(ElicitationUiState::default());
        chat
    }

    fn handle(
        chat: &mut ChatState,
        render: &mut RenderState,
        code: KeyCode,
    ) -> Vec<ElicitationResponseEffect> {
        handle_key(chat, render, key(code))
    }

    #[test]
    fn single_select_and_decline_emit_exact_typed_responses_without_resolving_state() {
        let mut chat = chat_with(select_state(ElicitationFieldKind::SingleSelect {
            options: vec![option("a", "Alpha"), option("b", "Beta")],
        }));
        let mut render = RenderState::new();

        assert!(handle(&mut chat, &mut render, KeyCode::Down).is_empty());
        assert_eq!(
            handle(&mut chat, &mut render, KeyCode::Enter),
            vec![ElicitationResponseEffect {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "choice": "b" })),
                outcome: ElicitationResponseOutcome::Selected(vec!["Beta".into()]),
            }]
        );
        assert!(chat.elicitation.is_some());

        assert_eq!(
            handle(&mut chat, &mut render, KeyCode::Esc),
            vec![ElicitationResponseEffect {
                elicitation_id: "test-id".into(),
                action: "decline".into(),
                content: None,
                outcome: ElicitationResponseOutcome::Declined,
            }]
        );
        assert!(chat.elicitation.is_some());
    }

    #[test]
    fn multi_select_preserves_wire_selection_order_and_schema_label_order() {
        let mut chat = chat_with(select_state(ElicitationFieldKind::MultiSelect {
            options: vec![option("a", "Alpha"), option("b", "Beta")],
        }));
        let mut render = RenderState::new();

        handle(&mut chat, &mut render, KeyCode::Down);
        handle(&mut chat, &mut render, KeyCode::Char(' '));
        handle(&mut chat, &mut render, KeyCode::Up);
        handle(&mut chat, &mut render, KeyCode::Char(' '));

        assert_eq!(
            handle(&mut chat, &mut render, KeyCode::Enter),
            vec![ElicitationResponseEffect {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "choice": ["b", "a"] })),
                outcome: ElicitationResponseOutcome::Selected(vec!["Alpha".into(), "Beta".into()]),
            }]
        );
    }

    #[test]
    fn text_and_number_fields_preserve_text_outcomes_and_wire_values() {
        let mut render = RenderState::new();
        let mut text = chat_with(ElicitationState::new_for_test(vec![field(
            "name",
            ElicitationFieldKind::TextInput,
        )]));

        handle(&mut text, &mut render, KeyCode::Char('界'));
        handle(&mut text, &mut render, KeyCode::Char('x'));
        handle(&mut text, &mut render, KeyCode::Backspace);
        assert_eq!(
            handle(&mut text, &mut render, KeyCode::Enter),
            vec![ElicitationResponseEffect {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "name": "界" })),
                outcome: ElicitationResponseOutcome::Text("界".into()),
            }]
        );

        let mut number = chat_with(ElicitationState::new_for_test(vec![field(
            "count",
            ElicitationFieldKind::NumberInput { integer: true },
        )]));
        for character in "42".chars() {
            handle(&mut number, &mut render, KeyCode::Char(character));
        }
        assert_eq!(
            handle(&mut number, &mut render, KeyCode::Enter),
            vec![ElicitationResponseEffect {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "count": 42 })),
                outcome: ElicitationResponseOutcome::Text("42".into()),
            }]
        );
    }

    #[test]
    fn boolean_requires_a_value_and_preserves_true_and_false_outcomes() {
        let mut chat = chat_with(ElicitationState::new_for_test(vec![field(
            "confirm",
            ElicitationFieldKind::BooleanToggle,
        )]));
        let mut render = RenderState::new();

        assert!(handle(&mut chat, &mut render, KeyCode::Enter).is_empty());
        handle(&mut chat, &mut render, KeyCode::Char(' '));
        assert_eq!(
            handle(&mut chat, &mut render, KeyCode::Enter)[0].outcome,
            ElicitationResponseOutcome::Boolean(true)
        );
        handle(&mut chat, &mut render, KeyCode::Char(' '));
        assert_eq!(
            handle(&mut chat, &mut render, KeyCode::Enter),
            vec![ElicitationResponseEffect {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "confirm": false })),
                outcome: ElicitationResponseOutcome::Boolean(false),
            }]
        );
    }

    #[test]
    fn custom_editor_routes_utf8_editing_hard_line_and_width_dependent_movement() {
        let mut chat = chat_with(select_state(ElicitationFieldKind::SingleSelect {
            options: vec![option("a", "Alpha")],
        }));
        let mut render = RenderState::new();
        handle(&mut chat, &mut render, KeyCode::Down);
        handle(&mut chat, &mut render, KeyCode::Enter);

        handle(&mut chat, &mut render, KeyCode::Char('界'));
        handle(&mut chat, &mut render, KeyCode::Char('é'));
        handle(&mut chat, &mut render, KeyCode::Left);
        handle(&mut chat, &mut render, KeyCode::Delete);
        assert_eq!(chat.elicitation.as_ref().unwrap().custom_input, "界");
        handle(&mut chat, &mut render, KeyCode::Backspace);
        assert!(chat.elicitation.as_ref().unwrap().custom_input.is_empty());

        for character in "ab\n界d".chars() {
            handle(&mut chat, &mut render, KeyCode::Char(character));
        }
        handle(&mut chat, &mut render, KeyCode::Home);
        assert_eq!(
            chat.elicitation_ui.as_ref().unwrap().custom_cursor,
            "ab\n".len()
        );
        handle(&mut chat, &mut render, KeyCode::End);
        assert_eq!(
            chat.elicitation_ui.as_ref().unwrap().custom_cursor,
            "ab\n界d".len()
        );
        handle(&mut chat, &mut render, KeyCode::Left);
        handle(&mut chat, &mut render, KeyCode::Right);
        assert_eq!(
            chat.elicitation_ui.as_ref().unwrap().custom_cursor,
            "ab\n界d".len()
        );

        chat.elicitation.as_mut().unwrap().custom_input = "abcdef".into();
        chat.elicitation_ui.as_mut().unwrap().custom_cursor = 4;
        render.prepare_elicitation_custom_layout("abcdef", 4, 4, 2);
        handle(&mut chat, &mut render, KeyCode::Up);
        assert_eq!(chat.elicitation_ui.as_ref().unwrap().custom_cursor, 2);
        handle(&mut chat, &mut render, KeyCode::Down);
        assert_eq!(chat.elicitation_ui.as_ref().unwrap().custom_cursor, 4);
    }

    #[test]
    fn custom_newline_accept_and_escape_preserve_trimming_and_geometry_scope() {
        let mut chat = chat_with(select_state(ElicitationFieldKind::SingleSelect {
            options: vec![option("a", "Alpha")],
        }));
        let mut render = RenderState::new();
        handle(&mut chat, &mut render, KeyCode::Down);
        handle(&mut chat, &mut render, KeyCode::Enter);
        handle(&mut chat, &mut render, KeyCode::Char(' '));
        assert!(handle(&mut chat, &mut render, KeyCode::Enter).is_empty());
        assert!(chat.elicitation.is_some());
        assert!(chat.elicitation_ui.as_ref().unwrap().custom_active);
        handle(&mut chat, &mut render, KeyCode::Backspace);
        handle(&mut chat, &mut render, KeyCode::Char(' '));
        handle(&mut chat, &mut render, KeyCode::Char('x'));
        handle_key(
            &mut chat,
            &mut render,
            modified_key(KeyCode::Enter, KeyModifiers::SHIFT),
        );
        handle(&mut chat, &mut render, KeyCode::Char('y'));
        handle(&mut chat, &mut render, KeyCode::Char(' '));

        assert_eq!(
            handle(&mut chat, &mut render, KeyCode::Enter),
            vec![ElicitationResponseEffect {
                elicitation_id: "test-id".into(),
                action: "accept".into(),
                content: Some(serde_json::json!({ "choice": "x\ny" })),
                outcome: ElicitationResponseOutcome::Selected(vec!["x\ny".into()]),
            }]
        );

        render.test_seed_elicitation_custom_geometry(19, 3);
        assert!(handle(&mut chat, &mut render, KeyCode::Esc).is_empty());
        assert!(!chat.elicitation_ui.as_ref().unwrap().custom_active);
        assert_eq!(render.test_elicitation_custom_geometry(), (19, 0, true));
    }

    #[test]
    fn missing_ui_empty_fields_control_chars_and_invalid_values_are_noops() {
        let mut render = RenderState::new();
        let mut chat = ChatState::new();
        assert!(handle(&mut chat, &mut render, KeyCode::Enter).is_empty());

        chat.elicitation = Some(ElicitationState::new_for_test(vec![field(
            "name",
            ElicitationFieldKind::TextInput,
        )]));
        assert!(handle(&mut chat, &mut render, KeyCode::Enter).is_empty());

        chat.elicitation_ui = Some(ElicitationUiState::default());
        assert!(handle(&mut chat, &mut render, KeyCode::Enter).is_empty());
        handle_key(
            &mut chat,
            &mut render,
            modified_key(KeyCode::Char('x'), KeyModifiers::CONTROL),
        );
        assert!(chat.elicitation.as_ref().unwrap().text_input.is_empty());

        chat.elicitation = Some(ElicitationState::new_for_test(Vec::new()));
        for code in [
            KeyCode::Down,
            KeyCode::Char('x'),
            KeyCode::Backspace,
            KeyCode::Enter,
        ] {
            assert!(handle(&mut chat, &mut render, code).is_empty());
        }
    }
}
