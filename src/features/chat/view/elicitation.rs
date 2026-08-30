use ratatui::{
    Frame,
    layout::Rect,
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Paragraph, Wrap},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::chat_state::ElicitationUiState;
use crate::domain::elicitation::{ElicitationFieldKind, ElicitationState};
use crate::render_state::RenderState;
use crate::theme::Theme;

const ARROW_DOWN: &str = "\u{2193}";
const ARROW_UP: &str = "\u{2191}";
const OUTCOME_BULLET: &str = "\u{25B8} ";
const RADIO_SELECTED: &str = "\u{25CF} ";
const RADIO_UNSELECTED: &str = "\u{25CB} ";
const CHECK_CHECKED: &str = "\u{2611} ";
const CHECK_UNCHECKED: &str = "\u{2610} ";

fn wrap_elicitation_message(text: &str, width: u16) -> Vec<String> {
    let width = width.max(1) as usize;
    let mut rows = Vec::new();

    for logical_line in text.lines() {
        let mut row = String::new();
        let mut row_width = 0;
        for word in logical_line.split_whitespace() {
            let word_width = UnicodeWidthStr::width(word);
            if !row.is_empty() && row_width + 1 + word_width > width {
                rows.push(std::mem::take(&mut row));
                row_width = 0;
            }

            if word_width > width {
                if !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                    row_width = 0;
                }
                for ch in word.chars() {
                    let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if row_width > 0 && row_width + ch_width > width {
                        rows.push(std::mem::take(&mut row));
                        row_width = 0;
                    }
                    row.push(ch);
                    row_width += ch_width;
                }
            } else {
                if !row.is_empty() {
                    row.push(' ');
                    row_width += 1;
                }
                row.push_str(word);
                row_width += word_width;
            }
        }
        rows.push(row);
    }

    if rows.is_empty() {
        rows.push(String::new());
    }
    rows
}

fn wrapped_text_rows(text: &str, width: u16) -> u16 {
    wrap_elicitation_message(text, width)
        .len()
        .min(u16::MAX as usize) as u16
}

fn elicitation_option_text(label: &str, description: Option<&str>) -> String {
    description
        .map(|description| format!("{label}  {description}"))
        .unwrap_or_else(|| label.to_string())
}

fn elicitation_option_rows(state: &ElicitationState, ui: &ElicitationUiState, width: u16) -> u16 {
    let schema_rows = match ui
        .current_field_index(state.fields.len())
        .and_then(|index| state.fields.get(index))
        .map(|field| &field.kind)
    {
        Some(ElicitationFieldKind::SingleSelect { options })
        | Some(ElicitationFieldKind::MultiSelect { options }) => options
            .iter()
            .map(|option| {
                wrapped_text_rows(
                    &elicitation_option_text(&option.label, option.description.as_deref()),
                    width,
                )
            })
            .sum(),
        _ => 1,
    };
    let option_count = match ui
        .current_field_index(state.fields.len())
        .and_then(|index| state.fields.get(index))
        .map(|field| &field.kind)
    {
        Some(ElicitationFieldKind::SingleSelect { options })
        | Some(ElicitationFieldKind::MultiSelect { options }) => options.len(),
        _ => 0,
    };
    schema_rows + u16::from(state.allow_custom && option_count > 0)
}

pub(crate) fn popup_height(
    state: Option<&ElicitationState>,
    ui: Option<&ElicitationUiState>,
    render: &mut RenderState,
    area: Rect,
) -> u16 {
    if let (Some(state), Some(ui)) = (state, ui) {
        let message_width = area.width.saturating_sub(4).max(1);
        let message_rows = wrapped_text_rows(&state.message, message_width);
        let option_width = area.width.saturating_sub(6).max(1);
        let option_rows = elicitation_option_rows(state, ui, option_width);
        let custom_rows = if ui.custom_active {
            let width = area.width.saturating_sub(4) as usize;
            render
                .prepare_elicitation_custom_layout(&state.custom_input, ui.custom_cursor, width, 2)
                .total_rows() as u16
        } else {
            0
        };
        // Top padding, title, wrapped message, choices/input, and hint, each separated as needed.
        let content_rows = option_rows.max(1) + custom_rows.min(5);
        let custom_spacing = u16::from(ui.custom_active);
        (7 + message_rows.max(1) + content_rows + custom_spacing).min(area.height.saturating_sub(3))
    } else {
        0
    }
}

pub(crate) fn draw_popup(
    f: &mut Frame,
    state: &ElicitationState,
    ui: &ElicitationUiState,
    render: &mut RenderState,
    area: Rect,
) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    f.render_widget(Block::default().style(Theme::popup_bg()), area);
    let inner = Rect {
        x: area.x + 1,
        y: area.y,
        width: area.width.saturating_sub(2),
        height: area.height,
    };
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let mut row = inner.y;
    let max_y = inner.y + inner.height;
    // Leave one row of padding before the header.
    row = row.saturating_add(1).min(max_y);
    if row < max_y {
        let title_style = Theme::status_accent().add_modifier(Modifier::BOLD);
        let title = Line::from(vec![
            Span::styled(OUTCOME_BULLET, title_style),
            Span::styled("Question", title_style),
        ]);
        f.render_widget(
            Paragraph::new(title).style(Theme::popup_bg()),
            Rect::new(inner.x, row, inner.width, 1),
        );
        row += 1;
    }
    // Keep the title visually separate from the question body.
    row = row.saturating_add(1).min(max_y);
    if row < max_y {
        let message_area = Rect::new(inner.x + 2, row, inner.width.saturating_sub(2), max_y - row);
        let message_lines: Vec<Line<'static>> =
            wrap_elicitation_message(&state.message, message_area.width)
                .into_iter()
                .map(|line| Line::from(Span::styled(line, Theme::fg())))
                .collect();
        let message_rows = message_lines.len().max(1) as u16;
        let message = Paragraph::new(message_lines).style(Theme::popup_bg());
        let visible_rows = message_rows.min(message_area.height);
        f.render_widget(
            message,
            Rect::new(
                message_area.x,
                message_area.y,
                message_area.width,
                visible_rows,
            ),
        );
        row += visible_rows;
    }
    // Keep the choices visually separate from the question text.
    row = row.saturating_add(1).min(max_y);

    let Some(field) = ui
        .current_field_index(state.fields.len())
        .and_then(|index| state.fields.get(index))
        .cloned()
    else {
        return;
    };
    match &field.kind {
        ElicitationFieldKind::SingleSelect { options }
        | ElicitationFieldKind::MultiSelect { options } => {
            let is_multi = matches!(&field.kind, ElicitationFieldKind::MultiSelect { .. });
            let selected_vals = state.selected.get(&field.name);
            for (idx, opt) in options.iter().enumerate() {
                if row >= max_y {
                    break;
                }
                let highlighted = !ui.custom_active && idx == ui.option_cursor;
                let is_chosen = if is_multi {
                    matches!(selected_vals, Some(serde_json::Value::Array(arr)) if arr.contains(&opt.value))
                } else {
                    selected_vals == Some(&opt.value)
                };
                let bullet = if is_multi {
                    if is_chosen {
                        CHECK_CHECKED
                    } else {
                        CHECK_UNCHECKED
                    }
                } else if highlighted {
                    RADIO_SELECTED
                } else {
                    RADIO_UNSELECTED
                };
                let style = if highlighted {
                    Theme::status_accent()
                } else {
                    Theme::status()
                };
                let text_width = inner.width.saturating_sub(4).max(1);
                let option_rows = wrapped_text_rows(
                    &elicitation_option_text(&opt.label, opt.description.as_deref()),
                    text_width,
                )
                .min(max_y - row);
                f.render_widget(
                    Paragraph::new(Span::styled(format!("  {bullet}"), style))
                        .style(Theme::popup_bg()),
                    Rect::new(inner.x, row, 4.min(inner.width), option_rows),
                );
                let option_line = Line::from(vec![
                    Span::styled(opt.label.clone(), style),
                    opt.description
                        .as_ref()
                        .map(|desc| Span::styled(format!("  {desc}"), Theme::dim()))
                        .unwrap_or_else(|| Span::raw("")),
                ]);
                f.render_widget(
                    Paragraph::new(option_line)
                        .style(Theme::popup_bg())
                        .wrap(Wrap { trim: true }),
                    Rect::new(inner.x + 4, row, text_width, option_rows),
                );
                row += option_rows;
            }

            if state.allow_custom && !options.is_empty() && row < max_y {
                let highlighted = ui.option_cursor == options.len();
                let style = if highlighted || ui.custom_active {
                    Theme::status_accent()
                } else {
                    Theme::status()
                };
                let bullet = if is_multi {
                    if ui.custom_active {
                        CHECK_CHECKED
                    } else {
                        CHECK_UNCHECKED
                    }
                } else if highlighted {
                    RADIO_SELECTED
                } else {
                    RADIO_UNSELECTED
                };
                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(format!("  {bullet}"), style),
                        Span::styled("Custom answer…", style),
                    ]))
                    .style(Theme::popup_bg()),
                    Rect::new(inner.x, row, inner.width, 1),
                );
                row += 1;
            }

            if ui.custom_active && row < max_y {
                // Keep the custom editor visually separate from the choices.
                row = row.saturating_add(1).min(max_y);
                let layout = render.prepare_elicitation_custom_layout(
                    &state.custom_input,
                    ui.custom_cursor,
                    inner.width.saturating_sub(2) as usize,
                    2,
                );
                let total_rows = layout.total_rows() as u16;
                // Reserve a spacer, the help row, and bottom padding below the editor.
                let visible = total_rows.min(max_y.saturating_sub(row + 3)).min(5);
                let scroll = render.ensure_elicitation_custom_cursor_visible(visible);

                let lines: Vec<Line<'static>> = layout
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(idx, visual_row)| {
                        if idx == 0 {
                            Line::from(vec![
                                Span::styled("> ", Theme::status_accent()),
                                Span::styled(visual_row.text.clone(), Theme::fg()),
                            ])
                        } else {
                            Line::from(Span::styled(visual_row.text.clone(), Theme::fg()))
                        }
                    })
                    .collect();
                f.render_widget(
                    Paragraph::new(lines)
                        .style(Theme::popup_bg())
                        .scroll((scroll, 0)),
                    Rect::new(inner.x + 2, row, inner.width.saturating_sub(2), visible),
                );
                let cursor_row = (layout.cursor_row as u16).saturating_sub(scroll);
                if cursor_row < visible {
                    f.set_cursor_position((
                        inner.x + 2 + layout.cursor_col as u16,
                        row + cursor_row,
                    ));
                }
                row += visible;
            }
        }
        ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
            let placeholder = if matches!(&field.kind, ElicitationFieldKind::NumberInput { .. }) {
                "enter number..."
            } else {
                "enter text..."
            };
            let display = if state.text_input.is_empty() {
                Span::styled(placeholder, Theme::dim())
            } else {
                Span::styled(state.text_input.clone(), Theme::fg())
            };
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("  > ", Theme::status_accent()),
                    display,
                ]))
                .style(Theme::popup_bg()),
                Rect::new(inner.x, row, inner.width, 1),
            );
            row += 1;
        }
        ElicitationFieldKind::BooleanToggle => {
            let value = state
                .selected
                .get(&field.name)
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        if value {
                            format!("  {CHECK_CHECKED}Yes")
                        } else {
                            format!("  {CHECK_UNCHECKED}No")
                        },
                        Theme::status_accent(),
                    ),
                    Span::styled("  (Space to toggle)", Theme::dim()),
                ]))
                .style(Theme::popup_bg()),
                Rect::new(inner.x, row, inner.width, 1),
            );
            row += 1;
        }
    }

    // Keep controls visually separate from the answer area.
    row = row.saturating_add(1).min(max_y);
    if row < max_y {
        let hint = if ui.custom_active {
            " type answer  Shift+Enter newline  Enter submit  Esc back".to_string()
        } else {
            match field.kind {
                ElicitationFieldKind::MultiSelect { .. } => format!(
                    " {ARROW_UP}{ARROW_DOWN} navigate  Space toggle  Enter submit  Esc decline"
                ),
                ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
                    " type answer  Enter submit  Esc decline".to_string()
                }
                ElicitationFieldKind::BooleanToggle => {
                    " Space toggle  Enter submit  Esc decline".to_string()
                }
                ElicitationFieldKind::SingleSelect { .. } => {
                    format!(" {ARROW_UP}{ARROW_DOWN} navigate  Enter select  Esc decline")
                }
            }
        };
        f.render_widget(
            Paragraph::new(Span::styled(hint, Theme::dim())).style(Theme::popup_bg()),
            Rect::new(inner.x, row, inner.width, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use ratatui::{
        backend::{Backend, TestBackend},
        layout::Position,
    };

    use super::*;
    use crate::domain::elicitation::{ElicitationField, ElicitationOption};

    fn field(kind: ElicitationFieldKind) -> ElicitationField {
        ElicitationField {
            name: "answer".into(),
            title: "Answer".into(),
            description: Some("field description".into()),
            required: true,
            kind,
        }
    }

    fn option(value: &str, label: &str, description: Option<&str>) -> ElicitationOption {
        ElicitationOption {
            value: serde_json::json!(value),
            label: label.into(),
            description: description.map(str::to_string),
        }
    }

    fn render(
        state: &ElicitationState,
        ui: &ElicitationUiState,
        render: &mut RenderState,
        width: u16,
        height: u16,
    ) -> (ratatui::buffer::Buffer, Position) {
        let backend = TestBackend::new(width, height);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| draw_popup(frame, state, ui, render, frame.area()))
            .unwrap();
        let cursor = terminal.backend_mut().get_cursor_position().unwrap();
        (terminal.backend().buffer().clone(), cursor)
    }

    fn find_text(buffer: &ratatui::buffer::Buffer, needle: &str) -> (u16, u16) {
        for y in 0..buffer.area.height {
            let line = (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>();
            if let Some(byte_index) = line.find(needle) {
                return (line[..byte_index].chars().count() as u16, y);
            }
        }
        panic!("missing {needle:?}");
    }

    #[derive(Debug, PartialEq)]
    struct SemanticSnapshot {
        elicitation_id: String,
        message: String,
        source: String,
        fields: Vec<ElicitationField>,
        selected: std::collections::HashMap<String, serde_json::Value>,
        text_input: String,
        custom_input: String,
        allow_custom: bool,
        field_cursor: usize,
        option_cursor: usize,
        text_cursor: usize,
        custom_active: bool,
        custom_cursor: usize,
    }

    fn semantic_snapshot(state: &ElicitationState, ui: &ElicitationUiState) -> SemanticSnapshot {
        SemanticSnapshot {
            elicitation_id: state.elicitation_id.clone(),
            message: state.message.clone(),
            source: state.source.clone(),
            fields: state.fields.clone(),
            selected: state.selected.clone(),
            text_input: state.text_input.clone(),
            custom_input: state.custom_input.clone(),
            allow_custom: state.allow_custom,
            field_cursor: ui.field_cursor,
            option_cursor: ui.option_cursor,
            text_cursor: ui.text_cursor,
            custom_active: ui.custom_active,
            custom_cursor: ui.custom_cursor,
        }
    }

    #[test]
    fn popup_renders_title_select_text_and_boolean_paths_without_editor_geometry() {
        let mut render_state = RenderState::new();
        let ui = ElicitationUiState::default();
        let mut select =
            ElicitationState::new_for_test(vec![field(ElicitationFieldKind::SingleSelect {
                options: vec![
                    option("a", "Alpha", Some("fast path")),
                    option("b", "Beta", None),
                ],
            })]);
        select.message = "Choose a path".into();
        let select_snapshot = semantic_snapshot(&select, &ui);

        let (buffer, _) = render(&select, &ui, &mut render_state, 48, 12);
        let title = find_text(&buffer, "Question");
        let alpha = find_text(&buffer, "Alpha  fast path");
        let beta = find_text(&buffer, "Beta");
        let hint = find_text(&buffer, "↑↓ navigate  Enter select  Esc decline");
        assert_eq!(buffer[title].style().fg, Theme::status_accent().fg);
        assert!(buffer[title].modifier.contains(Modifier::BOLD));
        assert_eq!(buffer[alpha].style().fg, Theme::status_accent().fg);
        assert_eq!(buffer[beta].style().fg, Theme::status().fg);
        assert!(alpha.1 < beta.1 && beta.1 < hint.1);
        assert_eq!(semantic_snapshot(&select, &ui), select_snapshot);
        assert_eq!(
            render_state.test_elicitation_custom_geometry(),
            (1, 0, false)
        );

        let mut text = ElicitationState::new_for_test(vec![field(ElicitationFieldKind::TextInput)]);
        text.message = "Explain".into();
        text.text_input = "typed answer".into();
        let text_snapshot = semantic_snapshot(&text, &ui);
        let (buffer, _) = render(&text, &ui, &mut render_state, 48, 10);
        let input = find_text(&buffer, "> typed answer");
        let hint = find_text(&buffer, "type answer  Enter submit  Esc decline");
        assert_eq!(buffer[input].style().fg, Theme::status_accent().fg);
        assert!(input.1 < hint.1);
        assert_eq!(semantic_snapshot(&text, &ui), text_snapshot);

        let mut boolean =
            ElicitationState::new_for_test(vec![field(ElicitationFieldKind::BooleanToggle)]);
        boolean.message = "Enable it?".into();
        boolean
            .selected
            .insert("answer".into(), serde_json::json!(true));
        let boolean_snapshot = semantic_snapshot(&boolean, &ui);
        let (buffer, _) = render(&boolean, &ui, &mut render_state, 48, 10);
        let value = find_text(&buffer, "☑ Yes  (Space to toggle)");
        let hint = find_text(&buffer, "Space toggle  Enter submit  Esc decline");
        assert_eq!(buffer[value].style().fg, Theme::status_accent().fg);
        assert!(value.1 < hint.1);
        assert_eq!(semantic_snapshot(&boolean, &ui), boolean_snapshot);
        assert_eq!(
            render_state.test_elicitation_custom_geometry(),
            (1, 0, false)
        );
    }

    #[test]
    fn custom_editor_scrolls_places_cursor_and_mutates_only_render_geometry() {
        let mut state =
            ElicitationState::new_for_test(vec![field(ElicitationFieldKind::SingleSelect {
                options: vec![option("a", "Alpha", None)],
            })]);
        state.message = "Choose or customize".into();
        state.custom_input = "a\nb\nc\nd\ne\nf\ng".into();
        state.text_input = "semantic text".into();
        state
            .selected
            .insert("answer".into(), serde_json::json!("a"));
        let ui = ElicitationUiState {
            option_cursor: 1,
            text_cursor: 3,
            custom_active: true,
            custom_cursor: state.custom_input.len(),
            ..Default::default()
        };
        let snapshot = semantic_snapshot(&state, &ui);
        let mut render_state = RenderState::new();
        render_state.test_seed_composer_input_geometry(9, 2);

        assert_eq!(
            popup_height(
                Some(&state),
                Some(&ui),
                &mut render_state,
                Rect::new(0, 0, 20, 18)
            ),
            15
        );
        let (buffer, cursor) = render(&state, &ui, &mut render_state, 40, 15);

        find_text(&buffer, "Custom answer…");
        find_text(&buffer, "Shift+Enter newline");
        assert_eq!(cursor, Position::new(4, 11));
        assert_eq!(
            render_state.test_elicitation_custom_geometry(),
            (36, 3, true)
        );
        assert_eq!(render_state.test_composer_input_geometry(), (9, 2, true));
        assert_eq!(semantic_snapshot(&state, &ui), snapshot);
    }

    #[test]
    fn empty_fields_and_tiny_areas_do_not_panic_or_publish_custom_geometry() {
        let mut state = ElicitationState::new_for_test(Vec::new());
        state.message = "Unsupported question".into();
        let ui = ElicitationUiState::default();
        let snapshot = semantic_snapshot(&state, &ui);
        let mut render_state = RenderState::new();

        let (buffer, _) = render(&state, &ui, &mut render_state, 24, 8);
        find_text(&buffer, "Question");
        find_text(&buffer, "Unsupported question");
        assert_eq!(popup_height(None, None, &mut render_state, buffer.area), 0);
        assert_eq!(
            render_state.test_elicitation_custom_geometry(),
            (1, 0, false)
        );

        let _ = render(&state, &ui, &mut render_state, 1, 1);
        let _ = render(&state, &ui, &mut render_state, 0, 0);
        assert_eq!(semantic_snapshot(&state, &ui), snapshot);
        assert_eq!(
            render_state.test_elicitation_custom_geometry(),
            (1, 0, false)
        );
    }

    #[test]
    fn custom_editor_draw_owns_scroll_resize_and_preserves_elicitation_semantics() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "choice".into(),
            title: "Choice".into(),
            description: Some("description".into()),
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![option("a", "Alpha", None)],
            },
        }]);
        state.custom_input = "a\nb\nc\nd\ne\nf\ng".into();
        state.text_input = "semantic text".into();
        state
            .selected
            .insert("choice".into(), serde_json::json!("a"));
        let mut ui = ElicitationUiState {
            field_cursor: 0,
            option_cursor: 1,
            text_cursor: 4,
            custom_active: true,
            custom_cursor: state.custom_input.len(),
        };
        let snapshot = semantic_snapshot(&state, &ui);
        let mut render_state = RenderState::new();

        let height = popup_height(
            Some(&state),
            Some(&ui),
            &mut render_state,
            Rect::new(0, 0, 40, 24),
        );
        render(&state, &ui, &mut render_state, 40, height);
        assert_eq!(
            render_state.test_elicitation_custom_geometry(),
            (36, 2, true)
        );
        assert_eq!(semantic_snapshot(&state, &ui), snapshot);

        ui.custom_cursor = 0;
        let height = popup_height(
            Some(&state),
            Some(&ui),
            &mut render_state,
            Rect::new(0, 0, 20, 24),
        );
        render(&state, &ui, &mut render_state, 20, height);
        assert_eq!(
            render_state.test_elicitation_custom_geometry(),
            (16, 0, true)
        );

        let height = popup_height(
            Some(&state),
            Some(&ui),
            &mut render_state,
            Rect::new(0, 0, 3, 4),
        );
        render(&state, &ui, &mut render_state, 3, height);
        assert_eq!(render_state.elicitation_custom_line_width(), 1);
    }

    #[test]
    fn draw_chat_does_not_panic_with_empty_elicitation_fields() {
        let state = ElicitationState::new_for_test(vec![]);
        let ui = ElicitationUiState::default();
        let mut render_state = RenderState::new();

        let _buffer = render(&state, &ui, &mut render_state, 80, 9);
    }

    #[test]
    fn draw_chat_custom_elicitation_wraps_and_expands_with_prefix() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "choice".into(),
            title: "Choice".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![option("a", "Alpha", None)],
            },
        }]);
        state.custom_input = "a deliberately long custom response that wraps\nsecond line".into();
        let ui = ElicitationUiState {
            option_cursor: 1,
            custom_active: true,
            custom_cursor: state.custom_input.len(),
            ..Default::default()
        };
        let mut render_state = RenderState::new();
        let height = popup_height(
            Some(&state),
            Some(&ui),
            &mut render_state,
            Rect::new(0, 0, 40, 21),
        );

        let (buffer, _) = render(&state, &ui, &mut render_state, 40, height);
        let rendered = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(rendered.contains("Custom answer…"));
        assert!(rendered.contains("> a deliberately long custom"));
        assert!(rendered.contains("second line"));
        assert!(rendered.contains("Shift+Enter newline"));
    }

    #[test]
    fn draw_chat_wraps_long_elicitation_question() {
        let mut state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "choice".into(),
            title: "Choice".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![option("a", "Alpha", None)],
            },
        }]);
        state.message = "When designing a new system-level tool, which approach best describes how you balance rapid prototyping and maintainability?".into();
        let ui = ElicitationUiState::default();
        let mut render_state = RenderState::new();
        let height = popup_height(
            Some(&state),
            Some(&ui),
            &mut render_state,
            Rect::new(0, 0, 50, 21),
        );

        let (buffer, _) = render(&state, &ui, &mut render_state, 50, height);
        let first = find_text(&buffer, "When designing");
        let second = find_text(&buffer, "best describes how");
        let option = find_text(&buffer, "Alpha");

        assert!(second.1 > first.1);
        assert!(option.1 > second.1);
    }

    #[test]
    fn draw_chat_wraps_long_elicitation_answers() {
        let state = ElicitationState::new_for_test(vec![ElicitationField {
            name: "choice".into(),
            title: "Choice".into(),
            description: None,
            required: true,
            kind: ElicitationFieldKind::SingleSelect {
                options: vec![option(
                    "a",
                    "Keep the validated prototype as the production foundation and improve it gradually through careful iteration",
                    None,
                )],
            },
        }]);
        let ui = ElicitationUiState::default();
        let mut render_state = RenderState::new();
        let height = popup_height(
            Some(&state),
            Some(&ui),
            &mut render_state,
            Rect::new(0, 0, 50, 21),
        );

        let (buffer, _) = render(&state, &ui, &mut render_state, 50, height);
        let first = find_text(&buffer, "Keep the validated");
        let second = find_text(&buffer, "foundation and improve");
        let custom = find_text(&buffer, "Custom answer…");

        assert!(second.1 > first.1);
        assert!(custom.1 > second.1);
    }
}
