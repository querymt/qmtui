use ratatui::{
    Frame,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Padding, Paragraph},
};

use crate::chat_state::ChatState;
use crate::composer_state::ComposerState;
use crate::domain::activity::{ActivityState, SessionOp};
use crate::input_layout::InputVisualLayout;
use crate::render_state::RenderState;
use crate::theme::Theme;

const ARROW_UP: &str = "\u{2191}";
const INPUT_OVERLINE: &str = "\u{00AF}";

pub(crate) fn input_layout_metrics(
    composer: &ComposerState,
    render: &mut RenderState,
    area: Rect,
) -> (u16, InputVisualLayout) {
    // Compute how many visual rows the input text needs when wrapped.
    let input_inner_width = area.width.saturating_sub(4) as usize;
    let prefix_width = 2usize; // "> "
    let input_layout = render.prepare_composer_input_layout(
        &composer.input,
        composer.input_cursor,
        input_inner_width,
        prefix_width,
    );
    let max_input_lines: u16 = 5;
    let input_height = (input_layout.total_rows() as u16).clamp(1, max_input_lines) + 1; // +1 bottom padding

    (input_height, input_layout)
}

pub(crate) fn draw_input_panel(
    f: &mut Frame,
    chat: &ChatState,
    agent_mode: &str,
    spinner_frame: &str,
    render: &mut RenderState,
    areas: (Rect, Rect),
    input_layout: InputVisualLayout,
) {
    let (border_area, input_area) = areas;
    // input border line reflects active session state
    let border_style = match &chat.activity {
        ActivityState::SessionOp(SessionOp::Undo) => Theme::input_border_undo(),
        ActivityState::SessionOp(SessionOp::Redo) => Theme::input_border_redo(),
        _ if chat.cancel_confirm_active() => Theme::input_border_cancel_confirm(),
        ActivityState::Compacting { .. } => Theme::input_border_compacting(),
        ActivityState::Thinking | ActivityState::Streaming | ActivityState::RunningTool { .. } => {
            Theme::input_border_thinking()
        }
        _ if chat.elicitation.is_some() => Theme::input_border_thinking(), // accent while waiting
        _ => Theme::mode_border(agent_mode),
    };
    let border_line =
        Paragraph::new(INPUT_OVERLINE.repeat(border_area.width as usize)).style(border_style);
    f.render_widget(border_line, border_area);

    // input area
    let input_bg = Block::default()
        .padding(Padding::new(2, 2, 0, 1))
        .style(Theme::input());
    let inner = input_bg.inner(input_area);
    f.render_widget(input_bg, input_area);

    let (label_text, label_style) = match &chat.activity {
        ActivityState::SessionOp(SessionOp::Undo) => {
            (format!("{spinner_frame} undoing "), Theme::input_undo())
        }
        ActivityState::SessionOp(SessionOp::Redo) => {
            (format!("{spinner_frame} redoing "), Theme::input_redo())
        }
        _ if chat.cancel_confirm_active() => (
            format!("{spinner_frame} Esc again to stop "),
            Theme::input_cancel_confirm(),
        ),
        ActivityState::Compacting { .. }
        | ActivityState::RunningTool { .. }
        | ActivityState::Thinking
        | ActivityState::Streaming => (format!("{spinner_frame} "), Theme::input_thinking()),
        _ if chat.elicitation.is_some() => (
            format!("  answer above {ARROW_UP} "),
            Theme::input_thinking(),
        ),
        _ => ("> ".into(), Theme::mode_border(agent_mode)),
    };
    let input_style = Theme::input();
    let hide_input_contents = chat.should_hide_input_contents();

    if hide_input_contents {
        render.ensure_composer_input_cursor_visible(inner.height, true);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(label_text, label_style))),
            inner,
        );
    } else {
        let layout = input_layout;
        let mut lines: Vec<Line<'static>> = Vec::new();
        for (idx, row) in layout.rows.iter().enumerate() {
            if idx == 0 {
                lines.push(Line::from(vec![
                    Span::styled(label_text.clone(), label_style),
                    Span::styled(row.text.clone(), input_style),
                ]));
            } else {
                lines.push(Line::from(Span::styled(row.text.clone(), input_style)));
            }
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled("", input_style)));
        }

        let visible = inner.height;
        let scroll = render.ensure_composer_input_cursor_visible(visible, false);

        f.render_widget(Paragraph::new(lines).scroll((scroll, 0)), inner);

        let visual_row = (layout.cursor_row as u16).saturating_sub(scroll);
        if visual_row < visible {
            f.set_cursor_position((inner.x + layout.cursor_col as u16, inner.y + visual_row));
        }
    }
}

#[cfg(test)]
mod tests {
    use ratatui::{
        backend::{Backend, TestBackend},
        layout::Position,
    };

    use super::*;

    fn render(
        composer: &ComposerState,
        chat: &ChatState,
        render: &mut RenderState,
        width: u16,
        input_height: u16,
    ) -> (ratatui::buffer::Buffer, Position) {
        let full_area = Rect::new(0, 0, width, input_height + 1);
        let (_, layout) = input_layout_metrics(composer, render, full_area);
        let backend = TestBackend::new(full_area.width, full_area.height);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                draw_input_panel(
                    frame,
                    chat,
                    "build",
                    "spin",
                    render,
                    (
                        Rect::new(0, 0, width, 1),
                        Rect::new(0, 1, width, input_height),
                    ),
                    layout.clone(),
                );
            })
            .unwrap();
        let cursor = terminal.backend_mut().get_cursor_position().unwrap();
        (terminal.backend().buffer().clone(), cursor)
    }

    fn buffer_line(buffer: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buffer.area.width)
            .map(|x| buffer[(x, y)].symbol())
            .collect()
    }

    #[test]
    fn visible_input_preserves_hard_newline_cursor_geometry_and_semantics() {
        let mut composer = ComposerState::new();
        composer.input = "alpha\nbeta".into();
        composer.input_cursor = "alpha\n".len();
        composer.input_preferred_col = Some(4);
        let chat = ChatState::new();
        let composer_snapshot = (
            composer.input.clone(),
            composer.input_cursor,
            composer.input_preferred_col,
            composer.file_index.clone(),
            composer.file_index_generated_at,
            composer.file_index_loading,
            composer.file_index_error.clone(),
            composer.mention_state.is_some(),
            composer.slash_state.is_some(),
        );
        let chat_snapshot = (
            chat.activity.clone(),
            chat.messages.len(),
            chat.streaming_content.clone(),
            chat.streaming_thinking.clone(),
            chat.elicitation.is_some(),
            chat.show_thinking,
            chat.suppress_turn_output,
        );
        let mut render_state = RenderState::new();

        let (buffer, cursor) = render(&composer, &chat, &mut render_state, 20, 3);

        assert_eq!(buffer_line(&buffer, 0), INPUT_OVERLINE.repeat(20));
        assert_eq!(buffer_line(&buffer, 1), format!("{:<20}", "  > alpha"));
        assert_eq!(buffer_line(&buffer, 2), format!("{:<20}", "  beta"));
        assert_eq!(cursor, Position::new(2, 2));
        assert_eq!(render_state.test_composer_input_geometry(), (16, 0, true));
        assert_eq!(buffer[(2, 1)].style().fg, Theme::mode_border("build").fg);
        assert_eq!(buffer[(4, 1)].style().fg, Theme::input().fg);
        assert_eq!(buffer[(4, 1)].style().bg, Theme::input().bg);
        assert_eq!(
            (
                composer.input.clone(),
                composer.input_cursor,
                composer.input_preferred_col,
                composer.file_index.clone(),
                composer.file_index_generated_at,
                composer.file_index_loading,
                composer.file_index_error.clone(),
                composer.mention_state.is_some(),
                composer.slash_state.is_some(),
            ),
            composer_snapshot
        );
        assert_eq!(
            (
                chat.activity.clone(),
                chat.messages.len(),
                chat.streaming_content.clone(),
                chat.streaming_thinking.clone(),
                chat.elicitation.is_some(),
                chat.show_thinking,
                chat.suppress_turn_output,
            ),
            chat_snapshot
        );
    }

    #[test]
    fn hidden_activity_label_resets_only_render_scroll() {
        let mut composer = ComposerState::new();
        composer.input = "a\nb\nc\nd\ne\nf\ng\nh".into();
        composer.input_cursor = composer.input.len();
        composer.input_preferred_col = Some(1);
        let visible_chat = ChatState::new();
        let mut render_state = RenderState::new();

        render(&composer, &visible_chat, &mut render_state, 20, 3);
        assert_eq!(render_state.test_composer_input_geometry(), (16, 6, true));

        let mut hidden_chat = ChatState::new();
        hidden_chat.activity = ActivityState::SessionOp(SessionOp::Undo);
        let composer_snapshot = (
            composer.input.clone(),
            composer.input_cursor,
            composer.input_preferred_col,
        );
        let chat_snapshot = hidden_chat.activity.clone();
        let (buffer, _) = render(&composer, &hidden_chat, &mut render_state, 20, 3);

        assert_eq!(
            buffer_line(&buffer, 1),
            format!("{:<20}", "  spin undoing ")
        );
        assert!(!buffer.content().iter().any(|cell| cell.symbol() == "h"));
        assert_eq!(buffer[(2, 1)].style().fg, Theme::input_undo().fg);
        assert_eq!(buffer[(2, 1)].style().bg, Theme::input_undo().bg);
        assert!(
            buffer[(2, 1)]
                .modifier
                .contains(ratatui::style::Modifier::BOLD)
        );
        assert_eq!(render_state.test_composer_input_geometry(), (16, 0, true));
        assert_eq!(
            (
                composer.input.clone(),
                composer.input_cursor,
                composer.input_preferred_col,
            ),
            composer_snapshot
        );
        assert_eq!(hidden_chat.activity, chat_snapshot);
    }
}
