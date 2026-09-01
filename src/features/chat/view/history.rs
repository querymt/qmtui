use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Cell, Clear, Paragraph, Row, Table, TableState},
};

use crate::chat_state::ChatState;
use crate::theme::Theme;
use crate::view_shared::{ELLIPSIS, popup_rect, scroll_input, truncate_with_ellipsis};

pub(crate) struct ForkPopupInput<'a> {
    pub(crate) chat: &'a ChatState,
}

pub(crate) fn draw_fork_turn_popup(f: &mut Frame, input: ForkPopupInput<'_>) {
    let chat = input.chat;
    let area = f.area();
    let popup_area = popup_rect(
        area,
        area.width.saturating_sub(4) as usize,
        area.height.saturating_sub(4) as usize,
        36..=84,
        6..=12,
        2,
    );

    f.render_widget(Clear, popup_area);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup_area);

    let inner = Rect {
        x: popup_area.x + 1,
        y: popup_area.y + 1,
        width: popup_area.width.saturating_sub(2),
        height: popup_area.height.saturating_sub(2),
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Span::styled("Fork Session", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );

    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (filter_display, filter_cur) =
        scroll_input(&chat.fork_filter, chat.fork_filter.len(), avail);
    let input_line = Line::from(vec![
        Span::styled("/ ", Theme::popup_title()),
        Span::styled(filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(input_line).style(Theme::popup_bg()),
        chunks[1],
    );
    if chunks[1].width > 2 && chunks[1].height > 0 {
        f.set_cursor_position((
            chunks[1]
                .x
                .saturating_add(2)
                .saturating_add(filter_cur as u16),
            chunks[1].y,
        ));
    }

    let turns = chat.visible_fork_turns();
    if turns.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("No forkable turns", Theme::status()))
                .style(Theme::popup_bg()),
            chunks[2],
        );
    } else {
        let row_width = chunks[2].width as usize;
        let preview_budget = row_width.saturating_sub(8) / 2;
        let selected_idx = chat.fork_cursor.min(turns.len().saturating_sub(1));
        let selected = Some(selected_idx);
        let dim_style = Theme::status().add_modifier(Modifier::DIM);
        let selected_style = Theme::selected();
        let selected_bg = selected_style.bg.unwrap_or_else(Theme::bg_hl);
        let selected_dim_style = Style::default()
            .fg(dim_style.fg.unwrap_or_else(Theme::dim))
            .bg(selected_bg)
            .add_modifier(Modifier::DIM);
        let rows: Vec<Row> = turns
            .iter()
            .enumerate()
            .map(|(idx, turn)| {
                let user_text = turn.user_preview.replace('\n', " ");
                let assistant_text = turn.assistant_preview.replace('\n', " ");
                let user = truncate_with_ellipsis(&user_text, preview_budget);
                let assistant = truncate_with_ellipsis(&assistant_text, preview_budget);
                let boundary_style = if idx == selected_idx {
                    selected_dim_style
                } else {
                    dim_style
                };
                let message_style = if idx == selected_idx {
                    Theme::selected()
                } else {
                    Theme::popup_bg()
                };
                Row::new(vec![
                    Cell::from(Span::styled(turn.turn_index.to_string(), boundary_style)),
                    Cell::from(Span::styled(user, message_style)),
                    Cell::from(Span::styled(ELLIPSIS, boundary_style)),
                    Cell::from(Span::styled(assistant, message_style)),
                ])
            })
            .collect();
        let table = Table::new(
            rows,
            [
                Constraint::Length(3),
                Constraint::Percentage(50),
                Constraint::Length(1),
                Constraint::Percentage(50),
            ],
        )
        .block(Block::default().style(Theme::popup_bg()))
        .style(Theme::popup_bg())
        .row_highlight_style(Theme::selected());
        let mut state = TableState::default().with_selected(selected);
        f.render_stateful_widget(table, chunks[2], &mut state);
    }

    let hint = Line::from(vec![
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("fork  ", Theme::status()),
        Span::styled("esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("type ", Theme::status_accent()),
        Span::styled("filter", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::chat::ChatEntry;

    fn buffer_line(buffer: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buffer.area.width)
            .map(|x| buffer[(x, y)].symbol())
            .collect::<String>()
    }

    fn find_buffer_text(buffer: &ratatui::buffer::Buffer, needle: &str) -> Option<(u16, u16)> {
        for y in 0..buffer.area.height {
            let line = buffer_line(buffer, y);
            if let Some(byte_idx) = line.find(needle) {
                let x = line[..byte_idx].chars().count() as u16;
                return Some((x, y));
            }
        }
        None
    }

    #[test]
    fn fork_popup_table_styles_dim_boundary_cells() {
        let selected_bg = Theme::selected().bg.expect("selected style must define bg");
        let dim_style = Theme::status().add_modifier(ratatui::style::Modifier::DIM);
        let selected_dim_style = ratatui::style::Style::default()
            .fg(dim_style.fg.unwrap_or_else(Theme::dim))
            .bg(selected_bg)
            .add_modifier(ratatui::style::Modifier::DIM);

        assert_eq!(dim_style.fg, Theme::status().fg);
        assert!(
            dim_style
                .add_modifier
                .contains(ratatui::style::Modifier::DIM)
        );
        assert_eq!(selected_dim_style.fg, Theme::status().fg);
        assert_eq!(selected_dim_style.bg, Some(selected_bg));
        assert!(
            selected_dim_style
                .add_modifier
                .contains(ratatui::style::Modifier::DIM)
        );
    }

    #[test]
    fn draw_fork_popup_renders_turns_as_table() {
        let mut chat = ChatState::new();
        chat.messages = vec![
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

        let backend = ratatui::backend::TestBackend::new(120, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| draw_fork_turn_popup(f, ForkPopupInput { chat: &chat }))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let rendered: String = buffer.content().iter().map(|c| c.symbol()).collect();

        assert!(
            !rendered.contains("#")
                && !rendered.contains("user message")
                && !rendered.contains("agent message"),
            "table headers should not render: {rendered}"
        );
        assert!(
            rendered.contains(ELLIPSIS),
            "missing ellipsis column: {rendered}"
        );
        assert!(
            rendered.contains("alpha prompt"),
            "missing user text: {rendered}"
        );
        assert!(
            rendered.contains("alpha reply"),
            "missing agent text: {rendered}"
        );
        assert!(
            rendered.contains("beta prompt"),
            "missing user-only text: {rendered}"
        );
        let (_, beta_row) = find_buffer_text(&buffer, "beta prompt").expect("missing beta row");
        let (_, alpha_row) = find_buffer_text(&buffer, "alpha prompt").expect("missing alpha row");
        assert!(
            beta_row < alpha_row,
            "latest forkable turn should render before older turn: {rendered}"
        );
        let (_, hint_row) = find_buffer_text(&buffer, "enter fork").expect("missing hint row");
        assert!(
            hint_row > alpha_row + 1,
            "expected a blank row between the last table row and hint: {rendered}"
        );
        assert!(
            buffer_line(&buffer, hint_row - 1).trim().is_empty(),
            "row before hint should be empty: {rendered}"
        );
        assert!(
            !rendered.contains("asst") && !rendered.contains("user beta prompt"),
            "old boundary labels should not render: {rendered}"
        );
    }
}
