use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::navigation_state::NavigationState;
use crate::theme::Theme;
use crate::view_shared::{scroll_input, truncate_with_ellipsis};

pub(crate) fn draw_command_palette_popup(f: &mut Frame, navigation: &NavigationState) {
    const PALETTE_MIN_W: u16 = 48;
    const PALETTE_MAX_W: u16 = 82;
    const PALETTE_MIN_H: u16 = 14;
    const PALETTE_MAX_H: u16 = 22;
    const TITLE_COL_W: usize = 24;
    const SHORTCUT_COL_W: usize = 10;

    let area = f.area();
    let commands = navigation.filtered_command_palette_commands();
    let desired_h = (commands.len() as u16).saturating_add(7);
    let popup_width = area
        .width
        .saturating_sub(4)
        .clamp(PALETTE_MIN_W.min(area.width), PALETTE_MAX_W);
    let popup_height = desired_h
        .clamp(PALETTE_MIN_H, PALETTE_MAX_H)
        .min(area.height.saturating_sub(2).max(1));
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(popup_width) / 2,
        y: area.y + area.height.saturating_sub(popup_height) / 3,
        width: popup_width,
        height: popup_height,
    };

    f.render_widget(Clear, popup_area);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup_area);

    let inner = Rect {
        x: popup_area.x + 2,
        y: popup_area.y + 1,
        width: popup_area.width.saturating_sub(4),
        height: popup_area.height.saturating_sub(2),
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title
            Constraint::Length(1), // filter
            Constraint::Length(1), // spacer
            Constraint::Min(1),    // commands
            Constraint::Length(1), // spacer
            Constraint::Length(1), // hints
        ])
        .split(inner);

    let title = Line::from(vec![
        Span::styled("command palette", Theme::popup_title()),
        Span::styled(
            format!(
                "  {} command{}",
                commands.len(),
                if commands.len() == 1 { "" } else { "s" }
            ),
            Theme::status(),
        ),
    ]);
    f.render_widget(Paragraph::new(title).style(Theme::popup_bg()), chunks[0]);

    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (filter_display, filter_cur) = scroll_input(
        &navigation.command_palette_filter,
        navigation.command_palette_filter.len(),
        avail,
    );
    let placeholder = if navigation.command_palette_filter.is_empty() {
        "type to filter..."
    } else {
        ""
    };
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(filter_display, Theme::popup_bg()),
        Span::styled(placeholder, Theme::status()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        chunks[1],
    );
    let cursor_area = chunks[1].intersection(area);
    if chunks[1].width > 0
        && chunks[1].height > 0
        && cursor_area.width > 0
        && cursor_area.height > 0
    {
        let cursor_x = chunks[1]
            .x
            .saturating_add(2)
            .saturating_add(filter_cur as u16)
            .clamp(cursor_area.x, cursor_area.right().saturating_sub(1));
        let cursor_y = chunks[1]
            .y
            .clamp(cursor_area.y, cursor_area.bottom().saturating_sub(1));
        f.set_cursor_position((cursor_x, cursor_y));
    }

    let list_w = chunks[3].width as usize;
    let desc_avail = list_w.saturating_sub(3 + TITLE_COL_W + 1 + SHORTCUT_COL_W + 1);
    let items: Vec<ListItem> = if commands.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "  no commands match current filter",
            Theme::status(),
        )))]
    } else {
        commands
            .iter()
            .enumerate()
            .map(|(i, command)| {
                let selected = i == navigation.command_palette_cursor;
                let row_style = if selected {
                    Theme::selected()
                } else {
                    Theme::popup_bg()
                };
                let dim_style = if selected {
                    Theme::selected()
                } else {
                    Theme::status()
                };
                let marker = if selected { "▸" } else { " " };
                let title = truncate_with_ellipsis(command.title, TITLE_COL_W);
                let shortcut = truncate_with_ellipsis(command.shortcut, SHORTCUT_COL_W);
                let description = truncate_with_ellipsis(command.description, desc_avail);

                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {marker} "), row_style),
                    Span::styled(format!("{title:<TITLE_COL_W$}"), row_style),
                    Span::styled(" ", row_style),
                    Span::styled(format!("{shortcut:<SHORTCUT_COL_W$}"), dim_style),
                    Span::styled(" ", row_style),
                    Span::styled(description, dim_style),
                ]))
            })
            .collect()
    };

    let list = List::new(items)
        .block(Block::default().style(Theme::popup_bg()))
        .highlight_style(Theme::selected())
        .highlight_symbol("");
    let visible_rows = chunks[3].height as usize;
    let selected = (!commands.is_empty()).then_some(
        navigation
            .command_palette_cursor
            .min(commands.len().saturating_sub(1)),
    );
    let offset = selected
        .unwrap_or(0)
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(selected);
    f.render_stateful_widget(list, chunks[3], &mut state);

    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("close  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("open  ", Theme::status()),
        Span::styled("↑↓ ", Theme::status_accent()),
        Span::styled("navigate", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[5]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::navigation_state::Screen;

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
    fn command_palette_cursor_stays_inside_tiny_frame() {
        let navigation = NavigationState::new();
        let backend = ratatui::backend::TestBackend::new(1, 1);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| draw_command_palette_popup(frame, &navigation))
            .unwrap();

        let cursor = terminal.get_cursor_position().unwrap();
        assert!(Rect::new(0, 0, 1, 1).contains(cursor));
    }

    #[test]
    fn draw_command_palette_popup_aligns_columns_and_highlights_selection() {
        let mut navigation = NavigationState::new();
        navigation.screen = Screen::Chat;
        navigation.command_palette_filter = "session switcher".into();
        navigation.command_palette_cursor = 0;

        let backend = ratatui::backend::TestBackend::new(100, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| draw_command_palette_popup(f, &navigation))
            .unwrap();
        let buffer = terminal.backend().buffer().clone();

        let (title_x, row_y) =
            find_buffer_text(&buffer, "Session switcher").expect("selected command row missing");
        let (shortcut_x, shortcut_y) =
            find_buffer_text(&buffer, "C-x l").expect("selected shortcut missing");
        let (description_x, description_y) =
            find_buffer_text(&buffer, "Browse and load sessions").expect("description missing");

        assert_eq!(row_y, shortcut_y);
        assert_eq!(row_y, description_y);
        assert!(shortcut_x > title_x);
        assert!(description_x > shortcut_x);
        assert!(
            find_buffer_text(&buffer, "action").is_none(),
            "table headers should not render"
        );
        assert_eq!(buffer[(shortcut_x, row_y)].style().bg, Theme::selected().bg);
    }
}
