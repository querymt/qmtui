use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::session_state::SessionsState;
use crate::theme::Theme;
use crate::view_shared::{popup_rect, scroll_input};

pub(crate) fn draw_new_session_popup(f: &mut Frame, sessions: &SessionsState) {
    let area = f.area();
    let show_completion = sessions
        .new_session_completion
        .as_ref()
        .map(|completion| !completion.results.is_empty())
        .unwrap_or(false);
    let max_height = if show_completion { 10 } else { 6 };
    let popup_area = popup_rect(
        area,
        area.width.saturating_sub(4) as usize,
        area.height.saturating_sub(4).min(max_height) as usize,
        24..=72,
        4..=max_height as usize,
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
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Span::styled("new session", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );
    f.render_widget(
        Paragraph::new(Span::styled(
            "workspace path (empty = default cwd)",
            Theme::status(),
        ))
        .style(Theme::popup_bg()),
        chunks[1],
    );
    let avail = chunks[2].width.saturating_sub(2) as usize;
    let (path_display, path_cur) = scroll_input(
        &sessions.new_session_path,
        sessions.new_session_cursor,
        avail,
    );
    let input_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(path_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(input_line).style(Theme::popup_bg()),
        chunks[2],
    );
    if chunks[2].width > 2 && chunks[2].height > 0 {
        f.set_cursor_position((
            chunks[2]
                .x
                .saturating_add(2)
                .saturating_add(path_cur as u16),
            chunks[2].y,
        ));
    }

    if let Some(completion) = &sessions.new_session_completion
        && !completion.results.is_empty()
    {
        let items: Vec<ListItem> = completion
            .results
            .iter()
            .map(|entry| {
                ListItem::new(Line::from(vec![Span::styled(
                    entry.path.clone(),
                    Theme::input(),
                )]))
            })
            .collect();
        let list = List::new(items)
            .block(Block::default().style(Theme::popup_bg()))
            .highlight_style(Theme::selected())
            .highlight_symbol("");
        let selected = (!completion.results.is_empty()).then_some(completion.selected_index);
        let mut state = ListState::default().with_selected(selected);
        f.render_stateful_widget(list, chunks[3], &mut state);
    }

    let hint = Line::from(vec![
        Span::styled("tab ", Theme::status_accent()),
        Span::styled("complete  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("start  ", Theme::status()),
        Span::styled("esc ", Theme::status_accent()),
        Span::styled("cancel", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}
