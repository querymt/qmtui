use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::diagnostics::{DiagnosticsState, LogLevel};
use crate::theme::Theme;
use crate::view_shared::{ELLIPSIS, centered_rect, scroll_input};

pub(crate) struct LogPopupInput<'a> {
    pub(crate) diagnostics: &'a DiagnosticsState,
}

fn popup_log_level_style(level: LogLevel) -> ratatui::style::Style {
    match level {
        LogLevel::Trace => Theme::status(),
        LogLevel::Debug => Theme::status_accent(),
        LogLevel::Info => ratatui::style::Style::default()
            .fg(Theme::info())
            .bg(Theme::bg_dim()),
        LogLevel::Warn => ratatui::style::Style::default()
            .fg(Theme::warn())
            .bg(Theme::bg_dim()),
        LogLevel::Error => ratatui::style::Style::default()
            .fg(Theme::err())
            .bg(Theme::bg_dim()),
    }
}

pub(crate) fn draw_log_popup(f: &mut Frame, input: LogPopupInput<'_>) {
    let diagnostics = input.diagnostics;
    let area = f.area();
    let popup_area = centered_rect(80, 70, area);

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
            Constraint::Length(1), // title
            Constraint::Length(1), // filter
            Constraint::Length(1), // level
            Constraint::Min(1),    // list
            Constraint::Length(1), // hint
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Span::styled("logs", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );

    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (log_filter_display, log_filter_cur) =
        scroll_input(&diagnostics.log_filter, diagnostics.log_filter.len(), avail);
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(log_filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        chunks[1],
    );
    f.set_cursor_position((chunks[1].x + 2 + log_filter_cur as u16, chunks[1].y));

    let level_line = Line::from(vec![
        Span::styled("level: ", Theme::status()),
        Span::styled(
            format!("{}+", diagnostics.log_level_filter.label()),
            popup_log_level_style(diagnostics.log_level_filter),
        ),
    ]);
    f.render_widget(
        Paragraph::new(level_line).style(Theme::popup_bg()),
        chunks[2],
    );

    let filtered = diagnostics.filtered_logs();
    let list_w = chunks[3].width as usize;
    let items: Vec<ListItem> = if filtered.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            " no log entries match current filter",
            Theme::status(),
        )))]
    } else {
        filtered
            .iter()
            .map(|entry| {
                let time_part = format!(
                    " {:>6}.{:01} ",
                    entry.elapsed.as_secs(),
                    entry.elapsed.subsec_millis() / 100,
                );
                let level_part = format!("{:<5}", entry.level.label());
                let target_part = format!(" {:<10} ", entry.target);
                let prefix_w = time_part.chars().count()
                    + level_part.chars().count()
                    + target_part.chars().count();
                let avail = list_w.saturating_sub(prefix_w);
                let message = if entry.message.chars().count() > avail {
                    let truncated: String = entry
                        .message
                        .chars()
                        .take(avail.saturating_sub(1))
                        .collect();
                    format!("{truncated}{ELLIPSIS}")
                } else {
                    entry.message.clone()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(time_part, Theme::status()),
                    Span::styled(level_part, popup_log_level_style(entry.level)),
                    Span::styled(target_part, Theme::status()),
                    Span::styled(message, Theme::popup_bg()),
                ]))
            })
            .collect()
    };

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let selected = (!filtered.is_empty())
        .then_some(diagnostics.log_cursor.min(filtered.len().saturating_sub(1)));
    let mut state = ListState::default().with_selected(selected);
    f.render_stateful_widget(list, chunks[3], &mut state);

    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("close  ", Theme::status()),
        Span::styled("tab ", Theme::status_accent()),
        Span::styled("level", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_log_popup_shows_filter_level_and_entries() {
        let mut diagnostics = DiagnosticsState::new();
        diagnostics.log_filter = "server".into();
        diagnostics.log_level_filter = LogLevel::Info;
        diagnostics.push_log(LogLevel::Info, "server", "starting local server");
        diagnostics.push_log(LogLevel::Error, "server", "start failed");
        diagnostics.log_cursor = diagnostics.filtered_logs().len().saturating_sub(1);

        let backend = ratatui::backend::TestBackend::new(100, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                draw_log_popup(
                    f,
                    LogPopupInput {
                        diagnostics: &diagnostics,
                    },
                )
            })
            .unwrap();
        let buffer = terminal.backend().buffer().clone();
        let rendered = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(rendered.contains("logs"));
        assert!(rendered.contains("level: INFO+"));
        assert!(rendered.contains("starting local server"));
        assert!(rendered.contains("start failed"));
        assert!(rendered.contains("server"));
        assert!(rendered.contains("esc close"));
        assert!(rendered.contains("tab level"));
    }
}
