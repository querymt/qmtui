use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::connection_state::ConnState;
use crate::theme::Theme;

const BRAILLE_SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const LINE_SPINNER: &[&str] = &["-", "\\", "|", "/"];
const DOTS_SPINNER: &[&str] = &[".  ", ".. ", "..."];

pub(crate) const CONN_ONLINE: &str = "\u{25CF}";
pub(crate) const CONN_OFFLINE: &str = "\u{25CB}";
const ICON_MESH: &str = "\u{1F5A7}";

pub(crate) const CHECK_CHECKED: &str = "\u{2611} ";
pub(crate) const CHECK_FAILED: &str = "\u{2612}";
pub(crate) const ELLIPSIS: &str = "\u{2026}";

pub(crate) fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Keep a single-line input cursor visible within the available columns.
pub(crate) fn scroll_input(text: &str, cursor_byte: usize, avail: usize) -> (String, usize) {
    let cursor_chars = text[..cursor_byte.min(text.len())].chars().count();
    scroll_input_chars(text, cursor_chars, avail)
}

pub(crate) fn scroll_input_chars(text: &str, cursor_chars: usize, avail: usize) -> (String, usize) {
    if avail == 0 {
        return (String::new(), 0);
    }
    let scroll = if cursor_chars >= avail {
        cursor_chars + 1 - avail
    } else {
        0
    };
    let visible = text.chars().skip(scroll).take(avail).collect();
    (visible, cursor_chars - scroll)
}

pub(crate) fn truncate_with_ellipsis(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let truncated: String = text.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{truncated}{ELLIPSIS}")
    } else {
        text.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpinnerKind {
    Braille,
    Line,
    Dots,
}

fn spinner_frames(kind: SpinnerKind) -> &'static [&'static str] {
    match kind {
        SpinnerKind::Braille => BRAILLE_SPINNER,
        SpinnerKind::Line => LINE_SPINNER,
        SpinnerKind::Dots => DOTS_SPINNER,
    }
}

pub(crate) fn spinner(kind: SpinnerKind, tick: u64) -> &'static str {
    let frames = spinner_frames(kind);
    frames[(tick as usize / 2) % frames.len()]
}

pub(crate) fn connection_indicator(connection: ConnState) -> Span<'static> {
    let (symbol, color) = match connection {
        ConnState::Connected => (CONN_ONLINE, Theme::ok()),
        ConnState::Connecting => (CONN_OFFLINE, Theme::warn()),
        ConnState::Disconnected => (CONN_ONLINE, Theme::err()),
    };
    Span::styled(
        format!("{symbol} "),
        Style::default().fg(color).bg(Theme::bg_dim()),
    )
}

pub(crate) fn mesh_header_span(node_count: Option<u32>) -> Option<Span<'static>> {
    let count = node_count.filter(|&count| count > 0)?;
    Some(Span::styled(
        format!(" {ICON_MESH} {count} "),
        Theme::status(),
    ))
}

pub(crate) fn draw_header(
    frame: &mut Frame,
    area: Rect,
    left: Vec<Span<'static>>,
    right: Vec<Span<'static>>,
    chord: bool,
    connection: ConnState,
) {
    let mut spans = vec![
        Span::styled(" query", Theme::title()),
        Span::styled("mt", Theme::title_accent()),
    ];
    if chord {
        spans.push(Span::styled(" C-x", Theme::status_accent()));
    }
    spans.extend(left);

    let connection = connection_indicator(connection);
    let left_len: usize = spans.iter().map(|span| span.content.chars().count()).sum();
    let right_len: usize = right.iter().map(|span| span.content.chars().count()).sum();
    let connection_len = connection.content.chars().count();
    let gap = (area.width as usize).saturating_sub(left_len + right_len + connection_len);
    spans.push(Span::styled(" ".repeat(gap), Theme::status()));
    spans.extend(right);
    spans.push(connection);

    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Theme::status()),
        area,
    );
}
