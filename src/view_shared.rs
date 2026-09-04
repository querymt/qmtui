use std::ops::RangeInclusive;

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
};
#[cfg(test)]
use ratatui::{buffer::Buffer, layout::Position};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::connection_state::ConnState;
use crate::theme::Theme;

const BRAILLE_SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
#[cfg(test)]
const LINE_SPINNER: &[&str] = &["-", "\\", "|", "/"];
#[cfg(test)]
const DOTS_SPINNER: &[&str] = &[".  ", ".. ", "..."];

pub(crate) const CONN_ONLINE: &str = "\u{25CF}";
pub(crate) const CONN_OFFLINE: &str = "\u{25CB}";
const ICON_MESH: &str = "\u{1F5A7}";

pub(crate) const CHECK_CHECKED: &str = "\u{2611} ";
pub(crate) const CHECK_FAILED: &str = "\u{2612}";
pub(crate) const ELLIPSIS: &str = "\u{2026}";

#[cfg(test)]
pub(crate) fn buffer_row(buffer: &Buffer, y: u16) -> String {
    assert!(
        y >= buffer.area.y && y < buffer.area.bottom(),
        "row {y} is outside buffer area {:?}",
        buffer.area
    );
    (buffer.area.x..buffer.area.right())
        .map(|x| buffer[(x, y)].symbol())
        .collect()
}

/// Find ASCII text by terminal cell, so Unicode elsewhere in the row cannot skew coordinates.
#[cfg(test)]
pub(crate) fn find_ascii_text(buffer: &Buffer, needle: &str) -> Option<Position> {
    assert!(!needle.is_empty() && needle.is_ascii());
    let symbols: Vec<String> = needle.chars().map(|ch| ch.to_string()).collect();
    let width = u16::try_from(symbols.len()).ok()?;
    if width > buffer.area.width {
        return None;
    }

    for y in buffer.area.y..buffer.area.bottom() {
        for x in buffer.area.x..=buffer.area.right().saturating_sub(width) {
            if symbols
                .iter()
                .enumerate()
                .all(|(offset, symbol)| buffer[(x + offset as u16, y)].symbol() == symbol)
            {
                return Some(Position::new(x, y));
            }
        }
    }
    None
}

#[cfg(test)]
pub(crate) fn find_ascii_text_rect(buffer: &Buffer, needle: &str) -> Option<Rect> {
    let position = find_ascii_text(buffer, needle)?;
    Some(Rect::new(
        position.x,
        position.y,
        u16::try_from(needle.len()).ok()?,
        1,
    ))
}

/// A rectangular buffer dump whose debug representation exposes its exact geometry.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BufferRegion {
    pub(crate) area: Rect,
    pub(crate) rows: Vec<String>,
}

/// Copy every cell in the intersection, including continuation cells and trailing blanks.
#[cfg(test)]
pub(crate) fn buffer_region(buffer: &Buffer, area: Rect) -> BufferRegion {
    let area = area.intersection(buffer.area);
    let rows = (area.y..area.bottom())
        .map(|y| {
            (area.x..area.right())
                .map(|x| buffer[(x, y)].symbol())
                .collect()
        })
        .collect();
    BufferRegion { area, rows }
}

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

/// Build a popup rectangle with wide calculations and a final frame-area clamp.
pub(crate) fn popup_rect(
    area: Rect,
    desired_width: usize,
    desired_height: usize,
    width_range: RangeInclusive<usize>,
    height_range: RangeInclusive<usize>,
    vertical_divisor: u16,
) -> Rect {
    fn dimension(desired: usize, range: RangeInclusive<usize>, available: u16) -> u16 {
        let min = *range.start();
        let max = (*range.end()).max(min);
        desired.max(min).min(max).min(available as usize) as u16
    }

    let width = dimension(desired_width, width_range, area.width);
    let height = dimension(desired_height, height_range, area.height);
    let rect = Rect::new(
        area.x.saturating_add(area.width.saturating_sub(width) / 2),
        area.y
            .saturating_add(area.height.saturating_sub(height) / vertical_divisor.max(1)),
        width,
        height,
    );
    rect.clamp(area)
}

/// Wrap text by terminal display columns without splitting UTF-8 characters.
pub(crate) fn wrap_display_width(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut line = String::new();
    let mut line_width = 0usize;
    for ch in text.chars() {
        if ch == '\n' {
            lines.push(std::mem::take(&mut line));
            line_width = 0;
            continue;
        }

        let ch_width = ch.width().unwrap_or(0).max(1);
        if !line.is_empty() && line_width.saturating_add(ch_width) > width {
            lines.push(std::mem::take(&mut line));
            line_width = 0;
        }
        line.push(ch);
        line_width = line_width.saturating_add(ch_width);
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
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
    #[cfg(test)]
    Line,
    #[cfg(test)]
    Dots,
}

fn spinner_frames(kind: SpinnerKind) -> &'static [&'static str] {
    match kind {
        SpinnerKind::Braille => BRAILLE_SPINNER,
        #[cfg(test)]
        SpinnerKind::Line => LINE_SPINNER,
        #[cfg(test)]
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
        ConnState::Disconnected => (CONN_OFFLINE, Theme::err()),
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
    let left_len: usize = spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum();
    let right_len: usize = right
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum();
    let connection_len = UnicodeWidthStr::width(connection.content.as_ref());
    let gap = (area.width as usize).saturating_sub(left_len + right_len + connection_len);
    spans.push(Span::styled(" ".repeat(gap), Theme::status()));
    spans.extend(right);
    spans.push(connection);

    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Theme::status()),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    #[test]
    fn buffer_helpers_handle_offset_and_wide_cells() {
        let mut buffer = Buffer::empty(Rect::new(5, 7, 8, 2));
        buffer.set_string(5, 7, "界abc", Style::default());

        assert_eq!(buffer[(6, 7)].symbol(), " ");
        assert_eq!(find_ascii_text(&buffer, "abc"), Some(Position::new(7, 7)));
        assert_eq!(
            find_ascii_text_rect(&buffer, "abc"),
            Some(Rect::new(7, 7, 3, 1))
        );
        assert_eq!(find_ascii_text(&buffer, "missing"), None);
        assert_eq!(find_ascii_text(&buffer, "123456789"), None);

        let region = buffer_region(&buffer, buffer.area);
        assert_eq!(region.area, buffer.area);
        assert_eq!(region.rows.len(), 2);
        assert_eq!(region.rows[0], format!("界 abc{}", " ".repeat(3)));
        assert_eq!(region.rows[1], " ".repeat(8));
    }

    #[test]
    fn buffer_region_clips_and_preserves_rectangular_geometry() {
        let mut buffer = Buffer::empty(Rect::new(5, 7, 6, 4));
        buffer.set_string(6, 8, "xy", Style::default());
        buffer.set_string(6, 9, "z", Style::default());

        let region = buffer_region(&buffer, Rect::new(6, 8, 4, 3));
        assert_eq!(region.area, Rect::new(6, 8, 4, 3));
        assert_eq!(
            region.rows,
            [
                format!("xy{}", " ".repeat(2)),
                format!("z{}", " ".repeat(3)),
                " ".repeat(4)
            ]
        );

        let outside = buffer_region(&buffer, Rect::new(0, 0, 2, 2));
        assert!(outside.area.is_empty());
        assert!(outside.rows.is_empty());
    }

    #[test]
    fn popup_rect_is_contained_for_zero_tiny_and_normal_areas() {
        for area in [
            Rect::new(0, 0, 0, 0),
            Rect::new(4, 7, 1, 1),
            Rect::new(3, 5, 8, 4),
            Rect::new(0, 0, 100, 40),
        ] {
            let popup = popup_rect(area, 80, 20, 24..=84, 6..=24, 2);
            assert!(popup.x >= area.x);
            assert!(popup.y >= area.y);
            assert!(popup.right() <= area.right());
            assert!(popup.bottom() <= area.bottom());
        }

        assert_eq!(
            popup_rect(Rect::new(0, 0, 100, 40), 80, 20, 24..=84, 6..=24, 2),
            Rect::new(10, 10, 80, 20)
        );
        assert_eq!(
            popup_rect(Rect::new(0, 0, 100, 40), 80, 20, 24..=84, 6..=24, 3),
            Rect::new(10, 6, 80, 20)
        );
    }

    #[test]
    fn wrap_display_width_handles_unicode_and_zero_width() {
        assert_eq!(wrap_display_width("ab界cd", 4), ["ab界", "cd"]);
        assert_eq!(wrap_display_width("e\u{301}x", 2), ["e\u{301}", "x"]);
        assert_eq!(wrap_display_width("界", 1), ["界"]);
        assert!(wrap_display_width("text", 0).is_empty());
    }

    fn rendered_right_x(left: &str) -> u16 {
        let backend = TestBackend::new(30, 1);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                draw_header(
                    frame,
                    frame.area(),
                    vec![Span::raw(left.to_string())],
                    vec![Span::raw("RIGHT")],
                    false,
                    ConnState::Connected,
                );
            })
            .unwrap();
        (0..30)
            .find(|&x| terminal.backend().buffer()[(x, 0)].symbol() == "R")
            .expect("right header span")
    }

    #[test]
    fn header_alignment_uses_display_width() {
        assert_eq!(rendered_right_x("ab"), rendered_right_x("界"));
        assert_eq!(rendered_right_x("ab"), rendered_right_x("😀"));
        assert_eq!(rendered_right_x("a"), rendered_right_x("e\u{301}"));
    }
}
