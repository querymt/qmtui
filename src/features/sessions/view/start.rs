use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

use crate::connection_state::ConnState;
use crate::render_state::RenderState;
use crate::session_state::{SessionsState, StartPageItem, session_group_count_text};
use crate::theme::Theme;
use crate::view_shared::{ELLIPSIS, draw_header, mesh_header_span};

// ── Start-page session list ────────────────────────────────────────────────────

pub(super) const COLLAPSE_OPEN: &str = "\u{25BE}"; // ▾ expanded group
pub(super) const COLLAPSE_CLOSED: &str = "\u{25B8}"; // ▸ collapsed group

pub(crate) struct StartScreenInput<'a> {
    pub(crate) sessions: &'a SessionsState,
    pub(crate) active_profile_label: &'a str,
    pub(crate) mesh_node_count: Option<u32>,
    pub(crate) chord: bool,
    pub(crate) connection: ConnState,
}

/// A single rendered row for the start-page session list.
///
/// Returned by [`build_start_page_rows`] and consumed by [`draw_start`].
pub(crate) struct StartPageRow {
    /// The logical item this row represents.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) item: StartPageItem,
    /// Pre-rendered line (spans already styled).
    pub(crate) line: Line<'static>,
    /// True when `session_cursor` points at this row.
    pub(crate) selected: bool,
}

/// Parse an ISO 8601 timestamp and return a human-readable relative time.
pub(super) fn relative_time(iso: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let Some(seconds) = iso.get(17..19).and_then(|value| value.parse::<u8>().ok()) else {
        return iso.to_string();
    };
    if seconds > 59 {
        return iso.to_string();
    }
    let Ok(timestamp) = chrono::DateTime::parse_from_rfc3339(iso) else {
        return iso.to_string();
    };
    let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return iso.to_string();
    };
    let Ok(now) = i64::try_from(now.as_secs()) else {
        return iso.to_string();
    };
    let Some(diff) = now.checked_sub(timestamp.timestamp()) else {
        return iso.to_string();
    };
    if diff < 0 {
        return "just now".into();
    }

    let secs = diff as u64;
    if secs < 60 {
        return "just now".into();
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("{mins}m ago");
    }
    let hours = mins / 60;
    if hours < 24 {
        return format!("{hours}h ago");
    }
    let days = hours / 24;
    if days < 7 {
        return format!("{days}d ago");
    }
    let weeks = days / 7;
    if weeks < 5 {
        return format!("{weeks}w ago");
    }
    let months = days / 30;
    if months < 12 {
        return format!("{months}mo ago");
    }
    let years = days / 365;
    format!("{years}y ago")
}

/// Build the displayable rows for the start-page session list.
///
/// Pure function — does not touch the frame. Accepts `area_width` for
/// truncating long paths/titles.
pub(crate) fn build_start_page_rows(
    sessions: &SessionsState,
    area_width: usize,
) -> Vec<StartPageRow> {
    let items = sessions.visible_start_items();
    let mut rows = Vec::with_capacity(items.len());

    for (idx, item) in items.iter().enumerate() {
        let selected = idx == sessions.session_cursor;

        let (header_style, dim_style) = if selected {
            (Theme::selected(), Theme::selected())
        } else {
            (Theme::start_header(), Theme::start_dim())
        };
        let session_style = if selected {
            Theme::selected()
        } else {
            Theme::start_session()
        };

        let line: Line<'static> = match &item {
            StartPageItem::GroupHeader {
                cwd,
                session_count,
                session_total,
                collapsed,
            } => {
                let indicator = if *collapsed {
                    COLLAPSE_CLOSED
                } else {
                    COLLAPSE_OPEN
                };
                let cwd_display = cwd.as_deref().unwrap_or("(no workspace)");
                // Shorten very long paths: keep last 3 components
                let cwd_short = short_cwd(cwd_display, area_width.saturating_sub(16));
                let count_text = session_group_count_text(*session_count, *session_total);
                Line::from(vec![
                    Span::styled(format!(" {indicator} "), header_style),
                    Span::styled(cwd_short, header_style),
                    Span::styled(format!("  ({count_text}) "), dim_style),
                ])
            }

            StartPageItem::Session {
                group_idx,
                path,
                depth,
            } => {
                let Some(session) = sessions.session_by_path(*group_idx, path) else {
                    continue;
                };
                let id_short: String = session.session_id.chars().take(8).collect();
                let title = session.title.as_deref().unwrap_or("(untitled)");
                let time_str = session
                    .updated_at
                    .as_deref()
                    .map(relative_time)
                    .unwrap_or_default();

                let fork_marker = if sessions.expandable_root_session(*group_idx, path) {
                    let indicator = if sessions
                        .expanded_session_children
                        .contains(&session.session_id)
                    {
                        COLLAPSE_OPEN
                    } else {
                        COLLAPSE_CLOSED
                    };
                    format!(" {indicator} ↳ {}", session.fork_count)
                } else if session.fork_count > 0 && *depth == 0 {
                    format!(" ↳ {}", session.fork_count)
                } else {
                    String::new()
                };

                let indent = "  ".repeat(*depth);
                let id_part = format!("   {indent}{id_short}  ");
                let time_part = format!("  {time_str} ");
                let overhead = id_part.chars().count()
                    + fork_marker.chars().count()
                    + time_part.chars().count();
                let avail = area_width.saturating_sub(overhead);
                let title_display: String = if title.chars().count() > avail && avail > 1 {
                    let t: String = title.chars().take(avail.saturating_sub(1)).collect();
                    format!("{t}{ELLIPSIS}")
                } else {
                    title.to_string()
                };
                let gap = avail.saturating_sub(title_display.chars().count());

                Line::from(vec![
                    Span::styled(id_part, dim_style),
                    Span::styled(title_display, session_style),
                    Span::styled(
                        fork_marker,
                        Theme::fork_count().bg(session_style.bg.unwrap_or(Theme::bg())),
                    ),
                    Span::styled(" ".repeat(gap), session_style),
                    Span::styled(time_part, dim_style),
                ])
            }

            // Per-group overflow row. The start page always opens the popup;
            // only the popup performs paginated load-more requests.
            StartPageItem::ShowMore { .. } => {
                let label = format!("   {ELLIPSIS} show all");
                Line::from(vec![Span::styled(label, dim_style)])
            }
        };

        rows.push(StartPageRow {
            item: item.clone(),
            line,
            selected,
        });
    }

    rows
}

/// Shorten a path to fit within `max_chars` by keeping the last N components.
pub(crate) fn short_cwd(path: &str, max_chars: usize) -> String {
    if path.chars().count() <= max_chars {
        return path.to_string();
    }
    // Walk backwards collecting components until we exceed max_chars
    let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    while !parts.is_empty() {
        let candidate = format!("\u{2026}/{}", parts.join("/"));
        if candidate.chars().count() <= max_chars {
            return candidate;
        }
        parts.remove(0);
    }
    // Absolute fallback: hard-truncate
    let t: String = path.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{t}{ELLIPSIS}")
}

pub(crate) fn draw_start(f: &mut Frame, input: StartScreenInput<'_>, render: &mut RenderState) {
    let area = f.area();

    // Minimum height to show the ASCII art: art (6 rows) + 2 padding = 8.
    // Below this threshold we skip the art and give all space to the list.
    const ART_MIN_HEIGHT: u16 = 16;
    const ART_ROWS: u16 = 6;

    let show_art = area.height >= ART_MIN_HEIGHT;
    let art_section_height = if show_art { ART_ROWS + 2 } else { 0 };

    // ── outer layout: header │ middle │ hints ─────────────────────────────────
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(1),    // middle (centred content)
            Constraint::Length(1), // hints
        ])
        .split(area);

    // ── header ────────────────────────────────────────────────────────────────
    let right = mesh_header_span(input.mesh_node_count)
        .into_iter()
        .collect();
    draw_header(
        f,
        outer[0],
        vec![Span::styled(
            format!(" profile:{} ", input.active_profile_label),
            Theme::status(),
        )],
        right,
        input.chord,
        input.connection,
    );

    let sessions = input.sessions;

    // ── hints ─────────────────────────────────────────────────────────────────
    // Rendered now (fixed bottom) so we can focus the rest on centring.
    let hint_area = outer[2];
    let hints = Line::from(vec![
        Span::styled(" \u{2191}\u{2193} ", Theme::status_accent()),
        Span::styled("navigate  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("open/collapse  ", Theme::status()),
        Span::styled("del ", Theme::status_accent()),
        Span::styled("delete  ", Theme::status()),
        Span::styled("C-x n ", Theme::status_accent()),
        Span::styled("new  ", Theme::status()),
        Span::styled("C-x s ", Theme::status_accent()),
        Span::styled("popup  ", Theme::status()),
        Span::styled("q ", Theme::status_accent()),
        Span::styled("quit", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hints).style(Theme::base()), hint_area);

    // ── glitch / wave variables (shared by art and button) ───────────────────
    const GLITCH_CHARS: &str = "░▒▓█▌▐▄▀┃╋╳";
    let tick = render.tick as usize;
    let prng = |seed: usize| -> usize {
        let mut h = seed.wrapping_mul(2654435761);
        h ^= h >> 16;
        h.wrapping_mul(0x45d9f3b) ^ (h >> 13)
    };
    let wave_colors = [
        Theme::accent(),
        Theme::info(),
        Theme::ok(),
        Theme::warn(),
        Theme::accent(),
    ];
    let glitch_frame = prng(tick / 3) % 7 == 0;

    // ── content column: fixed 64-char wide, centred under the ASCII art ───────
    const CONTENT_COL_W: u16 = 64;
    let col_w = CONTENT_COL_W.min(area.width);
    let col_x = area.x + area.width.saturating_sub(col_w) / 2;

    // ── compute content block height for vertical centring ────────────────────
    // Build rows first so we know how many there are.
    let rows = build_start_page_rows(sessions, col_w as usize);
    let rows_h = rows.len() as u16;

    // Button: 1 gap row + 1 text row (no border).
    const BUTTON_H: u16 = 2; // 1 gap + 1 button

    // Total content = art section + filter row + gap + session rows + button.
    // Cap rows_h to what fits so centring stays correct on short terminals.
    let middle = outer[1];
    let max_rows_h = middle
        .height
        .saturating_sub(art_section_height + 1 + 1 + BUTTON_H);
    let capped_rows_h = rows_h.min(max_rows_h);
    let content_h = (art_section_height + 1 + 1 + capped_rows_h + BUTTON_H).min(middle.height);

    // Centre the content block vertically inside `middle`.
    let top_pad = middle.height.saturating_sub(content_h) / 2;
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_pad),   // top spacer
            Constraint::Length(content_h), // content block
            Constraint::Min(0),            // bottom spacer
        ])
        .split(middle);

    let content_area = inner[1];

    // Sub-divide content_area into art | filter | gap | session rows | gap | button.
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(art_section_height), // [0] art
            Constraint::Length(1),                  // [1] filter
            Constraint::Length(1),                  // [2] gap after filter
            Constraint::Min(0),                     // [3] session rows
            Constraint::Length(1),                  // [4] gap before button
            Constraint::Length(1),                  // [5] button
        ])
        .split(content_area);

    // ── ASCII art ─────────────────────────────────────────────────────────────
    if show_art {
        let art_area = sections[0];
        let art = [
            r"  ___                        __  __ _____ ",
            r" / _ \ _   _  ___ _ __ _   _|  \/  |_   _|",
            r"| | | | | | |/ _ \ '__| | | | |\/| | | |  ",
            r"| |_| | |_| |  __/ |  | |_| | |  | | | |  ",
            r" \__\_\\__,_|\___|_|   \__, |_|  |_| |_|  ",
            r"                       |___/               ",
        ];
        let art_w = art.iter().map(|l| l.len()).max().unwrap_or(0) as u16;
        // Centre vertically within the art section (1 row padding top)
        let start_y = art_area.y + 1;
        let start_x = art_area.x + art_area.width.saturating_sub(art_w) / 2;

        let glitch_row = if glitch_frame {
            Some(prng(tick / 3 + 99) % art.len())
        } else {
            None
        };

        for (row, line) in art.iter().enumerate() {
            if start_y + row as u16 >= art_area.y + art_area.height {
                break;
            }
            let x_offset = if glitch_row == Some(row) {
                (prng(tick + row * 7) % 5) as i16 - 2
            } else {
                0
            };
            let row_x = (start_x as i16 + x_offset).max(art_area.x as i16) as u16;

            let spans: Vec<Span<'static>> = line
                .chars()
                .enumerate()
                .map(|(col, ch)| {
                    if ch == ' ' {
                        return Span::raw(" ");
                    }
                    let char_glitch =
                        glitch_frame && prng(tick.wrapping_mul(col + 1) + row * 31) % 12 == 0;
                    let display_ch = if char_glitch {
                        let idx = prng(tick + col * 13 + row * 7) % GLITCH_CHARS.chars().count();
                        GLITCH_CHARS.chars().nth(idx).unwrap_or(ch)
                    } else {
                        ch
                    };
                    let color = if char_glitch {
                        Theme::err()
                    } else {
                        let phase = (col + row * 3).wrapping_add(tick / 3);
                        wave_colors[phase % wave_colors.len()]
                    };
                    Span::styled(
                        display_ch.to_string(),
                        ratatui::style::Style::default()
                            .fg(color)
                            .add_modifier(ratatui::style::Modifier::BOLD),
                    )
                })
                .collect();
            f.render_widget(
                Paragraph::new(Line::from(spans)),
                Rect {
                    x: row_x,
                    y: start_y + row as u16,
                    width: art_w.min(art_area.width),
                    height: 1,
                },
            );
        }
    }

    // ── filter input ──────────────────────────────────────────────────────────
    let filter_area = Rect {
        x: col_x,
        y: sections[1].y,
        width: col_w,
        height: 1,
    };
    let filter_line = Line::from(vec![
        Span::styled(" > ", Theme::start_header()),
        Span::styled(sessions.session_filter.clone(), Theme::fg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::base()),
        filter_area,
    );
    // Show cursor at the end of the typed filter text
    f.set_cursor_position((
        filter_area.x + 3 + sessions.session_filter.chars().count() as u16,
        filter_area.y,
    ));

    // ── session list ──────────────────────────────────────────────────────────
    let list_area = sections[3];

    if rows.is_empty() {
        // Empty state
        let msg = if sessions.session_filter.is_empty() {
            "No sessions yet.  C-x n to start a new one."
        } else {
            "No sessions match the filter."
        };
        let msg_w = msg.len() as u16;
        let msg_x = col_x + col_w.saturating_sub(msg_w) / 2;
        let msg_y = list_area.y + list_area.height / 2;
        f.render_widget(
            Paragraph::new(Span::styled(msg, Theme::status())),
            Rect {
                x: msg_x,
                y: msg_y,
                width: msg_w.min(col_w),
                height: 1,
            },
        );
    } else {
        // Scroll: keep the selected row visible.
        let visible_rows = list_area.height as usize;
        let total_rows = rows.len();

        let scroll =
            render.clamp_start_page_scroll(sessions.session_cursor, total_rows, visible_rows);

        for (display_row, row) in rows.iter().skip(scroll).take(visible_rows).enumerate() {
            let y = list_area.y + display_row as u16;
            if y >= list_area.y + list_area.height {
                break;
            }
            let row_area = Rect {
                x: col_x,
                y,
                width: col_w,
                height: 1,
            };

            // Fill background for the entire row first
            let row_bg = if row.selected {
                Theme::selected()
            } else {
                Theme::base()
            };
            f.render_widget(Block::default().style(row_bg), row_area);
            f.render_widget(Paragraph::new(row.line.clone()), row_area);
        }
    }

    // ── New Session button ────────────────────────────────────────────────────
    const BUTTON_TEXT: &str = "+ New Session";
    let button_w = (BUTTON_TEXT.len() as u16).min(col_w);
    let button_x = col_x + col_w.saturating_sub(button_w) / 2;
    let button_rect = Rect {
        x: button_x,
        y: sections[5].y,
        width: button_w,
        height: 1,
    };

    let button_focused = sessions.session_cursor == rows.len();

    // Build text spans: glitch only when focused, always bold
    let text_spans: Vec<Span<'static>> = if button_focused {
        BUTTON_TEXT
            .chars()
            .enumerate()
            .map(|(col, ch)| {
                if ch == ' ' {
                    return Span::raw(" ");
                }
                let char_glitch = glitch_frame && prng(tick.wrapping_mul(col + 1) + 999) % 8 == 0;
                let display_ch = if char_glitch {
                    let idx = prng(tick + col * 17 + 333) % GLITCH_CHARS.chars().count();
                    GLITCH_CHARS.chars().nth(idx).unwrap_or(ch)
                } else {
                    ch
                };
                let color = if char_glitch {
                    Theme::err()
                } else {
                    let phase = (col + tick / 3) % wave_colors.len();
                    wave_colors[phase]
                };
                Span::styled(
                    display_ch.to_string(),
                    ratatui::style::Style::default()
                        .fg(color)
                        .add_modifier(ratatui::style::Modifier::BOLD),
                )
            })
            .collect()
    } else {
        vec![Span::styled(
            BUTTON_TEXT,
            Theme::start_header().add_modifier(ratatui::style::Modifier::BOLD),
        )]
    };

    f.render_widget(
        Paragraph::new(Line::from(text_spans)).alignment(Alignment::Center),
        button_rect,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::session::{SessionGroup, SessionSummary};
    use crate::view_shared::{buffer_region, buffer_row, find_ascii_text_rect};

    fn group(cwd: &str, count: usize) -> SessionGroup {
        SessionGroup {
            cwd: Some(cwd.into()),
            sessions: (0..count)
                .map(|index| SessionSummary {
                    session_id: format!("{cwd}-{index}"),
                    title: Some(format!("Session {index}")),
                    ..Default::default()
                })
                .collect(),
            total_count: Some(count as u64),
            ..Default::default()
        }
    }

    fn session_semantics(sessions: &SessionsState) -> String {
        format!(
            "groups={:?};cursor={};filter={:?};tab={};collapsed={:?};popup_collapsed={:?};discovery={};discovery_cursors={:?};pending_groups={:?};hydrated={:?};expanded={:?};pending_children={:?};session={:?};agent={:?};mode={:?};review={:?};new_path={:?};new_cursor={};completion={:?};activity={:?};remote={:?}",
            sessions.session_groups,
            sessions.session_cursor,
            sessions.session_filter,
            sessions.session_popup_tab,
            sessions.collapsed_groups,
            sessions.popup_collapsed_groups,
            sessions.session_discovery_in_progress,
            sessions.session_discovery_cursors,
            sessions.pending_session_group_loads,
            sessions.hydrated_session_groups,
            sessions.expanded_session_children,
            sessions.pending_session_child_loads,
            sessions.session_id,
            sessions.agent_id,
            sessions.agent_mode,
            sessions.mode_before_review,
            sessions.new_session_path,
            sessions.new_session_cursor,
            sessions.new_session_completion,
            sessions.session_activity,
            sessions.remote_session_locations,
        )
    }

    fn draw_at(
        sessions: &SessionsState,
        render: &mut RenderState,
        active_profile_label: &str,
        width: u16,
        height: u16,
    ) -> ratatui::buffer::Buffer {
        let backend = ratatui::backend::TestBackend::new(width, height);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                draw_start(
                    frame,
                    StartScreenInput {
                        sessions,
                        active_profile_label,
                        mesh_node_count: None,
                        chord: false,
                        connection: ConnState::Connecting,
                    },
                    render,
                )
            })
            .unwrap();
        terminal.backend().buffer().clone()
    }

    #[test]
    fn start_draw_clamps_resize_button_and_short_lists_without_mutating_sessions() {
        let mut sessions = SessionsState::new();
        let mut render = RenderState::new();
        sessions.session_groups = vec![group("repo-a", 4), group("repo-b", 4), group("repo-c", 4)];
        let row_count = build_start_page_rows(&sessions, 64).len();
        assert_eq!(row_count, 15);
        sessions.session_cursor = row_count;
        render.test_seed_start_page_scroll(usize::MAX);
        let before = session_semantics(&sessions);

        draw_at(&sessions, &mut render, "default", 100, 10);
        assert_eq!(render.start_page_scroll(), 11);
        assert_eq!(session_semantics(&sessions), before);

        draw_at(&sessions, &mut render, "default", 100, 14);
        assert_eq!(render.start_page_scroll(), 7);
        assert_eq!(session_semantics(&sessions), before);

        draw_at(&sessions, &mut render, "default", 100, 30);
        assert_eq!(render.start_page_scroll(), 0);
        assert_eq!(session_semantics(&sessions), before);

        sessions.session_groups = vec![group("short", 1)];
        sessions.session_cursor = 1;
        render.test_seed_start_page_scroll(9);
        let short_before = session_semantics(&sessions);
        draw_at(&sessions, &mut render, "default", 80, 20);
        assert_eq!(render.start_page_scroll(), 0);
        assert_eq!(session_semantics(&sessions), short_before);
    }

    #[test]
    fn start_draw_preserves_empty_scroll_and_handles_zero_height_list() {
        let mut sessions = SessionsState::new();
        let mut render = RenderState::new();
        render.test_seed_start_page_scroll(7);
        let before = session_semantics(&sessions);
        let buffer = draw_at(&sessions, &mut render, "default", 80, 20);
        let empty_message =
            find_ascii_text_rect(&buffer, "No sessions yet.  C-x n to start a new one.")
                .expect("empty-state message");
        assert_eq!(empty_message.y, 14);
        assert_eq!(
            buffer_region(&buffer, empty_message).rows,
            ["No sessions yet.  C-x n to start a new one."]
        );
        assert_eq!(render.start_page_scroll(), 7);
        assert_eq!(session_semantics(&sessions), before);

        sessions.session_groups = vec![group("tiny", 4)];
        sessions.session_cursor = 2;
        render.test_seed_start_page_scroll(0);
        let tiny_before = session_semantics(&sessions);
        draw_at(&sessions, &mut render, "default", 20, 3);
        assert_eq!(render.start_page_scroll(), 2);
        render.test_seed_start_page_scroll(0);
        draw_at(&sessions, &mut render, "default", 20, 2);
        assert_eq!(render.start_page_scroll(), 2);
        assert_eq!(session_semantics(&sessions), tiny_before);
    }

    #[test]
    fn start_page_header_uses_active_profile_label() {
        let sessions = SessionsState::new();
        let mut render = RenderState::new();
        let buffer = draw_at(&sessions, &mut render, "Fast", 100, 20);
        let profile = find_ascii_text_rect(&buffer, "profile:Fast").expect("profile label");
        assert_eq!(profile, Rect::new(9, 0, 12, 1));
        assert_eq!(&buffer_row(&buffer, 0)[8..22], " profile:Fast ");
    }

    #[test]
    fn start_page_empty_render_snapshot() {
        let sessions = SessionsState::new();
        let mut render = RenderState::new();
        render.replace_tick(3);

        let buffer = draw_at(&sessions, &mut render, "default", 80, 24);

        insta::assert_debug_snapshot!(
            "start_page_empty_80x24",
            buffer_region(&buffer, buffer.area)
        );
    }

    #[test]
    fn start_page_populated_render_snapshot() {
        let mut sessions = SessionsState::new();
        let mut querymt = group("/work/querymt", 2);
        querymt.sessions[0].session_id = "qmt-001".into();
        querymt.sessions[1].session_id = "qmt-002".into();
        let mut qmtui = group("/work/qmtui", 1);
        qmtui.sessions[0].session_id = "ui-001".into();
        sessions.session_groups = vec![querymt, qmtui];
        sessions.session_cursor = 1;
        let mut render = RenderState::new();
        render.replace_tick(3);

        let buffer = draw_at(&sessions, &mut render, "work", 80, 24);

        let selected = find_ascii_text_rect(&buffer, "qmt-001").expect("selected session");
        assert_eq!(selected.y, 14);
        assert_eq!(
            buffer[(selected.x, selected.y)].style().bg,
            Theme::selected().bg
        );
        insta::assert_debug_snapshot!(
            "start_page_populated_80x24",
            buffer_region(&buffer, buffer.area)
        );
    }

    #[test]
    fn relative_time_accepts_rfc3339_offsets_and_fractions() {
        for timestamp in [
            "2020-01-02T03:04:05Z",
            "2020-01-02T03:04:05.123456789Z",
            "2020-01-02T03:04:05+02:30",
        ] {
            assert_ne!(relative_time(timestamp), timestamp);
        }
    }

    #[test]
    fn relative_time_rejects_invalid_and_pathological_timestamps() {
        for timestamp in [
            "not-a-timestamp",
            "2020-13-02T03:04:05Z",
            "2021-02-29T03:04:05Z",
            "2020-01-02T24:04:05Z",
            "2020-01-02T03:60:05Z",
            "2020-01-02T03:04:60Z",
            "999999999999999999-01-01T00:00:00Z",
            "2020-01-02T03:04:05+99:00",
        ] {
            assert_eq!(relative_time(timestamp), timestamp);
        }
    }

    #[test]
    fn relative_time_treats_future_timestamps_as_just_now() {
        assert_eq!(relative_time("2999-01-01T00:00:00Z"), "just now");
    }
}
