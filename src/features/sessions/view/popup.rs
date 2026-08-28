use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::delegates_state::DelegatesState;
use crate::features::delegates::view::draw_tab as draw_delegate_tab;
use crate::render_state::RenderState;
use crate::session_state::{SessionsState, session_group_count_text};
use crate::theme::Theme;
use crate::view_shared::{ELLIPSIS, scroll_input};

use super::start::{COLLAPSE_CLOSED, COLLAPSE_OPEN, relative_time, short_cwd};

// ── Session popup ─────────────────────────────────────────────────────────────

pub(crate) fn draw_session_popup(
    f: &mut Frame,
    sessions: &SessionsState,
    delegates: &DelegatesState,
    render: &mut RenderState,
) {
    const SESSION_POPUP_MAX_W: u16 = 86;
    const SESSION_POPUP_MIN_W: u16 = 36;

    let area = f.area();
    let popup_width = area
        .width
        .saturating_sub(4)
        .clamp(SESSION_POPUP_MIN_W, SESSION_POPUP_MAX_W);
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(popup_width) / 2,
        y: area.y + area.height.saturating_sub(area.height * 60 / 100) / 2,
        width: popup_width,
        height: area.height * 60 / 100,
    };

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
            Constraint::Length(1), // title / tab bar
            Constraint::Length(1), // filter
            Constraint::Length(1), // spacer
            Constraint::Min(1),    // list
            Constraint::Length(1), // hint
        ])
        .split(inner);

    // Tab bar
    let tab_labels = ["sessions", "delegates"];
    let mut tab_spans = Vec::new();
    for (i, label) in tab_labels.iter().enumerate() {
        let is_active = i == sessions.session_popup_tab;
        let style = if is_active {
            Theme::popup_title().add_modifier(Modifier::UNDERLINED)
        } else {
            Theme::status()
        };
        if i > 0 {
            tab_spans.push(Span::styled(" \u{2502} ", Theme::status()));
        }
        tab_spans.push(Span::styled(format!(" {label} "), style));
    }
    f.render_widget(
        Paragraph::new(Line::from(tab_spans)).style(Theme::popup_bg()),
        chunks[0],
    );

    if sessions.session_popup_tab == 0 {
        draw_session_tab_content(f, sessions, delegates, render, &chunks);
    } else {
        draw_delegate_tab(f, sessions, delegates, render, &chunks);
    }
}

fn draw_session_tab_content(
    f: &mut Frame,
    sessions: &SessionsState,
    delegates: &DelegatesState,
    render: &mut RenderState,
    chunks: &std::rc::Rc<[Rect]>,
) {
    use crate::session_state::PopupItem;

    // filter
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (session_filter_display, session_filter_cur) = scroll_input(
        &sessions.session_filter,
        sessions.session_filter.len(),
        avail,
    );
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(session_filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        chunks[1],
    );
    f.set_cursor_position((chunks[1].x + 2 + session_filter_cur as u16, chunks[1].y));

    // grouped session list
    let popup_items = sessions.visible_popup_items();
    let list_w = chunks[3].width as usize;
    let visible_rows = chunks[3].height as usize;
    render.publish_session_popup_visible_rows(visible_rows);

    let items: Vec<ListItem> = popup_items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let selected = i == sessions.session_cursor;
            match item {
                PopupItem::GroupHeader {
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
                    let cwd_short = short_cwd(cwd_display, list_w.saturating_sub(16));
                    let (header_style, dim_style) = if selected {
                        (Theme::selected(), Theme::selected())
                    } else {
                        (Theme::status_accent(), Theme::status())
                    };
                    let count_text = session_group_count_text(*session_count, *session_total);
                    ListItem::new(Line::from(vec![
                        Span::styled(format!(" {indicator} "), header_style),
                        Span::styled(cwd_short, header_style),
                        Span::styled(format!("  ({count_text}) "), dim_style),
                    ]))
                }
                PopupItem::Session {
                    group_idx,
                    path,
                    depth,
                } => {
                    let Some(s) = sessions.session_by_path(*group_idx, path) else {
                        return ListItem::new(Line::from(""));
                    };
                    let id_short: String = s.session_id.chars().take(8).collect();
                    let time_str = s
                        .updated_at
                        .as_deref()
                        .map(relative_time)
                        .unwrap_or_default();
                    let title = s.title.as_deref().unwrap_or("(untitled)");

                    let is_active = sessions.session_id.as_deref() == Some(s.session_id.as_str());
                    let is_parent =
                        delegates.parent_session_id.as_deref() == Some(s.session_id.as_str());
                    let marker_part = if is_active {
                        " ● "
                    } else if is_parent {
                        " \u{2b11} "
                    } else {
                        "   "
                    };
                    let indent = "  ".repeat(*depth);
                    let id_part = format!(" {indent}{id_short} ");
                    let fork_marker = if sessions.expandable_root_session(*group_idx, path) {
                        let indicator =
                            if sessions.expanded_session_children.contains(&s.session_id) {
                                COLLAPSE_OPEN
                            } else {
                                COLLAPSE_CLOSED
                            };
                        format!(" {indicator} ↳ {}", s.fork_count)
                    } else if s.fork_count > 0 && *depth == 0 {
                        format!(" ↳ {}", s.fork_count)
                    } else {
                        String::new()
                    };
                    let time_part = format!(" {time_str:>7} ");
                    let avail = list_w.saturating_sub(
                        marker_part.chars().count()
                            + id_part.chars().count()
                            + fork_marker.chars().count()
                            + time_part.chars().count(),
                    );
                    let title_display = if title.chars().count() > avail {
                        let t: String = title.chars().take(avail.saturating_sub(1)).collect();
                        format!("{t}{ELLIPSIS}")
                    } else {
                        title.to_string()
                    };
                    let title_gap = avail.saturating_sub(title_display.chars().count());

                    let (main_style, dim_style, time_style, row_bg) = if selected {
                        (
                            Theme::selected(),
                            Theme::selected(),
                            Theme::selected(),
                            Theme::bg_hl(),
                        )
                    } else {
                        (
                            Theme::popup_bg(),
                            Theme::status(),
                            Theme::session_time(),
                            Theme::bg_dim(),
                        )
                    };
                    let active_style = Theme::status_accent().bg(row_bg);
                    let highlight = is_active || is_parent;
                    let marker_style = if highlight { active_style } else { dim_style };
                    let id_style = if highlight { active_style } else { dim_style };
                    let fork_style = Theme::fork_count().bg(row_bg);

                    let mut spans = vec![
                        Span::styled(marker_part, marker_style),
                        Span::styled(id_part, id_style),
                        Span::styled(title_display, main_style),
                        Span::styled(fork_marker, fork_style),
                        Span::styled(" ".repeat(title_gap), dim_style),
                    ];
                    spans.push(Span::styled(time_part, time_style));

                    ListItem::new(Line::from(spans))
                }
                PopupItem::LoadMore { parent_path, .. } => {
                    let style = if selected {
                        Theme::selected()
                    } else {
                        Theme::status()
                    };
                    let label = if parent_path.is_empty() {
                        format!("     {ELLIPSIS} load more")
                    } else {
                        format!("       {ELLIPSIS} load more forks")
                    };
                    ListItem::new(Line::from(vec![Span::styled(label, style)]))
                }
            }
        })
        .collect();

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let offset = sessions
        .session_cursor
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(Some(sessions.session_cursor));
    f.render_stateful_widget(list, chunks[3], &mut state);

    // hint
    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("load  ", Theme::status()),
        Span::styled("del ", Theme::status_accent()),
        Span::styled("delete  ", Theme::status()),
        Span::styled("tab ", Theme::status_accent()),
        Span::styled("switch", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}
