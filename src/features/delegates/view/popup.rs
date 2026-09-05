use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    text::{Line, Span},
    widgets::{Block, Cell, List, ListItem, ListState, Paragraph, Row, Table, TableState},
};
use unicode_width::UnicodeWidthStr;

use crate::delegates_state::DelegatesState;
use crate::domain::activity::DelegateStats;
use crate::render_state::RenderState;
use crate::session_state::SessionsState;
use crate::theme::Theme;
use crate::view_shared::{
    CHECK_CHECKED, CHECK_FAILED, SpinnerKind, scroll_input, spinner, truncate_with_ellipsis,
};

// ── Delegate session popup ─────────────────────────────────────────────────────

const DELEGATE_STATUS_COL_W: usize = 1;
const DELEGATE_ICON_TOOLS: &str = "\u{2692}"; // ⚒
const DELEGATE_ICON_MSG: &str = "\u{1F5E9}"; // 🗩
const DELEGATE_ICON_CONTEXT: &str = "\u{1F5AA}"; // 🖪

struct DelegateRowData {
    status_badge: String,
    badge_style: ratatui::style::Style,
    agent: String,
    objective_source: String,
    tools: String,
    msgs: String,
    ctx: String,
    cost: String,
    duration: String,
    pending_label: String,
    has_pending_input: bool,
    is_current: bool,
}

fn format_delegate_tools(stats: &DelegateStats) -> String {
    if stats.tool_calls > 0 {
        format!("{DELEGATE_ICON_TOOLS}{}", stats.tool_calls)
    } else {
        String::new()
    }
}

fn format_delegate_messages(stats: &DelegateStats) -> String {
    if stats.messages > 0 {
        format!("{DELEGATE_ICON_MSG}{}", stats.messages)
    } else {
        String::new()
    }
}

fn format_delegate_context(stats: &DelegateStats) -> String {
    if let Some(pct) = stats.context_pct() {
        format!("{DELEGATE_ICON_CONTEXT}{pct}%")
    } else if stats.context_tokens > 0 {
        let abbrev = if stats.context_tokens >= 1_000 {
            format!("{}k", stats.context_tokens / 1_000)
        } else {
            stats.context_tokens.to_string()
        };
        format!("{DELEGATE_ICON_CONTEXT}{abbrev}")
    } else {
        String::new()
    }
}

fn format_delegate_cost(stats: &DelegateStats) -> String {
    if stats.cost_usd > 0.0 {
        format!("${:.2}", stats.cost_usd)
    } else {
        String::new()
    }
}

fn delegate_display_width(text: &str) -> u16 {
    UnicodeWidthStr::width(text) as u16
}

pub(crate) fn draw_tab(
    f: &mut Frame,
    sessions: &SessionsState,
    delegates: &DelegatesState,
    render: &mut RenderState,
    chunks: &std::rc::Rc<[Rect]>,
) {
    use crate::domain::activity::DelegateStatus;

    // filter
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (filter_display, filter_cur) = scroll_input(
        &delegates.delegate_filter,
        delegates.delegate_filter.len(),
        avail,
    );
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        chunks[1],
    );
    let cursor_area = chunks[1].intersection(f.area());
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

    // delegate entry list (built from event stream)
    let visible_rows = chunks[3].height as usize;
    render.publish_delegate_popup_visible_rows(visible_rows);
    let entries = delegates.visible_entries();

    if entries.is_empty() {
        let list = List::new(vec![ListItem::new(Line::from(Span::styled(
            " no delegations",
            Theme::status(),
        )))])
        .block(Block::default().style(Theme::popup_bg()));
        let mut state = ListState::default();
        f.render_stateful_widget(list, chunks[3], &mut state);
    } else {
        let rows_data: Vec<DelegateRowData> = entries
            .iter()
            .map(|entry| {
                let status_badge = match entry.status {
                    DelegateStatus::InProgress => {
                        spinner(SpinnerKind::Braille, render.tick).to_string()
                    }
                    DelegateStatus::Completed => CHECK_CHECKED.to_string(),
                    DelegateStatus::Failed => CHECK_FAILED.to_string(),
                    DelegateStatus::Cancelled => "\u{2298}".to_string(), // ⊘
                };
                let badge_style = match entry.status {
                    DelegateStatus::InProgress => Theme::status_accent(),
                    DelegateStatus::Completed => Theme::session_time(),
                    DelegateStatus::Failed => Theme::error_on_dim(),
                    DelegateStatus::Cancelled => Theme::status(),
                };
                let objective_source = if entry.objective.is_empty() {
                    "(no objective)".to_string()
                } else {
                    entry.objective.clone()
                };

                let is_current =
                    entry.child_session_id.as_deref() == sessions.session_id.as_deref();
                let duration = entry
                    .started_at
                    .map(|start| {
                        let end = entry.ended_at.unwrap_or_else(|| {
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs() as i64)
                                .unwrap_or(start)
                        });
                        let secs = (end - start).max(0) as u64;
                        if secs < 60 {
                            format!("{secs}s")
                        } else {
                            format!("{}m{}s", secs / 60, secs % 60)
                        }
                    })
                    .unwrap_or_default();
                let pending_label = if entry.awaiting_input() {
                    "question pending".to_string()
                } else {
                    String::new()
                };

                DelegateRowData {
                    status_badge,
                    badge_style,
                    agent: entry.target_agent_id.clone().unwrap_or_default(),
                    objective_source,
                    tools: format_delegate_tools(&entry.stats),
                    msgs: format_delegate_messages(&entry.stats),
                    ctx: format_delegate_context(&entry.stats),
                    cost: format_delegate_cost(&entry.stats),
                    duration,
                    has_pending_input: !pending_label.is_empty(),
                    pending_label,
                    is_current,
                }
            })
            .collect();

        let agent_col_w = rows_data
            .iter()
            .map(|row| delegate_display_width(&row.agent))
            .max()
            .unwrap_or(0);
        let tools_col_w = rows_data
            .iter()
            .map(|row| delegate_display_width(&row.tools))
            .max()
            .unwrap_or(0);
        let msgs_col_w = rows_data
            .iter()
            .map(|row| delegate_display_width(&row.msgs))
            .max()
            .unwrap_or(0);
        let ctx_col_w = rows_data
            .iter()
            .map(|row| delegate_display_width(&row.ctx))
            .max()
            .unwrap_or(0);
        let cost_col_w = rows_data
            .iter()
            .map(|row| delegate_display_width(&row.cost))
            .max()
            .unwrap_or(0);
        let dur_col_w = rows_data
            .iter()
            .map(|row| delegate_display_width(&row.duration))
            .max()
            .unwrap_or(0);
        let show_agent = agent_col_w > 0;
        let show_tools = tools_col_w > 0;
        let show_msgs = msgs_col_w > 0;
        let show_ctx = ctx_col_w > 0;
        let show_cost = cost_col_w > 0;
        let show_dur = dur_col_w > 0;

        let mut fixed_w = DELEGATE_STATUS_COL_W as u16;
        if show_agent {
            fixed_w += agent_col_w;
        }
        if show_tools {
            fixed_w += tools_col_w;
        }
        if show_msgs {
            fixed_w += msgs_col_w;
        }
        if show_ctx {
            fixed_w += ctx_col_w;
        }
        if show_cost {
            fixed_w += cost_col_w;
        }
        if show_dur {
            fixed_w += dur_col_w;
        }
        // Keep stat columns visible; objective shrinks and ellipsizes first.
        // Ratatui tables insert one cell of spacing between adjacent columns, so
        // reserve that spacing up front to keep the trailing ellipsis visible.
        let visible_cols = 2
            + u16::from(show_agent)
            + u16::from(show_tools)
            + u16::from(show_msgs)
            + u16::from(show_ctx)
            + u16::from(show_cost)
            + u16::from(show_dur);
        let column_spacing = visible_cols.saturating_sub(1);
        let objective_w = chunks[3]
            .width
            .saturating_sub(fixed_w)
            .saturating_sub(column_spacing)
            .max(1) as usize;

        let main_style = Theme::popup_bg();
        let dim_style = Theme::status();
        let rows: Vec<Row> = rows_data
            .into_iter()
            .map(|row| {
                let objective_with_pending = if row.pending_label.is_empty() {
                    row.objective_source.clone()
                } else {
                    format!("{} [{}]", row.objective_source, row.pending_label)
                };
                let objective = truncate_with_ellipsis(&objective_with_pending, objective_w);
                let obj_style = if row.has_pending_input {
                    Theme::mode_badge("plan")
                } else if row.is_current {
                    Theme::status_accent()
                } else {
                    main_style
                };

                let mut cells = vec![Cell::from(Span::styled(row.status_badge, row.badge_style))];
                if show_agent {
                    cells.push(Cell::from(Span::styled(row.agent, dim_style)));
                }
                cells.push(Cell::from(Span::styled(objective, obj_style)));
                if show_tools {
                    cells.push(Cell::from(Span::styled(row.tools, dim_style)));
                }
                if show_msgs {
                    cells.push(Cell::from(Span::styled(row.msgs, dim_style)));
                }
                if show_ctx {
                    cells.push(Cell::from(Span::styled(row.ctx, dim_style)));
                }
                if show_cost {
                    cells.push(Cell::from(Span::styled(row.cost, dim_style)));
                }
                if show_dur {
                    cells.push(Cell::from(Span::styled(row.duration, dim_style)));
                }

                Row::new(cells)
            })
            .collect();

        let mut constraints = vec![Constraint::Length(DELEGATE_STATUS_COL_W as u16)];
        if show_agent {
            constraints.push(Constraint::Length(agent_col_w));
        }
        constraints.push(Constraint::Length(objective_w as u16));
        if show_tools {
            constraints.push(Constraint::Length(tools_col_w));
        }
        if show_msgs {
            constraints.push(Constraint::Length(msgs_col_w));
        }
        if show_ctx {
            constraints.push(Constraint::Length(ctx_col_w));
        }
        if show_cost {
            constraints.push(Constraint::Length(cost_col_w));
        }
        if show_dur {
            constraints.push(Constraint::Length(dur_col_w));
        }

        let table = Table::new(rows, constraints)
            .block(Block::default().style(Theme::popup_bg()))
            .style(Theme::popup_bg())
            .row_highlight_style(Theme::selected());

        let selected_idx = delegates
            .delegate_cursor
            .min(entries.len().saturating_sub(1));
        let offset = selected_idx.saturating_sub(visible_rows.saturating_sub(1));
        let selected = Some(selected_idx);
        let mut state = TableState::default()
            .with_offset(offset)
            .with_selected(selected);
        f.render_stateful_widget(table, chunks[3], &mut state);
    }

    // hint
    let selected_entry = entries.get(
        delegates
            .delegate_cursor
            .min(entries.len().saturating_sub(1)),
    );
    let awaiting_selected = selected_entry.is_some_and(|entry| entry.awaiting_input());
    let enter_help = if awaiting_selected {
        "open child to answer"
    } else {
        "load"
    };
    let enter_help_style = if awaiting_selected {
        Theme::mode_badge("plan")
    } else {
        Theme::status()
    };
    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled(format!("{enter_help}  "), enter_help_style),
        Span::styled("tab ", Theme::status_accent()),
        Span::styled("switch", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}

#[cfg(test)]
mod tests {
    use ratatui::{
        backend::TestBackend,
        layout::{Constraint, Direction, Layout, Rect},
        style::Modifier,
    };

    use super::*;
    use crate::domain::activity::{DelegateChildState, DelegateEntry, DelegateStatus};

    fn delegate_entry(
        delegation_id: &str,
        objective: &str,
        status: DelegateStatus,
    ) -> DelegateEntry {
        DelegateEntry {
            delegation_id: delegation_id.into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: objective.into(),
            status,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        }
    }

    fn delegate_chunks(area: Rect) -> std::rc::Rc<[Rect]> {
        let popup_width = area.width.saturating_sub(4).clamp(36, 86);
        let popup_area = Rect {
            x: area.x + area.width.saturating_sub(popup_width) / 2,
            y: area.y + area.height.saturating_sub(area.height * 60 / 100) / 2,
            width: popup_width,
            height: area.height * 60 / 100,
        };
        let inner = Rect {
            x: popup_area.x + 1,
            y: popup_area.y + 1,
            width: popup_area.width.saturating_sub(2),
            height: popup_area.height.saturating_sub(2),
        };

        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(inner)
    }

    fn render_delegate_popup(
        sessions: &SessionsState,
        delegates: &DelegatesState,
        width: u16,
        height: u16,
    ) -> ratatui::buffer::Buffer {
        let backend = TestBackend::new(width, height);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let mut render = RenderState::new();
        terminal
            .draw(|frame| {
                let chunks = delegate_chunks(frame.area());
                draw_tab(frame, sessions, delegates, &mut render, &chunks);
            })
            .unwrap();
        terminal.backend().buffer().clone()
    }

    fn buffer_line(buffer: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buffer.area.width)
            .map(|x| buffer[(x, y)].symbol())
            .collect()
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
    fn delegate_tab_cursor_stays_inside_tiny_session_popup() {
        let sessions = SessionsState {
            session_popup_tab: 1,
            ..SessionsState::new()
        };
        let delegates = DelegatesState::new();
        let backend = TestBackend::new(1, 1);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let mut render = RenderState::new();

        terminal
            .draw(|frame| {
                crate::features::sessions::view::draw_session_popup(
                    frame,
                    &sessions,
                    &delegates,
                    &mut render,
                );
            })
            .unwrap();

        let cursor = terminal.get_cursor_position().unwrap();
        assert!(Rect::new(0, 0, 1, 1).contains(cursor));
    }

    #[test]
    fn draw_delegate_popup_uses_aligned_stat_columns_without_header() {
        let mut sessions = SessionsState::new();
        sessions.session_id = Some("parent".into());
        let mut delegates = DelegatesState::new();
        let mut first = delegate_entry("del-1", "Fix the bug", DelegateStatus::Completed);
        first.child_session_id = Some("child-1".into());
        first.stats = DelegateStats {
            tool_calls: 7,
            messages: 3,
            cost_usd: 0.0125,
            context_tokens: 50_000,
            context_limit: 200_000,
        };
        let mut second = delegate_entry("del-2", "Write docs", DelegateStatus::InProgress);
        second.child_session_id = Some("child-2".into());
        second.stats = DelegateStats {
            tool_calls: 12,
            messages: 1,
            cost_usd: 0.0345,
            context_tokens: 20_000,
            context_limit: 200_000,
        };
        delegates.delegate_entries = vec![first, second];

        let buffer = render_delegate_popup(&sessions, &delegates, 90, 20);

        let (tool_x1, row1) = find_buffer_text(&buffer, "⚒7").expect("missing row1 tools");
        let (tool_x2, row2) = find_buffer_text(&buffer, "⚒12").expect("missing row2 tools");
        let (cost_x1, cost_row1) = find_buffer_text(&buffer, "$0.01").expect("missing row1 cost");
        let (cost_x2, cost_row2) = find_buffer_text(&buffer, "$0.03").expect("missing row2 cost");

        assert_ne!(row1, row2, "expected separate rows for each delegate entry");
        assert_eq!(tool_x1, tool_x2, "tool column must align across rows");
        assert_eq!(cost_x1, cost_x2, "cost column must align across rows");
        assert_eq!(row1, cost_row1, "row1 stats must share same row");
        assert_eq!(row2, cost_row2, "row2 stats must share same row");
    }

    #[test]
    fn draw_delegate_popup_shows_question_pending_marker_and_hint() {
        let sessions = SessionsState::new();
        let mut delegates = DelegatesState::new();
        delegates.delegate_cursor = 1;
        let mut entry = delegate_entry("del-1", "Pending task", DelegateStatus::InProgress);
        entry.child_session_id = Some("child-1".into());
        entry.target_agent_id = Some("coder".into());
        entry.child_state = DelegateChildState::PendingElicitation {
            elicitation_id: "elic-1".into(),
            message: "Need approval".into(),
            requested_schema: serde_json::json!({ "properties": {} }),
            source: "builtin:question".into(),
        };
        delegates.delegate_entries = vec![entry];

        let buffer = render_delegate_popup(&sessions, &delegates, 90, 20);
        let rendered: String = buffer.content().iter().map(|cell| cell.symbol()).collect();

        assert!(
            rendered.contains("question pending"),
            "missing popup marker: {rendered}"
        );
        let (x, y) = find_buffer_text(&buffer, "open child to answer").expect("missing popup hint");
        assert_eq!(buffer[(x, y)].style().fg, Some(Theme::mode_color("plan")));
        assert!(buffer[(x, y)].modifier.contains(Modifier::BOLD));
        assert!(
            rendered.contains("open child to answer"),
            "missing popup hint: {rendered}"
        );
    }

    #[test]
    fn draw_delegate_popup_hides_cost_column_when_all_rows_have_zero_cost() {
        let sessions = SessionsState::new();
        let mut delegates = DelegatesState::new();
        let mut first = delegate_entry("del-1", "Pending task", DelegateStatus::InProgress);
        first.stats = DelegateStats {
            tool_calls: 2,
            messages: 1,
            cost_usd: 0.0,
            context_tokens: 1000,
            context_limit: 200_000,
        };
        let mut second = delegate_entry("del-2", "Another task", DelegateStatus::Completed);
        second.stats = DelegateStats {
            tool_calls: 4,
            messages: 2,
            cost_usd: 0.0,
            context_tokens: 2000,
            context_limit: 200_000,
        };
        delegates.delegate_entries = vec![first, second];

        let buffer = render_delegate_popup(&sessions, &delegates, 90, 20);
        let rendered: String = buffer.content().iter().map(|cell| cell.symbol()).collect();

        assert!(rendered.contains("Pending task"));
        assert!(rendered.contains("Another task"));
        assert!(rendered.contains("⚒2"));
        assert!(rendered.contains("⚒4"));
        assert!(!rendered.contains("$0.0000"), "unexpected zero cost");
        assert!(
            !rendered.contains('$'),
            "cost column should be hidden when all rows have zero cost"
        );
    }

    #[test]
    fn draw_delegate_popup_truncates_long_objectives_with_unicode_ellipsis() {
        let sessions = SessionsState::new();
        let mut delegates = DelegatesState::new();
        let mut entry = delegate_entry(
            "del-1",
            "List the contents of the repository root directory as a simulated user request for the delegate popup",
            DelegateStatus::InProgress,
        );
        entry.stats = DelegateStats {
            tool_calls: 0,
            messages: 0,
            cost_usd: 0.0,
            context_tokens: 180_000,
            context_limit: 200_000,
        };
        delegates.delegate_entries = vec![entry];

        let buffer = render_delegate_popup(&sessions, &delegates, 70, 12);
        let rendered: String = buffer.content().iter().map(|cell| cell.symbol()).collect();

        assert!(rendered.contains('…'));
        assert!(!rendered.contains("..."));
    }

    #[test]
    fn draw_delegate_popup_failed_symbol_uses_dim_surface_background() {
        let sessions = SessionsState::new();
        let mut delegates = DelegatesState::new();
        delegates.delegate_cursor = 0;
        delegates.delegate_entries = vec![
            delegate_entry("del-0", "Selected row", DelegateStatus::Completed),
            delegate_entry("del-1", "Failed row", DelegateStatus::Failed),
        ];

        let buffer = render_delegate_popup(&sessions, &delegates, 90, 20);
        let (x, y) = find_buffer_text(&buffer, "☒").expect("missing failed symbol");
        assert_eq!(
            buffer[(x, y)].style().bg,
            Theme::error_on_dim().bg,
            "failed popup symbol should use dim-surface error background"
        );
    }

    #[test]
    fn draw_delegate_popup_highlights_full_selected_row() {
        let sessions = SessionsState::new();
        let mut delegates = DelegatesState::new();
        delegates.delegate_cursor = 1;
        let mut first = delegate_entry("del-1", "First row", DelegateStatus::Completed);
        first.stats = DelegateStats {
            tool_calls: 7,
            messages: 3,
            cost_usd: 0.0125,
            context_tokens: 50_000,
            context_limit: 200_000,
        };
        let mut second = delegate_entry("del-2", "Second row", DelegateStatus::InProgress);
        second.stats = DelegateStats {
            tool_calls: 4,
            messages: 1,
            cost_usd: 0.0,
            context_tokens: 10_000,
            context_limit: 200_000,
        };
        delegates.delegate_entries = vec![first, second];

        let buffer = render_delegate_popup(&sessions, &delegates, 90, 20);

        let (_, first_y) = find_buffer_text(&buffer, "First row").expect("missing first row");
        let (_, second_y) = find_buffer_text(&buffer, "Second row").expect("missing second row");
        assert_ne!(first_y, second_y);

        // Pick populated and empty cells across the selected row.
        let (obj_x, _) = find_buffer_text(&buffer, "Second row").expect("missing second row");
        let (tool_x, _) = find_buffer_text(&buffer, "⚒4").expect("missing second row tools");
        let (cost_x, _) = find_buffer_text(&buffer, "$0.01").expect("missing first row cost");

        let selected_bg = Theme::selected().bg.expect("selected style must define bg");
        assert_eq!(buffer[(obj_x, second_y)].style().bg, Some(selected_bg));
        assert_eq!(buffer[(tool_x, second_y)].style().bg, Some(selected_bg));
        assert_eq!(
            buffer[(cost_x, second_y)].style().bg,
            Some(selected_bg),
            "empty selected-row cost cell should also be highlighted"
        );
    }

    #[test]
    fn draw_delegate_popup_shows_agent_name_column() {
        let mut sessions = SessionsState::new();
        sessions.session_id = Some("parent".into());
        let mut delegates = DelegatesState::new();
        let mut first = delegate_entry("del-1", "Fix bug", DelegateStatus::Completed);
        first.target_agent_id = Some("coder".into());
        let mut second = delegate_entry("del-2", "Plan work", DelegateStatus::InProgress);
        second.target_agent_id = Some("planner".into());
        delegates.delegate_entries = vec![first, second];

        let buffer = render_delegate_popup(&sessions, &delegates, 90, 20);
        let rendered: String = buffer.content().iter().map(|cell| cell.symbol()).collect();

        assert!(
            rendered.contains("coder"),
            "agent name 'coder' must appear in popup, got: {rendered}"
        );
        assert!(
            rendered.contains("planner"),
            "agent name 'planner' must appear in popup"
        );
    }
}
