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
    f.set_cursor_position((chunks[1].x + 2 + filter_cur as u16, chunks[1].y));

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
