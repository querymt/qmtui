use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table, TableState},
};
use unicode_width::UnicodeWidthStr;

use crate::app::{App, AuthPanel, LogLevel};
use crate::protocol::OAuthStatus;
use crate::theme::Theme;

use super::chat::{CHECK_CHECKED, CHECK_FAILED, SpinnerKind, spinner};
use super::start::{COLLAPSE_CLOSED, COLLAPSE_OPEN, short_cwd};
use super::{ARROW_DOWN, ARROW_UP, COLOR_SWATCH, ELLIPSIS, relative_time};

// ── Single-line input scroll helper ──────────────────────────────────────────

/// Returns `(visible_text, cursor_col)` for a single-line input field so the
/// cursor is always within `[0, avail)`.
///
/// * `text`        – full input string
/// * `cursor_byte` – cursor offset in bytes within `text`
/// * `avail`       – available display columns (= field_width - prefix_width)
pub(crate) fn scroll_input(text: &str, cursor_byte: usize, avail: usize) -> (String, usize) {
    let cursor_chars = text[..cursor_byte.min(text.len())].chars().count();
    scroll_input_chars(text, cursor_chars, avail)
}

/// Same as [`scroll_input`] but takes a cursor position in chars instead of bytes.
/// Useful when the display text differs from the source (e.g. masked API key).
pub(crate) fn scroll_input_chars(text: &str, cursor_chars: usize, avail: usize) -> (String, usize) {
    if avail == 0 {
        return (String::new(), 0);
    }
    let scroll = if cursor_chars >= avail {
        cursor_chars + 1 - avail
    } else {
        0
    };
    let visible: String = text.chars().skip(scroll).take(avail).collect();
    let cursor_col = cursor_chars - scroll;
    (visible, cursor_col)
}

// ── Centered rect helper ──────────────────────────────────────────────────────

pub(crate) fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
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

/// Single-mode marker: returns vec with the mode name if this model matches
/// the preference for that specific mode, empty vec otherwise.
fn popup_single_mode_marker(
    app: &App,
    model: &crate::protocol::ModelEntry,
    mode: &'static str,
) -> Vec<&'static str> {
    let fallback_current = if app.agent_mode == mode {
        match (
            app.current_provider.as_deref(),
            app.current_model.as_deref(),
        ) {
            (Some(provider), Some(model_name)) => Some((provider, model_name)),
            _ => None,
        }
    } else {
        None
    };
    let target = app.get_mode_model_preference(mode).or(fallback_current);
    if target.is_some_and(|(provider, model_name)| {
        provider == model.provider && model_name == model.model
    }) {
        vec![mode]
    } else {
        vec![]
    }
}

/// Whether the given model matches a delegate agent's preferred model.
fn popup_delegate_marker(app: &App, model: &crate::protocol::ModelEntry, agent_id: &str) -> bool {
    app.get_delegate_model_preference(agent_id)
        .is_some_and(|(p, m)| p == model.provider && m == model.model)
}

// ── Model popup ───────────────────────────────────────────────────────────────

pub(super) fn draw_model_popup(f: &mut Frame, app: &App) {
    use crate::app::ModelPopupItem;

    const MODEL_MARKER_COL_W: u16 = 4;
    const MODEL_LABEL_MAX_W: u16 = 48;
    const MODEL_POPUP_MAX_W: u16 = MODEL_MARKER_COL_W + MODEL_LABEL_MAX_W + 2;
    const MODEL_POPUP_MIN_W: u16 = 30;

    let has_tabs = true;

    let area = f.area();
    let popup_width = area
        .width
        .saturating_sub(4)
        .clamp(MODEL_POPUP_MIN_W, MODEL_POPUP_MAX_W);
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

    // Layout: title, [tab bar], filter, separator, list, hints
    let constraints: Vec<Constraint> = if has_tabs {
        vec![
            Constraint::Length(1), // title
            Constraint::Length(1), // tab bar
            Constraint::Length(1), // filter
            Constraint::Length(1), // separator
            Constraint::Min(1),    // list
            Constraint::Length(1), // hints
        ]
    } else {
        vec![
            Constraint::Length(1), // title
            Constraint::Length(1), // filter
            Constraint::Length(1), // separator
            Constraint::Min(1),    // list
            Constraint::Length(1), // hints
        ]
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    // Chunk indices depend on whether tabs are shown.
    let (tab_idx, filter_idx, list_idx, hint_idx) = if has_tabs {
        (Some(1), 2, 4, 5)
    } else {
        (None, 1, 3, 4)
    };

    // Title
    f.render_widget(
        Paragraph::new(Span::styled("select model", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );

    // Tab bar (multi-agent only)
    if let Some(ti) = tab_idx {
        let mut tab_spans = Vec::new();
        for i in 0..app.model_popup_tab_count() {
            let label = app.model_popup_tab_label(i);
            let is_active = i == app.model_popup_agent_tab;
            let style = if let Some(mode) = app.model_popup_tab_mode(i) {
                let mut s = ratatui::style::Style::default()
                    .fg(Theme::mode_color(mode))
                    .bg(Theme::bg_dim())
                    .add_modifier(Modifier::BOLD);
                if is_active {
                    s = s.add_modifier(Modifier::UNDERLINED);
                }
                s
            } else if is_active {
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
            chunks[ti],
        );
    }

    // Filter input
    let filter_area = chunks[filter_idx];
    let avail = filter_area.width.saturating_sub(2) as usize;
    let (model_filter_display, model_filter_cur) =
        scroll_input(&app.model_filter, app.model_filter.len(), avail);
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(model_filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        filter_area,
    );
    f.set_cursor_position((filter_area.x + 2 + model_filter_cur as u16, filter_area.y));

    // Resolve current tab's mode (for mode tabs) or agent_id (for agent tabs).
    let active_mode = app.model_popup_tab_mode(app.model_popup_agent_tab);
    let active_agent_id = app
        .model_popup_tab_agent_id(app.model_popup_agent_tab)
        .map(str::to_string);

    let list_area = chunks[list_idx];
    let list_w = list_area.width as usize;

    let items: Vec<ListItem> = app
        .visible_model_popup_items()
        .iter()
        .enumerate()
        .map(|(i, item)| match item {
            ModelPopupItem::ProviderHeader {
                provider,
                model_count,
            } => {
                let selected = i == app.model_cursor;
                let marker = COLLAPSE_CLOSED;
                let count = format!(" {model_count}");
                let avail = list_w.saturating_sub(4 + count.chars().count());
                let label = if provider.chars().count() > avail {
                    let t: String = provider.chars().take(avail.saturating_sub(1)).collect();
                    format!("{t}{ELLIPSIS}")
                } else {
                    provider.clone()
                };
                let gap = avail.saturating_sub(label.chars().count());
                let marker_style = if selected {
                    Theme::selected()
                } else {
                    Theme::status_accent()
                };
                let provider_style = marker_style.add_modifier(Modifier::BOLD);
                let dim_style = if selected {
                    Theme::selected()
                } else {
                    Theme::status()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!(" {marker} "), marker_style),
                    Span::styled(label, provider_style),
                    Span::styled(" ".repeat(gap), dim_style),
                    Span::styled(count, dim_style),
                ]))
            }
            ModelPopupItem::Model { model_idx } => {
                let selected = i == app.model_cursor;
                let model = &app.models[*model_idx];

                // On a mode tab show that mode's marker;
                // on an agent tab show a delegate preference marker.
                let marker_modes: Vec<&str> = match (active_mode, active_agent_id.as_deref()) {
                    (Some(mode), _) => popup_single_mode_marker(app, model, mode),
                    (_, Some(aid)) => {
                        if popup_delegate_marker(app, model, aid) {
                            vec!["delegate"]
                        } else {
                            vec![]
                        }
                    }
                    _ => vec![],
                };

                let marker_bg = if selected {
                    Theme::bg_hl()
                } else {
                    Theme::bg_dim()
                };
                let marker_w = MODEL_MARKER_COL_W as usize;
                let avail = list_w.saturating_sub(marker_w);
                let label = if model.label.chars().count() > avail {
                    let t: String = model.label.chars().take(avail.saturating_sub(1)).collect();
                    format!("{t}{ELLIPSIS}")
                } else {
                    model.label.clone()
                };
                let gap = avail.saturating_sub(label.chars().count());
                let main_style = if selected {
                    Theme::selected()
                } else {
                    Theme::popup_bg()
                };
                let mut spans = Vec::with_capacity(1 + marker_modes.len() * 2 + 2);
                spans.push(Span::styled(" ", main_style));
                for mode in &marker_modes {
                    spans.push(Span::styled(
                        "\u{25cf}",
                        ratatui::style::Style::default()
                            .fg(Theme::mode_color(mode))
                            .bg(marker_bg),
                    ));
                }
                spans.push(Span::styled(
                    " ".repeat(marker_w.saturating_sub(1 + marker_modes.len())),
                    main_style,
                ));
                spans.push(Span::styled(label, main_style));
                spans.push(Span::styled(" ".repeat(gap), main_style));
                ListItem::new(Line::from(spans))
            }
        })
        .collect();

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let visible_rows = list_area.height as usize;
    let offset = app
        .model_cursor
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(Some(app.model_cursor));
    f.render_stateful_widget(list, list_area, &mut state);

    let mut hint_spans = vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("select", Theme::status()),
    ];
    hint_spans.push(Span::styled("  tab ", Theme::status_accent()));
    hint_spans.push(Span::styled("switch", Theme::status()));
    f.render_widget(
        Paragraph::new(Line::from(hint_spans)).style(Theme::popup_bg()),
        chunks[hint_idx],
    );
}

// ── Session popup ─────────────────────────────────────────────────────────────

pub(super) fn draw_session_popup(f: &mut Frame, app: &mut App) {
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
        let is_active = i == app.session_popup_tab;
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

    if app.session_popup_tab == 0 {
        draw_session_tab_content(f, app, &chunks);
    } else {
        draw_delegate_tab_content(f, app, &chunks);
    }
}

fn draw_session_tab_content(f: &mut Frame, app: &mut App, chunks: &std::rc::Rc<[Rect]>) {
    use crate::app::PopupItem;

    // filter
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (session_filter_display, session_filter_cur) =
        scroll_input(&app.session_filter, app.session_filter.len(), avail);
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
    let popup_items = app.visible_popup_items();
    let list_w = chunks[3].width as usize;
    let visible_rows = chunks[3].height as usize;
    app.session_popup_visible_rows = visible_rows;

    let items: Vec<ListItem> = popup_items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let selected = i == app.session_cursor;
            match item {
                PopupItem::GroupHeader {
                    cwd,
                    session_count,
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
                    ListItem::new(Line::from(vec![
                        Span::styled(format!(" {indicator} "), header_style),
                        Span::styled(cwd_short, header_style),
                        Span::styled(format!("  ({session_count}) "), dim_style),
                    ]))
                }
                PopupItem::Session {
                    group_idx,
                    session_idx,
                } => {
                    let s = &app.session_groups[*group_idx].sessions[*session_idx];
                    let id_short: String = s.session_id.chars().take(8).collect();
                    let time_str = s
                        .updated_at
                        .as_deref()
                        .map(relative_time)
                        .unwrap_or_default();
                    let title = s.title.as_deref().unwrap_or("(untitled)");

                    let is_active = app.session_id.as_deref() == Some(s.session_id.as_str());
                    let is_parent = app.parent_session_id.as_deref() == Some(s.session_id.as_str());
                    let marker_part = if is_active {
                        " ● "
                    } else if is_parent {
                        " \u{2b11} "
                    } else {
                        "   "
                    };
                    let id_part = format!(" {id_short} ");
                    let time_part = format!(" {time_str:>7} ");
                    let avail = list_w.saturating_sub(
                        marker_part.chars().count()
                            + id_part.chars().count()
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

                    let mut spans = vec![
                        Span::styled(marker_part, marker_style),
                        Span::styled(id_part, id_style),
                        Span::styled(title_display, main_style),
                        Span::styled(" ".repeat(title_gap), dim_style),
                    ];
                    spans.push(Span::styled(time_part, time_style));

                    ListItem::new(Line::from(spans))
                }
            }
        })
        .collect();

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let offset = app
        .session_cursor
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(Some(app.session_cursor));
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

// ── Delegate session popup ─────────────────────────────────────────────────────

const DELEGATE_POPUP_MAX_W: u16 = 72;
const DELEGATE_POPUP_MIN_W: u16 = 36;
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
    is_current: bool,
}

fn format_delegate_tools(stats: &crate::app::DelegateStats) -> String {
    if stats.tool_calls > 0 {
        format!("{DELEGATE_ICON_TOOLS}{}", stats.tool_calls)
    } else {
        String::new()
    }
}

fn format_delegate_messages(stats: &crate::app::DelegateStats) -> String {
    if stats.messages > 0 {
        format!("{DELEGATE_ICON_MSG}{}", stats.messages)
    } else {
        String::new()
    }
}

fn format_delegate_context(stats: &crate::app::DelegateStats) -> String {
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

fn format_delegate_cost(stats: &crate::app::DelegateStats) -> String {
    if stats.cost_usd > 0.0 {
        format!("${:.2}", stats.cost_usd)
    } else {
        String::new()
    }
}

fn delegate_display_width(text: &str) -> u16 {
    UnicodeWidthStr::width(text) as u16
}

fn draw_delegate_tab_content(f: &mut Frame, app: &mut App, chunks: &std::rc::Rc<[Rect]>) {
    use crate::app::DelegateStatus;

    // filter
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (filter_display, filter_cur) =
        scroll_input(&app.delegate_filter, app.delegate_filter.len(), avail);
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
    app.delegate_popup_visible_rows = visible_rows;
    let entries = app.visible_delegate_entries();

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
                        spinner(SpinnerKind::Braille, app.tick).to_string()
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

                let is_current = entry.child_session_id.as_deref() == app.session_id.as_deref();
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
                let objective = truncate_with_ellipsis(&row.objective_source, objective_w);
                let obj_style = if row.is_current {
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

        let selected_idx = app.delegate_cursor.min(entries.len().saturating_sub(1));
        let offset = selected_idx.saturating_sub(visible_rows.saturating_sub(1));
        let selected = Some(selected_idx);
        let mut state = TableState::default()
            .with_offset(offset)
            .with_selected(selected);
        f.render_stateful_widget(table, chunks[3], &mut state);
    }

    // hint
    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("load  ", Theme::status()),
        Span::styled("tab ", Theme::status_accent()),
        Span::styled("switch", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}

pub(crate) fn truncate_with_ellipsis(text: &str, max_chars: usize) -> String {
    if text.chars().count() > max_chars {
        let t: String = text.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{t}{ELLIPSIS}")
    } else {
        text.to_string()
    }
}

// ── Theme list item builder ───────────────────────────────────────────────────

/// Builds a single [`ListItem`] for the theme picker list.
///
/// Layout (mirrors session-popup column style):
/// ```text
/// [marker][label padded to avail][■■■■■■■■■■■■■■■■]
/// ```
/// * `marker`   – `"● "` when `orig_idx == current_idx`, otherwise `"  "`
/// * `label`    – theme display name, truncated with `…` if needed
/// * swatches   – 16 `■` chars, each coloured with its base16 slot colour
///
/// The row background comes from `row_bg` (selected = `bg_hl`, normal = `bg_dim`).
pub(crate) fn build_theme_list_item(
    t: &crate::themes_gen::Base16Palette,
    orig_idx: usize,
    current_idx: usize,
    list_w: usize,
    is_selected: bool,
) -> ListItem<'static> {
    const NUM_SWATCHES: usize = 16;
    // " " gap between label and swatches
    const GAP: usize = 1;

    let marker = if orig_idx == current_idx {
        "● "
    } else {
        "  "
    };
    let marker_w = marker.chars().count();
    let swatches_w = NUM_SWATCHES + GAP; // 16 ■ + 1 space

    // Styles ─────────────────────────────────────────────────────────────────
    let (main_style, dim_style, row_bg) = if is_selected {
        (Theme::selected(), Theme::selected(), Theme::bg_hl())
    } else {
        (Theme::popup_bg(), Theme::status(), Theme::bg_dim())
    };
    let marker_style = if orig_idx == current_idx {
        Theme::status_accent().bg(row_bg)
    } else {
        dim_style
    };

    // Label truncation (same pattern as session title) ───────────────────────
    let avail = list_w.saturating_sub(marker_w + swatches_w);
    let label: String = t.label.chars().collect();
    let label_display = if label.chars().count() > avail {
        let t: String = label.chars().take(avail.saturating_sub(1)).collect();
        format!("{t}{ELLIPSIS}")
    } else {
        label.clone()
    };
    let label_gap = avail.saturating_sub(label_display.chars().count());

    // Build spans ─────────────────────────────────────────────────────────────
    let mut spans: Vec<Span<'static>> = Vec::with_capacity(3 + NUM_SWATCHES + 1);
    spans.push(Span::styled(marker, marker_style));
    spans.push(Span::styled(label_display, main_style));
    spans.push(Span::styled(" ".repeat(label_gap + GAP), dim_style));

    // 16 colour swatches ──────────────────────────────────────────────────────
    for &c in &t.colors {
        let fg = crate::theme::u32_to_color(c);
        spans.push(Span::styled(
            COLOR_SWATCH,
            ratatui::style::Style::default().fg(fg).bg(row_bg),
        ));
    }

    ListItem::new(Line::from(spans))
}

// ── New session popup ─────────────────────────────────────────────────────────

pub(super) fn draw_new_session_popup(f: &mut Frame, app: &App) {
    let area = f.area();
    let show_completion = app
        .new_session_completion
        .as_ref()
        .map(|completion| !completion.results.is_empty())
        .unwrap_or(false);
    let popup_width = area.width.saturating_sub(4).clamp(24, 72);
    let popup_height = area
        .height
        .saturating_sub(4)
        .min(if show_completion { 10 } else { 6 })
        .max(4);
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(popup_width) / 2,
        y: area.y + area.height.saturating_sub(popup_height) / 2,
        width: popup_width,
        height: popup_height,
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
    let (path_display, path_cur) =
        scroll_input(&app.new_session_path, app.new_session_cursor, avail);
    let input_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(path_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(input_line).style(Theme::popup_bg()),
        chunks[2],
    );
    f.set_cursor_position((chunks[2].x + 2 + path_cur as u16, chunks[2].y));

    if let Some(completion) = &app.new_session_completion
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
        let selected = Some(completion.selected_index).filter(|_| !completion.results.is_empty());
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

// ── Theme popup ───────────────────────────────────────────────────────────────

pub(super) fn draw_theme_popup(f: &mut Frame, app: &App) {
    const THEME_MARKER_COL_W: u16 = 2;
    const THEME_LABEL_MAX_W: u16 = 44;
    const THEME_SWATCH_COL_W: u16 = 17;
    const THEME_ROW_MAX_W: u16 = THEME_MARKER_COL_W + THEME_LABEL_MAX_W + THEME_SWATCH_COL_W;
    const THEME_POPUP_MAX_W: u16 = THEME_ROW_MAX_W + 2;
    const THEME_POPUP_MIN_W: u16 = 28;

    let area = f.area();
    let popup_width = area
        .width
        .saturating_sub(4)
        .clamp(THEME_POPUP_MIN_W, THEME_POPUP_MAX_W);
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
            Constraint::Length(1), // title
            Constraint::Length(1), // filter
            Constraint::Length(1), // spacer
            Constraint::Min(1),    // list
            Constraint::Length(1), // hint
        ])
        .split(inner);

    // title
    f.render_widget(
        Paragraph::new(Span::styled("theme", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );

    // filter
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (theme_filter_display, theme_filter_cur) =
        scroll_input(&app.theme_filter, app.theme_filter.len(), avail);
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(theme_filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        chunks[1],
    );
    f.set_cursor_position((chunks[1].x + 2 + theme_filter_cur as u16, chunks[1].y));

    // theme list
    let all_themes = Theme::available_themes();
    let filter_lower = app.theme_filter.to_lowercase();
    let filtered: Vec<(usize, &crate::themes_gen::Base16Palette)> = all_themes
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            filter_lower.is_empty()
                || t.label.to_lowercase().contains(&filter_lower)
                || t.id.to_lowercase().contains(&filter_lower)
        })
        .collect();

    let current_idx = Theme::current_index();
    let list_w = chunks[3].width as usize;

    let items: Vec<ListItem> = filtered
        .iter()
        .enumerate()
        .map(|(i, (orig_idx, t))| {
            build_theme_list_item(t, *orig_idx, current_idx, list_w, i == app.theme_cursor)
        })
        .collect();

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let visible_rows = chunks[3].height as usize;
    let offset = app
        .theme_cursor
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(Some(app.theme_cursor));
    f.render_stateful_widget(list, chunks[3], &mut state);

    // hint
    let hint = Line::from(vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("apply", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}

// ── Log popup ─────────────────────────────────────────────────────────────────

pub(super) fn draw_log_popup(f: &mut Frame, app: &App) {
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
        scroll_input(&app.log_filter, app.log_filter.len(), avail);
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
            format!("{}+", app.log_level_filter.label()),
            popup_log_level_style(app.log_level_filter),
        ),
    ]);
    f.render_widget(
        Paragraph::new(level_line).style(Theme::popup_bg()),
        chunks[2],
    );

    let filtered = app.filtered_logs();
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
    let selected =
        Some(app.log_cursor.min(filtered.len().saturating_sub(1))).filter(|_| !filtered.is_empty());
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

// ── Help popup ────────────────────────────────────────────────────────────────

/// One section in the keyboard-shortcut reference.
pub(crate) struct ShortcutSection {
    pub title: &'static str,
    pub rows: &'static [(&'static str, &'static str)],
}

/// All shortcut sections shown in the help popup.
/// Keep entries sorted logically (not alphabetically).
pub(crate) fn shortcut_sections() -> &'static [ShortcutSection] {
    &[
        ShortcutSection {
            title: "global",
            rows: &[
                ("C-x \u{2026}", "chord prefix"),
                ("Tab", "cycle mode (build \u{2192} plan \u{2192} review)"),
                ("C-c", "clear input / quit"),
            ],
        },
        ShortcutSection {
            title: "chord  (C-x \u{2026})",
            rows: &[
                ("?", "this help"),
                ("a", "provider auth"),
                ("d", "delegate sessions"),
                ("e", "external editor"),
                ("m", "model selector"),
                ("n", "new session"),
                ("l", "logs popup"),
                ("p", "jump to parent session"),
                ("q", "quit"),
                ("r", "redo"),
                ("s", "session switcher"),
                ("t", "theme picker"),
                ("u", "undo"),
            ],
        },
        ShortcutSection {
            title: "chat",
            rows: &[
                ("Enter", "send message"),
                ("Esc", "cancel / dismiss mention"),
                ("\u{2191} \u{2193}", "scroll history / navigate mentions"),
                ("PgUp PgDn", "scroll fast"),
                ("\u{2190} \u{2192}", "move cursor"),
                ("Home  End", "start / end of input line"),
                ("End (empty)", "snap to bottom of history"),
                ("Backspace", "delete left"),
                ("Del", "delete right"),
                ("@", "mention a file"),
                (
                    "Ctrl+t",
                    "cycle thinking level (auto\u{2192}low\u{2192}medium\u{2192}high\u{2192}max)",
                ),
            ],
        },
        ShortcutSection {
            title: "sessions screen",
            rows: &[
                ("\u{2191} \u{2193}", "navigate sessions / groups"),
                ("Enter", "load session  /  collapse-expand group"),
                ("Del", "delete selected session"),
                ("type", "filter sessions by title or id"),
                ("Backspace", "clear last filter character"),
                ("q  Esc", "quit"),
            ],
        },
        ShortcutSection {
            title: "popups",
            rows: &[
                ("\u{2191} \u{2193}", "navigate"),
                ("Enter", "confirm"),
                ("Esc", "close"),
                ("type", "filter (sessions, models, themes)"),
            ],
        },
        ShortcutSection {
            title: "elicitation",
            rows: &[
                ("\u{2191} \u{2193}", "navigate fields / options"),
                ("Space", "toggle multi-select option"),
                ("Enter", "submit"),
                ("Esc", "decline"),
            ],
        },
        ShortcutSection {
            title: "slash commands",
            rows: &[
                ("/model [q]", "model selector (optional filter)"),
                ("/mode [m]", "switch mode (build, plan)"),
                ("/review", "enter review mode"),
                (
                    "/thinking [lvl]",
                    "set thinking (auto, low, med, high, max)",
                ),
                ("/theme", "open theme picker"),
                ("/sessions", "open session switcher"),
                ("/delegates", "list delegate sessions"),
                ("/new", "new session"),
                ("/help", "show help"),
                ("/logs", "open logs popup"),
                ("/auth", "provider auth"),
                ("/undo", "undo last turn"),
                ("/redo", "redo"),
                ("/editor", "open external editor"),
                ("/cancel", "cancel active turn"),
                ("/quit", "quit"),
            ],
        },
    ]
}

pub(super) fn draw_help_popup(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup_area = centered_rect(70, 80, area);

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
            Constraint::Length(1), // spacer
            Constraint::Min(1),    // list
            Constraint::Length(1), // hint
        ])
        .split(inner);

    // title
    f.render_widget(
        Paragraph::new(Span::styled("shortcuts", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );

    // shortcut list ───────────────────────────────────────────────────────────
    // Key column: 2-space left pad + key left-aligned in 12 chars = 14 total.
    const KEY_COL_W: usize = 14;

    let mut items: Vec<ListItem> = Vec::new();

    for (section_idx, section) in shortcut_sections().iter().enumerate() {
        // blank spacer row before every section except the first
        if section_idx > 0 {
            items.push(ListItem::new(Line::from(Span::raw(""))));
        }
        // section header
        items.push(ListItem::new(Line::from(Span::styled(
            format!("  {}", section.title),
            Theme::popup_title(),
        ))));
        // shortcut rows
        for &(key, desc) in section.rows {
            let key_col = format!("  {key:<KEY_COL_W$}");
            items.push(ListItem::new(Line::from(vec![
                Span::styled(key_col, Theme::status()),
                Span::styled(desc, Theme::popup_bg()),
            ])));
        }
    }

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let mut state = ListState::default().with_offset(app.help_scroll);
    f.render_stateful_widget(list, chunks[2], &mut state);

    // hint
    f.render_widget(
        Paragraph::new(Span::styled(
            format!(" {ARROW_UP}{ARROW_DOWN} scroll  esc close"),
            Theme::status(),
        ))
        .style(Theme::popup_bg()),
        chunks[3],
    );
}

// ── Auth popup ────────────────────────────────────────────────────────────────

const AUTH_POPUP_MAX_W: u16 = 68;
const AUTH_POPUP_MIN_W: u16 = 44;

pub(super) fn draw_auth_popup(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup_width = area
        .width
        .saturating_sub(4)
        .clamp(AUTH_POPUP_MIN_W, AUTH_POPUP_MAX_W);
    let popup_height = (area.height * 30 / 100).max(15).min(area.height);
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(popup_width) / 2,
        y: area.y + area.height.saturating_sub(popup_height) / 2,
        width: popup_width,
        height: popup_height,
    };

    f.render_widget(Clear, popup_area);
    f.render_widget(Block::default().style(Theme::popup_bg()), popup_area);

    let inner = Rect {
        x: popup_area.x + 1,
        y: popup_area.y + 1,
        width: popup_area.width.saturating_sub(2),
        height: popup_area.height.saturating_sub(2),
    };

    // If clipboard fallback is active, draw a simple URL display popup over everything.
    if let Some(ref url) = app.auth_clipboard_fallback {
        draw_auth_clipboard_fallback(f, inner, url);
        return;
    }

    // Determine detail area height based on panel state.
    let detail_height: u16 = match app.auth_panel {
        AuthPanel::List => {
            if app.auth_selected.is_some() {
                3 // status line + info
            } else {
                0
            }
        }
        AuthPanel::ApiKeyInput => 5,
        AuthPanel::OAuthFlow => 7,
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),             // title
            Constraint::Length(1),             // filter
            Constraint::Length(1),             // spacer
            Constraint::Min(1),                // provider list
            Constraint::Length(detail_height), // detail panel
            Constraint::Length(1),             // hint
        ])
        .split(inner);

    // title
    f.render_widget(
        Paragraph::new(Span::styled("provider auth", Theme::popup_title()))
            .style(Theme::popup_bg()),
        chunks[0],
    );

    // filter
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (auth_filter_display, auth_filter_cur) =
        scroll_input(&app.auth_filter, app.auth_filter.len(), avail);
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(auth_filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        chunks[1],
    );
    if app.auth_panel == AuthPanel::List {
        f.set_cursor_position((chunks[1].x + 2 + auth_filter_cur as u16, chunks[1].y));
    }

    // provider list
    let filtered = app.filtered_auth_providers();
    let list_w = chunks[3].width as usize;

    let items: Vec<ListItem> = filtered
        .iter()
        .enumerate()
        .map(|(i, (_, provider))| {
            let selected = i == app.auth_cursor;
            let badge = provider.auth_badge_label();
            let badge_active = provider.is_auth_active();
            let is_expired = provider.oauth_status == Some(OAuthStatus::Expired);

            // Badge styling
            let badge_style = if selected {
                Theme::selected()
            } else if is_expired {
                ratatui::style::Style::default()
                    .fg(Theme::warn())
                    .bg(Theme::bg_dim())
            } else if badge_active {
                ratatui::style::Style::default()
                    .fg(Theme::ok())
                    .bg(Theme::bg_dim())
            } else {
                Theme::status()
            };

            let name_style = if selected {
                Theme::selected()
            } else {
                Theme::popup_bg()
            };

            let badge_str = format!("[{badge}]");
            let badge_len = badge_str.chars().count();
            let name = &provider.display_name;
            let avail = list_w.saturating_sub(badge_len + 3);
            let name_display = if name.chars().count() > avail {
                let t: String = name.chars().take(avail.saturating_sub(1)).collect();
                format!("{t}{ELLIPSIS}")
            } else {
                name.to_string()
            };
            let gap = avail.saturating_sub(name_display.chars().count());

            ListItem::new(Line::from(vec![
                Span::styled(format!(" {name_display}"), name_style),
                Span::styled(" ".repeat(gap), name_style),
                Span::styled(format!(" {badge_str} "), badge_style),
            ]))
        })
        .collect();

    if items.is_empty() {
        let msg = if app.auth_filter.is_empty() {
            "loading providers..."
        } else {
            "no providers match filter"
        };
        f.render_widget(
            Paragraph::new(Span::styled(format!(" {msg}"), Theme::status()))
                .style(Theme::popup_bg()),
            chunks[3],
        );
    } else {
        let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
        let visible_rows = chunks[3].height as usize;
        let offset = app
            .auth_cursor
            .saturating_sub(visible_rows.saturating_sub(1));
        let mut state = ListState::default()
            .with_offset(offset)
            .with_selected(Some(app.auth_cursor));
        f.render_stateful_widget(list, chunks[3], &mut state);
    }

    // detail panel
    if detail_height > 0 {
        draw_auth_detail_panel(f, app, chunks[4]);
    }

    // hint
    let hint = match app.auth_panel {
        AuthPanel::List => {
            let mut spans = vec![
                Span::styled(" esc ", Theme::status_accent()),
                Span::styled("close  ", Theme::status()),
                Span::styled("enter ", Theme::status_accent()),
                Span::styled("select  ", Theme::status()),
            ];
            // Show C-d contextually when a provider with clearable credentials is selected
            if let Some(idx) = app.auth_selected
                && let Some(provider) = app.auth_providers.get(idx)
            {
                if provider.oauth_status == Some(OAuthStatus::Connected) {
                    spans.push(Span::styled("C-d ", Theme::status_accent()));
                    spans.push(Span::styled("disconnect  ", Theme::status()));
                } else if provider.has_stored_api_key {
                    spans.push(Span::styled("C-d ", Theme::status_accent()));
                    spans.push(Span::styled("clear key  ", Theme::status()));
                }
            }
            spans.push(Span::styled("C-k ", Theme::status_accent()));
            spans.push(Span::styled("api key  ", Theme::status()));
            spans.push(Span::styled("C-o ", Theme::status_accent()));
            spans.push(Span::styled("oauth", Theme::status()));
            Line::from(spans)
        }
        AuthPanel::ApiKeyInput => Line::from(vec![
            Span::styled(" esc ", Theme::status_accent()),
            Span::styled("back  ", Theme::status()),
            Span::styled("enter ", Theme::status_accent()),
            Span::styled("save  ", Theme::status()),
            Span::styled("tab ", Theme::status_accent()),
            Span::styled("toggle mask  ", Theme::status()),
            Span::styled("C-d ", Theme::status_accent()),
            Span::styled("clear key", Theme::status()),
        ]),
        AuthPanel::OAuthFlow => Line::from(vec![
            Span::styled(" esc ", Theme::status_accent()),
            Span::styled("back  ", Theme::status()),
            Span::styled("C-y ", Theme::status_accent()),
            Span::styled("copy url  ", Theme::status()),
            Span::styled("enter ", Theme::status_accent()),
            Span::styled("complete", Theme::status()),
        ]),
    };
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[5]);
}

fn draw_auth_detail_panel(f: &mut Frame, app: &App, area: Rect) {
    let Some(idx) = app.auth_selected else {
        return;
    };
    let Some(provider) = app.auth_providers.get(idx) else {
        return;
    };

    match app.auth_panel {
        AuthPanel::List => {
            // Show selected provider status summary
            let mut lines: Vec<Line<'static>> = Vec::new();

            let status_label = provider.auth_badge_label();
            let is_active = provider.is_auth_active();
            let status_style = if is_active {
                ratatui::style::Style::default()
                    .fg(Theme::ok())
                    .bg(Theme::bg_dim())
            } else {
                Theme::status()
            };

            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", provider.display_name), Theme::popup_title()),
                Span::styled(format!("[{status_label}]"), status_style),
            ]));

            if provider.is_unconfigurable() {
                lines.push(Line::from(Span::styled(
                    " OAuth required (not available in this build)",
                    Theme::status(),
                )));
            } else if let Some(ref env_var) = provider.env_var_name {
                lines.push(Line::from(Span::styled(
                    format!(" env: {env_var}"),
                    Theme::status(),
                )));
            }

            if let Some((success, ref msg)) = app.auth_result_message {
                let style = if success {
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim())
                } else {
                    ratatui::style::Style::default()
                        .fg(Theme::err())
                        .bg(Theme::bg_dim())
                };
                lines.push(Line::from(Span::styled(format!(" {msg}"), style)));
            }

            for (i, line) in lines.into_iter().enumerate() {
                if i as u16 >= area.height {
                    break;
                }
                let row = Rect {
                    x: area.x,
                    y: area.y + i as u16,
                    width: area.width,
                    height: 1,
                };
                f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
            }
        }

        AuthPanel::ApiKeyInput => {
            let mut lines: Vec<Line<'static>> = Vec::new();

            lines.push(Line::from(vec![
                Span::styled(
                    format!(" API Key for {} ", provider.display_name),
                    Theme::popup_title(),
                ),
                if let Some(ref env_var) = provider.env_var_name {
                    Span::styled(format!(" {env_var}"), Theme::status())
                } else {
                    Span::raw("")
                },
            ]));

            if provider.has_stored_api_key {
                lines.push(Line::from(Span::styled(
                    " key stored in keychain",
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim()),
                )));
            }

            // Input line
            let cursor_chars = app.auth_api_key_input[..app.auth_api_key_cursor]
                .chars()
                .count();
            let display_input = if app.auth_api_key_masked && !app.auth_api_key_input.is_empty() {
                "\u{2022}".repeat(app.auth_api_key_input.chars().count())
            } else {
                app.auth_api_key_input.clone()
            };
            let placeholder = if provider.has_stored_api_key {
                "new key to update..."
            } else {
                "enter API key..."
            };
            // " > " prefix = 3 cols
            let avail = area.width.saturating_sub(3) as usize;
            let (input_text, api_key_cur) = if app.auth_api_key_input.is_empty() {
                (placeholder.to_string(), 0usize)
            } else {
                let (vis, col) = scroll_input_chars(&display_input, cursor_chars, avail);
                (vis, col)
            };
            let input_style = if app.auth_api_key_input.is_empty() {
                Theme::status()
            } else {
                Theme::popup_bg()
            };
            lines.push(Line::from(vec![
                Span::styled(" > ", Theme::popup_title()),
                Span::styled(input_text, input_style),
            ]));

            if let Some((success, ref msg)) = app.auth_result_message {
                let style = if success {
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim())
                } else {
                    ratatui::style::Style::default()
                        .fg(Theme::err())
                        .bg(Theme::bg_dim())
                };
                lines.push(Line::from(Span::styled(format!(" {msg}"), style)));
            }

            for (i, line) in lines.into_iter().enumerate() {
                if i as u16 >= area.height {
                    break;
                }
                let row = Rect {
                    x: area.x,
                    y: area.y + i as u16,
                    width: area.width,
                    height: 1,
                };
                f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
            }

            // Position cursor in the input field
            let input_row_idx = if provider.has_stored_api_key { 2 } else { 1 };
            if (input_row_idx as u16) < area.height {
                f.set_cursor_position((
                    area.x + 3 + api_key_cur as u16,
                    area.y + input_row_idx as u16,
                ));
            }
        }

        AuthPanel::OAuthFlow => {
            let mut lines: Vec<Line<'static>> = Vec::new();

            if let Some(ref flow) = app.auth_oauth_flow {
                let is_device_poll = flow.flow_kind == crate::protocol::OAuthFlowKind::DevicePoll;

                lines.push(Line::from(Span::styled(
                    format!(" OAuth for {}", flow.provider),
                    Theme::popup_title(),
                )));

                // Truncate URL for display
                let url = &flow.authorization_url;
                let avail = area.width.saturating_sub(3) as usize;
                let url_display = if url.chars().count() > avail {
                    let t: String = url.chars().take(avail.saturating_sub(1)).collect();
                    format!("{t}{ELLIPSIS}")
                } else {
                    url.to_string()
                };
                lines.push(Line::from(Span::styled(
                    format!(" {url_display}"),
                    Theme::status(),
                )));

                if is_device_poll {
                    lines.push(Line::from(Span::styled(
                        " Open URL, approve, then press Enter to check",
                        Theme::status(),
                    )));
                } else {
                    // " > " prefix = 3 cols
                    let avail = area.width.saturating_sub(3) as usize;
                    let (response_display, _) = scroll_input(
                        &app.auth_oauth_response,
                        app.auth_oauth_response_cursor,
                        avail,
                    );
                    lines.push(Line::from(Span::styled(
                        " Open URL, approve, paste callback below:",
                        Theme::status(),
                    )));
                    lines.push(Line::from(vec![
                        Span::styled(" > ", Theme::popup_title()),
                        Span::styled(response_display, Theme::popup_bg()),
                    ]));
                }
            } else {
                lines.push(Line::from(Span::styled(
                    " Starting OAuth flow...",
                    Theme::status(),
                )));
            }

            if let Some((success, ref msg)) = app.auth_result_message {
                let style = if success {
                    ratatui::style::Style::default()
                        .fg(Theme::ok())
                        .bg(Theme::bg_dim())
                } else {
                    ratatui::style::Style::default()
                        .fg(Theme::err())
                        .bg(Theme::bg_dim())
                };
                lines.push(Line::from(Span::styled(format!(" {msg}"), style)));
            }

            for (i, line) in lines.into_iter().enumerate() {
                if i as u16 >= area.height {
                    break;
                }
                let row = Rect {
                    x: area.x,
                    y: area.y + i as u16,
                    width: area.width,
                    height: 1,
                };
                f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
            }

            // Position cursor on callback input (non-device-poll only, row 3)
            if app.auth_oauth_flow.is_some()
                && app
                    .auth_oauth_flow
                    .as_ref()
                    .is_some_and(|f| f.flow_kind != crate::protocol::OAuthFlowKind::DevicePoll)
            {
                let avail = area.width.saturating_sub(3) as usize;
                let (_, oauth_cur) = scroll_input(
                    &app.auth_oauth_response,
                    app.auth_oauth_response_cursor,
                    avail,
                );
                if 3 < area.height {
                    f.set_cursor_position((area.x + 3 + oauth_cur as u16, area.y + 3));
                }
            }
        }
    }
}

fn draw_auth_clipboard_fallback(f: &mut Frame, area: Rect, url: &str) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(Span::styled(
        " Clipboard not available",
        Theme::popup_title(),
    )));
    lines.push(Line::from(Span::styled(
        " Copy this URL manually:",
        Theme::status(),
    )));
    lines.push(Line::from(""));

    // Word-wrap the URL into available width
    let avail = area.width.saturating_sub(2) as usize;
    let mut remaining = url;
    while !remaining.is_empty() {
        let take = remaining.len().min(avail);
        lines.push(Line::from(Span::styled(
            format!(" {}", &remaining[..take]),
            Theme::popup_bg(),
        )));
        remaining = &remaining[take..];
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " press any key to dismiss",
        Theme::status(),
    )));

    for (i, line) in lines.into_iter().enumerate() {
        if i as u16 >= area.height {
            break;
        }
        let row = Rect {
            x: area.x,
            y: area.y + i as u16,
            width: area.width,
            height: 1,
        };
        f.render_widget(Paragraph::new(line).style(Theme::popup_bg()), row);
    }
}
