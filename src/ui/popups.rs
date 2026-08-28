use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table, TableState},
};

use super::{ARROW_DOWN, ARROW_UP, COLOR_SWATCH};

use crate::app::App;
use crate::diagnostics::LogLevel;
use crate::theme::Theme;
use crate::view_shared::{ELLIPSIS, scroll_input, truncate_with_ellipsis};

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

// ── Fork popup ────────────────────────────────────────────────────────────────

pub(super) fn draw_fork_turn_popup(f: &mut Frame, app: &App) {
    let area = f.area();
    let popup_width = area.width.saturating_sub(4).clamp(36, 84);
    let popup_height = area.height.saturating_sub(4).clamp(6, 12);
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
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Span::styled("Fork Session", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );

    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (filter_display, filter_cur) =
        scroll_input(&app.chat.fork_filter, app.chat.fork_filter.len(), avail);
    let input_line = Line::from(vec![
        Span::styled("/ ", Theme::popup_title()),
        Span::styled(filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(input_line).style(Theme::popup_bg()),
        chunks[1],
    );
    f.set_cursor_position((chunks[1].x + 2 + filter_cur as u16, chunks[1].y));

    let turns = app.chat.visible_fork_turns();
    if turns.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("No forkable turns", Theme::status()))
                .style(Theme::popup_bg()),
            chunks[2],
        );
    } else {
        let row_width = chunks[2].width as usize;
        let preview_budget = row_width.saturating_sub(8) / 2;
        let selected_idx = app.chat.fork_cursor.min(turns.len().saturating_sub(1));
        let selected = Some(selected_idx);
        let dim_style = Theme::status().add_modifier(Modifier::DIM);
        let selected_style = Theme::selected();
        let selected_bg = selected_style.bg.unwrap_or_else(Theme::bg_hl);
        let selected_dim_style = Style::default()
            .fg(dim_style.fg.unwrap_or_else(Theme::dim))
            .bg(selected_bg)
            .add_modifier(Modifier::DIM);
        let rows: Vec<Row> = turns
            .iter()
            .enumerate()
            .map(|(idx, turn)| {
                let user_text = turn.user_preview.replace('\n', " ");
                let assistant_text = turn.assistant_preview.replace('\n', " ");
                let user = truncate_with_ellipsis(&user_text, preview_budget);
                let assistant = truncate_with_ellipsis(&assistant_text, preview_budget);
                let boundary_style = if idx == selected_idx {
                    selected_dim_style
                } else {
                    dim_style
                };
                let message_style = if idx == selected_idx {
                    Theme::selected()
                } else {
                    Theme::popup_bg()
                };
                Row::new(vec![
                    Cell::from(Span::styled(turn.turn_index.to_string(), boundary_style)),
                    Cell::from(Span::styled(user, message_style)),
                    Cell::from(Span::styled(ELLIPSIS, boundary_style)),
                    Cell::from(Span::styled(assistant, message_style)),
                ])
            })
            .collect();
        let table = Table::new(
            rows,
            [
                Constraint::Length(3),
                Constraint::Percentage(50),
                Constraint::Length(1),
                Constraint::Percentage(50),
            ],
        )
        .block(Block::default().style(Theme::popup_bg()))
        .style(Theme::popup_bg())
        .row_highlight_style(Theme::selected());
        let mut state = TableState::default().with_selected(selected);
        f.render_stateful_widget(table, chunks[2], &mut state);
    }

    let hint = Line::from(vec![
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("fork  ", Theme::status()),
        Span::styled("esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("type ", Theme::status_accent()),
        Span::styled("filter", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[4]);
}

// ── Command palette ───────────────────────────────────────────────────────────

pub(super) fn draw_command_palette_popup(f: &mut Frame, app: &App) {
    const PALETTE_MIN_W: u16 = 48;
    const PALETTE_MAX_W: u16 = 82;
    const PALETTE_MIN_H: u16 = 14;
    const PALETTE_MAX_H: u16 = 22;
    const TITLE_COL_W: usize = 24;
    const SHORTCUT_COL_W: usize = 10;

    let area = f.area();
    let commands = app.navigation.filtered_command_palette_commands();
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
        &app.navigation.command_palette_filter,
        app.navigation.command_palette_filter.len(),
        avail,
    );
    let placeholder = if app.navigation.command_palette_filter.is_empty() {
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
    f.set_cursor_position((chunks[1].x + 2 + filter_cur as u16, chunks[1].y));

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
                let selected = i == app.navigation.command_palette_cursor;
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
        app.navigation
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
    let (theme_filter_display, theme_filter_cur) = scroll_input(
        &app.navigation.theme_filter,
        app.navigation.theme_filter.len(),
        avail,
    );
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
    let filtered = app.navigation.filtered_themes(all_themes);

    let current_idx = Theme::current_index();
    let list_w = chunks[3].width as usize;

    let items: Vec<ListItem> = filtered
        .iter()
        .enumerate()
        .map(|(i, (orig_idx, t))| {
            build_theme_list_item(
                t,
                *orig_idx,
                current_idx,
                list_w,
                i == app.navigation.theme_cursor,
            )
        })
        .collect();

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let visible_rows = chunks[3].height as usize;
    let offset = app
        .navigation
        .theme_cursor
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(Some(app.navigation.theme_cursor));
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
    let (log_filter_display, log_filter_cur) = scroll_input(
        &app.diagnostics.log_filter,
        app.diagnostics.log_filter.len(),
        avail,
    );
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
            format!("{}+", app.diagnostics.log_level_filter.label()),
            popup_log_level_style(app.diagnostics.log_level_filter),
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
    let selected = (!filtered.is_empty()).then_some(
        app.diagnostics
            .log_cursor
            .min(filtered.len().saturating_sub(1)),
    );
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
                ("C-p", "command palette"),
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
                ("p", "profile selector"),
                ("j", "jump to parent session"),
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
                ("C-p", "open command palette from anywhere"),
                ("\u{2191} \u{2193}", "navigate"),
                ("Enter", "confirm"),
                ("Esc", "close"),
                ("type", "filter"),
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
                ("/profile [q|id]", "set profile for new sessions"),
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
    let mut state = ListState::default().with_offset(app.navigation.help_scroll);
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
