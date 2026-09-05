use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Modifier,
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::domain::model::ModelEntry;
use crate::models_state::{ModelPopupItem, ModelsState};
use crate::theme::Theme;
use crate::view_shared::{ELLIPSIS, popup_rect, scroll_input};

const MODEL_GROUP_MARKER: &str = "\u{25B8}";

pub(crate) struct ModelPopupInput<'a> {
    pub(crate) models: &'a ModelsState,
    pub(crate) delegate_preference_profile_id: Option<&'a str>,
}

/// Whether the given model matches a delegate agent's preferred model.
fn popup_delegate_marker(
    models: &ModelsState,
    profile_id: Option<&str>,
    model: &ModelEntry,
    agent_id: &str,
) -> bool {
    profile_id
        .and_then(|profile_id| models.get_delegate_model_preference(profile_id, agent_id))
        .is_some_and(|preference| {
            preference.model_id == model.id && preference.node_id == model.node_id
        })
}

pub(crate) fn draw_model_popup(f: &mut Frame, input: ModelPopupInput<'_>) {
    const MODEL_MARKER_COL_W: u16 = 4;
    const MODEL_LABEL_MAX_W: u16 = 48;
    const MODEL_POPUP_MAX_W: u16 = MODEL_MARKER_COL_W + MODEL_LABEL_MAX_W + 2;
    const MODEL_POPUP_MIN_W: u16 = 30;

    let has_tabs = input.models.model_popup_has_tabs();

    let area = f.area();
    let popup_area = popup_rect(
        area,
        area.width.saturating_sub(4) as usize,
        (area.height as usize).saturating_mul(60) / 100,
        MODEL_POPUP_MIN_W as usize..=MODEL_POPUP_MAX_W as usize,
        0..=area.height as usize,
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
        for i in 0..input.models.model_popup_tab_count() {
            let label = input.models.model_popup_tab_label(i);
            let is_active = i == input.models.model_popup_agent_tab;
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
            chunks[ti],
        );
    }

    // Filter input
    let filter_area = chunks[filter_idx];
    let avail = filter_area.width.saturating_sub(2) as usize;
    let (model_filter_display, model_filter_cur) = scroll_input(
        &input.models.model_filter,
        input.models.model_filter.len(),
        avail,
    );
    let filter_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(model_filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(filter_line).style(Theme::popup_bg()),
        filter_area,
    );
    if filter_area.width > 2 && filter_area.height > 0 {
        f.set_cursor_position((
            filter_area
                .x
                .saturating_add(2)
                .saturating_add(model_filter_cur as u16),
            filter_area.y,
        ));
    }

    let on_session_tab = input
        .models
        .model_popup_is_session_tab(input.models.model_popup_agent_tab);
    let active_agent_id = input
        .models
        .model_popup_tab_agent_id(input.models.model_popup_agent_tab)
        .map(str::to_string);

    let list_area = chunks[list_idx];
    let list_w = list_area.width as usize;

    let items: Vec<ListItem> = input
        .models
        .visible_model_popup_items()
        .iter()
        .enumerate()
        .map(|(i, item)| match item {
            ModelPopupItem::ProviderHeader {
                provider,
                model_count,
                node_suffix,
            } => {
                let selected = i == input.models.model_cursor;
                let marker = MODEL_GROUP_MARKER;
                let count = format!(" {model_count}");
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
                let suffix_style = Theme::status_accent();

                let marker_w = 2usize;
                let count_w = count.chars().count();
                let suffix = node_suffix.as_ref().map(|n| format!("@ {n}"));
                let suffix_w = suffix.as_ref().map(|s| s.chars().count()).unwrap_or(0);
                let avail = list_w.saturating_sub(marker_w + 1 + count_w + suffix_w + 1);

                let provider_display = if provider.chars().count() > avail {
                    let t: String = provider.chars().take(avail.saturating_sub(1)).collect();
                    format!("{t}{ELLIPSIS}")
                } else {
                    provider.clone()
                };
                let gap = avail.saturating_sub(provider_display.chars().count());

                let mut line_spans = vec![
                    Span::styled(format!("{marker} "), marker_style),
                    Span::styled(provider_display, provider_style),
                    Span::styled(" ".repeat(gap), dim_style),
                ];
                if let Some(s) = suffix {
                    line_spans.push(Span::styled(s, suffix_style));
                }
                line_spans.push(Span::styled(count, dim_style));
                ListItem::new(Line::from(line_spans))
            }
            ModelPopupItem::Model { model_idx } => {
                let selected = i == input.models.model_cursor;
                let model = &input.models.models[*model_idx];

                let show_live =
                    on_session_tab && input.models.live_model_selection_matches_entry(model);
                let show_delegate = active_agent_id.as_deref().is_some_and(|aid| {
                    popup_delegate_marker(
                        input.models,
                        input.delegate_preference_profile_id,
                        model,
                        aid,
                    )
                });
                let marker_count = usize::from(show_live) + usize::from(show_delegate);

                let marker_bg = if selected {
                    Theme::bg_hl()
                } else {
                    Theme::bg_dim()
                };
                let marker_w = MODEL_MARKER_COL_W as usize;
                let avail = list_w.saturating_sub(marker_w);
                let base_label = model.label.clone();
                let label = if base_label.chars().count() > avail {
                    let t: String = base_label.chars().take(avail.saturating_sub(1)).collect();
                    format!("{t}{ELLIPSIS}")
                } else {
                    base_label
                };
                let gap = avail.saturating_sub(label.chars().count());
                let main_style = if selected {
                    Theme::selected()
                } else {
                    Theme::popup_bg()
                };
                let mut spans = Vec::with_capacity(4);
                spans.push(Span::styled(" ", main_style));
                if show_live {
                    spans.push(Span::styled(
                        "\u{25cf}",
                        Theme::status_accent().bg(marker_bg),
                    ));
                }
                if show_delegate {
                    spans.push(Span::styled(
                        "\u{25cf}",
                        Theme::status_accent()
                            .bg(marker_bg)
                            .add_modifier(Modifier::DIM),
                    ));
                }
                spans.push(Span::styled(
                    " ".repeat(marker_w.saturating_sub(1 + marker_count)),
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
    let offset = input
        .models
        .model_cursor
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(Some(input.models.model_cursor));
    f.render_stateful_widget(list, list_area, &mut state);

    let mut hint_spans = vec![
        Span::styled(" esc ", Theme::status_accent()),
        Span::styled("cancel  ", Theme::status()),
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("select", Theme::status()),
    ];
    if has_tabs {
        hint_spans.push(Span::styled("  tab ", Theme::status_accent()));
        hint_spans.push(Span::styled("agent", Theme::status()));
        if !on_session_tab {
            hint_spans.push(Span::styled("  del ", Theme::status_accent()));
            hint_spans.push(Span::styled("default", Theme::status()));
        }
    }
    f.render_widget(
        Paragraph::new(Line::from(hint_spans)).style(Theme::popup_bg()),
        chunks[hint_idx],
    );
}
