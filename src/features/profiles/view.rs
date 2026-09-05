use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Cell, Clear, Paragraph, Row, Table, TableState},
};
use unicode_width::UnicodeWidthStr;

use crate::domain::profile::ProfileInfo;
use crate::profiles_state::ProfilesState;
use crate::theme::Theme;
use crate::view_shared::{popup_rect, scroll_input, truncate_with_ellipsis};

pub(crate) struct ProfilePopupInput<'a> {
    pub(crate) profiles: &'a ProfilesState,
    pub(crate) current_session_profile_id: Option<&'a str>,
}

fn profile_source_symbol(profile: &ProfileInfo) -> &'static str {
    match profile
        .source
        .as_deref()
        .filter(|source| !source.is_empty())
    {
        Some(source) if source.starts_with("local:") => "L",
        Some(source) if source.starts_with("embedded:") => "E",
        Some(source)
            if source.starts_with("remote:")
                || source.starts_with("http://")
                || source.starts_with("https://") =>
        {
            "R"
        }
        Some(_) => "?",
        None => " ",
    }
}

fn profile_empty_text(profiles: &ProfilesState) -> &'static str {
    if profiles.profiles.is_empty() {
        "No profiles available"
    } else {
        "No profiles match filter"
    }
}

fn profile_display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn profile_table_widths(
    table_w: u16,
    name_max_w: usize,
    kind_max_w: usize,
) -> (usize, usize, usize) {
    const SOURCE_W: usize = 2;
    const COLUMN_SPACING: usize = 3;
    const KIND_CAP: usize = 16;
    const NAME_CAP: usize = 32;

    let content_w = (table_w as usize).saturating_sub(SOURCE_W + COLUMN_SPACING);
    if content_w == 0 {
        return (0, 0, 0);
    }

    // Keep at least one column for description; name/kind shrink and ellipsize first.
    let fixed_budget = content_w.saturating_sub(1);
    let kind_w = kind_max_w.min(KIND_CAP).min(fixed_budget);
    let remaining_after_kind = fixed_budget.saturating_sub(kind_w);
    let name_cap = NAME_CAP.min((content_w / 2).max(8));
    let mut name_w = name_max_w.min(name_cap).min(remaining_after_kind);
    if name_max_w > 0 && name_w == 0 && fixed_budget > kind_w {
        name_w = 1;
    }

    let desc_w = content_w.saturating_sub(kind_w).saturating_sub(name_w);
    (name_w, kind_w, desc_w)
}

pub(crate) fn draw_profile_popup(f: &mut Frame, input: ProfilePopupInput<'_>) {
    let area = f.area();
    let popup_area = popup_rect(
        area,
        area.width.saturating_sub(4) as usize,
        area.height.saturating_sub(4) as usize,
        32..=76,
        6..=16,
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
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Span::styled("profile", Theme::popup_title())).style(Theme::popup_bg()),
        chunks[0],
    );
    let avail = chunks[1].width.saturating_sub(2) as usize;
    let (filter_display, filter_cur) = scroll_input(
        &input.profiles.profile_filter,
        input.profiles.profile_filter.len(),
        avail,
    );
    let input_line = Line::from(vec![
        Span::styled("> ", Theme::popup_title()),
        Span::styled(filter_display, Theme::popup_bg()),
    ]);
    f.render_widget(
        Paragraph::new(input_line).style(Theme::popup_bg()),
        chunks[1],
    );
    if chunks[1].width > 2 && chunks[1].height > 0 {
        f.set_cursor_position((
            chunks[1]
                .x
                .saturating_add(2)
                .saturating_add(filter_cur as u16),
            chunks[1].y,
        ));
    }

    let active_id = input.profiles.active_profile_id.as_deref();
    let current_session_id = input.current_session_profile_id;
    let profiles = input.profiles.filtered_profiles();
    let has_matches = !profiles.is_empty();
    let row_data: Vec<(String, String, String, String)> = profiles
        .into_iter()
        .map(|profile| {
            let marker = if Some(profile.id.as_str()) == active_id {
                "*"
            } else if Some(profile.id.as_str()) == current_session_id {
                "@"
            } else {
                " "
            };
            let source = format!("{marker}{}", profile_source_symbol(profile));
            let kind = profile
                .config_kind
                .as_deref()
                .unwrap_or_default()
                .to_string();
            let description = profile
                .description
                .as_deref()
                .unwrap_or_default()
                .to_string();

            (source, profile.name.clone(), kind, description)
        })
        .collect();
    let name_max_w = row_data
        .iter()
        .map(|(_, name, _, _)| profile_display_width(name))
        .max()
        .unwrap_or_else(|| profile_display_width(profile_empty_text(input.profiles)));
    let kind_max_w = row_data
        .iter()
        .map(|(_, _, kind, _)| profile_display_width(kind))
        .max()
        .unwrap_or(0);
    let (name_w, kind_w, desc_w) = profile_table_widths(chunks[2].width, name_max_w, kind_max_w);
    let rows: Vec<Row> = if !has_matches {
        vec![Row::new(vec![
            Cell::from(Span::styled(" ", Theme::status())),
            Cell::from(Span::styled(
                truncate_with_ellipsis(profile_empty_text(input.profiles), name_w),
                Theme::status(),
            )),
            Cell::from(""),
            Cell::from(""),
        ])]
    } else {
        row_data
            .into_iter()
            .map(|(source, name, kind, description)| {
                Row::new(vec![
                    Cell::from(Span::styled(source, Theme::status_accent())),
                    Cell::from(Span::styled(
                        truncate_with_ellipsis(&name, name_w),
                        Theme::popup_bg(),
                    )),
                    Cell::from(Span::styled(
                        truncate_with_ellipsis(&kind, kind_w),
                        Theme::profile_kind(&kind),
                    )),
                    Cell::from(Span::styled(
                        truncate_with_ellipsis(&description, desc_w),
                        Theme::status(),
                    )),
                ])
            })
            .collect()
    };
    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Length(name_w as u16),
            Constraint::Length(kind_w as u16),
            Constraint::Length(desc_w as u16),
        ],
    )
    .block(Block::default().style(Theme::popup_bg()))
    .style(Theme::popup_bg())
    .row_highlight_style(Theme::selected());
    let selected = has_matches.then_some(input.profiles.profile_cursor);
    let mut state = TableState::default().with_selected(selected);
    f.render_stateful_widget(table, chunks[2], &mut state);

    let hint = Line::from(vec![
        Span::styled("enter ", Theme::status_accent()),
        Span::styled("set for new sessions  ", Theme::status()),
        Span::styled("esc ", Theme::status_accent()),
        Span::styled("close", Theme::status()),
    ]);
    f.render_widget(Paragraph::new(hint).style(Theme::popup_bg()), chunks[3]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(id: &str, name: &str) -> ProfileInfo {
        ProfileInfo {
            id: id.into(),
            name: name.into(),
            ..Default::default()
        }
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
                return Some((line[..byte_idx].chars().count() as u16, y));
            }
        }
        None
    }

    #[test]
    fn profile_popup_marks_active_before_current_and_highlights_filtered_row() {
        let mut profiles = ProfilesState::new();
        profiles.profiles = vec![
            ProfileInfo {
                id: "active".into(),
                name: "Active".into(),
                description: Some("selected".into()),
                ..Default::default()
            },
            profile("other", "Other"),
        ];
        profiles.active_profile_id = Some("active".into());
        profiles.profile_filter = "selected".into();

        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                draw_profile_popup(
                    frame,
                    ProfilePopupInput {
                        profiles: &profiles,
                        current_session_profile_id: Some("active"),
                    },
                )
            })
            .unwrap();
        let buffer = terminal.backend().buffer();

        let active = find_buffer_text(buffer, "Active").expect("active row");
        let active_line = buffer_line(buffer, active.1);
        assert!(active_line.contains('*'));
        assert!(!active_line.contains('@'));
        assert_eq!(buffer[active].bg, Theme::selected().bg.unwrap());
        assert!(find_buffer_text(buffer, "Other").is_none());
    }

    #[test]
    fn profile_popup_marks_distinct_current_session_with_at_symbol() {
        let mut profiles = ProfilesState::new();
        profiles.profiles = vec![profile("active", "Active"), profile("current", "Current")];
        profiles.active_profile_id = Some("active".into());

        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                draw_profile_popup(
                    frame,
                    ProfilePopupInput {
                        profiles: &profiles,
                        current_session_profile_id: Some("current"),
                    },
                )
            })
            .unwrap();
        let buffer = terminal.backend().buffer();

        let active_y = find_buffer_text(buffer, "Active").unwrap().1;
        let current_y = find_buffer_text(buffer, "Current").unwrap().1;
        assert!(buffer_line(buffer, active_y).contains('*'));
        assert!(buffer_line(buffer, current_y).contains('@'));
    }

    #[test]
    fn profile_source_symbol_maps_known_sources() {
        let mut profile = profile("fast", "Fast");
        assert_eq!(profile_source_symbol(&profile), " ");

        profile.source = Some("local:/tmp/profile.toml".into());
        assert_eq!(profile_source_symbol(&profile), "L");

        profile.source = Some("embedded:default".into());
        assert_eq!(profile_source_symbol(&profile), "E");

        profile.source = Some("remote:team/default".into());
        assert_eq!(profile_source_symbol(&profile), "R");

        profile.source = Some("https://example.com/profile.toml".into());
        assert_eq!(profile_source_symbol(&profile), "R");

        profile.source = Some("workspace".into());
        assert_eq!(profile_source_symbol(&profile), "?");
    }

    #[test]
    fn profile_empty_text_distinguishes_empty_and_no_match() {
        let mut profiles = ProfilesState::new();
        assert_eq!(profile_empty_text(&profiles), "No profiles available");

        profiles.profiles = vec![profile("fast", "Fast")];
        profiles.profile_filter = "missing".into();

        assert!(profiles.filtered_profiles().is_empty());
        assert_eq!(profile_empty_text(&profiles), "No profiles match filter");
    }

    #[test]
    fn profile_table_widths_are_dynamic_and_leave_description_space() {
        let (name_w, kind_w, desc_w) = profile_table_widths(58, 12, 5);
        assert_eq!((name_w, kind_w, desc_w), (12, 5, 36));

        let (name_w, kind_w, desc_w) = profile_table_widths(58, 80, 40);
        assert_eq!(kind_w, 16);
        assert!(name_w < 32);
        assert!(desc_w > 1);

        let (name_w, kind_w, desc_w) = profile_table_widths(8, 20, 20);
        assert_eq!(name_w + kind_w + desc_w, 3);
        assert_eq!(desc_w, 1);
    }
}
