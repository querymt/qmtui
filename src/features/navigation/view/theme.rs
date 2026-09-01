use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::navigation_state::NavigationState;
use crate::theme::Theme;
use crate::view_shared::{ELLIPSIS, popup_rect, scroll_input};

pub(crate) const COLOR_SWATCH: &str = "\u{25A0}";

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
    let marker_w = UnicodeWidthStr::width(marker);
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
    let label = t.label;
    let label_display = if UnicodeWidthStr::width(label) > avail {
        let text_width = avail.saturating_sub(UnicodeWidthStr::width(ELLIPSIS));
        let mut used = 0usize;
        let truncated: String = label
            .chars()
            .take_while(|character| {
                let width = UnicodeWidthChar::width(*character).unwrap_or(0).max(1);
                let fits = used.saturating_add(width) <= text_width;
                if fits {
                    used = used.saturating_add(width);
                }
                fits
            })
            .collect();
        format!("{truncated}{ELLIPSIS}")
    } else {
        label.to_string()
    };
    let label_gap = avail.saturating_sub(UnicodeWidthStr::width(label_display.as_str()));

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

pub(crate) fn draw_theme_popup(f: &mut Frame, navigation: &NavigationState) {
    const THEME_MARKER_COL_W: u16 = 2;
    const THEME_LABEL_MAX_W: u16 = 44;
    const THEME_SWATCH_COL_W: u16 = 17;
    const THEME_ROW_MAX_W: u16 = THEME_MARKER_COL_W + THEME_LABEL_MAX_W + THEME_SWATCH_COL_W;
    const THEME_POPUP_MAX_W: u16 = THEME_ROW_MAX_W + 2;
    const THEME_POPUP_MIN_W: u16 = 28;

    let area = f.area();
    let popup_area = popup_rect(
        area,
        area.width.saturating_sub(4) as usize,
        (area.height as usize).saturating_mul(60) / 100,
        THEME_POPUP_MIN_W as usize..=THEME_POPUP_MAX_W as usize,
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
        &navigation.theme_filter,
        navigation.theme_filter.len(),
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
    if chunks[1].width > 2 && chunks[1].height > 0 {
        f.set_cursor_position((
            chunks[1]
                .x
                .saturating_add(2)
                .saturating_add(theme_filter_cur as u16),
            chunks[1].y,
        ));
    }

    // theme list
    let all_themes = Theme::available_themes();
    let filtered = navigation.filtered_themes(all_themes);

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
                i == navigation.theme_cursor,
            )
        })
        .collect();

    let list = List::new(items).block(Block::default().style(Theme::popup_bg()));
    let visible_rows = chunks[3].height as usize;
    let offset = navigation
        .theme_cursor
        .saturating_sub(visible_rows.saturating_sub(1));
    let mut state = ListState::default()
        .with_offset(offset)
        .with_selected(Some(navigation.theme_cursor));
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

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn buffer_line(buffer: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buffer.area.width)
            .map(|x| buffer[(x, y)].symbol())
            .collect::<String>()
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
    fn draw_theme_popup_highlights_hint_keys() {
        let navigation = NavigationState::new();

        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| draw_theme_popup(f, &navigation)).unwrap();
        let buffer = terminal.backend().buffer().clone();

        let (esc_x, esc_y) = find_buffer_text(&buffer, "esc").expect("esc hint missing");
        let (cancel_x, cancel_y) =
            find_buffer_text(&buffer, "cancel").expect("cancel hint missing");
        let (enter_x, enter_y) = find_buffer_text(&buffer, "enter").expect("enter hint missing");
        let (apply_x, apply_y) = find_buffer_text(&buffer, "apply").expect("apply hint missing");

        assert_eq!(esc_y, cancel_y);
        assert_eq!(enter_y, apply_y);
        assert_eq!(buffer[(esc_x, esc_y)].fg, buffer[(enter_x, enter_y)].fg);
        assert_ne!(buffer[(esc_x, esc_y)].fg, buffer[(cancel_x, cancel_y)].fg);
        assert_ne!(buffer[(enter_x, enter_y)].fg, buffer[(apply_x, apply_y)].fg);
    }

    #[test]
    fn draw_theme_popup_truncates_long_labels_with_unicode_ellipsis() {
        let mut navigation = NavigationState::new();
        navigation.theme_filter = "Penumbra Dark Contrast Plus Plus".into();

        let backend = ratatui::backend::TestBackend::new(40, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| draw_theme_popup(f, &navigation)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let rendered = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(rendered.contains('…'));
        assert!(!rendered.contains("..."));
    }

    #[test]
    fn draw_theme_popup_uses_bounded_width_on_wide_terminal() {
        let navigation = NavigationState::new();

        let backend = ratatui::backend::TestBackend::new(120, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| draw_theme_popup(f, &navigation)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let (theme_x, _) = find_buffer_text(&buffer, "theme").expect("theme title missing");

        assert!(
            theme_x >= 28,
            "theme popup should be centered and use a bounded width"
        );
    }

    #[test]
    #[serial]
    fn draw_theme_popup_shows_current_theme_when_opened_on_short_terminal() {
        crate::theme::Theme::set_by_index(20);
        crate::theme::Theme::begin_frame();

        let mut navigation = NavigationState::new();
        navigation.theme_cursor = 20;

        let current_label = crate::theme::Theme::available_themes()[20].label;
        let backend = ratatui::backend::TestBackend::new(80, 10);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| draw_theme_popup(f, &navigation)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let rendered = buffer
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(
            rendered.contains(current_label),
            "current theme should be visible when popup opens"
        );

        crate::theme::Theme::set_by_index(0);
        crate::theme::Theme::begin_frame();
    }

    #[test]
    #[serial]
    fn draw_theme_popup_highlights_current_theme_marker_with_accent() {
        crate::theme::Theme::set_by_index(20);
        crate::theme::Theme::begin_frame();

        let mut navigation = NavigationState::new();
        navigation.theme_cursor = 20;

        let current_label = crate::theme::Theme::available_themes()[20].label;
        let backend = ratatui::backend::TestBackend::new(80, 10);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| draw_theme_popup(f, &navigation)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let (label_x, label_y) =
            find_buffer_text(&buffer, current_label).expect("current theme label missing");
        let marker_x = label_x.saturating_sub(2);

        assert_eq!(buffer[(marker_x, label_y)].symbol(), "●");
        assert_eq!(
            buffer[(marker_x, label_y)].fg,
            crate::theme::Theme::status_accent().fg.expect("accent fg")
        );
        assert_eq!(buffer[(marker_x, label_y)].bg, crate::theme::Theme::bg_hl());

        crate::theme::Theme::set_by_index(0);
        crate::theme::Theme::begin_frame();
    }

    /// Helper: build a minimal fake palette with all 16 colours set to the
    /// given value so tests can assert on specific fg colours.
    fn fake_palette(colors: [u32; 16]) -> crate::themes_gen::Base16Palette {
        crate::themes_gen::Base16Palette {
            id: "test-id",
            label: "Test Theme",
            colors,
        }
    }

    fn render_theme_item(
        palette: &crate::themes_gen::Base16Palette,
        orig_idx: usize,
        current_idx: usize,
        width: u16,
        selected: bool,
    ) -> ratatui::buffer::Buffer {
        let backend = ratatui::backend::TestBackend::new(width, 1);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let item =
                    build_theme_list_item(palette, orig_idx, current_idx, width as usize, selected);
                frame.render_widget(List::new(vec![item]), frame.area());
            })
            .unwrap();
        terminal.backend().buffer().clone()
    }

    #[test]
    fn rendered_theme_item_has_marker_label_and_sixteen_palette_swatches() {
        crate::theme::Theme::begin_frame();
        let colors = [
            0xff0000, 0x00ff00, 0x0000ff, 0xffffff, 0x000000, 0x111111, 0x222222, 0x333333,
            0x444444, 0x555555, 0x666666, 0x777777, 0x888888, 0x999999, 0xaaaaaa, 0xbbbbbb,
        ];
        let palette = fake_palette(colors);
        let buffer = render_theme_item(&palette, 3, 3, 40, true);

        assert_eq!(buffer[(0, 0)].symbol(), "●");
        assert_eq!(buffer[(0, 0)].fg, Theme::status_accent().fg.unwrap());
        assert_eq!(buffer[(0, 0)].bg, Theme::bg_hl());
        assert_eq!(find_buffer_text(&buffer, "Test Theme"), Some((2, 0)));
        for (index, color) in colors.into_iter().enumerate() {
            let x = 24 + index as u16;
            assert_eq!(buffer[(x, 0)].symbol(), COLOR_SWATCH);
            assert_eq!(buffer[(x, 0)].fg, crate::theme::u32_to_color(color));
            assert_eq!(buffer[(x, 0)].bg, Theme::bg_hl());
        }
    }

    #[test]
    fn rendered_theme_item_inactive_marker_and_styles_are_visible() {
        crate::theme::Theme::begin_frame();
        let palette = fake_palette([0x123456; 16]);
        let buffer = render_theme_item(&palette, 5, 3, 40, false);

        assert_eq!(buffer[(0, 0)].symbol(), " ");
        assert_eq!(buffer[(1, 0)].symbol(), " ");
        assert_eq!(buffer[(2, 0)].fg, Theme::popup_bg().fg.unwrap());
        assert_eq!(buffer[(24, 0)].symbol(), COLOR_SWATCH);
        assert_eq!(
            buffer[(24, 0)].fg,
            ratatui::style::Color::Rgb(0x12, 0x34, 0x56)
        );
        assert_eq!(buffer[(24, 0)].bg, Theme::bg_dim());
    }

    #[test]
    fn rendered_theme_item_truncates_unicode_by_display_width() {
        crate::theme::Theme::begin_frame();
        let palette = crate::themes_gen::Base16Palette {
            id: "t",
            label: "界界界界 Theme",
            colors: [0; 16],
        };
        let buffer = render_theme_item(&palette, 0, 99, 24, false);

        assert_eq!(buffer[(2, 0)].symbol(), "界");
        assert_eq!(buffer[(4, 0)].symbol(), "界");
        assert_eq!(buffer[(6, 0)].symbol(), ELLIPSIS);
        assert_eq!(buffer[(8, 0)].symbol(), COLOR_SWATCH);
        assert_eq!(buffer[(23, 0)].symbol(), COLOR_SWATCH);
    }

    /// u32_to_color converts RGB u32 correctly for all byte boundaries.
    #[test]
    fn u32_to_color_correct_rgb_extraction() {
        use ratatui::style::Color;
        assert_eq!(crate::theme::u32_to_color(0x000000), Color::Rgb(0, 0, 0));
        assert_eq!(
            crate::theme::u32_to_color(0xffffff),
            Color::Rgb(255, 255, 255)
        );
        assert_eq!(crate::theme::u32_to_color(0xff0000), Color::Rgb(255, 0, 0));
        assert_eq!(crate::theme::u32_to_color(0x00ff00), Color::Rgb(0, 255, 0));
        assert_eq!(crate::theme::u32_to_color(0x0000ff), Color::Rgb(0, 0, 255));
        assert_eq!(
            crate::theme::u32_to_color(0xaabbcc),
            Color::Rgb(0xaa, 0xbb, 0xcc)
        );
    }
}
