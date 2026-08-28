use ratatui::{
    Frame,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, List, ListItem, ListState},
};

use crate::composer_state::ComposerState;
use crate::theme::Theme;

pub(crate) fn completion_panel_height(composer: &ComposerState) -> u16 {
    if composer.slash_state.is_some()
        || composer.mention_state.is_some()
        || composer.file_index_loading
        || composer.file_index_error.is_some()
    {
        6
    } else {
        0
    }
}

pub(crate) fn draw_completion_panel(
    f: &mut Frame,
    composer: &ComposerState,
    spinner_frame: &str,
    area: Rect,
) {
    if composer.slash_state.is_some() {
        draw_slash_panel(f, composer, area);
    } else {
        draw_mention_panel(f, composer, spinner_frame, area);
    }
}

fn draw_slash_panel(f: &mut Frame, composer: &ComposerState, area: Rect) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let Some(state) = &composer.slash_state else {
        return;
    };

    let max_name_len = state
        .results
        .iter()
        .map(|cmd| cmd.name.len())
        .max()
        .unwrap_or(0);

    let items: Vec<ListItem> = state
        .results
        .iter()
        .map(|cmd| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("  /{:<width$}  ", cmd.name, width = max_name_len),
                    Theme::status_accent(),
                ),
                Span::styled(cmd.description, Theme::dim()),
            ]))
        })
        .collect();

    let title = Line::from(vec![
        Span::styled(" /", Theme::status_accent()),
        Span::styled(" commands ", Theme::dim()),
    ]);
    let list = List::new(items)
        .block(Block::default().title(title).style(Theme::popup_bg()))
        .highlight_style(Theme::selected())
        .highlight_symbol("");
    let mut list_state = ListState::default().with_selected(Some(state.selected_index));
    f.render_stateful_widget(list, area, &mut list_state);
}

fn draw_mention_panel(f: &mut Frame, composer: &ComposerState, spinner_frame: &str, area: Rect) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let mut items: Vec<ListItem> = Vec::new();
    if composer.file_index_loading && composer.file_index.is_empty() {
        items.push(ListItem::new(Line::from(vec![Span::styled(
            format!("{spinner_frame} indexing files"),
            Theme::thinking(),
        )])));
    } else if let Some(error) = &composer.file_index_error {
        items.push(ListItem::new(Line::from(vec![Span::styled(
            format!("file index error: {error}"),
            Theme::error_text(),
        )])));
    } else if let Some(mention) = &composer.mention_state {
        if mention.results.is_empty() {
            items.push(ListItem::new(Line::from(vec![Span::styled(
                format!("no matches for @{}", mention.query),
                Theme::info_text(),
            )])));
        } else {
            for entry in &mention.results {
                let icon = if entry.is_dir { "[D]" } else { "[F]" };
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(format!("{icon} "), Theme::status()),
                    Span::styled(entry.path.clone(), Theme::input()),
                ])));
            }
        }
    }

    if items.is_empty() {
        return;
    }

    let title = if let Some(mention) = &composer.mention_state {
        format!(" @ files - {} ", mention.query)
    } else {
        " @ files ".into()
    };
    let list = List::new(items)
        .block(Block::default().title(title).style(Theme::popup_bg()))
        .highlight_style(Theme::selected())
        .highlight_symbol("");
    let selected = composer
        .mention_state
        .as_ref()
        .map(|mention| mention.selected_index)
        .filter(|_| !composer.file_index_loading);
    let mut state = ListState::default().with_selected(selected);
    f.render_stateful_widget(list, area, &mut state);
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::composer_state::{FileIndexEntryLite, MentionState, SlashCompletionState};
    use crate::slash::SLASH_COMMANDS;

    fn render(composer: &ComposerState) -> ratatui::buffer::Buffer {
        render_at_width(composer, 60)
    }

    fn render_at_width(composer: &ComposerState, width: u16) -> ratatui::buffer::Buffer {
        let backend = TestBackend::new(width, 6);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| draw_completion_panel(frame, composer, "spinner", frame.area()))
            .unwrap();
        terminal.backend().buffer().clone()
    }

    fn find_text(buffer: &ratatui::buffer::Buffer, needle: &str) -> (u16, u16) {
        for y in 0..buffer.area.height {
            let line = (0..buffer.area.width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>();
            if let Some(byte_index) = line.find(needle) {
                return (line[..byte_index].chars().count() as u16, y);
            }
        }
        panic!("missing {needle:?}");
    }

    #[test]
    fn slash_and_mention_panels_preserve_rows_selection_and_precedence() {
        let mut composer = ComposerState::new();
        composer.slash_state = Some(SlashCompletionState {
            query: "mo".into(),
            selected_index: 1,
            results: vec![&SLASH_COMMANDS[0], &SLASH_COMMANDS[1]],
        });
        composer.mention_state = Some(MentionState {
            trigger_start: 0,
            query: "ignored".into(),
            selected_index: 0,
            results: vec![FileIndexEntryLite {
                path: "mention-must-not-render".into(),
                is_dir: false,
            }],
        });

        let slash = render(&composer);
        assert_eq!(completion_panel_height(&composer), 6);
        let selected_bg = Theme::selected()
            .bg
            .expect("selected style has a background");
        let mode = find_text(&slash, "switch mode");
        assert_eq!(slash[(0, mode.1)].bg, selected_bg);
        assert_ne!(
            slash[(0, find_text(&slash, "model selector").1)].bg,
            selected_bg
        );
        assert!(find_text(&slash, "model selector").1 < find_text(&slash, "switch mode").1);
        assert!(
            !slash
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>()
                .contains("mention-must-not-render")
        );

        composer.slash_state = None;
        composer.mention_state = Some(MentionState {
            trigger_start: 0,
            query: "src".into(),
            selected_index: 1,
            results: vec![
                FileIndexEntryLite {
                    path: "src".into(),
                    is_dir: true,
                },
                FileIndexEntryLite {
                    path: "src/main.rs".into(),
                    is_dir: false,
                },
            ],
        });

        let mention = render(&composer);
        let main = find_text(&mention, "src/main.rs");
        assert_eq!(mention[(0, main.1)].bg, selected_bg);
        let directory_row = find_text(&mention, "[D]").1;
        let directory_line = (0..mention.area.width)
            .map(|x| mention[(x, directory_row)].symbol())
            .collect::<String>();
        assert_ne!(mention[(0, directory_row)].bg, selected_bg);
        assert!(directory_line.contains("[D] src"));
        assert_eq!(find_text(&mention, "[F]").1, main.1);
    }

    #[test]
    fn narrow_panels_clip_long_rows_and_keep_filtered_empty_states() {
        let mut composer = ComposerState::new();
        composer.mention_state = Some(MentionState {
            trigger_start: 0,
            query: "src".into(),
            selected_index: 0,
            results: vec![FileIndexEntryLite {
                path: "src/a-deliberately-long-file-name.rs".into(),
                is_dir: false,
            }],
        });

        let clipped = render_at_width(&composer, 18)
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(clipped.contains("[F] src/a-deliber"));
        assert!(!clipped.contains("long-file-name"));

        composer.mention_state = None;
        composer.slash_state = Some(SlashCompletionState {
            query: "missing".into(),
            selected_index: 0,
            results: Vec::new(),
        });
        let filtered = render(&composer);
        find_text(&filtered, "/ commands");
        assert_eq!(completion_panel_height(&composer), 6);
    }

    #[test]
    fn mention_panel_preserves_empty_loading_and_error_text() {
        let mut composer = ComposerState::new();
        composer.mention_state = Some(MentionState {
            trigger_start: 0,
            query: "missing".into(),
            selected_index: 0,
            results: Vec::new(),
        });
        let empty = render(&composer);
        find_text(&empty, "no matches for @missing");

        composer.file_index_loading = true;
        let loading = render(&composer);
        find_text(&loading, "spinner indexing files");

        composer.file_index_loading = false;
        composer.file_index_error = Some("unavailable".into());
        let error = render(&composer);
        find_text(&error, "file index error: unavailable");

        composer.mention_state = None;
        composer.file_index_error = None;
        assert_eq!(completion_panel_height(&composer), 0);
    }
}
