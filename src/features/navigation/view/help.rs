use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListItem, ListState, Paragraph},
};

use crate::navigation_state::NavigationState;
use crate::theme::Theme;
use crate::view_shared::centered_rect;

const ARROW_UP: &str = "\u{2191}";
const ARROW_DOWN: &str = "\u{2193}";

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

pub(crate) fn draw_help_popup(f: &mut Frame, navigation: &NavigationState) {
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
    let mut state = ListState::default().with_offset(navigation.help_scroll);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every section must have at least one row.
    #[test]
    fn shortcut_sections_all_sections_nonempty() {
        for section in shortcut_sections() {
            assert!(
                !section.rows.is_empty(),
                "section '{}' has no rows",
                section.title
            );
        }
    }

    /// The chat section must contain the 'Ctrl+t' thinking-level cycling entry.
    #[test]
    fn shortcut_sections_chat_contains_ctrl_t_thinking_cycle() {
        let chat = shortcut_sections()
            .iter()
            .find(|s| s.title == "chat")
            .expect("chat section missing");
        assert!(
            chat.rows.iter().any(|&(k, _)| k == "Ctrl+t"),
            "chat section must have a 'Ctrl+t' row for cycling thinking level"
        );
    }

    /// The chord section must contain the '?' help entry.
    #[test]
    fn shortcut_sections_chord_contains_help_entry() {
        let chord = shortcut_sections()
            .iter()
            .find(|s| s.title.contains("chord"))
            .expect("chord section missing");
        assert!(
            chord.rows.iter().any(|&(k, _)| k == "?"),
            "chord section must have a '?' row"
        );
    }

    #[test]
    fn shortcut_sections_chord_contains_external_editor_entry() {
        let chord = shortcut_sections()
            .iter()
            .find(|s| s.title.contains("chord"))
            .expect("chord section missing");
        assert!(
            chord
                .rows
                .iter()
                .any(|&(key, desc)| key == "e" && desc == "external editor")
        );
    }

    #[test]
    fn shortcut_sections_chord_contains_logs_entry() {
        let chord = shortcut_sections()
            .iter()
            .find(|s| s.title.contains("chord"))
            .expect("chord section missing");
        assert!(
            chord
                .rows
                .iter()
                .any(|&(key, desc)| key == "l" && desc == "logs popup")
        );
    }

    /// Every section title must be unique.
    #[test]
    fn shortcut_sections_titles_are_unique() {
        let titles: Vec<_> = shortcut_sections().iter().map(|s| s.title).collect();
        let unique: std::collections::HashSet<_> = titles.iter().copied().collect();
        assert_eq!(titles.len(), unique.len(), "duplicate section titles found");
    }

    /// No key string within a single section appears more than once.
    #[test]
    fn shortcut_sections_no_duplicate_keys_within_section() {
        for section in shortcut_sections() {
            let keys: Vec<_> = section.rows.iter().map(|&(k, _)| k).collect();
            let unique: std::collections::HashSet<_> = keys.iter().copied().collect();
            assert_eq!(
                keys.len(),
                unique.len(),
                "section '{}' has duplicate key entries",
                section.title
            );
        }
    }

    /// The global section contains the chord prefix entry.
    #[test]
    fn shortcut_sections_global_has_chord_prefix() {
        let global = shortcut_sections()
            .iter()
            .find(|s| s.title == "global")
            .expect("global section missing");
        assert!(
            global.rows.iter().any(|&(_, desc)| desc.contains("chord")),
            "global section should document the chord prefix"
        );
    }

    /// The chat section documents the @ mention shortcut.
    #[test]
    fn shortcut_sections_chat_has_mention() {
        let chat = shortcut_sections()
            .iter()
            .find(|s| s.title == "chat")
            .expect("chat section missing");
        assert!(
            chat.rows.iter().any(|&(k, _)| k == "@"),
            "chat section must document @ mention"
        );
    }
}
