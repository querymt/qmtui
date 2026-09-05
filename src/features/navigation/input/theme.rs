use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::navigation_state::NavigationState;
use crate::themes_gen::Base16Palette;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ThemeInputResult {
    NotHandled,
    Close,
    Moved,
    Filtered,
    Apply { index: usize },
}

pub(crate) fn handle_key(
    navigation: &mut NavigationState,
    themes: &[Base16Palette],
    key: KeyEvent,
) -> ThemeInputResult {
    match key.code {
        KeyCode::Esc => ThemeInputResult::Close,
        KeyCode::Up => {
            navigation.move_theme_cursor_up();
            ThemeInputResult::Moved
        }
        KeyCode::Down => {
            navigation.move_theme_cursor_down(navigation.filtered_themes(themes).len());
            ThemeInputResult::Moved
        }
        KeyCode::Enter => navigation
            .selected_theme_index(themes)
            .map(|index| ThemeInputResult::Apply { index })
            .unwrap_or(ThemeInputResult::NotHandled),
        KeyCode::Backspace => {
            navigation.theme_filter_backspace();
            ThemeInputResult::Filtered
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            navigation.theme_filter_insert(character);
            ThemeInputResult::Filtered
        }
        _ => ThemeInputResult::NotHandled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THEMES: &[Base16Palette] = &[
        Base16Palette {
            id: "alpha",
            label: "Alpha",
            colors: [0; 16],
        },
        Base16Palette {
            id: "beta",
            label: "Beta",
            colors: [0; 16],
        },
    ];

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn movement_clamps_and_filter_edits_reset_the_cursor() {
        let mut navigation = NavigationState::new();
        assert_eq!(
            handle_key(&mut navigation, THEMES, key(KeyCode::Down)),
            ThemeInputResult::Moved
        );
        assert_eq!(navigation.theme_cursor, 1);
        handle_key(&mut navigation, THEMES, key(KeyCode::Down));
        assert_eq!(navigation.theme_cursor, 1);

        assert_eq!(
            handle_key(&mut navigation, THEMES, key(KeyCode::Char('b'))),
            ThemeInputResult::Filtered
        );
        assert_eq!(navigation.theme_filter, "b");
        assert_eq!(navigation.theme_cursor, 0);
        assert_eq!(
            handle_key(&mut navigation, THEMES, key(KeyCode::Backspace)),
            ThemeInputResult::Filtered
        );
        assert!(navigation.theme_filter.is_empty());
        assert_eq!(navigation.theme_cursor, 0);
    }

    #[test]
    fn enter_returns_original_theme_index_and_no_match_is_not_handled() {
        let mut navigation = NavigationState::new();
        navigation.theme_filter = "beta".into();
        assert_eq!(
            handle_key(&mut navigation, THEMES, key(KeyCode::Enter)),
            ThemeInputResult::Apply { index: 1 }
        );

        navigation.theme_filter = "missing".into();
        assert_eq!(
            handle_key(&mut navigation, THEMES, key(KeyCode::Enter)),
            ThemeInputResult::NotHandled
        );
    }

    #[test]
    fn empty_filter_preserves_out_of_range_apply_and_escape_closes() {
        let mut navigation = NavigationState::new();
        navigation.theme_cursor = THEMES.len() + 3;
        assert_eq!(
            handle_key(&mut navigation, THEMES, key(KeyCode::Enter)),
            ThemeInputResult::Apply {
                index: THEMES.len() + 3
            }
        );
        assert_eq!(
            handle_key(&mut navigation, THEMES, key(KeyCode::Esc)),
            ThemeInputResult::Close
        );
    }
}
