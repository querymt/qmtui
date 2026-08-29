use crossterm::event::{KeyCode, KeyEvent};

use crate::navigation_state::NavigationState;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum HelpInputResult {
    NotHandled,
    Close,
    Scroll { delta: isize },
}

pub(crate) fn handle_key(navigation: &mut NavigationState, key: KeyEvent) -> HelpInputResult {
    match key.code {
        KeyCode::Esc => HelpInputResult::Close,
        KeyCode::Up => {
            navigation.scroll_help_up();
            HelpInputResult::Scroll { delta: -1 }
        }
        KeyCode::Down => {
            navigation.scroll_help_down();
            HelpInputResult::Scroll { delta: 1 }
        }
        _ => HelpInputResult::NotHandled,
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyModifiers;

    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn scrolling_saturates_and_returns_the_exact_delta() {
        let mut navigation = NavigationState::new();
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Up)),
            HelpInputResult::Scroll { delta: -1 }
        );
        assert_eq!(navigation.help_scroll, 0);
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Down)),
            HelpInputResult::Scroll { delta: 1 }
        );
        assert_eq!(navigation.help_scroll, 1);
        handle_key(&mut navigation, key(KeyCode::Up));
        assert_eq!(navigation.help_scroll, 0);
    }

    #[test]
    fn escape_closes_and_other_keys_are_not_handled() {
        let mut navigation = NavigationState::new();
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Esc)),
            HelpInputResult::Close
        );
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Home)),
            HelpInputResult::NotHandled
        );
    }
}
