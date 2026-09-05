use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::navigation_state::{CommandPaletteAction, NavigationState};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PaletteInputResult {
    NotHandled,
    Close,
    Moved,
    Filtered,
    Execute(CommandPaletteAction),
}

pub(crate) fn handle_key(navigation: &mut NavigationState, key: KeyEvent) -> PaletteInputResult {
    match key.code {
        KeyCode::Esc => PaletteInputResult::Close,
        KeyCode::Up => {
            navigation.move_command_palette_cursor(-1);
            PaletteInputResult::Moved
        }
        KeyCode::Down => {
            navigation.move_command_palette_cursor(1);
            PaletteInputResult::Moved
        }
        KeyCode::Backspace => {
            navigation.command_palette_filter_backspace();
            PaletteInputResult::Filtered
        }
        KeyCode::Enter => navigation
            .selected_command_palette_action()
            .map(PaletteInputResult::Execute)
            .unwrap_or(PaletteInputResult::NotHandled),
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            navigation.command_palette_filter_insert(character);
            PaletteInputResult::Filtered
        }
        _ => PaletteInputResult::NotHandled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn movement_wraps_and_filter_edits_reset_the_cursor() {
        let mut navigation = NavigationState::new();

        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Up)),
            PaletteInputResult::Moved
        );
        assert_eq!(
            navigation.command_palette_cursor,
            navigation.filtered_command_palette_commands().len() - 1
        );
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Char('t'))),
            PaletteInputResult::Filtered
        );
        assert_eq!(navigation.command_palette_filter, "t");
        assert_eq!(navigation.command_palette_cursor, 0);

        navigation.command_palette_cursor = 2;
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Backspace)),
            PaletteInputResult::Filtered
        );
        assert!(navigation.command_palette_filter.is_empty());
        assert_eq!(navigation.command_palette_cursor, 0);
    }

    #[test]
    fn enter_returns_the_exact_selected_action_or_not_handled() {
        let mut navigation = NavigationState::new();
        navigation.command_palette_filter = "theme picker".into();

        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Enter)),
            PaletteInputResult::Execute(CommandPaletteAction::ThemeSelect)
        );

        navigation.command_palette_filter = "missing command".into();
        navigation.command_palette_cursor = 0;
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Enter)),
            PaletteInputResult::NotHandled
        );
    }

    #[test]
    fn escape_closes_and_control_characters_are_not_handled() {
        let mut navigation = NavigationState::new();
        assert_eq!(
            handle_key(&mut navigation, key(KeyCode::Esc)),
            PaletteInputResult::Close
        );
        assert_eq!(
            handle_key(
                &mut navigation,
                KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
            ),
            PaletteInputResult::NotHandled
        );
        assert!(navigation.command_palette_filter.is_empty());
    }
}
