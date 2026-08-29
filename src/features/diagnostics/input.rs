use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::diagnostics::DiagnosticsState;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DiagnosticsInputResult {
    NotHandled,
    Close,
    Moved,
    Paged,
    ToStart,
    ToEnd,
    Filtered,
    CycledLevel,
}

pub(crate) fn handle_key(
    diagnostics: &mut DiagnosticsState,
    key: KeyEvent,
) -> DiagnosticsInputResult {
    match key.code {
        KeyCode::Esc => DiagnosticsInputResult::Close,
        KeyCode::Up => {
            diagnostics.log_cursor = diagnostics.log_cursor.saturating_sub(1);
            DiagnosticsInputResult::Moved
        }
        KeyCode::Down => {
            let max = diagnostics.filtered_logs().len().saturating_sub(1);
            diagnostics.log_cursor = (diagnostics.log_cursor + 1).min(max);
            DiagnosticsInputResult::Moved
        }
        KeyCode::PageUp => {
            diagnostics.log_cursor = diagnostics.log_cursor.saturating_sub(10);
            DiagnosticsInputResult::Paged
        }
        KeyCode::PageDown => {
            let max = diagnostics.filtered_logs().len().saturating_sub(1);
            diagnostics.log_cursor = (diagnostics.log_cursor + 10).min(max);
            DiagnosticsInputResult::Paged
        }
        KeyCode::Home => {
            diagnostics.log_cursor = 0;
            DiagnosticsInputResult::ToStart
        }
        KeyCode::End => {
            diagnostics.log_cursor = diagnostics.filtered_logs().len().saturating_sub(1);
            DiagnosticsInputResult::ToEnd
        }
        KeyCode::Backspace => {
            diagnostics.log_filter.pop();
            diagnostics.log_cursor = diagnostics.filtered_logs().len().saturating_sub(1);
            DiagnosticsInputResult::Filtered
        }
        KeyCode::Tab => {
            diagnostics.cycle_log_level_filter();
            diagnostics.log_cursor = diagnostics.filtered_logs().len().saturating_sub(1);
            DiagnosticsInputResult::CycledLevel
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            diagnostics.log_filter.push(character);
            diagnostics.log_cursor = 0;
            DiagnosticsInputResult::Filtered
        }
        _ => DiagnosticsInputResult::NotHandled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn seed_logs(diagnostics: &mut DiagnosticsState, count: usize) {
        let level = diagnostics.log_level_filter;
        for index in 0..count {
            diagnostics.push_log(level, "test", format!("entry {index}"));
        }
    }

    #[test]
    fn row_page_start_and_end_movement_saturate() {
        let mut diagnostics = DiagnosticsState::new();
        seed_logs(&mut diagnostics, 15);
        diagnostics.log_cursor = 12;
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::Up)),
            DiagnosticsInputResult::Moved
        );
        assert_eq!(diagnostics.log_cursor, 11);
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::PageUp)),
            DiagnosticsInputResult::Paged
        );
        assert_eq!(diagnostics.log_cursor, 1);
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::Home)),
            DiagnosticsInputResult::ToStart
        );
        assert_eq!(diagnostics.log_cursor, 0);

        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::Down)),
            DiagnosticsInputResult::Moved
        );
        assert_eq!(diagnostics.log_cursor, 1);
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::PageDown)),
            DiagnosticsInputResult::Paged
        );
        assert_eq!(diagnostics.log_cursor, 11);
        handle_key(&mut diagnostics, key(KeyCode::PageDown));
        assert_eq!(diagnostics.log_cursor, 14);
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::End)),
            DiagnosticsInputResult::ToEnd
        );
        assert_eq!(diagnostics.log_cursor, 14);
    }

    #[test]
    fn filter_edits_and_level_cycle_reset_the_cursor() {
        let mut diagnostics = DiagnosticsState::new();
        seed_logs(&mut diagnostics, 3);
        diagnostics.log_cursor = 2;
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::Char('x'))),
            DiagnosticsInputResult::Filtered
        );
        assert_eq!(diagnostics.log_filter, "x");
        assert_eq!(diagnostics.log_cursor, 0);

        diagnostics.log_cursor = 8;
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::Backspace)),
            DiagnosticsInputResult::Filtered
        );
        assert!(diagnostics.log_filter.is_empty());
        assert_eq!(diagnostics.log_cursor, 2);

        diagnostics.cycle_log_level_filter();
        seed_logs(&mut diagnostics, 3);
        for _ in 0..4 {
            diagnostics.cycle_log_level_filter();
        }
        let previous_level = diagnostics.log_level_filter.label();
        diagnostics.log_cursor = 8;
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::Tab)),
            DiagnosticsInputResult::CycledLevel
        );
        assert_ne!(diagnostics.log_level_filter.label(), previous_level);
        assert_eq!(diagnostics.log_cursor, 2);
    }

    #[test]
    fn escape_closes_and_control_characters_are_not_handled() {
        let mut diagnostics = DiagnosticsState::new();
        assert_eq!(
            handle_key(&mut diagnostics, key(KeyCode::Esc)),
            DiagnosticsInputResult::Close
        );
        assert_eq!(
            handle_key(
                &mut diagnostics,
                KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL),
            ),
            DiagnosticsInputResult::NotHandled
        );
        assert!(diagnostics.log_filter.is_empty());
    }
}
