use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::session_state::SessionsState;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum NewSessionInputResult {
    NotHandled,
    Edited,
    MovedCompletion,
    AcceptCompletion,
    Cancel,
    Submit { raw_path: String },
}

pub(crate) fn handle_key(state: &mut SessionsState, key: KeyEvent) -> NewSessionInputResult {
    match key.code {
        KeyCode::Esc => NewSessionInputResult::Cancel,
        KeyCode::Up => {
            state.move_new_session_completion_selection(-1);
            NewSessionInputResult::MovedCompletion
        }
        KeyCode::Down => {
            state.move_new_session_completion_selection(1);
            NewSessionInputResult::MovedCompletion
        }
        KeyCode::Tab => NewSessionInputResult::AcceptCompletion,
        KeyCode::Left => {
            state.move_new_session_cursor_left();
            NewSessionInputResult::Edited
        }
        KeyCode::Right => {
            state.move_new_session_cursor_right();
            NewSessionInputResult::Edited
        }
        KeyCode::Home => {
            state.move_new_session_cursor_home();
            NewSessionInputResult::Edited
        }
        KeyCode::End => {
            state.move_new_session_cursor_end();
            NewSessionInputResult::Edited
        }
        KeyCode::Backspace => {
            state.new_session_backspace();
            NewSessionInputResult::Edited
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.new_session_insert(character);
            NewSessionInputResult::Edited
        }
        KeyCode::Enter => NewSessionInputResult::Submit {
            raw_path: state.new_session_path.clone(),
        },
        _ => NewSessionInputResult::NotHandled,
    }
}

pub(crate) fn accept_completion(state: &mut SessionsState, path: String) {
    state.accept_new_session_completion(path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{composer_state::FileIndexEntryLite, session_state::PathCompletionState};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn editing_and_cursor_keys_mutate_only_session_editing_state() {
        let mut state = SessionsState::new();
        state.new_session_path = "ac".into();
        state.new_session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, key(KeyCode::Char('b'))),
            NewSessionInputResult::Edited
        );
        assert_eq!(state.new_session_path, "abc");
        assert_eq!(state.new_session_cursor, 2);
        handle_key(&mut state, key(KeyCode::Backspace));
        assert_eq!(state.new_session_path, "ac");
        handle_key(&mut state, key(KeyCode::End));
        assert_eq!(state.new_session_cursor, 2);
    }

    #[test]
    fn completion_selection_and_acceptance_are_explicit_transitions() {
        let mut state = SessionsState::new();
        state.new_session_completion = Some(PathCompletionState {
            query: "a".into(),
            selected_index: 0,
            results: vec![
                FileIndexEntryLite {
                    path: "/a".into(),
                    is_dir: true,
                },
                FileIndexEntryLite {
                    path: "/b".into(),
                    is_dir: true,
                },
            ],
        });

        assert_eq!(
            handle_key(&mut state, key(KeyCode::Down)),
            NewSessionInputResult::MovedCompletion
        );
        assert_eq!(
            state
                .new_session_completion
                .as_ref()
                .unwrap()
                .selected_index,
            1
        );
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Tab)),
            NewSessionInputResult::AcceptCompletion
        );
        accept_completion(&mut state, "/b/".into());
        assert_eq!(state.new_session_path, "/b/");
        assert!(state.new_session_completion.is_none());
    }

    #[test]
    fn cancel_and_submit_return_data_only_intents() {
        let mut state = SessionsState::new();
        state.new_session_path = "relative/path".into();
        state.new_session_cursor = state.new_session_path.len();

        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            NewSessionInputResult::Submit {
                raw_path: "relative/path".into()
            }
        );
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Esc)),
            NewSessionInputResult::Cancel
        );
    }

    #[test]
    fn control_character_is_not_inserted() {
        let mut state = SessionsState::new();
        assert_eq!(
            handle_key(
                &mut state,
                KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL)
            ),
            NewSessionInputResult::NotHandled
        );
        assert!(state.new_session_path.is_empty());
    }
}
