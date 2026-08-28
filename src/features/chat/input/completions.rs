use crossterm::event::{KeyCode, KeyEvent};

use crate::composer_state::ComposerState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompletionResult {
    NoOp,
    Dismissed,
    SlashSelected,
    SlashAccepted,
    MentionSelected,
    MentionAccepted { request_file_index: bool },
}

pub(crate) fn handle_key(
    composer: &mut ComposerState,
    key: KeyEvent,
    editable: bool,
) -> CompletionResult {
    match key.code {
        KeyCode::Esc => {
            if composer.slash_state.is_none() && composer.mention_state.is_none() {
                return CompletionResult::NoOp;
            }
            composer.slash_state = None;
            composer.mention_state = None;
            CompletionResult::Dismissed
        }
        KeyCode::Up | KeyCode::Down if editable => {
            let delta = if key.code == KeyCode::Up { -1 } else { 1 };
            if composer.slash_state.is_some() {
                composer.move_slash_selection(delta);
                CompletionResult::SlashSelected
            } else if composer.mention_state.is_some() {
                composer.move_mention_selection(delta);
                CompletionResult::MentionSelected
            } else {
                CompletionResult::NoOp
            }
        }
        KeyCode::Enter => {
            if composer.slash_state.is_some() {
                return if composer.accept_selected_slash_completion() {
                    CompletionResult::SlashAccepted
                } else {
                    CompletionResult::NoOp
                };
            }
            accept_mention(composer)
        }
        KeyCode::Tab if editable => {
            if composer.slash_state.is_some() {
                if composer.accept_selected_slash_completion() {
                    CompletionResult::SlashAccepted
                } else {
                    CompletionResult::NoOp
                }
            } else {
                accept_mention(composer)
            }
        }
        _ => CompletionResult::NoOp,
    }
}

fn accept_mention(composer: &mut ComposerState) -> CompletionResult {
    if composer.mention_state.is_some() && composer.accept_selected_mention() {
        CompletionResult::MentionAccepted {
            request_file_index: composer.prepare_file_index_request(),
        }
    } else {
        CompletionResult::NoOp
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyModifiers;

    use super::*;
    use crate::composer_state::{FileIndexEntryLite, MentionState};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn entry(path: &str) -> FileIndexEntryLite {
        FileIndexEntryLite {
            path: path.into(),
            is_dir: false,
        }
    }

    fn composer_with_both_completions() -> ComposerState {
        let mut composer = ComposerState::new();
        composer.input = "/".into();
        composer.input_cursor = 1;
        composer.refresh_slash_state();
        composer.mention_state = Some(MentionState {
            trigger_start: 0,
            query: String::new(),
            selected_index: 0,
            results: vec![entry("src/main.rs"), entry("src/lib.rs")],
        });
        composer
    }

    #[test]
    fn escape_dismisses_both_completion_kinds_before_other_routing() {
        let mut composer = composer_with_both_completions();

        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Esc), false),
            CompletionResult::Dismissed
        );
        assert!(composer.slash_state.is_none());
        assert!(composer.mention_state.is_none());
        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Esc), true),
            CompletionResult::NoOp
        );
    }

    #[test]
    fn slash_selection_has_precedence_and_wraps_in_both_directions() {
        let mut composer = composer_with_both_completions();
        let total = composer.slash_state.as_ref().unwrap().results.len();
        composer.slash_state.as_mut().unwrap().selected_index = total - 1;

        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Down), true),
            CompletionResult::SlashSelected
        );
        assert_eq!(composer.slash_state.as_ref().unwrap().selected_index, 0);
        assert_eq!(composer.mention_state.as_ref().unwrap().selected_index, 0);
        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Up), true),
            CompletionResult::SlashSelected
        );
        assert_eq!(
            composer.slash_state.as_ref().unwrap().selected_index,
            total - 1
        );
    }

    #[test]
    fn mention_selection_wraps_when_no_slash_completion_is_active() {
        let mut composer = composer_with_both_completions();
        composer.slash_state = None;

        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Up), true),
            CompletionResult::MentionSelected
        );
        assert_eq!(composer.mention_state.as_ref().unwrap().selected_index, 1);
        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Down), true),
            CompletionResult::MentionSelected
        );
        assert_eq!(composer.mention_state.as_ref().unwrap().selected_index, 0);
    }

    #[test]
    fn slash_acceptance_precedes_mention_acceptance() {
        let mut composer = composer_with_both_completions();
        let command = composer.slash_state.as_ref().unwrap().results[0].name;

        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Enter), true),
            CompletionResult::SlashAccepted
        );
        assert_eq!(composer.input, format!("/{command} "));
        assert!(composer.mention_state.is_some());
    }

    #[test]
    fn slash_enter_and_tab_accept_without_executing() {
        for code in [KeyCode::Enter, KeyCode::Tab] {
            let mut composer = ComposerState::new();
            composer.input = "/hel".into();
            composer.input_cursor = composer.input.len();
            composer.refresh_slash_state();

            assert_eq!(
                handle_key(&mut composer, key(code), true),
                CompletionResult::SlashAccepted
            );
            assert_eq!(composer.input, "/help ");
            assert!(composer.slash_state.is_none());
        }
    }

    #[test]
    fn mention_enter_accepts_selection_and_reports_request_intent() {
        let mut composer = ComposerState::new();
        composer.input = "open @src/ma".into();
        composer.input_cursor = composer.input.len();
        composer.file_index = vec![entry("src/main.rs")];
        composer.refresh_mention_state();

        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Enter), true),
            CompletionResult::MentionAccepted {
                request_file_index: false
            }
        );
        assert_eq!(composer.input, "open @src/main.rs ");
        assert!(composer.mention_state.is_none());
        assert!(!composer.file_index_loading);
        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Enter), true),
            CompletionResult::NoOp
        );
    }

    #[test]
    fn blocked_selection_and_tab_are_noops_but_enter_preserves_local_behavior() {
        let mut composer = ComposerState::new();
        composer.input = "/hel".into();
        composer.input_cursor = composer.input.len();
        composer.refresh_slash_state();

        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Down), false),
            CompletionResult::NoOp
        );
        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Tab), false),
            CompletionResult::NoOp
        );
        assert_eq!(composer.input, "/hel");
        assert_eq!(
            handle_key(&mut composer, key(KeyCode::Enter), false),
            CompletionResult::SlashAccepted
        );
        assert_eq!(composer.input, "/help ");
    }
}
