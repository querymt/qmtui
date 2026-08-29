use crossterm::event::{KeyCode, KeyEvent};

use super::completions::{CompletionResult, handle_key as handle_completion_key};
use super::composer::{ComposerKeyResult, handle_key as handle_composer_key};
use crate::{composer_state::ComposerState, render_state::RenderState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChatInputContext {
    pub(crate) editable: bool,
    pub(crate) has_cancellable_activity: bool,
    pub(crate) cancel_confirmation_active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatInputResult {
    NotHandled,
    CompletionDismissed,
    Cancel(CancelIntent),
    Viewport(ChatViewportIntent),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelIntent {
    ArmConfirmation,
    ConfirmCancellation,
    ClearConfirmation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatViewportIntent {
    ScrollUp { rows: u16 },
    ScrollDown { rows: u16 },
    ToBottom,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChatCommandIntent {
    OpenExternalEditor { initial_text: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptSubmission {
    pub(crate) text: String,
    pub(crate) links: Vec<String>,
}

pub(crate) fn handle_coordination_key(
    composer: &mut ComposerState,
    render: &mut RenderState,
    key: KeyEvent,
    context: ChatInputContext,
) -> ChatInputResult {
    match key.code {
        KeyCode::Esc => {
            if handle_completion_key(composer, key, context.editable) == CompletionResult::Dismissed
            {
                return ChatInputResult::CompletionDismissed;
            }

            let intent = if !context.has_cancellable_activity {
                CancelIntent::ClearConfirmation
            } else if context.cancel_confirmation_active {
                CancelIntent::ConfirmCancellation
            } else {
                CancelIntent::ArmConfirmation
            };
            ChatInputResult::Cancel(intent)
        }
        KeyCode::PageUp => ChatInputResult::Viewport(ChatViewportIntent::ScrollUp { rows: 10 }),
        KeyCode::PageDown => ChatInputResult::Viewport(ChatViewportIntent::ScrollDown { rows: 10 }),
        KeyCode::End => {
            if handle_composer_key(composer, render, key, context.editable)
                == ComposerKeyResult::NotHandled
            {
                ChatInputResult::Viewport(ChatViewportIntent::ToBottom)
            } else {
                ChatInputResult::NotHandled
            }
        }
        _ => ChatInputResult::NotHandled,
    }
}

pub(crate) fn chat_command_intent(
    composer: &ComposerState,
    key: KeyEvent,
) -> Option<ChatCommandIntent> {
    match key.code {
        KeyCode::Char('e') => Some(ChatCommandIntent::OpenExternalEditor {
            initial_text: composer.input.clone(),
        }),
        _ => None,
    }
}

pub(crate) fn build_prompt_submission(composer: &ComposerState) -> Option<PromptSubmission> {
    if composer.input.is_empty() {
        return None;
    }

    let (text, links) = composer.build_prompt_text_and_links(&composer.input);
    Some(PromptSubmission {
        text: text.trim().to_string(),
        links,
    })
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyModifiers;

    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn context(
        editable: bool,
        has_cancellable_activity: bool,
        cancel_confirmation_active: bool,
    ) -> ChatInputContext {
        ChatInputContext {
            editable,
            has_cancellable_activity,
            cancel_confirmation_active,
        }
    }

    #[test]
    fn escape_dismisses_completion_before_cancellation() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();
        composer.input = "/mo".into();
        composer.input_cursor = composer.input.len();
        composer.refresh_slash_state();
        assert!(composer.slash_state.is_some());

        let result = handle_coordination_key(
            &mut composer,
            &mut render,
            key(KeyCode::Esc),
            context(true, true, true),
        );

        assert_eq!(result, ChatInputResult::CompletionDismissed);
        assert!(composer.slash_state.is_none());
    }

    #[test]
    fn escape_classifies_all_cancellation_intents() {
        for (input, expected) in [
            (context(true, true, false), CancelIntent::ArmConfirmation),
            (
                context(false, true, true),
                CancelIntent::ConfirmCancellation,
            ),
            (context(true, false, true), CancelIntent::ClearConfirmation),
        ] {
            let mut composer = ComposerState::new();
            let mut render = RenderState::new();
            assert_eq!(
                handle_coordination_key(&mut composer, &mut render, key(KeyCode::Esc), input,),
                ChatInputResult::Cancel(expected)
            );
        }
    }

    #[test]
    fn page_keys_return_exact_ten_row_viewport_intents() {
        for (code, expected) in [
            (KeyCode::PageUp, ChatViewportIntent::ScrollUp { rows: 10 }),
            (
                KeyCode::PageDown,
                ChatViewportIntent::ScrollDown { rows: 10 },
            ),
        ] {
            let mut composer = ComposerState::new();
            let mut render = RenderState::new();
            assert_eq!(
                handle_coordination_key(
                    &mut composer,
                    &mut render,
                    key(code),
                    context(false, false, false),
                ),
                ChatInputResult::Viewport(expected)
            );
        }
    }

    #[test]
    fn end_moves_editable_composer_or_requests_chat_bottom() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();
        composer.input = "draft".into();
        composer.input_cursor = 1;

        assert_eq!(
            handle_coordination_key(
                &mut composer,
                &mut render,
                key(KeyCode::End),
                context(true, false, false),
            ),
            ChatInputResult::NotHandled
        );
        assert_eq!(composer.input_cursor, composer.input.len());

        composer.input.clear();
        composer.input_cursor = 0;
        assert_eq!(
            handle_coordination_key(
                &mut composer,
                &mut render,
                key(KeyCode::End),
                context(true, false, false),
            ),
            ChatInputResult::Viewport(ChatViewportIntent::ToBottom)
        );

        composer.input = "blocked".into();
        composer.input_cursor = 1;
        assert_eq!(
            handle_coordination_key(
                &mut composer,
                &mut render,
                key(KeyCode::End),
                context(false, false, false),
            ),
            ChatInputResult::Viewport(ChatViewportIntent::ToBottom)
        );
        assert_eq!(composer.input_cursor, 1);
    }

    #[test]
    fn command_classifier_accepts_only_lowercase_e() {
        let mut composer = ComposerState::new();
        composer.input = "draft".into();

        assert_eq!(
            chat_command_intent(&composer, key(KeyCode::Char('e'))),
            Some(ChatCommandIntent::OpenExternalEditor {
                initial_text: "draft".into(),
            })
        );
        assert_eq!(
            chat_command_intent(&composer, key(KeyCode::Char('E'))),
            None
        );
        assert_eq!(chat_command_intent(&composer, key(KeyCode::Enter)), None);
        assert_eq!(composer.input, "draft");
    }

    #[test]
    fn prompt_submission_distinguishes_empty_and_whitespace_input() {
        let mut composer = ComposerState::new();
        assert_eq!(build_prompt_submission(&composer), None);

        composer.input = " \n  ".into();
        assert_eq!(
            build_prompt_submission(&composer),
            Some(PromptSubmission {
                text: String::new(),
                links: Vec::new(),
            })
        );
        assert_eq!(composer.input, " \n  ");
    }

    #[test]
    fn prompt_submission_trims_text_and_preserves_link_order_and_deduplication() {
        let mut composer = ComposerState::new();
        composer.input =
            "  check @src/main.rs then @src/lib.rs and @src/main.rs with @person  ".into();

        assert_eq!(
            build_prompt_submission(&composer),
            Some(PromptSubmission {
                text: "check @src/main.rs then @src/lib.rs and @src/main.rs with @person".into(),
                links: vec!["src/main.rs".into(), "src/lib.rs".into()],
            })
        );
        assert!(composer.input.starts_with("  check"));
    }
}
