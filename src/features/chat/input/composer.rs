use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::{composer_state::ComposerState, render_state::RenderState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComposerKeyResult {
    NotHandled,
    Handled,
    Edited,
    Navigated,
}

pub(crate) fn handle_key(
    composer: &mut ComposerState,
    render: &mut RenderState,
    key: KeyEvent,
    editable: bool,
) -> ComposerKeyResult {
    if !editable {
        return ComposerKeyResult::NotHandled;
    }

    match key.code {
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            composer.input_insert(character);
            ComposerKeyResult::Edited
        }
        KeyCode::Backspace => {
            if composer.input_cursor == 0 {
                ComposerKeyResult::Handled
            } else {
                composer.input_backspace();
                ComposerKeyResult::Edited
            }
        }
        KeyCode::Delete => {
            if composer.input_cursor == composer.input.len() {
                ComposerKeyResult::Handled
            } else {
                composer.input_delete();
                ComposerKeyResult::Edited
            }
        }
        KeyCode::Left => {
            composer.input_left();
            ComposerKeyResult::Navigated
        }
        KeyCode::Right => {
            composer.input_right();
            ComposerKeyResult::Navigated
        }
        KeyCode::Home => {
            composer.input_home();
            ComposerKeyResult::Navigated
        }
        KeyCode::End if composer.input.is_empty() => ComposerKeyResult::NotHandled,
        KeyCode::End => {
            composer.input_end();
            ComposerKeyResult::Navigated
        }
        KeyCode::Up => {
            composer.input_up_visual(render.composer_input_line_width(), 2);
            ComposerKeyResult::Navigated
        }
        KeyCode::Down => {
            composer.input_down_visual(render.composer_input_line_width(), 2);
            ComposerKeyResult::Navigated
        }
        _ => ComposerKeyResult::NotHandled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn utf8_editing_and_horizontal_navigation_preserve_byte_boundaries() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();
        composer.input = "éß".into();
        composer.input_cursor = "é".len();

        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Char('中')), true),
            ComposerKeyResult::Edited
        );
        assert_eq!(composer.input, "é中ß");
        assert_eq!(composer.input_cursor, "é中".len());

        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Backspace), true),
            ComposerKeyResult::Edited
        );
        assert_eq!(composer.input, "éß");
        assert_eq!(composer.input_cursor, "é".len());

        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Delete), true),
            ComposerKeyResult::Edited
        );
        assert_eq!(composer.input, "é");
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Left), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(composer.input_cursor, 0);
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Right), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(composer.input_cursor, "é".len());
    }

    #[test]
    fn edit_boundaries_are_handled_without_claiming_an_edit() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();

        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Backspace), true),
            ComposerKeyResult::Handled
        );
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Delete), true),
            ComposerKeyResult::Handled
        );
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Left), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Right), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(
            handle_key(
                &mut composer,
                &mut render,
                KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
                true,
            ),
            ComposerKeyResult::NotHandled
        );
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Char('x')), false),
            ComposerKeyResult::NotHandled
        );
        assert!(composer.input.is_empty());
    }

    #[test]
    fn mention_edit_prepares_file_index_request_only_once() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();

        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Char('@')), true),
            ComposerKeyResult::Edited
        );
        assert!(composer.prepare_file_index_request());
        assert!(!composer.prepare_file_index_request());
    }

    #[test]
    fn home_and_end_navigate_but_empty_end_remains_contextual() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();
        composer.input = "draft".into();
        composer.input_cursor = 2;

        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Home), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(composer.input_cursor, 0);
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::End), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(composer.input_cursor, composer.input.len());

        composer.clear_input();
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Home), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::End), true),
            ComposerKeyResult::NotHandled
        );
        composer.input = "blocked".into();
        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::End), false),
            ComposerKeyResult::NotHandled
        );
    }

    #[test]
    fn visual_navigation_uses_default_published_and_resized_widths() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();
        composer.input = "abc".into();
        composer.input_cursor = composer.input.len();

        assert_eq!(
            handle_key(&mut composer, &mut render, key(KeyCode::Up), true),
            ComposerKeyResult::Navigated
        );
        assert_eq!(render.composer_input_line_width(), 1);
        assert_eq!(composer.input_cursor, 2);

        composer.input_cursor = composer.input.len();
        composer.input_preferred_col = None;
        render.prepare_composer_input_layout("abc", 3, 4, 2);
        handle_key(&mut composer, &mut render, key(KeyCode::Up), true);
        assert_eq!(composer.input_cursor, 1);

        composer.input_preferred_col = None;
        render.prepare_composer_input_layout("abc", 1, 20, 2);
        handle_key(&mut composer, &mut render, key(KeyCode::Down), true);
        assert_eq!(composer.input_cursor, 1);
        assert_eq!(composer.input_preferred_col, Some(1));
    }

    #[test]
    fn visual_navigation_retains_preferred_column_across_unequal_rows() {
        let mut composer = ComposerState::new();
        let mut render = RenderState::new();
        composer.input = "abcd\nx\nwxyz".into();
        composer.input_cursor = composer.input.len();
        render.prepare_composer_input_layout(&composer.input, composer.input_cursor, 20, 2);

        handle_key(&mut composer, &mut render, key(KeyCode::Up), true);
        assert_eq!(composer.input_cursor, "abcd\nx".len());
        assert_eq!(composer.input_preferred_col, Some(4));
        handle_key(&mut composer, &mut render, key(KeyCode::Up), true);
        assert_eq!(composer.input_cursor, 4);
        handle_key(&mut composer, &mut render, key(KeyCode::Down), true);
        assert_eq!(composer.input_cursor, "abcd\nx".len());
        handle_key(&mut composer, &mut render, key(KeyCode::Down), true);
        assert_eq!(composer.input_cursor, composer.input.len());
        assert_eq!(composer.input_preferred_col, Some(4));
    }
}
