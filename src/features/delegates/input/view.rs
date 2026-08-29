use crossterm::event::KeyCode;

use crate::delegates_state::DelegatesState;

pub(crate) struct DelegateInputContext {
    pub(crate) page_step: usize,
    pub(crate) current_session_id: Option<String>,
    pub(crate) cwd: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DelegateViewportIntent {
    UpOne,
    DownOne,
    UpTen,
    DownTen,
    Top,
    Bottom,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DelegateInputResult {
    NotHandled,
    Moved,
    Filtered,
    Viewport(DelegateViewportIntent),
    ClosePopup,
    NavigateParent {
        session_id: String,
    },
    LoadChild {
        session_id: String,
        target_agent_id: Option<String>,
        current_session_id: Option<String>,
        parent_session_id: Option<String>,
        cwd: Option<String>,
    },
    PendingChild,
}

pub(crate) fn handle_view_key(state: &DelegatesState, key: KeyCode) -> DelegateInputResult {
    match key {
        KeyCode::Up => DelegateInputResult::Viewport(DelegateViewportIntent::UpOne),
        KeyCode::Down => DelegateInputResult::Viewport(DelegateViewportIntent::DownOne),
        KeyCode::PageUp => DelegateInputResult::Viewport(DelegateViewportIntent::UpTen),
        KeyCode::PageDown => DelegateInputResult::Viewport(DelegateViewportIntent::DownTen),
        KeyCode::Home => DelegateInputResult::Viewport(DelegateViewportIntent::Top),
        KeyCode::End => DelegateInputResult::Viewport(DelegateViewportIntent::Bottom),
        KeyCode::Esc => state
            .parent_session_id
            .clone()
            .map(|session_id| DelegateInputResult::NavigateParent { session_id })
            .unwrap_or(DelegateInputResult::NotHandled),
        _ => DelegateInputResult::NotHandled,
    }
}

pub(crate) fn handle_popup_key(
    state: &mut DelegatesState,
    key: KeyCode,
    context: DelegateInputContext,
) -> DelegateInputResult {
    match key {
        KeyCode::Esc => DelegateInputResult::ClosePopup,
        KeyCode::Up => {
            state.move_cursor_up();
            DelegateInputResult::Moved
        }
        KeyCode::Down => {
            state.move_cursor_down();
            DelegateInputResult::Moved
        }
        KeyCode::PageUp => {
            state.move_cursor_page(context.page_step, false);
            DelegateInputResult::Moved
        }
        KeyCode::PageDown => {
            state.move_cursor_page(context.page_step, true);
            DelegateInputResult::Moved
        }
        KeyCode::Enter => select_current(state, context),
        KeyCode::Backspace => {
            state.filter_backspace();
            DelegateInputResult::Filtered
        }
        KeyCode::Char(character) => {
            state.filter_insert(character);
            DelegateInputResult::Filtered
        }
        _ => DelegateInputResult::NotHandled,
    }
}

fn select_current(
    state: &mut DelegatesState,
    context: DelegateInputContext,
) -> DelegateInputResult {
    let selected = state.selected_entry().map(|entry| {
        (
            entry.child_session_id.clone(),
            entry.target_agent_id.clone(),
        )
    });
    let Some((child_session_id, target_agent_id)) = selected else {
        return DelegateInputResult::NotHandled;
    };
    let Some(session_id) = child_session_id else {
        return DelegateInputResult::PendingChild;
    };

    let parent_session_id = state.parent_session_id.clone();
    state.stage_parent_for_child_navigation(
        parent_session_id.clone(),
        context.current_session_id.clone(),
    );
    DelegateInputResult::LoadChild {
        session_id,
        target_agent_id,
        current_session_id: context.current_session_id,
        parent_session_id,
        cwd: context.cwd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::activity::{
        DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus,
    };

    fn entry(id: &str, child_session_id: Option<&str>) -> DelegateEntry {
        DelegateEntry {
            delegation_id: id.into(),
            child_session_id: child_session_id.map(str::to_string),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: format!("objective {id}"),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        }
    }

    fn context(page_step: usize) -> DelegateInputContext {
        DelegateInputContext {
            page_step,
            current_session_id: Some("child-old".into()),
            cwd: Some("/work".into()),
        }
    }

    #[test]
    fn view_keys_return_exact_viewport_and_parent_intents() {
        let mut state = DelegatesState::new();
        state.parent_session_id = Some("parent".into());

        for (key, expected) in [
            (KeyCode::Up, DelegateViewportIntent::UpOne),
            (KeyCode::Down, DelegateViewportIntent::DownOne),
            (KeyCode::PageUp, DelegateViewportIntent::UpTen),
            (KeyCode::PageDown, DelegateViewportIntent::DownTen),
            (KeyCode::Home, DelegateViewportIntent::Top),
            (KeyCode::End, DelegateViewportIntent::Bottom),
        ] {
            assert_eq!(
                handle_view_key(&state, key),
                DelegateInputResult::Viewport(expected)
            );
        }
        assert_eq!(
            handle_view_key(&state, KeyCode::Esc),
            DelegateInputResult::NavigateParent {
                session_id: "parent".into()
            }
        );
    }

    #[test]
    fn popup_movement_and_filtering_mutate_delegate_state() {
        let mut state = DelegatesState::new();
        state.delegate_entries = vec![entry("one", Some("child-1")), entry("two", Some("child-2"))];

        assert_eq!(
            handle_popup_key(&mut state, KeyCode::PageDown, context(2)),
            DelegateInputResult::Moved
        );
        assert_eq!(state.delegate_cursor, 1);
        assert_eq!(
            handle_popup_key(&mut state, KeyCode::Char('o'), context(1)),
            DelegateInputResult::Filtered
        );
        assert_eq!(state.delegate_filter, "o");
        assert_eq!(state.delegate_cursor, 0);
    }

    #[test]
    fn pending_and_load_results_carry_exact_coordination_payloads() {
        let mut state = DelegatesState::new();
        state.delegate_entries = vec![entry("pending", None)];
        assert_eq!(
            handle_popup_key(&mut state, KeyCode::Enter, context(1)),
            DelegateInputResult::PendingChild
        );
        assert!(state.pending_parent_session_id.is_none());

        state.delegate_entries = vec![entry("ready", Some("child-new"))];
        state.parent_session_id = Some("parent".into());
        assert_eq!(
            handle_popup_key(&mut state, KeyCode::Enter, context(1)),
            DelegateInputResult::LoadChild {
                session_id: "child-new".into(),
                target_agent_id: Some("coder".into()),
                current_session_id: Some("child-old".into()),
                parent_session_id: Some("parent".into()),
                cwd: Some("/work".into())
            }
        );
        assert_eq!(state.pending_parent_session_id.as_deref(), Some("parent"));
    }

    #[test]
    fn escape_returns_close_without_navigation_types() {
        let mut state = DelegatesState::new();
        assert_eq!(
            handle_popup_key(&mut state, KeyCode::Esc, context(1)),
            DelegateInputResult::ClosePopup
        );
    }
}
