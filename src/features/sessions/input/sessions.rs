use crossterm::event::KeyCode;

use crate::session_state::{SessionsState, StartPageItem};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SessionsInputResult {
    NotHandled,
    Moved {
        keep_visible_from_above: bool,
    },
    Filtered,
    ToggledGroup,
    OpenSession {
        session_id: String,
        cwd: Option<String>,
    },
    DeleteSession {
        session_id: String,
    },
    DismissRemoteSession {
        session_id: String,
    },
    OpenSessionPopup,
    OpenNewSession,
    LoadMore {
        group_idx: usize,
        parent_path: Vec<usize>,
    },
}

pub(crate) fn handle_key(state: &mut SessionsState, key: KeyCode) -> SessionsInputResult {
    match key {
        KeyCode::Up => {
            state.move_start_cursor_up();
            SessionsInputResult::Moved {
                keep_visible_from_above: true,
            }
        }
        KeyCode::Down => {
            state.move_start_cursor_down();
            SessionsInputResult::Moved {
                keep_visible_from_above: false,
            }
        }
        KeyCode::Enter => select_current(state),
        KeyCode::Delete => delete_current(state),
        KeyCode::Backspace => {
            state.start_filter_backspace();
            SessionsInputResult::Filtered
        }
        KeyCode::Char(character) => {
            state.start_filter_insert(character);
            SessionsInputResult::Filtered
        }
        _ => SessionsInputResult::NotHandled,
    }
}

pub(crate) fn toggle_selected_session_children(state: &mut SessionsState) -> SessionsInputResult {
    let selected = state
        .visible_start_items()
        .get(state.session_cursor)
        .cloned()
        .and_then(|item| match item {
            StartPageItem::Session {
                group_idx, path, ..
            } => Some((group_idx, path)),
            _ => None,
        });
    toggle_session_children(state, selected)
}

fn select_current(state: &mut SessionsState) -> SessionsInputResult {
    let items = state.visible_start_items();
    if state.session_cursor == items.len() {
        return SessionsInputResult::OpenNewSession;
    }

    let Some(item) = items.get(state.session_cursor).cloned() else {
        return SessionsInputResult::NotHandled;
    };
    match item {
        StartPageItem::GroupHeader { cwd, .. } => {
            state.toggle_group_collapse(cwd.as_deref());
            state.clamp_start_cursor();
            SessionsInputResult::ToggledGroup
        }
        StartPageItem::Session {
            group_idx, path, ..
        } => session_selection(state, group_idx, &path),
        StartPageItem::ShowMore { .. } => {
            state.reset_browser_for_open();
            SessionsInputResult::OpenSessionPopup
        }
    }
}

fn session_selection(
    state: &SessionsState,
    group_idx: usize,
    path: &[usize],
) -> SessionsInputResult {
    let Some(session) = state.session_by_path(group_idx, path) else {
        return SessionsInputResult::NotHandled;
    };
    SessionsInputResult::OpenSession {
        session_id: session.session_id.clone(),
        cwd: session
            .cwd
            .clone()
            .or_else(|| state.session_groups[group_idx].cwd.clone()),
    }
}

fn delete_current(state: &mut SessionsState) -> SessionsInputResult {
    let selected = state
        .visible_start_items()
        .get(state.session_cursor)
        .cloned();
    let Some(StartPageItem::Session {
        group_idx, path, ..
    }) = selected
    else {
        return SessionsInputResult::NotHandled;
    };
    let Some((session, is_remote)) = state.remove_session_at(group_idx, &path, false) else {
        return SessionsInputResult::NotHandled;
    };
    if is_remote {
        SessionsInputResult::DismissRemoteSession {
            session_id: session.session_id,
        }
    } else {
        SessionsInputResult::DeleteSession {
            session_id: session.session_id,
        }
    }
}

fn toggle_session_children(
    state: &mut SessionsState,
    selected: Option<(usize, Vec<usize>)>,
) -> SessionsInputResult {
    let Some((group_idx, parent_path)) = selected else {
        return SessionsInputResult::NotHandled;
    };
    if state.toggle_session_children(group_idx, &parent_path) {
        SessionsInputResult::LoadMore {
            group_idx,
            parent_path,
        }
    } else {
        SessionsInputResult::NotHandled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::session::{SessionGroup, SessionSummary};

    fn group(ids: &[&str]) -> SessionGroup {
        SessionGroup {
            cwd: Some("/work".into()),
            sessions: ids
                .iter()
                .map(|id| SessionSummary {
                    session_id: (*id).into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn movement_reports_scroll_coordination_without_owning_it() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Down),
            SessionsInputResult::Moved {
                keep_visible_from_above: false
            }
        );
        assert_eq!(state.session_cursor, 1);
        assert_eq!(
            handle_key(&mut state, KeyCode::Up),
            SessionsInputResult::Moved {
                keep_visible_from_above: true
            }
        );
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn button_header_and_show_more_return_exact_intents() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one", "two", "three", "four"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::ToggledGroup
        );
        assert!(state.collapsed_groups.contains("/work"));

        state.collapsed_groups.clear();
        state.session_cursor = 4;
        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenSessionPopup
        );
        assert_eq!(state.session_cursor, 0);

        state.session_cursor = state.visible_start_items().len();
        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenNewSession
        );
    }

    #[test]
    fn local_selection_and_delete_return_payloads_but_header_delete_is_noop() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one", "two"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete),
            SessionsInputResult::NotHandled
        );
        state.session_cursor = 1;
        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenSession {
                session_id: "one".into(),
                cwd: Some("/work".into())
            }
        );
        assert_eq!(
            handle_key(&mut state, KeyCode::Delete),
            SessionsInputResult::DeleteSession {
                session_id: "one".into()
            }
        );
        assert_eq!(state.session_groups[0].sessions[0].session_id, "two");
    }

    #[test]
    fn fork_toggle_returns_child_page_identity_only_when_loading_is_needed() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["root"])];
        state.session_groups[0].sessions[0].fork_count = 1;
        state.session_cursor = 1;

        assert_eq!(
            toggle_selected_session_children(&mut state),
            SessionsInputResult::LoadMore {
                group_idx: 0,
                parent_path: vec![0]
            }
        );
        assert!(state.expanded_session_children.contains("root"));
        assert_eq!(
            toggle_selected_session_children(&mut state),
            SessionsInputResult::NotHandled
        );
        assert!(!state.expanded_session_children.contains("root"));
    }
}
