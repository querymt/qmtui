use crossterm::event::KeyCode;

use crate::session_state::{PopupItem, SessionsState};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SessionPopupInputResult {
    NotHandled,
    Moved,
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
    LoadMore {
        group_idx: usize,
        parent_path: Vec<usize>,
    },
    ClosePopup,
}

pub(crate) fn switch_tab(state: &mut SessionsState) {
    state.switch_session_popup_tab();
}

pub(crate) fn handle_key(
    state: &mut SessionsState,
    key: KeyCode,
    page_step: usize,
) -> SessionPopupInputResult {
    match key {
        KeyCode::Esc => SessionPopupInputResult::ClosePopup,
        KeyCode::Up => {
            state.move_popup_cursor_up();
            SessionPopupInputResult::Moved
        }
        KeyCode::Down => {
            state.move_popup_cursor_down();
            SessionPopupInputResult::Moved
        }
        KeyCode::PageUp => {
            state.move_popup_cursor_page(page_step, false);
            SessionPopupInputResult::Moved
        }
        KeyCode::PageDown => {
            state.move_popup_cursor_page(page_step, true);
            SessionPopupInputResult::Moved
        }
        KeyCode::Enter => select_current(state),
        KeyCode::Delete => delete_current(state),
        KeyCode::Backspace => {
            state.popup_filter_backspace();
            SessionPopupInputResult::Filtered
        }
        KeyCode::Char(character) => {
            state.popup_filter_insert(character);
            SessionPopupInputResult::Filtered
        }
        _ => SessionPopupInputResult::NotHandled,
    }
}

pub(crate) fn toggle_selected_session_children(
    state: &mut SessionsState,
) -> SessionPopupInputResult {
    let selected = state
        .visible_popup_items()
        .get(state.session_cursor)
        .cloned()
        .and_then(|item| match item {
            PopupItem::Session {
                group_idx, path, ..
            } => Some((group_idx, path)),
            _ => None,
        });
    let Some((group_idx, parent_path)) = selected else {
        return SessionPopupInputResult::NotHandled;
    };
    if state.toggle_session_children(group_idx, &parent_path) {
        SessionPopupInputResult::LoadMore {
            group_idx,
            parent_path,
        }
    } else {
        SessionPopupInputResult::NotHandled
    }
}

fn select_current(state: &mut SessionsState) -> SessionPopupInputResult {
    let item = state
        .visible_popup_items()
        .get(state.session_cursor)
        .cloned();
    let Some(item) = item else {
        return SessionPopupInputResult::NotHandled;
    };
    match item {
        PopupItem::GroupHeader { cwd, .. } => {
            state.toggle_popup_group_collapse(cwd.as_deref());
            state.clamp_popup_cursor();
            SessionPopupInputResult::ToggledGroup
        }
        PopupItem::Session {
            group_idx, path, ..
        } => {
            let Some(session) = state.session_by_path(group_idx, &path) else {
                return SessionPopupInputResult::NotHandled;
            };
            SessionPopupInputResult::OpenSession {
                session_id: session.session_id.clone(),
                cwd: session
                    .cwd
                    .clone()
                    .or_else(|| state.session_groups[group_idx].cwd.clone()),
            }
        }
        PopupItem::LoadMore {
            group_idx,
            parent_path,
        } => SessionPopupInputResult::LoadMore {
            group_idx,
            parent_path,
        },
    }
}

fn delete_current(state: &mut SessionsState) -> SessionPopupInputResult {
    let selected = state
        .visible_popup_items()
        .get(state.session_cursor)
        .cloned();
    let Some(PopupItem::Session {
        group_idx, path, ..
    }) = selected
    else {
        return SessionPopupInputResult::NotHandled;
    };
    let Some((session, is_remote)) = state.remove_session_at(group_idx, &path, true) else {
        return SessionPopupInputResult::NotHandled;
    };
    if is_remote {
        SessionPopupInputResult::DismissRemoteSession {
            session_id: session.session_id,
        }
    } else {
        SessionPopupInputResult::DeleteSession {
            session_id: session.session_id,
        }
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
                    cwd: Some("/session".into()),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn tab_switch_preserves_each_owner_projection() {
        let mut state = SessionsState::new();
        state.session_filter = "sessions".into();
        state.session_cursor = 3;

        switch_tab(&mut state);
        assert_eq!(state.session_popup_tab, 1);
        assert_eq!(state.session_filter, "sessions");
        assert_eq!(state.session_cursor, 3);

        switch_tab(&mut state);
        assert_eq!(state.session_popup_tab, 0);
    }

    #[test]
    fn movement_traverses_rows_and_group_boundaries() {
        let mut state = SessionsState::new();
        let mut other_group = group(&["two"]);
        other_group.cwd = Some("/other".into());
        state.session_groups = vec![group(&["one"]), other_group];

        for cursor in [1, 2, 3] {
            assert_eq!(
                handle_key(&mut state, KeyCode::Down, 1),
                SessionPopupInputResult::Moved
            );
            assert_eq!(state.session_cursor, cursor);
        }
        for cursor in [2, 1, 0] {
            assert_eq!(
                handle_key(&mut state, KeyCode::Up, 1),
                SessionPopupInputResult::Moved
            );
            assert_eq!(state.session_cursor, cursor);
        }
    }

    #[test]
    fn movement_clamps_at_first_and_last_row() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Up, 1),
            SessionPopupInputResult::Moved
        );
        assert_eq!(state.session_cursor, 0);

        state.session_cursor = 1;
        assert_eq!(
            handle_key(&mut state, KeyCode::Down, 1),
            SessionPopupInputResult::Moved
        );
        assert_eq!(state.session_cursor, 1);
    }

    #[test]
    fn page_movement_uses_root_supplied_step() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one", "two", "three", "four"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::PageDown, 3),
            SessionPopupInputResult::Moved
        );
        assert_eq!(state.session_cursor, 3);
        handle_key(&mut state, KeyCode::PageUp, 2);
        assert_eq!(state.session_cursor, 1);
    }

    #[test]
    fn filtering_plain_characters_and_backspace_reset_cursor() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one", "two"])];

        for (character, expected) in [('x', "x"), ('o', "xo"), ('n', "xon")] {
            state.session_cursor = 2;
            assert_eq!(
                handle_key(&mut state, KeyCode::Char(character), 1),
                SessionPopupInputResult::Filtered
            );
            assert_eq!(state.session_filter, expected);
            assert_eq!(state.session_cursor, 0);
        }

        for expected in ["xo", "x", ""] {
            state.session_cursor = 2;
            assert_eq!(
                handle_key(&mut state, KeyCode::Backspace, 1),
                SessionPopupInputResult::Filtered
            );
            assert_eq!(state.session_filter, expected);
            assert_eq!(state.session_cursor, 0);
        }

        state.session_cursor = 2;
        assert_eq!(
            handle_key(&mut state, KeyCode::Backspace, 1),
            SessionPopupInputResult::Filtered
        );
        assert!(state.session_filter.is_empty());
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn collapse_and_load_more_preserve_popup_semantics() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one"]), group(&["two"])];
        state.session_cursor = 2;

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter, 1),
            SessionPopupInputResult::ToggledGroup
        );
        assert!(state.popup_collapsed_groups.contains("/work"));
        assert!(!state.collapsed_groups.contains("/work"));
        assert_eq!(state.session_cursor, 1);

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter, 1),
            SessionPopupInputResult::ToggledGroup
        );
        assert!(!state.popup_collapsed_groups.contains("/work"));
        assert!(!state.collapsed_groups.contains("/work"));

        state.session_groups = vec![group(&["one"])];
        state.session_groups[0].next_cursor = Some("next".into());
        state.session_cursor = 2;
        assert_eq!(
            handle_key(&mut state, KeyCode::Enter, 1),
            SessionPopupInputResult::LoadMore {
                group_idx: 0,
                parent_path: Vec::new()
            }
        );
    }

    #[test]
    fn selection_returns_identity_for_root_owned_remote_resolution() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["local", "remote", "missing"])];
        state.session_groups[0].sessions[1].node_id = Some("node-1".into());
        state.session_groups[0].sessions[2].node = Some("remote label".into());

        for (cursor, session_id) in [(1, "local"), (2, "remote"), (3, "missing")] {
            state.session_cursor = cursor;
            assert_eq!(
                handle_key(&mut state, KeyCode::Enter, 1),
                SessionPopupInputResult::OpenSession {
                    session_id: session_id.into(),
                    cwd: Some("/session".into())
                }
            );
        }
    }

    #[test]
    fn delete_returns_exact_local_or_remote_identity() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["remote", "local"])];
        state.session_groups[0].sessions[0].node_id = Some("node-1".into());
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete, 1),
            SessionPopupInputResult::DismissRemoteSession {
                session_id: "remote".into()
            }
        );
        assert_eq!(
            handle_key(&mut state, KeyCode::Delete, 1),
            SessionPopupInputResult::DeleteSession {
                session_id: "local".into()
            }
        );
        assert!(state.session_groups.is_empty());
    }

    #[test]
    fn delete_on_header_is_not_handled_and_preserves_projection() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["one"]), group(&["two"])];
        let projection = state.visible_popup_items();

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete, 1),
            SessionPopupInputResult::NotHandled
        );
        assert_eq!(state.visible_popup_items(), projection);
    }

    #[test]
    fn escape_is_data_only_close_intent() {
        let mut state = SessionsState::new();
        assert_eq!(
            handle_key(&mut state, KeyCode::Esc, 1),
            SessionPopupInputResult::ClosePopup
        );
    }
}
