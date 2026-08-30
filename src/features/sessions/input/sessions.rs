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
        group_at("/work", ids)
    }

    fn group_at(cwd: &str, ids: &[&str]) -> SessionGroup {
        SessionGroup {
            cwd: Some(cwd.into()),
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
    fn down_moves_cursor_forward() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1", "s2"])];

        for expected_cursor in [1, 2] {
            assert_eq!(
                handle_key(&mut state, KeyCode::Down),
                SessionsInputResult::Moved {
                    keep_visible_from_above: false
                }
            );
            assert_eq!(state.session_cursor, expected_cursor);
        }
    }

    #[test]
    fn down_from_last_item_reaches_button_slot() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Down),
            SessionsInputResult::Moved {
                keep_visible_from_above: false
            }
        );
        assert_eq!(state.session_cursor, 2);
    }

    #[test]
    fn up_moves_cursor_back() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1", "s2"])];
        state.session_cursor = 2;

        assert_eq!(
            handle_key(&mut state, KeyCode::Up),
            SessionsInputResult::Moved {
                keep_visible_from_above: true
            }
        );
        assert_eq!(state.session_cursor, 1);
    }

    #[test]
    fn up_does_not_go_below_zero() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Up),
            SessionsInputResult::Moved {
                keep_visible_from_above: true
            }
        );
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn enter_on_header_collapses_group() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::ToggledGroup
        );
        assert!(state.collapsed_groups.contains("/work"));
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn enter_on_collapsed_header_expands_group() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];
        state.collapsed_groups.insert("/work".into());

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::ToggledGroup
        );
        assert!(!state.collapsed_groups.contains("/work"));
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn show_more_returns_exact_intent_and_resets_browser() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1", "s2", "s3", "s4"])];
        state.session_filter = "s".into();
        state.session_cursor = 4;
        state.session_popup_tab = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenSessionPopup
        );
        assert_eq!(state.session_cursor, 0);
        assert!(state.session_filter.is_empty());
        assert_eq!(state.session_popup_tab, 0);
    }

    #[test]
    fn paginated_show_more_returns_exact_intent() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1", "s2", "s3"])];
        state.session_groups[0].next_cursor = Some("cursor-1".into());
        state.session_cursor = 4;

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenSessionPopup
        );
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn local_selection_returns_exact_payload_and_effective_cwd() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group_at("/group", &["one"])];
        state.session_groups[0].sessions[0].cwd = Some("/session".into());
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenSession {
                session_id: "one".into(),
                cwd: Some("/session".into())
            }
        );
    }

    #[test]
    fn remote_selection_returns_exact_payload_and_preserves_node_id() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group_at("/group", &["remote-1"])];
        state.session_groups[0].sessions[0].cwd = Some("/remote".into());
        state.session_groups[0].sessions[0].node_id = Some("node-1".into());
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenSession {
                session_id: "remote-1".into(),
                cwd: Some("/remote".into())
            }
        );
        assert_eq!(
            state.session_groups[0].sessions[0].node_id.as_deref(),
            Some("node-1")
        );
    }

    #[test]
    fn ctrl_o_on_expandable_root_requests_children() {
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
    }

    #[test]
    fn ctrl_o_on_expanded_root_collapses_without_loading_session() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["root"])];
        state.session_groups[0].sessions[0].fork_count = 1;
        state.expanded_session_children.insert("root".into());
        state.session_cursor = 1;

        assert_eq!(
            toggle_selected_session_children(&mut state),
            SessionsInputResult::NotHandled
        );
        assert!(!state.expanded_session_children.contains("root"));
    }

    #[test]
    fn delete_on_remote_session_returns_dismiss_action_and_removes() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["remote-1", "s2"])];
        state.session_groups[0].sessions[0].node_id = Some("node-1".into());
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete),
            SessionsInputResult::DismissRemoteSession {
                session_id: "remote-1".into()
            }
        );
        assert_eq!(state.session_groups[0].sessions.len(), 1);
        assert_eq!(state.session_groups[0].sessions[0].session_id, "s2");
    }

    #[test]
    fn enter_on_expandable_root_returns_load_action() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group_at("/a", &["root"])];
        state.session_groups[0].sessions[0].fork_count = 1;
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenSession {
                session_id: "root".into(),
                cwd: Some("/a".into())
            }
        );
        assert!(!state.expanded_session_children.contains("root"));
    }

    #[test]
    fn plain_o_still_filters_instead_of_toggling_forks() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["root"])];
        state.session_groups[0].sessions[0].fork_count = 1;
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Char('o')),
            SessionsInputResult::Filtered
        );
        assert_eq!(state.session_filter, "o");
        assert_eq!(state.session_cursor, 0);
        assert!(!state.expanded_session_children.contains("root"));
    }

    #[test]
    fn delete_on_session_returns_delete_action_and_removes() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1", "s2"])];
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete),
            SessionsInputResult::DeleteSession {
                session_id: "s1".into()
            }
        );
        assert_eq!(state.session_groups[0].sessions.len(), 1);
        assert_eq!(state.session_groups[0].sessions[0].session_id, "s2");
    }

    #[test]
    fn delete_removes_empty_group() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["only"])];
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete),
            SessionsInputResult::DeleteSession {
                session_id: "only".into()
            }
        );
        assert!(state.session_groups.is_empty());
    }

    #[test]
    fn delete_on_header_is_noop() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete),
            SessionsInputResult::NotHandled
        );
        assert_eq!(state.session_groups[0].sessions.len(), 1);
        assert_eq!(state.session_groups[0].sessions[0].session_id, "s1");
    }

    #[test]
    fn char_appends_to_filter_and_resets_cursor() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];
        state.session_cursor = 1;

        assert_eq!(
            handle_key(&mut state, KeyCode::Char('x')),
            SessionsInputResult::Filtered
        );
        assert_eq!(state.session_filter, "x");
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn backspace_removes_last_filter_char_and_resets_cursor() {
        let mut state = SessionsState::new();
        state.session_filter = "ab".into();
        state.session_cursor = 2;

        assert_eq!(
            handle_key(&mut state, KeyCode::Backspace),
            SessionsInputResult::Filtered
        );
        assert_eq!(state.session_filter, "a");
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn backspace_on_empty_filter_is_noop() {
        let mut state = SessionsState::new();

        assert_eq!(
            handle_key(&mut state, KeyCode::Backspace),
            SessionsInputResult::Filtered
        );
        assert!(state.session_filter.is_empty());
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn collapse_clamps_cursor_when_selected_row_disappears() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1", "s2"])];

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::ToggledGroup
        );
        assert!(state.collapsed_groups.contains("/work"));
        assert_eq!(state.visible_start_items().len(), 1);
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn down_does_not_exceed_button_slot() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];
        state.session_cursor = 2;

        assert_eq!(
            handle_key(&mut state, KeyCode::Down),
            SessionsInputResult::Moved {
                keep_visible_from_above: false
            }
        );
        assert_eq!(state.session_cursor, 2);
    }

    #[test]
    fn down_reaches_button_when_no_sessions() {
        let mut state = SessionsState::new();

        assert_eq!(
            handle_key(&mut state, KeyCode::Down),
            SessionsInputResult::Moved {
                keep_visible_from_above: false
            }
        );
        assert_eq!(state.session_cursor, 0);
        assert!(state.visible_start_items().is_empty());
    }

    #[test]
    fn enter_on_button_slot_returns_new_session() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];
        state.session_cursor = 2;

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenNewSession
        );
    }

    #[test]
    fn enter_on_button_slot_no_sessions_returns_new_session() {
        let mut state = SessionsState::new();

        assert_eq!(
            handle_key(&mut state, KeyCode::Enter),
            SessionsInputResult::OpenNewSession
        );
    }

    #[test]
    fn delete_on_button_slot_is_noop() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group(&["s1"])];
        state.session_cursor = 2;

        assert_eq!(
            handle_key(&mut state, KeyCode::Delete),
            SessionsInputResult::NotHandled
        );
        assert_eq!(state.session_groups[0].sessions.len(), 1);
        assert_eq!(state.session_groups[0].sessions[0].session_id, "s1");
    }
}
