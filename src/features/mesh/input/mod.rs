use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::mesh_state::{MeshFocus, MeshState};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MeshInputResult {
    NotHandled,
    Close,
    ToggleFocus,
    MoveNode { refresh_node_id: Option<String> },
    MoveRemoteSession,
    FocusSessions { node_id: Option<String> },
    AttachRemoteSession { node_id: String, session_id: String },
    CreateRemoteSession { node_id: String },
    RefreshNodes,
    InviteEdited,
    SubmitInvite,
    BackToInviteForm,
    ShowInviteUrl,
    CopyInviteUrl { url: String },
}

pub(crate) fn handle_popup_key(state: &mut MeshState, key: KeyEvent) -> MeshInputResult {
    match key.code {
        KeyCode::Esc => MeshInputResult::Close,
        KeyCode::Tab | KeyCode::Right | KeyCode::Left => {
            state.toggle_focus();
            MeshInputResult::ToggleFocus
        }
        KeyCode::Up => move_cursor(state, -1),
        KeyCode::Down => move_cursor(state, 1),
        KeyCode::Char('r') => MeshInputResult::RefreshNodes,
        KeyCode::Enter => match state.mesh_focus {
            MeshFocus::Nodes => {
                state.focus_sessions();
                MeshInputResult::FocusSessions {
                    node_id: state.selected_mesh_node_id().map(str::to_string),
                }
            }
            MeshFocus::Sessions => state
                .selected_remote_session()
                .map(|session| MeshInputResult::AttachRemoteSession {
                    node_id: session.node_id.clone(),
                    session_id: session.id.clone(),
                })
                .unwrap_or(MeshInputResult::NotHandled),
        },
        KeyCode::Char('n') => state
            .selected_mesh_node_id()
            .map(|node_id| MeshInputResult::CreateRemoteSession {
                node_id: node_id.to_string(),
            })
            .unwrap_or(MeshInputResult::NotHandled),
        _ => MeshInputResult::NotHandled,
    }
}

fn move_cursor(state: &mut MeshState, delta: isize) -> MeshInputResult {
    match state.mesh_focus {
        MeshFocus::Nodes => MeshInputResult::MoveNode {
            refresh_node_id: state.move_mesh_node_cursor(delta),
        },
        MeshFocus::Sessions => {
            state.move_remote_session_cursor(delta);
            MeshInputResult::MoveRemoteSession
        }
    }
}

pub(crate) fn handle_invite_key(state: &mut MeshState, key: KeyEvent) -> MeshInputResult {
    match key.code {
        KeyCode::Esc => MeshInputResult::Close,
        KeyCode::Up => {
            state.move_invite_form_field(-1);
            MeshInputResult::InviteEdited
        }
        KeyCode::Down | KeyCode::Tab => {
            state.move_invite_form_field(1);
            MeshInputResult::InviteEdited
        }
        KeyCode::Backspace => {
            state.invite_form_backspace();
            MeshInputResult::InviteEdited
        }
        KeyCode::Enter => MeshInputResult::SubmitInvite,
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.invite_form_insert(character);
            MeshInputResult::InviteEdited
        }
        _ => MeshInputResult::NotHandled,
    }
}

pub(crate) fn handle_invite_qr_key(state: &mut MeshState, key: KeyEvent) -> MeshInputResult {
    match key.code {
        KeyCode::Esc => MeshInputResult::BackToInviteForm,
        KeyCode::Char('u') => {
            if state.invite_url().is_some() {
                MeshInputResult::ShowInviteUrl
            } else {
                MeshInputResult::NotHandled
            }
        }
        KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => state
            .invite_url()
            .map(|url| MeshInputResult::CopyInviteUrl {
                url: url.to_string(),
            })
            .unwrap_or(MeshInputResult::NotHandled),
        _ => MeshInputResult::NotHandled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::{MeshInviteCreatedInfo, RemoteNodeInfo, RemoteSessionInfo};
    use crate::mesh_state::MeshInviteFormField;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn ctrl(character: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(character), KeyModifiers::CONTROL)
    }

    fn state() -> MeshState {
        let mut state = MeshState::new();
        state.mesh_nodes = vec![
            RemoteNodeInfo {
                id: "node-1".into(),
                ..Default::default()
            },
            RemoteNodeInfo {
                id: "node-2".into(),
                ..Default::default()
            },
        ];
        state.remote_sessions_by_node.insert(
            "node-1".into(),
            vec![RemoteSessionInfo {
                id: "session-1".into(),
                node_id: "node-1".into(),
                ..Default::default()
            }],
        );
        state
    }

    #[test]
    fn focus_and_cursor_results_preserve_refresh_identity() {
        let mut state = state();
        assert_eq!(
            handle_popup_key(&mut state, key(KeyCode::Down)),
            MeshInputResult::MoveNode {
                refresh_node_id: Some("node-2".into())
            }
        );
        assert_eq!(state.remote_session_cursor, 0);
        assert_eq!(
            handle_popup_key(&mut state, key(KeyCode::Tab)),
            MeshInputResult::ToggleFocus
        );
        assert_eq!(state.mesh_focus, MeshFocus::Sessions);
        assert_eq!(
            handle_popup_key(&mut state, key(KeyCode::Up)),
            MeshInputResult::MoveRemoteSession
        );
    }

    #[test]
    fn enter_attach_create_and_refresh_return_exact_payloads() {
        let mut state = state();
        assert_eq!(
            handle_popup_key(&mut state, key(KeyCode::Enter)),
            MeshInputResult::FocusSessions {
                node_id: Some("node-1".into())
            }
        );
        assert_eq!(
            handle_popup_key(&mut state, key(KeyCode::Enter)),
            MeshInputResult::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "session-1".into()
            }
        );
        assert_eq!(
            handle_popup_key(&mut state, key(KeyCode::Char('n'))),
            MeshInputResult::CreateRemoteSession {
                node_id: "node-1".into()
            }
        );
        assert_eq!(
            handle_popup_key(&mut state, key(KeyCode::Char('r'))),
            MeshInputResult::RefreshNodes
        );
    }

    #[test]
    fn invite_form_owns_field_movement_and_editing_but_not_submission() {
        let mut state = MeshState::new();
        assert_eq!(
            handle_invite_key(&mut state, key(KeyCode::Down)),
            MeshInputResult::InviteEdited
        );
        assert_eq!(state.mesh_invite_form_field, MeshInviteFormField::Ttl);
        handle_invite_key(&mut state, key(KeyCode::Char('1')));
        assert_eq!(state.mesh_invite_ttl, "24h1");
        handle_invite_key(&mut state, key(KeyCode::Backspace));
        assert_eq!(state.mesh_invite_ttl, "24h");
        assert_eq!(
            handle_invite_key(&mut state, key(KeyCode::Enter)),
            MeshInputResult::SubmitInvite
        );
    }

    #[test]
    fn invite_result_returns_back_show_and_copy_intents() {
        let mut state = MeshState::new();
        state.mesh_invite = Some(MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: "qmt://mesh/join/token".into(),
            qr_code: None,
            expires_at: 1,
            max_uses: 1,
            mesh_name: None,
        });
        assert_eq!(
            handle_invite_qr_key(&mut state, key(KeyCode::Char('u'))),
            MeshInputResult::ShowInviteUrl
        );
        assert!(state.mesh_clipboard_fallback.is_none());
        assert_eq!(
            handle_invite_qr_key(&mut state, ctrl('y')),
            MeshInputResult::CopyInviteUrl {
                url: "qmt://mesh/join/token".into()
            }
        );
        assert_eq!(
            handle_invite_qr_key(&mut state, key(KeyCode::Esc)),
            MeshInputResult::BackToInviteForm
        );
    }
}
