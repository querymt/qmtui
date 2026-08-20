use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::domain::mesh::{
    MeshInviteCreatedInfo, MeshStatusInfo, RemoteNodeInfo, RemoteSessionInfo,
};

const INVITE_ERROR_TTL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MeshFocus {
    #[default]
    Nodes,
    Sessions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum MeshInviteFormField {
    #[default]
    MeshName,
    Ttl,
    MaxUses,
}

pub(crate) struct MeshState {
    pub(crate) mesh_node_count: Option<u32>,
    pub(crate) mesh_status: Option<MeshStatusInfo>,
    pub(crate) mesh_nodes: Vec<RemoteNodeInfo>,
    pub(crate) remote_sessions_by_node: HashMap<String, Vec<RemoteSessionInfo>>,
    pub(crate) mesh_node_cursor: usize,
    pub(crate) remote_session_cursor: usize,
    pub(crate) mesh_focus: MeshFocus,
    pub(crate) mesh_error: Option<String>,
    pub(crate) mesh_error_until: Option<Instant>,
    pub(crate) mesh_invite: Option<MeshInviteCreatedInfo>,
    pub(crate) mesh_invite_name: String,
    pub(crate) mesh_invite_ttl: String,
    pub(crate) mesh_invite_max_uses: String,
    pub(crate) mesh_invite_form_field: MeshInviteFormField,
    pub(crate) mesh_clipboard_fallback: Option<String>,
}

impl MeshState {
    pub(crate) fn new() -> Self {
        Self {
            mesh_node_count: None,
            mesh_status: None,
            mesh_nodes: Vec::new(),
            remote_sessions_by_node: HashMap::new(),
            mesh_node_cursor: 0,
            remote_session_cursor: 0,
            mesh_focus: MeshFocus::Nodes,
            mesh_error: None,
            mesh_error_until: None,
            mesh_invite: None,
            mesh_invite_name: String::new(),
            mesh_invite_ttl: "24h".into(),
            mesh_invite_max_uses: "1".into(),
            mesh_invite_form_field: MeshInviteFormField::MeshName,
            mesh_clipboard_fallback: None,
        }
    }

    pub(crate) fn set_error(&mut self, message: impl Into<String>) {
        self.mesh_error = Some(message.into());
        self.mesh_error_until = Some(Instant::now() + INVITE_ERROR_TTL);
    }

    pub(crate) fn clear_error(&mut self) {
        self.mesh_error = None;
        self.mesh_error_until = None;
    }

    pub(crate) fn current_error(&self) -> Option<&str> {
        match (self.mesh_error.as_deref(), self.mesh_error_until) {
            (Some(message), Some(until)) if Instant::now() < until => Some(message),
            _ => None,
        }
    }

    pub(crate) fn reset_for_popup(&mut self) {
        self.mesh_focus = MeshFocus::Nodes;
        self.clear_error();
    }

    pub(crate) fn reset_invite_form(&mut self) {
        if self.mesh_invite_ttl.trim().is_empty() {
            self.mesh_invite_ttl = "24h".into();
        }
        if self.mesh_invite_max_uses.trim().is_empty() {
            self.mesh_invite_max_uses = "1".into();
        }
        self.mesh_invite_form_field = MeshInviteFormField::MeshName;
        self.clear_error();
    }

    pub(crate) fn validate_invite_form(
        &mut self,
    ) -> Option<(Option<String>, Option<String>, Option<u32>)> {
        let max_uses = match self.mesh_invite_max_uses.trim().parse::<u32>() {
            Ok(max_uses) if max_uses > 0 => max_uses,
            Ok(_) => {
                self.set_error("max uses must be at least 1");
                return None;
            }
            Err(_) => {
                self.set_error("max uses must be a number");
                return None;
            }
        };
        let ttl = match validate_invite_ttl(self.mesh_invite_ttl.trim()) {
            Ok(ttl) => ttl,
            Err(message) => {
                self.set_error(message);
                return None;
            }
        };
        let mesh_name = (!self.mesh_invite_name.trim().is_empty())
            .then(|| self.mesh_invite_name.trim().to_string());
        self.clear_error();
        Some((mesh_name, ttl, Some(max_uses)))
    }

    pub(crate) fn store_invite(&mut self, invite: MeshInviteCreatedInfo) -> String {
        let url = invite.url.clone();
        self.mesh_invite = Some(invite);
        self.mesh_clipboard_fallback = None;
        self.clear_error();
        url
    }

    pub(crate) fn invite_url(&self) -> Option<&str> {
        self.mesh_invite.as_ref().map(|invite| invite.url.as_str())
    }

    pub(crate) fn show_invite_url_fallback(&mut self) -> bool {
        let Some(url) = self.invite_url().map(str::to_string) else {
            return false;
        };
        self.mesh_clipboard_fallback = Some(url);
        true
    }

    pub(crate) fn set_clipboard_fallback(&mut self, url: String) {
        self.mesh_clipboard_fallback = Some(url);
    }

    pub(crate) fn consume_clipboard_fallback(&mut self) -> bool {
        self.mesh_clipboard_fallback.take().is_some()
    }

    pub(crate) fn replace_status(&mut self, status: MeshStatusInfo) -> (bool, u32) {
        let summary = (status.enabled, status.known_peer_count);
        self.mesh_status = Some(status);
        summary
    }

    pub(crate) fn replace_nodes(&mut self, nodes: Vec<RemoteNodeInfo>) -> (usize, Option<String>) {
        let count = nodes.len();
        self.mesh_node_count = Some(count as u32);
        self.mesh_nodes = nodes;
        if self.mesh_node_cursor >= self.mesh_nodes.len() {
            self.mesh_node_cursor = self.mesh_nodes.len().saturating_sub(1);
        }
        (count, self.selected_mesh_node_id().map(str::to_string))
    }

    pub(crate) fn replace_remote_sessions(
        &mut self,
        node_id: &str,
        sessions: Vec<RemoteSessionInfo>,
    ) -> usize {
        let count = sessions.len();
        self.remote_sessions_by_node
            .insert(node_id.to_string(), sessions);
        if self.remote_session_cursor >= count {
            self.remote_session_cursor = count.saturating_sub(1);
        }
        count
    }

    pub(crate) fn selected_mesh_node_id(&self) -> Option<&str> {
        self.mesh_nodes
            .get(self.mesh_node_cursor)
            .map(|node| node.id.as_str())
    }

    pub(crate) fn selected_mesh_node_label(&self) -> Option<&str> {
        self.mesh_nodes
            .get(self.mesh_node_cursor)
            .map(|node| node.label.as_str())
    }

    pub(crate) fn selected_remote_sessions(&self) -> &[RemoteSessionInfo] {
        self.selected_mesh_node_id()
            .and_then(|node_id| self.remote_sessions_by_node.get(node_id))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub(crate) fn selected_remote_session(&self) -> Option<&RemoteSessionInfo> {
        self.selected_remote_sessions()
            .get(self.remote_session_cursor)
    }

    pub(crate) fn toggle_focus(&mut self) {
        self.mesh_focus = match self.mesh_focus {
            MeshFocus::Nodes => MeshFocus::Sessions,
            MeshFocus::Sessions => MeshFocus::Nodes,
        };
    }

    pub(crate) fn focus_sessions(&mut self) {
        self.mesh_focus = MeshFocus::Sessions;
    }

    pub(crate) fn move_mesh_node_cursor(&mut self, delta: isize) -> Option<String> {
        let len = self.mesh_nodes.len();
        if len == 0 {
            self.mesh_node_cursor = 0;
            return None;
        }
        self.mesh_node_cursor =
            (self.mesh_node_cursor as isize + delta).rem_euclid(len as isize) as usize;
        self.remote_session_cursor = 0;
        self.selected_mesh_node_id().map(str::to_string)
    }

    pub(crate) fn move_remote_session_cursor(&mut self, delta: isize) {
        let len = self.selected_remote_sessions().len();
        if len == 0 {
            self.remote_session_cursor = 0;
            return;
        }
        self.remote_session_cursor =
            (self.remote_session_cursor as isize + delta).rem_euclid(len as isize) as usize;
    }

    pub(crate) fn move_invite_form_field(&mut self, delta: isize) {
        let current = match self.mesh_invite_form_field {
            MeshInviteFormField::MeshName => 0,
            MeshInviteFormField::Ttl => 1,
            MeshInviteFormField::MaxUses => 2,
        };
        self.mesh_invite_form_field = match (current + delta).rem_euclid(3) {
            0 => MeshInviteFormField::MeshName,
            1 => MeshInviteFormField::Ttl,
            _ => MeshInviteFormField::MaxUses,
        };
    }

    pub(crate) fn invite_form_backspace(&mut self) {
        match self.mesh_invite_form_field {
            MeshInviteFormField::MeshName => {
                self.mesh_invite_name.pop();
            }
            MeshInviteFormField::Ttl => {
                self.mesh_invite_ttl.pop();
            }
            MeshInviteFormField::MaxUses => {
                self.mesh_invite_max_uses.pop();
            }
        }
    }

    pub(crate) fn invite_form_insert(&mut self, character: char) {
        match self.mesh_invite_form_field {
            MeshInviteFormField::MeshName => self.mesh_invite_name.push(character),
            MeshInviteFormField::Ttl => self.mesh_invite_ttl.push(character),
            MeshInviteFormField::MaxUses if character.is_ascii_digit() => {
                self.mesh_invite_max_uses.push(character);
            }
            MeshInviteFormField::MaxUses => {}
        }
    }
}

fn validate_invite_ttl(value: &str) -> Result<Option<String>, String> {
    if value.is_empty() {
        return Ok(None);
    }

    let mut chars = value.char_indices().peekable();
    let mut segments = 0usize;
    while chars.peek().is_some() {
        let amount_start = chars.peek().map(|(idx, _)| *idx).unwrap_or(0);
        while matches!(chars.peek(), Some((_, c)) if c.is_ascii_digit()) {
            chars.next();
        }
        let amount_end = chars.peek().map(|(idx, _)| *idx).unwrap_or(value.len());
        if amount_start == amount_end {
            return Err("ttl must be like 30m, 1d3h, or 1d3h5m".into());
        }
        let amount = value[amount_start..amount_end]
            .parse::<u64>()
            .map_err(|_| "ttl amount is too large".to_string())?;
        if amount == 0 {
            return Err("ttl amounts must be greater than 0".into());
        }
        let Some((_, unit)) = chars.next() else {
            return Err("ttl must end with s, m, h, d, or w".into());
        };
        if !matches!(unit, 's' | 'm' | 'h' | 'd' | 'w') {
            return Err("ttl units must be s, m, h, d, or w".into());
        }
        segments += 1;
    }

    if segments == 0 {
        Err("ttl must be like 30m, 1d3h, or 1d3h5m".into())
    } else {
        Ok(Some(value.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> RemoteNodeInfo {
        RemoteNodeInfo {
            id: id.into(),
            label: format!("node {id}"),
            ..Default::default()
        }
    }

    fn session(id: &str, node_id: &str) -> RemoteSessionInfo {
        RemoteSessionInfo {
            id: id.into(),
            node_id: node_id.into(),
            ..Default::default()
        }
    }

    fn invite(url: &str) -> MeshInviteCreatedInfo {
        MeshInviteCreatedInfo {
            invite_id: "invite-1".into(),
            url: url.into(),
            qr_code: Some("QR".into()),
            expires_at: 1,
            max_uses: 1,
            mesh_name: None,
        }
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = MeshState::new();

        assert_eq!(state.mesh_node_count, None);
        assert!(state.mesh_status.is_none());
        assert!(state.mesh_nodes.is_empty());
        assert!(state.remote_sessions_by_node.is_empty());
        assert_eq!(state.mesh_node_cursor, 0);
        assert_eq!(state.remote_session_cursor, 0);
        assert_eq!(state.mesh_focus, MeshFocus::Nodes);
        assert!(state.mesh_error.is_none());
        assert!(state.mesh_error_until.is_none());
        assert!(state.mesh_invite.is_none());
        assert!(state.mesh_invite_name.is_empty());
        assert_eq!(state.mesh_invite_ttl, "24h");
        assert_eq!(state.mesh_invite_max_uses, "1");
        assert_eq!(state.mesh_invite_form_field, MeshInviteFormField::MeshName);
        assert!(state.mesh_clipboard_fallback.is_none());
    }

    #[test]
    fn popup_reset_changes_only_focus_and_error() {
        let mut state = MeshState::new();
        state.mesh_focus = MeshFocus::Sessions;
        state.set_error("stale");
        state.mesh_node_cursor = 2;
        state.remote_session_cursor = 3;
        state.mesh_invite_name = "preserved".into();

        state.reset_for_popup();

        assert_eq!(state.mesh_focus, MeshFocus::Nodes);
        assert!(state.mesh_error.is_none());
        assert!(state.mesh_error_until.is_none());
        assert_eq!(state.mesh_node_cursor, 2);
        assert_eq!(state.remote_session_cursor, 3);
        assert_eq!(state.mesh_invite_name, "preserved");
    }

    #[test]
    fn invite_reset_restores_only_blank_defaults_and_clears_error() {
        let mut state = MeshState::new();
        state.mesh_invite_ttl = "  ".into();
        state.mesh_invite_max_uses.clear();
        state.mesh_invite_form_field = MeshInviteFormField::MaxUses;
        state.set_error("stale");

        state.reset_invite_form();

        assert_eq!(state.mesh_invite_ttl, "24h");
        assert_eq!(state.mesh_invite_max_uses, "1");
        assert_eq!(state.mesh_invite_form_field, MeshInviteFormField::MeshName);
        assert!(state.mesh_error.is_none());

        state.mesh_invite_ttl = "2h".into();
        state.mesh_invite_max_uses = "4".into();
        state.reset_invite_form();
        assert_eq!(state.mesh_invite_ttl, "2h");
        assert_eq!(state.mesh_invite_max_uses, "4");
    }

    #[test]
    fn error_visibility_expires_lazily_after_five_seconds() {
        let mut state = MeshState::new();
        state.set_error("invalid invite");

        assert_eq!(state.current_error(), Some("invalid invite"));
        let until = state.mesh_error_until.expect("error deadline");
        assert!(until > Instant::now());
        assert!(until <= Instant::now() + Duration::from_secs(5));

        state.mesh_error_until = Some(Instant::now() - Duration::from_millis(1));
        assert_eq!(state.current_error(), None);
        assert_eq!(state.mesh_error.as_deref(), Some("invalid invite"));

        state.clear_error();
        assert!(state.mesh_error.is_none());
        assert!(state.mesh_error_until.is_none());
    }

    #[test]
    fn invite_validation_trims_and_normalizes_valid_values() {
        let mut state = MeshState::new();
        state.mesh_invite_name = "  Team Mesh  ".into();
        state.mesh_invite_ttl = "  1d3h5m  ".into();
        state.mesh_invite_max_uses = " 7 ".into();
        state.set_error("stale");

        assert_eq!(
            state.validate_invite_form(),
            Some((Some("Team Mesh".into()), Some("1d3h5m".into()), Some(7)))
        );
        assert!(state.mesh_error.is_none());
        assert!(state.mesh_error_until.is_none());

        state.mesh_invite_name = "  ".into();
        state.mesh_invite_ttl = "  ".into();
        state.mesh_invite_max_uses = "1".into();
        assert_eq!(state.validate_invite_form(), Some((None, None, Some(1))));
    }

    #[test]
    fn invite_validation_preserves_exact_max_use_errors_and_precedence() {
        let mut state = MeshState::new();
        state.mesh_invite_ttl = "invalid".into();
        state.mesh_invite_max_uses = "0".into();

        assert_eq!(state.validate_invite_form(), None);
        assert_eq!(
            state.mesh_error.as_deref(),
            Some("max uses must be at least 1")
        );

        state.mesh_invite_max_uses = "many".into();
        assert_eq!(state.validate_invite_form(), None);
        assert_eq!(
            state.mesh_error.as_deref(),
            Some("max uses must be a number")
        );
    }

    #[test]
    fn invite_validation_preserves_exact_ttl_errors() {
        let cases = [
            ("lol", "ttl must be like 30m, 1d3h, or 1d3h5m"),
            ("1", "ttl must end with s, m, h, d, or w"),
            ("1y", "ttl units must be s, m, h, d, or w"),
            ("0m", "ttl amounts must be greater than 0"),
            ("184467440737095516160m", "ttl amount is too large"),
        ];

        for (value, expected) in cases {
            let mut state = MeshState::new();
            state.mesh_invite_ttl = value.into();
            assert_eq!(state.validate_invite_form(), None, "value: {value}");
            assert_eq!(
                state.mesh_error.as_deref(),
                Some(expected),
                "value: {value}"
            );
        }
    }

    #[test]
    fn invite_storage_url_and_fallback_transitions_are_exact() {
        let mut state = MeshState::new();
        state.set_error("stale");
        state.set_clipboard_fallback("old".into());

        let url = state.store_invite(invite("qmt://mesh/join/token"));

        assert_eq!(url, "qmt://mesh/join/token");
        assert_eq!(state.invite_url(), Some("qmt://mesh/join/token"));
        assert!(state.mesh_error.is_none());
        assert!(state.mesh_clipboard_fallback.is_none());
        assert!(state.show_invite_url_fallback());
        assert_eq!(
            state.mesh_clipboard_fallback.as_deref(),
            Some("qmt://mesh/join/token")
        );
        assert!(state.consume_clipboard_fallback());
        assert!(!state.consume_clipboard_fallback());

        let mut empty = MeshState::new();
        assert!(!empty.show_invite_url_fallback());
        assert!(empty.mesh_clipboard_fallback.is_none());
    }

    #[test]
    fn status_replacement_returns_log_values_and_stores_full_status() {
        let mut state = MeshState::new();
        let status = MeshStatusInfo {
            enabled: true,
            peer_id: Some("peer-1".into()),
            known_peer_count: 3,
            has_invite_store: true,
            ..Default::default()
        };

        assert_eq!(state.replace_status(status), (true, 3));
        let stored = state.mesh_status.as_ref().expect("stored status");
        assert_eq!(stored.peer_id.as_deref(), Some("peer-1"));
        assert!(stored.has_invite_store);
    }

    #[test]
    fn node_replacement_preserves_valid_cursor_session_cursor_and_stale_lists() {
        let mut state = MeshState::new();
        state.mesh_node_cursor = 1;
        state.remote_session_cursor = 4;
        state
            .remote_sessions_by_node
            .insert("stale".into(), vec![session("old", "stale")]);

        let (count, selected) = state.replace_nodes(vec![node("first"), node("second")]);

        assert_eq!(count, 2);
        assert_eq!(selected.as_deref(), Some("second"));
        assert_eq!(state.mesh_node_count, Some(2));
        assert_eq!(state.mesh_node_cursor, 1);
        assert_eq!(state.remote_session_cursor, 4);
        assert_eq!(state.remote_sessions_by_node["stale"][0].id, "old");
    }

    #[test]
    fn node_replacement_clamps_only_invalid_cursor_values() {
        let mut state = MeshState::new();
        state.mesh_node_cursor = 8;
        let (_, selected) = state.replace_nodes(vec![node("first"), node("second")]);
        assert_eq!(state.mesh_node_cursor, 1);
        assert_eq!(selected.as_deref(), Some("second"));

        let (_, selected) = state.replace_nodes(Vec::new());
        assert_eq!(state.mesh_node_count, Some(0));
        assert_eq!(state.mesh_node_cursor, 0);
        assert_eq!(selected, None);
    }

    #[test]
    fn remote_session_replacement_is_per_node_and_clamps_global_cursor() {
        let mut state = MeshState::new();
        state.replace_nodes(vec![node("first"), node("second")]);
        state
            .remote_sessions_by_node
            .insert("second".into(), vec![session("preserved", "second")]);
        state.remote_session_cursor = 3;

        assert_eq!(
            state.replace_remote_sessions("first", vec![session("fresh", "first")]),
            1
        );
        assert_eq!(state.remote_session_cursor, 0);
        assert_eq!(state.remote_sessions_by_node["first"][0].id, "fresh");
        assert_eq!(state.remote_sessions_by_node["second"][0].id, "preserved");

        state.remote_session_cursor = 2;
        state.replace_remote_sessions(
            "first",
            vec![
                session("one", "first"),
                session("two", "first"),
                session("three", "first"),
            ],
        );
        assert_eq!(state.remote_session_cursor, 2);
    }

    #[test]
    fn selections_and_node_cursor_movement_wrap_and_reset_session_cursor() {
        let mut state = MeshState::new();
        state.replace_nodes(vec![node("first"), node("second")]);
        state.replace_remote_sessions("first", vec![session("a", "first")]);
        state.replace_remote_sessions("second", vec![session("b", "second")]);
        state.remote_session_cursor = 7;

        assert_eq!(state.selected_mesh_node_id(), Some("first"));
        assert_eq!(state.selected_mesh_node_label(), Some("node first"));
        assert_eq!(state.move_mesh_node_cursor(-1).as_deref(), Some("second"));
        assert_eq!(state.mesh_node_cursor, 1);
        assert_eq!(state.remote_session_cursor, 0);
        assert_eq!(
            state.selected_remote_session().map(|item| item.id.as_str()),
            Some("b")
        );
        assert_eq!(state.move_mesh_node_cursor(1).as_deref(), Some("first"));

        state.replace_nodes(Vec::new());
        state.mesh_node_cursor = 5;
        assert_eq!(state.move_mesh_node_cursor(1), None);
        assert_eq!(state.mesh_node_cursor, 0);
    }

    #[test]
    fn remote_session_cursor_wraps_and_empty_selection_resets_it() {
        let mut state = MeshState::new();
        state.replace_nodes(vec![node("node")]);
        state.replace_remote_sessions("node", vec![session("one", "node"), session("two", "node")]);

        state.move_remote_session_cursor(-1);
        assert_eq!(state.remote_session_cursor, 1);
        state.move_remote_session_cursor(1);
        assert_eq!(state.remote_session_cursor, 0);

        state.replace_remote_sessions("node", Vec::new());
        state.remote_session_cursor = 9;
        state.move_remote_session_cursor(1);
        assert_eq!(state.remote_session_cursor, 0);
        assert!(state.selected_remote_sessions().is_empty());
        assert!(state.selected_remote_session().is_none());
    }

    #[test]
    fn focus_and_invite_field_movement_wrap() {
        let mut state = MeshState::new();
        state.toggle_focus();
        assert_eq!(state.mesh_focus, MeshFocus::Sessions);
        state.toggle_focus();
        assert_eq!(state.mesh_focus, MeshFocus::Nodes);
        state.focus_sessions();
        assert_eq!(state.mesh_focus, MeshFocus::Sessions);

        state.move_invite_form_field(-1);
        assert_eq!(state.mesh_invite_form_field, MeshInviteFormField::MaxUses);
        state.move_invite_form_field(1);
        assert_eq!(state.mesh_invite_form_field, MeshInviteFormField::MeshName);
        state.move_invite_form_field(1);
        assert_eq!(state.mesh_invite_form_field, MeshInviteFormField::Ttl);
    }

    #[test]
    fn invite_input_targets_selected_field_and_filters_max_uses() {
        let mut state = MeshState::new();
        state.invite_form_insert('M');
        state.invite_form_insert('é');
        assert_eq!(state.mesh_invite_name, "Mé");
        state.invite_form_backspace();
        assert_eq!(state.mesh_invite_name, "M");

        state.mesh_invite_form_field = MeshInviteFormField::Ttl;
        state.mesh_invite_ttl.clear();
        state.invite_form_insert('2');
        state.invite_form_insert('h');
        assert_eq!(state.mesh_invite_ttl, "2h");
        state.invite_form_backspace();
        assert_eq!(state.mesh_invite_ttl, "2");

        state.mesh_invite_form_field = MeshInviteFormField::MaxUses;
        state.mesh_invite_max_uses.clear();
        state.invite_form_insert('x');
        state.invite_form_insert('7');
        assert_eq!(state.mesh_invite_max_uses, "7");
        state.invite_form_backspace();
        assert!(state.mesh_invite_max_uses.is_empty());
    }
}
