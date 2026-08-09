use std::time::{Duration, Instant};

use crate::app::{App, LogLevel, Popup};
use crate::command::Command;
use crate::protocol::{
    MeshInviteCreatedInfo, MeshNodesInfo, MeshStatusInfo, RemoteSessionInfo, RemoteSessionListInfo,
};

const INVITE_ERROR_TTL: Duration = Duration::from_secs(5);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshFocus {
    #[default]
    Nodes,
    Sessions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshInviteFormField {
    #[default]
    MeshName,
    Ttl,
    MaxUses,
}

impl App {
    pub fn set_mesh_error(&mut self, message: impl Into<String>) {
        self.mesh_error = Some(message.into());
        self.mesh_error_until = Some(Instant::now() + INVITE_ERROR_TTL);
    }

    pub fn clear_mesh_error(&mut self) {
        self.mesh_error = None;
        self.mesh_error_until = None;
    }

    pub fn current_mesh_error(&self) -> Option<&str> {
        match (self.mesh_error.as_deref(), self.mesh_error_until) {
            (Some(message), Some(until)) if Instant::now() < until => Some(message),
            _ => None,
        }
    }

    pub fn open_mesh_popup(&mut self) {
        self.popup = Popup::Mesh;
        self.mesh_focus = MeshFocus::Nodes;
        self.clear_mesh_error();
        self.set_status(LogLevel::Debug, "mesh", "refreshing mesh");
    }

    pub fn open_mesh_invite_form(&mut self) {
        self.popup = Popup::MeshInvite;
        if self.mesh_invite_ttl.trim().is_empty() {
            self.mesh_invite_ttl = "24h".into();
        }
        if self.mesh_invite_max_uses.trim().is_empty() {
            self.mesh_invite_max_uses = "1".into();
        }
        self.mesh_invite_form_field = MeshInviteFormField::MeshName;
        self.clear_mesh_error();
    }

    pub fn apply_mesh_invite_created(&mut self, invite: MeshInviteCreatedInfo) {
        let url = invite.url.clone();
        self.mesh_invite = Some(invite);
        self.popup = Popup::MeshInviteQr;
        self.mesh_clipboard_fallback = None;
        self.clear_mesh_error();
        self.set_status(LogLevel::Info, "mesh", "mesh invite created");
        self.push_log(LogLevel::Info, "mesh", format!("mesh invite: {url}"));
    }

    pub fn mesh_invite_form_command(&mut self) -> Option<Command> {
        let max_uses = match self.mesh_invite_max_uses.trim().parse::<u32>() {
            Ok(max_uses) if max_uses > 0 => max_uses,
            Ok(_) => {
                self.set_mesh_error("max uses must be at least 1");
                return None;
            }
            Err(_) => {
                self.set_mesh_error("max uses must be a number");
                return None;
            }
        };
        let ttl = match validate_invite_ttl(self.mesh_invite_ttl.trim()) {
            Ok(ttl) => ttl,
            Err(message) => {
                self.set_mesh_error(message);
                return None;
            }
        };
        let mesh_name = (!self.mesh_invite_name.trim().is_empty())
            .then(|| self.mesh_invite_name.trim().to_string());
        self.clear_mesh_error();
        Some(Command::CreateMeshInvite {
            mesh_name,
            ttl,
            max_uses: Some(max_uses),
        })
    }

    pub fn mesh_invite_url(&self) -> Option<&str> {
        self.mesh_invite.as_ref().map(|invite| invite.url.as_str())
    }

    pub fn apply_mesh_status(&mut self, status: MeshStatusInfo) {
        let enabled = status.enabled;
        let peer_count = status.known_peer_count;
        self.mesh_status = Some(status);
        self.push_log(
            LogLevel::Debug,
            "mesh",
            format!("mesh status: enabled={enabled}, peers={peer_count}"),
        );
    }

    pub fn apply_mesh_nodes(&mut self, nodes: MeshNodesInfo) -> Vec<Command> {
        self.mesh_node_count = Some(nodes.nodes.len() as u32);
        self.mesh_nodes = nodes.nodes;
        if self.mesh_node_cursor >= self.mesh_nodes.len() {
            self.mesh_node_cursor = self.mesh_nodes.len().saturating_sub(1);
        }
        self.push_log(
            LogLevel::Info,
            "mesh",
            format!("mesh nodes: {}", self.mesh_nodes.len()),
        );
        self.selected_mesh_node_id()
            .map(|node_id| Command::ListRemoteSessions {
                node_id: node_id.to_string(),
                offset: 0,
                limit: 50,
            })
            .into_iter()
            .collect()
    }

    pub fn apply_remote_sessions(&mut self, list: RemoteSessionListInfo) {
        let count = list.sessions.len();
        self.remote_sessions_by_node
            .insert(list.node_id.clone(), list.sessions);
        if self.remote_session_cursor >= count {
            self.remote_session_cursor = count.saturating_sub(1);
        }
        self.push_log(
            LogLevel::Info,
            "mesh",
            format!("remote sessions: {count} for {}", list.node_id),
        );
    }

    pub fn apply_remote_session_attached(
        &mut self,
        session_id: &str,
        node_id: &str,
        attached: bool,
    ) -> Vec<Command> {
        self.remember_remote_session_node(session_id, node_id);
        if attached {
            self.popup = Popup::None;
            self.set_status(LogLevel::Info, "mesh", "remote session attached");
            Command::load_session_commands(
                session_id.to_string(),
                self.current_session_cwd(),
                self.agent_id.clone(),
            )
            .into()
        } else {
            self.set_status(LogLevel::Info, "mesh", "remote session created");
            vec![Command::ListRemoteSessions {
                node_id: node_id.to_string(),
                offset: 0,
                limit: 50,
            }]
        }
    }

    pub fn selected_mesh_node_id(&self) -> Option<&str> {
        self.mesh_nodes
            .get(self.mesh_node_cursor)
            .map(|node| node.id.as_str())
    }

    pub fn selected_mesh_node_label(&self) -> Option<&str> {
        self.mesh_nodes
            .get(self.mesh_node_cursor)
            .map(|node| node.label.as_str())
    }

    pub fn selected_remote_sessions(&self) -> &[RemoteSessionInfo] {
        self.selected_mesh_node_id()
            .and_then(|node_id| self.remote_sessions_by_node.get(node_id))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn selected_remote_session(&self) -> Option<&RemoteSessionInfo> {
        self.selected_remote_sessions()
            .get(self.remote_session_cursor)
    }

    pub fn move_mesh_node_cursor(&mut self, delta: isize) -> Option<Command> {
        let len = self.mesh_nodes.len();
        if len == 0 {
            self.mesh_node_cursor = 0;
            return None;
        }
        self.mesh_node_cursor =
            (self.mesh_node_cursor as isize + delta).rem_euclid(len as isize) as usize;
        self.remote_session_cursor = 0;
        self.selected_mesh_node_id()
            .map(|node_id| Command::ListRemoteSessions {
                node_id: node_id.to_string(),
                offset: 0,
                limit: 50,
            })
    }

    pub fn move_remote_session_cursor(&mut self, delta: isize) {
        let len = self.selected_remote_sessions().len();
        if len == 0 {
            self.remote_session_cursor = 0;
            return;
        }
        self.remote_session_cursor =
            (self.remote_session_cursor as isize + delta).rem_euclid(len as isize) as usize;
    }
}
