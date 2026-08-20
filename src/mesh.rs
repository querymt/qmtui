use crate::app::{App, Popup};
use crate::command::Command;
use crate::diagnostics::LogLevel;
use crate::domain::mesh::{
    MeshInviteCreatedInfo, MeshNodesInfo, MeshStatusInfo, RemoteSessionAttachInfo,
    RemoteSessionListInfo,
};

impl App {
    pub fn open_mesh_popup(&mut self) {
        self.popup = Popup::Mesh;
        self.mesh.reset_for_popup();
        self.set_status(LogLevel::Debug, "mesh", "refreshing mesh");
    }

    pub fn open_mesh_invite_form(&mut self) {
        self.popup = Popup::MeshInvite;
        self.mesh.reset_invite_form();
    }

    pub fn apply_mesh_invite_created(&mut self, invite: MeshInviteCreatedInfo) {
        let url = self.mesh.store_invite(invite);
        self.popup = Popup::MeshInviteQr;
        self.set_status(LogLevel::Info, "mesh", "mesh invite created");
        self.push_log(LogLevel::Info, "mesh", format!("mesh invite: {url}"));
    }

    pub fn mesh_invite_form_command(&mut self) -> Option<Command> {
        let (mesh_name, ttl, max_uses) = self.mesh.validate_invite_form()?;
        Some(Command::CreateMeshInvite {
            mesh_name,
            ttl,
            max_uses,
        })
    }

    pub fn apply_mesh_status(&mut self, status: MeshStatusInfo) {
        let (enabled, peer_count) = self.mesh.replace_status(status);
        self.push_log(
            LogLevel::Debug,
            "mesh",
            format!("mesh status: enabled={enabled}, peers={peer_count}"),
        );
    }

    pub fn apply_mesh_nodes(&mut self, nodes: MeshNodesInfo) -> Vec<Command> {
        let (count, selected_node_id) = self.mesh.replace_nodes(nodes.nodes);
        self.push_log(LogLevel::Info, "mesh", format!("mesh nodes: {count}"));
        selected_node_id
            .map(|node_id| Command::ListRemoteSessions {
                node_id,
                offset: 0,
                limit: 50,
            })
            .into_iter()
            .collect()
    }

    pub fn apply_remote_sessions(&mut self, list: RemoteSessionListInfo) {
        for session in &list.sessions {
            self.remember_remote_session_location(
                &session.id,
                &session.node_id,
                session.cwd.clone(),
            );
        }
        let count = self
            .mesh
            .replace_remote_sessions(&list.node_id, list.sessions);
        self.push_log(
            LogLevel::Info,
            "mesh",
            format!("remote sessions: {count} for {}", list.node_id),
        );
    }

    pub fn apply_remote_session_attached(
        &mut self,
        attached: RemoteSessionAttachInfo,
    ) -> Vec<Command> {
        // ACP session/load is authoritative for history and typed config; applying this
        // extension snapshot/config directly would duplicate replay or configuration.
        self.remember_remote_session_node(&attached.session_id, &attached.node_id);
        if attached.attached {
            self.popup = Popup::None;
            self.set_status(LogLevel::Info, "mesh", "remote session attached");
            Command::load_session_commands(
                attached.session_id.clone(),
                self.session_remote_cwd(&attached.session_id),
                self.agent_id.clone(),
            )
            .into()
        } else {
            self.set_status(LogLevel::Info, "mesh", "remote session created");
            vec![Command::ListRemoteSessions {
                node_id: attached.node_id,
                offset: 0,
                limit: 50,
            }]
        }
    }
}
