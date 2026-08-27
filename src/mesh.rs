use crate::app::App;
use crate::application::Effect;
use crate::command::Command;
use crate::diagnostics::LogLevel;
use crate::domain::mesh::{
    MeshInviteCreatedInfo, MeshNodesInfo, MeshStatusInfo, RemoteSessionAttachInfo,
    RemoteSessionListInfo,
};
use crate::mesh_state::{MeshAction, MeshContext, MeshCoordination, MeshOutcome};
use crate::navigation_state::Popup;

impl App {
    pub fn open_mesh_popup(&mut self) {
        self.navigation.popup = Popup::Mesh;
        self.mesh.reset_for_popup();
        self.set_status(LogLevel::Debug, "mesh", "refreshing mesh");
    }

    pub fn open_mesh_invite_form(&mut self) {
        self.navigation.popup = Popup::MeshInvite;
        self.mesh.reset_invite_form();
    }

    pub fn apply_mesh_invite_created(&mut self, invite: MeshInviteCreatedInfo) {
        let outcome = self
            .mesh
            .reduce(MeshAction::InviteCreated(invite), MeshContext::default());
        let effects = self.apply_mesh_outcome(outcome);
        debug_assert!(effects.is_empty());
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
        let outcome = self
            .mesh
            .reduce(MeshAction::Status(status), MeshContext::default());
        let effects = self.apply_mesh_outcome(outcome);
        debug_assert!(effects.is_empty());
    }

    pub fn apply_mesh_nodes(&mut self, nodes: MeshNodesInfo) -> Vec<Command> {
        let outcome = self
            .mesh
            .reduce(MeshAction::Nodes(nodes), MeshContext::default());
        self.apply_mesh_outcome(outcome)
    }

    pub fn apply_remote_sessions(&mut self, list: RemoteSessionListInfo) {
        let outcome = self
            .mesh
            .reduce(MeshAction::RemoteSessions(list), MeshContext::default());
        let effects = self.apply_mesh_outcome(outcome);
        debug_assert!(effects.is_empty());
    }

    pub fn apply_remote_session_attached(
        &mut self,
        attached: RemoteSessionAttachInfo,
    ) -> Vec<Command> {
        // ACP session/load remains authoritative for history and typed config.
        let context = MeshContext {
            agent_id: self.sessions.agent_id.clone(),
            remote_session_cwd: self.sessions.session_remote_cwd(&attached.session_id),
        };
        let outcome = self
            .mesh
            .reduce(MeshAction::RemoteSessionAttached(attached), context);
        self.apply_mesh_outcome(outcome)
    }

    pub(crate) fn apply_mesh_clipboard_result(&mut self, success: bool) -> Vec<Effect> {
        let outcome = self.mesh.reduce(
            MeshAction::ClipboardFinished { success },
            MeshContext::default(),
        );
        self.apply_mesh_outcome(outcome)
            .into_iter()
            .map(Effect::Command)
            .collect()
    }

    fn apply_mesh_outcome(&mut self, outcome: MeshOutcome) -> Vec<Command> {
        for coordination in outcome.coordination {
            match coordination {
                MeshCoordination::Log { level, message } => {
                    self.push_log(level, "mesh", message);
                }
                MeshCoordination::Status { level, message } => {
                    self.set_status(level, "mesh", message);
                }
                MeshCoordination::SetPopup(popup) => self.navigation.popup = popup,
                MeshCoordination::RememberRemoteSession {
                    session_id,
                    node_id,
                    cwd,
                } => self
                    .sessions
                    .remember_remote_session_location(&session_id, &node_id, cwd),
            }
        }
        outcome
            .effects
            .into_iter()
            .map(|effect| match effect {
                Effect::Command(command) => command,
                _ => unreachable!("mesh reducers only emit command effects"),
            })
            .collect()
    }
}
