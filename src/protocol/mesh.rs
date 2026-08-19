use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshStatusDto {
    pub enabled: bool,
    #[serde(default)]
    pub peer_id: Option<String>,
    #[serde(default)]
    pub transport: Option<String>,
    #[serde(default)]
    pub known_peer_count: u32,
    #[serde(default)]
    pub has_invite_store: bool,
    #[serde(default)]
    pub has_mesh_state_store: bool,
    #[serde(default)]
    pub scopes: Vec<MeshScopeDto>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshScopeDto {
    pub kind: String,
    pub id: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteNodeDto {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub active_sessions: u32,
    #[serde(default)]
    pub transport: String,
    #[serde(default)]
    pub last_seen_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshNodesDto {
    #[serde(default)]
    pub nodes: Vec<RemoteNodeDto>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteSessionDto {
    pub id: String,
    pub node_id: String,
    #[serde(default)]
    pub node_label: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RemoteSessionListDto {
    pub node_id: String,
    #[serde(default)]
    pub sessions: Vec<RemoteSessionDto>,
    #[serde(default)]
    pub next_offset: Option<u32>,
    #[serde(default)]
    pub total_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RemoteSessionAttachDto {
    pub session_id: String,
    pub node_id: String,
    #[serde(default)]
    pub attached: bool,
    #[serde(default)]
    pub config_options: Vec<serde_json::Value>,
    #[serde(default)]
    pub snapshot: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MeshInviteCreatedDto {
    pub invite_id: String,
    pub url: String,
    #[serde(default)]
    pub qr_code: Option<String>,
    #[serde(default)]
    pub expires_at: u64,
    #[serde(default)]
    pub max_uses: u32,
    #[serde(default)]
    pub mesh_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{
        MeshInviteCreatedDto, MeshNodesDto, MeshStatusDto, RemoteNodeDto, RemoteSessionAttachDto,
        RemoteSessionDto, RemoteSessionListDto,
    };
    use serde_json::json;

    #[test]
    fn mesh_dtos_deserialize_representative_values_and_ignore_unknown_fields() {
        let status: MeshStatusDto = serde_json::from_value(json!({
            "enabled": true,
            "peer_id": "peer-1",
            "transport": "webrtc",
            "known_peer_count": 2,
            "has_invite_store": true,
            "has_mesh_state_store": true,
            "scopes": [{ "kind": "team", "id": "scope-1" }],
            "unknown": "ignored"
        }))
        .unwrap();
        assert!(status.enabled);
        assert_eq!(status.scopes[0].kind, "team");
        assert_eq!(status.scopes[0].id, "scope-1");

        let nodes: MeshNodesDto = serde_json::from_value(json!({
            "nodes": [{
                "id": "node-1",
                "label": "Framework",
                "capabilities": ["sessions"],
                "active_sessions": 3,
                "transport": "relay",
                "last_seen_at": "2025-01-01T00:00:00Z"
            }]
        }))
        .unwrap();
        assert_eq!(nodes.nodes[0].id, "node-1");
        assert_eq!(nodes.nodes[0].capabilities, ["sessions"]);
        assert_eq!(nodes.nodes[0].active_sessions, 3);

        let node_defaults: RemoteNodeDto = serde_json::from_value(json!({
            "id": "node-2",
            "label": "Defaults"
        }))
        .unwrap();
        assert!(node_defaults.capabilities.is_empty());
        assert_eq!(node_defaults.active_sessions, 0);
        assert!(node_defaults.transport.is_empty());
        assert_eq!(node_defaults.last_seen_at, None);

        let sessions: RemoteSessionListDto = serde_json::from_value(json!({
            "node_id": "node-1",
            "sessions": [{
                "id": "session-1",
                "node_id": "node-1",
                "node_label": "Framework",
                "title": "Fix boundary",
                "cwd": "/repo",
                "updated_at": "now",
                "profile_id": "profile-1",
                "model_id": "model-1"
            }],
            "next_offset": 50,
            "total_count": 51
        }))
        .unwrap();
        assert_eq!(sessions.sessions[0].model_id.as_deref(), Some("model-1"));
        assert_eq!(sessions.next_offset, Some(50));
        assert_eq!(sessions.total_count, 51);

        let session_defaults: RemoteSessionDto = serde_json::from_value(json!({
            "id": "session-2",
            "node_id": "node-1"
        }))
        .unwrap();
        assert_eq!(session_defaults.title, None);
        assert_eq!(session_defaults.cwd, None);
        assert_eq!(session_defaults.model_id, None);

        let attach: RemoteSessionAttachDto = serde_json::from_value(json!({
            "session_id": "session-1",
            "node_id": "node-1",
            "attached": true,
            "config_options": [],
            "snapshot": {
                "audit": [{ "kind": "message", "data": { "nested": [1, 2] } }],
                "cursor": { "position": 1 },
                "delegationUpdates": [{ "id": "delegate-1" }]
            }
        }))
        .unwrap();
        assert!(attach.config_options.is_empty());
        assert_eq!(
            attach.snapshot.as_ref().unwrap()["audit"][0]["data"]["nested"],
            json!([1, 2])
        );
        assert_eq!(
            attach.snapshot.as_ref().unwrap()["delegationUpdates"][0]["id"],
            "delegate-1"
        );

        let detached: RemoteSessionAttachDto = serde_json::from_value(json!({
            "session_id": "session-2",
            "node_id": "node-1",
            "attached": false,
            "config_options": []
        }))
        .unwrap();
        assert_eq!(detached.snapshot, None);

        let null_snapshot: RemoteSessionAttachDto = serde_json::from_value(json!({
            "session_id": "session-3",
            "node_id": "node-1",
            "snapshot": null
        }))
        .unwrap();
        assert_eq!(null_snapshot.snapshot, None);

        let invite: MeshInviteCreatedDto = serde_json::from_value(json!({
            "invite_id": "invite-1",
            "url": "qmt://mesh/join/token",
            "qr_code": "QR",
            "expires_at": 123,
            "max_uses": 2,
            "mesh_name": "Team"
        }))
        .unwrap();
        assert_eq!(invite.mesh_name.as_deref(), Some("Team"));
    }

    #[test]
    fn mesh_dto_defaults_preserve_existing_wire_behavior() {
        let status: MeshStatusDto = serde_json::from_value(json!({ "enabled": false })).unwrap();
        assert!(status.scopes.is_empty());
        assert_eq!(status.known_peer_count, 0);

        let nodes: MeshNodesDto = serde_json::from_value(json!({})).unwrap();
        assert!(nodes.nodes.is_empty());

        let sessions: RemoteSessionListDto =
            serde_json::from_value(json!({ "node_id": "node-1" })).unwrap();
        assert!(sessions.sessions.is_empty());
        assert_eq!(sessions.next_offset, None);
        assert_eq!(sessions.total_count, 0);

        let attach: RemoteSessionAttachDto =
            serde_json::from_value(json!({ "session_id": "session-1", "node_id": "node-1" }))
                .unwrap();
        assert!(!attach.attached);
        assert!(attach.config_options.is_empty());
        assert_eq!(attach.snapshot, None);

        let invite: MeshInviteCreatedDto = serde_json::from_value(json!({
            "invite_id": "invite-1",
            "url": "qmt://mesh/join/token"
        }))
        .unwrap();
        assert_eq!(invite.expires_at, 0);
        assert_eq!(invite.max_uses, 0);
        assert_eq!(invite.qr_code, None);
    }
}
