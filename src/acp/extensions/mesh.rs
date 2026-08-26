use agent_client_protocol as acp_sdk;
use serde_json::{Value, json};

use crate::domain::mesh::{
    MeshInviteCreatedInfo, MeshNodesInfo, MeshScopeInfo, MeshStatusInfo, RemoteNodeInfo,
    RemoteSessionAttachInfo, RemoteSessionInfo, RemoteSessionListInfo,
};
use crate::protocol::mesh::{
    MeshInviteCreatedDto, MeshNodesDto, MeshScopeDto, MeshStatusDto, RemoteNodeDto,
    RemoteSessionAttachDto, RemoteSessionDto, RemoteSessionListDto,
};

use super::{call, payload};
use crate::acp::connection::AcpConnection;

pub(in crate::acp) async fn status<C: AcpConnection>(
    connection: &C,
) -> Result<Option<MeshStatusInfo>, acp_sdk::Error> {
    tolerant_call(
        connection,
        "querymt/mesh/status",
        json!({}),
        status_from_wire,
    )
    .await
}

pub(in crate::acp) async fn nodes<C: AcpConnection>(
    connection: &C,
) -> Result<Option<MeshNodesInfo>, acp_sdk::Error> {
    tolerant_call(connection, "querymt/mesh/nodes", json!({}), nodes_from_wire).await
}

pub(in crate::acp) async fn remote_sessions<C: AcpConnection>(
    connection: &C,
    node_id: String,
    offset: u32,
    limit: u32,
) -> Result<Option<RemoteSessionListInfo>, acp_sdk::Error> {
    tolerant_call(
        connection,
        "querymt/remote/sessions",
        json!({ "node_id": node_id, "offset": offset, "limit": limit }),
        remote_list_from_wire,
    )
    .await
}

pub(in crate::acp) async fn create_remote_session<C: AcpConnection>(
    connection: &C,
    node_id: String,
    cwd: Option<String>,
) -> Result<Option<RemoteSessionAttachInfo>, acp_sdk::Error> {
    tolerant_call(
        connection,
        "querymt/remote/createSession",
        json!({ "node_id": node_id, "cwd": cwd, "attach": true }),
        remote_attach_from_wire,
    )
    .await
}

pub(in crate::acp) async fn attach_remote_session<C: AcpConnection>(
    connection: &C,
    node_id: String,
    session_id: String,
) -> Result<Option<RemoteSessionAttachInfo>, acp_sdk::Error> {
    tolerant_call(
        connection,
        "querymt/remote/attachSession",
        json!({ "node_id": node_id, "session_id": session_id }),
        remote_attach_from_wire,
    )
    .await
}

pub(in crate::acp) async fn create_invite<C: AcpConnection>(
    connection: &C,
    mesh_name: Option<String>,
    ttl: Option<String>,
    max_uses: Option<u32>,
) -> Result<Option<MeshInviteCreatedInfo>, acp_sdk::Error> {
    tolerant_call(
        connection,
        "querymt/mesh/createInvite",
        json!({ "mesh_name": mesh_name, "ttl": ttl, "max_uses": max_uses }),
        invite_from_wire,
    )
    .await
}

async fn tolerant_call<C, D, T>(
    connection: &C,
    method: &str,
    params: Value,
    convert: impl FnOnce(D) -> T,
) -> Result<Option<T>, acp_sdk::Error>
where
    C: AcpConnection,
    D: serde::de::DeserializeOwned,
{
    let response = call(connection, method, params).await?;
    Ok(serde_json::from_value::<D>(payload(&response).clone())
        .ok()
        .map(convert))
}

fn status_from_wire(status: MeshStatusDto) -> MeshStatusInfo {
    MeshStatusInfo {
        enabled: status.enabled,
        peer_id: status.peer_id,
        transport: status.transport,
        known_peer_count: status.known_peer_count,
        has_invite_store: status.has_invite_store,
        has_mesh_state_store: status.has_mesh_state_store,
        scopes: status.scopes.into_iter().map(scope_from_wire).collect(),
    }
}

fn scope_from_wire(scope: MeshScopeDto) -> MeshScopeInfo {
    MeshScopeInfo {
        kind: scope.kind,
        id: scope.id,
    }
}

fn node_from_wire(node: RemoteNodeDto) -> RemoteNodeInfo {
    RemoteNodeInfo {
        id: node.id,
        label: node.label,
        capabilities: node.capabilities,
        active_sessions: node.active_sessions,
        transport: node.transport,
        last_seen_at: node.last_seen_at,
    }
}

fn nodes_from_wire(nodes: MeshNodesDto) -> MeshNodesInfo {
    MeshNodesInfo {
        nodes: nodes.nodes.into_iter().map(node_from_wire).collect(),
    }
}

fn remote_from_wire(session: RemoteSessionDto) -> RemoteSessionInfo {
    RemoteSessionInfo {
        id: session.id,
        node_id: session.node_id,
        node_label: session.node_label,
        title: session.title,
        cwd: session.cwd,
        updated_at: session.updated_at,
        profile_id: session.profile_id,
        model_id: session.model_id,
    }
}

fn remote_list_from_wire(list: RemoteSessionListDto) -> RemoteSessionListInfo {
    RemoteSessionListInfo {
        node_id: list.node_id,
        sessions: list.sessions.into_iter().map(remote_from_wire).collect(),
        next_offset: list.next_offset,
        total_count: list.total_count,
    }
}

fn remote_attach_from_wire(attached: RemoteSessionAttachDto) -> RemoteSessionAttachInfo {
    RemoteSessionAttachInfo {
        session_id: attached.session_id,
        node_id: attached.node_id,
        attached: attached.attached,
        config_options: attached.config_options,
        snapshot: attached.snapshot,
    }
}

fn invite_from_wire(invite: MeshInviteCreatedDto) -> MeshInviteCreatedInfo {
    MeshInviteCreatedInfo {
        invite_id: invite.invite_id,
        url: invite.url,
        qr_code: invite.qr_code,
        expires_at: invite.expires_at,
        max_uses: invite.max_uses,
        mesh_name: invite.mesh_name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_preserve_mesh_and_remote_fields() {
        let status = status_from_wire(MeshStatusDto {
            enabled: true,
            peer_id: Some("peer".into()),
            transport: Some("relay".into()),
            known_peer_count: 2,
            has_invite_store: true,
            has_mesh_state_store: true,
            scopes: vec![MeshScopeDto {
                kind: "team".into(),
                id: "scope".into(),
            }],
        });
        assert!(status.enabled);
        assert_eq!(status.scopes[0].id, "scope");

        let list = remote_list_from_wire(RemoteSessionListDto {
            node_id: "node".into(),
            sessions: vec![RemoteSessionDto {
                id: "session".into(),
                node_id: "node".into(),
                node_label: Some("remote".into()),
                title: Some("title".into()),
                cwd: Some("/repo".into()),
                updated_at: Some("now".into()),
                profile_id: Some("profile".into()),
                model_id: Some("model".into()),
            }],
            next_offset: Some(10),
            total_count: 11,
        });
        assert_eq!(list.sessions[0].cwd.as_deref(), Some("/repo"));
        assert_eq!(list.next_offset, Some(10));
    }

    #[test]
    fn empty_nodes_default_and_malformed_attach_is_tolerated() {
        let nodes: MeshNodesDto = serde_json::from_value(json!({})).expect("default nodes");
        assert!(nodes.nodes.is_empty());
        assert!(serde_json::from_value::<RemoteSessionAttachDto>(json!({})).is_err());
    }
}
