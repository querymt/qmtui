use agent_client_protocol as acp_sdk;

use crate::acp_state::AcpAppEvent;

use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::extensions::mesh;

pub(super) async fn list_nodes<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
) -> Result<(), acp_sdk::Error> {
    if let Some(status) = mesh::status(ctx.connection).await? {
        ctx.events.send(AcpAppEvent::MeshStatus(status));
    }
    if let Some(nodes) = mesh::nodes(ctx.connection).await? {
        ctx.events.send(AcpAppEvent::MeshNodes(nodes));
    }
    Ok(())
}

pub(super) async fn list_sessions<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    node_id: String,
    offset: u32,
    limit: u32,
) -> Result<(), acp_sdk::Error> {
    if let Some(list) = mesh::remote_sessions(ctx.connection, node_id, offset, limit).await? {
        ctx.events.send(AcpAppEvent::RemoteSessions(list));
    }
    Ok(())
}

pub(super) async fn create_session<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    node_id: String,
    cwd: Option<String>,
) -> Result<(), acp_sdk::Error> {
    if let Some(attached) = mesh::create_remote_session(ctx.connection, node_id, cwd).await? {
        ctx.events
            .send(AcpAppEvent::RemoteSessionAttached(attached));
    }
    Ok(())
}

pub(super) async fn attach_session<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    node_id: String,
    session_id: String,
) -> Result<(), acp_sdk::Error> {
    if let Some(attached) = mesh::attach_remote_session(ctx.connection, node_id, session_id).await?
    {
        ctx.events
            .send(AcpAppEvent::RemoteSessionAttached(attached));
    }
    Ok(())
}

pub(super) async fn create_invite<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    mesh_name: Option<String>,
    ttl: Option<String>,
    max_uses: Option<u32>,
) -> Result<(), acp_sdk::Error> {
    if let Some(invite) = mesh::create_invite(ctx.connection, mesh_name, ttl, max_uses).await? {
        ctx.events.send(AcpAppEvent::MeshInviteCreated(invite));
    }
    Ok(())
}
