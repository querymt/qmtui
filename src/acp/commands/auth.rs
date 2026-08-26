use agent_client_protocol as acp_sdk;

use crate::acp_state::AcpAppEvent;

use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::extensions::auth;

pub(super) async fn list<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
) -> Result<(), acp_sdk::Error> {
    if let Some(response) = auth::providers(ctx.connection).await? {
        ctx.events
            .send(AcpAppEvent::AuthProviders(response.providers));
    }
    Ok(())
}

pub(super) async fn start<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    provider: String,
) -> Result<(), acp_sdk::Error> {
    if let Some(flow) = auth::start(ctx.connection, provider).await? {
        ctx.events.send(AcpAppEvent::OAuthFlowStarted(flow));
    }
    Ok(())
}

pub(super) async fn complete<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    flow_id: String,
    response: String,
) -> Result<(), acp_sdk::Error> {
    if let Some(result) = auth::complete(ctx.connection, flow_id, response).await? {
        ctx.events.send(AcpAppEvent::OAuthResult(result));
    }
    Ok(())
}

pub(super) async fn logout<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    provider: String,
) -> Result<(), acp_sdk::Error> {
    if let Some(result) = auth::logout(ctx.connection, provider).await? {
        ctx.events.send(AcpAppEvent::OAuthResult(result));
    }
    Ok(())
}
