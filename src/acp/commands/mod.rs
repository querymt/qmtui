use agent_client_protocol as acp_sdk;

use crate::command::Command;

use super::connection::AcpConnection;
use super::context::CommandContext;

mod auth;
mod catalog;
mod elicitation;
mod history;
mod initialize;
mod mesh;
mod session;
mod unsupported;

pub(super) async fn dispatch<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    command: Command,
) -> Result<(), acp_sdk::Error> {
    match command {
        Command::Init => initialize::run(ctx).await?,
        Command::ListSessions { request, cursor } => session::list(ctx, request, cursor).await,
        Command::NewSession { cwd, profile_id } => session::new(ctx, cwd, profile_id).await?,
        Command::LoadSession { session_id, cwd } => session::load(ctx, session_id, cwd).await?,
        Command::Prompt { prompt, local_id } => session::prompt(ctx, prompt, local_id).await?,
        Command::CancelSession => session::cancel(ctx).await?,
        Command::DeleteSession { session_id } => session::delete(ctx, session_id).await?,
        Command::SetAgentMode { mode } => catalog::set_config(ctx, "mode", &mode).await?,
        Command::SetReasoningEffort { reasoning_effort } => {
            catalog::set_config(ctx, "reasoning_effort", &reasoning_effort).await?
        }
        Command::SetSessionModel {
            session_id,
            model_id,
            node_id,
        } => catalog::set_session_model(ctx, session_id, model_id, node_id).await?,
        Command::ListAllModels { refresh } => catalog::list_models(ctx, refresh).await?,
        Command::ListProfiles => catalog::list_profiles(ctx).await,
        Command::ListProfileAgents { profile_id } => {
            catalog::list_profile_agents(ctx, profile_id).await
        }
        Command::SetDelegateModel {
            session_id,
            agent_id,
            model_id,
            node_id,
        } => catalog::set_delegate_model(ctx, session_id, agent_id, model_id, node_id).await?,
        Command::ListAuthProviders => auth::list(ctx).await?,
        Command::StartOAuthLogin { provider } => auth::start(ctx, provider).await?,
        Command::CompleteOAuthLogin { flow_id, response } => {
            auth::complete(ctx, flow_id, response).await?
        }
        Command::DisconnectOAuth { provider } => auth::logout(ctx, provider).await?,
        Command::ElicitationResponse {
            elicitation_id,
            action,
            content,
        } => elicitation::respond(ctx, elicitation_id, action, content).await,
        Command::ForkSession { message_id } => history::fork(ctx, message_id).await?,
        Command::Undo { message_id } => history::undo(ctx, message_id).await?,
        Command::Redo => history::redo(ctx).await?,
        Command::ListRemoteNodes => mesh::list_nodes(ctx).await?,
        Command::ListRemoteSessions {
            node_id,
            offset,
            limit,
        } => mesh::list_sessions(ctx, node_id, offset, limit).await?,
        Command::CreateRemoteSession { node_id, cwd } => {
            mesh::create_session(ctx, node_id, cwd).await?
        }
        Command::AttachRemoteSession {
            node_id,
            session_id,
        } => mesh::attach_session(ctx, node_id, session_id).await?,
        Command::CreateMeshInvite {
            mesh_name,
            ttl,
            max_uses,
        } => mesh::create_invite(ctx, mesh_name, ttl, max_uses).await?,
        Command::SubscribeSession { .. } => {}
        Command::GetFileIndex => unsupported::file_index(ctx),
        command @ (Command::ListSessionChildren { .. }
        | Command::DismissRemoteSession { .. }
        | Command::SetApiToken { .. }
        | Command::ClearApiToken { .. }) => unsupported::command(ctx, &command),
    }
    Ok(())
}
