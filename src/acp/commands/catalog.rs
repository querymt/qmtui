use agent_client_protocol::{self as acp_sdk, schema::v1 as acp};

use crate::acp_state::{AcpAppEvent, AcpModelsMetaInfo};

use super::super::configuration;
use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::extensions::{delegation, models, profiles};

pub(super) async fn set_config<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    config_id: &str,
    value: &str,
) -> Result<(), acp_sdk::Error> {
    let Some(session_id) = ctx.state.current_session_id().await else {
        ctx.events
            .error(format!("cannot set {config_id} before a session is loaded"));
        return Ok(());
    };
    let response = ctx
        .connection
        .request(acp::SetSessionConfigOptionRequest::new(
            session_id,
            config_id.to_string(),
            acp::SessionConfigOptionValue::from(value),
        ))
        .await?;
    configuration::apply(ctx.state, ctx.events, response.config_options).await;
    Ok(())
}

pub(super) async fn set_session_model<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    session_id: String,
    model_id: String,
    node_id: Option<String>,
) -> Result<(), acp_sdk::Error> {
    let model = ctx
        .state
        .model_by_id(&model_id)
        .await
        .unwrap_or_else(|| configuration::fallback_model(&model_id));
    let effective_node = node_id.as_deref().or(model.node_id.as_deref());
    let node_part = effective_node
        .map(|node| format!(" node={node}"))
        .unwrap_or_default();
    ctx.events.info(
        "acp",
        format!(
            "ACP SetSessionModel: provider={} model={} id={}{node_part}",
            model.provider, model.model, model_id
        ),
    );
    let response = ctx
        .connection
        .request(
            acp::SetSessionConfigOptionRequest::new(session_id, "model", model_id.as_str())
                .meta(configuration::model_meta(&model, node_id.as_deref())),
        )
        .await?;
    ctx.state.select_model(model_id).await;
    configuration::send_provider_changed(ctx.events, &model);
    configuration::apply(ctx.state, ctx.events, response.config_options).await;
    Ok(())
}

pub(super) async fn list_models<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    refresh: bool,
) -> Result<(), acp_sdk::Error> {
    let response = load_models(ctx.connection, refresh).await?;
    ctx.state.set_models(response.models.clone()).await;
    send_models(ctx, &response);
    if let Some(model) = ctx.state.selected_or_default_model().await {
        ctx.state.select_model(model.id.clone()).await;
        configuration::send_provider_changed(ctx.events, &model);
    }
    Ok(())
}

async fn load_models<C: AcpConnection>(
    connection: &C,
    refresh: bool,
) -> Result<models::ModelsResponse, acp_sdk::Error> {
    let mut response = models::list(connection, refresh).await?;
    if !refresh && response.should_retry_empty() {
        for attempt in 0..3 {
            tokio::time::sleep(std::time::Duration::from_millis(250 * (attempt + 1))).await;
            response = models::list(connection, false).await?;
            if !response.should_retry_empty() {
                return Ok(response);
            }
        }
        response = models::list(connection, true).await?;
    }
    if response.should_retry_empty() {
        for attempt in 0..3 {
            tokio::time::sleep(std::time::Duration::from_millis(300 * (attempt + 1))).await;
            response = models::list(connection, false).await?;
            if !response.should_retry_empty() {
                break;
            }
        }
    }
    Ok(response)
}

fn send_models<C: AcpConnection>(ctx: CommandContext<'_, C>, response: &models::ModelsResponse) {
    ctx.events.send(AcpAppEvent::Models {
        models: response
            .models
            .iter()
            .map(models::Model::to_app_model)
            .collect(),
        meta: response.meta.as_ref().map(|meta| AcpModelsMetaInfo {
            remote_node_count: meta.remote_node_count,
            remote_timeout_count: meta.remote_timeout_count,
        }),
    });
}

pub(super) async fn list_profiles<C: AcpConnection>(ctx: CommandContext<'_, C>) {
    match profiles::list(ctx.connection).await {
        Ok(response) => ctx.events.send(AcpAppEvent::Profiles {
            profiles: response.profiles,
            active_profile_id: response.active_profile_id,
        }),
        Err(err) => ctx
            .events
            .info("profiles", format!("profile catalog unavailable: {err}")),
    }
}

pub(super) async fn list_profile_agents<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    profile_id: String,
) {
    match profiles::agents(ctx.connection, &profile_id).await {
        Ok(response) => ctx.events.send(AcpAppEvent::ProfileAgents {
            profile_id: response.profile_id,
            agents: response.agents,
        }),
        Err(err) => ctx.events.info(
            "profiles",
            format!("profile agents unavailable for {profile_id}: {err}"),
        ),
    }
}

pub(super) async fn set_delegate_model<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    session_id: String,
    agent_id: String,
    model_id: Option<String>,
    node_id: Option<String>,
) -> Result<(), acp_sdk::Error> {
    let response =
        delegation::set_model(ctx.connection, session_id, agent_id, model_id, node_id).await?;
    ctx.events.send(AcpAppEvent::DelegateModelSet {
        session_id: response.session_id,
        agent_id: response.agent_id,
        model: response.model,
    });
    Ok(())
}
