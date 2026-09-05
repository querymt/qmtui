use agent_client_protocol::{self as acp_sdk, schema::ProtocolVersion, schema::v1 as acp};
use serde_json::Value;

use crate::acp_state::AcpAppEvent;

use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::extensions::{capabilities, mesh, profiles};
use super::super::runtime::AgentIdentity;

pub(super) async fn run<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
) -> Result<(), acp_sdk::Error> {
    let response = ctx
        .connection
        .request(
            acp::InitializeRequest::new(ProtocolVersion::V1)
                .client_capabilities(client_capabilities())
                .client_info(acp::Implementation::new("qmtui", env!("CARGO_PKG_VERSION"))),
        )
        .await?;
    let identity = response
        .agent_info
        .map(|info| AgentIdentity {
            id: info.name.clone(),
            name: info.title.unwrap_or(info.name),
        })
        .unwrap_or_default();
    ctx.state.set_agent_identity(identity.clone()).await;
    ctx.events.send(AcpAppEvent::Initialized {
        agent_id: identity.id,
        agent_name: identity.name,
        profiles: Vec::new(),
        active_profile_id: None,
        agent_mode: Some("build".to_string()),
        reasoning_effort: Some(None),
    });
    post_connect(ctx).await;
    Ok(())
}

async fn post_connect<C: AcpConnection>(ctx: CommandContext<'_, C>) {
    match capabilities::get(ctx.connection).await {
        Ok(response) => {
            let payload = super::super::extensions::payload(&response).clone();
            ctx.events
                .send(AcpAppEvent::ControlCapabilities(payload.clone()));
            let methods = methods(&payload);
            if methods.iter().any(|method| method == "querymt/mesh/nodes")
                && let Ok(Some(nodes)) = mesh::nodes(ctx.connection).await
            {
                ctx.events.send(AcpAppEvent::MeshNodes(nodes));
            }
            if methods.iter().any(|method| method == "querymt/profiles") {
                match profiles::list(ctx.connection).await {
                    Ok(response) => ctx.events.send(AcpAppEvent::Profiles {
                        profiles: response.profiles,
                        active_profile_id: response.active_profile_id,
                    }),
                    Err(err) => ctx.events.error(format!("failed to load profiles: {err}")),
                }
            } else if payload
                .get("features")
                .and_then(|features| features.get("profiles"))
                .and_then(Value::as_bool)
                == Some(false)
            {
                ctx.events.send(AcpAppEvent::Profiles {
                    profiles: Vec::new(),
                    active_profile_id: None,
                });
            }
        }
        Err(err) => ctx
            .events
            .send(AcpAppEvent::ControlCapabilitiesUnavailable(err.to_string())),
    }
}

fn methods(payload: &Value) -> Vec<String> {
    payload
        .get("methods")
        .and_then(Value::as_array)
        .map(|methods| {
            methods
                .iter()
                .filter_map(|method| method.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn client_capabilities() -> acp::ClientCapabilities {
    acp::ClientCapabilities::new()
        .fs(acp::FileSystemCapabilities::new())
        .terminal(false)
        .elicitation(
            acp::ElicitationCapabilities::new().form(acp::ElicitationFormCapabilities::new()),
        )
}
