use agent_client_protocol as acp_sdk;
use serde_json::json;

use crate::domain::auth::{OAuthFlow, OAuthResult, OAuthResultStatus};
use crate::protocol::auth::{AuthProvidersDto, OAuthFlowDto, OAuthResultDto};

use super::{call, payload};
use crate::acp::connection::AcpConnection;

pub(super) async fn providers<C: AcpConnection>(
    connection: &C,
) -> Result<Option<AuthProvidersDto>, acp_sdk::Error> {
    let response = call(connection, "querymt/auth/status", json!({})).await?;
    Ok(serde_json::from_value(payload(&response).clone()).ok())
}

pub(super) async fn start<C: AcpConnection>(
    connection: &C,
    provider: String,
) -> Result<Option<OAuthFlow>, acp_sdk::Error> {
    let response = call(
        connection,
        "querymt/auth/start",
        json!({ "provider": provider }),
    )
    .await?;
    Ok(
        serde_json::from_value::<OAuthFlowDto>(payload(&response).clone())
            .ok()
            .map(flow_from_wire),
    )
}

pub(super) async fn complete<C: AcpConnection>(
    connection: &C,
    flow_id: String,
    response: String,
) -> Result<Option<OAuthResult>, acp_sdk::Error> {
    result_call(
        connection,
        "querymt/auth/complete",
        json!({ "flow_id": flow_id, "response": response }),
    )
    .await
}

pub(super) async fn logout<C: AcpConnection>(
    connection: &C,
    provider: String,
) -> Result<Option<OAuthResult>, acp_sdk::Error> {
    result_call(
        connection,
        "querymt/auth/logout",
        json!({ "provider": provider }),
    )
    .await
}

async fn result_call<C: AcpConnection>(
    connection: &C,
    method: &str,
    params: serde_json::Value,
) -> Result<Option<OAuthResult>, acp_sdk::Error> {
    let response = call(connection, method, params).await?;
    Ok(
        serde_json::from_value::<OAuthResultDto>(payload(&response).clone())
            .ok()
            .map(result_from_wire),
    )
}

fn flow_from_wire(flow: OAuthFlowDto) -> OAuthFlow {
    OAuthFlow {
        flow_id: flow.flow_id,
        provider: flow.provider,
        authorization_url: flow.authorization_url,
        flow_kind: flow.flow_kind,
    }
}

fn result_from_wire(result: OAuthResultDto) -> OAuthResult {
    OAuthResult {
        provider: result.provider,
        status: if result.success {
            OAuthResultStatus::Success
        } else {
            OAuthResultStatus::Failure
        },
        message: result.message,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn auth_decoding_is_tolerant() {
        let valid: Option<AuthProvidersDto> =
            serde_json::from_value(payload(&json!({ "data": { "providers": [] } })).clone()).ok();
        assert!(valid.is_some());
        let malformed: Option<AuthProvidersDto> = serde_json::from_value(json!({})).ok();
        assert!(malformed.is_none());
    }
}
