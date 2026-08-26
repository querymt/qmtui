use agent_client_protocol as acp_sdk;
use serde::Deserialize;
use serde_json::json;

use crate::domain::profile::{AgentInfo, ProfileInfo};

use super::{call, payload};
use crate::acp::connection::AcpConnection;

#[derive(Debug, Clone, Default, Deserialize)]
pub(super) struct ProfilesResponse {
    pub(super) profiles: Vec<ProfileInfo>,
    #[serde(default)]
    pub(super) active_profile_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct ProfileAgentsResponse {
    pub(super) profile_id: String,
    pub(super) agents: Vec<AgentInfo>,
}

pub(super) async fn list<C: AcpConnection>(
    connection: &C,
) -> Result<ProfilesResponse, acp_sdk::Error> {
    let response = call(connection, "querymt/profiles", json!({})).await?;
    serde_json::from_value(payload(&response).clone()).map_err(acp_sdk::Error::into_internal_error)
}

pub(super) async fn agents<C: AcpConnection>(
    connection: &C,
    profile_id: &str,
) -> Result<ProfileAgentsResponse, acp_sdk::Error> {
    let response = call(
        connection,
        "querymt/profile/agents",
        json!({ "profile_id": profile_id }),
    )
    .await?;
    serde_json::from_value(payload(&response).clone()).map_err(acp_sdk::Error::into_internal_error)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn profile_decoding_is_strict_for_direct_and_wrapped_payloads() {
        let direct: ProfilesResponse = serde_json::from_value(json!({
            "profiles": [{ "id": "fast", "name": "Fast" }],
            "active_profile_id": "fast"
        }))
        .expect("direct");
        assert_eq!(direct.profiles[0].id, "fast");
        let wrapped: ProfileAgentsResponse = serde_json::from_value(
            payload(&json!({ "data": {
                "profile_id": "fast",
                "agents": [{ "id": "primary", "name": "Session" }]
            }}))
            .clone(),
        )
        .expect("wrapped");
        assert_eq!(wrapped.agents[0].id, "primary");
        assert!(serde_json::from_value::<ProfilesResponse>(json!({})).is_err());
        assert!(
            serde_json::from_value::<ProfileAgentsResponse>(json!({ "profile_id": "x" })).is_err()
        );
    }
}
