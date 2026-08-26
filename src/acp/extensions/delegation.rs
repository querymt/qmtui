use agent_client_protocol as acp_sdk;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::domain::activity::{DelegationState, DelegationUpdate};
use crate::domain::model::DelegateModelOverrideInfo;
use crate::protocol::delegation::{DelegationUpdateDto, DelegationUpdateStateDto};

use super::{call, payload};
use crate::acp::connection::AcpConnection;

#[derive(Debug, Clone, Deserialize)]
pub(in crate::acp) struct DelegateModelResponse {
    pub(in crate::acp) session_id: String,
    pub(in crate::acp) agent_id: String,
    #[serde(default)]
    pub(in crate::acp) model: Option<DelegateModelOverrideInfo>,
}

pub(in crate::acp) async fn set_model<C: AcpConnection>(
    connection: &C,
    session_id: String,
    agent_id: String,
    model_id: Option<String>,
    node_id: Option<String>,
) -> Result<DelegateModelResponse, acp_sdk::Error> {
    let response = call(
        connection,
        "querymt/session/setDelegateModel",
        json!({
            "session_id": session_id,
            "agent_id": agent_id,
            "model_id": model_id,
            "node_id": node_id,
        }),
    )
    .await?;
    serde_json::from_value(payload(&response).clone()).map_err(acp_sdk::Error::into_internal_error)
}

pub(in crate::acp) fn from_value(
    value: Value,
) -> Result<Option<DelegationUpdate>, serde_json::Error> {
    serde_json::from_value(value).map(from_wire)
}

pub(in crate::acp) fn from_wire(update: DelegationUpdateDto) -> Option<DelegationUpdate> {
    if update.version != 1 {
        return None;
    }
    Some(DelegationUpdate {
        session_id: update.session_id,
        delegation_id: update.delegation_id,
        tool_call_id: update.tool_call_id,
        state: match update.state {
            DelegationUpdateStateDto::Requested => DelegationState::Requested,
            DelegationUpdateStateDto::Forked => DelegationState::Forked,
            DelegationUpdateStateDto::Completed => DelegationState::Completed,
            DelegationUpdateStateDto::Failed => DelegationState::Failed,
            DelegationUpdateStateDto::Cancelled => DelegationState::Cancelled,
        },
        target_agent_id: update.target_agent_id,
        objective: update.objective,
        child_session_id: update.child_session_id,
        requested_at: update.requested_at,
        forked_at: update.forked_at,
        finished_at: update.finished_at,
        updated_at: update.updated_at,
        result_summary: update.result_summary,
        error: update.error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_conversion_accepts_supported_versions_only() {
        let value = json!({
            "version": 1,
            "sessionId": "parent",
            "delegationId": "d1",
            "state": "completed",
            "targetAgentId": "coder",
            "objective": "implement",
            "requestedAt": 1,
            "updatedAt": 2
        });
        assert!(from_value(value.clone()).expect("wire").is_some());
        let mut unsupported = value;
        unsupported["version"] = json!(2);
        assert!(from_value(unsupported).expect("wire").is_none());
    }

    #[test]
    fn delegate_model_decode_is_strict_and_accepts_data_wrapper() {
        let response: DelegateModelResponse = serde_json::from_value(
            payload(&json!({ "data": {
                "session_id": "s",
                "agent_id": "a",
                "model": { "model_id": "p/m", "node_id": null }
            }}))
            .clone(),
        )
        .expect("response");
        assert_eq!(response.model.expect("model").model_id, "p/m");
        assert!(serde_json::from_value::<DelegateModelResponse>(json!({})).is_err());
    }
}
