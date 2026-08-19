use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelegationUpdateStateDto {
    Requested,
    Forked,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DelegationUpdateDto {
    pub version: u32,
    pub session_id: String,
    pub delegation_id: String,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    pub state: DelegationUpdateStateDto,
    pub target_agent_id: String,
    pub objective: String,
    #[serde(default)]
    pub child_session_id: Option<String>,
    pub requested_at: i64,
    #[serde(default)]
    pub forked_at: Option<i64>,
    #[serde(default)]
    pub finished_at: Option<i64>,
    pub updated_at: i64,
    #[serde(default)]
    pub result_summary: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{DelegationUpdateDto, DelegationUpdateStateDto};
    use serde_json::json;

    #[test]
    fn delegation_update_dto_deserializes_camel_case_fields_and_ignores_unknown_fields() {
        let update: DelegationUpdateDto = serde_json::from_value(json!({
            "version": 1,
            "sessionId": "parent",
            "delegationId": "delegation-1",
            "toolCallId": "call-1",
            "state": "forked",
            "targetAgentId": "coder",
            "objective": "Implement it",
            "childSessionId": "child-1",
            "requestedAt": 100,
            "forkedAt": 110,
            "finishedAt": null,
            "updatedAt": 110,
            "resultSummary": null,
            "error": null,
            "unknownField": "ignored"
        }))
        .expect("delegation update");

        assert_eq!(update.version, 1);
        assert_eq!(update.session_id, "parent");
        assert_eq!(update.delegation_id, "delegation-1");
        assert_eq!(update.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(update.state, DelegationUpdateStateDto::Forked);
        assert_eq!(update.target_agent_id, "coder");
        assert_eq!(update.objective, "Implement it");
        assert_eq!(update.child_session_id.as_deref(), Some("child-1"));
        assert_eq!(update.requested_at, 100);
        assert_eq!(update.forked_at, Some(110));
        assert_eq!(update.finished_at, None);
        assert_eq!(update.updated_at, 110);
        assert_eq!(update.result_summary, None);
        assert_eq!(update.error, None);
    }

    #[test]
    fn delegation_update_dto_defaults_optional_fields() {
        let update: DelegationUpdateDto = serde_json::from_value(json!({
            "version": 1,
            "sessionId": "parent",
            "delegationId": "delegation-1",
            "state": "requested",
            "targetAgentId": "coder",
            "objective": "Implement it",
            "requestedAt": 100,
            "updatedAt": 100
        }))
        .expect("delegation update defaults");

        assert_eq!(update.tool_call_id, None);
        assert_eq!(update.child_session_id, None);
        assert_eq!(update.forked_at, None);
        assert_eq!(update.finished_at, None);
        assert_eq!(update.result_summary, None);
        assert_eq!(update.error, None);
    }

    #[test]
    fn delegation_update_state_dto_accepts_every_snake_case_lifecycle_state() {
        for (wire_state, expected) in [
            ("requested", DelegationUpdateStateDto::Requested),
            ("forked", DelegationUpdateStateDto::Forked),
            ("completed", DelegationUpdateStateDto::Completed),
            ("failed", DelegationUpdateStateDto::Failed),
            ("cancelled", DelegationUpdateStateDto::Cancelled),
        ] {
            let state: DelegationUpdateStateDto =
                serde_json::from_value(json!(wire_state)).expect("delegation state");
            assert_eq!(state, expected);
        }
    }
}
