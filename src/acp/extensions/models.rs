use agent_client_protocol as acp_sdk;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::domain::model::ModelEntry;

use super::{call, payload};
use crate::acp::connection::AcpConnection;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(in crate::acp) struct Model {
    pub(in crate::acp) id: String,
    pub(in crate::acp) label: String,
    #[serde(default)]
    pub(in crate::acp) source: Option<String>,
    pub(in crate::acp) provider: String,
    pub(in crate::acp) model: String,
    #[serde(default)]
    pub(in crate::acp) node_id: Option<String>,
    #[serde(default)]
    pub(in crate::acp) node_label: Option<String>,
    #[serde(default)]
    pub(in crate::acp) family: Option<String>,
    #[serde(default)]
    pub(in crate::acp) quant: Option<String>,
}

impl Model {
    pub(super) fn to_app_model(&self) -> ModelEntry {
        ModelEntry {
            id: self.id.clone(),
            label: self.label.clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            node_id: self.node_id.clone(),
            node_label: self.node_label.clone(),
            family: self.family.clone(),
            quant: self.quant.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(in crate::acp) struct ModelsMeta {
    #[serde(default)]
    pub(super) stale: bool,
    #[serde(default)]
    pub(super) refresh_in_progress: bool,
    #[serde(default)]
    pub(super) remote_node_count: u32,
    #[serde(default)]
    pub(super) remote_timeout_count: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in crate::acp) struct ModelsResponse {
    pub(in crate::acp) models: Vec<Model>,
    pub(in crate::acp) meta: Option<ModelsMeta>,
}

impl ModelsResponse {
    pub(super) fn should_retry_empty(&self) -> bool {
        self.models.is_empty()
            || self
                .meta
                .as_ref()
                .is_some_and(|meta| meta.stale || meta.refresh_in_progress)
    }
}

pub(super) async fn list<C: AcpConnection>(
    connection: &C,
    refresh: bool,
) -> Result<ModelsResponse, acp_sdk::Error> {
    let method = if refresh {
        "querymt/refreshModels"
    } else {
        "querymt/models"
    };
    let params = if refresh {
        json!({ "wait_for_completion": true })
    } else {
        json!({})
    };
    let response = call(connection, method, params).await?;
    Ok(normalize(response))
}

pub(super) fn normalize(response: Value) -> ModelsResponse {
    let data = payload(&response);
    let models = data
        .get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|model| serde_json::from_value::<Model>(model.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    let meta = data
        .get("meta")
        .or_else(|| response.get("meta"))
        .and_then(|meta| serde_json::from_value(meta.clone()).ok());
    ModelsResponse { models, meta }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_is_tolerant_and_accepts_direct_or_wrapped_data() {
        let direct = normalize(json!({
            "models": [{
                "id": "openai/gpt-5",
                "label": "gpt-5",
                "provider": "openai",
                "model": "gpt-5"
            }, { "malformed": true }],
            "meta": { "stale": true }
        }));
        assert_eq!(direct.models.len(), 1);
        assert!(direct.meta.expect("meta").stale);

        let wrapped = normalize(json!({
            "data": {
                "models": [{
                    "id": "anthropic/claude",
                    "label": "claude",
                    "provider": "anthropic",
                    "model": "claude"
                }],
                "meta": { "refresh_in_progress": true }
            }
        }));
        assert_eq!(wrapped.models[0].provider, "anthropic");
        assert!(wrapped.should_retry_empty());
    }
}
