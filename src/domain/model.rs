use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DelegateModelOverrideInfo {
    pub model_id: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub node_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegateModelPreference {
    pub model_id: String,
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
}

// Provider metadata remains part of the decoded model contract.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub label: String,
    pub provider: String,
    pub model: String,
    pub node_id: Option<String>,
    #[serde(default)]
    pub node_label: Option<String>,
    pub family: Option<String>,
    pub quant: Option<String>,
}
