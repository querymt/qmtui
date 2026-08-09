use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegateModelPreference {
    pub model_id: String,
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub label: String,
    pub provider: String,
    pub model: String,
    pub node_id: Option<String>,
    #[serde(default)]
    pub node_label: Option<String>,
    #[allow(dead_code)] // Preserved from ACP model metadata for future presentation.
    pub family: Option<String>,
    #[allow(dead_code)] // Preserved from ACP model metadata for future presentation.
    pub quant: Option<String>,
}
