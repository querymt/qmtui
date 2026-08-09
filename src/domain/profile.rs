use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProfileInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[allow(dead_code)] // Preserved from the profile catalog wire contract.
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub config_kind: Option<String>,
    #[allow(dead_code)] // Preserved from the profile catalog wire contract.
    #[serde(default)]
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    #[allow(dead_code)] // Preserved from the profile-agent wire contract.
    #[serde(default)]
    pub description: Option<String>,
    #[allow(dead_code)] // Preserved from the profile-agent wire contract.
    #[serde(default)]
    pub capabilities: Vec<String>,
}
