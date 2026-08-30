use serde::Deserialize;

// Backend profile metadata is retained even when not currently displayed.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProfileInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub config_kind: Option<String>,
    #[serde(default)]
    pub fingerprint: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}
