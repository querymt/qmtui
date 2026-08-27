use std::path::PathBuf;

use tokio::sync::Mutex;

use super::assistant_buffer::AssistantBuffers;
use super::elicitation::ElicitationRegistry;
use super::extensions::models::Model;
use super::replay::ReplayBuffer;

#[derive(Debug, Clone)]
pub(super) struct AgentIdentity {
    pub(super) id: String,
    pub(super) name: String,
}

impl Default for AgentIdentity {
    fn default() -> Self {
        Self {
            id: "querymt".to_string(),
            name: "QueryMT".to_string(),
        }
    }
}

pub(super) struct RuntimeState {
    agent: Mutex<AgentIdentity>,
    current_session_id: Mutex<Option<String>>,
    pub(super) replay: ReplayBuffer,
    pub(super) assistants: AssistantBuffers,
    pub(super) elicitations: ElicitationRegistry,
    models: Mutex<Vec<Model>>,
    selected_model_id: Mutex<Option<String>>,
    launch_cwd: Option<String>,
}

impl RuntimeState {
    pub(super) fn new(launch_cwd: Option<String>) -> Self {
        Self {
            agent: Mutex::new(AgentIdentity::default()),
            current_session_id: Mutex::new(None),
            replay: ReplayBuffer::default(),
            assistants: AssistantBuffers::default(),
            elicitations: ElicitationRegistry::default(),
            models: Mutex::new(Vec::new()),
            selected_model_id: Mutex::new(None),
            launch_cwd,
        }
    }

    pub(super) async fn agent_identity(&self) -> AgentIdentity {
        self.agent.lock().await.clone()
    }

    pub(super) async fn set_agent_identity(&self, identity: AgentIdentity) {
        *self.agent.lock().await = identity;
    }

    pub(super) async fn current_session_id(&self) -> Option<String> {
        self.current_session_id.lock().await.clone()
    }

    pub(super) async fn set_current_session_id(&self, session_id: impl Into<String>) {
        *self.current_session_id.lock().await = Some(session_id.into());
    }

    pub(super) async fn set_models(&self, models: Vec<Model>) {
        *self.models.lock().await = models;
    }

    pub(super) async fn model_by_id(&self, model_id: &str) -> Option<Model> {
        self.models
            .lock()
            .await
            .iter()
            .find(|model| model.id == model_id)
            .cloned()
    }

    pub(super) async fn select_model(&self, model_id: impl Into<String>) {
        *self.selected_model_id.lock().await = Some(model_id.into());
    }

    pub(super) async fn selected_or_default_model(&self) -> Option<Model> {
        let selected = self.selected_model_id.lock().await.clone();
        let models = self.models.lock().await;
        selected
            .as_deref()
            .and_then(|id| models.iter().find(|model| model.id == id))
            .or_else(|| models.first())
            .cloned()
    }

    pub(super) fn default_cwd(&self) -> PathBuf {
        self.launch_cwd
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }
}
