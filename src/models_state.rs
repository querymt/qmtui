use std::collections::{BTreeMap, HashMap, HashSet};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::application::Effect;
use crate::command::Command;
use crate::diagnostics::LogLevel;
use crate::domain::model::{DelegateModelPreference, ModelEntry};
use crate::domain::profile::AgentInfo;

#[derive(Debug, Clone)]
pub(crate) enum ModelAction {
    InitializePrimaryAgent(AgentInfo),
    InitializeReasoningEffort(Option<String>),
    ReasoningEffort(Option<String>),
    ReplaceCatalog {
        models: Vec<ModelEntry>,
        remote_node_count: u32,
        remote_timeout_count: u32,
    },
    SynchronizeProfileCatalog {
        active_profile_changed: bool,
    },
    ReplaceProfileAgents {
        profile_id: String,
        agents: Vec<AgentInfo>,
    },
    ProviderChanged {
        provider: String,
        model: String,
        node_id: Option<String>,
        context_limit: Option<u64>,
    },
    DelegateModelSet {
        session_id: String,
        agent_id: String,
        model_id: Option<String>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ModelContext {
    pub(crate) desired_profile_id: Option<String>,
    pub(crate) root_session_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelCoordination {
    Log {
        level: LogLevel,
        target: &'static str,
        message: String,
    },
    Status {
        level: LogLevel,
        target: &'static str,
        message: String,
    },
    SetContextLimit(u64),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ModelOutcome {
    pub(crate) coordination: Vec<ModelCoordination>,
    pub(crate) effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelPopupItem {
    /// When set, `node_suffix` renders the remote mesh node on the right.
    ProviderHeader {
        provider: String,
        model_count: usize,
        node_suffix: Option<String>,
    },
    Model {
        model_idx: usize,
    },
}

pub(crate) struct ModelsState {
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) current_model: Option<String>,
    pub(crate) current_provider: Option<String>,
    pub(crate) current_model_node_id: Option<String>,
    pub(crate) models: Vec<ModelEntry>,
    pub(crate) model_cursor: usize,
    pub(crate) model_filter: String,
    pub(crate) agents: Vec<AgentInfo>,
    pub(crate) agents_profile_id: Option<String>,
    pub(crate) model_popup_agent_tab: usize,
    pub(crate) delegate_model_preferences:
        HashMap<String, HashMap<String, DelegateModelPreference>>,
}

/// Validate and normalize a reasoning-effort string.
///
/// Returns `Some(None)` for the automatic level and `None` for an invalid level.
pub(crate) fn validate_reasoning_effort(s: Option<&str>) -> Option<Option<String>> {
    match s {
        None | Some("auto") | Some("") => Some(None),
        Some("low") => Some(Some("low".to_string())),
        Some("medium") | Some("med") => Some(Some("medium".to_string())),
        Some("high") => Some(Some("high".to_string())),
        Some("max") => Some(Some("max".to_string())),
        Some(_) => None,
    }
}

impl ModelsState {
    pub(crate) fn reduce(&mut self, action: ModelAction, context: ModelContext) -> ModelOutcome {
        match action {
            ModelAction::InitializePrimaryAgent(agent) => {
                self.initialize_primary_agent(agent);
                ModelOutcome::default()
            }
            ModelAction::InitializeReasoningEffort(reasoning_effort) => {
                self.reasoning_effort = reasoning_effort;
                ModelOutcome::default()
            }
            ModelAction::ReasoningEffort(reasoning_effort) => {
                if let Some(validated) = validate_reasoning_effort(reasoning_effort.as_deref()) {
                    self.reasoning_effort = validated;
                }
                ModelOutcome::default()
            }
            ModelAction::ReplaceCatalog {
                models,
                remote_node_count,
                remote_timeout_count,
            } => {
                let (total_models, remote_models) = self.replace_catalog(models);
                let mut message = format!("models: {total_models} total, {remote_models} remote");
                if remote_node_count > 0 || remote_timeout_count > 0 {
                    message.push_str(&format!(
                        " (inventory nodes={remote_node_count}, timeouts={remote_timeout_count})"
                    ));
                }
                ModelOutcome {
                    coordination: vec![ModelCoordination::Log {
                        level: LogLevel::Info,
                        target: "models",
                        message,
                    }],
                    effects: Vec::new(),
                }
            }
            ModelAction::SynchronizeProfileCatalog {
                active_profile_changed,
            } => {
                let desired_profile_id = context.desired_profile_id;
                if active_profile_changed
                    || self.agents_profile_id.as_deref() != desired_profile_id.as_deref()
                    || self.agents.len() <= 1
                {
                    self.clear_profile_agents();
                    ModelOutcome {
                        coordination: Vec::new(),
                        effects: desired_profile_id
                            .map(|profile_id| {
                                Effect::Command(Command::ListProfileAgents { profile_id })
                            })
                            .into_iter()
                            .collect(),
                    }
                } else {
                    ModelOutcome::default()
                }
            }
            ModelAction::ReplaceProfileAgents { profile_id, agents } => {
                if context.desired_profile_id.as_deref() != Some(profile_id.as_str()) {
                    return ModelOutcome::default();
                }
                self.replace_profile_agents(profile_id.clone(), agents);
                ModelOutcome {
                    coordination: Vec::new(),
                    effects: context
                        .root_session_id
                        .map(|session_id| {
                            self.delegate_model_commands_for_session(&session_id, &profile_id)
                        })
                        .unwrap_or_default()
                        .into_iter()
                        .map(Effect::Command)
                        .collect(),
                }
            }
            ModelAction::ProviderChanged {
                provider,
                model,
                node_id,
                context_limit,
            } => {
                self.replace_live_selection(provider, model, node_id);
                ModelOutcome {
                    coordination: context_limit
                        .map(ModelCoordination::SetContextLimit)
                        .into_iter()
                        .collect(),
                    effects: Vec::new(),
                }
            }
            ModelAction::DelegateModelSet {
                session_id,
                agent_id,
                model_id,
            } => ModelOutcome {
                coordination: vec![ModelCoordination::Status {
                    level: LogLevel::Info,
                    target: "model",
                    message: match model_id {
                        Some(model_id) => {
                            format!("delegate model set for {agent_id} in {session_id}: {model_id}")
                        }
                        None => format!("delegate model reset for {agent_id} in {session_id}"),
                    },
                }],
                effects: Vec::new(),
            },
        }
    }

    /// Valid explicit reasoning levels; automatic effort is represented by `None`.
    #[cfg(test)]
    pub const EFFORT_LEVELS: &[&str] = &["low", "medium", "high", "max"];

    pub(crate) fn new() -> Self {
        Self {
            reasoning_effort: None,
            current_model: None,
            current_provider: None,
            current_model_node_id: None,
            models: Vec::new(),
            model_cursor: 0,
            model_filter: String::new(),
            agents: Vec::new(),
            agents_profile_id: None,
            model_popup_agent_tab: 0,
            delegate_model_preferences: HashMap::new(),
        }
    }

    pub(crate) fn reasoning_effort_label(&self) -> &str {
        self.reasoning_effort.as_deref().unwrap_or("auto")
    }

    /// Set a validated effort and return the wire value expected by the server.
    pub(crate) fn set_reasoning_effort(&mut self, level: Option<&str>) -> Option<String> {
        let normalized = validate_reasoning_effort(level)?;
        self.reasoning_effort = normalized;
        Some(self.reasoning_effort_label().to_string())
    }

    /// Advance `auto -> low -> medium -> high -> max -> auto`.
    pub(crate) fn cycle_reasoning_effort(&mut self) -> Option<String> {
        const LEVELS: &[Option<&str>] =
            &[None, Some("low"), Some("medium"), Some("high"), Some("max")];
        let current = self.reasoning_effort.as_deref();
        let idx = LEVELS
            .iter()
            .position(|level| level.as_deref() == current)?;
        self.set_reasoning_effort(LEVELS[(idx + 1) % LEVELS.len()])
    }

    /// Replace the authoritative catalog without reconciling any other model state.
    pub(crate) fn replace_catalog(&mut self, models: Vec<ModelEntry>) -> (usize, usize) {
        self.models = models;
        let remote = self
            .models
            .iter()
            .filter(|model| model.node_id.is_some())
            .count();
        (self.models.len(), remote)
    }

    pub(crate) fn replace_live_selection(
        &mut self,
        provider: String,
        model: String,
        node_id: Option<String>,
    ) {
        self.current_provider = Some(provider);
        self.current_model = Some(model);
        self.current_model_node_id = node_id;
    }

    pub(crate) fn apply_model_selection_from_entry(&mut self, entry: &ModelEntry) {
        self.replace_live_selection(
            entry.provider.clone(),
            entry.model.clone(),
            entry.node_id.clone(),
        );
    }

    fn filtered_model_indices(&self) -> Vec<usize> {
        if self.model_filter.is_empty() {
            (0..self.models.len()).collect()
        } else {
            let matcher = SkimMatcherV2::default();
            let mut scored: Vec<(i64, usize)> = self
                .models
                .iter()
                .enumerate()
                .filter_map(|(index, model)| {
                    let score = [&model.label, &model.provider, &model.model]
                        .iter()
                        .filter_map(|field| matcher.fuzzy_match(field, &self.model_filter))
                        .max();
                    score.map(|score| (score, index))
                })
                .collect();
            scored.sort_by_key(|item| std::cmp::Reverse(item.0));
            scored.into_iter().map(|(_, index)| index).collect()
        }
    }

    #[cfg(test)]
    pub(crate) fn filtered_models(&self) -> Vec<&ModelEntry> {
        self.filtered_model_indices()
            .into_iter()
            .map(|index| &self.models[index])
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn model_index_for_entry(&self, entry: &ModelEntry) -> Option<usize> {
        self.models
            .iter()
            .position(|model| model.id == entry.id && model.node_id == entry.node_id)
    }

    /// Match a catalog row to a provider/model pair, disambiguating local and mesh rows.
    pub(crate) fn model_entry_matches_node(
        entry: &ModelEntry,
        provider: &str,
        model: &str,
        node_id: Option<&str>,
    ) -> bool {
        if entry.provider != provider || entry.model != model {
            return false;
        }
        match node_id {
            Some(node) => entry.node_id.as_deref() == Some(node),
            None => entry.node_id.is_none(),
        }
    }

    pub(crate) fn live_model_selection_matches_entry(&self, entry: &ModelEntry) -> bool {
        let (Some(provider), Some(model)) = (
            self.current_provider.as_deref(),
            self.current_model.as_deref(),
        ) else {
            return false;
        };
        Self::model_entry_matches_node(
            entry,
            provider,
            model,
            self.current_model_node_id.as_deref(),
        )
    }

    pub(crate) fn model_popup_open_cursor(&self) -> usize {
        let items = self.visible_model_popup_items();
        items
            .iter()
            .position(|item| match item {
                ModelPopupItem::Model { model_idx } => self
                    .models
                    .get(*model_idx)
                    .is_some_and(|entry| self.live_model_selection_matches_entry(entry)),
                ModelPopupItem::ProviderHeader { .. } => false,
            })
            .or_else(|| Self::first_model_cursor(&items))
            .unwrap_or(0)
    }

    pub(crate) fn visible_model_popup_items(&self) -> Vec<ModelPopupItem> {
        let mut groups: BTreeMap<(String, Option<String>), Vec<usize>> = BTreeMap::new();
        for model_idx in self.filtered_model_indices() {
            let model = &self.models[model_idx];
            groups
                .entry((model.provider.clone(), model.node_id.clone()))
                .or_default()
                .push(model_idx);
        }

        let mut items = Vec::new();
        for ((provider, node_id), models_in_group) in groups {
            let node_suffix = node_id.as_ref().map(|node_id| {
                models_in_group
                    .first()
                    .and_then(|&index| self.models[index].node_label.clone())
                    .unwrap_or_else(|| node_id.clone())
            });
            items.push(ModelPopupItem::ProviderHeader {
                provider,
                model_count: models_in_group.len(),
                node_suffix,
            });
            items.extend(
                models_in_group
                    .into_iter()
                    .map(|model_idx| ModelPopupItem::Model { model_idx }),
            );
        }
        items
    }

    pub(crate) fn reset_for_open(&mut self) {
        self.model_filter.clear();
        self.model_popup_agent_tab = 0;
        self.model_cursor = self.model_popup_open_cursor();
    }

    pub(crate) fn replace_filter(&mut self, filter: String) {
        self.model_filter = filter;
        self.model_cursor = 0;
    }

    pub(crate) fn model_filter_insert(&mut self, character: char) {
        self.model_filter.push(character);
        self.model_cursor = 0;
    }

    pub(crate) fn model_filter_backspace(&mut self) {
        self.model_filter.pop();
        self.model_cursor = 0;
    }

    pub(crate) fn move_model_cursor_up(&mut self) {
        self.model_cursor = self.model_cursor.saturating_sub(1);
    }

    pub(crate) fn move_model_cursor_down(&mut self) {
        let max = self.visible_model_popup_items().len().saturating_sub(1);
        self.model_cursor = (self.model_cursor + 1).min(max);
    }

    /// Switch popup tabs and return the newly selected delegate agent, if any.
    pub(crate) fn switch_model_popup_tab(&mut self, backwards: bool) -> Option<String> {
        if !self.model_popup_has_tabs() {
            return None;
        }
        let tab_count = self.model_popup_tab_count();
        self.model_popup_agent_tab = if backwards {
            if self.model_popup_agent_tab == 0 {
                tab_count - 1
            } else {
                self.model_popup_agent_tab - 1
            }
        } else {
            (self.model_popup_agent_tab + 1) % tab_count
        };
        self.model_filter.clear();
        self.model_popup_tab_agent_id(self.model_popup_agent_tab)
            .map(str::to_string)
    }

    pub(crate) fn initialize_primary_agent(&mut self, agent: AgentInfo) {
        self.agents = vec![agent];
        self.agents_profile_id = None;
    }

    pub(crate) fn replace_profile_agents(&mut self, profile_id: String, agents: Vec<AgentInfo>) {
        self.agents = agents;
        self.agents_profile_id = Some(profile_id);
        self.model_popup_agent_tab = self
            .model_popup_agent_tab
            .min(self.model_popup_tab_count().saturating_sub(1));
    }

    pub(crate) fn clear_profile_agents(&mut self) {
        self.agents.clear();
        self.agents_profile_id = None;
        self.model_popup_agent_tab = 0;
    }

    pub(crate) fn is_multi_agent(&self) -> bool {
        self.agents.len() > 1
    }

    /// Tabs are the session plus direct agent indices when delegates are available.
    pub(crate) fn model_popup_tab_count(&self) -> usize {
        if self.is_multi_agent() {
            self.agents.len()
        } else {
            1
        }
    }

    pub(crate) fn model_popup_has_tabs(&self) -> bool {
        self.is_multi_agent()
    }

    pub(crate) fn model_popup_tab_label(&self, tab_idx: usize) -> &str {
        if self.model_popup_is_session_tab(tab_idx) {
            "session"
        } else {
            self.agents
                .get(tab_idx)
                .map(|agent| agent.name.as_str())
                .unwrap_or("???")
        }
    }

    pub(crate) fn model_popup_tab_agent_id(&self, tab_idx: usize) -> Option<&str> {
        if self.model_popup_is_session_tab(tab_idx) {
            None
        } else {
            self.agents.get(tab_idx).map(|agent| agent.id.as_str())
        }
    }

    pub(crate) fn model_popup_is_session_tab(&self, tab_idx: usize) -> bool {
        tab_idx == 0
    }

    pub(crate) fn set_delegate_model_preference(
        &mut self,
        profile_id: &str,
        agent_id: &str,
        model: &ModelEntry,
    ) {
        self.delegate_model_preferences
            .entry(profile_id.to_string())
            .or_default()
            .insert(
                agent_id.to_string(),
                DelegateModelPreference {
                    model_id: model.id.clone(),
                    provider: model.provider.clone(),
                    model: model.model.clone(),
                    node_id: model.node_id.clone(),
                },
            );
    }

    pub(crate) fn clear_delegate_model_preference(&mut self, profile_id: &str, agent_id: &str) {
        if let Some(preferences) = self.delegate_model_preferences.get_mut(profile_id) {
            preferences.remove(agent_id);
            if preferences.is_empty() {
                self.delegate_model_preferences.remove(profile_id);
            }
        }
    }

    pub(crate) fn get_delegate_model_preference(
        &self,
        profile_id: &str,
        agent_id: &str,
    ) -> Option<&DelegateModelPreference> {
        self.delegate_model_preferences
            .get(profile_id)
            .and_then(|preferences| preferences.get(agent_id))
    }

    pub(crate) fn delegate_model_commands_for_session(
        &self,
        session_id: &str,
        profile_id: &str,
    ) -> Vec<Command> {
        let known_agents: HashSet<&str> = self
            .agents
            .iter()
            .skip(1)
            .map(|agent| agent.id.as_str())
            .collect();
        self.delegate_model_preferences
            .get(profile_id)
            .into_iter()
            .flat_map(|preferences| preferences.iter())
            .filter(|(agent_id, _)| known_agents.contains(agent_id.as_str()))
            .map(|(agent_id, preference)| Command::SetDelegateModel {
                session_id: session_id.to_string(),
                agent_id: agent_id.clone(),
                model_id: Some(preference.model_id.clone()),
                node_id: preference.node_id.clone(),
            })
            .collect()
    }

    pub(crate) fn delegate_model_cursor(&self, profile_id: Option<&str>, agent_id: &str) -> usize {
        let items = self.visible_model_popup_items();
        let Some(profile_id) = profile_id else {
            return Self::first_model_cursor(&items).unwrap_or(0);
        };
        let Some(preference) = self.get_delegate_model_preference(profile_id, agent_id) else {
            return Self::first_model_cursor(&items).unwrap_or(0);
        };
        items
            .iter()
            .position(|item| match item {
                ModelPopupItem::Model { model_idx } => {
                    self.models[*model_idx].id == preference.model_id
                        && self.models[*model_idx].node_id == preference.node_id
                }
                ModelPopupItem::ProviderHeader { .. } => false,
            })
            .unwrap_or(0)
    }

    fn first_model_cursor(items: &[ModelPopupItem]) -> Option<usize> {
        items
            .iter()
            .position(|item| matches!(item, ModelPopupItem::Model { .. }))
    }

    #[cfg(test)]
    pub(crate) fn test_model_entry(
        id: &str,
        provider: &str,
        model: &str,
        node_id: Option<&str>,
        node_label: Option<&str>,
    ) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            label: model.into(),
            provider: provider.into(),
            model: model.into(),
            node_id: node_id.map(str::to_string),
            node_label: node_label.map(str::to_string),
            family: None,
            quant: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(
        id: &str,
        label: &str,
        provider: &str,
        model: &str,
        node_id: Option<&str>,
        node_label: Option<&str>,
    ) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            label: label.into(),
            provider: provider.into(),
            model: model.into(),
            node_id: node_id.map(str::to_string),
            node_label: node_label.map(str::to_string),
            family: None,
            quant: None,
        }
    }

    fn agent(id: &str, name: &str) -> AgentInfo {
        AgentInfo {
            id: id.into(),
            name: name.into(),
            description: None,
            capabilities: Vec::new(),
        }
    }

    #[test]
    fn reducer_initialization_and_reasoning_effort_preserve_validation_contracts() {
        let mut state = ModelsState::new();
        state.model_popup_agent_tab = 3;
        state.reasoning_effort = Some("low".into());

        assert_eq!(
            state.reduce(
                ModelAction::InitializePrimaryAgent(agent("primary", "Primary")),
                ModelContext::default(),
            ),
            ModelOutcome::default()
        );
        assert_eq!(state.agents[0].id, "primary");
        assert_eq!(state.agents_profile_id, None);
        assert_eq!(state.model_popup_agent_tab, 3);

        state.reduce(
            ModelAction::InitializeReasoningEffort(Some("backend-level".into())),
            ModelContext::default(),
        );
        assert_eq!(state.reasoning_effort.as_deref(), Some("backend-level"));
        state.reduce(
            ModelAction::ReasoningEffort(Some("med".into())),
            ModelContext::default(),
        );
        assert_eq!(state.reasoning_effort.as_deref(), Some("medium"));
        state.reduce(
            ModelAction::ReasoningEffort(Some("invalid".into())),
            ModelContext::default(),
        );
        assert_eq!(state.reasoning_effort.as_deref(), Some("medium"));
    }

    #[test]
    fn reducer_catalog_preserves_model_ui_state_and_reports_exact_diagnostics() {
        let mut state = ModelsState::new();
        state.current_model = Some("old-model".into());
        state.current_provider = Some("old-provider".into());
        state.current_model_node_id = Some("old-node".into());
        state.model_cursor = 9;
        state.model_filter = "old-filter".into();
        state.model_popup_agent_tab = 3;
        state.reasoning_effort = Some("high".into());

        let outcome = state.reduce(
            ModelAction::ReplaceCatalog {
                models: vec![
                    model("local", "local", "p", "m", None, None),
                    model("remote", "remote", "p", "m", Some("node"), None),
                ],
                remote_node_count: 2,
                remote_timeout_count: 1,
            },
            ModelContext::default(),
        );

        assert_eq!(state.models.len(), 2);
        assert_eq!(state.current_model.as_deref(), Some("old-model"));
        assert_eq!(state.current_provider.as_deref(), Some("old-provider"));
        assert_eq!(state.current_model_node_id.as_deref(), Some("old-node"));
        assert_eq!(state.model_cursor, 9);
        assert_eq!(state.model_filter, "old-filter");
        assert_eq!(state.model_popup_agent_tab, 3);
        assert_eq!(state.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(
            outcome,
            ModelOutcome {
                coordination: vec![ModelCoordination::Log {
                    level: LogLevel::Info,
                    target: "models",
                    message: "models: 2 total, 1 remote (inventory nodes=2, timeouts=1)".into(),
                }],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn reducer_profile_synchronization_preserves_refresh_and_stale_response_contracts() {
        let mut state = ModelsState::new();
        state.agents_profile_id = Some("quorum".into());
        state.agents = vec![agent("primary", "Primary"), agent("coder", "Coder")];
        state.model_popup_agent_tab = 1;

        let outcome = state.reduce(
            ModelAction::SynchronizeProfileCatalog {
                active_profile_changed: false,
            },
            ModelContext {
                desired_profile_id: Some("quorum".into()),
                root_session_id: None,
            },
        );
        assert_eq!(outcome, ModelOutcome::default());
        assert_eq!(state.agents.len(), 2);

        let outcome = state.reduce(
            ModelAction::SynchronizeProfileCatalog {
                active_profile_changed: true,
            },
            ModelContext {
                desired_profile_id: Some("fast".into()),
                root_session_id: None,
            },
        );
        assert!(state.agents.is_empty());
        assert_eq!(state.agents_profile_id, None);
        assert_eq!(state.model_popup_agent_tab, 0);
        assert_eq!(
            outcome.effects,
            vec![Effect::Command(Command::ListProfileAgents {
                profile_id: "fast".into(),
            })]
        );

        state.agents = vec![agent("sentinel", "Sentinel")];
        let outcome = state.reduce(
            ModelAction::ReplaceProfileAgents {
                profile_id: "stale".into(),
                agents: vec![agent("stale", "Stale")],
            },
            ModelContext {
                desired_profile_id: Some("fast".into()),
                root_session_id: Some("session".into()),
            },
        );
        assert_eq!(outcome, ModelOutcome::default());
        assert_eq!(state.agents[0].id, "sentinel");
    }

    #[test]
    fn reducer_current_profile_agents_update_tabs_before_reapplying_root_preferences() {
        let mut state = ModelsState::new();
        state.delegate_model_preferences.insert(
            "quorum".into(),
            [(
                "coder".into(),
                DelegateModelPreference {
                    model_id: "openai/gpt-5".into(),
                    provider: "openai".into(),
                    model: "gpt-5".into(),
                    node_id: Some("node-1".into()),
                },
            )]
            .into_iter()
            .collect(),
        );

        let outcome = state.reduce(
            ModelAction::ReplaceProfileAgents {
                profile_id: "quorum".into(),
                agents: vec![agent("primary", "Primary"), agent("coder", "Coder")],
            },
            ModelContext {
                desired_profile_id: Some("quorum".into()),
                root_session_id: Some("parent".into()),
            },
        );

        assert_eq!(state.agents_profile_id.as_deref(), Some("quorum"));
        assert_eq!(state.model_popup_tab_agent_id(1), Some("coder"));
        assert_eq!(
            outcome.effects,
            vec![Effect::Command(Command::SetDelegateModel {
                session_id: "parent".into(),
                agent_id: "coder".into(),
                model_id: Some("openai/gpt-5".into()),
                node_id: Some("node-1".into()),
            })]
        );
    }

    #[test]
    fn reducer_provider_and_delegate_events_return_ordered_coordination() {
        let mut state = ModelsState::new();
        let outcome = state.reduce(
            ModelAction::ProviderChanged {
                provider: "provider".into(),
                model: "model".into(),
                node_id: Some("node".into()),
                context_limit: Some(128_000),
            },
            ModelContext::default(),
        );
        assert_eq!(state.current_provider.as_deref(), Some("provider"));
        assert_eq!(state.current_model.as_deref(), Some("model"));
        assert_eq!(state.current_model_node_id.as_deref(), Some("node"));
        assert_eq!(
            outcome.coordination,
            vec![ModelCoordination::SetContextLimit(128_000)]
        );

        let outcome = state.reduce(
            ModelAction::DelegateModelSet {
                session_id: "session".into(),
                agent_id: "coder".into(),
                model_id: Some("openai/gpt-5".into()),
            },
            ModelContext::default(),
        );
        assert_eq!(
            outcome.coordination,
            vec![ModelCoordination::Status {
                level: LogLevel::Info,
                target: "model",
                message: "delegate model set for coder in session: openai/gpt-5".into(),
            }]
        );
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = ModelsState::new();
        assert_eq!(state.reasoning_effort, None);
        assert_eq!(state.current_model, None);
        assert_eq!(state.current_provider, None);
        assert_eq!(state.current_model_node_id, None);
        assert!(state.models.is_empty());
        assert_eq!(state.model_cursor, 0);
        assert!(state.model_filter.is_empty());
        assert!(state.agents.is_empty());
        assert_eq!(state.agents_profile_id, None);
        assert_eq!(state.model_popup_agent_tab, 0);
        assert!(state.delegate_model_preferences.is_empty());
    }

    #[test]
    fn reasoning_validation_normalizes_only_supported_case_sensitive_values() {
        assert_eq!(ModelsState::EFFORT_LEVELS, ["low", "medium", "high", "max"]);
        assert_eq!(validate_reasoning_effort(None), Some(None));
        assert_eq!(validate_reasoning_effort(Some("")), Some(None));
        assert_eq!(validate_reasoning_effort(Some("auto")), Some(None));
        assert_eq!(
            validate_reasoning_effort(Some("low")),
            Some(Some("low".into()))
        );
        assert_eq!(
            validate_reasoning_effort(Some("medium")),
            Some(Some("medium".into()))
        );
        assert_eq!(
            validate_reasoning_effort(Some("med")),
            Some(Some("medium".into()))
        );
        assert_eq!(
            validate_reasoning_effort(Some("high")),
            Some(Some("high".into()))
        );
        assert_eq!(
            validate_reasoning_effort(Some("max")),
            Some(Some("max".into()))
        );
        assert_eq!(validate_reasoning_effort(Some("MED")), None);
        assert_eq!(validate_reasoning_effort(Some("ultra")), None);
    }

    #[test]
    fn reasoning_label_cycle_and_unknown_state_preserve_contract() {
        let mut state = ModelsState::new();
        assert_eq!(state.reasoning_effort_label(), "auto");
        for expected in ["low", "medium", "high", "max", "auto"] {
            assert_eq!(state.cycle_reasoning_effort().as_deref(), Some(expected));
            assert_eq!(state.reasoning_effort_label(), expected);
        }
        state.reasoning_effort = Some("ultra".into());
        assert_eq!(state.reasoning_effort_label(), "ultra");
        assert_eq!(state.cycle_reasoning_effort(), None);
        assert_eq!(state.reasoning_effort.as_deref(), Some("ultra"));
    }

    #[test]
    fn invalid_reasoning_set_does_not_mutate_state() {
        let mut state = ModelsState::new();
        state.reasoning_effort = Some("high".into());
        assert_eq!(state.set_reasoning_effort(Some("HIGH")), None);
        assert_eq!(state.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(
            state.set_reasoning_effort(Some("auto")).as_deref(),
            Some("auto")
        );
        assert_eq!(state.reasoning_effort, None);
    }

    #[test]
    fn authoritative_catalog_replacement_reports_counts_and_preserves_stale_state() {
        let mut state = ModelsState::new();
        state.current_model = Some("old-model".into());
        state.current_provider = Some("old-provider".into());
        state.current_model_node_id = Some("old-node".into());
        state.model_cursor = 9;
        state.model_filter = "old-filter".into();
        state.model_popup_agent_tab = 3;
        state.reasoning_effort = Some("high".into());
        state.set_delegate_model_preference(
            "profile",
            "coder",
            &model("old", "old", "old-provider", "old-model", None, None),
        );

        let counts = state.replace_catalog(vec![
            model("local", "local", "p", "m", None, None),
            model("remote", "remote", "p", "m", Some("node"), None),
        ]);
        assert_eq!(counts, (2, 1));
        assert_eq!(state.current_model.as_deref(), Some("old-model"));
        assert_eq!(state.current_provider.as_deref(), Some("old-provider"));
        assert_eq!(state.current_model_node_id.as_deref(), Some("old-node"));
        assert_eq!(state.model_cursor, 9);
        assert_eq!(state.model_filter, "old-filter");
        assert_eq!(state.model_popup_agent_tab, 3);
        assert_eq!(state.reasoning_effort.as_deref(), Some("high"));
        assert!(
            state
                .get_delegate_model_preference("profile", "coder")
                .is_some()
        );

        assert_eq!(state.replace_catalog(Vec::new()), (0, 0));
        assert_eq!(state.model_cursor, 9);
        assert_eq!(state.model_filter, "old-filter");
    }

    #[test]
    fn filtering_searches_only_label_provider_and_model_with_stable_ties() {
        let mut state = ModelsState::new();
        state.models = vec![
            model("first-id", "same", "provider", "model", None, None),
            model("second-id", "same", "provider", "model", None, None),
            model(
                "secret-id-match",
                "other",
                "elsewhere",
                "different",
                None,
                None,
            ),
            ModelEntry {
                family: Some("secret-id-match".into()),
                quant: Some("secret-id-match".into()),
                node_label: Some("secret-id-match".into()),
                ..model(
                    "excluded",
                    "other",
                    "elsewhere",
                    "different",
                    Some("secret-id-match"),
                    None,
                )
            },
        ];
        assert_eq!(
            state
                .filtered_models()
                .into_iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            ["first-id", "second-id", "secret-id-match", "excluded"]
        );

        state.model_filter = "same".into();
        assert_eq!(
            state
                .filtered_models()
                .into_iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            ["first-id", "second-id"]
        );
        state.model_filter = "secret-id-match".into();
        assert!(state.filtered_models().is_empty());
    }

    #[test]
    fn filtering_matches_each_searchable_field() {
        let mut state = ModelsState::new();
        state.models = vec![
            model("provider-hit", "zzz", "alpha", "zzz", None, None),
            model("model-hit", "zzz", "zzz", "alpha", None, None),
            model("label-hit", "alpha", "zzz", "zzz", None, None),
        ];
        state.model_filter = "alpha".into();
        let ids: Vec<&str> = state
            .filtered_models()
            .into_iter()
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&"provider-hit"));
        assert!(ids.contains(&"model-hit"));
        assert!(ids.contains(&"label-hit"));
    }

    #[test]
    fn filtering_uses_the_maximum_score_across_searchable_fields() {
        let mut state = ModelsState::new();
        state.models = vec![
            model("best", "zz", "alpha", "al", None, None),
            model("weaker", "al", "zz", "zz", None, None),
        ];
        state.model_filter = "alpha".into();
        let ids: Vec<&str> = state
            .filtered_models()
            .into_iter()
            .map(|entry| entry.id.as_str())
            .collect();
        assert_eq!(ids.first().copied(), Some("best"));
    }

    #[test]
    fn visible_items_group_by_provider_and_node_with_local_first() {
        let mut state = ModelsState::new();
        state.models = vec![
            model(
                "z-remote-2",
                "r2",
                "zeta",
                "r2",
                Some("node-b"),
                Some("box"),
            ),
            model("a-remote", "ar", "alpha", "ar", Some("node-a"), None),
            model("z-local-1", "l1", "zeta", "l1", None, None),
            model(
                "z-remote-1",
                "r1",
                "zeta",
                "r1",
                Some("node-b"),
                Some("box"),
            ),
            model("z-local-2", "l2", "zeta", "l2", None, None),
        ];

        assert_eq!(
            state.visible_model_popup_items(),
            vec![
                ModelPopupItem::ProviderHeader {
                    provider: "alpha".into(),
                    model_count: 1,
                    node_suffix: Some("node-a".into()),
                },
                ModelPopupItem::Model { model_idx: 1 },
                ModelPopupItem::ProviderHeader {
                    provider: "zeta".into(),
                    model_count: 2,
                    node_suffix: None,
                },
                ModelPopupItem::Model { model_idx: 2 },
                ModelPopupItem::Model { model_idx: 4 },
                ModelPopupItem::ProviderHeader {
                    provider: "zeta".into(),
                    model_count: 2,
                    node_suffix: Some("box".into()),
                },
                ModelPopupItem::Model { model_idx: 0 },
                ModelPopupItem::Model { model_idx: 3 },
            ]
        );
    }

    #[test]
    fn visible_items_preserve_exact_catalog_indices_with_filter_and_duplicates() {
        let mut state = ModelsState::new();
        state.models = vec![
            model("duplicate", "match first", "p", "same", Some("node"), None),
            model("other", "excluded", "p", "other", None, None),
            model("duplicate", "match second", "p", "same", Some("node"), None),
        ];
        state.model_filter = "match".into();

        let indices = state
            .visible_model_popup_items()
            .into_iter()
            .filter_map(|item| match item {
                ModelPopupItem::Model { model_idx } => Some(model_idx),
                ModelPopupItem::ProviderHeader { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(indices, [0, 2]);
    }

    #[test]
    fn model_identity_and_live_selection_include_exact_node_identity() {
        let local = model("same", "local", "p", "m", None, None);
        let remote = model("same", "remote", "p", "m", Some("node"), None);
        let other_remote = model("same", "remote", "p", "m", Some("other"), None);
        let mut state = ModelsState::new();
        state.models = vec![local.clone(), remote.clone(), other_remote.clone()];
        assert_eq!(state.model_index_for_entry(&local), Some(0));
        assert_eq!(state.model_index_for_entry(&remote), Some(1));

        state.apply_model_selection_from_entry(&remote);
        assert!(state.live_model_selection_matches_entry(&remote));
        assert!(!state.live_model_selection_matches_entry(&local));
        assert!(!state.live_model_selection_matches_entry(&other_remote));
        assert!(ModelsState::model_entry_matches_node(
            &local, "p", "m", None
        ));
        assert!(!ModelsState::model_entry_matches_node(
            &remote, "p", "m", None
        ));
    }

    #[test]
    fn popup_reset_cursor_and_filter_edits_preserve_header_inclusive_semantics() {
        let mut state = ModelsState::new();
        state.models = vec![
            model("local", "local", "p", "m", None, None),
            model("remote", "remote", "p", "m", Some("node"), None),
        ];
        state.apply_model_selection_from_entry(&state.models[1].clone());
        state.model_cursor = 99;
        state.model_filter = "stale".into();
        state.model_popup_agent_tab = 7;
        state.reset_for_open();
        assert_eq!(state.model_filter, "");
        assert_eq!(state.model_popup_agent_tab, 0);
        assert!(matches!(
            state.visible_model_popup_items()[state.model_cursor],
            ModelPopupItem::Model { model_idx: 1 }
        ));

        state.model_filter_insert('x');
        assert_eq!(state.model_cursor, 0);
        state.model_cursor = 8;
        state.model_filter_backspace();
        assert_eq!(state.model_cursor, 0);
        state.replace_filter("none".into());
        assert_eq!(state.model_filter, "none");
        assert_eq!(state.model_cursor, 0);
        state.move_model_cursor_down();
        assert_eq!(state.model_cursor, 0);
        state.move_model_cursor_up();
        assert_eq!(state.model_cursor, 0);
    }

    #[test]
    fn cursor_navigation_selects_headers_and_clamps_to_visible_items() {
        let mut state = ModelsState::new();
        state.models = vec![model("local", "local", "p", "m", None, None)];
        assert_eq!(state.visible_model_popup_items().len(), 2);
        state.move_model_cursor_up();
        assert_eq!(state.model_cursor, 0);
        state.move_model_cursor_down();
        assert_eq!(state.model_cursor, 1);
        state.move_model_cursor_down();
        assert_eq!(state.model_cursor, 1);
    }

    #[test]
    fn tabs_wrap_clear_filter_and_map_directly_to_agent_indices() {
        let mut state = ModelsState::new();
        assert_eq!(state.model_popup_tab_count(), 1);
        assert!(!state.model_popup_has_tabs());
        state.agents = vec![
            agent("main", "Main"),
            agent("coder", "Coder"),
            agent("reviewer", "Reviewer"),
        ];
        state.model_filter = "query".into();
        assert_eq!(
            state.switch_model_popup_tab(false).as_deref(),
            Some("coder")
        );
        assert_eq!(state.model_popup_agent_tab, 1);
        assert_eq!(state.model_filter, "");
        assert_eq!(state.model_popup_tab_label(0), "session");
        assert_eq!(state.model_popup_tab_label(1), "Coder");
        assert_eq!(state.model_popup_tab_agent_id(0), None);
        assert_eq!(state.model_popup_tab_agent_id(2), Some("reviewer"));
        assert_eq!(state.model_popup_tab_label(99), "???");
        assert_eq!(state.model_popup_tab_agent_id(99), None);

        state.model_popup_agent_tab = 0;
        assert_eq!(
            state.switch_model_popup_tab(true).as_deref(),
            Some("reviewer")
        );
        state.model_popup_agent_tab = 2;
        assert_eq!(state.switch_model_popup_tab(false), None);
        assert_eq!(state.model_popup_agent_tab, 0);
    }

    #[test]
    fn agent_initialization_replacement_clamping_and_clear_preserve_boundaries() {
        let mut state = ModelsState::new();
        state.model_popup_agent_tab = 4;
        state
            .delegate_model_preferences
            .insert("profile".into(), HashMap::new());
        state.initialize_primary_agent(agent("main", "Main"));
        assert_eq!(state.agents.len(), 1);
        assert_eq!(state.agents_profile_id, None);
        assert_eq!(state.model_popup_agent_tab, 4);
        assert!(state.delegate_model_preferences.contains_key("profile"));

        state.replace_profile_agents(
            "profile".into(),
            vec![agent("main", "Main"), agent("coder", "Coder")],
        );
        assert_eq!(state.agents_profile_id.as_deref(), Some("profile"));
        assert_eq!(state.model_popup_agent_tab, 1);
        state.clear_profile_agents();
        assert!(state.agents.is_empty());
        assert_eq!(state.agents_profile_id, None);
        assert_eq!(state.model_popup_agent_tab, 0);
        assert!(state.delegate_model_preferences.contains_key("profile"));
    }

    #[test]
    fn preferences_copy_remote_identity_and_prune_empty_profiles() {
        let mut state = ModelsState::new();
        let remote = model(
            "remote-id",
            "remote",
            "provider",
            "model",
            Some("node"),
            None,
        );
        state.set_delegate_model_preference("profile", "coder", &remote);
        let preference = state
            .get_delegate_model_preference("profile", "coder")
            .expect("stored preference");
        assert_eq!(preference.model_id, "remote-id");
        assert_eq!(preference.provider, "provider");
        assert_eq!(preference.model, "model");
        assert_eq!(preference.node_id.as_deref(), Some("node"));

        state.clear_delegate_model_preference("profile", "coder");
        assert!(!state.delegate_model_preferences.contains_key("profile"));
    }

    #[test]
    fn live_selection_replacement_stores_exact_provider_model_and_node() {
        let mut state = ModelsState::new();
        state.replace_live_selection("provider".into(), "model".into(), Some("node".into()));
        assert_eq!(state.current_provider.as_deref(), Some("provider"));
        assert_eq!(state.current_model.as_deref(), Some("model"));
        assert_eq!(state.current_model_node_id.as_deref(), Some("node"));
    }

    #[test]
    fn popup_open_cursor_falls_back_to_first_model_or_zero() {
        let mut state = ModelsState::new();
        assert_eq!(state.model_popup_open_cursor(), 0);
        state.models = vec![model("local", "local", "p", "m", None, None)];
        assert_eq!(state.model_popup_open_cursor(), 1);
    }

    #[test]
    fn tab_switch_without_delegates_preserves_popup_state() {
        let mut state = ModelsState::new();
        state.model_popup_agent_tab = 4;
        state.model_filter = "filter".into();
        assert_eq!(state.switch_model_popup_tab(false), None);
        assert_eq!(state.model_popup_agent_tab, 4);
        assert_eq!(state.model_filter, "filter");
    }

    #[test]
    fn empty_profile_agent_replacement_clamps_tab_and_retains_profile_identity() {
        let mut state = ModelsState::new();
        state.model_popup_agent_tab = 3;
        state.replace_profile_agents("profile".into(), Vec::new());
        assert!(state.agents.is_empty());
        assert_eq!(state.agents_profile_id.as_deref(), Some("profile"));
        assert_eq!(state.model_popup_agent_tab, 0);
    }

    #[test]
    fn clearing_missing_preference_does_not_remove_other_agents() {
        let mut state = ModelsState::new();
        let local = model("local", "local", "p", "m", None, None);
        state.set_delegate_model_preference("profile", "coder", &local);
        state.clear_delegate_model_preference("profile", "reviewer");
        assert!(
            state
                .get_delegate_model_preference("profile", "coder")
                .is_some()
        );
    }

    #[test]
    fn delegate_cursor_uses_profile_preference_and_exact_node() {
        let mut state = ModelsState::new();
        let local = model("same", "local", "p", "m", None, None);
        let remote = model("same", "remote", "p", "m", Some("node"), None);
        state.models = vec![local, remote.clone()];
        state.set_delegate_model_preference("profile", "coder", &remote);
        let cursor = state.delegate_model_cursor(Some("profile"), "coder");
        assert!(matches!(
            state.visible_model_popup_items()[cursor],
            ModelPopupItem::Model { model_idx: 1 }
        ));
        assert!(matches!(
            state.visible_model_popup_items()[state.delegate_model_cursor(None, "coder")],
            ModelPopupItem::Model { .. }
        ));
    }
}
