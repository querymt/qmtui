use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::domain::model::ModelEntry;
use crate::models_state::{ModelPopupItem, ModelsState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModelTab {
    Session,
    Delegate { agent_id: String, label: String },
}

#[derive(Debug)]
pub(crate) enum ModelInputResult {
    NotHandled,
    Close,
    Moved,
    SwitchedTab,
    Filtered,
    SelectModel {
        model: Box<ModelEntry>,
        tab: ModelTab,
    },
    ClearDelegatePreference {
        agent_id: String,
    },
}

pub(crate) fn handle_key(
    state: &mut ModelsState,
    key: KeyEvent,
    delegate_profile_id: Option<&str>,
) -> ModelInputResult {
    match key.code {
        KeyCode::Esc => ModelInputResult::Close,
        KeyCode::Tab | KeyCode::BackTab => {
            if !state.model_popup_has_tabs() {
                return ModelInputResult::NotHandled;
            }
            let agent_id = state.switch_model_popup_tab(key.code == KeyCode::BackTab);
            state.model_cursor = if state.model_popup_is_session_tab(state.model_popup_agent_tab) {
                state.model_popup_open_cursor()
            } else if let Some(agent_id) = agent_id {
                state.delegate_model_cursor(delegate_profile_id, &agent_id)
            } else {
                0
            };
            ModelInputResult::SwitchedTab
        }
        KeyCode::Up => {
            state.move_model_cursor_up();
            ModelInputResult::Moved
        }
        KeyCode::Down => {
            state.move_model_cursor_down();
            ModelInputResult::Moved
        }
        KeyCode::Enter => select_current(state),
        KeyCode::Delete if !state.model_popup_is_session_tab(state.model_popup_agent_tab) => state
            .model_popup_tab_agent_id(state.model_popup_agent_tab)
            .map(str::to_string)
            .map(|agent_id| ModelInputResult::ClearDelegatePreference { agent_id })
            .unwrap_or(ModelInputResult::NotHandled),
        KeyCode::Backspace => {
            state.model_filter_backspace();
            ModelInputResult::Filtered
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.model_filter_insert(character);
            ModelInputResult::Filtered
        }
        _ => ModelInputResult::NotHandled,
    }
}

fn select_current(state: &ModelsState) -> ModelInputResult {
    let model = state
        .visible_model_popup_items()
        .get(state.model_cursor)
        .and_then(|item| match item {
            ModelPopupItem::Model { model_idx } => state.models.get(*model_idx),
            ModelPopupItem::ProviderHeader { .. } => None,
        })
        .cloned();
    let Some(model) = model else {
        return ModelInputResult::NotHandled;
    };

    let tab = if state.model_popup_is_session_tab(state.model_popup_agent_tab) {
        ModelTab::Session
    } else {
        let Some(agent_id) = state
            .model_popup_tab_agent_id(state.model_popup_agent_tab)
            .map(str::to_string)
        else {
            return ModelInputResult::NotHandled;
        };
        ModelTab::Delegate {
            agent_id,
            label: state
                .model_popup_tab_label(state.model_popup_agent_tab)
                .to_string(),
        }
    };
    ModelInputResult::SelectModel {
        model: Box::new(model),
        tab,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::profile::AgentInfo;

    fn model(id: &str, provider: &str, node_id: Option<&str>) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            label: id.into(),
            provider: provider.into(),
            model: id.into(),
            node_id: node_id.map(str::to_string),
            node_label: None,
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

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn state() -> ModelsState {
        let mut state = ModelsState::new();
        state.models = vec![
            model("local", "alpha", None),
            model("remote", "beta", Some("node-1")),
        ];
        state.agents = vec![agent("primary", "Session"), agent("coder", "Coder")];
        state
    }

    #[test]
    fn movement_filtering_and_header_selection_return_exact_results() {
        let mut state = state();
        assert!(matches!(
            handle_key(&mut state, key(KeyCode::Down), None),
            ModelInputResult::Moved
        ));
        assert_eq!(state.model_cursor, 1);
        state.model_cursor = 0;
        assert!(matches!(
            handle_key(&mut state, key(KeyCode::Enter), None),
            ModelInputResult::NotHandled
        ));
        assert!(matches!(
            handle_key(&mut state, key(KeyCode::Char('r')), None),
            ModelInputResult::Filtered
        ));
        assert_eq!(state.model_filter, "r");
        assert_eq!(state.model_cursor, 0);
    }

    #[test]
    fn tabs_wrap_and_use_profile_scoped_delegate_cursor() {
        let mut state = state();
        let remote = state.models[1].clone();
        state.set_delegate_model_preference("profile", "coder", &remote);

        assert!(matches!(
            handle_key(&mut state, key(KeyCode::BackTab), Some("profile")),
            ModelInputResult::SwitchedTab
        ));
        assert_eq!(state.model_popup_agent_tab, 1);
        assert!(matches!(
            handle_key(&mut state, key(KeyCode::Enter), Some("profile")),
            ModelInputResult::SelectModel {
                model,
                tab: ModelTab::Delegate { agent_id, label },
            } if model.id == "remote" && model.node_id.as_deref() == Some("node-1")
                && agent_id == "coder" && label == "Coder"
        ));

        handle_key(&mut state, key(KeyCode::Tab), Some("profile"));
        assert_eq!(state.model_popup_agent_tab, 0);
    }

    #[test]
    fn session_selection_and_delegate_clear_return_owned_payloads() {
        let mut state = state();
        state.model_cursor = 1;
        assert!(matches!(
            handle_key(&mut state, key(KeyCode::Enter), None),
            ModelInputResult::SelectModel { model, tab: ModelTab::Session }
                if model.id == "local"
        ));

        state.model_popup_agent_tab = 1;
        assert!(matches!(
            handle_key(&mut state, key(KeyCode::Delete), None),
            ModelInputResult::ClearDelegatePreference { agent_id } if agent_id == "coder"
        ));
    }

    #[test]
    fn tab_without_delegate_tabs_is_a_noop() {
        let mut state = ModelsState::new();
        state.model_filter = "keep".into();
        assert!(matches!(
            handle_key(&mut state, key(KeyCode::Tab), None),
            ModelInputResult::NotHandled
        ));
        assert_eq!(state.model_filter, "keep");
    }
}
