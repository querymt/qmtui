use agent_client_protocol::schema::v1 as acp;
use serde_json::{Value, json};

use crate::acp_state::AcpAppEvent;
use crate::domain::profile::ProfileInfo;

use super::events::EventSink;
use super::extensions::models::Model;
use super::runtime::RuntimeState;

pub(super) async fn apply(
    state: &RuntimeState,
    events: &EventSink,
    config_options: Vec<acp::SessionConfigOption>,
) {
    let options_json = serde_json::to_value(&config_options).unwrap_or(Value::Null);
    let Some(options) = options_json.as_array() else {
        return;
    };
    let mut profiles = Vec::new();
    let mut active_profile_id = None;
    for option in options {
        let id = option.get("id").and_then(Value::as_str).unwrap_or_default();
        let category = option
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let name = option
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let current = option.get("currentValue").and_then(Value::as_str);
        let normalized = normalize_key(id, category, name);
        if normalized == "model" {
            if let Some(model_id) = current {
                state.select_model(model_id).await;
                if let Some(model) = state
                    .model_by_id(model_id)
                    .await
                    .or_else(|| model_from_option(option, model_id))
                {
                    send_provider_changed(events, &model);
                }
            }
        } else if normalized == "mode" {
            if let Some(mode) = current {
                events.send(AcpAppEvent::AgentMode {
                    mode: mode.to_string(),
                });
            }
        } else if matches!(
            normalized.as_str(),
            "thought_level" | "reasoning" | "reasoning_effort" | "thought"
        ) {
            if let Some(effort) = current {
                events.send(AcpAppEvent::ReasoningEffort {
                    reasoning_effort: Some(effort.to_string()),
                });
            }
        } else if normalized == "profile" {
            active_profile_id = current.map(str::to_string);
            profiles = profiles_from_option(option);
        }
    }
    if !profiles.is_empty() {
        events.send(AcpAppEvent::Profiles {
            profiles,
            active_profile_id,
        });
    }
}

pub(super) fn profile_id(config_options: &[acp::SessionConfigOption]) -> Option<String> {
    let options_json = serde_json::to_value(config_options).ok()?;
    for option in options_json.as_array()? {
        let id = option.get("id").and_then(Value::as_str).unwrap_or_default();
        let category = option
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let name = option
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if normalize_key(id, category, name) == "profile" {
            return option
                .get("currentValue")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
    }
    None
}

pub(super) fn model_meta(
    model: &Model,
    node_id_override: Option<&str>,
) -> Option<serde_json::Map<String, Value>> {
    let mut entry = serde_json::to_value(model).ok()?;
    if let Some(node_id) = node_id_override
        && let Some(object) = entry.as_object_mut()
    {
        object.insert("node_id".to_string(), Value::String(node_id.to_string()));
    }
    let mut meta = serde_json::Map::new();
    meta.insert("querymt".to_string(), json!({ "modelEntry": entry }));
    Some(meta)
}

pub(super) fn fallback_model(model_id: &str) -> Model {
    let (provider, model) = model_id
        .split_once('/')
        .map(|(provider, model)| (provider.to_string(), model.to_string()))
        .unwrap_or_else(|| ("unknown".to_string(), model_id.to_string()));
    Model {
        id: model_id.to_string(),
        label: model.clone(),
        source: Some("qmtui-fallback".to_string()),
        provider,
        model,
        node_id: None,
        node_label: None,
        family: None,
        quant: None,
    }
}

pub(super) fn send_provider_changed(events: &EventSink, model: &Model) {
    events.send(AcpAppEvent::ProviderChanged {
        provider: model.provider.clone(),
        model: model.model.clone(),
        context_limit: None,
        provider_node_id: model.node_id.clone(),
    });
}

fn normalize_key(id: &str, category: &str, name: &str) -> String {
    for value in [id, category, name] {
        let normalized = value
            .to_ascii_lowercase()
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .trim_matches('_')
            .to_string();
        if !normalized.is_empty() {
            return normalized;
        }
    }
    String::new()
}

fn profiles_from_option(option: &Value) -> Vec<ProfileInfo> {
    flatten_select_entries(option)
        .into_iter()
        .filter_map(|entry| {
            let id = entry.get("value")?.as_str()?;
            let name = entry.get("name").and_then(Value::as_str).unwrap_or(id);
            Some(ProfileInfo {
                id: id.to_string(),
                name: name.to_string(),
                description: entry
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                tags: Vec::new(),
                source: None,
                config_kind: None,
                fingerprint: None,
            })
        })
        .collect()
}

fn model_from_option(option: &Value, model_id: &str) -> Option<Model> {
    flatten_select_entries(option)
        .into_iter()
        .find(|entry| entry.get("value").and_then(Value::as_str) == Some(model_id))
        .map(|entry| {
            let label = entry
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(model_id)
                .to_string();
            let (provider, model) = model_id
                .split_once('/')
                .map(|(provider, model)| (provider.to_string(), model.to_string()))
                .unwrap_or_else(|| ("unknown".to_string(), label.clone()));
            Model {
                id: model_id.to_string(),
                label,
                source: None,
                provider,
                model,
                node_id: None,
                node_label: None,
                family: None,
                quant: None,
            }
        })
}

fn flatten_select_entries(option: &Value) -> Vec<&Value> {
    option
        .get("options")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|entry| {
            if let Some(group_options) = entry.get("options").and_then(Value::as_array) {
                group_options.iter().collect::<Vec<_>>()
            } else {
                vec![entry]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_metadata_preserves_full_entry_and_override_node() {
        let model = fallback_model("openrouter/openai/gpt-5");
        let meta = model_meta(&model, Some("node-1")).expect("meta");
        let entry = &meta["querymt"]["modelEntry"];
        assert_eq!(entry["provider"], "openrouter");
        assert_eq!(entry["model"], "openai/gpt-5");
        assert_eq!(entry["node_id"], "node-1");
    }

    #[test]
    fn option_parser_reads_grouped_models_and_profiles() {
        let option = json!({
            "options": [{ "name": "group", "options": [
                { "value": "openai/gpt-5", "name": "gpt-5" }
            ] }]
        });
        assert_eq!(
            model_from_option(&option, "openai/gpt-5")
                .expect("model")
                .provider,
            "openai"
        );
        assert_eq!(profiles_from_option(&option)[0].id, "openai/gpt-5");
    }
}
