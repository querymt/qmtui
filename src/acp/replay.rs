use std::collections::HashMap;

use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};
use crate::domain::activity::DelegationUpdate;
use crate::protocol::delegation::DelegationUpdateDto;

use super::extensions::delegation::from_wire;

const SESSION_LOAD_SNAPSHOT_META_KEY: &str = "querymt/sessionLoadSnapshot.v1";

#[derive(Debug, Default)]
struct SessionLoad {
    updates: Vec<AcpSessionUpdate>,
    completing: bool,
}

#[derive(Debug, Default)]
pub(super) struct ReplayBuffer {
    sessions: Mutex<HashMap<String, SessionLoad>>,
}

impl ReplayBuffer {
    pub(super) async fn begin(&self, session_id: &str) {
        self.sessions
            .lock()
            .await
            .insert(session_id.to_string(), SessionLoad::default());
    }

    pub(super) async fn route(
        &self,
        session_id: &str,
        update: AcpSessionUpdate,
    ) -> Option<AcpSessionUpdate> {
        let mut sessions = self.sessions.lock().await;
        let Some(load) = sessions.get_mut(session_id) else {
            return Some(update);
        };
        load.updates.push(update);
        None
    }

    pub(super) async fn start_completion(&self, session_id: &str) -> Vec<AcpSessionUpdate> {
        let mut sessions = self.sessions.lock().await;
        let Some(load) = sessions.get_mut(session_id) else {
            return Vec::new();
        };
        load.completing = true;
        std::mem::take(&mut load.updates)
    }

    pub(super) async fn drain_completion(&self, session_id: &str) -> Option<Vec<AcpSessionUpdate>> {
        let mut sessions = self.sessions.lock().await;
        let load = sessions.get_mut(session_id)?;
        debug_assert!(load.completing);
        if load.updates.is_empty() {
            sessions.remove(session_id);
            None
        } else {
            Some(std::mem::take(&mut load.updates))
        }
    }

    pub(super) async fn abort(&self, session_id: &str) {
        self.sessions.lock().await.remove(session_id);
    }

    #[cfg(test)]
    async fn contains(&self, session_id: &str) -> bool {
        self.sessions.lock().await.contains_key(session_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SnapshotProviderChange {
    pub(super) provider: String,
    pub(super) model: String,
    pub(super) context_limit: Option<u64>,
    pub(super) provider_node_id: Option<String>,
}

fn session_load_snapshot(response: &Value) -> Option<&Value> {
    response
        .get("_meta")
        .or_else(|| response.get("meta"))
        .and_then(|meta| meta.get(SESSION_LOAD_SNAPSHOT_META_KEY))
}

fn session_load_audit(response: &Value) -> Value {
    session_load_snapshot(response)
        .and_then(|snapshot| snapshot.get("audit"))
        .cloned()
        .unwrap_or_else(|| json!({ "events": [] }))
}

pub(super) fn delegation_updates(response: &Value) -> Vec<DelegationUpdate> {
    session_load_snapshot(response)
        .and_then(|snapshot| snapshot.get("delegationUpdates"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|update| serde_json::from_value::<DelegationUpdateDto>(update.clone()).ok())
        .filter_map(from_wire)
        .collect()
}

pub(super) fn provider_change(response: &Value) -> Option<SnapshotProviderChange> {
    let audit = session_load_audit(response);
    audit
        .get("events")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(provider_change_from_event)
        .next_back()
}

fn provider_change_from_event(event: &Value) -> Option<SnapshotProviderChange> {
    let kind = event.get("kind")?;
    if kind.get("type").and_then(Value::as_str) != Some("provider_changed") {
        return None;
    }
    let data = kind.get("data")?;
    Some(SnapshotProviderChange {
        provider: string_field(data, "provider")?,
        model: string_field(data, "model")?,
        context_limit: u64_field(data, "context_limit"),
        provider_node_id: string_field(data, "provider_node_id"),
    })
}

pub(super) fn provider_change_event(change: SnapshotProviderChange) -> AcpAppEvent {
    AcpAppEvent::ProviderChanged {
        provider: change.provider,
        model: change.model,
        context_limit: change.context_limit,
        provider_node_id: change.provider_node_id,
    }
}

pub(super) fn merge_snapshot_stats(
    replay_updates: Vec<AcpSessionUpdate>,
    snapshot_updates: Vec<AcpSessionUpdate>,
) -> Vec<AcpSessionUpdate> {
    if replay_updates.is_empty() {
        return snapshot_updates;
    }

    let replay_has_usage = replay_updates
        .iter()
        .any(|update| matches!(update, AcpSessionUpdate::UsageUpdate { .. }));
    let replay_has_timing = replay_updates
        .iter()
        .any(|update| matches!(update, AcpSessionUpdate::TimingUpdate { .. }));
    let mut merged = replay_updates;
    merged.extend(snapshot_updates.into_iter().filter(|update| match update {
        AcpSessionUpdate::UsageUpdate { .. } => !replay_has_usage,
        AcpSessionUpdate::TimingUpdate { .. } => !replay_has_timing,
        _ => false,
    }));
    merged
}

pub(super) fn snapshot_updates(response: &Value) -> Vec<AcpSessionUpdate> {
    let audit = session_load_audit(response);
    let Some(events) = audit.get("events").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut updates = Vec::new();
    let mut context_limit = 0;
    let mut llm_started_at: Option<i64> = None;
    for event in events {
        let Some(kind) = event.get("kind") else {
            continue;
        };
        let Some(kind_type) = kind.get("type").and_then(Value::as_str) else {
            continue;
        };
        let data = kind.get("data").unwrap_or(&Value::Null);
        let timestamp = i64_field(event, "timestamp");
        match kind_type {
            "llm_request_start" => llm_started_at = timestamp,
            "provider_changed" => {
                if let Some(limit) = u64_field(data, "context_limit") {
                    context_limit = limit;
                    if limit > 0 {
                        updates.push(AcpSessionUpdate::UsageUpdate {
                            used: 0,
                            size: limit,
                            cost_usd: None,
                        });
                    }
                }
            }
            "llm_request_end" => {
                push_timing(&mut updates, llm_started_at.take(), timestamp);
                updates.push(AcpSessionUpdate::UsageUpdate {
                    used: u64_field(data, "context_tokens").unwrap_or(0),
                    size: context_limit,
                    cost_usd: f64_field(data, "cumulative_cost_usd"),
                });
                updates.push(AcpSessionUpdate::Finished {
                    finish_reason: string(data, "finish_reason")
                        .unwrap_or_else(|| "completed".to_string()),
                });
            }
            "cancelled" | "error" => {
                push_timing(&mut updates, llm_started_at.take(), timestamp);
                if let Some(update) = snapshot_event_to_update(kind_type, data) {
                    updates.push(update);
                }
            }
            _ => {
                if let Some(update) = snapshot_event_to_update(kind_type, data) {
                    updates.push(update);
                }
            }
        }
    }
    updates
}

fn push_timing(updates: &mut Vec<AcpSessionUpdate>, started: Option<i64>, ended: Option<i64>) {
    if let (Some(started), Some(ended)) = (started, ended)
        && ended >= started
    {
        updates.push(AcpSessionUpdate::TimingUpdate {
            duration_secs: (ended - started) as u64,
        });
    }
}

fn snapshot_event_to_update(kind: &str, data: &Value) -> Option<AcpSessionUpdate> {
    match kind {
        "prompt_received" => Some(AcpSessionUpdate::UserMessage {
            content: data.get("content").cloned().unwrap_or(Value::Null),
            message_id: string(data, "message_id"),
        }),
        "assistant_content_delta" => Some(AcpSessionUpdate::AssistantContentDelta {
            content: string(data, "content").unwrap_or_default(),
            message_id: string(data, "message_id"),
        }),
        "assistant_thinking_delta" => Some(AcpSessionUpdate::AssistantThinkingDelta {
            content: string(data, "content").unwrap_or_default(),
            message_id: string(data, "message_id"),
        }),
        "assistant_message_stored" => Some(AcpSessionUpdate::AssistantMessage {
            content: string(data, "content").unwrap_or_default(),
            thinking: string(data, "thinking").filter(|text| !text.is_empty()),
            message_id: string(data, "message_id"),
        }),
        "tool_call_start" => Some(AcpSessionUpdate::ToolCallStart {
            tool_call_id: string(data, "tool_call_id"),
            name: string(data, "tool_name").unwrap_or_else(|| "tool".to_string()),
            arguments: data.get("arguments").or_else(|| data.get("input")).cloned(),
        }),
        "tool_call_end" => Some(AcpSessionUpdate::ToolCallEnd {
            tool_call_id: string(data, "tool_call_id"),
            name: string(data, "tool_name").unwrap_or_else(|| "tool".to_string()),
            is_error: data
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            result: string(data, "result")
                .or_else(|| string(data, "output"))
                .or_else(|| string(data, "content")),
        }),
        "cancelled" => Some(AcpSessionUpdate::Cancelled),
        _ => None,
    }
}

fn i64_field(data: &Value, key: &str) -> Option<i64> {
    match data.get(key)? {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn u64_field(data: &Value, key: &str) -> Option<u64> {
    match data.get(key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn f64_field(data: &Value, key: &str) -> Option<f64> {
    match data.get(key)? {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn string(data: &Value, key: &str) -> Option<String> {
    match data.get(key)? {
        Value::String(text) => Some(text.clone()),
        Value::Null => None,
        other => serde_json::to_string(other).ok(),
    }
}

fn string_field(data: &Value, key: &str) -> Option<String> {
    match data.get(key)? {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn user(id: &str) -> AcpSessionUpdate {
        AcpSessionUpdate::UserMessage {
            content: json!(id),
            message_id: Some(id.to_string()),
        }
    }

    #[tokio::test]
    async fn buffers_are_per_session_and_preserve_arrival_order() {
        let buffer = ReplayBuffer::default();
        buffer.begin("one").await;
        buffer.begin("two").await;
        assert!(buffer.route("one", user("a")).await.is_none());
        assert!(buffer.route("two", user("x")).await.is_none());
        assert!(buffer.route("one", user("b")).await.is_none());

        let one = buffer.start_completion("one").await;
        let two = buffer.start_completion("two").await;
        assert!(
            matches!(&one[..], [AcpSessionUpdate::UserMessage { message_id: Some(a), .. }, AcpSessionUpdate::UserMessage { message_id: Some(b), .. }] if a == "a" && b == "b")
        );
        assert!(
            matches!(&two[..], [AcpSessionUpdate::UserMessage { message_id: Some(x), .. }] if x == "x")
        );
    }

    #[tokio::test]
    async fn completion_race_neither_loses_nor_strands_an_update() {
        let buffer = Arc::new(ReplayBuffer::default());
        buffer.begin("session").await;
        assert!(buffer.route("session", user("before")).await.is_none());
        let initial = buffer.start_completion("session").await;

        let racing = buffer.clone();
        let task = tokio::spawn(async move { racing.route("session", user("racing")).await });
        assert!(task.await.expect("task").is_none());
        let tail = buffer
            .drain_completion("session")
            .await
            .expect("completion tail");
        assert_eq!(initial.len() + tail.len(), 2);
        assert!(buffer.drain_completion("session").await.is_none());
        assert!(!buffer.contains("session").await);
        assert!(buffer.route("session", user("live")).await.is_some());
    }

    #[test]
    fn native_replay_owns_history_and_snapshot_only_fills_missing_stats() {
        let replay = vec![user("native")];
        let snapshot = vec![
            user("snapshot"),
            AcpSessionUpdate::TimingUpdate { duration_secs: 12 },
            AcpSessionUpdate::UsageUpdate {
                used: 3,
                size: 10,
                cost_usd: Some(0.1),
            },
        ];
        let merged = merge_snapshot_stats(replay, snapshot);
        assert_eq!(merged.len(), 3);
        assert!(
            matches!(&merged[0], AcpSessionUpdate::UserMessage { message_id: Some(id), .. } if id == "native")
        );
        assert!(matches!(
            &merged[1],
            AcpSessionUpdate::TimingUpdate { duration_secs: 12 }
        ));
        assert!(matches!(
            &merged[2],
            AcpSessionUpdate::UsageUpdate { used: 3, .. }
        ));
    }

    #[test]
    fn missing_snapshot_returns_no_updates_or_provider() {
        assert!(snapshot_updates(&json!({})).is_empty());
        assert!(provider_change(&json!({})).is_none());
        assert!(delegation_updates(&json!({})).is_empty());
    }

    #[test]
    fn native_usage_and_timing_are_not_duplicated() {
        let replay = vec![
            AcpSessionUpdate::TimingUpdate { duration_secs: 1 },
            AcpSessionUpdate::UsageUpdate {
                used: 1,
                size: 10,
                cost_usd: None,
            },
        ];
        let merged = merge_snapshot_stats(
            replay,
            vec![
                AcpSessionUpdate::TimingUpdate { duration_secs: 2 },
                AcpSessionUpdate::UsageUpdate {
                    used: 2,
                    size: 10,
                    cost_usd: Some(1.0),
                },
            ],
        );
        assert_eq!(merged.len(), 2);
        assert!(matches!(
            &merged[0],
            AcpSessionUpdate::TimingUpdate { duration_secs: 1 }
        ));
        assert!(matches!(
            &merged[1],
            AcpSessionUpdate::UsageUpdate { used: 1, .. }
        ));
    }

    #[test]
    fn provider_change_uses_last_valid_snapshot_event() {
        let change = provider_change(&json!({
            "_meta": {
                SESSION_LOAD_SNAPSHOT_META_KEY: {
                    "audit": { "events": [
                        { "kind": { "type": "provider_changed", "data": {
                            "provider": "first", "model": "one"
                        } } },
                        { "kind": { "type": "provider_changed", "data": {
                            "provider": "second", "model": "two",
                            "context_limit": "100", "provider_node_id": "node"
                        } } }
                    ] }
                }
            }
        }))
        .expect("provider change");
        assert_eq!(change.provider, "second");
        assert_eq!(change.model, "two");
        assert_eq!(change.context_limit, Some(100));
        assert_eq!(change.provider_node_id.as_deref(), Some("node"));
    }

    #[test]
    fn snapshot_decoding_restores_stats_without_wall_clock_age() {
        let updates = snapshot_updates(&json!({
            "_meta": {
                SESSION_LOAD_SNAPSHOT_META_KEY: {
                    "audit": { "events": [
                        { "timestamp": 10, "kind": { "type": "session_created" } },
                        { "timestamp": 20, "kind": { "type": "provider_changed", "data": { "context_limit": 100 } } },
                        { "timestamp": 30, "kind": { "type": "llm_request_start" } },
                        { "timestamp": 42, "kind": { "type": "llm_request_end", "data": { "context_tokens": 25, "cumulative_cost_usd": 0.2 } } }
                    ] }
                }
            }
        }));
        assert!(
            updates.iter().any(|update| matches!(
                update,
                AcpSessionUpdate::TimingUpdate { duration_secs: 12 }
            ))
        );
        assert!(updates.iter().any(|update| matches!(
            update,
            AcpSessionUpdate::UsageUpdate {
                used: 25,
                size: 100,
                ..
            }
        )));
        assert_eq!(
            updates
                .iter()
                .filter(|update| matches!(update, AcpSessionUpdate::TimingUpdate { .. }))
                .count(),
            1
        );
    }
}
