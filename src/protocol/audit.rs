use serde::Deserialize;

// Audit DTOs intentionally deserialize the full backend event contract.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum EventKind {
    TurnStarted,
    PromptReceived {
        content: serde_json::Value,
        message_id: Option<String>,
    },
    UserMessageStored {
        content: serde_json::Value,
    },
    AssistantMessageStored {
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    },
    AssistantContentDelta {
        content: String,
        message_id: Option<String>,
    },
    AssistantThinkingDelta {
        content: String,
        message_id: Option<String>,
    },
    CompactionStart {
        token_estimate: u32,
    },
    CompactionEnd {
        summary: String,
        summary_len: u32,
    },
    LlmRequestStart {
        message_count: Option<u32>,
    },
    LlmRequestEnd {
        finish_reason: Option<String>,
        cost_usd: Option<f64>,
        cumulative_cost_usd: Option<f64>,
        context_tokens: Option<u64>,
        tool_calls: Option<u32>,
        metrics: Option<serde_json::Value>,
    },
    ToolCallStart {
        tool_call_id: Option<String>,
        tool_name: String,
        arguments: Option<serde_json::Value>,
    },
    ToolCallEnd {
        tool_call_id: Option<String>,
        tool_name: String,
        is_error: Option<bool>,
        result: Option<String>,
    },
    SnapshotStart {
        policy: String,
    },
    SnapshotEnd {
        summary: Option<String>,
    },
    ProgressRecorded {
        progress_entry: ProgressEntry,
    },
    ArtifactRecorded {
        artifact: ArtifactInfo,
    },
    SessionQueued {
        reason: String,
    },
    SessionConfigured {
        cwd: Option<String>,
        #[serde(default)]
        mcp_servers: Vec<serde_json::Value>,
        limits: Option<SessionLimits>,
    },
    ToolsAvailable {
        #[serde(default)]
        tools: Vec<ToolInfo>,
        #[serde(default)]
        tools_hash: Option<serde_json::Value>,
    },
    ProviderChanged {
        provider: String,
        model: String,
        config_id: Option<i64>,
        context_limit: Option<u64>,
        /// Mesh node hosting the provider when the session routes LLM calls remotely.
        #[serde(default)]
        provider_node_id: Option<String>,
    },
    ElicitationRequested {
        elicitation_id: String,
        session_id: String,
        message: String,
        requested_schema: serde_json::Value,
        source: String,
    },
    /// Emitted when a session's mode changes (per-session mode in actor model).
    /// Durable — appears in the audit journal and replayed on session load.
    /// The last occurrence in a session's audit gives the session's last-used mode.
    SessionModeChanged {
        mode: String,
    },
    Error {
        message: String,
    },
    Cancelled,
    SessionCreated,
    DelegationRequested {
        delegation: DelegationData,
    },
    DelegationCompleted {
        delegation_id: String,
        #[serde(default)]
        result: Option<String>,
    },
    DelegationFailed {
        delegation_id: String,
        #[serde(default)]
        error: Option<String>,
    },
    DelegationCancelled {
        delegation_id: String,
    },
    SessionForked {
        #[serde(default)]
        child_session_id: Option<String>,
        #[serde(default)]
        origin: Option<String>,
        /// Delegation public_id when origin="delegation".
        #[serde(default)]
        fork_point_ref: Option<String>,
        /// The agent the child session was delegated to.
        #[serde(default)]
        target_agent_id: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct ProgressEntry {
    pub kind: ProgressKind,
    pub content: String,
    pub metadata: Option<String>,
    pub created_at: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct ArtifactInfo {
    pub kind: String,
    pub uri: Option<String>,
    pub path: Option<String>,
    pub summary: Option<String>,
    pub created_at: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProgressKind {
    ToolCall,
    Artifact,
    Note,
    Checkpoint,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct SessionLimits {
    pub max_steps: Option<u32>,
    pub max_turns: Option<u32>,
    pub max_cost_usd: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct ToolInfo {
    #[serde(rename = "type", default)]
    pub tool_type: String,
    #[serde(default)]
    pub function: Option<FunctionToolInfo>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct FunctionToolInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// Subset of the server-side `Delegation` struct that we care about.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct DelegationData {
    pub public_id: String,
    #[serde(default)]
    pub target_agent_id: Option<String>,
    #[serde(default)]
    pub objective: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{EventKind, ProgressKind};
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Deserialize)]
    struct TestAgentEvent {
        kind: EventKind,
        timestamp: Option<i64>,
    }

    #[test]
    fn backend_next_protocol_events_deserialize() {
        let session_queued = json!({
            "kind": {
                "type": "session_queued",
                "data": {
                    "reason": "waiting for previous operation to complete",
                    "unknown_field": "ignored"
                }
            },
            "timestamp": 123,
            "unknown_envelope_field": true
        });
        let session_configured = json!({
            "kind": {
                "type": "session_configured",
                "data": {
                    "cwd": "/workspace/project",
                    "mcp_servers": [],
                    "limits": {
                        "max_steps": 200,
                        "max_turns": 50,
                        "max_cost_usd": null
                    }
                }
            },
            "timestamp": null
        });
        let tools_available = json!({
            "kind": {
                "type": "tools_available",
                "data": {
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "search_text",
                            "description": "Search file contents",
                            "parameters": { "type": "object" }
                        }
                    }],
                    "tools_hash": "123456789"
                }
            },
            "timestamp": null
        });
        let artifact_recorded = json!({
            "kind": {
                "type": "artifact_recorded",
                "data": {
                    "artifact": {
                        "kind": "file",
                        "uri": null,
                        "path": "src/generated.txt",
                        "summary": "Produced by write_file",
                        "created_at": "2026-04-29T14:25:09Z"
                    }
                }
            },
            "timestamp": null
        });

        let queued: TestAgentEvent = serde_json::from_value(session_queued).unwrap();
        assert_eq!(queued.timestamp, Some(123));
        assert!(
            matches!(queued.kind, EventKind::SessionQueued { reason } if reason == "waiting for previous operation to complete")
        );

        let configured: TestAgentEvent = serde_json::from_value(session_configured).unwrap();
        assert!(
            matches!(configured.kind, EventKind::SessionConfigured { cwd, mcp_servers, limits } if cwd.as_deref() == Some("/workspace/project") && mcp_servers.is_empty() && limits.as_ref().and_then(|l| l.max_steps) == Some(200))
        );

        let available: TestAgentEvent = serde_json::from_value(tools_available).unwrap();
        assert!(
            matches!(available.kind, EventKind::ToolsAvailable { tools, tools_hash } if tools.first().and_then(|tool| tool.function.as_ref()).map(|function| function.name.as_str()) == Some("search_text") && tools_hash.is_some())
        );

        let artifact: TestAgentEvent = serde_json::from_value(artifact_recorded).unwrap();
        assert!(
            matches!(artifact.kind, EventKind::ArtifactRecorded { artifact } if artifact.kind == "file" && artifact.path.as_deref() == Some("src/generated.txt") && artifact.summary.as_deref() == Some("Produced by write_file"))
        );
    }

    #[test]
    fn backend_snapshot_and_progress_events_deserialize() {
        let snapshot_start = json!({
            "kind": {
                "type": "snapshot_start",
                "data": { "policy": "diff" }
            },
            "timestamp": null
        });
        let snapshot_end = json!({
            "kind": {
                "type": "snapshot_end",
                "data": { "summary": "1 modified" }
            },
            "timestamp": null
        });
        let progress_recorded = json!({
            "kind": {
                "type": "progress_recorded",
                "data": {
                    "progress_entry": {
                        "kind": "tool_call",
                        "content": "Calling tool: shell",
                        "metadata": "{\"tool\":\"shell\"}",
                        "created_at": "2026-04-13T00:00:00Z"
                    }
                }
            },
            "timestamp": null
        });

        let start: TestAgentEvent = serde_json::from_value(snapshot_start).unwrap();
        assert!(matches!(start.kind, EventKind::SnapshotStart { policy } if policy == "diff"));

        let end: TestAgentEvent = serde_json::from_value(snapshot_end).unwrap();
        assert!(
            matches!(end.kind, EventKind::SnapshotEnd { summary } if summary.as_deref() == Some("1 modified"))
        );

        let progress: TestAgentEvent = serde_json::from_value(progress_recorded).unwrap();
        assert!(
            matches!(progress.kind, EventKind::ProgressRecorded { progress_entry } if progress_entry.kind == ProgressKind::ToolCall && progress_entry.content == "Calling tool: shell")
        );
    }

    #[test]
    fn unknown_event_type_falls_back_to_unknown() {
        let event: TestAgentEvent = serde_json::from_value(json!({
            "kind": { "type": "future_event" },
            "timestamp": null
        }))
        .unwrap();

        assert!(matches!(event.kind, EventKind::Unknown));
    }

    #[test]
    fn current_optional_fields_keep_their_defaults() {
        let configured: EventKind = serde_json::from_value(json!({
            "type": "session_configured",
            "data": { "cwd": null, "limits": null }
        }))
        .unwrap();
        assert!(
            matches!(configured, EventKind::SessionConfigured { mcp_servers, .. } if mcp_servers.is_empty())
        );

        let available: EventKind = serde_json::from_value(json!({
            "type": "tools_available",
            "data": {}
        }))
        .unwrap();
        assert!(
            matches!(available, EventKind::ToolsAvailable { tools, tools_hash } if tools.is_empty() && tools_hash.is_none())
        );

        let provider: EventKind = serde_json::from_value(json!({
            "type": "provider_changed",
            "data": {
                "provider": "openai",
                "model": "gpt",
                "config_id": null,
                "context_limit": null
            }
        }))
        .unwrap();
        assert!(matches!(
            provider,
            EventKind::ProviderChanged {
                provider_node_id: None,
                ..
            }
        ));
    }
}
