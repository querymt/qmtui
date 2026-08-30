use std::sync::Arc;

use agent_client_protocol::schema::v1 as acp;
use serde_json::Value;

use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};

use super::configuration;
use super::events::EventSink;
use super::runtime::RuntimeState;

#[derive(Debug)]
pub(super) enum Translation {
    Update(AcpSessionUpdate),
    AssistantChunk {
        text: String,
        message_id: Option<String>,
        thinking: bool,
    },
    ToolStart(AcpSessionUpdate),
    AgentMode(String),
    ConfigOptions(Vec<acp::SessionConfigOption>),
    Ignore,
}

pub(super) fn translate(notification: acp::SessionNotification) -> (String, Translation) {
    let session_id = notification.session_id.to_string();
    let translation = match notification.update {
        acp::SessionUpdate::UserMessageChunk(chunk) => {
            Translation::Update(AcpSessionUpdate::UserMessage {
                content: content_block_to_json(&chunk.content),
                message_id: chunk.message_id.map(|id| id.to_string()),
            })
        }
        acp::SessionUpdate::AgentMessageChunk(chunk) => Translation::AssistantChunk {
            text: content_block_text(&chunk.content),
            message_id: chunk.message_id.map(|id| id.to_string()),
            thinking: false,
        },
        acp::SessionUpdate::AgentThoughtChunk(chunk) => Translation::AssistantChunk {
            text: content_block_text(&chunk.content),
            message_id: chunk.message_id.map(|id| id.to_string()),
            thinking: true,
        },
        acp::SessionUpdate::ToolCall(tool_call) => {
            Translation::ToolStart(tool_start_update(&tool_call))
        }
        acp::SessionUpdate::ToolCallUpdate(update) => Translation::Update(tool_call_update(update)),
        acp::SessionUpdate::CurrentModeUpdate(update) => {
            Translation::AgentMode(update.current_mode_id.to_string())
        }
        acp::SessionUpdate::ConfigOptionUpdate(update) => {
            Translation::ConfigOptions(update.config_options)
        }
        acp::SessionUpdate::UsageUpdate(update) => Translation::Update(usage_update(update)),
        acp::SessionUpdate::SessionInfoUpdate(_)
        | acp::SessionUpdate::Plan(_)
        | acp::SessionUpdate::AvailableCommandsUpdate(_) => Translation::Ignore,
        _ => Translation::Ignore,
    };
    (session_id, translation)
}

pub(super) async fn apply(
    state: &Arc<RuntimeState>,
    events: &EventSink,
    session_id: String,
    translation: Translation,
) {
    match translation {
        Translation::Update(update) => emit_or_buffer(state, events, &session_id, update).await,
        Translation::ToolStart(update) => match state.replay.route(&session_id, update).await {
            None => {}
            Some(update) => {
                flush_assistant(state, events, &session_id).await;
                events.session_update(&session_id, update);
            }
        },
        Translation::AssistantChunk {
            text,
            message_id,
            thinking,
        } => {
            let replay_update = if thinking {
                AcpSessionUpdate::AssistantThinkingDelta {
                    content: text.clone(),
                    message_id: message_id.clone(),
                }
            } else {
                AcpSessionUpdate::AssistantMessage {
                    content: text.clone(),
                    thinking: None,
                    message_id: message_id.clone(),
                }
            };
            if state
                .replay
                .route(&session_id, replay_update)
                .await
                .is_none()
            {
                return;
            }
            if let Some(update) = state
                .assistants
                .flush_for_message(&session_id, message_id.as_deref())
                .await
            {
                events.session_update(&session_id, update);
            }
            state
                .assistants
                .remember(&session_id, message_id.clone(), &text, thinking)
                .await;
            events.session_update(
                &session_id,
                if thinking {
                    AcpSessionUpdate::AssistantThinkingDelta {
                        content: text,
                        message_id,
                    }
                } else {
                    AcpSessionUpdate::AssistantContentDelta {
                        content: text,
                        message_id,
                    }
                },
            );
        }
        Translation::AgentMode(mode) => events.send(AcpAppEvent::AgentMode { mode }),
        Translation::ConfigOptions(options) => {
            configuration::apply(state, events, options).await;
        }
        Translation::Ignore => {}
    }
}

async fn emit_or_buffer(
    state: &Arc<RuntimeState>,
    events: &EventSink,
    session_id: &str,
    update: AcpSessionUpdate,
) {
    if let Some(update) = state.replay.route(session_id, update).await {
        events.session_update(session_id, update);
    }
}

pub(super) async fn flush_assistant(
    state: &Arc<RuntimeState>,
    events: &EventSink,
    session_id: &str,
) {
    if let Some(update) = state.assistants.flush(session_id).await {
        events.session_update(session_id, update);
    }
}

fn tool_start_update(tool_call: &acp::ToolCall) -> AcpSessionUpdate {
    AcpSessionUpdate::ToolCallStart {
        tool_call_id: Some(tool_call.tool_call_id.to_string()),
        name: tool_name(&tool_call.title),
        arguments: tool_call.raw_input.clone(),
    }
}

fn usage_update(update: acp::UsageUpdate) -> AcpSessionUpdate {
    AcpSessionUpdate::UsageUpdate {
        used: update.used,
        size: update.size,
        cost_usd: update
            .cost
            .filter(|cost| cost.currency.eq_ignore_ascii_case("usd"))
            .map(|cost| cost.amount),
    }
}

fn tool_call_update(update: acp::ToolCallUpdate) -> AcpSessionUpdate {
    let status = update.fields.status;
    let name = update
        .fields
        .title
        .as_deref()
        .map(tool_name)
        .unwrap_or_else(|| "tool".to_string());
    if matches!(
        status,
        Some(acp::ToolCallStatus::Completed | acp::ToolCallStatus::Failed)
    ) {
        AcpSessionUpdate::ToolCallEnd {
            tool_call_id: Some(update.tool_call_id.to_string()),
            name,
            is_error: matches!(status, Some(acp::ToolCallStatus::Failed)),
            result: tool_result(&update.fields),
        }
    } else {
        AcpSessionUpdate::ToolCallStart {
            tool_call_id: Some(update.tool_call_id.to_string()),
            name,
            arguments: update.fields.raw_input,
        }
    }
}

fn tool_result(fields: &acp::ToolCallUpdateFields) -> Option<String> {
    if let Some(value) = fields.raw_output.as_ref() {
        return Some(value_to_text(value));
    }
    fields.content.as_ref().map(|content| {
        content
            .iter()
            .map(|entry| {
                serde_json::to_value(entry)
                    .map(|value| value_to_text(&value))
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn tool_name(title: &str) -> String {
    title.strip_prefix("Run ").unwrap_or(title).to_string()
}

fn content_block_to_json(block: &acp::ContentBlock) -> Value {
    serde_json::to_value(block).unwrap_or(Value::Null)
}

fn content_block_text(block: &acp::ContentBlock) -> String {
    match block {
        acp::ContentBlock::Text(text) => text.text.clone(),
        acp::ContentBlock::ResourceLink(link) => link.uri.clone(),
        other => serde_json::to_value(other)
            .map(|value| value_to_text(&value))
            .unwrap_or_default(),
    }
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notification(update: acp::SessionUpdate) -> acp::SessionNotification {
        acp::SessionNotification::new("session-1", update)
    }

    #[test]
    fn standard_updates_translate_without_runtime_dependencies() {
        let cases = [
            (
                notification(acp::SessionUpdate::UserMessageChunk(
                    acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new("user")))
                        .message_id(Some(acp::MessageId::from("u1"))),
                )),
                "user",
            ),
            (
                notification(acp::SessionUpdate::AgentMessageChunk(
                    acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(
                        "assistant",
                    )))
                    .message_id(Some(acp::MessageId::from("a1"))),
                )),
                "assistant",
            ),
            (
                notification(acp::SessionUpdate::AgentThoughtChunk(
                    acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(
                        "thought",
                    )))
                    .message_id(Some(acp::MessageId::from("a1"))),
                )),
                "thought",
            ),
        ];
        for (notification, expected) in cases {
            let (session_id, translated) = translate(notification);
            assert_eq!(session_id, "session-1");
            match (expected, translated) {
                ("user", Translation::Update(AcpSessionUpdate::UserMessage { message_id, .. })) => {
                    assert_eq!(message_id.as_deref(), Some("u1"));
                }
                (
                    "assistant",
                    Translation::AssistantChunk {
                        text,
                        thinking: false,
                        ..
                    },
                ) => {
                    assert_eq!(text, "assistant");
                }
                (
                    "thought",
                    Translation::AssistantChunk {
                        text,
                        thinking: true,
                        ..
                    },
                ) => {
                    assert_eq!(text, "thought");
                }
                (_, other) => panic!("unexpected translation: {other:?}"),
            }
        }
    }

    #[test]
    fn tool_start_and_pending_updates_preserve_names_and_arguments() {
        let (_, started) = translate(notification(acp::SessionUpdate::ToolCall(
            acp::ToolCall::new("tool-1", "Run shell")
                .raw_input(serde_json::json!({ "cmd": "cargo test" })),
        )));
        assert!(matches!(
            started,
            Translation::ToolStart(AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some(id),
                name,
                arguments: Some(arguments),
            }) if id == "tool-1"
                && name == "shell"
                && arguments == serde_json::json!({ "cmd": "cargo test" })
        ));

        let (_, pending) = translate(notification(acp::SessionUpdate::ToolCallUpdate(
            acp::ToolCallUpdate::new(
                "tool-2",
                acp::ToolCallUpdateFields::new().raw_input(serde_json::json!({ "path": "src" })),
            ),
        )));
        assert!(matches!(
            pending,
            Translation::Update(AcpSessionUpdate::ToolCallStart {
                tool_call_id: Some(id),
                name,
                arguments: Some(arguments),
            }) if id == "tool-2"
                && name == "tool"
                && arguments == serde_json::json!({ "path": "src" })
        ));
    }

    #[test]
    fn completed_and_failed_tool_updates_choose_raw_output_then_content() {
        let (_, completed) = translate(notification(acp::SessionUpdate::ToolCallUpdate(
            acp::ToolCallUpdate::new(
                "tool-1",
                acp::ToolCallUpdateFields::new()
                    .title("Run shell".to_string())
                    .status(acp::ToolCallStatus::Completed)
                    .raw_output(serde_json::json!({ "exit": 0 }))
                    .content(vec![
                        acp::ContentBlock::Text(acp::TextContent::new("fallback")).into(),
                    ]),
            ),
        )));
        assert!(matches!(
            completed,
            Translation::Update(AcpSessionUpdate::ToolCallEnd {
                tool_call_id: Some(id),
                name,
                is_error: false,
                result: Some(result),
            }) if id == "tool-1" && name == "shell" && result == r#"{"exit":0}"#
        ));

        let (_, failed) = translate(notification(acp::SessionUpdate::ToolCallUpdate(
            acp::ToolCallUpdate::new(
                "tool-2",
                acp::ToolCallUpdateFields::new()
                    .status(acp::ToolCallStatus::Failed)
                    .content(vec![
                        acp::ContentBlock::Text(acp::TextContent::new("failure")).into(),
                        acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                            "log",
                            "file:///tmp/error.log",
                        ))
                        .into(),
                    ]),
            ),
        )));
        assert!(matches!(
            failed,
            Translation::Update(AcpSessionUpdate::ToolCallEnd {
                tool_call_id: Some(id),
                name,
                is_error: true,
                result: Some(result),
            }) if id == "tool-2"
                && name == "tool"
                && result.contains("failure")
                && result.contains("file:///tmp/error.log")
        ));
    }

    #[test]
    fn assistant_content_converts_resource_links_and_non_text_blocks() {
        let (_, resource) = translate(notification(acp::SessionUpdate::AgentMessageChunk(
            acp::ContentChunk::new(acp::ContentBlock::ResourceLink(acp::ResourceLink::new(
                "guide",
                "file:///repo/guide.md",
            ))),
        )));
        assert!(matches!(
            resource,
            Translation::AssistantChunk { text, thinking: false, .. }
                if text == "file:///repo/guide.md"
        ));

        let (_, image) = translate(notification(acp::SessionUpdate::AgentThoughtChunk(
            acp::ContentChunk::new(acp::ContentBlock::Image(acp::ImageContent::new(
                "YWJj",
                "image/png",
            ))),
        )));
        assert!(matches!(
            image,
            Translation::AssistantChunk { text, thinking: true, .. }
                if text == r#"{"type":"image","data":"YWJj","mimeType":"image/png"}"#
        ));
    }

    #[test]
    fn mode_config_and_ignored_variants_stay_at_the_translation_boundary() {
        let option = acp::SessionConfigOption::select(
            "mode",
            "Mode",
            "plan",
            vec![acp::SessionConfigSelectOption::new("plan", "Plan")],
        );
        let (_, mode) = translate(notification(acp::SessionUpdate::CurrentModeUpdate(
            acp::CurrentModeUpdate::new("plan"),
        )));
        assert!(matches!(mode, Translation::AgentMode(value) if value == "plan"));

        let (_, config) = translate(notification(acp::SessionUpdate::ConfigOptionUpdate(
            acp::ConfigOptionUpdate::new(vec![option.clone()]),
        )));
        assert!(matches!(
            config,
            Translation::ConfigOptions(options) if options == vec![option]
        ));

        for update in [
            acp::SessionUpdate::SessionInfoUpdate(acp::SessionInfoUpdate::new().title("ignored")),
            acp::SessionUpdate::Plan(acp::Plan::new(vec![])),
            acp::SessionUpdate::AvailableCommandsUpdate(acp::AvailableCommandsUpdate::new(vec![])),
        ] {
            assert!(matches!(
                translate(notification(update)).1,
                Translation::Ignore
            ));
        }
    }

    #[test]
    fn usage_translation_keeps_only_usd_cost() {
        for (cost, expected) in [
            (Some(acp::Cost::new(0.25, "USD")), Some(0.25)),
            (Some(acp::Cost::new(1.5, "eur")), None),
            (None, None),
        ] {
            let update = if let Some(cost) = cost {
                acp::UsageUpdate::new(5, 10).cost(cost)
            } else {
                acp::UsageUpdate::new(5, 10)
            };
            let (_, translated) = translate(notification(acp::SessionUpdate::UsageUpdate(update)));
            assert!(matches!(
                translated,
                Translation::Update(AcpSessionUpdate::UsageUpdate {
                    used: 5,
                    size: 10,
                    cost_usd,
                }) if cost_usd == expected
            ));
        }
    }
}
