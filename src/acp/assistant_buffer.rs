use std::collections::HashMap;

use tokio::sync::Mutex;

use crate::acp_state::AcpSessionUpdate;

#[derive(Debug, Default)]
struct AssistantBuffer {
    content: String,
    thinking: String,
    content_message_id: Option<String>,
    thinking_message_id: Option<String>,
}

impl AssistantBuffer {
    fn message_id(&self) -> Option<String> {
        self.content_message_id
            .clone()
            .or_else(|| self.thinking_message_id.clone())
    }

    fn has_different_message(&self, incoming_message_id: Option<&str>) -> bool {
        let Some(incoming) = incoming_message_id else {
            return false;
        };
        self.message_id()
            .as_deref()
            .is_some_and(|current| current != incoming)
    }

    fn into_update(self) -> Option<AcpSessionUpdate> {
        if self.content.is_empty() && self.thinking.is_empty() {
            return None;
        }
        let message_id = self.message_id();
        Some(AcpSessionUpdate::AssistantMessage {
            content: self.content,
            thinking: (!self.thinking.is_empty()).then_some(self.thinking),
            message_id,
        })
    }
}

#[derive(Default)]
pub(super) struct AssistantBuffers {
    buffers: Mutex<HashMap<String, AssistantBuffer>>,
}

impl AssistantBuffers {
    pub(super) async fn flush_for_message(
        &self,
        session_id: &str,
        incoming_message_id: Option<&str>,
    ) -> Option<AcpSessionUpdate> {
        let mut buffers = self.buffers.lock().await;
        if buffers
            .get(session_id)
            .is_some_and(|buffer| buffer.has_different_message(incoming_message_id))
        {
            return buffers
                .remove(session_id)
                .and_then(AssistantBuffer::into_update);
        }
        None
    }

    pub(super) async fn remember(
        &self,
        session_id: &str,
        message_id: Option<String>,
        text: &str,
        thinking: bool,
    ) {
        if text.is_empty() {
            return;
        }
        let mut buffers = self.buffers.lock().await;
        let buffer = buffers.entry(session_id.to_string()).or_default();
        if thinking {
            if buffer.thinking_message_id.is_none() && message_id.is_some() {
                buffer.thinking_message_id = message_id;
            }
            buffer.thinking.push_str(text);
        } else {
            if buffer.content_message_id.is_none() && message_id.is_some() {
                buffer.content_message_id = message_id;
            }
            buffer.content.push_str(text);
        }
    }

    pub(super) async fn flush(&self, session_id: &str) -> Option<AcpSessionUpdate> {
        self.buffers
            .lock()
            .await
            .remove(session_id)
            .and_then(AssistantBuffer::into_update)
    }

    pub(super) async fn clear(&self) {
        self.buffers.lock().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(update: Option<AcpSessionUpdate>) -> (String, Option<String>, Option<String>) {
        match update.expect("assistant update") {
            AcpSessionUpdate::AssistantMessage {
                content,
                thinking,
                message_id,
            } => (content, thinking, message_id),
            other => panic!("unexpected update: {other:?}"),
        }
    }

    #[tokio::test]
    async fn chunks_merge_and_message_change_flushes_the_previous_boundary() {
        let buffers = AssistantBuffers::default();
        buffers
            .remember("session-1", Some("a1".into()), "think", true)
            .await;
        buffers
            .remember("session-1", Some("a1".into()), "answer", false)
            .await;
        let flushed = buffers.flush_for_message("session-1", Some("a2")).await;
        assert_eq!(
            assistant(flushed),
            ("answer".into(), Some("think".into()), Some("a1".into()))
        );
    }

    #[tokio::test]
    async fn tool_finish_cancel_and_elicitation_boundaries_use_the_same_flush() {
        let buffers = AssistantBuffers::default();
        for boundary in ["tool", "finish", "cancel", "elicitation"] {
            buffers
                .remember("session-1", Some(boundary.into()), boundary, false)
                .await;
            let (content, _, message_id) = assistant(buffers.flush("session-1").await);
            assert_eq!(content, boundary);
            assert_eq!(message_id.as_deref(), Some(boundary));
        }
    }

    #[tokio::test]
    async fn buffers_are_isolated_per_session() {
        let buffers = AssistantBuffers::default();
        buffers
            .remember("one", Some("a1".into()), "first", false)
            .await;
        buffers
            .remember("two", Some("a2".into()), "second", false)
            .await;
        assert_eq!(assistant(buffers.flush("one").await).0, "first");
        assert_eq!(assistant(buffers.flush("two").await).0, "second");
    }
}
