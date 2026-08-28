use crate::domain::tool::ToolDetail;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElicitationResponseOutcome {
    Selected(Vec<String>),
    Text(String),
    Boolean(bool),
    Declined,
    Cancelled,
    UnsupportedSchema,
    Responded,
}

#[derive(Debug, Clone)]
pub enum ChatEntry {
    User {
        text: String,
        message_id: Option<String>,
    },
    Assistant {
        content: String,
        thinking: Option<String>,
        message_id: Option<String>,
    },
    Thinking {
        content: String,
        message_id: Option<String>,
    },
    ToolCall {
        tool_call_id: Option<String>,
        name: String,
        is_error: bool,
        detail: ToolDetail,
    },
    CompactionStart {
        token_estimate: u32,
    },
    CompactionEnd {
        token_estimate: Option<u32>,
        summary: String,
        summary_len: u32,
    },
    Info(String),
    Error(String),
    Elicitation {
        elicitation_id: String,
        message: String,
        source: String,
        /// None is pending; Some records the semantic response state.
        outcome: Option<ElicitationResponseOutcome>,
    },
}

#[cfg(test)]
mod tests {
    use super::ElicitationResponseOutcome;

    #[test]
    fn elicitation_response_outcomes_preserve_semantic_payloads_and_states() {
        let selected = ElicitationResponseOutcome::Selected(vec!["Alpha".into(), "Beta".into()]);
        assert!(matches!(
            selected,
            ElicitationResponseOutcome::Selected(labels)
                if labels == ["Alpha".to_string(), "Beta".to_string()]
        ));

        let text = ElicitationResponseOutcome::Text("line one\nline two".into());
        assert!(matches!(
            text,
            ElicitationResponseOutcome::Text(value) if value == "line one\nline two"
        ));
        assert_ne!(
            ElicitationResponseOutcome::Boolean(true),
            ElicitationResponseOutcome::Boolean(false),
        );
        assert_ne!(
            ElicitationResponseOutcome::Declined,
            ElicitationResponseOutcome::Cancelled,
        );
        assert_ne!(
            ElicitationResponseOutcome::UnsupportedSchema,
            ElicitationResponseOutcome::Responded,
        );
    }
}
