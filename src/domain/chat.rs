use crate::domain::tool::ToolDetail;

pub const OUTCOME_BULLET: &str = "\u{25B8} ";

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
    #[allow(dead_code)] // Retained for compaction event rendering.
    CompactionStart {
        token_estimate: u32,
    },
    #[allow(dead_code)] // Retained for compaction event rendering.
    CompactionEnd {
        token_estimate: Option<u32>,
        summary: String,
        summary_len: u32,
    },
    #[allow(dead_code)] // Retained for informational replay entries.
    Info(String),
    Error(String),
    Elicitation {
        elicitation_id: String,
        message: String,
        #[allow(dead_code)] // Retained to preserve elicitation origin metadata.
        source: String,
        /// None = pending; Some = responded with this outcome label.
        outcome: Option<String>,
    },
}

pub fn format_outcome_label(label: &str) -> String {
    let mut formatted = String::with_capacity(OUTCOME_BULLET.len() + label.len());
    formatted.push_str(OUTCOME_BULLET);
    formatted.push_str(label);
    formatted
}

pub fn format_outcome_labels<'a>(labels: impl IntoIterator<Item = &'a str>) -> String {
    let mut labels = labels.into_iter();
    let Some(first) = labels.next() else {
        return String::new();
    };

    let mut formatted = format_outcome_label(first);
    for label in labels {
        formatted.push('\n');
        formatted.push_str(OUTCOME_BULLET);
        formatted.push_str(label);
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::{OUTCOME_BULLET, format_outcome_label, format_outcome_labels};

    #[test]
    fn format_outcome_label_prefixes_single_label() {
        assert_eq!(
            format_outcome_label("Beta"),
            format!("{OUTCOME_BULLET}Beta")
        );
    }

    #[test]
    fn format_outcome_labels_joins_multiple_labels_with_newlines() {
        assert_eq!(
            format_outcome_labels(["Alpha", "Beta", "Gamma"]),
            format!("{OUTCOME_BULLET}Alpha\n{OUTCOME_BULLET}Beta\n{OUTCOME_BULLET}Gamma"),
        );
    }

    #[test]
    fn format_outcome_labels_returns_empty_for_no_labels() {
        assert!(format_outcome_labels(std::iter::empty()).is_empty());
    }
}
