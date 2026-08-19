use std::time::{Duration, Instant};

/// Per-delegation stats accumulated from child session events.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DelegateStats {
    pub tool_calls: u32,
    pub messages: u32,
    /// Cumulative cost in USD from LLM request completions.
    pub cost_usd: f64,
    /// Latest context token count.
    pub context_tokens: u64,
    /// Context limit from the active provider.
    pub context_limit: u64,
}

impl DelegateStats {
    pub fn context_pct(&self) -> Option<u32> {
        if self.context_limit > 0 {
            Some(
                ((self.context_tokens as f64 / self.context_limit as f64) * 100.0)
                    .round()
                    .min(100.0) as u32,
            )
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DelegateChildState {
    #[default]
    None,
    PendingElicitation {
        elicitation_id: String,
        message: String,
        requested_schema: serde_json::Value,
        source: String,
    },
    QuestionToolFinished,
    AssistantMessage,
    UserMessage,
    OtherProgress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegationState {
    Requested,
    Forked,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelegationUpdate {
    pub session_id: String,
    pub delegation_id: String,
    pub tool_call_id: Option<String>,
    pub state: DelegationState,
    pub target_agent_id: String,
    pub objective: String,
    pub child_session_id: Option<String>,
    pub requested_at: i64,
    pub forked_at: Option<i64>,
    pub finished_at: Option<i64>,
    pub updated_at: i64,
    pub result_summary: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DelegateEntry {
    pub delegation_id: String,
    pub child_session_id: Option<String>,
    /// Parent delegate tool call this row renders for, when known.
    pub delegate_tool_call_id: Option<String>,
    pub target_agent_id: Option<String>,
    pub objective: String,
    pub status: DelegateStatus,
    pub stats: DelegateStats,
    /// Server timestamp (unix seconds) when delegation was requested.
    pub started_at: Option<i64>,
    /// Server timestamp (unix seconds) when delegation completed/failed/cancelled.
    pub ended_at: Option<i64>,
    /// Compact state derived from the latest significant child-session event.
    pub child_state: DelegateChildState,
}

impl DelegateEntry {
    pub fn awaiting_input(&self) -> bool {
        self.status == DelegateStatus::InProgress
            && matches!(
                self.child_state,
                DelegateChildState::PendingElicitation { .. }
            )
    }

    pub fn pending_elicitation(&self) -> Option<(&str, &str, &str)> {
        match &self.child_state {
            DelegateChildState::PendingElicitation {
                elicitation_id,
                message,
                source,
                ..
            } => Some((elicitation_id.as_str(), message.as_str(), source.as_str())),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegateStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingDelegateToolCall {
    pub tool_call_id: String,
    pub target_agent_id: Option<String>,
    pub objective: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SessionStatsLite {
    pub active_llm_duration: Duration,
    pub open_llm_request_ts: Option<i64>,
    pub open_llm_request_instant: Option<Instant>,
    pub latest_context_tokens: Option<u64>,
    pub total_tool_calls: u32,
}

#[derive(Debug, Clone)]
pub struct SessionActivity {
    pub last_event_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionOp {
    Undo,
    Redo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivityState {
    Idle,
    Thinking,
    Streaming,
    RunningTool { name: String },
    Compacting { token_estimate: u32 },
    SessionOp(SessionOp),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_percentage_rounds_and_caps_at_one_hundred() {
        let stats = DelegateStats {
            context_tokens: 50_000,
            context_limit: 200_000,
            ..DelegateStats::default()
        };
        assert_eq!(stats.context_pct(), Some(25));

        let over_limit = DelegateStats {
            context_tokens: 201,
            context_limit: 200,
            ..DelegateStats::default()
        };
        assert_eq!(over_limit.context_pct(), Some(100));
        assert_eq!(DelegateStats::default().context_pct(), None);
    }

    #[test]
    fn delegate_entry_reports_only_active_pending_elicitations() {
        let mut entry = DelegateEntry {
            delegation_id: "d1".into(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: None,
            objective: "Answer a question".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::PendingElicitation {
                elicitation_id: "e1".into(),
                message: "Choose".into(),
                requested_schema: serde_json::Value::Null,
                source: "builtin:question".into(),
            },
        };

        assert!(entry.awaiting_input());
        assert_eq!(
            entry.pending_elicitation(),
            Some(("e1", "Choose", "builtin:question"))
        );
        entry.status = DelegateStatus::Completed;
        assert!(!entry.awaiting_input());
    }
}
