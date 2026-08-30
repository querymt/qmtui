use std::collections::{HashMap, HashSet};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use serde_json::Value;

use crate::application::Effect;
use crate::domain::activity::{
    DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus, DelegationState,
    DelegationUpdate, PendingDelegateToolCall,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DelegateChildActivity {
    ToolCallStarted,
    AssistantMessage {
        message_id: Option<String>,
    },
    AssistantContent {
        message_id: Option<String>,
    },
    Progress,
    UserMessage,
    Usage {
        used: u64,
        size: u64,
        cost_usd: Option<f64>,
    },
    PendingElicitation {
        elicitation_id: String,
        message: String,
        requested_schema: Value,
        source: String,
    },
    QuestionToolFinished,
    Unchanged,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DelegateAction {
    LifecycleUpdate(DelegationUpdate),
    ProvisionalToolDelegate {
        tool_call_id: Option<String>,
        arguments: Option<Value>,
    },
    InactiveChildActivity {
        session_id: String,
        activity: DelegateChildActivity,
    },
    ResolveLoadedSessionParent(Option<String>),
    ClearNewRootParent,
    ClearRootSessionState,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DelegateContext {
    pub(crate) active_session_id: Option<String>,
    pub(crate) inactive_activity_session_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DelegateOutcome {
    pub(crate) changed: bool,
    pub(crate) presentation_changed: bool,
    pub(crate) effects: Vec<Effect>,
}

struct DelegateLifecycleUpdate {
    pub(crate) delegation_id: String,
    pub(crate) tool_call_id: Option<String>,
    pub(crate) target_agent_id: String,
    pub(crate) objective: String,
    pub(crate) child_session_id: Option<String>,
    pub(crate) status: DelegateStatus,
    pub(crate) lifecycle_rank: u8,
    pub(crate) requested_at: i64,
    pub(crate) finished_at: Option<i64>,
    pub(crate) updated_at: i64,
    pub(crate) result_summary: Option<String>,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DelegatesState {
    pub(crate) delegate_entries: Vec<DelegateEntry>,
    pub(crate) delegate_cursor: usize,
    pub(crate) delegate_filter: String,
    pub(crate) parent_session_id: Option<String>,
    pub(crate) pending_parent_session_id: Option<String>,
    pub(crate) suppress_delegation_result: bool,
    pub(crate) pending_delegate_child_states: HashMap<String, DelegateChildState>,
    pub(crate) pending_delegate_child_stats: HashMap<String, DelegateStats>,
    pub(crate) delegate_child_message_ids: HashMap<String, HashSet<String>>,
    pub(crate) delegation_update_times: HashMap<String, i64>,
    pub(crate) delegation_result_summaries: HashMap<String, String>,
    pub(crate) delegation_errors: HashMap<String, String>,
    pub(crate) pending_delegate_tool_calls: Vec<PendingDelegateToolCall>,
}

impl DelegatesState {
    pub(crate) fn reduce(
        &mut self,
        action: DelegateAction,
        context: DelegateContext,
    ) -> DelegateOutcome {
        match action {
            DelegateAction::LifecycleUpdate(update) => {
                if context.active_session_id.as_deref() != Some(update.session_id.as_str()) {
                    return DelegateOutcome::default();
                }
                let status = match update.state {
                    DelegationState::Requested | DelegationState::Forked => {
                        DelegateStatus::InProgress
                    }
                    DelegationState::Completed => DelegateStatus::Completed,
                    DelegationState::Failed => DelegateStatus::Failed,
                    DelegationState::Cancelled => DelegateStatus::Cancelled,
                };
                let lifecycle_rank = delegation_state_rank(update.state);
                let before = self.clone();
                let accepted = self.apply_lifecycle_update(DelegateLifecycleUpdate {
                    delegation_id: update.delegation_id,
                    tool_call_id: update.tool_call_id,
                    target_agent_id: update.target_agent_id,
                    objective: update.objective,
                    child_session_id: update.child_session_id,
                    status,
                    lifecycle_rank,
                    requested_at: update.requested_at,
                    finished_at: update.finished_at,
                    updated_at: update.updated_at,
                    result_summary: update.result_summary,
                    error: update.error,
                });
                Self::presentation_outcome(accepted && *self != before)
            }
            DelegateAction::ProvisionalToolDelegate {
                tool_call_id,
                arguments,
            } => {
                let Some(tool_call_id) = tool_call_id else {
                    return DelegateOutcome::default();
                };
                let target_agent_id = arguments
                    .as_ref()
                    .and_then(|value| value.get("target_agent_id"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let objective = arguments
                    .as_ref()
                    .and_then(|value| value.get("objective"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let changed =
                    self.upsert_provisional_delegate(&tool_call_id, target_agent_id, objective);
                Self::presentation_outcome(changed)
            }
            DelegateAction::InactiveChildActivity {
                session_id,
                activity,
            } => {
                if context.inactive_activity_session_id.as_deref() != Some(session_id.as_str()) {
                    return DelegateOutcome::default();
                }
                let changed = self.apply_child_activity(&session_id, activity);
                Self::presentation_outcome(changed)
            }
            DelegateAction::ResolveLoadedSessionParent(discovered_parent) => {
                let parent_before = self.parent_session_id.clone();
                let pending_before = self.pending_parent_session_id.clone();
                self.resolve_parent_session_id(discovered_parent);
                DelegateOutcome {
                    changed: self.parent_session_id != parent_before
                        || self.pending_parent_session_id != pending_before,
                    ..DelegateOutcome::default()
                }
            }
            DelegateAction::ClearNewRootParent => {
                let changed = self.parent_session_id.take().is_some()
                    | self.pending_parent_session_id.take().is_some();
                DelegateOutcome {
                    changed,
                    ..DelegateOutcome::default()
                }
            }
            DelegateAction::ClearRootSessionState => {
                if self.parent_session_id.is_some() {
                    return DelegateOutcome::default();
                }
                let before = self.clone();
                self.clear_for_root_session();
                DelegateOutcome {
                    changed: *self != before,
                    ..DelegateOutcome::default()
                }
            }
        }
    }

    fn presentation_outcome(changed: bool) -> DelegateOutcome {
        DelegateOutcome {
            changed,
            presentation_changed: changed,
            effects: Vec::new(),
        }
    }

    fn apply_child_activity(&mut self, session_id: &str, activity: DelegateChildActivity) -> bool {
        let (mut state, mut stats) = self.child_snapshot(session_id);
        match activity {
            DelegateChildActivity::ToolCallStarted => {
                stats.tool_calls = stats.tool_calls.saturating_add(1);
                state = DelegateChildState::OtherProgress;
            }
            DelegateChildActivity::AssistantMessage { message_id } => {
                if self.record_child_message_id(session_id, message_id.as_deref(), true) {
                    stats.messages = stats.messages.saturating_add(1);
                }
                state = DelegateChildState::AssistantMessage;
            }
            DelegateChildActivity::AssistantContent { message_id } => {
                if self.record_child_message_id(session_id, message_id.as_deref(), false) {
                    stats.messages = stats.messages.saturating_add(1);
                }
                state = DelegateChildState::AssistantMessage;
            }
            DelegateChildActivity::Progress => state = DelegateChildState::OtherProgress,
            DelegateChildActivity::UserMessage => state = DelegateChildState::UserMessage,
            DelegateChildActivity::Usage {
                used,
                size,
                cost_usd,
            } => {
                if used > 0 {
                    stats.context_tokens = used;
                }
                if size > 0 {
                    stats.context_limit = size;
                }
                if let Some(cost) = cost_usd {
                    stats.cost_usd = cost;
                }
            }
            DelegateChildActivity::PendingElicitation {
                elicitation_id,
                message,
                requested_schema,
                source,
            } => {
                state = DelegateChildState::PendingElicitation {
                    elicitation_id,
                    message,
                    requested_schema,
                    source,
                };
            }
            DelegateChildActivity::QuestionToolFinished => {
                state = DelegateChildState::QuestionToolFinished;
            }
            DelegateChildActivity::Unchanged => {}
        }
        self.apply_child_snapshot(session_id, state, stats)
    }

    pub(crate) fn new() -> Self {
        Self {
            delegate_entries: Vec::new(),
            delegate_cursor: 0,
            delegate_filter: String::new(),
            parent_session_id: None,
            pending_parent_session_id: None,
            suppress_delegation_result: false,
            pending_delegate_child_states: HashMap::new(),
            pending_delegate_child_stats: HashMap::new(),
            delegate_child_message_ids: HashMap::new(),
            delegation_update_times: HashMap::new(),
            delegation_result_summaries: HashMap::new(),
            delegation_errors: HashMap::new(),
            pending_delegate_tool_calls: Vec::new(),
        }
    }

    pub(crate) fn visible_entries(&self) -> Vec<&DelegateEntry> {
        if self.delegate_filter.is_empty() {
            return self.delegate_entries.iter().collect();
        }
        let matcher = SkimMatcherV2::default();
        let query = self.delegate_filter.to_lowercase();
        let mut scored: Vec<(i64, &DelegateEntry)> = self
            .delegate_entries
            .iter()
            .filter_map(|entry| {
                [
                    matcher.fuzzy_match(&entry.objective, &query),
                    matcher.fuzzy_match(&entry.delegation_id, &query),
                    matcher.fuzzy_match(entry.target_agent_id.as_deref().unwrap_or(""), &query),
                ]
                .into_iter()
                .flatten()
                .max()
                .map(|score| (score, entry))
            })
            .collect();
        scored.sort_by_key(|item| std::cmp::Reverse(item.0));
        scored.into_iter().map(|(_, entry)| entry).collect()
    }

    pub(crate) fn selected_entry(&self) -> Option<&DelegateEntry> {
        self.visible_entries().get(self.delegate_cursor).copied()
    }

    pub(crate) fn reset_popup(&mut self) {
        self.delegate_cursor = 0;
        self.delegate_filter.clear();
    }

    pub(crate) fn move_cursor_up(&mut self) {
        self.delegate_cursor = self.delegate_cursor.saturating_sub(1);
    }

    pub(crate) fn move_cursor_down(&mut self) {
        let max = self.visible_entries().len().saturating_sub(1);
        self.delegate_cursor = self.delegate_cursor.saturating_add(1).min(max);
    }

    pub(crate) fn move_cursor_page(&mut self, step: usize, down: bool) {
        if down {
            let max = self.visible_entries().len().saturating_sub(1);
            self.delegate_cursor = self.delegate_cursor.saturating_add(step).min(max);
        } else {
            self.delegate_cursor = self.delegate_cursor.saturating_sub(step);
        }
    }

    pub(crate) fn filter_insert(&mut self, character: char) {
        self.delegate_filter.push(character);
        self.delegate_cursor = 0;
    }

    pub(crate) fn filter_backspace(&mut self) {
        self.delegate_filter.pop();
        self.delegate_cursor = 0;
    }

    pub(crate) fn upsert_provisional_delegate(
        &mut self,
        tool_call_id: &str,
        target_agent_id: Option<String>,
        objective: String,
    ) -> bool {
        if self
            .delegate_entries
            .iter()
            .any(|entry| entry.delegate_tool_call_id.as_deref() == Some(tool_call_id))
        {
            return false;
        }
        self.delegate_entries.push(DelegateEntry {
            delegation_id: format!("tool:{tool_call_id}"),
            child_session_id: None,
            delegate_tool_call_id: Some(tool_call_id.to_string()),
            target_agent_id: target_agent_id.clone(),
            objective: objective.clone(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        self.pending_delegate_tool_calls
            .push(PendingDelegateToolCall {
                tool_call_id: tool_call_id.to_string(),
                target_agent_id,
                objective,
            });
        true
    }

    fn apply_lifecycle_update(&mut self, update: DelegateLifecycleUpdate) -> bool {
        let DelegateLifecycleUpdate {
            delegation_id,
            mut tool_call_id,
            target_agent_id,
            objective,
            child_session_id,
            status,
            lifecycle_rank: incoming_lifecycle_rank,
            requested_at,
            finished_at,
            updated_at,
            result_summary,
            error,
        } = update;
        let existing_index = self
            .delegate_entries
            .iter()
            .position(|entry| entry.delegation_id == delegation_id);
        if let Some(existing_timestamp) = self.delegation_update_times.get(&delegation_id) {
            let existing_rank = existing_index
                .map(|index| entry_lifecycle_rank(&self.delegate_entries[index]))
                .unwrap_or(0);
            if *existing_timestamp > updated_at
                || (*existing_timestamp == updated_at && existing_rank > incoming_lifecycle_rank)
            {
                return false;
            }
        }

        if tool_call_id.is_none() {
            tool_call_id = self
                .take_pending_tool_call(Some(&target_agent_id), Some(&objective))
                .map(|pending| pending.tool_call_id);
        } else if let Some(id) = tool_call_id.as_deref() {
            self.pending_delegate_tool_calls
                .retain(|pending| pending.tool_call_id != id);
        }

        let index = existing_index.or_else(|| {
            tool_call_id.as_deref().and_then(|id| {
                self.delegate_entries
                    .iter()
                    .position(|entry| entry.delegate_tool_call_id.as_deref() == Some(id))
            })
        });
        let index = if let Some(index) = index {
            index
        } else {
            self.delegate_entries.push(DelegateEntry {
                delegation_id: delegation_id.clone(),
                child_session_id: None,
                delegate_tool_call_id: tool_call_id.clone(),
                target_agent_id: Some(target_agent_id.clone()),
                objective: objective.clone(),
                status,
                stats: DelegateStats::default(),
                started_at: Some(requested_at),
                ended_at: None,
                child_state: DelegateChildState::None,
            });
            self.delegate_entries.len() - 1
        };

        let entry = &mut self.delegate_entries[index];
        entry.delegation_id = delegation_id.clone();
        if tool_call_id.is_some() {
            entry.delegate_tool_call_id = tool_call_id;
        }
        entry.target_agent_id = Some(target_agent_id);
        entry.objective = objective;
        entry.status = status;
        entry.started_at = Some(requested_at);
        entry.ended_at = finished_at;
        entry.child_session_id = child_session_id.clone();
        if status != DelegateStatus::InProgress {
            entry.child_state = DelegateChildState::None;
        }

        self.delegation_update_times
            .insert(delegation_id.clone(), updated_at);
        replace_optional_map_value(
            &mut self.delegation_result_summaries,
            &delegation_id,
            result_summary,
        );
        replace_optional_map_value(&mut self.delegation_errors, &delegation_id, error);

        if let Some(child_session_id) = child_session_id {
            if let Some(stats) = self.pending_delegate_child_stats.remove(&child_session_id) {
                self.delegate_entries[index].stats = stats;
            }
            if let Some(state) = self.pending_delegate_child_states.remove(&child_session_id) {
                self.delegate_entries[index].child_state = state;
            }
        }
        true
    }

    pub(crate) fn child_snapshot(&self, session_id: &str) -> (DelegateChildState, DelegateStats) {
        let index = self
            .delegate_entries
            .iter()
            .position(|entry| entry.child_session_id.as_deref() == Some(session_id));
        let state = index
            .map(|index| self.delegate_entries[index].child_state.clone())
            .or_else(|| self.pending_delegate_child_states.get(session_id).cloned())
            .unwrap_or_default();
        let stats = index
            .map(|index| self.delegate_entries[index].stats.clone())
            .or_else(|| self.pending_delegate_child_stats.get(session_id).cloned())
            .unwrap_or_default();
        (state, stats)
    }

    pub(crate) fn record_child_message_id(
        &mut self,
        session_id: &str,
        message_id: Option<&str>,
        increment_without_id: bool,
    ) -> bool {
        match message_id {
            Some(message_id) => self
                .delegate_child_message_ids
                .entry(session_id.to_string())
                .or_default()
                .insert(message_id.to_string()),
            None => increment_without_id,
        }
    }

    pub(crate) fn apply_child_snapshot(
        &mut self,
        session_id: &str,
        state: DelegateChildState,
        stats: DelegateStats,
    ) -> bool {
        if let Some(index) = self
            .delegate_entries
            .iter()
            .position(|entry| entry.child_session_id.as_deref() == Some(session_id))
        {
            if self.delegate_entries[index].stats != stats
                || self.delegate_entries[index].child_state != state
            {
                self.delegate_entries[index].stats = stats;
                self.delegate_entries[index].child_state = state;
                return true;
            }
        } else if state != DelegateChildState::None || stats != DelegateStats::default() {
            self.pending_delegate_child_states
                .insert(session_id.to_string(), state);
            self.pending_delegate_child_stats
                .insert(session_id.to_string(), stats);
        }
        false
    }

    pub(crate) fn stage_parent_for_child_navigation(
        &mut self,
        current_parent_session_id: Option<String>,
        current_session_id: Option<String>,
    ) {
        self.pending_parent_session_id = current_parent_session_id.or(current_session_id);
    }

    pub(crate) fn resolve_parent_session_id(&mut self, discovered_parent: Option<String>) {
        self.parent_session_id = self.pending_parent_session_id.take().or(discovered_parent);
    }

    pub(crate) fn clear_for_root_session(&mut self) {
        self.delegate_entries.clear();
        self.pending_delegate_child_states.clear();
        self.pending_delegate_child_stats.clear();
        self.delegate_child_message_ids.clear();
        self.delegation_update_times.clear();
        self.delegation_result_summaries.clear();
        self.delegation_errors.clear();
        self.pending_delegate_tool_calls.clear();
    }

    fn take_pending_tool_call(
        &mut self,
        target_agent_id: Option<&str>,
        objective: Option<&str>,
    ) -> Option<PendingDelegateToolCall> {
        let index = self
            .pending_delegate_tool_calls
            .iter()
            .position(|pending| {
                target_agent_id
                    .is_none_or(|agent| pending.target_agent_id.as_deref() == Some(agent))
                    && objective.is_none_or(|value| pending.objective == value)
            })?;
        Some(self.pending_delegate_tool_calls.remove(index))
    }
}

fn replace_optional_map_value(
    values: &mut HashMap<String, String>,
    key: &str,
    value: Option<String>,
) {
    if let Some(value) = value {
        values.insert(key.to_string(), value);
    } else {
        values.remove(key);
    }
}

fn lifecycle_rank(status: DelegateStatus, child_session_id: Option<&str>) -> u8 {
    match status {
        DelegateStatus::Completed | DelegateStatus::Failed | DelegateStatus::Cancelled => 3,
        DelegateStatus::InProgress if child_session_id.is_some() => 2,
        DelegateStatus::InProgress => 1,
    }
}

fn entry_lifecycle_rank(entry: &DelegateEntry) -> u8 {
    lifecycle_rank(entry.status, entry.child_session_id.as_deref())
}

fn delegation_state_rank(state: DelegationState) -> u8 {
    match state {
        DelegationState::Requested => 1,
        DelegationState::Forked => 2,
        DelegationState::Completed | DelegationState::Failed | DelegationState::Cancelled => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        delegation_id: &str,
        objective: &str,
        agent: Option<&str>,
        status: DelegateStatus,
    ) -> DelegateEntry {
        DelegateEntry {
            delegation_id: delegation_id.to_string(),
            child_session_id: None,
            delegate_tool_call_id: None,
            target_agent_id: agent.map(str::to_string),
            objective: objective.to_string(),
            status,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        }
    }

    fn apply(
        state: &mut DelegatesState,
        delegation_id: &str,
        status: DelegateStatus,
        updated_at: i64,
        child_session_id: Option<&str>,
        terminal: (Option<&str>, Option<&str>),
    ) -> bool {
        state.apply_lifecycle_update(DelegateLifecycleUpdate {
            delegation_id: delegation_id.to_string(),
            tool_call_id: None,
            target_agent_id: "coder".to_string(),
            objective: "build feature".to_string(),
            child_session_id: child_session_id.map(str::to_string),
            status,
            lifecycle_rank: lifecycle_rank(status, child_session_id),
            requested_at: 10,
            finished_at: (status != DelegateStatus::InProgress).then_some(20),
            updated_at,
            result_summary: terminal.0.map(str::to_string),
            error: terminal.1.map(str::to_string),
        })
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = DelegatesState::new();

        assert!(state.delegate_entries.is_empty());
        assert_eq!(state.delegate_cursor, 0);
        assert!(state.delegate_filter.is_empty());
        assert_eq!(state.parent_session_id, None);
        assert_eq!(state.pending_parent_session_id, None);
        assert!(!state.suppress_delegation_result);
        assert!(state.pending_delegate_child_states.is_empty());
        assert!(state.pending_delegate_child_stats.is_empty());
        assert!(state.delegate_child_message_ids.is_empty());
        assert!(state.delegation_update_times.is_empty());
        assert!(state.delegation_result_summaries.is_empty());
        assert!(state.delegation_errors.is_empty());
        assert!(state.pending_delegate_tool_calls.is_empty());
    }

    #[test]
    fn filtering_is_case_insensitive_across_objective_id_and_agent() {
        let mut state = DelegatesState::new();
        assert!(state.visible_entries().is_empty());

        state.delegate_entries = vec![
            entry(
                "DEL-ONE",
                "Build Feature",
                Some("Coder"),
                DelegateStatus::InProgress,
            ),
            entry(
                "del-two",
                "Write docs",
                Some("Planner"),
                DelegateStatus::Completed,
            ),
        ];

        assert_eq!(state.visible_entries().len(), 2);
        for query in ["BUILD", "del-one", "CODER"] {
            state.delegate_filter = query.to_string();
            assert_eq!(state.visible_entries().len(), 1);
            assert_eq!(state.visible_entries()[0].delegation_id, "DEL-ONE");
        }

        state.delegate_filter = "planner".into();
        assert_eq!(state.visible_entries()[0].delegation_id, "del-two");
    }

    #[test]
    fn stats_context_percentage_handles_limits() {
        let stats = DelegateStats {
            context_tokens: 50_000,
            context_limit: 200_000,
            ..DelegateStats::default()
        };
        assert_eq!(stats.context_pct(), Some(25));

        let no_limit = DelegateStats {
            context_tokens: 1000,
            context_limit: 0,
            ..DelegateStats::default()
        };
        assert_eq!(no_limit.context_pct(), None);
    }

    #[test]
    fn selection_cursor_page_filter_and_popup_reset_preserve_filtered_semantics() {
        let mut state = DelegatesState::new();
        state.delegate_entries = (0..7)
            .map(|index| {
                entry(
                    &format!("d{index}"),
                    if index % 2 == 0 { "docs" } else { "code" },
                    None,
                    DelegateStatus::InProgress,
                )
            })
            .collect();

        state.move_cursor_down();
        state.move_cursor_page(3, true);
        assert_eq!(state.delegate_cursor, 4);
        state.move_cursor_page(3, false);
        assert_eq!(state.delegate_cursor, 1);
        state.filter_insert('d');
        state.filter_insert('o');
        assert_eq!(state.delegate_cursor, 0);
        state.move_cursor_down();
        assert_eq!(state.selected_entry().unwrap().objective, "docs");
        state.filter_backspace();
        assert_eq!(state.delegate_cursor, 0);
        state.reset_popup();
        assert!(state.delegate_filter.is_empty());
        assert_eq!(state.delegate_cursor, 0);
    }

    #[test]
    fn provisional_entries_deduplicate_and_reconcile_by_tool_id() {
        let mut state = DelegatesState::new();
        assert!(state.upsert_provisional_delegate(
            "tool-1",
            Some("coder".to_string()),
            "build feature".to_string(),
        ));
        assert!(!state.upsert_provisional_delegate("tool-1", None, String::new()));
        assert_eq!(state.delegate_entries.len(), 1);
        assert_eq!(state.pending_delegate_tool_calls.len(), 1);

        assert!(state.apply_lifecycle_update(DelegateLifecycleUpdate {
            delegation_id: "delegation-1".to_string(),
            tool_call_id: Some("tool-1".to_string()),
            target_agent_id: "coder".to_string(),
            objective: "build feature".to_string(),
            child_session_id: None,
            status: DelegateStatus::InProgress,
            lifecycle_rank: 1,
            requested_at: 10,
            finished_at: None,
            updated_at: 10,
            result_summary: None,
            error: None,
        }));
        assert_eq!(state.delegate_entries.len(), 1);
        assert_eq!(state.delegate_entries[0].delegation_id, "delegation-1");
        assert!(state.pending_delegate_tool_calls.is_empty());
    }

    #[test]
    fn pending_tool_call_attaches_by_narrow_agent_and_objective_values() {
        let mut state = DelegatesState::new();
        state.upsert_provisional_delegate(
            "tool-1",
            Some("coder".to_string()),
            "build feature".to_string(),
        );

        assert!(apply(
            &mut state,
            "delegation-1",
            DelegateStatus::InProgress,
            10,
            None,
            (None, None),
        ));
        assert_eq!(
            state.delegate_entries[0].delegate_tool_call_id.as_deref(),
            Some("tool-1")
        );
        assert!(state.pending_delegate_tool_calls.is_empty());
    }

    #[test]
    fn lifecycle_rejects_stale_and_equal_timestamp_regressions() {
        let mut state = DelegatesState::new();
        assert!(apply(
            &mut state,
            "d1",
            DelegateStatus::Completed,
            20,
            Some("child"),
            (Some("done"), None),
        ));
        assert!(!apply(
            &mut state,
            "d1",
            DelegateStatus::InProgress,
            19,
            None,
            (None, None),
        ));
        assert!(!apply(
            &mut state,
            "d1",
            DelegateStatus::InProgress,
            20,
            Some("child"),
            (None, None),
        ));
        assert_eq!(state.delegate_entries[0].status, DelegateStatus::Completed);
        assert_eq!(state.delegation_result_summaries["d1"], "done");
    }

    #[test]
    fn lifecycle_summary_and_error_values_replace_and_remove_exactly() {
        let mut state = DelegatesState::new();
        apply(
            &mut state,
            "d1",
            DelegateStatus::Completed,
            10,
            None,
            (Some("done"), None),
        );
        assert_eq!(state.delegation_result_summaries["d1"], "done");
        assert!(!state.delegation_errors.contains_key("d1"));

        apply(
            &mut state,
            "d1",
            DelegateStatus::Failed,
            11,
            None,
            (None, Some("boom")),
        );
        assert!(!state.delegation_result_summaries.contains_key("d1"));
        assert_eq!(state.delegation_errors["d1"], "boom");

        apply(
            &mut state,
            "d1",
            DelegateStatus::Cancelled,
            12,
            None,
            (None, None),
        );
        assert!(!state.delegation_errors.contains_key("d1"));
    }

    #[test]
    fn child_updates_before_linkage_attach_stats_and_pending_elicitation() {
        let mut state = DelegatesState::new();
        let pending = DelegateChildState::PendingElicitation {
            elicitation_id: "elic-1".to_string(),
            message: "Need approval".to_string(),
            requested_schema: serde_json::json!({"type": "object"}),
            source: "builtin:question".to_string(),
        };
        let stats = DelegateStats {
            tool_calls: 2,
            messages: 1,
            ..DelegateStats::default()
        };
        assert!(!state.apply_child_snapshot("child", pending.clone(), stats.clone()));

        apply(
            &mut state,
            "d1",
            DelegateStatus::InProgress,
            10,
            Some("child"),
            (None, None),
        );
        assert_eq!(state.delegate_entries[0].child_state, pending);
        assert_eq!(state.delegate_entries[0].stats, stats);
        assert!(state.pending_delegate_child_states.is_empty());
        assert!(state.pending_delegate_child_stats.is_empty());
    }

    #[test]
    fn duplicate_child_assistant_message_ids_are_suppressed() {
        let mut state = DelegatesState::new();
        assert!(state.record_child_message_id("child", Some("m1"), true));
        assert!(!state.record_child_message_id("child", Some("m1"), true));
        assert!(state.record_child_message_id("child", None, true));
        assert!(!state.record_child_message_id("child", None, false));
    }

    #[test]
    fn root_clear_and_child_load_preservation_keep_existing_asymmetry() {
        let mut state = DelegatesState::new();
        state
            .delegate_entries
            .push(entry("d1", "task", None, DelegateStatus::InProgress));
        state
            .pending_delegate_child_states
            .insert("child".to_string(), DelegateChildState::OtherProgress);
        state
            .delegate_child_message_ids
            .insert("child".to_string(), HashSet::from(["m1".to_string()]));
        state.delegation_update_times.insert("d1".to_string(), 10);
        state
            .delegation_result_summaries
            .insert("d1".to_string(), "done".to_string());
        state
            .delegation_errors
            .insert("d1".to_string(), "boom".to_string());
        state.delegate_filter = "stale".to_string();
        state.delegate_cursor = 3;
        state.parent_session_id = Some("parent".to_string());

        let entries_before = state.delegate_entries.clone();
        if state.parent_session_id.is_none() {
            state.clear_for_root_session();
        }
        assert_eq!(state.delegate_entries, entries_before);

        state.parent_session_id = None;
        state.clear_for_root_session();
        assert!(state.delegate_entries.is_empty());
        assert!(state.pending_delegate_child_states.is_empty());
        assert!(state.delegate_child_message_ids.is_empty());
        assert!(state.delegation_update_times.is_empty());
        assert!(state.delegation_result_summaries.is_empty());
        assert!(state.delegation_errors.is_empty());
        assert_eq!(state.delegate_filter, "stale");
        assert_eq!(state.delegate_cursor, 3);
    }

    #[test]
    fn reducer_lifecycle_filters_and_preserves_timestamp_rank_and_tie_semantics() {
        let mut state = DelegatesState::new();
        let update = |session_id: &str, lifecycle, updated_at| DelegationUpdate {
            session_id: session_id.into(),
            delegation_id: "d1".into(),
            tool_call_id: Some("tool-1".into()),
            state: lifecycle,
            target_agent_id: "coder".into(),
            objective: "build feature".into(),
            child_session_id: (lifecycle != DelegationState::Requested).then(|| "child".into()),
            requested_at: 10,
            forked_at: (lifecycle != DelegationState::Requested).then_some(15),
            finished_at: matches!(
                lifecycle,
                DelegationState::Completed | DelegationState::Failed | DelegationState::Cancelled
            )
            .then_some(20),
            updated_at,
            result_summary: (lifecycle == DelegationState::Completed).then(|| "done".into()),
            error: (lifecycle == DelegationState::Failed).then(|| "boom".into()),
        };
        let context = DelegateContext {
            active_session_id: Some("parent".into()),
            inactive_activity_session_id: None,
        };

        let filtered = state.reduce(
            DelegateAction::LifecycleUpdate(update("other-parent", DelegationState::Requested, 10)),
            context.clone(),
        );
        assert_eq!(filtered, DelegateOutcome::default());
        assert!(state.delegate_entries.is_empty());

        let accepted = state.reduce(
            DelegateAction::LifecycleUpdate(update("parent", DelegationState::Completed, 20)),
            context.clone(),
        );
        assert!(accepted.changed);
        assert!(accepted.presentation_changed);
        assert!(accepted.effects.is_empty());

        let stale = state.reduce(
            DelegateAction::LifecycleUpdate(update("parent", DelegationState::Forked, 19)),
            context.clone(),
        );
        assert_eq!(stale, DelegateOutcome::default());
        let equal_lower_rank = state.reduce(
            DelegateAction::LifecycleUpdate(update("parent", DelegationState::Forked, 20)),
            context.clone(),
        );
        assert_eq!(equal_lower_rank, DelegateOutcome::default());

        let equal_rank_tie = state.reduce(
            DelegateAction::LifecycleUpdate(update("parent", DelegationState::Failed, 20)),
            context.clone(),
        );
        assert!(equal_rank_tie.changed);
        assert!(equal_rank_tie.presentation_changed);
        assert_eq!(state.delegate_entries[0].status, DelegateStatus::Failed);
        assert_eq!(state.delegation_errors["d1"], "boom");
        assert!(!state.delegation_result_summaries.contains_key("d1"));

        let repeated = state.reduce(
            DelegateAction::LifecycleUpdate(update("parent", DelegationState::Failed, 20)),
            context,
        );
        assert_eq!(repeated, DelegateOutcome::default());
    }

    #[test]
    fn reducer_provisional_action_reconciles_authoritative_metadata_without_duplicates() {
        let mut state = DelegatesState::new();
        let action = DelegateAction::ProvisionalToolDelegate {
            tool_call_id: Some("tool-1".into()),
            arguments: Some(serde_json::json!({
                "target_agent_id": "coder",
                "objective": "build feature"
            })),
        };

        let inserted = state.reduce(action.clone(), DelegateContext::default());
        assert!(inserted.changed);
        assert!(inserted.presentation_changed);
        assert!(inserted.effects.is_empty());
        assert_eq!(
            state.reduce(action, DelegateContext::default()),
            DelegateOutcome::default()
        );
        assert_eq!(
            state.reduce(
                DelegateAction::ProvisionalToolDelegate {
                    tool_call_id: None,
                    arguments: None,
                },
                DelegateContext::default(),
            ),
            DelegateOutcome::default()
        );

        let authoritative = state.reduce(
            DelegateAction::LifecycleUpdate(DelegationUpdate {
                session_id: "parent".into(),
                delegation_id: "d1".into(),
                tool_call_id: Some("tool-1".into()),
                state: DelegationState::Forked,
                target_agent_id: "reviewer".into(),
                objective: "review feature".into(),
                child_session_id: Some("child".into()),
                requested_at: 10,
                forked_at: Some(11),
                finished_at: None,
                updated_at: 11,
                result_summary: None,
                error: None,
            }),
            DelegateContext {
                active_session_id: Some("parent".into()),
                inactive_activity_session_id: None,
            },
        );
        assert!(authoritative.changed);
        assert!(authoritative.presentation_changed);
        assert_eq!(state.delegate_entries.len(), 1);
        let entry = &state.delegate_entries[0];
        assert_eq!(entry.delegation_id, "d1");
        assert_eq!(entry.delegate_tool_call_id.as_deref(), Some("tool-1"));
        assert_eq!(entry.target_agent_id.as_deref(), Some("reviewer"));
        assert_eq!(entry.objective, "review feature");
        assert_eq!(entry.child_session_id.as_deref(), Some("child"));
        assert!(state.pending_delegate_tool_calls.is_empty());
    }

    #[test]
    fn reducer_inactive_activity_stages_then_attaches_and_reports_presentation_changes() {
        let mut state = DelegatesState::new();
        let context = DelegateContext {
            active_session_id: Some("parent".into()),
            inactive_activity_session_id: Some("child".into()),
        };
        let rejected = state.reduce(
            DelegateAction::InactiveChildActivity {
                session_id: "child".into(),
                activity: DelegateChildActivity::AssistantContent {
                    message_id: Some("m1".into()),
                },
            },
            DelegateContext {
                active_session_id: Some("parent".into()),
                inactive_activity_session_id: None,
            },
        );
        assert!(!rejected.changed);
        assert!(!rejected.presentation_changed);
        assert!(state.pending_delegate_child_states.is_empty());
        assert!(state.pending_delegate_child_stats.is_empty());

        let staged = state.reduce(
            DelegateAction::InactiveChildActivity {
                session_id: "child".into(),
                activity: DelegateChildActivity::AssistantContent {
                    message_id: Some("m1".into()),
                },
            },
            context.clone(),
        );
        assert!(!staged.changed);
        assert!(!staged.presentation_changed);
        assert!(staged.effects.is_empty());
        assert_eq!(
            state.pending_delegate_child_states["child"],
            DelegateChildState::AssistantMessage
        );
        assert_eq!(state.pending_delegate_child_stats["child"].messages, 1);

        let linked = state.reduce(
            DelegateAction::LifecycleUpdate(DelegationUpdate {
                session_id: "parent".into(),
                delegation_id: "d1".into(),
                tool_call_id: None,
                state: DelegationState::Forked,
                target_agent_id: "coder".into(),
                objective: "build feature".into(),
                child_session_id: Some("child".into()),
                requested_at: 10,
                forked_at: Some(11),
                finished_at: None,
                updated_at: 11,
                result_summary: None,
                error: None,
            }),
            context.clone(),
        );
        assert!(linked.changed);
        assert!(linked.presentation_changed);
        assert_eq!(state.delegate_entries[0].stats.messages, 1);
        assert_eq!(
            state.delegate_entries[0].child_state,
            DelegateChildState::AssistantMessage
        );
        assert!(!state.pending_delegate_child_states.contains_key("child"));
        assert!(!state.pending_delegate_child_stats.contains_key("child"));

        let changed = state.reduce(
            DelegateAction::InactiveChildActivity {
                session_id: "child".into(),
                activity: DelegateChildActivity::ToolCallStarted,
            },
            context.clone(),
        );
        assert!(changed.changed);
        assert!(changed.presentation_changed);
        assert_eq!(state.delegate_entries[0].stats.tool_calls, 1);

        let unchanged = state.reduce(
            DelegateAction::InactiveChildActivity {
                session_id: "child".into(),
                activity: DelegateChildActivity::Unchanged,
            },
            context,
        );
        assert!(!unchanged.changed);
        assert!(!unchanged.presentation_changed);
    }

    #[test]
    fn reducer_resolve_and_clear_actions_report_exact_changes() {
        let mut state = DelegatesState::new();
        state.pending_parent_session_id = Some("staged-parent".into());
        state
            .delegate_entries
            .push(entry("d1", "task", None, DelegateStatus::InProgress));

        let resolved = state.reduce(
            DelegateAction::ResolveLoadedSessionParent(Some("catalog-parent".into())),
            DelegateContext::default(),
        );
        assert!(resolved.changed);
        assert!(!resolved.presentation_changed);
        assert!(resolved.effects.is_empty());
        assert_eq!(state.parent_session_id.as_deref(), Some("staged-parent"));
        assert_eq!(state.pending_parent_session_id, None);
        assert_eq!(
            state.reduce(
                DelegateAction::ResolveLoadedSessionParent(Some("staged-parent".into())),
                DelegateContext::default(),
            ),
            DelegateOutcome::default()
        );

        let protected = state.reduce(
            DelegateAction::ClearRootSessionState,
            DelegateContext::default(),
        );
        assert_eq!(protected, DelegateOutcome::default());
        assert_eq!(state.delegate_entries.len(), 1);

        let cleared_parent = state.reduce(
            DelegateAction::ClearNewRootParent,
            DelegateContext::default(),
        );
        assert!(cleared_parent.changed);
        assert!(!cleared_parent.presentation_changed);
        assert_eq!(
            state.reduce(
                DelegateAction::ClearNewRootParent,
                DelegateContext::default(),
            ),
            DelegateOutcome::default()
        );
        let cleared_root = state.reduce(
            DelegateAction::ClearRootSessionState,
            DelegateContext::default(),
        );
        assert!(cleared_root.changed);
        assert!(!cleared_root.presentation_changed);
        assert!(state.delegate_entries.is_empty());
        assert_eq!(
            state.reduce(
                DelegateAction::ClearRootSessionState,
                DelegateContext::default(),
            ),
            DelegateOutcome::default()
        );
    }

    #[test]
    fn sibling_parent_staging_prefers_real_parent_and_survives_resolution() {
        let mut state = DelegatesState::new();
        state.stage_parent_for_child_navigation(
            Some("root".to_string()),
            Some("child-a".to_string()),
        );
        state.resolve_parent_session_id(Some("catalog-parent".to_string()));
        assert_eq!(state.parent_session_id.as_deref(), Some("root"));
        assert_eq!(state.pending_parent_session_id, None);

        state.stage_parent_for_child_navigation(None, Some("root".to_string()));
        state.resolve_parent_session_id(None);
        assert_eq!(state.parent_session_id.as_deref(), Some("root"));
    }
}
