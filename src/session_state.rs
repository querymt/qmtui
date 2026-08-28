use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::application::Effect;
use crate::command::{Command, SessionListRequest};
use crate::composer_state::FileIndexEntryLite;
use crate::diagnostics::LogLevel;
use crate::domain::activity::SessionActivity;
use crate::domain::mesh::RemoteSessionLocation;
use crate::domain::session::{SessionChildrenPage, SessionGroup, SessionListPage, SessionSummary};

/// Maximum number of recent sessions shown per group on the start page.
pub(crate) const MAX_RECENT_SESSIONS: usize = 3;
/// Maximum discovery-preview sessions retained before the workspace page arrives.
pub(crate) const POPUP_SESSION_PAGE_TARGET: usize = 10;
pub(crate) const SESSION_CHILD_PAGE_LIMIT: u32 = 10;
/// Maximum number of workspace groups shown on the start page.
pub(crate) const MAX_VISIBLE_GROUPS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PathCompletionState {
    pub(crate) query: String,
    pub(crate) selected_index: usize,
    pub(crate) results: Vec<FileIndexEntryLite>,
}

/// A single visible row on the start-page session list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StartPageItem {
    GroupHeader {
        cwd: Option<String>,
        session_count: usize,
        session_total: Option<u64>,
        collapsed: bool,
    },
    Session {
        group_idx: usize,
        path: Vec<usize>,
        depth: usize,
    },
    ShowMore {
        group_idx: usize,
        remaining: usize,
        has_more: bool,
    },
}

/// A single visible row in the sessions popup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PopupItem {
    GroupHeader {
        cwd: Option<String>,
        session_count: usize,
        session_total: Option<u64>,
        collapsed: bool,
    },
    Session {
        group_idx: usize,
        path: Vec<usize>,
        depth: usize,
    },
    LoadMore {
        group_idx: usize,
        parent_path: Vec<usize>,
    },
}

pub(crate) fn session_group_count_text(session_count: usize, session_total: Option<u64>) -> String {
    session_total
        .map(|total| format!("{session_count}/{total}"))
        .unwrap_or_else(|| session_count.to_string())
}

#[derive(Debug, Clone)]
pub(crate) enum SessionAction {
    CatalogPage {
        request: SessionListRequest,
        page: SessionListPage,
    },
    CatalogFailed {
        request: SessionListRequest,
        message: String,
    },
    Created {
        agent_id: String,
        session_id: String,
        profile_id: Option<String>,
    },
    Loaded {
        agent_id: String,
        session_id: String,
        profile_id: Option<String>,
    },
    InitializeAgentId(String),
    ReplaceAgentMode(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SessionCoordination {
    PushChatError(String),
    ClearDelegateParent,
    SetChatIdle,
    ResolveDelegateParent(Option<String>),
    ApplyProfileBinding {
        session_id: String,
        profile_id: Option<String>,
        remove_if_missing: bool,
    },
    ResetActiveSessionView,
    SelectChat,
    SelectLoadedSessionScreen,
    Status {
        level: LogLevel,
        target: &'static str,
        message: String,
    },
    SynchronizeProfileCommands {
        session_id: String,
        root_only: bool,
    },
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SessionOutcome {
    pub(crate) coordination: Vec<SessionCoordination>,
    pub(crate) effects: Vec<Effect>,
}

pub(crate) struct SessionsState {
    pub(crate) session_groups: Vec<SessionGroup>,
    pub(crate) session_cursor: usize,
    pub(crate) session_filter: String,
    pub(crate) session_popup_tab: usize,
    pub(crate) collapsed_groups: HashSet<String>,
    pub(crate) popup_collapsed_groups: HashSet<String>,
    pub(crate) session_discovery_in_progress: bool,
    pub(crate) session_discovery_cursors: HashSet<String>,
    pub(crate) pending_session_group_loads: HashSet<Option<String>>,
    pub(crate) hydrated_session_groups: HashSet<String>,
    pub(crate) expanded_session_children: HashSet<String>,
    pub(crate) pending_session_child_loads: HashSet<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) agent_id: Option<String>,
    pub(crate) agent_mode: String,
    pub(crate) mode_before_review: Option<String>,
    pub(crate) new_session_path: String,
    pub(crate) new_session_cursor: usize,
    pub(crate) new_session_completion: Option<PathCompletionState>,
    pub(crate) session_activity: HashMap<String, SessionActivity>,
    pub(crate) remote_session_locations: HashMap<String, RemoteSessionLocation>,
}

impl SessionsState {
    pub(crate) fn reduce(&mut self, action: SessionAction) -> SessionOutcome {
        match action {
            SessionAction::CatalogPage { request, page } => {
                let SessionListPage {
                    groups,
                    next_cursor,
                    total_count: _,
                } = page;
                let commands = match request {
                    SessionListRequest::Discovery => {
                        let (workspaces, next_discovery_cursor) =
                            self.apply_discovery_page(groups, next_cursor);
                        let mut commands = workspaces
                            .into_iter()
                            .map(Command::list_sessions_workspace)
                            .collect::<Vec<_>>();
                        if let Some(cursor) = next_discovery_cursor {
                            commands.push(Command::list_sessions_discovery(Some(cursor)));
                        }
                        commands
                    }
                    SessionListRequest::WorkspaceFirstPage { cwd } => {
                        self.apply_workspace_first_page(cwd, groups);
                        Vec::new()
                    }
                    SessionListRequest::WorkspaceContinuation { cwd } => {
                        self.apply_workspace_continuation(cwd, groups);
                        Vec::new()
                    }
                };
                SessionOutcome {
                    coordination: Vec::new(),
                    effects: commands.into_iter().map(Effect::Command).collect(),
                }
            }
            SessionAction::CatalogFailed { request, message } => {
                match request {
                    SessionListRequest::Discovery => self.fail_discovery(),
                    SessionListRequest::WorkspaceFirstPage { cwd }
                    | SessionListRequest::WorkspaceContinuation { cwd } => {
                        self.fail_workspace_request(&cwd);
                    }
                }
                SessionOutcome {
                    coordination: vec![
                        SessionCoordination::PushChatError(message.clone()),
                        SessionCoordination::Status {
                            level: LogLevel::Error,
                            target: "acp",
                            message: format!("error: {message}"),
                        },
                    ],
                    effects: Vec::new(),
                }
            }
            SessionAction::Created {
                agent_id,
                session_id,
                profile_id,
            } => {
                let remove_if_missing = self.is_remote_session_id(&session_id);
                self.session_id = Some(session_id.clone());
                self.agent_id = Some(agent_id);
                self.mode_before_review = None;
                SessionOutcome {
                    coordination: vec![
                        SessionCoordination::ClearDelegateParent,
                        SessionCoordination::ApplyProfileBinding {
                            session_id: session_id.clone(),
                            profile_id,
                            remove_if_missing,
                        },
                        SessionCoordination::ResetActiveSessionView,
                        SessionCoordination::SelectChat,
                        SessionCoordination::Status {
                            level: LogLevel::Info,
                            target: "session",
                            message: "session created".into(),
                        },
                        SessionCoordination::SynchronizeProfileCommands {
                            session_id: session_id.clone(),
                            root_only: false,
                        },
                    ],
                    effects: vec![Effect::Command(Command::SubscribeSession {
                        session_id,
                        agent_id: self.agent_id.clone(),
                    })],
                }
            }
            SessionAction::Loaded {
                agent_id,
                session_id,
                profile_id,
            } => {
                let discovered_parent = self.session_parent_id(&session_id).map(str::to_owned);
                let remove_if_missing = self.is_remote_session_id(&session_id);
                self.session_id = Some(session_id.clone());
                self.agent_id = Some(agent_id);
                self.mode_before_review = None;
                SessionOutcome {
                    coordination: vec![
                        SessionCoordination::SetChatIdle,
                        SessionCoordination::ResolveDelegateParent(discovered_parent),
                        SessionCoordination::ApplyProfileBinding {
                            session_id: session_id.clone(),
                            profile_id,
                            remove_if_missing,
                        },
                        SessionCoordination::ResetActiveSessionView,
                        SessionCoordination::SelectLoadedSessionScreen,
                        SessionCoordination::Status {
                            level: LogLevel::Debug,
                            target: "activity",
                            message: "ready".into(),
                        },
                        SessionCoordination::SynchronizeProfileCommands {
                            session_id,
                            root_only: true,
                        },
                    ],
                    effects: vec![Effect::Command(Command::SetAgentMode {
                        mode: self.agent_mode.clone(),
                    })],
                }
            }
            SessionAction::InitializeAgentId(agent_id) => {
                self.agent_id = Some(agent_id);
                SessionOutcome::default()
            }
            SessionAction::ReplaceAgentMode(mode) => {
                self.replace_agent_mode(mode);
                SessionOutcome::default()
            }
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            session_groups: Vec::new(),
            session_cursor: 0,
            session_filter: String::new(),
            session_popup_tab: 0,
            collapsed_groups: HashSet::new(),
            popup_collapsed_groups: HashSet::new(),
            session_discovery_in_progress: false,
            session_discovery_cursors: HashSet::new(),
            pending_session_group_loads: HashSet::new(),
            hydrated_session_groups: HashSet::new(),
            expanded_session_children: HashSet::new(),
            pending_session_child_loads: HashSet::new(),
            session_id: None,
            agent_id: None,
            agent_mode: "build".into(),
            mode_before_review: None,
            new_session_path: String::new(),
            new_session_cursor: 0,
            new_session_completion: None,
            session_activity: HashMap::new(),
            remote_session_locations: HashMap::new(),
        }
    }

    pub(crate) fn prepare_session_discovery(&mut self) -> bool {
        if self.session_discovery_in_progress || !self.pending_session_group_loads.is_empty() {
            return false;
        }
        self.session_groups.clear();
        self.session_discovery_cursors.clear();
        self.pending_session_group_loads.clear();
        self.hydrated_session_groups.clear();
        self.session_discovery_in_progress = true;
        true
    }

    pub(crate) fn prepare_session_group_page(
        &mut self,
        group_idx: usize,
    ) -> Option<(String, String)> {
        let group = self.session_groups.get(group_idx)?;
        let cursor = group.next_cursor.clone()?;
        let cwd = group.cwd.clone()?;
        if !self.pending_session_group_loads.insert(Some(cwd.clone())) {
            return None;
        }
        Some((cwd, cursor))
    }

    pub(crate) fn prepare_session_child_page(
        &mut self,
        group_idx: usize,
        parent_path: &[usize],
    ) -> Option<(String, Option<String>)> {
        let parent = self.session_by_path(group_idx, parent_path)?;
        let parent_session_id = parent.session_id.clone();
        let cursor = parent.children_next_cursor.clone();
        if !self
            .pending_session_child_loads
            .insert(parent_session_id.clone())
        {
            return None;
        }
        Some((parent_session_id, cursor))
    }

    pub(crate) fn apply_discovery_page(
        &mut self,
        mut groups: Vec<SessionGroup>,
        next_cursor: Option<String>,
    ) -> (Vec<String>, Option<String>) {
        self.normalize_session_groups(&mut groups);
        self.session_discovery_in_progress = false;
        let mut workspace_requests = Vec::new();

        for mut group in groups {
            group.next_cursor = None;
            match group.cwd.clone() {
                Some(cwd) => {
                    let hydrated = self.hydrated_session_groups.contains(&cwd);
                    if let Some(existing) = self
                        .session_groups
                        .iter_mut()
                        .find(|existing| existing.cwd.as_deref() == Some(cwd.as_str()))
                    {
                        if !hydrated {
                            Self::merge_session_page(
                                existing,
                                group.sessions,
                                Some(POPUP_SESSION_PAGE_TARGET),
                            );
                        }
                    } else {
                        group.sessions.truncate(POPUP_SESSION_PAGE_TARGET);
                        self.session_groups.push(group);
                    }

                    if !hydrated && self.pending_session_group_loads.insert(Some(cwd.clone())) {
                        workspace_requests.push(cwd);
                    }
                }
                None => {
                    if let Some(existing) = self
                        .session_groups
                        .iter_mut()
                        .find(|existing| existing.cwd.is_none())
                    {
                        Self::merge_session_page(
                            existing,
                            group.sessions,
                            Some(POPUP_SESSION_PAGE_TARGET),
                        );
                    } else {
                        group.sessions.truncate(POPUP_SESSION_PAGE_TARGET);
                        self.session_groups.push(group);
                    }
                }
            }
        }

        let next_request = next_cursor.filter(|cursor| {
            let queued = self.session_discovery_cursors.insert(cursor.clone());
            if queued {
                self.session_discovery_in_progress = true;
            }
            queued
        });
        self.finish_catalog_update();
        (workspace_requests, next_request)
    }

    pub(crate) fn apply_workspace_first_page(
        &mut self,
        cwd: String,
        mut groups: Vec<SessionGroup>,
    ) {
        self.normalize_session_groups(&mut groups);
        self.pending_session_group_loads.remove(&Some(cwd.clone()));
        self.hydrated_session_groups.insert(cwd.clone());
        let mut group = groups
            .into_iter()
            .find(|group| group.cwd.as_deref() == Some(cwd.as_str()))
            .unwrap_or_else(|| SessionGroup {
                cwd: Some(cwd.clone()),
                ..SessionGroup::default()
            });
        group.cwd = Some(cwd.clone());
        group.total_count = None;

        if group.sessions.is_empty() {
            self.session_groups
                .retain(|existing| existing.cwd.as_deref() != Some(cwd.as_str()));
        } else if let Some(existing) = self
            .session_groups
            .iter_mut()
            .find(|existing| existing.cwd.as_deref() == Some(cwd.as_str()))
        {
            *existing = group;
        } else {
            self.session_groups.push(group);
        }
        self.finish_catalog_update();
    }

    pub(crate) fn apply_workspace_continuation(
        &mut self,
        cwd: String,
        mut groups: Vec<SessionGroup>,
    ) {
        self.normalize_session_groups(&mut groups);
        self.pending_session_group_loads.remove(&Some(cwd.clone()));
        let page = groups
            .into_iter()
            .find(|group| group.cwd.as_deref() == Some(cwd.as_str()))
            .unwrap_or_else(|| SessionGroup {
                cwd: Some(cwd.clone()),
                ..SessionGroup::default()
            });
        if let Some(existing) = self
            .session_groups
            .iter_mut()
            .find(|existing| existing.cwd.as_deref() == Some(cwd.as_str()))
        {
            existing.next_cursor = page.next_cursor.clone();
            Self::merge_session_page(existing, page.sessions, None);
        } else if !page.sessions.is_empty() {
            self.session_groups.push(page);
        }
        self.finish_catalog_update();
    }

    pub(crate) fn fail_discovery(&mut self) {
        self.session_discovery_in_progress = false;
    }

    pub(crate) fn fail_workspace_request(&mut self, cwd: &str) {
        self.pending_session_group_loads
            .remove(&Some(cwd.to_string()));
    }

    fn normalize_session_groups(&mut self, groups: &mut [SessionGroup]) {
        for group in groups {
            group.total_count = None;
            for session in &group.sessions {
                if let Some(node_id) = session
                    .node_id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty())
                {
                    self.remember_remote_session_location(
                        &session.session_id,
                        node_id,
                        session.cwd.clone(),
                    );
                }
            }
            group.sessions.sort_by(|a, b| {
                b.updated_at
                    .cmp(&a.updated_at)
                    .then_with(|| b.session_id.cmp(&a.session_id))
            });
            group.latest_activity = group
                .sessions
                .first()
                .and_then(|session| session.updated_at.clone());
        }
    }

    fn finish_catalog_update(&mut self) {
        self.session_groups
            .retain(|group| !group.sessions.is_empty());
        self.session_groups.sort_by(|a, b| {
            b.latest_activity
                .cmp(&a.latest_activity)
                .then_with(|| a.cwd.cmp(&b.cwd))
        });
        self.clamp_start_cursor();
    }

    pub(crate) fn reset_browser_for_open(&mut self) {
        self.session_popup_tab = 0;
        self.session_cursor = 0;
        self.session_filter.clear();
    }

    pub(crate) fn switch_session_popup_tab(&mut self) {
        self.session_popup_tab = 1 - self.session_popup_tab;
    }

    pub(crate) fn move_start_cursor_up(&mut self) {
        self.session_cursor = self.session_cursor.saturating_sub(1);
    }

    pub(crate) fn move_start_cursor_down(&mut self) {
        let max = self.visible_start_items().len();
        if self.session_cursor < max {
            self.session_cursor += 1;
        }
    }

    pub(crate) fn move_popup_cursor_up(&mut self) {
        self.session_cursor = self.session_cursor.saturating_sub(1);
    }

    pub(crate) fn move_popup_cursor_down(&mut self) {
        let max = self.visible_popup_items().len().saturating_sub(1);
        self.session_cursor = self.session_cursor.saturating_add(1).min(max);
    }

    pub(crate) fn move_popup_cursor_page(&mut self, step: usize, down: bool) {
        if down {
            let max = self.visible_popup_items().len().saturating_sub(1);
            self.session_cursor = self.session_cursor.saturating_add(step).min(max);
        } else {
            self.session_cursor = self.session_cursor.saturating_sub(step);
        }
    }

    pub(crate) fn start_filter_insert(&mut self, character: char) {
        self.session_filter.push(character);
        self.session_cursor = 0;
    }

    pub(crate) fn start_filter_backspace(&mut self) {
        self.session_filter.pop();
        self.session_cursor = 0;
    }

    pub(crate) fn popup_filter_insert(&mut self, character: char) {
        self.session_filter.push(character);
        self.session_cursor = 0;
    }

    pub(crate) fn popup_filter_backspace(&mut self) {
        self.session_filter.pop();
        self.session_cursor = 0;
    }

    pub(crate) fn clamp_start_cursor(&mut self) {
        let visible_len = self.visible_start_items().len();
        if visible_len > 0 && self.session_cursor >= visible_len {
            self.session_cursor = visible_len - 1;
        }
    }

    pub(crate) fn clamp_popup_cursor(&mut self) {
        let visible_len = self.visible_popup_items().len();
        if visible_len > 0 && self.session_cursor >= visible_len {
            self.session_cursor = visible_len - 1;
        }
    }

    pub(crate) fn remove_session_at(
        &mut self,
        group_idx: usize,
        path: &[usize],
        popup_items: bool,
    ) -> Option<(SessionSummary, bool)> {
        let session = self.session_by_path(group_idx, path)?.clone();
        let is_remote = self.is_remote_session_id(&session.session_id);
        if path.len() == 1 {
            self.session_groups
                .get_mut(group_idx)?
                .sessions
                .remove(path[0]);
        } else if let Some(parent_path) = path.get(..path.len() - 1)
            && let Some(parent) = self.session_by_path_mut(group_idx, parent_path)
            && let Some(child_idx) = path.last()
            && *child_idx < parent.children.len()
        {
            parent.children.remove(*child_idx);
        }
        self.session_groups
            .retain(|group| !group.sessions.is_empty());
        if popup_items {
            self.clamp_popup_cursor();
        } else {
            self.clamp_start_cursor();
        }
        Some((session, is_remote))
    }

    pub(crate) fn session_summary_by_id(&self, session_id: &str) -> Option<&SessionSummary> {
        fn find<'a>(
            sessions: &'a [SessionSummary],
            session_id: &str,
        ) -> Option<&'a SessionSummary> {
            for session in sessions {
                if session.session_id == session_id {
                    return Some(session);
                }
                if let Some(child) = find(&session.children, session_id) {
                    return Some(child);
                }
            }
            None
        }

        self.session_groups
            .iter()
            .find_map(|group| find(&group.sessions, session_id))
    }

    pub(crate) fn session_parent_id(&self, session_id: &str) -> Option<&str> {
        self.session_summary_by_id(session_id)
            .and_then(|session| session.parent_session_id.as_deref())
    }

    pub(crate) fn remember_remote_session_location(
        &mut self,
        session_id: &str,
        node_id: &str,
        cwd: Option<String>,
    ) {
        let location = self
            .remote_session_locations
            .entry(session_id.to_string())
            .or_default();
        location.node_id = node_id.to_string();
        if let Some(cwd) = cwd.filter(|cwd| !cwd.trim().is_empty()) {
            location.cwd = Some(cwd);
        }
    }

    pub(crate) fn remember_remote_session_node(&mut self, session_id: &str, node_id: &str) {
        self.remember_remote_session_location(session_id, node_id, None);
    }

    pub(crate) fn session_remote_node_id(&self, session_id: &str) -> Option<&str> {
        self.session_summary_by_id(session_id)
            .and_then(|session| session.node_id.as_deref())
            .or_else(|| {
                self.remote_session_locations
                    .get(session_id)
                    .map(|location| location.node_id.as_str())
            })
    }

    pub(crate) fn session_remote_cwd(&self, session_id: &str) -> Option<String> {
        self.session_summary_by_id(session_id)
            .filter(|session| session.node_id.is_some() || session.node.is_some())
            .and_then(|session| session.cwd.as_deref())
            .filter(|cwd| !cwd.trim().is_empty())
            .map(str::to_string)
            .or_else(|| {
                self.remote_session_locations
                    .get(session_id)
                    .and_then(|location| location.cwd.as_deref())
                    .filter(|cwd| !cwd.trim().is_empty())
                    .map(str::to_string)
            })
    }

    pub(crate) fn is_remote_session_id(&self, session_id: &str) -> bool {
        self.session_remote_node_id(session_id).is_some()
            || self
                .session_summary_by_id(session_id)
                .map(|session| session.node.is_some())
                .unwrap_or(false)
    }

    pub(crate) fn current_session_is_remote(&self) -> bool {
        self.session_id
            .as_deref()
            .map(|session_id| self.is_remote_session_id(session_id))
            .unwrap_or(false)
    }

    pub(crate) fn session_by_path(
        &self,
        group_idx: usize,
        path: &[usize],
    ) -> Option<&SessionSummary> {
        let (first, rest) = path.split_first()?;
        let mut session = self.session_groups.get(group_idx)?.sessions.get(*first)?;
        for idx in rest {
            session = session.children.get(*idx)?;
        }
        Some(session)
    }

    pub(crate) fn session_by_path_mut(
        &mut self,
        group_idx: usize,
        path: &[usize],
    ) -> Option<&mut SessionSummary> {
        fn descend<'a>(
            sessions: &'a mut [SessionSummary],
            path: &[usize],
        ) -> Option<&'a mut SessionSummary> {
            let (first, rest) = path.split_first()?;
            let session = sessions.get_mut(*first)?;
            if rest.is_empty() {
                Some(session)
            } else {
                descend(&mut session.children, rest)
            }
        }
        descend(&mut self.session_groups.get_mut(group_idx)?.sessions, path)
    }

    pub(crate) fn expandable_root_session(&self, group_idx: usize, path: &[usize]) -> bool {
        let Some(session) = self.session_by_path(group_idx, path) else {
            return false;
        };
        path.len() == 1
            && session.parent_session_id.is_none()
            && session.fork_count > 0
            && session.node.is_none()
            && session.node_id.is_none()
    }

    pub(crate) fn toggle_session_children(&mut self, group_idx: usize, path: &[usize]) -> bool {
        if !self.expandable_root_session(group_idx, path) {
            return false;
        }
        let Some(session) = self.session_by_path(group_idx, path) else {
            return false;
        };
        let session_id = session.session_id.clone();
        let should_load = session.children.is_empty()
            && session.children_next_cursor.is_none()
            && !self.pending_session_child_loads.contains(&session_id);
        if !self.expanded_session_children.remove(&session_id) {
            self.expanded_session_children.insert(session_id);
            return should_load;
        }
        false
    }

    pub(crate) fn merge_session_children(&mut self, data: SessionChildrenPage) {
        let remote_locations: Vec<(String, String, Option<String>)> = data
            .sessions
            .iter()
            .filter_map(|session| {
                Some((
                    session.session_id.clone(),
                    session.node_id.clone()?,
                    session.cwd.clone(),
                ))
            })
            .collect();
        for (session_id, node_id, cwd) in remote_locations {
            self.remember_remote_session_location(&session_id, &node_id, cwd);
        }

        let had_pending_request = self
            .pending_session_child_loads
            .remove(&data.parent_session_id);
        if let Some(parent) = self
            .session_groups
            .iter_mut()
            .find_map(|group| session_by_id_mut(&mut group.sessions, &data.parent_session_id))
        {
            let mut sessions: Vec<SessionSummary> = data
                .sessions
                .into_iter()
                .filter(fork_browsing_child)
                .collect();
            let append = had_pending_request
                && !parent.children.is_empty()
                && parent.children_next_cursor.is_some();
            if append {
                let mut seen: HashSet<String> = parent
                    .children
                    .iter()
                    .map(|session| session.session_id.clone())
                    .collect();
                parent.children.extend(
                    sessions
                        .into_iter()
                        .filter(|session| seen.insert(session.session_id.clone())),
                );
            } else {
                let mut seen = HashSet::new();
                sessions.retain(|session| seen.insert(session.session_id.clone()));
                parent.children = sessions;
            }
            parent.children_next_cursor = data.next_cursor;
            parent.children_total_count = data.total_count;
            parent.has_children = !parent.children.is_empty() || parent.fork_count > 0;
            if let Some(total) = parent.children_total_count {
                parent.fork_count = total;
            }
        }
    }

    pub(crate) fn visible_start_items(&self) -> Vec<StartPageItem> {
        let query = self.session_filter.to_lowercase();
        let mut items = Vec::new();

        for (group_idx, group) in self
            .session_groups
            .iter()
            .enumerate()
            .take(MAX_VISIBLE_GROUPS)
        {
            let collapse_key = group.cwd.clone().unwrap_or_default();
            let collapsed = self.collapsed_groups.contains(&collapse_key);
            let matching = matching_session_indices(group, &query);
            if !query.is_empty() && matching.is_empty() {
                continue;
            }

            items.push(StartPageItem::GroupHeader {
                cwd: group.cwd.clone(),
                session_count: group.sessions.len(),
                session_total: group.total_count,
                collapsed,
            });

            if !collapsed {
                let visible: Vec<usize> =
                    matching.iter().copied().take(MAX_RECENT_SESSIONS).collect();
                let hidden = matching.len().saturating_sub(MAX_RECENT_SESSIONS);
                for session_idx in visible {
                    items.push(StartPageItem::Session {
                        group_idx,
                        path: vec![session_idx],
                        depth: 0,
                    });
                    let session = &group.sessions[session_idx];
                    if self.expanded_session_children.contains(&session.session_id) {
                        for (child_idx, child) in session.children.iter().enumerate() {
                            if session_matches(child, &query) {
                                items.push(StartPageItem::Session {
                                    group_idx,
                                    path: vec![session_idx, child_idx],
                                    depth: 1,
                                });
                            }
                        }
                    }
                }

                if hidden > 0 || group.next_cursor.is_some() {
                    items.push(StartPageItem::ShowMore {
                        group_idx,
                        remaining: hidden,
                        has_more: group.next_cursor.is_some(),
                    });
                }
            }
        }

        items
    }

    pub(crate) fn toggle_group_collapse(&mut self, cwd: Option<&str>) {
        let key = cwd.unwrap_or("").to_string();
        if !self.collapsed_groups.remove(&key) {
            self.collapsed_groups.insert(key);
        }
    }

    pub(crate) fn toggle_popup_group_collapse(&mut self, cwd: Option<&str>) {
        let key = cwd.unwrap_or("").to_string();
        if !self.popup_collapsed_groups.remove(&key) {
            self.popup_collapsed_groups.insert(key);
        }
    }

    pub(crate) fn visible_popup_items(&self) -> Vec<PopupItem> {
        let query = self.session_filter.to_lowercase();
        let mut items = Vec::new();

        for (group_idx, group) in self.session_groups.iter().enumerate() {
            let collapse_key = group.cwd.clone().unwrap_or_default();
            let collapsed = self.popup_collapsed_groups.contains(&collapse_key);
            let matching = matching_session_indices(group, &query);
            if !query.is_empty() && matching.is_empty() {
                continue;
            }

            items.push(PopupItem::GroupHeader {
                cwd: group.cwd.clone(),
                session_count: group.sessions.len(),
                session_total: group.total_count,
                collapsed,
            });

            if !collapsed {
                for session_idx in matching {
                    items.push(PopupItem::Session {
                        group_idx,
                        path: vec![session_idx],
                        depth: 0,
                    });
                    let session = &group.sessions[session_idx];
                    if self.expanded_session_children.contains(&session.session_id) {
                        for (child_idx, child) in session.children.iter().enumerate() {
                            if session_matches(child, &query) {
                                items.push(PopupItem::Session {
                                    group_idx,
                                    path: vec![session_idx, child_idx],
                                    depth: 1,
                                });
                            }
                        }
                        if session.children_next_cursor.is_some() {
                            items.push(PopupItem::LoadMore {
                                group_idx,
                                parent_path: vec![session_idx],
                            });
                        }
                    }
                }
                if group.next_cursor.is_some()
                    && !self.pending_session_group_loads.contains(&group.cwd)
                {
                    items.push(PopupItem::LoadMore {
                        group_idx,
                        parent_path: Vec::new(),
                    });
                }
            }
        }

        items
    }

    pub(crate) fn merge_session_page(
        group: &mut SessionGroup,
        sessions: Vec<SessionSummary>,
        cap: Option<usize>,
    ) {
        let mut seen = group
            .sessions
            .iter()
            .map(|session| session.session_id.clone())
            .collect::<HashSet<_>>();
        group.sessions.extend(
            sessions
                .into_iter()
                .filter(|session| seen.insert(session.session_id.clone())),
        );
        group.sessions.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| b.session_id.cmp(&a.session_id))
        });
        if let Some(cap) = cap {
            group.sessions.truncate(cap);
        }
        group.latest_activity = group
            .sessions
            .first()
            .and_then(|session| session.updated_at.clone());
        group.total_count = None;
    }

    pub(crate) fn note_session_activity(&mut self, session_id: &str) {
        self.session_activity.insert(
            session_id.to_string(),
            SessionActivity {
                last_event_at: Instant::now(),
            },
        );
    }

    pub(crate) fn active_session_count(&self) -> usize {
        const ACTIVE_SESSION_WINDOW: Duration = Duration::from_secs(5);
        let now = Instant::now();
        self.session_activity
            .values()
            .filter(|activity| now.duration_since(activity.last_event_at) <= ACTIVE_SESSION_WINDOW)
            .count()
    }

    pub(crate) fn other_active_session_count(&self) -> usize {
        const ACTIVE_SESSION_WINDOW: Duration = Duration::from_secs(5);
        let now = Instant::now();
        self.session_activity
            .iter()
            .filter(|(session_id, activity)| {
                now.duration_since(activity.last_event_at) <= ACTIVE_SESSION_WINDOW
                    && self.session_id.as_deref() != Some(session_id.as_str())
            })
            .count()
    }

    pub(crate) fn next_mode(&self) -> String {
        match self.agent_mode.as_str() {
            "build" => "plan".into(),
            "plan" => "build".into(),
            "review" => self
                .mode_before_review
                .clone()
                .unwrap_or_else(|| "build".into()),
            _ => "build".into(),
        }
    }

    pub(crate) fn replace_agent_mode(&mut self, mode: String) {
        self.agent_mode = mode;
        if self.agent_mode != "review" {
            self.mode_before_review = None;
        }
    }

    pub(crate) fn apply_mode_transition(&mut self, target: &str) {
        if target == "review" {
            if self.agent_mode != "review" {
                self.mode_before_review = Some(self.agent_mode.clone());
            }
        } else {
            self.mode_before_review = None;
        }
        self.agent_mode = target.to_string();
    }

    pub(crate) fn move_new_session_cursor_left(&mut self) {
        self.new_session_cursor = self.new_session_cursor.saturating_sub(1);
    }

    pub(crate) fn move_new_session_cursor_right(&mut self) {
        self.new_session_cursor = (self.new_session_cursor + 1).min(self.new_session_path.len());
    }

    pub(crate) fn move_new_session_cursor_home(&mut self) {
        self.new_session_cursor = 0;
    }

    pub(crate) fn move_new_session_cursor_end(&mut self) {
        self.new_session_cursor = self.new_session_path.len();
    }

    pub(crate) fn new_session_backspace(&mut self) {
        if self.new_session_cursor > 0 && !self.new_session_path.is_empty() {
            let index = self.new_session_cursor - 1;
            self.new_session_path.remove(index);
            self.new_session_cursor = index;
        }
    }

    pub(crate) fn new_session_insert(&mut self, character: char) {
        self.new_session_path
            .insert(self.new_session_cursor, character);
        self.new_session_cursor += 1;
    }

    pub(crate) fn replace_new_session_completion(
        &mut self,
        query: String,
        results: Vec<FileIndexEntryLite>,
    ) {
        self.new_session_completion = Some(PathCompletionState {
            query,
            selected_index: 0,
            results,
        });
    }

    pub(crate) fn accept_new_session_completion(&mut self, path: String) {
        self.new_session_path = path;
        self.new_session_cursor = self.new_session_path.len();
        self.new_session_completion = None;
    }

    pub(crate) fn move_new_session_completion_selection(&mut self, delta: isize) {
        if let Some(completion) = self.new_session_completion.as_mut() {
            let len = completion.results.len();
            if len == 0 {
                completion.selected_index = 0;
                return;
            }
            completion.selected_index =
                (completion.selected_index as isize + delta).rem_euclid(len as isize) as usize;
        }
    }
}

fn session_matches(session: &SessionSummary, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    let matcher = SkimMatcherV2::default();
    matcher
        .fuzzy_match(session.title.as_deref().unwrap_or(""), query)
        .is_some()
        || matcher.fuzzy_match(&session.session_id, query).is_some()
}

fn matching_session_indices(group: &SessionGroup, query: &str) -> Vec<usize> {
    if query.is_empty() {
        return (0..group.sessions.len()).collect();
    }
    let matcher = SkimMatcherV2::default();
    let mut scored: Vec<(i64, usize)> = group
        .sessions
        .iter()
        .enumerate()
        .filter_map(|(index, session)| {
            let score = [
                matcher.fuzzy_match(session.title.as_deref().unwrap_or(""), query),
                matcher.fuzzy_match(&session.session_id, query),
            ]
            .into_iter()
            .flatten()
            .max();
            score.map(|score| (score, index))
        })
        .collect();
    scored.sort_by_key(|item| std::cmp::Reverse(item.0));
    scored.into_iter().map(|(_, index)| index).collect()
}

fn session_by_id_mut<'a>(
    sessions: &'a mut [SessionSummary],
    session_id: &str,
) -> Option<&'a mut SessionSummary> {
    for session in sessions {
        if session.session_id == session_id {
            return Some(session);
        }
        if let Some(child) = session_by_id_mut(&mut session.children, session_id) {
            return Some(child);
        }
    }
    None
}

fn fork_browsing_child(session: &SessionSummary) -> bool {
    session.fork_origin.as_deref() != Some("delegation")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(id: &str) -> SessionSummary {
        SessionSummary {
            session_id: id.into(),
            ..Default::default()
        }
    }

    fn group(cwd: &str, ids: &[&str]) -> SessionGroup {
        SessionGroup {
            cwd: Some(cwd.into()),
            sessions: ids.iter().map(|id| session(id)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn reducer_catalog_pages_preserve_request_order_and_remote_locations() {
        let mut state = SessionsState::new();
        state.session_discovery_in_progress = true;
        state.remember_remote_session_location("remote", "node-1", Some("/remote".into()));

        let outcome = state.reduce(SessionAction::CatalogPage {
            request: SessionListRequest::Discovery,
            page: SessionListPage {
                groups: vec![group("/first", &["remote"]), group("/second", &["local"])],
                next_cursor: Some("discovery-2".into()),
                total_count: Some(2),
            },
        });

        assert_eq!(
            outcome,
            SessionOutcome {
                coordination: Vec::new(),
                effects: vec![
                    Effect::Command(Command::list_sessions_workspace("/first".into())),
                    Effect::Command(Command::list_sessions_workspace("/second".into())),
                    Effect::Command(Command::list_sessions_discovery(Some("discovery-2".into()))),
                ],
            }
        );
        assert_eq!(state.remote_session_locations["remote"].node_id, "node-1");
        assert_eq!(
            state.remote_session_locations["remote"].cwd.as_deref(),
            Some("/remote")
        );

        let first_page = state.reduce(SessionAction::CatalogPage {
            request: SessionListRequest::WorkspaceFirstPage {
                cwd: "/first".into(),
            },
            page: SessionListPage {
                groups: vec![group("/first", &["first-page"])],
                ..Default::default()
            },
        });
        assert_eq!(first_page, SessionOutcome::default());
        let continuation = state.reduce(SessionAction::CatalogPage {
            request: SessionListRequest::WorkspaceContinuation {
                cwd: "/first".into(),
            },
            page: SessionListPage {
                groups: vec![group("/first", &["continued"])],
                ..Default::default()
            },
        });
        assert_eq!(continuation, SessionOutcome::default());
        assert_eq!(
            state.session_groups[0]
                .sessions
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["first-page", "continued"]
        );
    }

    #[test]
    fn reducer_catalog_failures_clear_only_the_requested_pending_state() {
        let mut state = SessionsState::new();
        state.session_discovery_in_progress = true;
        state
            .pending_session_group_loads
            .insert(Some("/repo".into()));
        state
            .pending_session_group_loads
            .insert(Some("/keep".into()));

        let discovery = state.reduce(SessionAction::CatalogFailed {
            request: SessionListRequest::Discovery,
            message: "discovery failed".into(),
        });
        assert!(!state.session_discovery_in_progress);
        assert_eq!(
            discovery,
            SessionOutcome {
                coordination: vec![
                    SessionCoordination::PushChatError("discovery failed".into()),
                    SessionCoordination::Status {
                        level: LogLevel::Error,
                        target: "acp",
                        message: "error: discovery failed".into(),
                    },
                ],
                effects: Vec::new(),
            }
        );

        let workspace = state.reduce(SessionAction::CatalogFailed {
            request: SessionListRequest::WorkspaceContinuation {
                cwd: "/repo".into(),
            },
            message: "workspace failed".into(),
        });
        assert!(
            !state
                .pending_session_group_loads
                .contains(&Some("/repo".into()))
        );
        assert!(
            state
                .pending_session_group_loads
                .contains(&Some("/keep".into()))
        );
        assert_eq!(workspace.coordination.len(), 2);
        assert!(workspace.effects.is_empty());
    }

    #[test]
    fn reducer_created_owns_active_ids_and_orders_external_coordination() {
        let mut state = SessionsState::new();
        state.mode_before_review = Some("plan".into());

        let outcome = state.reduce(SessionAction::Created {
            agent_id: "agent-1".into(),
            session_id: "session-1".into(),
            profile_id: Some("code".into()),
        });

        assert_eq!(state.session_id.as_deref(), Some("session-1"));
        assert_eq!(state.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(state.mode_before_review, None);
        assert_eq!(
            outcome,
            SessionOutcome {
                coordination: vec![
                    SessionCoordination::ClearDelegateParent,
                    SessionCoordination::ApplyProfileBinding {
                        session_id: "session-1".into(),
                        profile_id: Some("code".into()),
                        remove_if_missing: false,
                    },
                    SessionCoordination::ResetActiveSessionView,
                    SessionCoordination::SelectChat,
                    SessionCoordination::Status {
                        level: LogLevel::Info,
                        target: "session",
                        message: "session created".into(),
                    },
                    SessionCoordination::SynchronizeProfileCommands {
                        session_id: "session-1".into(),
                        root_only: false,
                    },
                ],
                effects: vec![Effect::Command(Command::SubscribeSession {
                    session_id: "session-1".into(),
                    agent_id: Some("agent-1".into()),
                })],
            }
        );
    }

    #[test]
    fn reducer_loaded_reports_parent_remote_binding_and_mode_prefix() {
        let mut child = session("child");
        child.parent_session_id = Some("parent".into());
        let mut state = SessionsState::new();
        state.agent_mode = "plan".into();
        state.mode_before_review = Some("build".into());
        state.session_groups = vec![SessionGroup {
            sessions: vec![child],
            ..Default::default()
        }];
        state.remember_remote_session_location("child", "node-1", Some("/remote".into()));

        let outcome = state.reduce(SessionAction::Loaded {
            agent_id: "agent-1".into(),
            session_id: "child".into(),
            profile_id: None,
        });

        assert_eq!(state.session_id.as_deref(), Some("child"));
        assert_eq!(state.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(state.mode_before_review, None);
        assert_eq!(
            outcome.coordination,
            vec![
                SessionCoordination::SetChatIdle,
                SessionCoordination::ResolveDelegateParent(Some("parent".into())),
                SessionCoordination::ApplyProfileBinding {
                    session_id: "child".into(),
                    profile_id: None,
                    remove_if_missing: true,
                },
                SessionCoordination::ResetActiveSessionView,
                SessionCoordination::SelectLoadedSessionScreen,
                SessionCoordination::Status {
                    level: LogLevel::Debug,
                    target: "activity",
                    message: "ready".into(),
                },
                SessionCoordination::SynchronizeProfileCommands {
                    session_id: "child".into(),
                    root_only: true,
                },
            ]
        );
        assert_eq!(
            outcome.effects,
            vec![Effect::Command(Command::SetAgentMode {
                mode: "plan".into(),
            })]
        );
    }

    #[test]
    fn reducer_loaded_marks_missing_local_profile_as_non_removing() {
        let mut state = SessionsState::new();
        let outcome = state.reduce(SessionAction::Loaded {
            agent_id: "agent-1".into(),
            session_id: "local".into(),
            profile_id: None,
        });
        assert!(matches!(
            outcome.coordination.get(2),
            Some(SessionCoordination::ApplyProfileBinding {
                session_id,
                profile_id: None,
                remove_if_missing: false,
            }) if session_id == "local"
        ));
    }

    #[test]
    fn reducer_agent_initialization_changes_only_identity_with_empty_outcome() {
        let mut state = SessionsState::new();
        state.session_id = Some("session-1".into());
        state.agent_id = Some("old-agent".into());
        state.agent_mode = "review".into();
        state.mode_before_review = Some("plan".into());
        state.session_filter = "keep".into();
        state.session_cursor = 4;
        state.new_session_path = "/keep".into();
        state.note_session_activity("session-1");
        state.remember_remote_session_location("remote-1", "node-1", Some("/remote/repo".into()));
        let activity_at = state.session_activity["session-1"].last_event_at;

        let outcome = state.reduce(SessionAction::InitializeAgentId("agent-1".into()));

        assert_eq!(outcome, SessionOutcome::default());
        assert_eq!(state.agent_id.as_deref(), Some("agent-1"));
        assert_eq!(state.session_id.as_deref(), Some("session-1"));
        assert_eq!(state.agent_mode, "review");
        assert_eq!(state.mode_before_review.as_deref(), Some("plan"));
        assert_eq!(state.session_filter, "keep");
        assert_eq!(state.session_cursor, 4);
        assert_eq!(state.new_session_path, "/keep");
        assert_eq!(
            state.session_activity["session-1"].last_event_at,
            activity_at
        );
        assert_eq!(state.session_remote_node_id("remote-1"), Some("node-1"));
        assert_eq!(
            state.session_remote_cwd("remote-1"),
            Some("/remote/repo".into())
        );
    }

    #[test]
    fn reducer_agent_mode_replacement_preserves_review_return_semantics() {
        let mut state = SessionsState::new();
        state.agent_mode = "review".into();
        state.mode_before_review = Some("plan".into());

        assert_eq!(
            state.reduce(SessionAction::ReplaceAgentMode("build".into())),
            SessionOutcome::default()
        );
        assert_eq!(state.agent_mode, "build");
        assert_eq!(state.mode_before_review, None);

        state.mode_before_review = Some("plan".into());
        assert_eq!(
            state.reduce(SessionAction::ReplaceAgentMode("review".into())),
            SessionOutcome::default()
        );
        assert_eq!(state.agent_mode, "review");
        assert_eq!(state.mode_before_review.as_deref(), Some("plan"));
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = SessionsState::new();
        assert!(state.session_groups.is_empty());
        assert_eq!(state.session_cursor, 0);
        assert!(state.session_filter.is_empty());
        assert_eq!(state.session_popup_tab, 0);
        assert!(state.collapsed_groups.is_empty());
        assert!(state.popup_collapsed_groups.is_empty());
        assert!(!state.session_discovery_in_progress);
        assert!(state.session_discovery_cursors.is_empty());
        assert!(state.pending_session_group_loads.is_empty());
        assert!(state.hydrated_session_groups.is_empty());
        assert!(state.expanded_session_children.is_empty());
        assert!(state.pending_session_child_loads.is_empty());
        assert_eq!(state.session_id, None);
        assert_eq!(state.agent_id, None);
        assert_eq!(state.agent_mode, "build");
        assert_eq!(state.mode_before_review, None);
        assert!(state.new_session_path.is_empty());
        assert_eq!(state.new_session_cursor, 0);
        assert_eq!(state.new_session_completion, None);
        assert!(state.session_activity.is_empty());
        assert!(state.remote_session_locations.is_empty());
    }

    #[test]
    fn discovery_preparation_rejects_pending_and_clears_only_catalog_state() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group("/repo", &["s1"])];
        state.session_discovery_cursors.insert("old".into());
        state.hydrated_session_groups.insert("/repo".into());
        state.expanded_session_children.insert("s1".into());
        state.session_id = Some("active".into());
        state
            .pending_session_group_loads
            .insert(Some("/repo".into()));

        assert!(!state.prepare_session_discovery());
        assert_eq!(state.session_groups.len(), 1);
        state.pending_session_group_loads.clear();
        assert!(state.prepare_session_discovery());
        assert!(state.session_groups.is_empty());
        assert!(state.session_discovery_cursors.is_empty());
        assert!(state.hydrated_session_groups.is_empty());
        assert!(state.session_discovery_in_progress);
        assert!(state.expanded_session_children.contains("s1"));
        assert_eq!(state.session_id.as_deref(), Some("active"));
    }

    #[test]
    fn group_and_child_request_preparation_record_pending_before_deduplication() {
        let mut state = SessionsState::new();
        let mut root = session("root");
        root.children_next_cursor = Some("child-cursor".into());
        state.session_groups = vec![SessionGroup {
            cwd: Some("/repo".into()),
            sessions: vec![root],
            next_cursor: Some("group-cursor".into()),
            ..Default::default()
        }];

        assert_eq!(
            state.prepare_session_group_page(0),
            Some(("/repo".into(), "group-cursor".into()))
        );
        assert_eq!(state.prepare_session_group_page(0), None);
        assert_eq!(
            state.prepare_session_child_page(0, &[0]),
            Some(("root".into(), Some("child-cursor".into())))
        );
        assert!(state.pending_session_child_loads.contains("root"));
    }

    #[test]
    fn merge_page_deduplicates_sorts_and_caps() {
        let mut existing = session("existing");
        existing.updated_at = Some("2025-01-01".into());
        let mut newer = session("newer");
        newer.updated_at = Some("2025-02-01".into());
        let mut group = SessionGroup {
            sessions: vec![existing.clone()],
            total_count: Some(9),
            ..Default::default()
        };

        SessionsState::merge_session_page(&mut group, vec![existing, newer], Some(1));
        assert_eq!(group.sessions.len(), 1);
        assert_eq!(group.sessions[0].session_id, "newer");
        assert_eq!(group.latest_activity.as_deref(), Some("2025-02-01"));
        assert_eq!(group.total_count, None);
    }

    #[test]
    fn zero_visible_rows_preserve_stale_cursor() {
        let mut state = SessionsState::new();
        state.session_cursor = 7;
        state.clamp_start_cursor();
        state.clamp_popup_cursor();
        assert_eq!(state.session_cursor, 7);
    }

    #[test]
    fn start_and_popup_collapses_are_independent() {
        let mut state = SessionsState::new();
        state.session_groups = vec![group("/repo", &["s1"])];
        state.toggle_group_collapse(Some("/repo"));
        assert_eq!(state.visible_start_items().len(), 1);
        assert_eq!(state.visible_popup_items().len(), 2);
        state.toggle_popup_group_collapse(Some("/repo"));
        assert_eq!(state.visible_popup_items().len(), 1);
    }

    #[test]
    fn start_and_popup_filter_transitions_reset_semantic_cursor() {
        let mut state = SessionsState::new();
        state.session_cursor = 4;
        state.start_filter_insert('x');
        assert_eq!(state.session_filter, "x");
        assert_eq!(state.session_cursor, 0);

        state.session_cursor = 4;
        state.popup_filter_backspace();
        assert_eq!(state.session_cursor, 0);
    }

    #[test]
    fn review_mode_transition_and_next_mode_preserve_return_mode() {
        let mut state = SessionsState::new();
        state.agent_mode = "plan".into();
        state.apply_mode_transition("review");
        assert_eq!(state.next_mode(), "plan");
        assert_eq!(state.mode_before_review.as_deref(), Some("plan"));
        state.apply_mode_transition("plan");
        assert_eq!(state.mode_before_review, None);
    }

    #[test]
    fn completion_selection_wraps_and_acceptance_clears_state() {
        let mut state = SessionsState::new();
        state.replace_new_session_completion(
            "repo".into(),
            vec![
                FileIndexEntryLite {
                    path: "/a".into(),
                    is_dir: true,
                },
                FileIndexEntryLite {
                    path: "/b".into(),
                    is_dir: true,
                },
            ],
        );
        state.move_new_session_completion_selection(-1);
        assert_eq!(
            state
                .new_session_completion
                .as_ref()
                .unwrap()
                .selected_index,
            1
        );
        state.accept_new_session_completion("/b/".into());
        assert_eq!(state.new_session_path, "/b/");
        assert_eq!(state.new_session_cursor, 3);
        assert_eq!(state.new_session_completion, None);
    }

    #[test]
    fn activity_window_excludes_only_current_session() {
        let mut state = SessionsState::new();
        state.session_id = Some("current".into());
        state.note_session_activity("current");
        state.note_session_activity("other");
        state.session_activity.insert(
            "stale".into(),
            SessionActivity {
                last_event_at: Instant::now() - Duration::from_secs(6),
            },
        );
        assert_eq!(state.active_session_count(), 2);
        assert_eq!(state.other_active_session_count(), 1);
    }

    #[test]
    fn discovery_pages_protect_hydrated_groups_and_dedupe_cursors() {
        let mut state = SessionsState::new();
        state.session_discovery_in_progress = true;
        state.hydrated_session_groups.insert("/repo".into());
        state.session_groups = vec![group("/repo", &["authoritative"])];

        let (workspaces, cursor) = state.apply_discovery_page(
            vec![group("/repo", &["preview"]), group("/later", &["later-1"])],
            Some("cursor-2".into()),
        );
        assert_eq!(workspaces, vec!["/later"]);
        assert_eq!(cursor.as_deref(), Some("cursor-2"));
        assert!(state.session_discovery_in_progress);
        assert_eq!(
            state
                .session_groups
                .iter()
                .find(|group| group.cwd.as_deref() == Some("/repo"))
                .unwrap()
                .sessions[0]
                .session_id,
            "authoritative"
        );

        let (_, duplicate_cursor) = state.apply_discovery_page(Vec::new(), Some("cursor-2".into()));
        assert_eq!(duplicate_cursor, None);
        assert!(!state.session_discovery_in_progress);
    }

    #[test]
    fn child_merge_replaces_then_appends_deduped_non_delegate_children() {
        let mut parent = session("parent");
        parent.children = vec![session("stale")];
        let mut state = SessionsState::new();
        state.session_groups = vec![SessionGroup {
            sessions: vec![parent],
            ..Default::default()
        }];
        state.pending_session_child_loads.insert("parent".into());
        let mut delegated = session("delegated");
        delegated.fork_origin = Some("delegation".into());

        state.merge_session_children(SessionChildrenPage {
            parent_session_id: "parent".into(),
            sessions: vec![session("child-1"), delegated, session("child-1")],
            next_cursor: Some("cursor-2".into()),
            total_count: Some(2),
        });
        assert_eq!(
            state
                .session_summary_by_id("parent")
                .unwrap()
                .children
                .iter()
                .map(|child| child.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["child-1"]
        );

        state.pending_session_child_loads.insert("parent".into());
        state.merge_session_children(SessionChildrenPage {
            parent_session_id: "parent".into(),
            sessions: vec![session("child-1"), session("child-2")],
            total_count: Some(2),
            ..Default::default()
        });
        assert_eq!(
            state
                .session_summary_by_id("parent")
                .unwrap()
                .children
                .iter()
                .map(|child| child.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["child-1", "child-2"]
        );
    }

    #[test]
    fn catalog_remote_metadata_precedes_remembered_location() {
        let mut remote = session("remote");
        remote.node_id = Some("catalog-node".into());
        remote.cwd = Some("/catalog".into());
        let mut state = SessionsState::new();
        state.session_groups = vec![SessionGroup {
            sessions: vec![remote],
            ..Default::default()
        }];
        state.remember_remote_session_location(
            "remote",
            "remembered-node",
            Some("/remembered".into()),
        );

        assert_eq!(state.session_remote_node_id("remote"), Some("catalog-node"));
        assert_eq!(state.session_remote_cwd("remote"), Some("/catalog".into()));
    }

    #[test]
    fn remote_node_only_refresh_replaces_node_and_preserves_cwd() {
        let mut state = SessionsState::new();
        state.remember_remote_session_location("remote", "node-1", Some("/repo".into()));
        state.remember_remote_session_location("remote", "node-2", Some("  ".into()));
        assert_eq!(state.session_remote_node_id("remote"), Some("node-2"));
        assert_eq!(state.session_remote_cwd("remote"), Some("/repo".into()));
    }
}
