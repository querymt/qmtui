use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::app::*;
use crate::protocol::*;

/// Returns indices of sessions in `group` whose title or ID matches `q`.
/// When `q` is empty every session matches in original order.
/// When `q` is non-empty, results are sorted by fuzzy match score (best first).
fn session_matches(session: &SessionSummary, q: &str) -> bool {
    if q.is_empty() {
        return true;
    }
    let matcher = SkimMatcherV2::default();
    matcher
        .fuzzy_match(session.title.as_deref().unwrap_or(""), q)
        .is_some()
        || matcher.fuzzy_match(&session.session_id, q).is_some()
}

fn matching_session_indices(group: &SessionGroup, q: &str) -> Vec<usize> {
    if q.is_empty() {
        return (0..group.sessions.len()).collect();
    }
    let matcher = SkimMatcherV2::default();
    let mut scored: Vec<(i64, usize)> = group
        .sessions
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let score = [
                matcher.fuzzy_match(s.title.as_deref().unwrap_or(""), q),
                matcher.fuzzy_match(&s.session_id, q),
            ]
            .into_iter()
            .flatten()
            .max();
            score.map(|s| (s, i))
        })
        .collect();
    scored.sort_by_key(|item| std::cmp::Reverse(item.0));
    scored.into_iter().map(|(_, i)| i).collect()
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

impl App {
    pub fn session_summary_by_id(&self, session_id: &str) -> Option<&SessionSummary> {
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

    pub fn remember_remote_session_node(&mut self, session_id: &str, node_id: &str) {
        self.remote_session_nodes
            .insert(session_id.to_string(), node_id.to_string());
    }

    pub fn session_remote_node_id(&self, session_id: &str) -> Option<&str> {
        self.session_summary_by_id(session_id)
            .and_then(|session| session.node_id.as_deref())
            .or_else(|| {
                self.remote_session_nodes
                    .get(session_id)
                    .map(String::as_str)
            })
    }

    pub fn is_remote_session_id(&self, session_id: &str) -> bool {
        self.session_remote_node_id(session_id).is_some()
            || self
                .session_summary_by_id(session_id)
                .map(|session| session.node.is_some())
                .unwrap_or(false)
    }

    pub fn current_session_is_remote(&self) -> bool {
        self.session_id
            .as_deref()
            .map(|session_id| self.is_remote_session_id(session_id))
            .unwrap_or(false)
    }

    pub fn apply_session_profile_binding(&mut self, session_id: &str, profile_id: Option<String>) {
        if let Some(profile_id) = profile_id {
            self.session_profiles
                .insert(session_id.to_string(), profile_id);
        } else if self.is_remote_session_id(session_id) {
            self.session_profiles.remove(session_id);
        }
    }

    pub fn session_by_path(&self, group_idx: usize, path: &[usize]) -> Option<&SessionSummary> {
        let (first, rest) = path.split_first()?;
        let mut session = self.session_groups.get(group_idx)?.sessions.get(*first)?;
        for idx in rest {
            session = session.children.get(*idx)?;
        }
        Some(session)
    }

    pub fn session_by_path_mut(
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

    pub fn expandable_root_session(&self, group_idx: usize, path: &[usize]) -> bool {
        let Some(session) = self.session_by_path(group_idx, path) else {
            return false;
        };
        path.len() == 1
            && session.parent_session_id.is_none()
            && session.fork_count > 0
            && session.node.is_none()
            && session.node_id.is_none()
    }

    pub fn toggle_session_children(&mut self, group_idx: usize, path: &[usize]) -> bool {
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

    pub fn merge_session_children(&mut self, data: SessionChildrenData) {
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
            let remote_nodes: Vec<(String, String)> = sessions
                .iter()
                .filter_map(|session| {
                    Some((
                        session.session_id.clone(),
                        session.node_id.as_ref()?.clone(),
                    ))
                })
                .collect();
            for (session_id, node_id) in remote_nodes {
                self.remote_session_nodes.insert(session_id, node_id);
            }
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

    /// Flat list of sessions that match the current filter, across all groups.
    ///
    /// Used by the session popup (which shows a flat list) for backward compatibility.
    pub fn filtered_sessions(&self) -> Vec<&SessionSummary> {
        let q = self.session_filter.to_lowercase();
        self.session_groups
            .iter()
            .flat_map(|g| {
                matching_session_indices(g, &q)
                    .into_iter()
                    .map(move |i| &g.sessions[i])
            })
            .collect()
    }

    /// Build the flat list of visible rows for the start-page session list.
    ///
    /// Each call re-evaluates the current `session_filter` and `collapsed_groups`.
    /// Group headers are always included; session rows are included only when
    /// the group is expanded *and* the session matches the filter.
    /// Groups with zero matching sessions are omitted entirely when a filter is
    /// active.
    pub fn visible_start_items(&self) -> Vec<StartPageItem> {
        let q = self.session_filter.to_lowercase();
        let mut items = Vec::new();

        let groups_iter = self
            .session_groups
            .iter()
            .enumerate()
            .take(MAX_VISIBLE_GROUPS);

        for (group_idx, group) in groups_iter {
            let collapse_key = group.cwd.clone().unwrap_or_default();
            let collapsed = self.collapsed_groups.contains(&collapse_key);

            let matching = matching_session_indices(group, &q);

            // When a filter is active, skip groups with no matches entirely.
            if !q.is_empty() && matching.is_empty() {
                continue;
            }

            items.push(StartPageItem::GroupHeader {
                cwd: group.cwd.clone(),
                session_count: group.sessions.len(),
                session_total: group.total_count,
                collapsed,
            });

            if !collapsed {
                // Cap at MAX_RECENT_SESSIONS and append a ShowMore row if needed.
                let visible: Vec<usize> =
                    matching.iter().copied().take(MAX_RECENT_SESSIONS).collect();
                let hidden = matching.len().saturating_sub(MAX_RECENT_SESSIONS);

                for session_idx in visible {
                    let path = vec![session_idx];
                    items.push(StartPageItem::Session {
                        group_idx,
                        path: path.clone(),
                        depth: 0,
                    });
                    let session = &group.sessions[session_idx];
                    if self.expanded_session_children.contains(&session.session_id) {
                        for (child_idx, child) in session.children.iter().enumerate() {
                            if session_matches(child, &q) {
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

    /// Toggle the collapsed state of the group identified by `cwd`.
    ///
    /// `None` cwd is stored under the empty-string key so it can still be
    /// toggled independently.
    pub fn toggle_group_collapse(&mut self, cwd: Option<&str>) {
        let key = cwd.unwrap_or("").to_string();
        if !self.collapsed_groups.remove(&key) {
            self.collapsed_groups.insert(key);
        }
    }

    /// Toggle the collapsed state of a group *in the session popup*.
    ///
    /// Uses `popup_collapsed_groups` — fully independent of the start-page
    /// `collapsed_groups` so the two views never interfere.
    pub fn toggle_popup_group_collapse(&mut self, cwd: Option<&str>) {
        let key = cwd.unwrap_or("").to_string();
        if !self.popup_collapsed_groups.remove(&key) {
            self.popup_collapsed_groups.insert(key);
        }
    }

    /// Build the flat list of visible rows for the session popup.
    ///
    /// Mirrors [`visible_start_items`] but uses popup-local collapse state and
    /// appends a per-group load-more row when the backend has another page.
    pub fn visible_popup_items(&self) -> Vec<PopupItem> {
        let q = self.session_filter.to_lowercase();
        let mut items = Vec::new();

        for (group_idx, group) in self.session_groups.iter().enumerate() {
            let collapse_key = group.cwd.clone().unwrap_or_default();
            let collapsed = self.popup_collapsed_groups.contains(&collapse_key);

            let matching = matching_session_indices(group, &q);

            // When a filter is active, skip groups with no matches entirely.
            if !q.is_empty() && matching.is_empty() {
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
                            if session_matches(child, &q) {
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
                if group.next_cursor.is_some() {
                    items.push(PopupItem::LoadMore {
                        group_idx,
                        parent_path: Vec::new(),
                    });
                }
            }
        }

        items
    }

    /// Flat list of delegate entries that match `delegate_filter`.
    /// Built from the parent session's event stream (DelegationRequested /
    /// SessionForked / DelegationCompleted / DelegationFailed events).
    /// When the filter is empty every entry matches in original order.
    /// When the filter is non-empty, results are sorted by fuzzy match score (best first).
    pub fn visible_delegate_entries(&self) -> Vec<&DelegateEntry> {
        if self.delegate_filter.is_empty() {
            return self.delegate_entries.iter().collect();
        }
        let matcher = SkimMatcherV2::default();
        let q = self.delegate_filter.to_lowercase();
        let mut scored: Vec<(i64, &DelegateEntry)> = self
            .delegate_entries
            .iter()
            .filter_map(|e| {
                let score = [
                    matcher.fuzzy_match(&e.objective, &q),
                    matcher.fuzzy_match(&e.delegation_id, &q),
                    matcher.fuzzy_match(e.target_agent_id.as_deref().unwrap_or(""), &q),
                ]
                .into_iter()
                .flatten()
                .max();
                score.map(|s| (s, e))
            })
            .collect();
        scored.sort_by_key(|item| std::cmp::Reverse(item.0));
        scored.into_iter().map(|(_, e)| e).collect()
    }

    pub fn resolve_new_session_default_cwd(&self) -> Option<String> {
        if let Some(active_session_id) = self.session_id.as_deref() {
            for group in &self.session_groups {
                for session in &group.sessions {
                    if session.session_id == active_session_id {
                        if let Some(cwd) = session.cwd.as_ref().filter(|cwd| !cwd.trim().is_empty())
                        {
                            return Some(cwd.clone());
                        }
                        if let Some(cwd) = group.cwd.as_ref().filter(|cwd| !cwd.trim().is_empty()) {
                            return Some(cwd.clone());
                        }
                    }
                }
            }
        }

        self.launch_cwd
            .as_ref()
            .filter(|cwd| !cwd.trim().is_empty())
            .cloned()
    }

    /// Semantic alias for the active session's effective cwd.
    pub fn current_session_cwd(&self) -> Option<String> {
        self.resolve_new_session_default_cwd()
    }

    pub fn open_delegate_popup(&mut self) {
        self.popup = Popup::SessionSelect;
        self.session_popup_tab = 1;
        self.delegate_cursor = 0;
        self.delegate_filter.clear();
    }

    pub fn open_new_session_popup(&mut self) {
        self.popup = Popup::NewSession;
        self.new_session_path = self.resolve_new_session_default_cwd().unwrap_or_default();
        self.new_session_cursor = self.new_session_path.chars().count();
        self.refresh_new_session_completion();
    }

    pub fn new_session_base_dir(&self) -> PathBuf {
        self.launch_cwd
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn expand_user_path(&self, input: &str) -> PathBuf {
        if input == "~" {
            return dirs::home_dir().unwrap_or_else(|| PathBuf::from(input));
        }
        if let Some(rest) = input.strip_prefix("~/")
            && let Some(home) = dirs::home_dir()
        {
            return home.join(rest);
        }
        PathBuf::from(input)
    }

    fn normalize_lexical_path(&self, path: &Path) -> PathBuf {
        use std::path::Component;

        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
                Component::RootDir => normalized.push(Component::RootDir.as_os_str()),
                Component::CurDir => {}
                Component::ParentDir => {
                    if !normalized.pop() {
                        normalized.push(Component::RootDir.as_os_str());
                    }
                }
                Component::Normal(part) => normalized.push(part),
            }
        }
        normalized
    }

    pub fn normalize_new_session_path(&self, input: &str) -> Option<String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return self.resolve_new_session_default_cwd().map(|cwd| {
                self.normalize_lexical_path(&PathBuf::from(cwd))
                    .to_string_lossy()
                    .into_owned()
            });
        }

        let path = self.expand_user_path(trimmed);
        let absolute = if path.is_absolute() {
            path
        } else {
            self.new_session_base_dir().join(path)
        };
        Some(
            self.normalize_lexical_path(&absolute)
                .to_string_lossy()
                .into_owned(),
        )
    }

    pub fn collect_path_completion_candidates(&self, query: &str) -> Vec<FileIndexEntryLite> {
        let base_dir = self.new_session_base_dir();
        let typed = query.trim();
        let candidate_root = if typed.is_empty() {
            base_dir.clone()
        } else {
            let raw = PathBuf::from(typed);
            if raw.is_absolute() {
                raw.parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("/"))
            } else {
                let joined = base_dir.join(raw);
                joined
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or(base_dir.clone())
            }
        };

        let Ok(entries) = std::fs::read_dir(&candidate_root) else {
            return Vec::new();
        };

        let mut candidates = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
            if !is_dir {
                continue;
            }
            candidates.push(FileIndexEntryLite {
                path: path.to_string_lossy().into_owned(),
                is_dir,
            });
        }
        candidates
    }

    pub fn rank_path_completion_matches(&self, query: &str) -> Vec<FileIndexEntryLite> {
        let matcher = SkimMatcherV2::default();
        let mut scored: Vec<(i64, bool, usize, FileIndexEntryLite)> = self
            .collect_path_completion_candidates(query)
            .into_iter()
            .filter_map(|entry| {
                let path = entry.path.as_str();
                let filename = path.rsplit('/').next().unwrap_or(path);
                let lower_path = path.to_lowercase();
                let lower_filename = filename.to_lowercase();
                let lower_query = query.trim().to_lowercase();

                let mut score = if lower_query.is_empty() {
                    0
                } else {
                    matcher
                        .fuzzy_match(path, query.trim())
                        .or_else(|| matcher.fuzzy_match(filename, query.trim()))?
                };
                if !lower_query.is_empty() && lower_path.starts_with(&lower_query) {
                    score += 10_000;
                }
                if !lower_query.is_empty() && lower_filename.starts_with(&lower_query) {
                    score += 7_500;
                }
                if !lower_query.is_empty() && lower_path.contains(&lower_query) {
                    score += 3_000;
                }

                Some((score, entry.is_dir, path.len(), entry))
            })
            .collect();

        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.path.cmp(&b.3.path))
        });

        scored
            .into_iter()
            .take(6)
            .map(|(_, _, _, entry)| entry)
            .collect()
    }

    pub fn refresh_new_session_completion(&mut self) {
        let query = self.new_session_path.clone();
        let results = self.rank_path_completion_matches(&query);
        self.new_session_completion = Some(PathCompletionState {
            query,
            selected_index: 0,
            results,
        });
    }

    pub fn move_new_session_completion_selection(&mut self, delta: isize) {
        if let Some(completion) = self.new_session_completion.as_mut() {
            let len = completion.results.len();
            if len == 0 {
                completion.selected_index = 0;
                return;
            }
            let next =
                (completion.selected_index as isize + delta).rem_euclid(len as isize) as usize;
            completion.selected_index = next;
        }
    }

    pub fn accept_selected_new_session_completion(&mut self) -> bool {
        let Some(completion) = self.new_session_completion.clone() else {
            return false;
        };
        let Some(selected) = completion.results.get(completion.selected_index) else {
            return false;
        };

        let mut normalized = self
            .normalize_new_session_path(&selected.path)
            .unwrap_or_else(|| selected.path.clone());
        if selected.is_dir && !normalized.ends_with('/') {
            normalized.push('/');
        }
        self.new_session_path = normalized;
        self.new_session_cursor = self.new_session_path.len();
        self.new_session_completion = None;
        true
    }

    pub fn note_session_activity(&mut self, session_id: &str) {
        self.session_activity.insert(
            session_id.to_string(),
            SessionActivity {
                last_event_at: Instant::now(),
            },
        );
    }

    pub fn active_session_count(&self) -> usize {
        const ACTIVE_SESSION_WINDOW: Duration = Duration::from_secs(5);
        let now = Instant::now();
        self.session_activity
            .values()
            .filter(|activity| now.duration_since(activity.last_event_at) <= ACTIVE_SESSION_WINDOW)
            .count()
    }

    pub fn other_active_session_count(&self) -> usize {
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
}
