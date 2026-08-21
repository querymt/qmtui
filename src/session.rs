use std::path::{Path, PathBuf};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::app::{App, FileIndexEntryLite};
use crate::domain::activity::DelegateEntry;
use crate::navigation_state::Popup;

impl App {
    pub fn apply_session_profile_binding(&mut self, session_id: &str, profile_id: Option<String>) {
        if let Some(profile_id) = profile_id {
            self.profiles
                .bind_session_profile(session_id.to_string(), profile_id);
        } else if self.sessions.is_remote_session_id(session_id) {
            self.profiles.remove_session_profile(session_id);
        }
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
        if let Some(active_session_id) = self.sessions.session_id.as_deref() {
            for group in &self.sessions.session_groups {
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
        self.navigation.popup = Popup::SessionSelect;
        self.sessions.session_popup_tab = 1;
        self.delegate_cursor = 0;
        self.delegate_filter.clear();
    }

    pub fn open_new_session_popup(&mut self) {
        self.navigation.popup = Popup::NewSession;
        self.sessions.new_session_path = self.resolve_new_session_default_cwd().unwrap_or_default();
        self.sessions.new_session_cursor = self.sessions.new_session_path.chars().count();
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
        let query = self.sessions.new_session_path.clone();
        let results = self.rank_path_completion_matches(&query);
        self.sessions.replace_new_session_completion(query, results);
    }

    pub fn accept_selected_new_session_completion(&mut self) -> bool {
        let Some(completion) = self.sessions.new_session_completion.clone() else {
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
        self.sessions.accept_new_session_completion(normalized);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::session::{SessionChildrenPage, SessionGroup, SessionSummary};

    fn session(id: &str) -> SessionSummary {
        SessionSummary {
            session_id: id.into(),
            ..Default::default()
        }
    }

    #[test]
    fn remote_session_location_preserves_cwd_across_node_only_refreshes() {
        let mut app = App::new();
        app.sessions.remember_remote_session_location(
            "remote-1",
            "node-1",
            Some("/remote/repo".into()),
        );
        app.sessions
            .remember_remote_session_node("remote-1", "node-2");

        assert_eq!(
            app.sessions.session_remote_node_id("remote-1"),
            Some("node-2")
        );
        assert_eq!(
            app.sessions.session_remote_cwd("remote-1"),
            Some("/remote/repo".into())
        );
    }

    #[test]
    fn current_profile_label_uses_binding_active_fallback_and_remote_fallback() {
        let mut app = App::new();
        app.profiles.profiles = vec![crate::domain::profile::ProfileInfo {
            id: "fast".into(),
            name: "Fast".into(),
            ..Default::default()
        }];
        app.profiles.active_profile_id = Some("fast".into());
        app.sessions.session_id = Some("local".into());

        assert_eq!(app.current_profile_label(), "Fast");

        app.profiles
            .bind_session_profile("local".into(), "unknown".into());
        assert_eq!(app.current_profile_label(), "unknown");

        app.sessions.session_id = Some("remote".into());
        app.sessions
            .remember_remote_session_node("remote", "node-1");
        assert_eq!(app.current_profile_label(), "remote");
    }

    #[test]
    fn profile_binding_some_inserts_and_overwrites() {
        let mut app = App::new();

        app.apply_session_profile_binding("session", Some("fast".into()));
        assert_eq!(app.profiles.session_profile_id("session"), Some("fast"));

        app.apply_session_profile_binding("session", Some("deep".into()));
        assert_eq!(app.profiles.session_profile_id("session"), Some("deep"));
    }

    #[test]
    fn missing_local_profile_preserves_existing_binding() {
        let mut app = App::new();
        app.profiles
            .bind_session_profile("local".into(), "fast".into());

        app.apply_session_profile_binding("local", None);

        assert_eq!(app.profiles.session_profile_id("local"), Some("fast"));
    }

    #[test]
    fn missing_remote_profile_removes_existing_binding() {
        let mut app = App::new();
        app.sessions
            .remember_remote_session_node("remote", "node-1");
        app.profiles
            .bind_session_profile("remote".into(), "fast".into());

        app.apply_session_profile_binding("remote", None);

        assert!(app.profiles.session_profile_id("remote").is_none());
    }

    #[test]
    fn session_parent_id_finds_explicit_parent_at_any_depth() {
        let mut app = App::new();
        let mut grandchild = session("grandchild");
        grandchild.parent_session_id = Some("child".into());
        let mut child = session("child");
        child.parent_session_id = Some("parent".into());
        child.children = vec![grandchild];
        let mut parent = session("parent");
        parent.children = vec![child];
        app.sessions.session_groups = vec![
            SessionGroup {
                cwd: Some("/unrelated".into()),
                sessions: vec![session("unrelated")],
                ..Default::default()
            },
            SessionGroup {
                cwd: Some("/repo".into()),
                sessions: vec![parent],
                ..Default::default()
            },
        ];

        assert_eq!(app.sessions.session_parent_id("child"), Some("parent"));
        assert_eq!(app.sessions.session_parent_id("grandchild"), Some("child"));
        assert_eq!(app.sessions.session_parent_id("unrelated"), None);
        assert_eq!(app.sessions.session_parent_id("missing"), None);
    }

    #[test]
    fn merge_session_children_replaces_then_appends_filtered_nested_children() {
        let mut app = App::new();
        let mut parent = session("parent");
        parent.children = vec![session("stale")];
        let mut root = session("root");
        root.children = vec![parent];
        app.sessions.session_groups = vec![SessionGroup {
            cwd: Some("/repo".into()),
            sessions: vec![root],
            ..Default::default()
        }];
        app.sessions
            .pending_session_child_loads
            .insert("parent".into());

        let mut remote = session("remote-child");
        remote.node_id = Some("node-1".into());
        remote.cwd = Some("/remote/repo".into());
        let mut delegated = session("delegated-child");
        delegated.fork_origin = Some("delegation".into());
        app.sessions.merge_session_children(SessionChildrenPage {
            parent_session_id: "parent".into(),
            sessions: vec![session("child-1"), remote, delegated, session("child-1")],
            next_cursor: Some("cursor-2".into()),
            total_count: Some(3),
        });

        let parent = app.sessions.session_summary_by_id("parent").unwrap();
        assert_eq!(
            parent
                .children
                .iter()
                .map(|child| child.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["child-1", "remote-child"]
        );
        assert_eq!(parent.children_next_cursor.as_deref(), Some("cursor-2"));
        assert_eq!(parent.children_total_count, Some(3));
        assert_eq!(parent.fork_count, 3);
        assert!(parent.has_children);
        assert_eq!(
            app.sessions.session_remote_node_id("remote-child"),
            Some("node-1")
        );
        assert_eq!(
            app.sessions.session_remote_cwd("remote-child"),
            Some("/remote/repo".into())
        );
        assert!(app.sessions.pending_session_child_loads.is_empty());

        app.sessions
            .pending_session_child_loads
            .insert("parent".into());
        app.sessions.merge_session_children(SessionChildrenPage {
            parent_session_id: "parent".into(),
            sessions: vec![session("remote-child"), session("child-2")],
            next_cursor: None,
            total_count: Some(3),
        });

        let parent = app.sessions.session_summary_by_id("parent").unwrap();
        assert_eq!(
            parent
                .children
                .iter()
                .map(|child| child.session_id.as_str())
                .collect::<Vec<_>>(),
            vec!["child-1", "remote-child", "child-2"]
        );
        assert_eq!(parent.children_next_cursor, None);
        assert!(app.sessions.pending_session_child_loads.is_empty());

        app.sessions
            .pending_session_child_loads
            .insert("missing-parent".into());
        app.sessions.merge_session_children(SessionChildrenPage {
            parent_session_id: "missing-parent".into(),
            ..Default::default()
        });
        assert!(app.sessions.pending_session_child_loads.is_empty());
    }
}
