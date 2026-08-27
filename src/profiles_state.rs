use std::collections::HashMap;

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::domain::profile::ProfileInfo;

#[derive(Debug, Clone)]
pub(crate) enum ProfileAction {
    ReplaceCatalog {
        profiles: Vec<ProfileInfo>,
        backend_active_profile_id: Option<String>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProfileOutcome {
    pub(crate) active_profile_changed: bool,
}

pub(crate) struct ProfilesState {
    pub(crate) profiles: Vec<ProfileInfo>,
    pub(crate) active_profile_id: Option<String>,
    pub(crate) session_profiles: HashMap<String, String>,
    pub(crate) profile_cursor: usize,
    pub(crate) profile_filter: String,
}

impl ProfilesState {
    pub(crate) fn reduce(&mut self, action: ProfileAction) -> ProfileOutcome {
        match action {
            ProfileAction::ReplaceCatalog {
                profiles,
                backend_active_profile_id,
            } => ProfileOutcome {
                active_profile_changed: self.apply_catalog(profiles, backend_active_profile_id),
            },
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            profiles: Vec::new(),
            active_profile_id: None,
            session_profiles: HashMap::new(),
            profile_cursor: 0,
            profile_filter: String::new(),
        }
    }

    pub(crate) fn profile_by_id(&self, profile_id: &str) -> Option<&ProfileInfo> {
        self.profiles
            .iter()
            .find(|profile| profile.id == profile_id)
    }

    pub(crate) fn active_profile(&self) -> Option<&ProfileInfo> {
        self.active_profile_id
            .as_deref()
            .and_then(|profile_id| self.profile_by_id(profile_id))
    }

    pub(crate) fn profile_display_name(&self, profile_id: &str) -> String {
        self.profile_by_id(profile_id)
            .map(|profile| profile.name.clone())
            .unwrap_or_else(|| profile_id.to_string())
    }

    fn profile_label_or(
        &self,
        profile_id: Option<&str>,
        fallback: impl FnOnce() -> String,
    ) -> String {
        profile_id
            .map(|profile_id| self.profile_display_name(profile_id))
            .unwrap_or_else(fallback)
    }

    pub(crate) fn active_profile_label(&self) -> String {
        self.profile_label_or(self.active_profile_id.as_deref(), || "default".to_string())
    }

    pub(crate) fn filtered_profiles(&self) -> Vec<&ProfileInfo> {
        if self.profile_filter.is_empty() {
            self.profiles.iter().collect()
        } else {
            let matcher = SkimMatcherV2::default();
            let mut scored: Vec<(i64, &ProfileInfo)> = self
                .profiles
                .iter()
                .filter_map(|profile| {
                    let score = [
                        matcher.fuzzy_match(&profile.name, &self.profile_filter),
                        matcher.fuzzy_match(&profile.id, &self.profile_filter),
                        profile.description.as_deref().and_then(|description| {
                            matcher.fuzzy_match(description, &self.profile_filter)
                        }),
                    ]
                    .into_iter()
                    .flatten()
                    .max();
                    score.map(|score| (score, profile))
                })
                .collect();
            scored.sort_by_key(|item| std::cmp::Reverse(item.0));
            scored.into_iter().map(|(_, profile)| profile).collect()
        }
    }

    pub(crate) fn move_profile_cursor(&mut self, delta: isize) {
        let len = self.filtered_profiles().len();
        self.profile_cursor = if len == 0 {
            0
        } else {
            (self.profile_cursor as isize + delta).rem_euclid(len as isize) as usize
        };
    }

    pub(crate) fn selected_profile(&self) -> Option<&ProfileInfo> {
        self.filtered_profiles().get(self.profile_cursor).copied()
    }

    pub(crate) fn reset_for_open(&mut self) {
        self.profile_filter.clear();
        self.profile_cursor = self
            .active_profile_id
            .as_deref()
            .and_then(|active_id| {
                self.profiles
                    .iter()
                    .position(|profile| profile.id == active_id)
            })
            .unwrap_or(0);
    }

    pub(crate) fn find_profile_id(&self, query: &str) -> Option<String> {
        let needle = query.trim();
        if needle.is_empty() {
            return None;
        }
        self.profiles
            .iter()
            .find(|profile| profile.id == needle || profile.name.eq_ignore_ascii_case(needle))
            .map(|profile| profile.id.clone())
    }

    pub(crate) fn session_profile_id(&self, session_id: &str) -> Option<&str> {
        self.session_profiles.get(session_id).map(String::as_str)
    }

    pub(crate) fn session_profile(&self, session_id: &str) -> Option<&ProfileInfo> {
        self.session_profile_id(session_id)
            .and_then(|profile_id| self.profile_by_id(profile_id))
    }

    pub(crate) fn bind_session_profile(
        &mut self,
        session_id: String,
        profile_id: String,
    ) -> Option<String> {
        self.session_profiles.insert(session_id, profile_id)
    }

    pub(crate) fn remove_session_profile(&mut self, session_id: &str) -> Option<String> {
        self.session_profiles.remove(session_id)
    }

    pub(crate) fn apply_catalog(
        &mut self,
        profiles: Vec<ProfileInfo>,
        backend_active_profile_id: Option<String>,
    ) -> bool {
        let previous_profile_id = self.active_profile_id.clone();
        let is_available =
            |profile_id: &str| profiles.iter().any(|profile| profile.id == profile_id);
        let selected = self
            .active_profile_id
            .take()
            .filter(|profile_id| is_available(profile_id));
        let backend_default =
            backend_active_profile_id.filter(|profile_id| is_available(profile_id));
        self.active_profile_id = selected
            .or(backend_default)
            .or_else(|| profiles.first().map(|profile| profile.id.clone()));
        self.profiles = profiles;
        if self.profile_cursor >= self.profiles.len() {
            self.profile_cursor = self.profiles.len().saturating_sub(1);
        }
        self.active_profile_id != previous_profile_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(id: &str, name: &str, description: Option<&str>) -> ProfileInfo {
        ProfileInfo {
            id: id.into(),
            name: name.into(),
            description: description.map(str::to_string),
            ..Default::default()
        }
    }

    fn ids(profiles: Vec<&ProfileInfo>) -> Vec<&str> {
        profiles
            .into_iter()
            .map(|profile| profile.id.as_str())
            .collect()
    }

    #[test]
    fn reducer_catalog_preserves_selection_precedence_and_reports_change() {
        let mut state = ProfilesState::new();
        state.active_profile_id = Some("deep".into());

        let outcome = state.reduce(ProfileAction::ReplaceCatalog {
            profiles: vec![profile("fast", "Fast", None), profile("deep", "Deep", None)],
            backend_active_profile_id: Some("fast".into()),
        });
        assert_eq!(outcome, ProfileOutcome::default());
        assert_eq!(state.active_profile_id.as_deref(), Some("deep"));

        state.active_profile_id = Some("removed".into());
        let outcome = state.reduce(ProfileAction::ReplaceCatalog {
            profiles: vec![profile("fast", "Fast", None), profile("deep", "Deep", None)],
            backend_active_profile_id: Some("deep".into()),
        });
        assert_eq!(
            outcome,
            ProfileOutcome {
                active_profile_changed: true,
            }
        );
        assert_eq!(state.active_profile_id.as_deref(), Some("deep"));
    }

    #[test]
    fn reducer_empty_catalog_clears_selection_without_unowned_normalization() {
        let mut state = ProfilesState::new();
        state.profiles = vec![profile("old", "Old", None)];
        state.active_profile_id = Some("old".into());
        state.profile_cursor = 4;
        state.profile_filter = "keep".into();
        state.bind_session_profile("session".into(), "old".into());

        let outcome = state.reduce(ProfileAction::ReplaceCatalog {
            profiles: Vec::new(),
            backend_active_profile_id: None,
        });

        assert!(outcome.active_profile_changed);
        assert!(state.profiles.is_empty());
        assert!(state.active_profile_id.is_none());
        assert_eq!(state.profile_cursor, 0);
        assert_eq!(state.profile_filter, "keep");
        assert_eq!(state.session_profile_id("session"), Some("old"));
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = ProfilesState::new();

        assert!(state.profiles.is_empty());
        assert!(state.active_profile_id.is_none());
        assert!(state.session_profiles.is_empty());
        assert_eq!(state.profile_cursor, 0);
        assert!(state.profile_filter.is_empty());
    }

    #[test]
    fn lookup_active_profile_and_labels_preserve_unknown_ids() {
        let mut state = ProfilesState::new();
        state.profiles = vec![profile("fast", "Fast", None)];
        state.active_profile_id = Some("fast".into());

        assert_eq!(
            state.profile_by_id("fast").map(|p| p.name.as_str()),
            Some("Fast")
        );
        assert!(state.profile_by_id("FAST").is_none());
        assert_eq!(state.active_profile().map(|p| p.id.as_str()), Some("fast"));
        assert_eq!(state.profile_display_name("fast"), "Fast");
        assert_eq!(state.profile_display_name("missing"), "missing");
        assert_eq!(state.active_profile_label(), "Fast");

        state.active_profile_id = Some("missing".into());
        assert!(state.active_profile().is_none());
        assert_eq!(state.active_profile_label(), "missing");
        state.active_profile_id = None;
        assert_eq!(state.active_profile_label(), "default");
    }

    #[test]
    fn filtering_matches_name_id_and_description() {
        let mut state = ProfilesState::new();
        state.profiles = vec![
            profile("fast-id", "Quick", Some("low latency")),
            profile("deep-id", "Reasoner", Some("thorough analysis")),
        ];

        state.profile_filter = "Quick".into();
        assert_eq!(ids(state.filtered_profiles()), vec!["fast-id"]);
        state.profile_filter = "deep-id".into();
        assert_eq!(ids(state.filtered_profiles()), vec!["deep-id"]);
        state.profile_filter = "latency".into();
        assert_eq!(ids(state.filtered_profiles()), vec!["fast-id"]);
    }

    #[test]
    fn filtering_uses_max_score_descending_with_stable_ties() {
        let mut state = ProfilesState::new();
        state.profiles = vec![
            profile("first", "Common", Some("shared")),
            profile("second", "Common", Some("shared")),
            profile("common", "Other", None),
        ];
        state.profile_filter = "common".into();

        assert_eq!(
            ids(state.filtered_profiles()),
            vec!["common", "first", "second"]
        );
    }

    #[test]
    fn filtering_excludes_profile_metadata_and_preserves_catalog_order_when_empty() {
        let mut state = ProfilesState::new();
        let mut first = profile("first", "One", None);
        first.tags = vec!["metadata-query".into()];
        first.source = Some("metadata-query".into());
        first.config_kind = Some("metadata-query".into());
        first.fingerprint = Some("metadata-query".into());
        state.profiles = vec![first, profile("second", "Two", None)];

        assert_eq!(ids(state.filtered_profiles()), vec!["first", "second"]);
        state.profile_filter = "metadata-query".into();
        assert!(state.filtered_profiles().is_empty());
    }

    #[test]
    fn cursor_wraps_selects_filtered_rows_and_resets_when_empty() {
        let mut state = ProfilesState::new();
        state.profiles = vec![
            profile("one", "One", None),
            profile("two", "Two", None),
            profile("three", "Three", None),
        ];

        state.move_profile_cursor(-1);
        assert_eq!(state.profile_cursor, 2);
        assert_eq!(
            state.selected_profile().map(|p| p.id.as_str()),
            Some("three")
        );
        state.move_profile_cursor(1);
        assert_eq!(state.profile_cursor, 0);

        state.profile_filter = "Two".into();
        assert_eq!(state.selected_profile().map(|p| p.id.as_str()), Some("two"));
        state.profile_filter = "missing".into();
        state.profile_cursor = 9;
        state.move_profile_cursor(1);
        assert_eq!(state.profile_cursor, 0);
        assert!(state.selected_profile().is_none());
    }

    #[test]
    fn reset_for_open_clears_filter_and_uses_raw_active_position() {
        let mut state = ProfilesState::new();
        state.profiles = vec![
            profile("one", "One", None),
            profile("two", "Two", None),
            profile("three", "Three", None),
        ];
        state.active_profile_id = Some("three".into());
        state.profile_filter = "One".into();
        state.profile_cursor = 7;

        state.reset_for_open();

        assert!(state.profile_filter.is_empty());
        assert_eq!(state.profile_cursor, 2);

        state.active_profile_id = Some("missing".into());
        state.profile_filter = "again".into();
        state.reset_for_open();
        assert!(state.profile_filter.is_empty());
        assert_eq!(state.profile_cursor, 0);
    }

    #[test]
    fn catalog_selection_precedence_and_change_result_are_exact() {
        let mut state = ProfilesState::new();
        state.active_profile_id = Some("deep".into());

        assert!(!state.apply_catalog(
            vec![profile("fast", "Fast", None), profile("deep", "Deep", None)],
            Some("fast".into()),
        ));
        assert_eq!(state.active_profile_id.as_deref(), Some("deep"));

        state.active_profile_id = Some("removed".into());
        assert!(state.apply_catalog(
            vec![profile("fast", "Fast", None), profile("deep", "Deep", None)],
            Some("deep".into()),
        ));
        assert_eq!(state.active_profile_id.as_deref(), Some("deep"));

        state.active_profile_id = Some("removed-again".into());
        assert!(state.apply_catalog(
            vec![profile("fast", "Fast", None), profile("deep", "Deep", None)],
            Some("missing".into()),
        ));
        assert_eq!(state.active_profile_id.as_deref(), Some("fast"));
    }

    #[test]
    fn catalog_empty_clamps_cursor_and_preserves_filter_and_session_map() {
        let mut state = ProfilesState::new();
        state.profiles = vec![profile("old", "Old", None)];
        state.active_profile_id = Some("old".into());
        state.profile_cursor = 4;
        state.profile_filter = "keep".into();
        state.bind_session_profile("session".into(), "old".into());

        assert!(state.apply_catalog(Vec::new(), None));

        assert!(state.profiles.is_empty());
        assert!(state.active_profile_id.is_none());
        assert_eq!(state.profile_cursor, 0);
        assert_eq!(state.profile_filter, "keep");
        assert_eq!(state.session_profile_id("session"), Some("old"));
    }

    #[test]
    fn catalog_clamps_cursor_to_last_entry_without_other_normalization() {
        let mut state = ProfilesState::new();
        state.profile_cursor = 8;

        state.apply_catalog(
            vec![profile("one", "One", None), profile("two", "Two", None)],
            None,
        );

        assert_eq!(state.profile_cursor, 1);
        assert_eq!(state.active_profile_id.as_deref(), Some("one"));
    }

    #[test]
    fn find_profile_id_trims_and_uses_exact_id_or_ascii_insensitive_name() {
        let mut state = ProfilesState::new();
        state.profiles = vec![
            profile("UPPER", "Duplicate", None),
            profile("lower", "duplicate", None),
            profile("exact", "Other", None),
        ];

        assert_eq!(state.find_profile_id("  exact  ").as_deref(), Some("exact"));
        assert_eq!(state.find_profile_id("duplicate").as_deref(), Some("UPPER"));
        assert!(state.find_profile_id("upper").is_none());
        assert!(state.find_profile_id("   ").is_none());
    }

    #[test]
    fn session_map_insert_overwrite_lookup_profile_and_remove() {
        let mut state = ProfilesState::new();
        state.profiles = vec![profile("fast", "Fast", None), profile("deep", "Deep", None)];

        assert!(
            state
                .bind_session_profile("session".into(), "fast".into())
                .is_none()
        );
        assert_eq!(state.session_profile_id("session"), Some("fast"));
        assert_eq!(
            state
                .session_profile("session")
                .map(|profile| profile.name.as_str()),
            Some("Fast")
        );
        assert_eq!(
            state.bind_session_profile("session".into(), "deep".into()),
            Some("fast".into())
        );
        assert_eq!(state.session_profile_id("session"), Some("deep"));
        assert_eq!(state.remove_session_profile("session"), Some("deep".into()));
        assert!(state.session_profile_id("session").is_none());
        assert!(state.remove_session_profile("missing").is_none());
    }
}
