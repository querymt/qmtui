use std::collections::HashSet;

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::input_layout::build_input_visual_layout;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileIndexEntryLite {
    pub(crate) path: String,
    pub(crate) is_dir: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct MentionState {
    pub(crate) trigger_start: usize,
    pub(crate) query: String,
    pub(crate) selected_index: usize,
    pub(crate) results: Vec<FileIndexEntryLite>,
}

#[derive(Debug, Clone)]
pub(crate) struct SlashCompletionState {
    /// The text typed after the leading `/` (e.g. `"mo"` while typing `/mo`).
    pub(crate) query: String,
    pub(crate) selected_index: usize,
    pub(crate) results: Vec<&'static crate::slash::SlashCommandDef>,
}

pub(crate) struct ComposerState {
    pub(crate) input: String,
    pub(crate) input_cursor: usize,
    pub(crate) input_preferred_col: Option<usize>,
    pub(crate) file_index: Vec<FileIndexEntryLite>,
    pub(crate) file_index_generated_at: Option<u64>,
    pub(crate) file_index_loading: bool,
    pub(crate) file_index_error: Option<String>,
    pub(crate) mention_state: Option<MentionState>,
    pub(crate) slash_state: Option<SlashCompletionState>,
}

impl ComposerState {
    pub(crate) fn new() -> Self {
        Self {
            input: String::new(),
            input_cursor: 0,
            input_preferred_col: None,
            file_index: Vec::new(),
            file_index_generated_at: None,
            file_index_loading: false,
            file_index_error: None,
            mention_state: None,
            slash_state: None,
        }
    }

    fn reset_input_preferred_col(&mut self) {
        self.input_preferred_col = None;
    }

    pub(crate) fn clear_input(&mut self) {
        self.input.clear();
        self.input_cursor = 0;
    }

    pub(crate) fn replace_input(&mut self, input: String) {
        self.input = input;
        self.input_cursor = self.input.len();
    }

    pub(crate) fn input_insert(&mut self, character: char) {
        self.reset_input_preferred_col();
        self.input.insert(self.input_cursor, character);
        self.input_cursor += character.len_utf8();
        self.refresh_mention_state();
        self.refresh_slash_state();
    }

    pub(crate) fn input_backspace(&mut self) {
        if self.input_cursor > 0 {
            self.reset_input_preferred_col();
            let previous = self.input[..self.input_cursor]
                .char_indices()
                .last()
                .map(|(index, _)| index)
                .unwrap_or(0);
            self.input.drain(previous..self.input_cursor);
            self.input_cursor = previous;
            self.refresh_mention_state();
            self.refresh_slash_state();
        }
    }

    pub(crate) fn input_delete(&mut self) {
        if self.input_cursor < self.input.len() {
            self.reset_input_preferred_col();
            let next = self.input[self.input_cursor..]
                .char_indices()
                .nth(1)
                .map(|(index, _)| self.input_cursor + index)
                .unwrap_or(self.input.len());
            self.input.drain(self.input_cursor..next);
            self.refresh_mention_state();
            self.refresh_slash_state();
        }
    }

    pub(crate) fn input_left(&mut self) {
        if self.input_cursor > 0 {
            self.reset_input_preferred_col();
            self.input_cursor = self.input[..self.input_cursor]
                .char_indices()
                .last()
                .map(|(index, _)| index)
                .unwrap_or(0);
            self.refresh_mention_state();
            self.refresh_slash_state();
        }
    }

    pub(crate) fn input_right(&mut self) {
        if self.input_cursor < self.input.len() {
            self.reset_input_preferred_col();
            self.input_cursor = self.input[self.input_cursor..]
                .char_indices()
                .nth(1)
                .map(|(index, _)| self.input_cursor + index)
                .unwrap_or(self.input.len());
            self.refresh_mention_state();
            self.refresh_slash_state();
        }
    }

    pub(crate) fn input_home(&mut self) {
        self.reset_input_preferred_col();
        self.input_cursor = 0;
        self.refresh_mention_state();
        self.refresh_slash_state();
    }

    pub(crate) fn input_end(&mut self) {
        self.reset_input_preferred_col();
        self.input_cursor = self.input.len();
        self.refresh_mention_state();
        self.refresh_slash_state();
    }

    pub(crate) fn input_up_visual(&mut self, line_width: usize, prefix_width: usize) {
        let layout =
            build_input_visual_layout(&self.input, self.input_cursor, line_width, prefix_width);
        if layout.cursor_row == 0 {
            self.input_preferred_col = Some(layout.cursor_text_col);
            return;
        }
        let preferred_col = self.input_preferred_col.unwrap_or(layout.cursor_text_col);
        self.input_cursor = layout.cursor_offset_for_row_col(layout.cursor_row - 1, preferred_col);
        self.input_preferred_col = Some(preferred_col);
        self.refresh_mention_state();
        self.refresh_slash_state();
    }

    pub(crate) fn input_down_visual(&mut self, line_width: usize, prefix_width: usize) {
        let layout =
            build_input_visual_layout(&self.input, self.input_cursor, line_width, prefix_width);
        if layout.cursor_row + 1 >= layout.total_rows() {
            self.input_preferred_col = Some(layout.cursor_text_col);
            return;
        }
        let preferred_col = self.input_preferred_col.unwrap_or(layout.cursor_text_col);
        self.input_cursor = layout.cursor_offset_for_row_col(layout.cursor_row + 1, preferred_col);
        self.input_preferred_col = Some(preferred_col);
        self.refresh_mention_state();
        self.refresh_slash_state();
    }

    pub(crate) fn active_mention_query_from(
        &self,
        input: &str,
        cursor: usize,
    ) -> Option<(usize, String)> {
        if cursor > input.len() || !input.is_char_boundary(cursor) {
            return None;
        }

        let before_cursor = &input[..cursor];
        let trigger_start = before_cursor.rfind('@')?;
        let prefix = &before_cursor[..trigger_start];
        if !prefix.is_empty() && !prefix.ends_with(char::is_whitespace) {
            return None;
        }

        let token = &before_cursor[trigger_start + 1..];
        if token.chars().any(char::is_whitespace) {
            return None;
        }

        Some((trigger_start, token.to_string()))
    }

    pub(crate) fn rank_file_matches(&self, query: &str) -> Vec<FileIndexEntryLite> {
        let matcher = SkimMatcherV2::default();
        let mut scored: Vec<(i64, bool, usize, &FileIndexEntryLite)> = self
            .file_index
            .iter()
            .filter_map(|entry| {
                let path = entry.path.as_str();
                let filename = path.rsplit('/').next().unwrap_or(path);
                let lower_path = path.to_lowercase();
                let lower_filename = filename.to_lowercase();
                let lower_query = query.to_lowercase();

                let mut score = matcher.fuzzy_match(path, query)?;
                if query.is_empty() {
                    score = 0;
                }
                if !query.is_empty() && lower_path.starts_with(&lower_query) {
                    score += 10_000;
                }
                if !query.is_empty() && lower_filename.starts_with(&lower_query) {
                    score += 7_500;
                }
                if !query.is_empty() && lower_path.contains(&lower_query) {
                    score += 3_000;
                }

                Some((score, entry.is_dir, path.len(), entry))
            })
            .collect();

        scored.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.3.path.cmp(&right.3.path))
        });

        scored
            .into_iter()
            .take(8)
            .map(|(_, _, _, entry)| entry.clone())
            .collect()
    }

    pub(crate) fn refresh_mention_state(&mut self) {
        let Some((trigger_start, query)) =
            self.active_mention_query_from(&self.input, self.input_cursor)
        else {
            self.mention_state = None;
            return;
        };

        let results = self.rank_file_matches(&query);
        self.mention_state = Some(MentionState {
            trigger_start,
            query,
            selected_index: 0,
            results,
        });
    }

    pub(crate) fn prepare_file_index_request(&mut self) -> bool {
        if self.mention_state.is_some() && self.file_index.is_empty() && !self.file_index_loading {
            self.file_index_loading = true;
            self.file_index_error = None;
            return true;
        }
        false
    }

    pub(crate) fn move_mention_selection(&mut self, delta: isize) {
        if let Some(mention) = self.mention_state.as_mut() {
            let len = mention.results.len();
            if len == 0 {
                mention.selected_index = 0;
                return;
            }
            mention.selected_index =
                (mention.selected_index as isize + delta).rem_euclid(len as isize) as usize;
        }
    }

    pub(crate) fn accept_selected_mention(&mut self) -> bool {
        let Some(mention) = self.mention_state.clone() else {
            return false;
        };
        let Some(selected) = mention.results.get(mention.selected_index).cloned() else {
            return false;
        };

        let replacement = format!("@{} ", selected.path);
        let replace_end = mention.trigger_start + 1 + mention.query.len();
        self.input
            .replace_range(mention.trigger_start..replace_end, &replacement);
        self.input_cursor = mention.trigger_start + replacement.len();
        self.mention_state = None;
        true
    }

    pub(crate) fn build_prompt_text_and_links(&self, input: &str) -> (String, Vec<String>) {
        let mut links = Vec::new();
        let mut seen = HashSet::new();
        let bytes = input.as_bytes();
        let mut index = 0usize;

        while index < bytes.len() {
            if bytes[index] == b'@' {
                let start = index + 1;
                let mut end = start;
                while end < bytes.len() {
                    let character = input[end..].chars().next().unwrap_or(' ');
                    if character.is_whitespace() {
                        break;
                    }
                    end += character.len_utf8();
                }
                if end > start {
                    let candidate = &input[start..end];
                    let looks_like_path = candidate.contains('/') || candidate.contains('.');
                    if looks_like_path && seen.insert(candidate.to_string()) {
                        links.push(candidate.to_string());
                    }
                }
                index = end.max(index + 1);
                continue;
            }
            index += 1;
        }

        (input.to_string(), links)
    }

    pub(crate) fn replace_input_from_editor(&mut self, input: String) {
        self.replace_input(input);
        self.refresh_mention_state();
        self.refresh_slash_state();
    }

    pub(crate) fn active_slash_query(&self) -> Option<String> {
        let end = self.input_cursor.min(self.input.len());
        let before_cursor = &self.input[..end];
        let after_slash = before_cursor.strip_prefix('/')?;
        if after_slash.chars().any(char::is_whitespace) {
            return None;
        }
        Some(after_slash.to_string())
    }

    pub(crate) fn refresh_slash_state(&mut self) {
        let Some(query) = self.active_slash_query() else {
            self.slash_state = None;
            return;
        };

        let query_lower = query.to_lowercase();
        let results: Vec<&'static crate::slash::SlashCommandDef> = crate::slash::SLASH_COMMANDS
            .iter()
            .filter(|command| command.name.starts_with(query_lower.as_str()))
            .collect();

        if results.is_empty() {
            self.slash_state = None;
            return;
        }

        let selected_index = self
            .slash_state
            .as_ref()
            .map(|state| state.selected_index.min(results.len().saturating_sub(1)))
            .unwrap_or(0);

        self.slash_state = Some(SlashCompletionState {
            query,
            selected_index,
            results,
        });
    }

    pub(crate) fn move_slash_selection(&mut self, delta: isize) {
        if let Some(state) = self.slash_state.as_mut() {
            let len = state.results.len();
            if len == 0 {
                state.selected_index = 0;
                return;
            }
            state.selected_index =
                (state.selected_index as isize + delta).rem_euclid(len as isize) as usize;
        }
    }

    pub(crate) fn accept_selected_slash_completion(&mut self) -> bool {
        let Some(state) = self.slash_state.clone() else {
            return false;
        };
        let Some(&command) = state.results.get(state.selected_index) else {
            self.slash_state = None;
            return false;
        };

        let end = self
            .input
            .find(char::is_whitespace)
            .unwrap_or(self.input.len());
        let replacement = format!("/{} ", command.name);
        self.input.replace_range(..end, &replacement);
        self.input_cursor = replacement.len();
        self.slash_state = None;
        true
    }

    pub(crate) fn reset_for_session_switch(&mut self) {
        self.file_index.clear();
        self.file_index_generated_at = None;
        self.file_index_loading = false;
        self.file_index_error = None;
        self.mention_state = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: impl Into<String>, is_dir: bool) -> FileIndexEntryLite {
        FileIndexEntryLite {
            path: path.into(),
            is_dir,
        }
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = ComposerState::new();

        assert!(state.input.is_empty());
        assert_eq!(state.input_cursor, 0);
        assert_eq!(state.input_preferred_col, None);
        assert!(state.file_index.is_empty());
        assert_eq!(state.file_index_generated_at, None);
        assert!(!state.file_index_loading);
        assert_eq!(state.file_index_error, None);
        assert!(state.mention_state.is_none());
        assert!(state.slash_state.is_none());
    }

    #[test]
    fn clear_and_replace_input_preserve_reset_boundaries() {
        let mut state = ComposerState::new();
        state.input = "old".into();
        state.input_cursor = 2;

        state.clear_input();
        assert!(state.input.is_empty());
        assert_eq!(state.input_cursor, 0);

        state.replace_input("new".into());
        assert_eq!(state.input, "new");
        assert_eq!(state.input_cursor, 3);
    }

    #[test]
    fn utf8_editing_and_horizontal_movement_use_byte_offsets() {
        let mut state = ComposerState::new();
        state.input = "éß".into();
        state.input_cursor = "é".len();

        state.input_insert('x');
        assert_eq!(state.input, "éxß");
        assert_eq!(state.input_cursor, "éx".len());
        state.input_backspace();
        assert_eq!(state.input, "éß");
        assert_eq!(state.input_cursor, "é".len());
        state.input_delete();
        assert_eq!(state.input, "é");
        state.input_left();
        assert_eq!(state.input_cursor, 0);
        state.input_right();
        assert_eq!(state.input_cursor, "é".len());
        state.input_home();
        assert_eq!(state.input_cursor, 0);
        state.input_end();
        assert_eq!(state.input_cursor, state.input.len());
    }

    #[test]
    fn visual_layout_wraps_long_lines_and_tracks_cursor() {
        let layout = build_input_visual_layout("abcdef", 4, 4, 2);
        assert_eq!(layout.total_rows(), 2);
        assert_eq!(layout.rows[0].text, "ab");
        assert_eq!(layout.rows[1].text, "cdef");
        assert_eq!((layout.cursor_row, layout.cursor_col), (1, 2));
    }

    #[test]
    fn visual_layout_preserves_hard_break_rows() {
        let layout = build_input_visual_layout("ab\ncd", 3, 6, 2);
        assert_eq!(layout.total_rows(), 2);
        assert_eq!(layout.rows[0].text, "ab");
        assert_eq!(layout.rows[1].text, "cd");
        assert_eq!((layout.cursor_row, layout.cursor_col), (1, 0));
    }

    #[test]
    fn visual_movement_crosses_wraps_and_newlines_and_horizontal_resets_preference() {
        let mut state = ComposerState::new();
        state.input = "abcdef".into();
        state.input_cursor = 4;

        state.input_up_visual(4, 2);
        assert_eq!(state.input_cursor, 2);
        assert_eq!(state.input_preferred_col, Some(2));
        state.input_down_visual(4, 2);
        assert_eq!(state.input_cursor, 4);

        state.input = "ab\ncd".into();
        state.input_cursor = 1;
        state.input_preferred_col = None;
        state.input_down_visual(6, 2);
        assert_eq!(state.input_cursor, 4);

        state.input_left();
        assert_eq!(state.input_cursor, 3);
        assert_eq!(state.input_preferred_col, None);
    }

    #[test]
    fn visual_movement_retains_preferred_column_across_unequal_rows_and_boundaries() {
        let mut state = ComposerState::new();
        state.input = "abcd\nx\nwxyz".into();
        state.input_cursor = state.input.len();

        state.input_up_visual(20, 2);
        assert_eq!(state.input_cursor, "abcd\nx".len());
        assert_eq!(state.input_preferred_col, Some(4));
        state.input_up_visual(20, 2);
        assert_eq!(state.input_cursor, 4);
        state.input_up_visual(20, 2);
        assert_eq!(state.input_cursor, 4);
        assert_eq!(state.input_preferred_col, Some(4));
        state.input_down_visual(20, 2);
        assert_eq!(state.input_cursor, "abcd\nx".len());
        state.input_down_visual(20, 2);
        assert_eq!(state.input_cursor, state.input.len());
        state.input_down_visual(20, 2);
        assert_eq!(state.input_cursor, state.input.len());
        assert_eq!(state.input_preferred_col, Some(4));
    }

    #[test]
    fn mention_query_detects_trigger_and_rejects_email() {
        let state = ComposerState::new();

        assert_eq!(
            state.active_mention_query_from("fix @src/ma", "fix @src/ma".len()),
            Some((4, "src/ma".into()))
        );
        assert_eq!(
            state.active_mention_query_from("email@test.com", "email@test.com".len()),
            None
        );
        assert_eq!(
            state.active_mention_query_from("foo @", 5),
            Some((4, String::new()))
        );
        assert_eq!(
            state.active_mention_query_from("foo @bar baz", 8),
            Some((4, "bar".into()))
        );
        assert_eq!(state.active_mention_query_from("foo @bar baz", 12), None);
    }

    #[test]
    fn mention_ranking_preserves_bonuses_and_caps_results_at_eight() {
        let mut state = ComposerState::new();
        state.file_index = vec![
            entry("src/main.rs", false),
            entry("tests/main_spec.rs", false),
            entry("src/manifest.toml", false),
            entry("src", true),
        ];
        let results = state.rank_file_matches("ma");
        let ranked: Vec<&str> = results.iter().map(|result| result.path.as_str()).collect();
        assert_eq!(ranked[0], "src/main.rs");
        assert!(ranked.contains(&"src/manifest.toml"));
        assert!(ranked.contains(&"tests/main_spec.rs"));

        state
            .file_index
            .extend((0..10).map(|index| entry(format!("other/main-{index}.rs"), false)));
        assert_eq!(state.rank_file_matches("ma").len(), 8);
    }

    #[test]
    fn empty_mention_query_ranks_directories_before_shorter_paths() {
        let mut state = ComposerState::new();
        state.file_index = vec![
            entry("long/file.rs", false),
            entry("z-dir", true),
            entry("a.rs", false),
        ];

        let ranked = state.rank_file_matches("");

        assert_eq!(ranked[0].path, "z-dir");
        assert_eq!(ranked[1].path, "a.rs");
    }

    #[test]
    fn mention_selection_wraps_and_empty_results_clamp_to_zero() {
        let mut state = ComposerState::new();
        state.mention_state = Some(MentionState {
            trigger_start: 0,
            query: String::new(),
            selected_index: 1,
            results: vec![entry("a.rs", false), entry("b.rs", false)],
        });

        state.move_mention_selection(1);
        assert_eq!(state.mention_state.as_ref().unwrap().selected_index, 0);
        state.move_mention_selection(-1);
        assert_eq!(state.mention_state.as_ref().unwrap().selected_index, 1);
        state.mention_state.as_mut().unwrap().results.clear();
        state.move_mention_selection(1);
        assert_eq!(state.mention_state.as_ref().unwrap().selected_index, 0);
    }

    #[test]
    fn mention_acceptance_and_resource_links_preserve_friendly_tokens() {
        let mut state = ComposerState::new();
        state.input = "open @src/ma now".into();
        state.input_cursor = "open @src/ma".len();
        state.file_index = vec![entry("src/main.rs", false)];
        state.refresh_mention_state();

        assert!(state.accept_selected_mention());
        assert_eq!(state.input, "open @src/main.rs  now");
        assert_eq!(state.input_cursor, "open @src/main.rs ".len());
        assert!(state.mention_state.is_none());

        let (text, links) = state
            .build_prompt_text_and_links("check @src/main.rs and @src/lib.rs then @src/main.rs");
        assert_eq!(text, "check @src/main.rs and @src/lib.rs then @src/main.rs");
        assert_eq!(links, vec!["src/main.rs", "src/lib.rs"]);
    }

    #[test]
    fn resource_links_skip_plain_mentions_and_deduplicate_paths() {
        let state = ComposerState::new();
        let (_, links) = state.build_prompt_text_and_links(
            "hello @person then @src/main.rs and @src/main.rs and @folder/file",
        );

        assert_eq!(links, vec!["src/main.rs", "folder/file"]);
    }

    #[test]
    fn file_index_request_preparation_is_one_shot() {
        let mut state = ComposerState::new();
        state.mention_state = Some(MentionState {
            trigger_start: 0,
            query: String::new(),
            selected_index: 0,
            results: Vec::new(),
        });
        state.file_index_error = Some("stale".into());

        assert!(state.prepare_file_index_request());
        assert!(state.file_index_loading);
        assert_eq!(state.file_index_error, None);
        assert!(!state.prepare_file_index_request());
    }

    #[test]
    fn file_index_request_requires_an_active_empty_index_mention() {
        let mut state = ComposerState::new();
        assert!(!state.prepare_file_index_request());

        state.mention_state = Some(MentionState {
            trigger_start: 0,
            query: String::new(),
            selected_index: 0,
            results: Vec::new(),
        });
        state.file_index.push(entry("src/main.rs", false));
        assert!(!state.prepare_file_index_request());
        assert!(!state.file_index_loading);
    }

    #[test]
    fn slash_query_filter_clamp_wrap_and_acceptance_preserve_behavior() {
        let mut state = ComposerState::new();
        state.input = "/".into();
        state.input_cursor = 1;
        assert_eq!(state.active_slash_query(), Some(String::new()));
        state.refresh_slash_state();
        let total = state.slash_state.as_ref().unwrap().results.len();
        state.slash_state.as_mut().unwrap().selected_index = total - 1;
        state.move_slash_selection(1);
        assert_eq!(state.slash_state.as_ref().unwrap().selected_index, 0);
        state.move_slash_selection(-1);
        assert_eq!(
            state.slash_state.as_ref().unwrap().selected_index,
            total - 1
        );

        state.input = "/mo".into();
        state.input_cursor = 3;
        state.refresh_slash_state();
        let slash_state = state.slash_state.as_ref().unwrap();
        assert!(
            slash_state
                .results
                .iter()
                .all(|command| command.name.starts_with("mo"))
        );
        assert!(slash_state.selected_index < slash_state.results.len());
        let model_index = slash_state
            .results
            .iter()
            .position(|command| command.name == "model")
            .unwrap();
        state.slash_state.as_mut().unwrap().selected_index = model_index;
        assert!(state.accept_selected_slash_completion());
        assert_eq!(state.input, "/model ");
        assert_eq!(state.input_cursor, "/model ".len());
        assert!(state.slash_state.is_none());

        state.input = "/zzz".into();
        state.input_cursor = 4;
        state.refresh_slash_state();
        assert!(state.slash_state.is_none());
        assert_eq!(state.active_slash_query(), Some("zzz".into()));
        state.input.push(' ');
        state.input_cursor += 1;
        assert_eq!(state.active_slash_query(), None);
    }

    #[test]
    fn slash_refresh_clamps_stale_selection() {
        let mut state = ComposerState::new();
        state.input = "/".into();
        state.input_cursor = 1;
        state.refresh_slash_state();
        state.slash_state.as_mut().unwrap().selected_index = usize::MAX;
        state.input = "/mo".into();
        state.input_cursor = 3;

        state.refresh_slash_state();

        let slash_state = state.slash_state.as_ref().unwrap();
        assert_eq!(slash_state.query, "mo");
        assert_eq!(
            slash_state.selected_index,
            slash_state.results.len().saturating_sub(1)
        );
    }

    #[test]
    fn slash_acceptance_without_state_returns_false() {
        let mut state = ComposerState::new();
        state.input = "/model".into();
        state.input_cursor = state.input.len();

        assert!(!state.accept_selected_slash_completion());
        assert_eq!(state.input, "/model");
    }

    #[test]
    fn editor_replacement_refreshes_completion_state() {
        let mut state = ComposerState::new();
        state.replace_input_from_editor("/mo".into());

        assert_eq!(state.input_cursor, 3);
        assert!(state.mention_state.is_none());
        assert!(state.slash_state.is_some());

        state.replace_input_from_editor("@src".into());
        assert!(state.mention_state.is_some());
        assert!(state.slash_state.is_none());
    }

    #[test]
    fn visual_movement_at_boundaries_only_records_preferred_column() {
        let mut state = ComposerState::new();
        state.input = "abc".into();
        state.input_cursor = 2;

        state.input_up_visual(10, 2);
        assert_eq!(state.input_cursor, 2);
        assert_eq!(state.input_preferred_col, Some(2));
        state.input_preferred_col = None;
        state.input_down_visual(10, 2);
        assert_eq!(state.input_cursor, 2);
        assert_eq!(state.input_preferred_col, Some(2));
    }

    #[test]
    fn session_switch_reset_clears_only_session_scoped_composer_data() {
        let mut state = ComposerState::new();
        state.input = "draft".into();
        state.input_cursor = 3;
        state.input_preferred_col = Some(4);
        state.file_index = vec![entry("src/main.rs", false)];
        state.file_index_generated_at = Some(7);
        state.file_index_loading = true;
        state.file_index_error = Some("error".into());
        state.mention_state = Some(MentionState {
            trigger_start: 0,
            query: "src".into(),
            selected_index: 0,
            results: Vec::new(),
        });
        state.input = "/mo".into();
        state.input_cursor = 3;
        state.refresh_slash_state();

        state.reset_for_session_switch();

        assert_eq!(state.input, "/mo");
        assert_eq!(state.input_cursor, 3);
        assert_eq!(state.input_preferred_col, Some(4));
        assert!(state.file_index.is_empty());
        assert_eq!(state.file_index_generated_at, None);
        assert!(!state.file_index_loading);
        assert_eq!(state.file_index_error, None);
        assert!(state.mention_state.is_none());
        assert!(state.slash_state.is_some());
    }
}
