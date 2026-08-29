use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::profiles_state::ProfilesState;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ProfileInputResult {
    NotHandled,
    Close,
    Moved,
    Filtered,
    SelectProfile { profile_id: String },
    NoMatchingProfile,
}

pub(crate) fn handle_key(state: &mut ProfilesState, key: KeyEvent) -> ProfileInputResult {
    match key.code {
        KeyCode::Esc => ProfileInputResult::Close,
        KeyCode::Up => {
            state.move_profile_cursor(-1);
            ProfileInputResult::Moved
        }
        KeyCode::Down => {
            state.move_profile_cursor(1);
            ProfileInputResult::Moved
        }
        KeyCode::Backspace => {
            state.profile_filter.pop();
            state.profile_cursor = 0;
            ProfileInputResult::Filtered
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.profile_filter.push(character);
            state.profile_cursor = 0;
            ProfileInputResult::Filtered
        }
        KeyCode::Enter => state
            .selected_profile()
            .map(|profile| ProfileInputResult::SelectProfile {
                profile_id: profile.id.clone(),
            })
            .unwrap_or(ProfileInputResult::NoMatchingProfile),
        _ => ProfileInputResult::NotHandled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::profile::ProfileInfo;

    fn profile(id: &str, name: &str) -> ProfileInfo {
        ProfileInfo {
            id: id.into(),
            name: name.into(),
            ..Default::default()
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    #[test]
    fn movement_and_filtering_are_owned_by_profile_state() {
        let mut state = ProfilesState::new();
        state.profiles = vec![profile("fast", "Fast"), profile("deep", "Deep")];

        assert_eq!(
            handle_key(&mut state, key(KeyCode::Down)),
            ProfileInputResult::Moved
        );
        assert_eq!(state.profile_cursor, 1);
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Char('f'))),
            ProfileInputResult::Filtered
        );
        assert_eq!(state.profile_filter, "f");
        assert_eq!(state.profile_cursor, 0);
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Backspace)),
            ProfileInputResult::Filtered
        );
        assert!(state.profile_filter.is_empty());
    }

    #[test]
    fn selection_returns_exact_id_and_empty_filters_report_no_match() {
        let mut state = ProfilesState::new();
        state.profiles = vec![profile("fast", "Fast"), profile("deep", "Deep")];
        state.profile_cursor = 1;
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            ProfileInputResult::SelectProfile {
                profile_id: "deep".into()
            }
        );

        state.profile_filter = "missing".into();
        state.profile_cursor = 0;
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            ProfileInputResult::NoMatchingProfile
        );

        state.profiles.clear();
        state.profile_filter.clear();
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            ProfileInputResult::NoMatchingProfile
        );
    }
}
