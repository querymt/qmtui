use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::auth_state::{AuthPanel, AuthState};
use crate::domain::auth::{OAuthFlowKind, OAuthStatus};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum AuthInputResult {
    NotHandled,
    Moved,
    Filtered,
    Edited,
    SelectedProvider,
    ToggledApiKeyMask,
    ClosePopup,
    CloseDetail,
    BackToList,
    EnterApiKeyPanel,
    StartOAuth { provider: String },
    DisconnectOAuth { provider: String },
    ClearApiToken { provider: String },
    SetApiToken { provider: String, api_key: String },
    CopyOAuthUrl { provider: String, url: String },
    CompleteOAuth { flow_id: String, response: String },
}

pub(crate) fn handle_key(state: &mut AuthState, key: KeyEvent) -> AuthInputResult {
    match state.panel {
        AuthPanel::List => handle_list_key(state, key),
        AuthPanel::ApiKeyInput => handle_api_key(state, key),
        AuthPanel::OAuthFlow => handle_oauth_key(state, key),
    }
}

fn handle_list_key(state: &mut AuthState, key: KeyEvent) -> AuthInputResult {
    match key.code {
        KeyCode::Esc => {
            if state.selected.is_some() {
                state.close_detail();
                AuthInputResult::CloseDetail
            } else {
                AuthInputResult::ClosePopup
            }
        }
        KeyCode::Up => {
            let max = state.filtered_providers().len().saturating_sub(1);
            state.cursor = state.cursor.saturating_sub(1).min(max);
            AuthInputResult::Moved
        }
        KeyCode::Down => {
            let max = state.filtered_providers().len().saturating_sub(1);
            state.cursor = (state.cursor + 1).min(max);
            AuthInputResult::Moved
        }
        KeyCode::Enter => select_provider(state),
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            let Some(provider) = state
                .selected
                .and_then(|idx| state.providers.get(idx))
                .cloned()
            else {
                return AuthInputResult::NotHandled;
            };
            if provider.oauth_status == Some(OAuthStatus::Connected) {
                AuthInputResult::DisconnectOAuth {
                    provider: provider.provider,
                }
            } else if provider.has_stored_api_key {
                AuthInputResult::ClearApiToken {
                    provider: provider.provider,
                }
            } else {
                AuthInputResult::NotHandled
            }
        }
        KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            let Some((real_idx, supports_api_key)) =
                selected_filtered_provider(state).map(|(idx, provider)| {
                    (
                        idx,
                        provider.env_var_name.is_some() || provider.has_stored_api_key,
                    )
                })
            else {
                return AuthInputResult::NotHandled;
            };
            if !supports_api_key {
                return AuthInputResult::NotHandled;
            }
            state.ui_notice = None;
            enter_api_key_panel(state, real_idx);
            AuthInputResult::EnterApiKeyPanel
        }
        KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            let Some((real_idx, provider, supports_oauth)) = selected_filtered_provider(state)
                .map(|(idx, provider)| (idx, provider.provider.clone(), provider.supports_oauth))
            else {
                return AuthInputResult::NotHandled;
            };
            if !supports_oauth {
                return AuthInputResult::NotHandled;
            }
            state.ui_notice = None;
            state.selected = Some(real_idx);
            AuthInputResult::StartOAuth { provider }
        }
        KeyCode::Backspace => {
            state.filter.pop();
            state.cursor = 0;
            AuthInputResult::Filtered
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.filter.push(character);
            state.cursor = 0;
            AuthInputResult::Filtered
        }
        _ => AuthInputResult::NotHandled,
    }
}

fn select_provider(state: &mut AuthState) -> AuthInputResult {
    let Some((real_idx, provider)) = selected_filtered_provider(state).map(|(idx, provider)| {
        (
            idx,
            (
                provider.provider.clone(),
                provider.is_unconfigurable(),
                provider.is_api_key_only(),
                provider.is_oauth_only(),
                provider.oauth_status.clone(),
            ),
        )
    }) else {
        return AuthInputResult::NotHandled;
    };
    let (provider_id, unconfigurable, api_key_only, oauth_only, oauth_status) = provider;

    state.last_result = None;
    state.ui_notice = None;
    state.selected = Some(real_idx);
    if unconfigurable {
        AuthInputResult::SelectedProvider
    } else if api_key_only {
        enter_api_key_panel(state, real_idx);
        AuthInputResult::EnterApiKeyPanel
    } else if oauth_only && oauth_status != Some(OAuthStatus::Connected) {
        AuthInputResult::StartOAuth {
            provider: provider_id,
        }
    } else {
        AuthInputResult::SelectedProvider
    }
}

fn selected_filtered_provider(
    state: &AuthState,
) -> Option<(usize, &crate::domain::auth::AuthProviderEntry)> {
    state.filtered_providers().get(state.cursor).copied()
}

fn enter_api_key_panel(state: &mut AuthState, real_idx: usize) {
    state.selected = Some(real_idx);
    state.panel = AuthPanel::ApiKeyInput;
    state.api_key_input.clear();
    state.api_key_cursor = 0;
}

fn handle_api_key(state: &mut AuthState, key: KeyEvent) -> AuthInputResult {
    match key.code {
        KeyCode::Esc => {
            state.panel = AuthPanel::List;
            state.api_key_input.clear();
            state.api_key_cursor = 0;
            AuthInputResult::BackToList
        }
        KeyCode::Enter => {
            let Some(provider) = state
                .selected
                .and_then(|idx| state.providers.get(idx))
                .map(|provider| provider.provider.clone())
            else {
                return AuthInputResult::NotHandled;
            };
            let api_key = state.api_key_input.trim().to_string();
            if api_key.is_empty() {
                AuthInputResult::NotHandled
            } else {
                AuthInputResult::SetApiToken { provider, api_key }
            }
        }
        KeyCode::Tab => {
            state.api_key_masked = !state.api_key_masked;
            AuthInputResult::ToggledApiKeyMask
        }
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => state
            .selected
            .and_then(|idx| state.providers.get(idx))
            .map(|provider| AuthInputResult::ClearApiToken {
                provider: provider.provider.clone(),
            })
            .unwrap_or(AuthInputResult::NotHandled),
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.api_key_input.insert(state.api_key_cursor, character);
            state.api_key_cursor += character.len_utf8();
            AuthInputResult::Edited
        }
        KeyCode::Backspace if state.api_key_cursor > 0 => {
            remove_previous_char(&mut state.api_key_input, &mut state.api_key_cursor);
            AuthInputResult::Edited
        }
        KeyCode::Left if state.api_key_cursor > 0 => {
            move_cursor_left(&state.api_key_input, &mut state.api_key_cursor);
            AuthInputResult::Edited
        }
        KeyCode::Right if state.api_key_cursor < state.api_key_input.len() => {
            move_cursor_right(&state.api_key_input, &mut state.api_key_cursor);
            AuthInputResult::Edited
        }
        _ => AuthInputResult::NotHandled,
    }
}

fn handle_oauth_key(state: &mut AuthState, key: KeyEvent) -> AuthInputResult {
    match key.code {
        KeyCode::Esc => {
            state.oauth_flow = None;
            state.panel = AuthPanel::List;
            state.oauth_response.clear();
            state.oauth_response_cursor = 0;
            AuthInputResult::BackToList
        }
        KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            let Some((provider, url)) = state
                .oauth_flow
                .as_ref()
                .map(|flow| (flow.provider.clone(), flow.authorization_url.clone()))
            else {
                return AuthInputResult::NotHandled;
            };
            AuthInputResult::CopyOAuthUrl { provider, url }
        }
        KeyCode::Enter => {
            let Some(flow) = state.oauth_flow.as_ref() else {
                return AuthInputResult::NotHandled;
            };
            let is_device_poll = flow.flow_kind == OAuthFlowKind::DevicePoll;
            let response = if is_device_poll {
                String::new()
            } else {
                state.oauth_response.trim().to_string()
            };
            if !is_device_poll && response.is_empty() {
                return AuthInputResult::NotHandled;
            }
            AuthInputResult::CompleteOAuth {
                flow_id: flow.flow_id.clone(),
                response,
            }
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state
                .oauth_response
                .insert(state.oauth_response_cursor, character);
            state.oauth_response_cursor += character.len_utf8();
            AuthInputResult::Edited
        }
        KeyCode::Backspace if state.oauth_response_cursor > 0 => {
            remove_previous_char(&mut state.oauth_response, &mut state.oauth_response_cursor);
            AuthInputResult::Edited
        }
        KeyCode::Left if state.oauth_response_cursor > 0 => {
            move_cursor_left(&state.oauth_response, &mut state.oauth_response_cursor);
            AuthInputResult::Edited
        }
        KeyCode::Right if state.oauth_response_cursor < state.oauth_response.len() => {
            move_cursor_right(&state.oauth_response, &mut state.oauth_response_cursor);
            AuthInputResult::Edited
        }
        _ => AuthInputResult::NotHandled,
    }
}

fn remove_previous_char(input: &mut String, cursor: &mut usize) {
    let character = input[..*cursor].chars().next_back().unwrap();
    input.remove(*cursor - character.len_utf8());
    *cursor -= character.len_utf8();
}

fn move_cursor_left(input: &str, cursor: &mut usize) {
    let character = input[..*cursor].chars().next_back().unwrap();
    *cursor -= character.len_utf8();
}

fn move_cursor_right(input: &str, cursor: &mut usize) {
    let character = input[*cursor..].chars().next().unwrap();
    *cursor += character.len_utf8();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::auth::{AuthProviderEntry, OAuthFlow};

    fn provider(id: &str, oauth: bool, env_var: bool) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: id.into(),
            display_name: id.into(),
            oauth_status: oauth.then_some(OAuthStatus::NotAuthenticated),
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: env_var.then(|| format!("{}_KEY", id.to_uppercase())),
            supports_oauth: oauth,
            preferred_method: None,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::empty())
    }

    fn ctrl(character: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(character), KeyModifiers::CONTROL)
    }

    #[test]
    fn list_transitions_and_capability_gates_are_exact() {
        let mut state = AuthState::new();
        state.providers = vec![
            provider("none", false, false),
            provider("api", false, true),
            provider("oauth", true, false),
            provider("both", true, true),
        ];

        assert_eq!(
            handle_key(&mut state, ctrl('k')),
            AuthInputResult::NotHandled
        );
        assert_eq!(
            handle_key(&mut state, ctrl('o')),
            AuthInputResult::NotHandled
        );
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            AuthInputResult::SelectedProvider
        );
        assert_eq!(state.selected, Some(0));
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Esc)),
            AuthInputResult::CloseDetail
        );
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Esc)),
            AuthInputResult::ClosePopup
        );

        state.cursor = 1;
        assert_eq!(
            handle_key(&mut state, ctrl('k')),
            AuthInputResult::EnterApiKeyPanel
        );
        assert_eq!(state.panel, AuthPanel::ApiKeyInput);
        handle_key(&mut state, key(KeyCode::Esc));

        state.cursor = 2;
        assert_eq!(
            handle_key(&mut state, ctrl('o')),
            AuthInputResult::StartOAuth {
                provider: "oauth".into()
            }
        );
    }

    #[test]
    fn list_disconnect_prefers_oauth_then_stored_api_key() {
        let mut state = AuthState::new();
        let mut entry = provider("both", true, true);
        entry.oauth_status = Some(OAuthStatus::Connected);
        entry.has_stored_api_key = true;
        state.providers.push(entry);
        state.selected = Some(0);
        assert_eq!(
            handle_key(&mut state, ctrl('d')),
            AuthInputResult::DisconnectOAuth {
                provider: "both".into()
            }
        );

        state.providers[0].oauth_status = Some(OAuthStatus::NotAuthenticated);
        assert_eq!(
            handle_key(&mut state, ctrl('d')),
            AuthInputResult::ClearApiToken {
                provider: "both".into()
            }
        );
    }

    #[test]
    fn api_key_editing_uses_utf8_byte_offsets_and_normalizes_submit() {
        let mut state = AuthState::new();
        state.providers.push(provider("api", false, true));
        state.selected = Some(0);
        state.panel = AuthPanel::ApiKeyInput;

        assert_eq!(
            handle_key(&mut state, key(KeyCode::Char('é'))),
            AuthInputResult::Edited
        );
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Char('x'))),
            AuthInputResult::Edited
        );
        assert_eq!(state.api_key_cursor, 3);
        handle_key(&mut state, key(KeyCode::Left));
        assert_eq!(state.api_key_cursor, 2);
        handle_key(&mut state, key(KeyCode::Backspace));
        assert_eq!(state.api_key_input, "x");
        assert_eq!(state.api_key_cursor, 0);
        handle_key(&mut state, key(KeyCode::Right));
        assert_eq!(state.api_key_cursor, 1);

        state.api_key_input = "  secret  ".into();
        state.api_key_cursor = state.api_key_input.len();
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            AuthInputResult::SetApiToken {
                provider: "api".into(),
                api_key: "secret".into()
            }
        );
        state.api_key_input = "  ".into();
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            AuthInputResult::NotHandled
        );
    }

    #[test]
    fn oauth_copy_editing_and_completion_return_exact_payloads() {
        let mut state = AuthState::new();
        state.panel = AuthPanel::OAuthFlow;
        state.oauth_flow = Some(OAuthFlow {
            flow_id: "flow-1".into(),
            provider: "oauth".into(),
            authorization_url: "https://example.test/auth".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        });
        assert_eq!(
            handle_key(&mut state, ctrl('y')),
            AuthInputResult::CopyOAuthUrl {
                provider: "oauth".into(),
                url: "https://example.test/auth".into()
            }
        );
        handle_key(&mut state, key(KeyCode::Char('é')));
        handle_key(&mut state, key(KeyCode::Char('x')));
        assert_eq!(state.oauth_response_cursor, 3);
        handle_key(&mut state, key(KeyCode::Left));
        handle_key(&mut state, key(KeyCode::Backspace));
        assert_eq!(state.oauth_response, "x");
        assert_eq!(state.oauth_response_cursor, 0);

        state.oauth_response = "  code  ".into();
        state.oauth_response_cursor = state.oauth_response.len();
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            AuthInputResult::CompleteOAuth {
                flow_id: "flow-1".into(),
                response: "code".into()
            }
        );
        state.oauth_response = "  ".into();
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            AuthInputResult::NotHandled
        );

        state.oauth_flow.as_mut().unwrap().flow_kind = OAuthFlowKind::DevicePoll;
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Enter)),
            AuthInputResult::CompleteOAuth {
                flow_id: "flow-1".into(),
                response: String::new()
            }
        );
    }

    #[test]
    fn panel_escape_and_mask_toggle_preserve_local_transitions() {
        let mut state = AuthState::new();
        state.panel = AuthPanel::ApiKeyInput;
        state.api_key_input = "secret".into();
        state.api_key_cursor = 6;
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Tab)),
            AuthInputResult::ToggledApiKeyMask
        );
        assert!(!state.api_key_masked);
        assert_eq!(
            handle_key(&mut state, key(KeyCode::Esc)),
            AuthInputResult::BackToList
        );
        assert_eq!(state.panel, AuthPanel::List);
        assert!(state.api_key_input.is_empty());
    }
}
