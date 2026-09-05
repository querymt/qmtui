use crate::application::Effect;
use crate::command::Command;
use crate::diagnostics::LogLevel;
use crate::domain::auth::{AuthProviderEntry, OAuthFlow, OAuthResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AuthUiNotice {
    pub(crate) provider: Option<String>,
    pub(crate) success: bool,
    pub(crate) message: String,
}

#[derive(Debug, Clone)]
pub(crate) enum AuthAction {
    Providers(Vec<AuthProviderEntry>),
    OAuthFlowStarted(OAuthFlow),
    OAuthResult(OAuthResult),
    ClipboardFinished { provider: String, success: bool },
    ClearUiNotice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AuthCoordination {
    pub(crate) level: LogLevel,
    pub(crate) message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AuthOutcome {
    pub(crate) diagnostics: Vec<AuthCoordination>,
    pub(crate) effects: Vec<Effect>,
}

/// Which sub-panel is active in the provider auth popup.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum AuthPanel {
    /// Browsing the provider list.
    #[default]
    List,
    /// Editing an API key for the selected provider.
    ApiKeyInput,
    /// Active OAuth flow showing URL and callback/device-poll input.
    OAuthFlow,
}

pub(crate) struct AuthState {
    pub(crate) providers: Vec<AuthProviderEntry>,
    pub(crate) cursor: usize,
    pub(crate) filter: String,
    pub(crate) selected: Option<usize>,
    pub(crate) panel: AuthPanel,
    pub(crate) api_key_input: String,
    pub(crate) api_key_cursor: usize,
    pub(crate) api_key_masked: bool,
    pub(crate) oauth_flow: Option<OAuthFlow>,
    pub(crate) oauth_response: String,
    pub(crate) oauth_response_cursor: usize,
    pub(crate) last_result: Option<OAuthResult>,
    pub(crate) ui_notice: Option<AuthUiNotice>,
    pub(crate) clipboard_fallback: Option<String>,
}

impl AuthState {
    pub(crate) fn reduce(&mut self, action: AuthAction) -> AuthOutcome {
        match action {
            AuthAction::Providers(providers) => {
                let selected_provider = self
                    .selected
                    .and_then(|index| self.providers.get(index))
                    .map(|provider| provider.provider.clone());
                self.providers = providers;
                self.selected = selected_provider.as_ref().and_then(|selected| {
                    self.providers
                        .iter()
                        .position(|provider| provider.provider == *selected)
                });
                let selected_was_removed = selected_provider.is_some() && self.selected.is_none();
                let panel_is_compatible = match self.panel {
                    AuthPanel::List => true,
                    AuthPanel::ApiKeyInput => self
                        .selected
                        .and_then(|index| self.providers.get(index))
                        .is_some_and(|provider| {
                            provider.env_var_name.is_some() || provider.has_stored_api_key
                        }),
                    AuthPanel::OAuthFlow => self
                        .selected
                        .and_then(|index| self.providers.get(index))
                        .is_some_and(|provider| {
                            provider.supports_oauth
                                && self
                                    .oauth_flow
                                    .as_ref()
                                    .is_none_or(|flow| flow.provider == provider.provider)
                        }),
                };
                if selected_was_removed || !panel_is_compatible {
                    self.clear_detail_state();
                }
                self.cursor = self
                    .cursor
                    .min(self.filtered_providers().len().saturating_sub(1));
                AuthOutcome {
                    diagnostics: vec![AuthCoordination {
                        level: LogLevel::Debug,
                        message: format!("{} auth provider(s)", self.providers.len()),
                    }],
                    effects: Vec::new(),
                }
            }
            AuthAction::OAuthFlowStarted(flow) => {
                let provider = flow.provider.clone();
                self.begin_oauth_flow(flow);
                AuthOutcome {
                    diagnostics: vec![AuthCoordination {
                        level: LogLevel::Info,
                        message: format!("OAuth flow started for {provider}"),
                    }],
                    effects: Vec::new(),
                }
            }
            AuthAction::OAuthResult(result) => {
                let is_success = result.is_success();
                let level = if is_success {
                    LogLevel::Info
                } else {
                    LogLevel::Warn
                };
                let message = result.message.clone();
                let applied_success = self.apply_oauth_result(result);
                debug_assert_eq!(applied_success, is_success);
                AuthOutcome {
                    diagnostics: vec![AuthCoordination { level, message }],
                    effects: vec![Effect::Command(Command::ListAuthProviders)],
                }
            }
            AuthAction::ClipboardFinished { provider, success } => {
                if success {
                    self.ui_notice = Some(AuthUiNotice {
                        provider: Some(provider),
                        success: true,
                        message: "Copied to clipboard".into(),
                    });
                    self.clipboard_fallback = None;
                } else {
                    self.ui_notice = None;
                    self.clipboard_fallback = self
                        .oauth_flow
                        .as_ref()
                        .map(|flow| flow.authorization_url.clone());
                }
                AuthOutcome::default()
            }
            AuthAction::ClearUiNotice => {
                self.ui_notice = None;
                AuthOutcome::default()
            }
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            providers: Vec::new(),
            cursor: 0,
            filter: String::new(),
            selected: None,
            panel: AuthPanel::List,
            api_key_input: String::new(),
            api_key_cursor: 0,
            api_key_masked: true,
            oauth_flow: None,
            oauth_response: String::new(),
            oauth_response_cursor: 0,
            last_result: None,
            ui_notice: None,
            clipboard_fallback: None,
        }
    }

    pub(crate) fn filtered_providers(&self) -> Vec<(usize, &AuthProviderEntry)> {
        if self.filter.is_empty() {
            self.providers.iter().enumerate().collect()
        } else {
            let query = self.filter.to_lowercase();
            self.providers
                .iter()
                .enumerate()
                .filter(|(_, provider)| {
                    provider.display_name.to_lowercase().contains(&query)
                        || provider.provider.to_lowercase().contains(&query)
                })
                .collect()
        }
    }

    pub(crate) fn reset_for_open(&mut self) {
        self.cursor = 0;
        self.filter.clear();
        self.selected = None;
        self.panel = AuthPanel::List;
        self.api_key_input.clear();
        self.api_key_cursor = 0;
        self.api_key_masked = true;
        self.oauth_flow = None;
        self.oauth_response.clear();
        self.oauth_response_cursor = 0;
        self.last_result = None;
        self.ui_notice = None;
        self.clipboard_fallback = None;
    }

    pub(crate) fn close_detail(&mut self) {
        self.selected = None;
        self.clear_detail_state();
    }

    fn clear_detail_state(&mut self) {
        self.panel = AuthPanel::List;
        self.api_key_input.clear();
        self.api_key_cursor = 0;
        self.oauth_flow = None;
        self.oauth_response.clear();
        self.oauth_response_cursor = 0;
        self.last_result = None;
        self.ui_notice = None;
        self.clipboard_fallback = None;
    }

    pub(crate) fn feedback_for_provider(&self, provider: &str) -> Option<(bool, &str)> {
        if let Some(notice) = self.ui_notice.as_ref().filter(|notice| {
            notice
                .provider
                .as_deref()
                .is_none_or(|notice_provider| notice_provider == provider)
        }) {
            return Some((notice.success, notice.message.as_str()));
        }

        self.last_result
            .as_ref()
            .filter(|result| result.provider == provider)
            .map(|result| (result.is_success(), result.message.as_str()))
    }

    pub(crate) fn begin_oauth_flow(&mut self, flow: OAuthFlow) {
        self.oauth_flow = Some(flow);
        self.panel = AuthPanel::OAuthFlow;
        self.oauth_response.clear();
        self.oauth_response_cursor = 0;
        self.last_result = None;
        self.ui_notice = None;
    }

    pub(crate) fn apply_oauth_result(&mut self, result: OAuthResult) -> bool {
        let is_success = result.is_success();
        self.ui_notice = None;
        self.last_result = Some(result);
        if is_success {
            self.oauth_flow = None;
            self.panel = AuthPanel::List;
        }
        is_success
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::*;
    use crate::domain::auth::{OAuthFlowKind, OAuthResultStatus, OAuthStatus};
    use crate::features::auth::input::{AuthInputResult, handle_key};

    fn provider(id: &str, display_name: &str) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: id.into(),
            display_name: display_name.into(),
            oauth_status: Some(OAuthStatus::NotAuthenticated),
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: Some(format!("{}_API_KEY", id.to_uppercase())),
            supports_oauth: true,
            preferred_method: None,
        }
    }

    fn oauth_flow(provider: &str) -> OAuthFlow {
        OAuthFlow {
            flow_id: "flow-1".into(),
            provider: provider.into(),
            authorization_url: "https://example.com/authorize".into(),
            flow_kind: OAuthFlowKind::RedirectCode,
        }
    }

    fn oauth_result(provider: &str, status: OAuthResultStatus, message: &str) -> OAuthResult {
        OAuthResult {
            provider: provider.into(),
            status,
            message: message.into(),
        }
    }

    #[test]
    fn reducer_replaces_provider_catalog_and_reports_exact_count() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("stale", "Stale")];

        let outcome = auth.reduce(AuthAction::Providers(vec![
            provider("openai", "OpenAI"),
            provider("anthropic", "Anthropic"),
        ]));

        assert_eq!(auth.providers.len(), 2);
        assert_eq!(auth.providers[0].provider, "openai");
        assert_eq!(
            outcome,
            AuthOutcome {
                diagnostics: vec![AuthCoordination {
                    level: LogLevel::Debug,
                    message: "2 auth provider(s)".into(),
                }],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn provider_replacement_normalizes_cursor_and_selection_identity() {
        let mut auth = AuthState::new();
        auth.providers = vec![
            provider("one", "One"),
            provider("two", "Two"),
            provider("three", "Three"),
        ];
        auth.cursor = 2;
        auth.selected = Some(1);
        auth.filter = "t".into();

        auth.reduce(AuthAction::Providers(vec![
            provider("two", "Two updated"),
            provider("four", "Four"),
        ]));
        assert_eq!(auth.cursor, 0);
        assert_eq!(auth.selected, Some(0));

        auth.cursor = 9;
        auth.selected = Some(0);
        auth.filter = "four".into();
        auth.reduce(AuthAction::Providers(vec![provider("four", "Four")]));
        assert_eq!(auth.cursor, 0);
        assert_eq!(auth.selected, None);

        auth.cursor = 4;
        auth.filter = "missing".into();
        auth.reduce(AuthAction::Providers(Vec::new()));
        assert_eq!(auth.cursor, 0);
        assert_eq!(auth.selected, None);
    }

    fn populate_detail_state(auth: &mut AuthState, provider_id: &str, panel: AuthPanel) {
        auth.selected = Some(0);
        auth.panel = panel;
        auth.api_key_input = "secret".into();
        auth.api_key_cursor = auth.api_key_input.len();
        auth.oauth_flow = Some(oauth_flow(provider_id));
        auth.oauth_response = "callback".into();
        auth.oauth_response_cursor = auth.oauth_response.len();
        auth.last_result = Some(oauth_result(
            provider_id,
            OAuthResultStatus::Failure,
            "stale result",
        ));
        auth.ui_notice = Some(AuthUiNotice {
            provider: Some(provider_id.into()),
            success: false,
            message: "stale notice".into(),
        });
        auth.clipboard_fallback = Some("https://example.com/fallback".into());
    }

    fn assert_detail_state_cleared(auth: &AuthState) {
        assert_eq!(auth.panel, AuthPanel::List);
        assert!(auth.api_key_input.is_empty());
        assert_eq!(auth.api_key_cursor, 0);
        assert!(auth.oauth_flow.is_none());
        assert!(auth.oauth_response.is_empty());
        assert_eq!(auth.oauth_response_cursor, 0);
        assert!(auth.last_result.is_none());
        assert!(auth.ui_notice.is_none());
        assert!(auth.clipboard_fallback.is_none());
    }

    #[test]
    fn provider_removal_clears_api_key_detail_state() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("removed", "Removed")];
        populate_detail_state(&mut auth, "removed", AuthPanel::ApiKeyInput);

        auth.reduce(AuthAction::Providers(Vec::new()));

        assert!(auth.selected.is_none());
        assert_detail_state_cleared(&auth);
        assert_eq!(
            handle_key(&mut auth, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AuthInputResult::NotHandled
        );
    }

    #[test]
    fn provider_removal_clears_oauth_detail_state() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("removed", "Removed")];
        populate_detail_state(&mut auth, "removed", AuthPanel::OAuthFlow);

        auth.reduce(AuthAction::Providers(Vec::new()));

        assert!(auth.selected.is_none());
        assert_detail_state_cleared(&auth);
        assert_eq!(
            handle_key(&mut auth, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AuthInputResult::NotHandled
        );
    }

    #[test]
    fn provider_capability_loss_clears_incompatible_api_key_panel() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("retained", "Retained")];
        populate_detail_state(&mut auth, "retained", AuthPanel::ApiKeyInput);
        let mut oauth_only = provider("retained", "Retained");
        oauth_only.env_var_name = None;
        oauth_only.has_stored_api_key = false;

        auth.reduce(AuthAction::Providers(vec![oauth_only]));

        assert_eq!(auth.selected, Some(0));
        assert_detail_state_cleared(&auth);
    }

    #[test]
    fn provider_capability_loss_clears_incompatible_oauth_panel() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("retained", "Retained")];
        populate_detail_state(&mut auth, "retained", AuthPanel::OAuthFlow);
        let mut api_key_only = provider("retained", "Retained");
        api_key_only.supports_oauth = false;
        api_key_only.oauth_status = None;

        auth.reduce(AuthAction::Providers(vec![api_key_only]));

        assert_eq!(auth.selected, Some(0));
        assert_detail_state_cleared(&auth);
    }

    #[test]
    fn provider_reorder_preserves_compatible_detail_state_by_identity() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("selected", "Selected"), provider("other", "Other")];
        populate_detail_state(&mut auth, "selected", AuthPanel::OAuthFlow);
        let expected_flow = auth.oauth_flow.clone();
        let expected_result = auth.last_result.clone();
        let expected_notice = auth.ui_notice.clone();

        auth.reduce(AuthAction::Providers(vec![
            provider("other", "Other updated"),
            provider("selected", "Selected updated"),
        ]));

        assert_eq!(auth.selected, Some(1));
        assert_eq!(auth.panel, AuthPanel::OAuthFlow);
        assert_eq!(auth.api_key_input, "secret");
        assert_eq!(auth.api_key_cursor, 6);
        assert_eq!(auth.oauth_flow, expected_flow);
        assert_eq!(auth.oauth_response, "callback");
        assert_eq!(auth.oauth_response_cursor, 8);
        assert_eq!(auth.last_result, expected_result);
        assert_eq!(auth.ui_notice, expected_notice);
        assert_eq!(
            auth.clipboard_fallback.as_deref(),
            Some("https://example.com/fallback")
        );
    }

    #[test]
    fn reducer_begins_oauth_flow_with_exact_state_and_diagnostic() {
        let mut auth = AuthState::new();
        auth.oauth_response = "stale response".into();
        auth.oauth_response_cursor = auth.oauth_response.len();
        auth.last_result = Some(oauth_result(
            "openai",
            OAuthResultStatus::Failure,
            "old result",
        ));
        let flow = oauth_flow("openai");

        let outcome = auth.reduce(AuthAction::OAuthFlowStarted(flow.clone()));

        assert_eq!(auth.oauth_flow, Some(flow));
        assert_eq!(auth.panel, AuthPanel::OAuthFlow);
        assert!(auth.oauth_response.is_empty());
        assert_eq!(auth.oauth_response_cursor, 0);
        assert!(auth.last_result.is_none());
        assert_eq!(
            outcome,
            AuthOutcome {
                diagnostics: vec![AuthCoordination {
                    level: LogLevel::Info,
                    message: "OAuth flow started for openai".into(),
                }],
                effects: Vec::new(),
            }
        );
    }

    #[test]
    fn reducer_oauth_results_always_refresh_providers_and_preserve_failure_flow() {
        let mut auth = AuthState::new();
        let flow = oauth_flow("openai");
        auth.oauth_flow = Some(flow.clone());
        auth.panel = AuthPanel::OAuthFlow;

        let failed = auth.reduce(AuthAction::OAuthResult(oauth_result(
            "openai",
            OAuthResultStatus::Failure,
            "authorization denied",
        )));

        assert_eq!(auth.oauth_flow, Some(flow));
        assert_eq!(auth.panel, AuthPanel::OAuthFlow);
        assert_eq!(
            failed,
            AuthOutcome {
                diagnostics: vec![AuthCoordination {
                    level: LogLevel::Warn,
                    message: "authorization denied".into(),
                }],
                effects: vec![Effect::Command(Command::ListAuthProviders)],
            }
        );

        let succeeded = auth.reduce(AuthAction::OAuthResult(oauth_result(
            "openai",
            OAuthResultStatus::Success,
            "connected",
        )));

        assert!(auth.oauth_flow.is_none());
        assert_eq!(auth.panel, AuthPanel::List);
        assert_eq!(
            succeeded,
            AuthOutcome {
                diagnostics: vec![AuthCoordination {
                    level: LogLevel::Info,
                    message: "connected".into(),
                }],
                effects: vec![Effect::Command(Command::ListAuthProviders)],
            }
        );
    }

    #[test]
    fn reducer_clipboard_results_preserve_exact_feedback_and_fallback() {
        let mut auth = AuthState::new();
        auth.oauth_flow = Some(oauth_flow("openai"));
        auth.ui_notice = Some(AuthUiNotice {
            provider: None,
            success: false,
            message: "old notice".into(),
        });

        let failed = auth.reduce(AuthAction::ClipboardFinished {
            provider: "openai".into(),
            success: false,
        });

        assert_eq!(failed, AuthOutcome::default());
        assert!(auth.ui_notice.is_none());
        assert_eq!(
            auth.clipboard_fallback.as_deref(),
            Some("https://example.com/authorize")
        );

        let succeeded = auth.reduce(AuthAction::ClipboardFinished {
            provider: "openai".into(),
            success: true,
        });

        assert_eq!(succeeded, AuthOutcome::default());
        assert!(matches!(
            auth.ui_notice.as_ref(),
            Some(notice)
                if notice.provider.as_deref() == Some("openai")
                    && notice.success
                    && notice.message == "Copied to clipboard"
        ));
        assert!(auth.clipboard_fallback.is_none());
    }

    #[test]
    fn reducer_notice_clear_preserves_all_other_auth_state_with_empty_outcome() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("openai", "OpenAI")];
        auth.cursor = 4;
        auth.filter = "keep".into();
        auth.selected = Some(0);
        auth.panel = AuthPanel::OAuthFlow;
        auth.api_key_input = "secret".into();
        auth.api_key_cursor = 3;
        auth.api_key_masked = false;
        auth.oauth_flow = Some(oauth_flow("openai"));
        auth.oauth_response = "response".into();
        auth.oauth_response_cursor = 5;
        auth.last_result = Some(oauth_result(
            "openai",
            OAuthResultStatus::Failure,
            "old result",
        ));
        auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: false,
            message: "old notice".into(),
        });
        auth.clipboard_fallback = Some("https://example.com/fallback".into());

        let outcome = auth.reduce(AuthAction::ClearUiNotice);

        assert_eq!(outcome, AuthOutcome::default());
        assert!(auth.ui_notice.is_none());
        assert_eq!(auth.providers, vec![provider("openai", "OpenAI")]);
        assert_eq!(auth.cursor, 4);
        assert_eq!(auth.filter, "keep");
        assert_eq!(auth.selected, Some(0));
        assert_eq!(auth.panel, AuthPanel::OAuthFlow);
        assert_eq!(auth.api_key_input, "secret");
        assert_eq!(auth.api_key_cursor, 3);
        assert!(!auth.api_key_masked);
        assert_eq!(auth.oauth_flow, Some(oauth_flow("openai")));
        assert_eq!(auth.oauth_response, "response");
        assert_eq!(auth.oauth_response_cursor, 5);
        assert_eq!(
            auth.last_result,
            Some(oauth_result(
                "openai",
                OAuthResultStatus::Failure,
                "old result",
            ))
        );
        assert_eq!(
            auth.clipboard_fallback.as_deref(),
            Some("https://example.com/fallback")
        );
    }

    #[test]
    fn constructor_uses_expected_defaults() {
        let auth = AuthState::new();

        assert!(auth.providers.is_empty());
        assert_eq!(auth.cursor, 0);
        assert!(auth.filter.is_empty());
        assert!(auth.selected.is_none());
        assert_eq!(auth.panel, AuthPanel::List);
        assert!(auth.api_key_input.is_empty());
        assert_eq!(auth.api_key_cursor, 0);
        assert!(auth.api_key_masked);
        assert!(auth.oauth_flow.is_none());
        assert!(auth.oauth_response.is_empty());
        assert_eq!(auth.oauth_response_cursor, 0);
        assert!(auth.last_result.is_none());
        assert!(auth.ui_notice.is_none());
        assert!(auth.clipboard_fallback.is_none());
    }

    #[test]
    fn reset_for_open_clears_transient_state_and_preserves_providers() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("openai", "OpenAI")];
        auth.cursor = 5;
        auth.filter = "test".into();
        auth.selected = Some(2);
        auth.panel = AuthPanel::OAuthFlow;
        auth.api_key_input = "secret".into();
        auth.api_key_cursor = 6;
        auth.api_key_masked = false;
        auth.oauth_flow = Some(oauth_flow("openai"));
        auth.oauth_response = "response".into();
        auth.oauth_response_cursor = 8;
        auth.last_result = Some(oauth_result(
            "openai",
            OAuthResultStatus::Failure,
            "old result",
        ));
        auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        auth.clipboard_fallback = Some("https://example.com".into());

        auth.reset_for_open();

        assert_eq!(auth.providers.len(), 1);
        assert_eq!(auth.providers[0].provider, "openai");
        assert_eq!(auth.cursor, 0);
        assert!(auth.filter.is_empty());
        assert!(auth.selected.is_none());
        assert_eq!(auth.panel, AuthPanel::List);
        assert!(auth.api_key_input.is_empty());
        assert_eq!(auth.api_key_cursor, 0);
        assert!(auth.api_key_masked);
        assert!(auth.oauth_flow.is_none());
        assert!(auth.oauth_response.is_empty());
        assert_eq!(auth.oauth_response_cursor, 0);
        assert!(auth.last_result.is_none());
        assert!(auth.ui_notice.is_none());
        assert!(auth.clipboard_fallback.is_none());
    }

    #[test]
    fn filtered_providers_with_empty_filter_returns_all_rows() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("openai", "OpenAI"), provider("groq", "Groq")];

        let filtered = auth.filtered_providers();

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, 0);
        assert_eq!(filtered[1].0, 1);
    }

    #[test]
    fn filtered_providers_matches_id_and_name_case_insensitively() {
        let mut auth = AuthState::new();
        auth.providers = vec![
            provider("openai", "OpenAI"),
            provider("anthropic", "Claude Provider"),
        ];

        auth.filter = "CLAUDE".into();
        let by_name = auth.filtered_providers();
        assert_eq!(by_name.len(), 1);
        assert_eq!(by_name[0].0, 1);

        auth.filter = "OPENAI".into();
        let by_id = auth.filtered_providers();
        assert_eq!(by_id.len(), 1);
        assert_eq!(by_id[0].0, 0);
    }

    #[test]
    fn close_detail_resets_panel_state_and_preserves_list_state_and_mask() {
        let mut auth = AuthState::new();
        auth.providers = vec![provider("openai", "OpenAI")];
        auth.cursor = 3;
        auth.filter = "open".into();
        auth.selected = Some(0);
        auth.panel = AuthPanel::ApiKeyInput;
        auth.api_key_input = "secret".into();
        auth.api_key_cursor = 6;
        auth.api_key_masked = false;
        auth.oauth_flow = Some(oauth_flow("openai"));
        auth.oauth_response = "response".into();
        auth.oauth_response_cursor = 8;
        auth.last_result = Some(oauth_result(
            "openai",
            OAuthResultStatus::Success,
            "connected",
        ));
        auth.ui_notice = Some(AuthUiNotice {
            provider: None,
            success: true,
            message: "saved".into(),
        });
        auth.clipboard_fallback = Some("https://example.com".into());

        auth.close_detail();

        assert_eq!(auth.providers.len(), 1);
        assert_eq!(auth.cursor, 3);
        assert_eq!(auth.filter, "open");
        assert!(!auth.api_key_masked);
        assert!(auth.selected.is_none());
        assert_eq!(auth.panel, AuthPanel::List);
        assert!(auth.api_key_input.is_empty());
        assert_eq!(auth.api_key_cursor, 0);
        assert!(auth.oauth_flow.is_none());
        assert!(auth.oauth_response.is_empty());
        assert_eq!(auth.oauth_response_cursor, 0);
        assert!(auth.last_result.is_none());
        assert!(auth.ui_notice.is_none());
        assert!(auth.clipboard_fallback.is_none());
    }

    #[test]
    fn feedback_scopes_oauth_result_to_its_provider() {
        let mut auth = AuthState::new();
        auth.last_result = Some(oauth_result(
            "openai",
            OAuthResultStatus::Failure,
            "authorization denied",
        ));

        assert_eq!(auth.feedback_for_provider("anthropic"), None);
        assert_eq!(
            auth.feedback_for_provider("openai"),
            Some((false, "authorization denied"))
        );
    }

    #[test]
    fn feedback_scopes_ui_notice_and_takes_precedence() {
        let mut auth = AuthState::new();
        auth.last_result = Some(oauth_result(
            "openai",
            OAuthResultStatus::Failure,
            "authorization denied",
        ));
        auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "Copied to clipboard".into(),
        });

        assert_eq!(auth.feedback_for_provider("anthropic"), None);
        assert_eq!(
            auth.feedback_for_provider("openai"),
            Some((true, "Copied to clipboard"))
        );
    }

    #[test]
    fn feedback_supports_generic_ui_notice() {
        let mut auth = AuthState::new();
        auth.ui_notice = Some(AuthUiNotice {
            provider: None,
            success: false,
            message: "Clipboard unavailable".into(),
        });

        assert_eq!(
            auth.feedback_for_provider("openai"),
            Some((false, "Clipboard unavailable"))
        );
        assert_eq!(
            auth.feedback_for_provider("anthropic"),
            Some((false, "Clipboard unavailable"))
        );
    }

    #[test]
    fn begin_oauth_flow_replaces_flow_and_clears_response_and_feedback() {
        let mut auth = AuthState::new();
        auth.oauth_response = "stale response".into();
        auth.oauth_response_cursor = auth.oauth_response.len();
        auth.last_result = Some(oauth_result(
            "openai",
            OAuthResultStatus::Failure,
            "old result",
        ));
        auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let flow = oauth_flow("openai");

        auth.begin_oauth_flow(flow.clone());

        assert_eq!(auth.oauth_flow, Some(flow));
        assert_eq!(auth.panel, AuthPanel::OAuthFlow);
        assert!(auth.oauth_response.is_empty());
        assert_eq!(auth.oauth_response_cursor, 0);
        assert!(auth.last_result.is_none());
        assert!(auth.ui_notice.is_none());
    }

    #[test]
    fn successful_oauth_result_clears_flow_and_returns_true() {
        let mut auth = AuthState::new();
        auth.oauth_flow = Some(oauth_flow("openai"));
        auth.panel = AuthPanel::OAuthFlow;
        auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let result = oauth_result("openai", OAuthResultStatus::Success, "connected");

        assert!(auth.apply_oauth_result(result.clone()));
        assert!(auth.oauth_flow.is_none());
        assert_eq!(auth.panel, AuthPanel::List);
        assert_eq!(auth.last_result, Some(result));
        assert!(auth.ui_notice.is_none());
    }

    #[test]
    fn failed_oauth_result_preserves_flow_and_returns_false() {
        let mut auth = AuthState::new();
        let flow = oauth_flow("openai");
        auth.oauth_flow = Some(flow.clone());
        auth.panel = AuthPanel::OAuthFlow;
        auth.ui_notice = Some(AuthUiNotice {
            provider: Some("openai".into()),
            success: true,
            message: "old notice".into(),
        });
        let result = oauth_result("openai", OAuthResultStatus::Failure, "denied");

        assert!(!auth.apply_oauth_result(result.clone()));
        assert_eq!(auth.oauth_flow, Some(flow));
        assert_eq!(auth.panel, AuthPanel::OAuthFlow);
        assert_eq!(auth.last_result, Some(result));
        assert!(auth.ui_notice.is_none());
    }
}
