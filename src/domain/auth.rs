use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    #[serde(rename = "oauth")]
    OAuth,
    ApiKey,
    EnvVar,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OAuthStatus {
    Connected,
    Expired,
    NotAuthenticated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthStatus {
    Unconfigurable,
    Expired,
    Active(AuthMethod),
    NotConfigured,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AuthProviderEntry {
    pub provider: String,
    pub display_name: String,
    pub oauth_status: Option<OAuthStatus>,
    pub has_stored_api_key: bool,
    pub has_env_api_key: bool,
    pub env_var_name: Option<String>,
    pub supports_oauth: bool,
    pub preferred_method: Option<AuthMethod>,
}

impl AuthProviderEntry {
    /// Provider supports only OAuth (no API key env var).
    pub fn is_oauth_only(&self) -> bool {
        self.supports_oauth && self.env_var_name.is_none()
    }

    /// Provider supports only API key (no OAuth).
    pub fn is_api_key_only(&self) -> bool {
        !self.supports_oauth && self.env_var_name.is_some()
    }

    /// Provider supports multiple auth methods (both OAuth and API key).
    pub fn has_multiple_auth_methods(&self) -> bool {
        self.supports_oauth && self.env_var_name.is_some()
    }

    /// Provider requires OAuth but the build doesn't include it.
    pub fn is_unconfigurable(&self) -> bool {
        !self.supports_oauth && self.env_var_name.is_none()
    }

    /// Resolve which auth method is effectively active.
    pub fn effective_auth(&self) -> Option<AuthMethod> {
        let pref = self.preferred_method;
        let order: &[AuthMethod] = if let Some(p) = pref {
            // Preferred first, then defaults
            match p {
                AuthMethod::OAuth => &[AuthMethod::OAuth, AuthMethod::ApiKey, AuthMethod::EnvVar],
                AuthMethod::ApiKey => &[AuthMethod::ApiKey, AuthMethod::OAuth, AuthMethod::EnvVar],
                AuthMethod::EnvVar => &[AuthMethod::EnvVar, AuthMethod::OAuth, AuthMethod::ApiKey],
            }
        } else if self.supports_oauth {
            &[AuthMethod::OAuth, AuthMethod::ApiKey, AuthMethod::EnvVar]
        } else {
            &[AuthMethod::ApiKey, AuthMethod::EnvVar]
        };

        for method in order {
            match method {
                AuthMethod::OAuth => {
                    if self.oauth_status == Some(OAuthStatus::Connected) {
                        return Some(AuthMethod::OAuth);
                    }
                }
                AuthMethod::ApiKey => {
                    if self.has_stored_api_key {
                        return Some(AuthMethod::ApiKey);
                    }
                }
                AuthMethod::EnvVar => {
                    if self.has_env_api_key {
                        return Some(AuthMethod::EnvVar);
                    }
                }
            }
        }
        None
    }

    /// Classify the provider's current authentication state.
    pub fn auth_status(&self) -> AuthStatus {
        if self.is_unconfigurable() {
            AuthStatus::Unconfigurable
        } else if self.oauth_status == Some(OAuthStatus::Expired) {
            AuthStatus::Expired
        } else if let Some(method) = self.effective_auth() {
            AuthStatus::Active(method)
        } else {
            AuthStatus::NotConfigured
        }
    }

    /// Whether the badge indicates a successful/active auth.
    pub fn is_auth_active(&self) -> bool {
        self.effective_auth().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OAuthFlowKind {
    RedirectCode,
    DevicePoll,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthFlow {
    pub flow_id: String,
    pub provider: String,
    pub authorization_url: String,
    pub flow_kind: OAuthFlowKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthResultStatus {
    Success,
    Failure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthResult {
    pub provider: String,
    pub status: OAuthResultStatus,
    pub message: String,
}

impl OAuthResult {
    pub fn is_success(&self) -> bool {
        matches!(self.status, OAuthResultStatus::Success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_provider(name: &str) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: name.to_lowercase(),
            display_name: name.to_string(),
            oauth_status: Some(OAuthStatus::NotAuthenticated),
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: Some(format!("{}_API_KEY", name.to_uppercase())),
            supports_oauth: true,
            preferred_method: None,
        }
    }

    fn make_oauth_only(name: &str) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: name.to_lowercase(),
            display_name: name.to_string(),
            oauth_status: Some(OAuthStatus::NotAuthenticated),
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: None,
            supports_oauth: true,
            preferred_method: None,
        }
    }

    fn make_api_key_only(name: &str) -> AuthProviderEntry {
        AuthProviderEntry {
            provider: name.to_lowercase(),
            display_name: name.to_string(),
            oauth_status: None,
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: Some(format!("{}_API_KEY", name.to_uppercase())),
            supports_oauth: false,
            preferred_method: None,
        }
    }

    #[test]
    fn auth_provider_entry_effective_auth_oauth_connected() {
        let mut p = make_provider("OpenAI");
        p.oauth_status = Some(OAuthStatus::Connected);
        assert_eq!(p.effective_auth(), Some(AuthMethod::OAuth));
        assert_eq!(p.auth_status(), AuthStatus::Active(AuthMethod::OAuth));
        assert!(p.is_auth_active());
    }

    #[test]
    fn auth_provider_entry_effective_auth_api_key_stored() {
        let mut p = make_api_key_only("Groq");
        p.has_stored_api_key = true;
        assert_eq!(p.effective_auth(), Some(AuthMethod::ApiKey));
        assert_eq!(p.auth_status(), AuthStatus::Active(AuthMethod::ApiKey));
        assert!(p.is_auth_active());
    }

    #[test]
    fn auth_provider_entry_effective_auth_env_var() {
        let mut p = make_api_key_only("DeepSeek");
        p.has_env_api_key = true;
        assert_eq!(p.effective_auth(), Some(AuthMethod::EnvVar));
        assert_eq!(p.auth_status(), AuthStatus::Active(AuthMethod::EnvVar));
    }

    #[test]
    fn auth_provider_entry_not_configured() {
        let p = make_provider("OpenAI");
        assert_eq!(p.effective_auth(), None);
        assert_eq!(p.auth_status(), AuthStatus::NotConfigured);
        assert!(!p.is_auth_active());
    }

    #[test]
    fn auth_provider_entry_expired_status() {
        let mut p = make_oauth_only("Codex");
        p.oauth_status = Some(OAuthStatus::Expired);
        assert_eq!(p.auth_status(), AuthStatus::Expired);
    }

    #[test]
    fn auth_provider_entry_unconfigurable() {
        let p = AuthProviderEntry {
            provider: "codex".into(),
            display_name: "Codex".into(),
            oauth_status: None,
            has_stored_api_key: false,
            has_env_api_key: false,
            env_var_name: None,
            supports_oauth: false,
            preferred_method: None,
        };
        assert!(p.is_unconfigurable());
        assert_eq!(p.auth_status(), AuthStatus::Unconfigurable);
    }

    #[test]
    fn auth_provider_entry_preferred_method_controls_effective_auth_order() {
        for preferred in [AuthMethod::OAuth, AuthMethod::ApiKey, AuthMethod::EnvVar] {
            let mut provider = make_provider("OpenAI");
            provider.oauth_status = Some(OAuthStatus::Connected);
            provider.has_stored_api_key = true;
            provider.has_env_api_key = true;
            provider.preferred_method = Some(preferred);

            assert_eq!(provider.effective_auth(), Some(preferred));
            assert_eq!(provider.auth_status(), AuthStatus::Active(preferred));
        }
    }

    #[test]
    fn auth_provider_classification_helpers() {
        let multi = make_provider("OpenAI");
        assert!(multi.has_multiple_auth_methods());
        assert!(!multi.is_oauth_only());
        assert!(!multi.is_api_key_only());

        let oauth_only = make_oauth_only("Codex");
        assert!(oauth_only.is_oauth_only());
        assert!(!oauth_only.has_multiple_auth_methods());

        let api_only = make_api_key_only("Groq");
        assert!(api_only.is_api_key_only());
        assert!(!api_only.has_multiple_auth_methods());
    }

    #[test]
    fn auth_method_oauth_serde_matches_server_wire_format() {
        // Server explicitly renames OAuth to "oauth" (not "o_auth")
        let json = serde_json::json!("oauth");
        let method: AuthMethod = serde_json::from_value(json).unwrap();
        assert_eq!(method, AuthMethod::OAuth);

        // Round-trip: our serialization must also produce "oauth"
        let serialized = serde_json::to_string(&AuthMethod::OAuth).unwrap();
        assert_eq!(serialized, "\"oauth\"");
    }

    #[test]
    fn auth_enum_wire_values_remain_compatible() {
        assert_eq!(
            serde_json::from_value::<AuthMethod>(json!("api_key")).unwrap(),
            AuthMethod::ApiKey
        );
        assert_eq!(
            serde_json::from_value::<AuthMethod>(json!("env_var")).unwrap(),
            AuthMethod::EnvVar
        );
        assert_eq!(
            serde_json::from_value::<OAuthStatus>(json!("connected")).unwrap(),
            OAuthStatus::Connected
        );
        assert_eq!(
            serde_json::from_value::<OAuthStatus>(json!("expired")).unwrap(),
            OAuthStatus::Expired
        );
        assert_eq!(
            serde_json::from_value::<OAuthStatus>(json!("not_authenticated")).unwrap(),
            OAuthStatus::NotAuthenticated
        );
        assert_eq!(
            serde_json::from_value::<OAuthFlowKind>(json!("redirect_code")).unwrap(),
            OAuthFlowKind::RedirectCode
        );
        assert_eq!(
            serde_json::from_value::<OAuthFlowKind>(json!("device_poll")).unwrap(),
            OAuthFlowKind::DevicePoll
        );
    }

    #[test]
    fn oauth_result_reports_success_from_semantic_status() {
        let mut result = OAuthResult {
            provider: "openai".into(),
            status: OAuthResultStatus::Success,
            message: "connected".into(),
        };
        assert!(result.is_success());

        result.status = OAuthResultStatus::Failure;
        assert!(!result.is_success());
    }

    #[test]
    fn auth_provider_entry_deserializes_with_oauth_preferred_method() {
        // Exact JSON shape the server sends for a provider with OAuth preference
        let json = serde_json::json!({
            "provider": "anthropic",
            "display_name": "Anthropic",
            "oauth_status": "connected",
            "has_stored_api_key": true,
            "has_env_api_key": false,
            "env_var_name": "ANTHROPIC_API_KEY",
            "supports_oauth": true,
            "preferred_method": "oauth"
        });
        let entry: AuthProviderEntry = serde_json::from_value(json).unwrap();
        assert_eq!(entry.provider, "anthropic");
        assert_eq!(entry.oauth_status, Some(OAuthStatus::Connected));
        assert_eq!(entry.preferred_method, Some(AuthMethod::OAuth));
    }
}
