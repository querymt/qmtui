use serde::Deserialize;

use crate::domain::auth::{AuthProviderEntry, OAuthFlowKind};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OAuthFlowDto {
    pub flow_id: String,
    pub provider: String,
    pub authorization_url: String,
    pub flow_kind: OAuthFlowKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct OAuthResultDto {
    pub provider: String,
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AuthProvidersDto {
    pub providers: Vec<AuthProviderEntry>,
}

#[cfg(test)]
mod tests {
    use super::{AuthProvidersDto, OAuthFlowDto, OAuthResultDto};
    use crate::domain::auth::{AuthMethod, OAuthFlowKind, OAuthStatus};
    use serde_json::json;

    #[test]
    fn oauth_flow_dto_deserializes_both_flow_kinds_and_requires_fields() {
        for (wire_kind, expected_kind) in [
            ("redirect_code", OAuthFlowKind::RedirectCode),
            ("device_poll", OAuthFlowKind::DevicePoll),
        ] {
            let flow: OAuthFlowDto = serde_json::from_value(json!({
                "flow_id": "flow-123",
                "provider": "openai",
                "authorization_url": "https://example.test/authorize",
                "flow_kind": wire_kind
            }))
            .unwrap();
            assert_eq!(flow.flow_id, "flow-123");
            assert_eq!(flow.provider, "openai");
            assert_eq!(flow.authorization_url, "https://example.test/authorize");
            assert_eq!(flow.flow_kind, expected_kind);
        }

        let complete = json!({
            "flow_id": "flow-123",
            "provider": "openai",
            "authorization_url": "https://example.test/authorize",
            "flow_kind": "redirect_code"
        });
        for field in ["flow_id", "provider", "authorization_url", "flow_kind"] {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<OAuthFlowDto>(missing).is_err());
        }
    }

    #[test]
    fn oauth_result_dto_deserializes_success_failure_and_requires_fields() {
        let success: OAuthResultDto = serde_json::from_value(json!({
            "provider": "openai",
            "success": true,
            "message": "connected"
        }))
        .unwrap();
        assert_eq!(success.provider, "openai");
        assert!(success.success);
        assert_eq!(success.message, "connected");

        let failure: OAuthResultDto = serde_json::from_value(json!({
            "provider": "anthropic",
            "success": false,
            "message": "authorization denied"
        }))
        .unwrap();
        assert_eq!(failure.provider, "anthropic");
        assert!(!failure.success);
        assert_eq!(failure.message, "authorization denied");

        let complete = json!({
            "provider": "openai",
            "success": true,
            "message": "connected"
        });
        for field in ["provider", "success", "message"] {
            let mut missing = complete.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<OAuthResultDto>(missing).is_err());
        }
    }

    #[test]
    fn auth_providers_dto_deserializes_mixed_providers() {
        let data: AuthProvidersDto = serde_json::from_value(json!({
            "providers": [
                {
                    "provider": "openai",
                    "display_name": "OpenAI",
                    "oauth_status": "not_authenticated",
                    "has_stored_api_key": false,
                    "has_env_api_key": true,
                    "env_var_name": "OPENAI_API_KEY",
                    "supports_oauth": true,
                    "preferred_method": "oauth"
                },
                {
                    "provider": "groq",
                    "display_name": "Groq",
                    "oauth_status": null,
                    "has_stored_api_key": true,
                    "has_env_api_key": false,
                    "env_var_name": "GROQ_API_KEY",
                    "supports_oauth": false,
                    "preferred_method": "api_key"
                },
                {
                    "provider": "local",
                    "display_name": "Local",
                    "oauth_status": null,
                    "has_stored_api_key": false,
                    "has_env_api_key": false,
                    "env_var_name": null,
                    "supports_oauth": false,
                    "preferred_method": null
                }
            ],
            "unknown_field": "ignored"
        }))
        .unwrap();

        assert_eq!(data.providers.len(), 3);
        assert_eq!(
            data.providers[0].oauth_status,
            Some(OAuthStatus::NotAuthenticated)
        );
        assert_eq!(data.providers[0].preferred_method, Some(AuthMethod::OAuth));
        assert_eq!(data.providers[1].preferred_method, Some(AuthMethod::ApiKey));
        assert_eq!(data.providers[2].preferred_method, None);
    }
}
