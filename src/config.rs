// src/config.rs — TUI persistent configuration and session cache
//
//   ~/.qmt/qmtui.toml  - user config (theme, ACP defaults, delegate models)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

// ── path overrides for tests ─────────────────────────────────────────────────

static CONFIG_PATH_OVERRIDE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static TEST_PERSISTENCE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn config_path_override() -> &'static Mutex<Option<PathBuf>> {
    CONFIG_PATH_OVERRIDE.get_or_init(|| Mutex::new(None))
}

fn test_persistence_lock() -> &'static Mutex<()> {
    TEST_PERSISTENCE_LOCK.get_or_init(|| Mutex::new(()))
}

/// Override the config path used by `TuiConfig::load()` / `save()`.
/// Intended for tests only; production code should not call this.
pub fn test_set_config_path_override(path: Option<PathBuf>) {
    *config_path_override().lock().unwrap() = path;
}

#[cfg(test)]
pub struct TestPersistenceGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl TestPersistenceGuard {
    pub fn new(label: &str) -> Self {
        let lock = test_persistence_lock().lock().unwrap();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("qmt-persistence-tests-{label}-{pid}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        test_set_config_path_override(Some(dir.join("qmtui.toml")));
        Self { _lock: lock }
    }
}

#[cfg(test)]
impl Drop for TestPersistenceGuard {
    fn drop(&mut self) {
        test_set_config_path_override(None);
    }
}

// -- TuiConfig - ~/.qmt/qmtui.toml -------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AcpTransportMode {
    #[default]
    Stdio,
    WebSocket,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AcpConfig {
    /// ACP transport to use. Only stdio is wired today; WebSocket is reserved
    /// because qmtcode can expose ACP at ws://host/ws once the CLI grows a flag.
    pub transport: Option<AcpTransportMode>,
    /// Reserved for future ACP WebSocket support.
    pub websocket_url: Option<String>,
    /// Path to the `qmtcode` binary. Falls back to `$PATH` lookup when absent.
    pub binary_path: Option<String>,
    /// Extra CLI arguments passed to the spawned ACP agent.
    /// Default (when absent): `["--acp"]`.
    pub binary_args: Option<Vec<String>>,
    /// Automatically start a local ACP stdio agent. Default: `true`.
    pub auto_start: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileConfig {
    pub id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiConfig {
    pub theme: Option<String>,
    pub show_thinking: Option<bool>,
    pub acp: AcpConfig,
    /// Per-agent-id model preferences: agent_id → "provider/model".
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub delegate_models: HashMap<String, String>,
    pub profile: ProfileConfig,
}

impl TuiConfig {
    pub fn config_path() -> PathBuf {
        if let Some(path) = config_path_override().lock().unwrap().clone() {
            return path;
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".qmt")
            .join("qmtui.toml")
    }

    /// Load from the default path (`~/.qmt/qmtui.toml`).
    pub fn load() -> Self {
        Self::load_from_path(&Self::config_path())
    }

    /// Load from an explicit path. Returns `Default` on any I/O or parse error.
    pub fn load_from_path(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| toml::from_str(&t).ok())
            .unwrap_or_default()
    }

    /// Save to the default path (`~/.qmt/qmtui.toml`).
    pub fn save(&self) {
        #[cfg(test)]
        assert!(
            config_path_override().lock().unwrap().is_some(),
            "TuiConfig::save() called in test without config path override! \
             Use TestPersistenceGuard to avoid writing to the real ~/.qmt/qmtui.toml."
        );
        self.save_to_path(&Self::config_path());
    }

    /// Save to an explicit path. Creates parent directories if needed.
    /// Errors are intentionally ignored (best-effort persistence).
    pub fn save_to_path(&self, path: &Path) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }

    /// Return a copy of this config with app-owned UI fields refreshed.
    /// This intentionally preserves unrelated persisted settings like `acp.*`.
    pub fn with_app_settings(&self, app: &crate::app::App) -> Self {
        let mut merged = self.clone();
        merged.theme = Some(crate::theme::Theme::current_id().to_string());
        merged.show_thinking = Some(app.show_thinking);
        merged.delegate_models = app
            .delegate_model_preferences
            .iter()
            .map(|(id, (p, m))| (id.clone(), format!("{p}/{m}")))
            .collect();
        merged.profile.id = app.active_profile_id.clone();
        merged
    }

    pub fn acp_args(&self) -> Vec<String> {
        self.acp
            .binary_args
            .clone()
            .unwrap_or_else(|| vec!["--acp".to_string()])
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use serial_test::serial;

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("qmt-tui-tests-{label}-{pid}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    struct TestPathGuard(TestPersistenceGuard);

    impl TestPathGuard {
        fn new(label: &str) -> Self {
            Self(TestPersistenceGuard::new(label))
        }
    }

    // ── TuiConfig ─────────────────────────────────────────────────────────────

    #[test]
    fn config_round_trip() {
        let cfg = TuiConfig {
            theme: Some("base16-ocean".into()),
            show_thinking: Some(false),
            acp: AcpConfig {
                binary_path: Some("/usr/local/bin/qmtcode".into()),
                binary_args: Some(vec!["--acp".into()]),
                auto_start: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        let text = toml::to_string_pretty(&cfg).unwrap();
        assert_eq!(toml::from_str::<TuiConfig>(&text).unwrap(), cfg);
    }

    #[test]
    fn config_empty_deserializes_to_default() {
        let cfg = toml::from_str::<TuiConfig>("").unwrap();
        assert_eq!(cfg, TuiConfig::default());
        // show_thinking defaults to None (treated as true at startup)
        assert_eq!(cfg.show_thinking, None);
    }

    #[test]
    fn config_show_thinking_round_trips() {
        let cfg = TuiConfig {
            show_thinking: Some(false),
            ..Default::default()
        };
        let text = toml::to_string_pretty(&cfg).unwrap();
        let loaded: TuiConfig = toml::from_str(&text).unwrap();
        assert_eq!(loaded.show_thinking, Some(false));
    }

    #[test]
    fn config_show_thinking_absent_means_none() {
        let cfg: TuiConfig = toml::from_str("[acp]\n").unwrap();
        assert_eq!(cfg.show_thinking, None);
    }

    #[test]
    fn config_acp_args_default_to_acp() {
        let cfg: TuiConfig = toml::from_str("[acp]\n").unwrap();
        assert_eq!(cfg.acp_args(), vec!["--acp"]);
    }

    #[test]
    fn config_acp_transport_round_trips_websocket() {
        let cfg: TuiConfig = toml::from_str(
            "[acp]\ntransport = \"websocket\"\nwebsocket_url = \"ws://127.0.0.1:9123/ws\"\n",
        )
        .unwrap();
        assert_eq!(cfg.acp.transport, Some(AcpTransportMode::WebSocket));
        assert_eq!(
            cfg.acp.websocket_url.as_deref(),
            Some("ws://127.0.0.1:9123/ws")
        );

        let text = toml::to_string_pretty(&cfg).unwrap();
        let loaded: TuiConfig = toml::from_str(&text).unwrap();
        assert_eq!(loaded.acp.transport, Some(AcpTransportMode::WebSocket));
    }

    #[test]
    fn config_load_from_path_missing_returns_default() {
        let dir = unique_temp_dir("cfg-missing");
        let path = dir.join("missing.toml");
        let loaded = TuiConfig::load_from_path(&path);
        assert_eq!(loaded, TuiConfig::default());
    }

    #[test]
    fn config_load_from_path_malformed_returns_default() {
        let dir = unique_temp_dir("cfg-bad");
        let path = dir.join("bad.toml");
        std::fs::write(&path, "not toml ???").unwrap();
        let loaded = TuiConfig::load_from_path(&path);
        assert_eq!(loaded, TuiConfig::default());
    }

    #[test]
    fn config_save_to_path_and_load_round_trip() {
        let dir = unique_temp_dir("cfg-save");
        let path = dir.join("nested").join("qmtui.toml");
        let cfg = TuiConfig {
            theme: Some("base16-ocean".into()),
            show_thinking: None,
            acp: AcpConfig {
                binary_path: Some("/usr/local/bin/qmtcode".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        cfg.save_to_path(&path);
        let loaded = TuiConfig::load_from_path(&path);
        assert_eq!(loaded, cfg);
    }

    #[test]
    #[serial]
    fn config_default_load_save_respects_override_path() {
        let _guard = TestPathGuard::new("cfg-override");
        let cfg = TuiConfig {
            theme: Some("base16-ocean".into()),
            show_thinking: None,
            acp: AcpConfig::default(),
            ..Default::default()
        };
        cfg.save();
        let loaded = TuiConfig::load();
        assert_eq!(loaded, cfg);
    }

    #[test]
    #[serial]
    fn config_save_load_round_trip_preserves_acp_fields() {
        let _guard = TestPathGuard::new("cfg-preserve-acp");
        let cfg = TuiConfig {
            theme: Some("base16-ocean".into()),
            show_thinking: Some(false),
            acp: AcpConfig {
                transport: Some(AcpTransportMode::Stdio),
                websocket_url: Some("ws://127.0.0.1:9123/ws".into()),
                binary_path: Some("/usr/local/bin/qmtcode".into()),
                binary_args: Some(vec!["--acp".into()]),
                auto_start: Some(false),
            },
            ..Default::default()
        };
        cfg.save();
        let loaded = TuiConfig::load();
        assert_eq!(loaded, cfg);
    }

    // ── from_app ──────────────────────────────────────────────────────────────

    #[test]
    fn with_app_settings_preserves_acp_settings() {
        let mut app = App::new();
        app.show_thinking = false;

        let existing = TuiConfig {
            theme: Some("base16-ocean".into()),
            show_thinking: Some(true),
            acp: AcpConfig {
                transport: Some(AcpTransportMode::Stdio),
                websocket_url: Some("ws://127.0.0.1:9123/ws".into()),
                binary_path: Some("/usr/local/bin/qmtcode".into()),
                binary_args: Some(vec!["--acp".into()]),
                auto_start: Some(false),
            },
            ..Default::default()
        };

        let merged = existing.with_app_settings(&app);
        assert_eq!(merged.show_thinking, Some(false));
        assert_eq!(merged.acp, existing.acp);
    }

    // ── delegate_models persistence (via TuiConfig) ────────────────────────

    #[test]
    #[serial]
    fn config_round_trip_with_delegate_models() {
        let _guard = TestPathGuard::new("delegate-models-rt");
        let mut app = App::new();
        app.set_delegate_model_preference("coder", "anthropic", "claude-sonnet");
        app.set_delegate_model_preference("planner", "openai", "gpt-4o");

        let cfg = TuiConfig::load().with_app_settings(&app);
        cfg.save();
        let loaded = TuiConfig::load();

        assert_eq!(
            loaded.delegate_models.get("coder").map(String::as_str),
            Some("anthropic/claude-sonnet")
        );
        assert_eq!(
            loaded.delegate_models.get("planner").map(String::as_str),
            Some("openai/gpt-4o")
        );
    }

    #[test]
    fn config_with_app_settings_empty_delegate_models() {
        let app = App::new();
        let cfg = TuiConfig::default().with_app_settings(&app);
        assert!(cfg.delegate_models.is_empty());
    }

    #[test]
    fn config_delegate_models_deserializes_from_toml() {
        let toml_str = r#"
[delegate_models]
coder = "anthropic/claude-sonnet"
"#;
        let cfg: TuiConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            cfg.delegate_models.get("coder").map(String::as_str),
            Some("anthropic/claude-sonnet")
        );
    }

}
