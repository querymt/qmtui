// src/server_manager.rs - Local qmtcode ACP discovery helpers.
//
// qmtui is ACP-only: it starts `qmtcode --acp` over stdio and does not fall
// back to the deprecated UI API/dashboard transports.

use std::ffi::OsString;
use std::path::Path;
use std::process::Stdio;

/// Events sent from the ACP launcher to the TUI run_loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerEvent {
    /// Launcher is about to start the ACP process.
    Starting,
    /// ACP process is running and the client connection is active.
    Started,
    /// No `qmtcode` binary could be found.
    BinaryNotFound,
    /// ACP process failed to start or connect.
    StartFailed { error: String },
    /// ACP process exited.
    Stopped { reason: String },
}

/// Local ACP process state stored on [`crate::app::App`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ServerState {
    #[default]
    Disabled,
    BinaryNotFound,
    Starting,
    Running,
    StartFailed {
        error: String,
    },
    Restarting {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryDiscovery {
    pub binary: Option<OsString>,
    pub configured_path: Option<String>,
    pub configured_exists: bool,
    pub used_path_lookup: bool,
}

/// Locate the `qmtcode` binary.
///
/// Checks `configured_path` first (if provided), then falls back to a `$PATH`
/// lookup by running `qmtcode --version`.
pub fn find_binary_info(configured_path: Option<&str>) -> BinaryDiscovery {
    let configured_path = configured_path.map(str::to_string);
    let configured_exists = configured_path
        .as_deref()
        .is_some_and(|p| Path::new(p).exists());
    if configured_exists {
        return BinaryDiscovery {
            binary: configured_path.clone().map(OsString::from),
            configured_path,
            configured_exists: true,
            used_path_lookup: false,
        };
    }

    let used_path_lookup = true;
    let ok = std::process::Command::new("qmtcode")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    BinaryDiscovery {
        binary: ok.then(|| OsString::from("qmtcode")),
        configured_path,
        configured_exists,
        used_path_lookup,
    }
}

pub fn find_binary(configured_path: Option<&str>) -> Option<OsString> {
    find_binary_info(configured_path).binary
}

pub fn build_acp_argv(binary: OsString, args: Vec<String>) -> Vec<String> {
    let mut argv = vec![binary.to_string_lossy().to_string()];
    if args.is_empty() {
        argv.push("--acp".to_string());
    } else {
        argv.extend(args);
    }
    argv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acp_args_default_to_acp_flag() {
        assert_eq!(
            build_acp_argv(OsString::from("qmtcode"), vec![]),
            vec!["qmtcode", "--acp"]
        );
    }

    #[test]
    fn acp_args_custom_override_default() {
        assert_eq!(
            build_acp_argv(OsString::from("qmtcode"), vec!["--custom".into()]),
            vec!["qmtcode", "--custom"]
        );
    }

    #[test]
    fn find_binary_info_reports_missing_configured_path_and_path_lookup_attempt() {
        let info = find_binary_info(Some("/nonexistent/path/to/qmtcode"));

        assert_eq!(
            info.configured_path,
            Some("/nonexistent/path/to/qmtcode".into())
        );
        assert!(!info.configured_exists);
        assert!(info.used_path_lookup);
        if let Some(ref binary) = info.binary {
            assert_ne!(binary, "/nonexistent/path/to/qmtcode");
        }
    }
}
