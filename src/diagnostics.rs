use std::time::{Duration, Instant};

const MAX_DIAGNOSTIC_ENTRIES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }

    pub(crate) fn next(self) -> Self {
        match self {
            Self::Trace => Self::Debug,
            Self::Debug => Self::Info,
            Self::Info => Self::Warn,
            Self::Warn => Self::Error,
            Self::Error => Self::Trace,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppLogEntry {
    pub(crate) elapsed: Duration,
    pub(crate) level: LogLevel,
    pub(crate) target: &'static str,
    pub(crate) message: String,
}

pub(crate) struct DiagnosticsState {
    pub(crate) started_at: Instant,
    pub(crate) logs: Vec<AppLogEntry>,
    pub(crate) log_cursor: usize,
    pub(crate) log_filter: String,
    pub(crate) log_level_filter: LogLevel,
    pub(crate) status: String,
}

impl DiagnosticsState {
    pub(crate) fn new() -> Self {
        Self {
            started_at: Instant::now(),
            logs: Vec::new(),
            log_cursor: 0,
            log_filter: String::new(),
            log_level_filter: LogLevel::Info,
            status: "connecting...".into(),
        }
    }

    pub(crate) fn push_log(
        &mut self,
        level: LogLevel,
        target: &'static str,
        message: impl Into<String>,
    ) {
        let message = message.into();
        if self.logs.last().is_some_and(|entry| {
            entry.level == level && entry.target == target && entry.message == message
        }) {
            return;
        }
        if self.logs.len() >= MAX_DIAGNOSTIC_ENTRIES {
            let evicted_was_visible = self
                .logs
                .first()
                .is_some_and(|entry| self.log_is_visible(entry));
            self.logs.remove(0);
            if evicted_was_visible {
                self.log_cursor = self.log_cursor.saturating_sub(1);
            }
        }
        self.logs.push(AppLogEntry {
            elapsed: self.started_at.elapsed(),
            level,
            target,
            message,
        });
        self.log_cursor = self
            .log_cursor
            .min(self.filtered_logs().len().saturating_sub(1));
    }

    pub(crate) fn set_status(
        &mut self,
        level: LogLevel,
        target: &'static str,
        message: impl Into<String>,
    ) {
        let message = message.into();
        self.status = message.clone();
        self.push_log(level, target, message);
    }

    fn log_is_visible(&self, entry: &AppLogEntry) -> bool {
        if entry.level < self.log_level_filter {
            return false;
        }
        let query = self.log_filter.to_lowercase();
        query.is_empty()
            || entry.message.to_lowercase().contains(&query)
            || entry.target.to_lowercase().contains(&query)
            || entry.level.label().to_lowercase().contains(&query)
    }

    pub(crate) fn filtered_logs(&self) -> Vec<&AppLogEntry> {
        self.logs
            .iter()
            .filter(|entry| self.log_is_visible(entry))
            .collect()
    }

    pub(crate) fn cycle_log_level_filter(&mut self) {
        self.log_level_filter = self.log_level_filter.next();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{DiagnosticsState, LogLevel, MAX_DIAGNOSTIC_ENTRIES};

    #[test]
    fn constructor_uses_expected_defaults() {
        let before = Instant::now();
        let diagnostics = DiagnosticsState::new();
        let after = Instant::now();

        assert!(diagnostics.started_at >= before);
        assert!(diagnostics.started_at <= after);
        assert!(diagnostics.logs.is_empty());
        assert_eq!(diagnostics.log_cursor, 0);
        assert!(diagnostics.log_filter.is_empty());
        assert_eq!(diagnostics.log_level_filter, LogLevel::Info);
        assert_eq!(diagnostics.status, "connecting...");
    }

    #[test]
    fn push_log_suppresses_only_consecutive_exact_duplicates() {
        let mut diagnostics = DiagnosticsState::new();

        diagnostics.push_log(LogLevel::Info, "server", "starting local server");
        diagnostics.push_log(LogLevel::Info, "server", "starting local server");
        diagnostics.push_log(LogLevel::Warn, "server", "starting local server");
        diagnostics.push_log(LogLevel::Warn, "runtime", "starting local server");
        diagnostics.push_log(LogLevel::Warn, "runtime", "ready");
        diagnostics.push_log(LogLevel::Info, "server", "starting local server");

        assert_eq!(diagnostics.logs.len(), 5);
        assert_eq!(diagnostics.logs[0].level, LogLevel::Info);
        assert_eq!(diagnostics.logs[1].level, LogLevel::Warn);
        assert_eq!(diagnostics.logs[2].target, "runtime");
        assert_eq!(diagnostics.logs[3].message, "ready");
        assert_eq!(diagnostics.logs[4].level, diagnostics.logs[0].level);
        assert_eq!(diagnostics.logs[4].target, diagnostics.logs[0].target);
        assert_eq!(diagnostics.logs[4].message, diagnostics.logs[0].message);
    }

    #[test]
    fn push_log_bounds_capacity_and_retains_newest_entries() {
        let mut diagnostics = DiagnosticsState::new();
        for index in 0..=MAX_DIAGNOSTIC_ENTRIES {
            diagnostics.push_log(LogLevel::Info, "test", format!("entry-{index}"));
        }

        assert_eq!(diagnostics.logs.len(), MAX_DIAGNOSTIC_ENTRIES);
        assert_eq!(diagnostics.logs[0].message, "entry-1");
        assert_eq!(diagnostics.logs.last().unwrap().message, "entry-1024");
    }

    #[test]
    fn eviction_adjusts_cursor_only_when_evicted_entry_is_visible() {
        let mut visible = DiagnosticsState::new();
        visible.log_level_filter = LogLevel::Info;
        visible.push_log(LogLevel::Info, "test", "visible-oldest");
        for index in 1..MAX_DIAGNOSTIC_ENTRIES {
            visible.push_log(LogLevel::Info, "test", format!("visible-{index}"));
        }
        visible.log_cursor = 10;
        visible.push_log(LogLevel::Info, "test", "visible-newest");
        assert_eq!(visible.log_cursor, 9);

        let mut hidden = DiagnosticsState::new();
        hidden.log_level_filter = LogLevel::Info;
        hidden.push_log(LogLevel::Debug, "test", "hidden-oldest");
        for index in 1..MAX_DIAGNOSTIC_ENTRIES {
            hidden.push_log(LogLevel::Info, "test", format!("visible-{index}"));
        }
        hidden.log_cursor = 10;
        hidden.push_log(LogLevel::Info, "test", "visible-newest");
        assert_eq!(hidden.log_cursor, 10);
    }

    #[test]
    fn eviction_clamps_cursor_to_filtered_list() {
        let mut diagnostics = DiagnosticsState::new();
        diagnostics.log_filter = "matching".into();
        diagnostics.push_log(LogLevel::Info, "test", "matching oldest");
        for index in 1..MAX_DIAGNOSTIC_ENTRIES {
            diagnostics.push_log(LogLevel::Info, "test", format!("hidden-{index}"));
        }
        diagnostics.log_cursor = usize::MAX;
        diagnostics.push_log(LogLevel::Info, "test", "still hidden");

        assert!(diagnostics.filtered_logs().is_empty());
        assert_eq!(diagnostics.log_cursor, 0);
    }

    #[test]
    fn set_status_updates_visible_status_and_appends_log() {
        let mut diagnostics = DiagnosticsState::new();

        diagnostics.set_status(LogLevel::Info, "connection", "connected");

        assert_eq!(diagnostics.status, "connected");
        let last = diagnostics.logs.last().expect("missing log entry");
        assert_eq!(last.level, LogLevel::Info);
        assert_eq!(last.target, "connection");
        assert_eq!(last.message, "connected");
    }

    #[test]
    fn filtered_logs_match_message_target_and_level_case_insensitively() {
        let mut diagnostics = DiagnosticsState::new();
        diagnostics.push_log(LogLevel::Debug, "activity", "ready");
        diagnostics.push_log(LogLevel::Warn, "Server", "waiting for lock");
        diagnostics.push_log(LogLevel::Error, "runtime", "START FAILED");
        diagnostics.log_level_filter = LogLevel::Trace;

        diagnostics.log_filter = "READY".into();
        assert_eq!(diagnostics.filtered_logs()[0].target, "activity");

        diagnostics.log_filter = "server".into();
        assert_eq!(diagnostics.filtered_logs()[0].level, LogLevel::Warn);

        diagnostics.log_filter = "wArN".into();
        assert_eq!(diagnostics.filtered_logs()[0].message, "waiting for lock");

        diagnostics.log_filter = "failed".into();
        assert_eq!(diagnostics.filtered_logs()[0].level, LogLevel::Error);
    }

    #[test]
    fn filtered_logs_apply_minimum_level_threshold() {
        let mut diagnostics = DiagnosticsState::new();
        diagnostics.push_log(LogLevel::Debug, "activity", "ready");
        diagnostics.push_log(LogLevel::Warn, "server", "waiting for lock");
        diagnostics.push_log(LogLevel::Error, "server", "start failed");

        diagnostics.log_level_filter = LogLevel::Warn;
        let filtered = diagnostics.filtered_logs();

        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|entry| entry.level >= LogLevel::Warn));
    }

    #[test]
    fn cycle_log_level_filter_wraps_in_display_order() {
        let mut diagnostics = DiagnosticsState::new();
        diagnostics.log_level_filter = LogLevel::Trace;

        for expected in [
            LogLevel::Debug,
            LogLevel::Info,
            LogLevel::Warn,
            LogLevel::Error,
            LogLevel::Trace,
        ] {
            diagnostics.cycle_log_level_filter();
            assert_eq!(diagnostics.log_level_filter, expected);
        }
    }

    #[test]
    fn log_elapsed_uses_diagnostics_started_at() {
        let mut diagnostics = DiagnosticsState::new();
        let started_at = diagnostics.started_at;
        let elapsed_before = started_at.elapsed();

        diagnostics.push_log(LogLevel::Info, "server", "ready");

        let elapsed = diagnostics.logs[0].elapsed;
        let elapsed_after = started_at.elapsed();
        assert!(elapsed >= elapsed_before);
        assert!(elapsed <= elapsed_after);
    }
}
