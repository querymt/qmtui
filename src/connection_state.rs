#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConnState {
    Connecting,
    Connected,
    Disconnected,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum ServerState {
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

pub(crate) struct ConnectionState {
    pub(crate) launch_cwd: Option<String>,
    pub(crate) conn: ConnState,
    pub(crate) reconnect_attempt: u32,
    pub(crate) reconnect_delay_ms: Option<u64>,
    pub(crate) server_state: ServerState,
}

impl ConnectionState {
    pub(crate) fn new() -> Self {
        Self {
            launch_cwd: None,
            conn: ConnState::Connecting,
            reconnect_attempt: 0,
            reconnect_delay_ms: None,
            server_state: ServerState::Disabled,
        }
    }

    pub(crate) fn apply_connecting(&mut self, attempt: u32, delay_ms: u64) {
        self.conn = ConnState::Connecting;
        self.reconnect_attempt = attempt;
        self.reconnect_delay_ms = Some(delay_ms);
    }

    pub(crate) fn apply_connected(&mut self) {
        self.conn = ConnState::Connected;
        self.reconnect_attempt = 0;
        self.reconnect_delay_ms = None;
    }

    pub(crate) fn apply_disconnected(&mut self) {
        self.conn = ConnState::Disconnected;
        self.reconnect_delay_ms = None;
    }

    pub(crate) fn apply_server_starting(&mut self) {
        self.server_state = ServerState::Starting;
    }

    pub(crate) fn apply_server_started(&mut self) {
        self.server_state = ServerState::Running;
    }

    pub(crate) fn apply_server_binary_not_found(&mut self) {
        self.server_state = ServerState::BinaryNotFound;
    }

    pub(crate) fn apply_server_start_failed(&mut self, error: String) {
        self.server_state = ServerState::StartFailed { error };
    }

    pub(crate) fn apply_server_stopped(&mut self, reason: String) {
        self.server_state = ServerState::Restarting { reason };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_state() -> ConnectionState {
        ConnectionState {
            launch_cwd: Some("/workspace".into()),
            conn: ConnState::Connected,
            reconnect_attempt: 7,
            reconnect_delay_ms: Some(8000),
            server_state: ServerState::Running,
        }
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = ConnectionState::new();

        assert_eq!(state.launch_cwd, None);
        assert_eq!(state.conn, ConnState::Connecting);
        assert_eq!(state.reconnect_attempt, 0);
        assert_eq!(state.reconnect_delay_ms, None);
        assert_eq!(state.server_state, ServerState::Disabled);
    }

    #[test]
    fn connection_transitions_replace_retry_metadata_and_preserve_launch_and_server_state() {
        let mut state = seeded_state();

        state.apply_connecting(3, 2000);
        assert_eq!(state.conn, ConnState::Connecting);
        assert_eq!(state.reconnect_attempt, 3);
        assert_eq!(state.reconnect_delay_ms, Some(2000));
        assert_eq!(state.launch_cwd.as_deref(), Some("/workspace"));
        assert_eq!(state.server_state, ServerState::Running);

        state.apply_connected();
        assert_eq!(state.conn, ConnState::Connected);
        assert_eq!(state.reconnect_attempt, 0);
        assert_eq!(state.reconnect_delay_ms, None);
        assert_eq!(state.launch_cwd.as_deref(), Some("/workspace"));
        assert_eq!(state.server_state, ServerState::Running);
    }

    #[test]
    fn disconnected_clears_delay_and_preserves_stale_attempt_launch_and_server_state() {
        let mut state = seeded_state();

        state.apply_disconnected();

        assert_eq!(state.conn, ConnState::Disconnected);
        assert_eq!(state.reconnect_attempt, 7);
        assert_eq!(state.reconnect_delay_ms, None);
        assert_eq!(state.launch_cwd.as_deref(), Some("/workspace"));
        assert_eq!(state.server_state, ServerState::Running);
    }

    #[test]
    fn server_transitions_replace_only_server_state_and_preserve_payloads() {
        let mut state = seeded_state();
        let expected_connection = (
            state.launch_cwd.clone(),
            state.conn,
            state.reconnect_attempt,
            state.reconnect_delay_ms,
        );

        state.apply_server_starting();
        assert_eq!(state.server_state, ServerState::Starting);
        state.apply_server_started();
        assert_eq!(state.server_state, ServerState::Running);
        state.apply_server_binary_not_found();
        assert_eq!(state.server_state, ServerState::BinaryNotFound);
        state.apply_server_start_failed("invalid command".into());
        assert_eq!(
            state.server_state,
            ServerState::StartFailed {
                error: "invalid command".into()
            }
        );
        state.apply_server_stopped("process exited".into());
        assert_eq!(
            state.server_state,
            ServerState::Restarting {
                reason: "process exited".into()
            }
        );
        assert_eq!(
            (
                state.launch_cwd,
                state.conn,
                state.reconnect_attempt,
                state.reconnect_delay_ms,
            ),
            expected_connection
        );
    }
}
