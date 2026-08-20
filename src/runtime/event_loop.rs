use std::time::Duration;

use crossterm::event::{self, Event, EventStream};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::{
    app::{self, App},
    command::Command,
    diagnostics::LogLevel,
    handlers::{AppAction, handle_key, handle_mouse},
    server_manager::{self, ServerEvent, ServerState},
    ui,
};

use super::{
    ConnectionManagerEvent, ServerChannelMsg,
    terminal::{AppTerminal, open_external_editor_with_terminal},
};

/// Derive a UI tick from wall-clock elapsed time.
/// Each tick step is about 80 ms so animations do not depend on event-loop activity.
fn tick_from_elapsed(elapsed: Duration) -> u64 {
    (elapsed.as_millis() / 80) as u64
}

fn send_command(cmd_tx: &mpsc::UnboundedSender<Command>, command: Command) -> anyhow::Result<()> {
    cmd_tx.send(command)?;
    Ok(())
}

fn reconnect_session_commands(app: &mut App) -> Vec<Command> {
    let Some(session_id) = app.session_id.clone() else {
        return Vec::new();
    };

    if let Some(node_id) = app.session_remote_node_id(&session_id) {
        return vec![Command::AttachRemoteSession {
            node_id: node_id.to_string(),
            session_id,
        }];
    }
    if app.is_remote_session_id(&session_id) {
        app.set_status(
            LogLevel::Warn,
            "session",
            "remote session is missing node id; reconnect attach skipped",
        );
        return Vec::new();
    }

    Command::load_session_commands(session_id, app.current_session_cwd(), app.agent_id.clone())
        .into()
}

pub(super) async fn run_loop(
    terminal: &mut AppTerminal,
    app: &mut App,
    srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
    conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
    sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
    cmd_tx: &mpsc::UnboundedSender<Command>,
) -> anyhow::Result<()> {
    let mut term_events = EventStream::new();

    loop {
        app.tick = tick_from_elapsed(app.diagnostics.started_at.elapsed());
        app.clear_expired_cancel_confirm();
        terminal.draw(|frame| ui::draw(frame, app))?;

        tokio::select! {
            biased;
            Some(event) = conn_rx.recv() => {
                match event {
                    ConnectionManagerEvent::State(state) => {
                        let was_connected = app.conn == app::ConnState::Connected;
                        app.handle_connection_event(state);
                        if app.conn == app::ConnState::Connected {
                            cmd_tx.send(Command::Init)?;
                            cmd_tx.send(Command::list_sessions_browse())?;
                            cmd_tx.send(Command::ListAllModels { refresh: false })?;
                            for command in reconnect_session_commands(app) {
                                cmd_tx.send(command)?;
                            }
                        } else if was_connected && app.conn == app::ConnState::Disconnected {
                            app.set_status(
                                LogLevel::Warn,
                                "connection",
                                "connection lost - reconnecting...",
                            );
                        }
                    }
                }
            }
            Some(ServerChannelMsg::Acp(event)) = srv_rx.recv() => {
                for command in app.handle_acp_event(event) {
                    send_command(cmd_tx, command)?;
                }
            }
            Some(sup_event) = sup_rx.recv() => {
                match sup_event {
                    ServerEvent::Starting => {
                        app.server_state = ServerState::Starting;
                        if app.conn != app::ConnState::Connected {
                            app.set_status(LogLevel::Info, "acp", "starting qmtcode ACP agent...");
                        }
                    }
                    ServerEvent::Started => {
                        app.server_state = ServerState::Running;
                        if app.conn != app::ConnState::Connected {
                            app.set_status(LogLevel::Info, "acp", "qmtcode ACP agent started");
                        }
                    }
                    ServerEvent::BinaryNotFound => {
                        app.server_state = ServerState::BinaryNotFound;
                        if app.conn != app::ConnState::Connected {
                            app.set_status(
                                LogLevel::Warn,
                                "acp",
                                "qmtcode not found; install it or set acp.binary_path in ~/.qmt/qmtui.toml",
                            );
                        }
                    }
                    ServerEvent::StartFailed { error } => {
                        app.server_state = ServerState::StartFailed { error: error.clone() };
                        app.set_status(
                            LogLevel::Error,
                            "acp",
                            format!("ACP start failed: {error}"),
                        );
                    }
                    ServerEvent::Stopped { reason } => {
                        app.server_state = ServerState::Restarting { reason: reason.clone() };
                        app.set_status(
                            LogLevel::Warn,
                            "acp",
                            format!("ACP agent stopped ({reason})"),
                        );
                    }
                }
            }
            Some(event_result) = term_events.next() => {
                match event_result {
                    Ok(Event::Key(key)) if key.kind == crossterm::event::KeyEventKind::Press => {
                        let action = handle_key(app, key, cmd_tx)?;
                        if matches!(action, AppAction::OpenExternalEditor) {
                            open_external_editor_with_terminal(terminal, app)?;
                        }
                        if app.should_quit {
                            return Ok(());
                        }
                        while let Ok(true) = event::poll(Duration::from_millis(0)) {
                            match event::read() {
                                Ok(Event::Key(key)) if key.kind == crossterm::event::KeyEventKind::Press => {
                                    let action = handle_key(app, key, cmd_tx)?;
                                    if matches!(action, AppAction::OpenExternalEditor) {
                                        open_external_editor_with_terminal(terminal, app)?;
                                    }
                                    if app.should_quit {
                                        return Ok(());
                                    }
                                }
                                Ok(Event::Mouse(mouse)) => {
                                    handle_mouse(app, mouse);
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(Event::Mouse(mouse)) => {
                        handle_mouse(app, mouse);
                    }
                    _ => {}
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(80)) => {}
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::sync::mpsc;

    use super::{reconnect_session_commands, send_command, tick_from_elapsed};
    use crate::app::App;
    use crate::command::Command;

    #[test]
    fn send_command_sends_each_command_once_without_implicit_subscribe() {
        let (tx, mut rx) = mpsc::unbounded_channel();

        send_command(
            &tx,
            Command::LoadSession {
                session_id: "session-1".into(),
                cwd: Some("/repo".into()),
            },
        )
        .unwrap();

        assert!(matches!(
            rx.try_recv().unwrap(),
            Command::LoadSession {
                session_id,
                cwd: Some(cwd),
            } if session_id == "session-1" && cwd == "/repo"
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn reconnect_attaches_remembered_remote_session_without_loading() {
        let mut app = App::new();
        app.session_id = Some("remote-1".into());
        app.remember_remote_session_location("remote-1", "node-1", Some("/remote/repo".into()));

        assert_eq!(
            reconnect_session_commands(&mut app),
            vec![Command::AttachRemoteSession {
                node_id: "node-1".into(),
                session_id: "remote-1".into(),
            }]
        );
    }

    #[test]
    fn reconnect_warns_for_remote_session_missing_node() {
        let mut app = App::new();
        app.session_id = Some("remote-1".into());
        app.session_groups = vec![crate::domain::session::SessionGroup {
            sessions: vec![crate::domain::session::SessionSummary {
                session_id: "remote-1".into(),
                node: Some("remote".into()),
                ..Default::default()
            }],
            ..Default::default()
        }];

        assert!(reconnect_session_commands(&mut app).is_empty());
        assert!(app.diagnostics.status.contains("missing node id"));
    }

    #[test]
    fn reconnect_loads_and_subscribes_local_session() {
        let mut app = App::new();
        app.session_id = Some("local-1".into());
        app.agent_id = Some("agent-1".into());
        app.launch_cwd = Some("/local/repo".into());

        assert_eq!(
            reconnect_session_commands(&mut app),
            vec![
                Command::LoadSession {
                    session_id: "local-1".into(),
                    cwd: Some("/local/repo".into()),
                },
                Command::SubscribeSession {
                    session_id: "local-1".into(),
                    agent_id: Some("agent-1".into()),
                },
            ]
        );
    }

    #[test]
    fn reconnect_without_active_session_does_nothing() {
        assert!(reconnect_session_commands(&mut App::new()).is_empty());
    }

    #[test]
    fn tick_from_elapsed_zero_is_zero() {
        assert_eq!(tick_from_elapsed(Duration::ZERO), 0);
    }

    #[test]
    fn tick_from_elapsed_advances_every_80ms() {
        assert_eq!(tick_from_elapsed(Duration::from_millis(0)), 0);
        assert_eq!(tick_from_elapsed(Duration::from_millis(79)), 0);
        assert_eq!(tick_from_elapsed(Duration::from_millis(80)), 1);
        assert_eq!(tick_from_elapsed(Duration::from_millis(159)), 1);
        assert_eq!(tick_from_elapsed(Duration::from_millis(160)), 2);
    }

    #[test]
    fn tick_from_elapsed_spinner_frame_changes_every_two_ticks() {
        let tick_at_0ms = tick_from_elapsed(Duration::from_millis(0));
        let tick_at_150ms = tick_from_elapsed(Duration::from_millis(150));
        let tick_at_160ms = tick_from_elapsed(Duration::from_millis(160));

        assert_eq!(tick_at_0ms / 2, tick_at_150ms / 2);
        assert_ne!(tick_at_0ms / 2, tick_at_160ms / 2);
    }
}
