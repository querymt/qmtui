use std::time::Duration;

use crossterm::event::{self, Event, EventStream};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::{
    app::{self, App},
    handlers::{AppAction, handle_key, handle_mouse},
    protocol::ClientMsg,
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

pub(super) async fn run_loop(
    terminal: &mut AppTerminal,
    app: &mut App,
    srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
    conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
    sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
    cmd_tx: &mpsc::UnboundedSender<ClientMsg>,
) -> anyhow::Result<()> {
    let mut term_events = EventStream::new();

    loop {
        app.tick = tick_from_elapsed(app.started_at.elapsed());
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
                            cmd_tx.send(ClientMsg::Init)?;
                            cmd_tx.send(ClientMsg::list_sessions_browse())?;
                            cmd_tx.send(ClientMsg::ListAllModels { refresh: false })?;
                            if let Some(session_id) = app.session_id.clone() {
                                if let Some(node_id) = app.session_remote_node_id(&session_id) {
                                    cmd_tx.send(ClientMsg::AttachRemoteSession {
                                        node_id: node_id.to_string(),
                                        session_id,
                                    })?;
                                } else if app.is_remote_session_id(&session_id) {
                                    app.set_status(
                                        app::LogLevel::Warn,
                                        "session",
                                        "remote session is missing node id; reconnect attach skipped",
                                    );
                                } else {
                                    cmd_tx.send(ClientMsg::LoadSession {
                                        session_id: session_id.clone(),
                                        cwd: app.current_session_cwd(),
                                    })?;
                                    cmd_tx.send(ClientMsg::SubscribeSession {
                                        session_id,
                                        agent_id: app.agent_id.clone(),
                                    })?;
                                }
                            }
                        } else if was_connected && app.conn == app::ConnState::Disconnected {
                            app.set_status(
                                app::LogLevel::Warn,
                                "connection",
                                "connection lost - reconnecting...",
                            );
                        }
                    }
                }
            }
            Some(ServerChannelMsg::Acp(event)) = srv_rx.recv() => {
                for reply in app.handle_acp_event(event) {
                    if let ClientMsg::LoadSession { ref session_id, .. } = reply {
                        let sid = session_id.clone();
                        cmd_tx.send(reply)?;
                        cmd_tx.send(ClientMsg::SubscribeSession {
                            session_id: sid,
                            agent_id: app.agent_id.clone(),
                        })?;
                    } else {
                        cmd_tx.send(reply)?;
                    }
                }
            }
            Some(sup_event) = sup_rx.recv() => {
                match sup_event {
                    ServerEvent::Starting => {
                        app.server_state = ServerState::Starting;
                        if app.conn != app::ConnState::Connected {
                            app.set_status(app::LogLevel::Info, "acp", "starting qmtcode ACP agent...");
                        }
                    }
                    ServerEvent::Started => {
                        app.server_state = ServerState::Running;
                        if app.conn != app::ConnState::Connected {
                            app.set_status(app::LogLevel::Info, "acp", "qmtcode ACP agent started");
                        }
                    }
                    ServerEvent::BinaryNotFound => {
                        app.server_state = ServerState::BinaryNotFound;
                        if app.conn != app::ConnState::Connected {
                            app.set_status(
                                app::LogLevel::Warn,
                                "acp",
                                "qmtcode not found; install it or set acp.binary_path in ~/.qmt/qmtui.toml",
                            );
                        }
                    }
                    ServerEvent::StartFailed { error } => {
                        app.server_state = ServerState::StartFailed { error: error.clone() };
                        app.set_status(
                            app::LogLevel::Error,
                            "acp",
                            format!("ACP start failed: {error}"),
                        );
                    }
                    ServerEvent::Stopped { reason } => {
                        app.server_state = ServerState::Restarting { reason: reason.clone() };
                        app.set_status(
                            app::LogLevel::Warn,
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

    use super::tick_from_elapsed;

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
