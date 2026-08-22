use std::time::Duration;

use crossterm::event::{self, Event, EventStream};
use futures::StreamExt;
use tokio::sync::mpsc;

use crate::{
    app::App,
    application::{self, AppEvent},
    server_manager, ui,
};

use super::{ConnectionManagerEvent, EffectExecutor, ServerChannelMsg, terminal::AppTerminal};

pub(super) async fn run_loop(
    terminal: &mut AppTerminal,
    app: &mut App,
    srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
    conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
    sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
    executor: &mut EffectExecutor<'_>,
) -> anyhow::Result<()> {
    let mut term_events = EventStream::new();

    loop {
        terminal.draw(|frame| ui::draw(frame, app))?;

        let event = tokio::select! {
            biased;
            Some(event) = conn_rx.recv() => match event {
                ConnectionManagerEvent::State(state) => Some(AppEvent::Connection(state)),
            },
            Some(ServerChannelMsg::Acp(event)) = srv_rx.recv() => Some(AppEvent::Acp(event)),
            Some(event) = sup_rx.recv() => Some(AppEvent::Supervisor(event)),
            Some(event_result) = term_events.next() => terminal_event(event_result),
            _ = tokio::time::sleep(Duration::from_millis(80)) => Some(AppEvent::Tick),
        };

        if let Some(event) = event {
            let effects = application::update(app, event);
            executor.execute(terminal, app, effects)?;
        }

        while let Ok(true) = event::poll(Duration::from_millis(0)) {
            let Ok(event) = event::read() else {
                break;
            };
            if let Some(event) = crossterm_event(event) {
                let effects = application::update(app, event);
                executor.execute(terminal, app, effects)?;
            }
            if app.should_quit {
                break;
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

fn terminal_event(event: std::io::Result<Event>) -> Option<AppEvent> {
    event.ok().and_then(crossterm_event)
}

fn crossterm_event(event: Event) -> Option<AppEvent> {
    match event {
        Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
            Some(AppEvent::Key(key))
        }
        Event::Mouse(mouse) => Some(AppEvent::Mouse(mouse)),
        _ => None,
    }
}
