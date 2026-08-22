use std::time::Duration;

use crossterm::event::{self, Event, EventStream};
use futures::{Stream, StreamExt};
use tokio::{
    sync::mpsc,
    time::{Instant, Interval, MissedTickBehavior},
};

use crate::{
    app::App,
    application::{self, AppEvent},
    server_manager, ui,
};

use super::{ConnectionManagerEvent, EffectExecutor, ServerChannelMsg, terminal::AppTerminal};

const TICK_INTERVAL: Duration = Duration::from_millis(80);

pub(super) async fn run_loop(
    terminal: &mut AppTerminal,
    app: &mut App,
    srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
    conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
    sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
    executor: &mut EffectExecutor<'_>,
) -> anyhow::Result<()> {
    let mut term_events = EventStream::new();
    let mut tick_interval = application_tick_interval();

    loop {
        terminal.draw(|frame| ui::draw(frame, app))?;

        let event = next_event(
            &mut tick_interval,
            conn_rx,
            srv_rx,
            sup_rx,
            &mut term_events,
        )
        .await;

        if let Some(event) = event
            && execute_event(terminal, app, executor, event)?
        {
            return Ok(());
        }

        while let Ok(true) = event::poll(Duration::ZERO) {
            let Ok(event) = event::read() else {
                break;
            };
            if let Some(event) = crossterm_event(event)
                && execute_event(terminal, app, executor, event)?
            {
                return Ok(());
            }
        }
    }
}

fn execute_event(
    terminal: &mut AppTerminal,
    app: &mut App,
    executor: &mut EffectExecutor<'_>,
    event: AppEvent,
) -> anyhow::Result<bool> {
    let effects = application::update(app, event);
    executor.execute(terminal, app, effects)?;
    Ok(app.should_quit)
}

fn application_tick_interval() -> Interval {
    let mut interval = tokio::time::interval_at(Instant::now() + TICK_INTERVAL, TICK_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    interval
}

async fn next_event<S>(
    tick_interval: &mut Interval,
    conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
    srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
    sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
    term_events: &mut S,
) -> Option<AppEvent>
where
    S: Stream<Item = std::io::Result<Event>> + Unpin,
{
    tokio::select! {
        biased;
        _ = tick_interval.tick() => Some(AppEvent::Tick),
        Some(event) = conn_rx.recv() => match event {
            ConnectionManagerEvent::State(state) => Some(AppEvent::Connection(state)),
        },
        Some(ServerChannelMsg::Acp(event)) = srv_rx.recv() => Some(AppEvent::Acp(event)),
        Some(event) = sup_rx.recv() => Some(AppEvent::Supervisor(event)),
        Some(event_result) = term_events.next() => terminal_event(event_result),
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

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use futures::stream;

    use super::*;

    fn test_terminal() -> AppTerminal {
        ratatui::Terminal::with_options(
            ratatui::backend::CrosstermBackend::new(std::io::stdout()),
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Fixed(ratatui::layout::Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap()
    }

    #[test]
    fn execute_event_reports_quit_immediately_after_effect_execution() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = App::new();

        let should_stop = execute_event(
            &mut terminal,
            &mut app,
            &mut executor,
            AppEvent::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        )
        .unwrap();

        assert!(should_stop);
        assert!(app.should_quit);
    }

    #[tokio::test]
    async fn persistent_tick_wins_when_connection_channel_stays_busy() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (_srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (_sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let mut term_events = stream::pending::<std::io::Result<Event>>();
        let mut tick_interval = application_tick_interval();

        conn_tx
            .send(ConnectionManagerEvent::State(
                crate::app::ConnectionEvent::Connecting {
                    attempt: 1,
                    delay_ms: 10,
                },
            ))
            .unwrap();
        tokio::time::sleep(TICK_INTERVAL + Duration::from_millis(10)).await;

        assert!(matches!(
            next_event(
                &mut tick_interval,
                &mut conn_rx,
                &mut srv_rx,
                &mut sup_rx,
                &mut term_events,
            )
            .await,
            Some(AppEvent::Tick)
        ));
        assert_eq!(conn_rx.len(), 1);
    }

    #[tokio::test]
    async fn persistent_tick_reaches_each_interval_boundary_under_busy_traffic() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (_srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (_sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let mut term_events: Pin<Box<dyn Stream<Item = std::io::Result<Event>> + Send>> =
            Box::pin(stream::pending());
        let mut tick_interval = application_tick_interval();

        for attempt in 1..=3 {
            conn_tx
                .send(ConnectionManagerEvent::State(
                    crate::app::ConnectionEvent::Connecting {
                        attempt,
                        delay_ms: 10,
                    },
                ))
                .unwrap();
            tokio::time::sleep(TICK_INTERVAL + Duration::from_millis(10)).await;

            assert!(matches!(
                next_event(
                    &mut tick_interval,
                    &mut conn_rx,
                    &mut srv_rx,
                    &mut sup_rx,
                    &mut term_events,
                )
                .await,
                Some(AppEvent::Tick)
            ));
        }

        assert_eq!(conn_rx.len(), 3);
    }
}
