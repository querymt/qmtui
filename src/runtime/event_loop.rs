use std::time::Duration;

use crossterm::event::{Event, EventStream};
use futures::{FutureExt, Stream, StreamExt};
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
const NON_TICK_SOURCE_COUNT: usize = 4;

struct EventScheduler {
    prefer_tick: bool,
    next_non_tick_source: usize,
}

impl EventScheduler {
    fn new() -> Self {
        Self {
            prefer_tick: true,
            next_non_tick_source: 0,
        }
    }

    async fn next_event<S>(
        &mut self,
        tick_interval: &mut Interval,
        conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
        srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
        sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
        term_events: &mut S,
    ) -> Option<AppEvent>
    where
        S: Stream<Item = std::io::Result<Event>> + Unpin,
    {
        let prefer_tick = self.prefer_tick;
        let (is_tick, event) = if prefer_tick {
            tokio::select! {
                biased;
                _ = tick_interval.tick() => (true, None),
                event = self.next_non_tick_event(conn_rx, srv_rx, sup_rx, term_events) => {
                    (false, event)
                },
            }
        } else {
            tokio::select! {
                biased;
                event = self.next_non_tick_event(conn_rx, srv_rx, sup_rx, term_events) => {
                    (false, event)
                },
                _ = tick_interval.tick() => (true, None),
            }
        };

        if is_tick {
            self.prefer_tick = false;
            Some(AppEvent::Tick)
        } else {
            self.prefer_tick = true;
            event
        }
    }

    async fn next_non_tick_event<S>(
        &mut self,
        conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
        srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
        sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
        term_events: &mut S,
    ) -> Option<AppEvent>
    where
        S: Stream<Item = std::io::Result<Event>> + Unpin,
    {
        for offset in 0..NON_TICK_SOURCE_COUNT {
            let source = (self.next_non_tick_source + offset) % NON_TICK_SOURCE_COUNT;
            let event = match source {
                0 => conn_rx.try_recv().ok().map(|event| match event {
                    ConnectionManagerEvent::State(state) => Some(AppEvent::Connection(state)),
                }),
                1 => srv_rx.try_recv().ok().map(|event| match event {
                    ServerChannelMsg::Acp(event) => Some(AppEvent::Acp(event)),
                }),
                2 => sup_rx
                    .try_recv()
                    .ok()
                    .map(|event| Some(AppEvent::Supervisor(event))),
                3 => term_events
                    .next()
                    .now_or_never()
                    .flatten()
                    .map(terminal_event),
                _ => unreachable!("non-tick source index must be in range"),
            };
            if let Some(event) = event {
                self.next_non_tick_source = (source + 1) % NON_TICK_SOURCE_COUNT;
                return event;
            }
        }

        let (source, event) = tokio::select! {
            Some(event) = conn_rx.recv() => (0, match event {
                ConnectionManagerEvent::State(state) => Some(AppEvent::Connection(state)),
            }),
            Some(event) = srv_rx.recv() => (1, match event {
                ServerChannelMsg::Acp(event) => Some(AppEvent::Acp(event)),
            }),
            Some(event) = sup_rx.recv() => (2, Some(AppEvent::Supervisor(event))),
            Some(event_result) = term_events.next() => (3, terminal_event(event_result)),
            else => return std::future::pending().await,
        };
        self.next_non_tick_source = (source + 1) % NON_TICK_SOURCE_COUNT;
        event
    }
}

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
    let mut scheduler = EventScheduler::new();

    loop {
        terminal.draw(|frame| ui::draw(frame, app))?;

        let event = scheduler
            .next_event(
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
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use futures::stream;

    use super::*;
    use crate::acp_state::AcpAppEvent;

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

    fn connection_event(attempt: u32) -> ConnectionManagerEvent {
        ConnectionManagerEvent::State(crate::app::ConnectionEvent::Connecting {
            attempt,
            delay_ms: 10,
        })
    }

    #[tokio::test]
    async fn acp_progresses_while_connection_source_stays_ready() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (_sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let mut term_events = stream::pending::<std::io::Result<Event>>();
        let mut tick_interval = application_tick_interval();
        let mut scheduler = EventScheduler::new();

        for attempt in 1..=8 {
            conn_tx.send(connection_event(attempt)).unwrap();
        }
        srv_tx
            .send(ServerChannelMsg::Acp(AcpAppEvent::Error {
                message: "acp-ready".into(),
            }))
            .unwrap();

        let mut saw_acp = false;
        for _ in 0..NON_TICK_SOURCE_COUNT {
            let event = scheduler
                .next_event(
                    &mut tick_interval,
                    &mut conn_rx,
                    &mut srv_rx,
                    &mut sup_rx,
                    &mut term_events,
                )
                .await;
            saw_acp |= matches!(
                event,
                Some(AppEvent::Acp(AcpAppEvent::Error { message })) if message == "acp-ready"
            );
            if saw_acp {
                break;
            }
        }

        assert!(saw_acp);
        assert!(!conn_rx.is_empty());
    }

    #[tokio::test]
    async fn supervisor_progresses_while_connection_source_stays_ready() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (_srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let mut term_events = stream::pending::<std::io::Result<Event>>();
        let mut tick_interval = application_tick_interval();
        let mut scheduler = EventScheduler::new();

        for attempt in 1..=8 {
            conn_tx.send(connection_event(attempt)).unwrap();
        }
        sup_tx.send(server_manager::ServerEvent::Started).unwrap();

        let mut saw_supervisor = false;
        for _ in 0..NON_TICK_SOURCE_COUNT {
            let event = scheduler
                .next_event(
                    &mut tick_interval,
                    &mut conn_rx,
                    &mut srv_rx,
                    &mut sup_rx,
                    &mut term_events,
                )
                .await;
            saw_supervisor |= matches!(
                event,
                Some(AppEvent::Supervisor(server_manager::ServerEvent::Started))
            );
            if saw_supervisor {
                break;
            }
        }

        assert!(saw_supervisor);
        assert!(!conn_rx.is_empty());
    }

    #[tokio::test]
    async fn continuous_terminal_input_does_not_starve_other_sources_or_ticks() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let mut term_events = stream::repeat_with(|| {
            Ok(Event::Key(KeyEvent::new(
                KeyCode::Char('x'),
                KeyModifiers::NONE,
            )))
        });
        let mut tick_interval = application_tick_interval();
        let mut scheduler = EventScheduler::new();

        conn_tx.send(connection_event(1)).unwrap();
        srv_tx
            .send(ServerChannelMsg::Acp(AcpAppEvent::Error {
                message: "acp-ready".into(),
            }))
            .unwrap();
        sup_tx.send(server_manager::ServerEvent::Started).unwrap();
        tokio::time::sleep(TICK_INTERVAL + Duration::from_millis(20)).await;

        let mut saw_connection = false;
        let mut saw_acp = false;
        let mut saw_supervisor = false;
        let mut saw_tick = false;
        let mut terminal_events = 0;
        for _ in 0..(NON_TICK_SOURCE_COUNT * 2 + 2) {
            match scheduler
                .next_event(
                    &mut tick_interval,
                    &mut conn_rx,
                    &mut srv_rx,
                    &mut sup_rx,
                    &mut term_events,
                )
                .await
            {
                Some(AppEvent::Connection(_)) => saw_connection = true,
                Some(AppEvent::Acp(_)) => saw_acp = true,
                Some(AppEvent::Supervisor(_)) => saw_supervisor = true,
                Some(AppEvent::Tick) => saw_tick = true,
                Some(AppEvent::Key(_)) => terminal_events += 1,
                _ => {}
            }
        }

        assert!(saw_connection);
        assert!(saw_acp);
        assert!(saw_supervisor);
        assert!(saw_tick);
        assert!(terminal_events >= 2);
    }

    #[tokio::test]
    async fn tick_progresses_while_non_tick_sources_are_busy() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let mut term_events = stream::pending::<std::io::Result<Event>>();
        let mut tick_interval = application_tick_interval();
        let mut scheduler = EventScheduler::new();

        conn_tx.send(connection_event(1)).unwrap();
        srv_tx
            .send(ServerChannelMsg::Acp(AcpAppEvent::Error {
                message: "busy".into(),
            }))
            .unwrap();
        sup_tx.send(server_manager::ServerEvent::Started).unwrap();
        tokio::time::sleep(TICK_INTERVAL + Duration::from_millis(20)).await;

        assert!(matches!(
            scheduler
                .next_event(
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

    #[tokio::test]
    async fn overdue_tick_alternates_with_ready_non_tick_work() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (_srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (_sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let mut term_events = stream::pending::<std::io::Result<Event>>();
        let mut tick_interval = application_tick_interval();
        let mut scheduler = EventScheduler::new();

        for attempt in 1..=4 {
            conn_tx.send(connection_event(attempt)).unwrap();
        }

        for expect_tick in [true, false, true, false] {
            tokio::time::sleep(TICK_INTERVAL * 2 + Duration::from_millis(20)).await;
            let event = scheduler
                .next_event(
                    &mut tick_interval,
                    &mut conn_rx,
                    &mut srv_rx,
                    &mut sup_rx,
                    &mut term_events,
                )
                .await;
            assert_eq!(matches!(event, Some(AppEvent::Tick)), expect_tick);
        }

        assert_eq!(conn_rx.len(), 2);
    }
}
