use std::{io, sync::mpsc::TryRecvError, time::Duration};

use anyhow::Context;
use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use tokio::{
    sync::mpsc,
    time::{Instant, Interval, MissedTickBehavior},
};

use crate::{
    app::App,
    application::{self, AppEvent},
    navigation_state::{Popup, Screen},
    server_manager, ui,
};

use super::{
    ConnectionManagerEvent, EffectExecutor, ServerChannelMsg,
    terminal::{AppTerminal, TerminalEventReceiver, TerminalEventResult},
};

const TICK_INTERVAL: Duration = Duration::from_millis(80);
const MAX_TERMINAL_BATCH_EVENTS: usize = 32;
const MAX_TERMINAL_BATCH_DURATION: Duration = Duration::from_millis(8);
const NON_TICK_SOURCE_COUNT: usize = 4;

struct EventScheduler {
    prefer_tick: bool,
    next_non_tick_source: usize,
    pending_terminal: Option<TerminalEventResult>,
}

enum ScheduledEvent {
    Application(Box<AppEvent>),
    Terminal(Option<TerminalEventResult>),
}

impl EventScheduler {
    fn new() -> Self {
        Self {
            prefer_tick: true,
            next_non_tick_source: 0,
            pending_terminal: None,
        }
    }

    fn defer_terminal(&mut self, event: TerminalEventResult) {
        debug_assert!(self.pending_terminal.is_none());
        self.pending_terminal = Some(event);
    }

    async fn next_event(
        &mut self,
        tick_interval: &mut Interval,
        conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
        srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
        sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
        term_events: &TerminalEventReceiver,
    ) -> ScheduledEvent {
        let prefer_tick = self.prefer_tick;
        let (is_tick, event) = if prefer_tick {
            tokio::select! {
                biased;
                _ = tick_interval.tick() => (true, None),
                event = self.next_non_tick_event(conn_rx, srv_rx, sup_rx, term_events) => {
                    (false, Some(event))
                },
            }
        } else {
            tokio::select! {
                biased;
                event = self.next_non_tick_event(conn_rx, srv_rx, sup_rx, term_events) => {
                    (false, Some(event))
                },
                _ = tick_interval.tick() => (true, None),
            }
        };

        if is_tick {
            self.prefer_tick = false;
            ScheduledEvent::Application(Box::new(AppEvent::Tick))
        } else {
            self.prefer_tick = true;
            event.expect("non-tick event must be present")
        }
    }

    async fn next_non_tick_event(
        &mut self,
        conn_rx: &mut mpsc::UnboundedReceiver<ConnectionManagerEvent>,
        srv_rx: &mut mpsc::UnboundedReceiver<ServerChannelMsg>,
        sup_rx: &mut mpsc::UnboundedReceiver<server_manager::ServerEvent>,
        term_events: &TerminalEventReceiver,
    ) -> ScheduledEvent {
        for offset in 0..NON_TICK_SOURCE_COUNT {
            let source = (self.next_non_tick_source + offset) % NON_TICK_SOURCE_COUNT;
            let event = match source {
                0 => conn_rx.try_recv().ok().map(|event| match event {
                    ConnectionManagerEvent::State(state) => {
                        ScheduledEvent::Application(Box::new(AppEvent::Connection(state)))
                    }
                }),
                1 => srv_rx.try_recv().ok().map(|event| match event {
                    ServerChannelMsg::Acp(event) => {
                        ScheduledEvent::Application(Box::new(AppEvent::Acp(event)))
                    }
                }),
                2 => sup_rx.try_recv().ok().map(|event| {
                    ScheduledEvent::Application(Box::new(AppEvent::Supervisor(event)))
                }),
                3 => self.pending_terminal.take().map_or_else(
                    || match term_events.try_recv() {
                        Ok(event) => Some(ScheduledEvent::Terminal(Some(event))),
                        Err(TryRecvError::Empty) => None,
                        Err(TryRecvError::Disconnected) => Some(ScheduledEvent::Terminal(None)),
                    },
                    |event| Some(ScheduledEvent::Terminal(Some(event))),
                ),
                _ => unreachable!("non-tick source index must be in range"),
            };
            if let Some(event) = event {
                self.next_non_tick_source = (source + 1) % NON_TICK_SOURCE_COUNT;
                return event;
            }
        }

        let (source, event) = tokio::select! {
            Some(event) = conn_rx.recv() => (0, match event {
                ConnectionManagerEvent::State(state) => {
                    ScheduledEvent::Application(Box::new(AppEvent::Connection(state)))
                }
            }),
            Some(event) = srv_rx.recv() => (1, match event {
                ServerChannelMsg::Acp(event) => ScheduledEvent::Application(Box::new(AppEvent::Acp(event))),
            }),
            Some(event) = sup_rx.recv() => {
                (2, ScheduledEvent::Application(Box::new(AppEvent::Supervisor(event))))
            },
            event = term_events.recv() => (3, ScheduledEvent::Terminal(event)),
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
    term_events: &TerminalEventReceiver,
    executor: &mut EffectExecutor<'_>,
) -> anyhow::Result<()> {
    let mut tick_interval = application_tick_interval();
    let mut scheduler = EventScheduler::new();

    loop {
        terminal.draw(|frame| ui::draw(frame, app))?;

        match scheduler
            .next_event(&mut tick_interval, conn_rx, srv_rx, sup_rx, term_events)
            .await
        {
            ScheduledEvent::Application(event) => {
                if execute_event(terminal, app, executor, *event)?.should_quit {
                    return Ok(());
                }
            }
            ScheduledEvent::Terminal(first) => {
                let started = std::time::Instant::now();
                let mut batch_state = TerminalBatchState::default();
                let batch = process_terminal_batch(
                    term_events,
                    first,
                    |event, is_first| {
                        process_terminal_input(
                            terminal,
                            app,
                            executor,
                            &mut batch_state,
                            event,
                            is_first,
                        )
                    },
                    || started.elapsed() >= MAX_TERMINAL_BATCH_DURATION,
                )?;
                debug_assert!(batch.processed > 0);
                if let Some(event) = batch.deferred {
                    scheduler.defer_terminal(event);
                }
                if batch.should_quit {
                    return Ok(());
                }
            }
        }
    }
}

#[derive(Debug)]
enum BatchControl {
    Continue,
    Boundary,
    Quit,
    Defer(TerminalEventResult),
}

struct BatchResult {
    processed: usize,
    should_quit: bool,
    deferred: Option<TerminalEventResult>,
}

fn process_terminal_batch<F, B>(
    term_events: &TerminalEventReceiver,
    first: Option<TerminalEventResult>,
    mut process: F,
    mut duration_exhausted: B,
) -> anyhow::Result<BatchResult>
where
    F: FnMut(Option<TerminalEventResult>, bool) -> anyhow::Result<BatchControl>,
    B: FnMut() -> bool,
{
    let mut processed = 1;
    match process(first, true)? {
        BatchControl::Quit => {
            return Ok(BatchResult {
                processed,
                should_quit: true,
                deferred: None,
            });
        }
        BatchControl::Boundary => {
            return Ok(BatchResult {
                processed,
                should_quit: false,
                deferred: None,
            });
        }
        BatchControl::Defer(_) => unreachable!("the first terminal event cannot be deferred"),
        BatchControl::Continue => {}
    }

    while processed < MAX_TERMINAL_BATCH_EVENTS && !duration_exhausted() {
        let event = match term_events.try_recv() {
            Ok(event) => Some(event),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => None,
        };
        match process(event, false)? {
            BatchControl::Continue => processed += 1,
            BatchControl::Boundary => {
                processed += 1;
                break;
            }
            BatchControl::Quit => {
                processed += 1;
                return Ok(BatchResult {
                    processed,
                    should_quit: true,
                    deferred: None,
                });
            }
            BatchControl::Defer(event) => {
                return Ok(BatchResult {
                    processed,
                    should_quit: false,
                    deferred: Some(event),
                });
            }
        }
    }

    Ok(BatchResult {
        processed,
        should_quit: false,
        deferred: None,
    })
}

enum ClassifiedTerminalEvent {
    Application(Box<AppEvent>),
    Resize,
    Ignored,
    ReadError(io::Error),
}

fn classify_terminal_event(event: TerminalEventResult) -> ClassifiedTerminalEvent {
    match event {
        Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
            ClassifiedTerminalEvent::Application(Box::new(AppEvent::Key(key)))
        }
        Ok(Event::Mouse(mouse)) => {
            ClassifiedTerminalEvent::Application(Box::new(AppEvent::Mouse(mouse)))
        }
        Ok(Event::Resize(_, _)) => ClassifiedTerminalEvent::Resize,
        Ok(_) => ClassifiedTerminalEvent::Ignored,
        Err(error) => ClassifiedTerminalEvent::ReadError(error),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TerminalInputRoute {
    Elicitation,
    Popup(Popup),
    Screen(Screen),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TerminalBatchAffinity {
    Chord,
    Typing(TerminalInputRoute),
    SessionsNavigation(KeyCode),
    ChatViewport(KeyCode),
    DelegateViewport(KeyCode),
    ComposerNavigation(KeyCode),
    MouseScroll {
        screen: Screen,
        kind: MouseEventKind,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TerminalBatchDisposition {
    FreshFrame,
    Affinity(TerminalBatchAffinity),
}

#[derive(Default)]
struct TerminalBatchState {
    affinity: Option<TerminalBatchAffinity>,
}

fn process_terminal_input(
    terminal: &mut AppTerminal,
    app: &mut App,
    executor: &mut EffectExecutor<'_>,
    batch_state: &mut TerminalBatchState,
    event: Option<TerminalEventResult>,
    is_first: bool,
) -> anyhow::Result<BatchControl> {
    let disposition = terminal_event_batch_disposition(app, event.as_ref());
    if !is_first {
        // A batch may skip intermediate draws only while input keeps the same semantic route/action.
        let compatible = matches!(
            &disposition,
            Some(TerminalBatchDisposition::Affinity(affinity))
                if batch_state.affinity.as_ref() == Some(affinity)
        );
        if disposition.is_some() && !compatible {
            return Ok(BatchControl::Defer(event.expect(
                "only application terminal events have a batch disposition",
            )));
        }
    }

    let event = event.context("terminal event reader closed")?;
    match classify_terminal_event(event) {
        ClassifiedTerminalEvent::Application(event) => {
            let execution = execute_event(terminal, app, executor, *event)?;
            if execution.should_quit {
                Ok(BatchControl::Quit)
            } else if matches!(disposition, Some(TerminalBatchDisposition::FreshFrame))
                || execution.ends_terminal_batch
            {
                Ok(BatchControl::Boundary)
            } else {
                if is_first {
                    let Some(TerminalBatchDisposition::Affinity(affinity)) = disposition else {
                        unreachable!("batchable application input must have an affinity");
                    };
                    batch_state.affinity = Some(affinity);
                }
                Ok(BatchControl::Continue)
            }
        }
        ClassifiedTerminalEvent::Resize | ClassifiedTerminalEvent::Ignored => {
            Ok(BatchControl::Boundary)
        }
        ClassifiedTerminalEvent::ReadError(error) => {
            Err(error).context("failed to read terminal event")
        }
    }
}

fn terminal_event_batch_disposition(
    app: &App,
    event: Option<&TerminalEventResult>,
) -> Option<TerminalBatchDisposition> {
    match event {
        Some(Ok(Event::Key(key))) if key.kind == KeyEventKind::Press => {
            Some(terminal_key_batch_disposition(app, key))
        }
        Some(Ok(Event::Mouse(mouse))) => Some(terminal_mouse_batch_disposition(app, mouse)),
        _ => None,
    }
}

fn terminal_input_route(app: &App) -> TerminalInputRoute {
    if app.chat.elicitation.is_some() {
        TerminalInputRoute::Elicitation
    } else if app.navigation.popup != Popup::None {
        TerminalInputRoute::Popup(app.navigation.popup.clone())
    } else {
        TerminalInputRoute::Screen(app.navigation.screen.clone())
    }
}

fn terminal_key_batch_disposition(app: &App, key: &KeyEvent) -> TerminalBatchDisposition {
    if app.navigation.chord
        || key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('x')
    {
        return TerminalBatchDisposition::Affinity(TerminalBatchAffinity::Chord);
    }

    let route = terminal_input_route(app);
    if matches!(route, TerminalInputRoute::Popup(_))
        && matches!(key.code, KeyCode::PageUp | KeyCode::PageDown)
        || matches!(route, TerminalInputRoute::Popup(Popup::SessionSelect))
            && matches!(key.code, KeyCode::Tab | KeyCode::BackTab)
        || matches!(route, TerminalInputRoute::Elicitation)
            && matches!(
                key.code,
                KeyCode::Enter | KeyCode::Esc | KeyCode::Up | KeyCode::Down
            )
        || matches!(route, TerminalInputRoute::Screen(Screen::Chat))
            && matches!(key.code, KeyCode::Up | KeyCode::Down)
    {
        return TerminalBatchDisposition::FreshFrame;
    }

    let affinity = match (&route, key.code) {
        (TerminalInputRoute::Screen(Screen::Sessions), KeyCode::Up | KeyCode::Down) => {
            TerminalBatchAffinity::SessionsNavigation(key.code)
        }
        (TerminalInputRoute::Screen(Screen::Chat), KeyCode::PageUp | KeyCode::PageDown) => {
            TerminalBatchAffinity::ChatViewport(key.code)
        }
        (TerminalInputRoute::Screen(Screen::Chat), KeyCode::End)
            if app.chat.input_blocked_by_activity() || app.composer.input.is_empty() =>
        {
            TerminalBatchAffinity::ChatViewport(key.code)
        }
        (TerminalInputRoute::Screen(Screen::Chat), KeyCode::End) => {
            TerminalBatchAffinity::ComposerNavigation(key.code)
        }
        (
            TerminalInputRoute::Screen(Screen::Delegate),
            KeyCode::Up
            | KeyCode::Down
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Home
            | KeyCode::End,
        ) => TerminalBatchAffinity::DelegateViewport(key.code),
        (
            TerminalInputRoute::Screen(Screen::Chat),
            KeyCode::Left | KeyCode::Right | KeyCode::Home,
        ) => TerminalBatchAffinity::ComposerNavigation(key.code),
        _ if matches!(key.code, KeyCode::Char(_))
            && !key.modifiers.contains(KeyModifiers::CONTROL) =>
        {
            TerminalBatchAffinity::Typing(route)
        }
        _ => return TerminalBatchDisposition::FreshFrame,
    };
    TerminalBatchDisposition::Affinity(affinity)
}

fn terminal_mouse_batch_disposition(app: &App, mouse: &MouseEvent) -> TerminalBatchDisposition {
    if app.navigation.popup == Popup::None
        && matches!(app.navigation.screen, Screen::Chat | Screen::Delegate)
        && matches!(
            mouse.kind,
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        )
    {
        TerminalBatchDisposition::Affinity(TerminalBatchAffinity::MouseScroll {
            screen: app.navigation.screen.clone(),
            kind: mouse.kind,
        })
    } else {
        TerminalBatchDisposition::FreshFrame
    }
}

struct EventExecution {
    should_quit: bool,
    ends_terminal_batch: bool,
}

fn execute_event(
    terminal: &mut AppTerminal,
    app: &mut App,
    executor: &mut EffectExecutor<'_>,
    event: AppEvent,
) -> anyhow::Result<EventExecution> {
    let screen_before = app.navigation.screen.clone();
    let popup_before = app.navigation.popup.clone();
    let effects = application::update(app, event);
    let effect_boundary = executor.execute(terminal, app, effects)?;
    Ok(EventExecution {
        should_quit: app.should_quit,
        ends_terminal_batch: effect_boundary
            || app.navigation.screen != screen_before
            || app.navigation.popup != popup_before,
    })
}

fn application_tick_interval() -> Interval {
    let mut interval = tokio::time::interval_at(Instant::now() + TICK_INTERVAL, TICK_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    interval
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::VecDeque};

    use crossterm::event::{KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

    use super::*;
    use crate::{
        acp_state::AcpAppEvent,
        composer_state::ComposerState,
        connection_state::ConnState,
        domain::{
            chat::ChatEntry,
            session::{SessionGroup, SessionSummary},
        },
        runtime::terminal::{TerminalEventSender, terminal_event_queue},
    };

    fn test_terminal() -> AppTerminal {
        ratatui::Terminal::with_options(
            ratatui::backend::CrosstermBackend::new(std::io::stdout()),
            ratatui::TerminalOptions {
                viewport: ratatui::Viewport::Fixed(ratatui::layout::Rect::new(0, 0, 80, 24)),
            },
        )
        .unwrap()
    }

    fn draw_test_app(app: &mut App, width: u16, height: u16) {
        let backend = ratatui::backend::TestBackend::new(width, height);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| ui::draw(frame, app)).unwrap();
    }

    fn key(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn mouse(column: u16) -> Event {
        Event::Mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row: 0,
            modifiers: KeyModifiers::NONE,
        })
    }

    fn scroll(kind: MouseEventKind) -> Event {
        Event::Mouse(MouseEvent {
            kind,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        })
    }

    fn queue_with_events(
        events: impl IntoIterator<Item = TerminalEventResult>,
    ) -> (TerminalEventSender, TerminalEventReceiver) {
        let events = events.into_iter().collect::<Vec<_>>();
        let (sender, receiver) = terminal_event_queue(events.len().max(1));
        for event in events {
            sender.send(event);
        }
        (sender, receiver)
    }

    fn run_batch<F>(receiver: &TerminalEventReceiver, process: F) -> anyhow::Result<BatchResult>
    where
        F: FnMut(Option<TerminalEventResult>, bool) -> anyhow::Result<BatchControl>,
    {
        let first = Some(receiver.try_recv().unwrap());
        process_terminal_batch(receiver, first, process, || false)
    }

    fn scheduled_application(event: ScheduledEvent) -> Option<AppEvent> {
        match event {
            ScheduledEvent::Application(event) => Some(*event),
            ScheduledEvent::Terminal(_) => None,
        }
    }

    fn run_app_batch(app: &mut App, events: &[Event]) -> BatchResult {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let (_sender, receiver) = queue_with_events(events.iter().cloned().map(Ok));
        let mut batch_state = TerminalBatchState::default();
        run_batch(&receiver, |event, is_first| {
            process_terminal_input(
                &mut terminal,
                app,
                &mut executor,
                &mut batch_state,
                event,
                is_first,
            )
        })
        .unwrap()
    }

    fn run_batched_with_draws(
        app: &mut App,
        events: &[Event],
        width: u16,
        height: u16,
    ) -> Vec<usize> {
        let (_sender, receiver) = queue_with_events(events.iter().cloned().map(Ok));
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut deferred = None;
        let mut batch_sizes = Vec::new();
        let mut processed = 0;

        while processed < events.len() {
            let first = deferred.take().or_else(|| receiver.try_recv().ok());
            let mut batch_state = TerminalBatchState::default();
            let batch = process_terminal_batch(
                &receiver,
                first,
                |event, is_first| {
                    process_terminal_input(
                        &mut terminal,
                        app,
                        &mut executor,
                        &mut batch_state,
                        event,
                        is_first,
                    )
                },
                || false,
            )
            .unwrap();
            deferred = batch.deferred;
            processed += batch.processed;
            batch_sizes.push(batch.processed);
            draw_test_app(app, width, height);
        }
        batch_sizes
    }

    fn run_reference_with_draws(app: &mut App, events: &[Event], width: u16, height: u16) {
        for event in events {
            let effects = application::update(
                app,
                match event {
                    Event::Key(key) => AppEvent::Key(*key),
                    Event::Mouse(mouse) => AppEvent::Mouse(*mouse),
                    _ => panic!("reference input must be an application event"),
                },
            );
            assert!(effects.is_empty());
            draw_test_app(app, width, height);
        }
    }

    fn populate_chat(app: &mut App) {
        app.chat.messages = (0..12)
            .map(|index| ChatEntry::User {
                text: format!("message {index} with enough text to occupy a viewport row"),
                message_id: Some(format!("user-{index}")),
            })
            .collect();
    }

    #[test]
    fn execute_event_reports_quit_immediately_after_effect_execution() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = App::new();

        let execution = execute_event(
            &mut terminal,
            &mut app,
            &mut executor,
            AppEvent::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        )
        .unwrap();

        assert!(execution.should_quit);
        assert!(execution.ends_terminal_batch);
        assert!(app.should_quit);
    }

    #[test]
    fn terminal_event_classification_is_explicit() {
        let press =
            KeyEvent::new_with_kind(KeyCode::Char('p'), KeyModifiers::NONE, KeyEventKind::Press);
        let repeat =
            KeyEvent::new_with_kind(KeyCode::Char('r'), KeyModifiers::NONE, KeyEventKind::Repeat);
        let release = KeyEvent::new_with_kind(
            KeyCode::Char('u'),
            KeyModifiers::NONE,
            KeyEventKind::Release,
        );

        assert!(matches!(
            classify_terminal_event(Ok(Event::Key(press))),
            ClassifiedTerminalEvent::Application(event)
                if matches!(*event, AppEvent::Key(key) if key == press)
        ));
        assert!(matches!(
            classify_terminal_event(Ok(mouse(7))),
            ClassifiedTerminalEvent::Application(event)
                if matches!(*event, AppEvent::Mouse(event) if event.column == 7)
        ));
        assert!(matches!(
            classify_terminal_event(Ok(Event::Resize(80, 24))),
            ClassifiedTerminalEvent::Resize
        ));
        for ignored in [
            Event::Key(repeat),
            Event::Key(release),
            Event::FocusGained,
            Event::FocusLost,
            Event::Paste("ignored".into()),
        ] {
            assert!(matches!(
                classify_terminal_event(Ok(ignored)),
                ClassifiedTerminalEvent::Ignored
            ));
        }
        match classify_terminal_event(Err(io::Error::other("boom"))) {
            ClassifiedTerminalEvent::ReadError(error) => assert_eq!(error.to_string(), "boom"),
            _ => panic!("read error was not preserved"),
        }
    }

    #[test]
    fn terminal_batch_respects_count_limits() {
        for queued in [1, MAX_TERMINAL_BATCH_EVENTS - 1, MAX_TERMINAL_BATCH_EVENTS] {
            let (_sender, receiver) =
                queue_with_events((0..queued).map(|_| Ok(key(KeyCode::Char('x')))));
            let processed = Cell::new(0);
            let result = run_batch(&receiver, |_, _| {
                processed.set(processed.get() + 1);
                Ok(BatchControl::Continue)
            })
            .unwrap();
            assert_eq!(processed.get(), queued);
            assert_eq!(result.processed, queued);
            assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
        }

        let queued = MAX_TERMINAL_BATCH_EVENTS + 1;
        let (_sender, receiver) =
            queue_with_events((0..queued).map(|_| Ok(key(KeyCode::Char('x')))));
        let result = run_batch(&receiver, |_, _| Ok(BatchControl::Continue)).unwrap();
        assert_eq!(result.processed, MAX_TERMINAL_BATCH_EVENTS);
        assert!(receiver.try_recv().is_ok());
    }

    #[test]
    fn terminal_batch_elapsed_cutoff_is_checked_before_each_additional_event() {
        let (_sender, receiver) = queue_with_events((0..4).map(|_| Ok(key(KeyCode::Char('x')))));
        let checks = Cell::new(0);
        let processed = Cell::new(0);
        let first = Some(receiver.try_recv().unwrap());
        let result = process_terminal_batch(
            &receiver,
            first,
            |_, _| {
                processed.set(processed.get() + 1);
                Ok(BatchControl::Continue)
            },
            || {
                checks.set(checks.get() + 1);
                checks.get() == 2
            },
        )
        .unwrap();

        assert_eq!(result.processed, 2);
        assert_eq!(processed.get(), 2);
        assert_eq!(checks.get(), 2);
        assert_eq!(
            receiver.try_recv().unwrap().unwrap(),
            key(KeyCode::Char('x'))
        );
    }

    #[test]
    fn terminal_batch_stops_at_boundaries_and_quit() {
        let mut quit_at_count_limit = (0..MAX_TERMINAL_BATCH_EVENTS - 1)
            .map(|_| BatchControl::Continue)
            .collect::<Vec<_>>();
        quit_at_count_limit.push(BatchControl::Quit);
        for (controls, expected, should_quit) in [
            (vec![BatchControl::Boundary], 1, false),
            (
                vec![BatchControl::Continue, BatchControl::Boundary],
                2,
                false,
            ),
            (vec![BatchControl::Continue, BatchControl::Quit], 2, true),
            (vec![BatchControl::Quit], 1, true),
            (quit_at_count_limit, MAX_TERMINAL_BATCH_EVENTS, true),
        ] {
            let queued = controls.len().max(4);
            let (_sender, receiver) =
                queue_with_events((0..queued).map(|_| Ok(key(KeyCode::Char('x')))));
            let mut controls = VecDeque::from(controls);
            let result = run_batch(&receiver, |_, _| Ok(controls.pop_front().unwrap())).unwrap();
            assert_eq!(result.processed, expected);
            assert_eq!(result.should_quit, should_quit);
        }
    }

    #[test]
    fn a_thousand_mixed_terminal_events_are_delivered_once_in_fifo_order() {
        let expected = (0..1_000)
            .map(|index| match index % 4 {
                0 => key(KeyCode::Down),
                1 => key(KeyCode::Up),
                2 => key(KeyCode::Char(
                    char::from_u32('0' as u32 + index % 10).unwrap(),
                )),
                _ => key(KeyCode::Char(
                    char::from_u32('0' as u32 + (index + 1) % 10).unwrap(),
                )),
            })
            .collect::<Vec<_>>();
        let (_sender, receiver) = queue_with_events(expected.iter().cloned().map(Ok));
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = App::new();
        app.sessions.session_groups = vec![SessionGroup {
            cwd: Some("/workspace".into()),
            sessions: vec![SessionSummary {
                session_id: "session-1".into(),
                ..SessionSummary::default()
            }],
            ..SessionGroup::default()
        }];
        let mut actual = Vec::new();
        let mut deferred = None;
        let mut batches = 0;

        while actual.len() < expected.len() {
            let first = deferred.take().or_else(|| receiver.try_recv().ok());
            let mut batch_state = TerminalBatchState::default();
            let result = process_terminal_batch(
                &receiver,
                first,
                |event, is_first| {
                    if let Some(raw) = event
                        .as_ref()
                        .and_then(|event| event.as_ref().ok().cloned())
                    {
                        let control = process_terminal_input(
                            &mut terminal,
                            &mut app,
                            &mut executor,
                            &mut batch_state,
                            event,
                            is_first,
                        )?;
                        if !matches!(control, BatchControl::Defer(_)) {
                            actual.push(raw);
                        }
                        Ok(control)
                    } else {
                        process_terminal_input(
                            &mut terminal,
                            &mut app,
                            &mut executor,
                            &mut batch_state,
                            event,
                            is_first,
                        )
                    }
                },
                || false,
            )
            .unwrap();
            assert!(result.processed <= MAX_TERMINAL_BATCH_EVENTS);
            deferred = result.deferred;
            batches += 1;
        }

        assert_eq!(actual, expected);
        assert_eq!(batches, 750);
        assert!(deferred.is_none());
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn incompatible_followup_is_deferred_until_after_redraw() {
        let (_sender, receiver) = queue_with_events([
            Ok(key(KeyCode::Down)),
            Ok(key(KeyCode::Up)),
            Ok(key(KeyCode::Down)),
        ]);
        let mut app = App::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut batch_state = TerminalBatchState::default();
        let batch = run_batch(&receiver, |event, is_first| {
            process_terminal_input(
                &mut terminal,
                &mut app,
                &mut executor,
                &mut batch_state,
                event,
                is_first,
            )
        })
        .unwrap();

        assert_eq!(batch.processed, 1);
        assert_eq!(batch.deferred.unwrap().unwrap(), key(KeyCode::Up));
        assert_eq!(receiver.try_recv().unwrap().unwrap(), key(KeyCode::Down));
    }

    #[test]
    fn resize_and_ignored_events_end_batches_without_consuming_following_input() {
        for boundary in [
            Event::Resize(120, 40),
            Event::FocusGained,
            Event::Paste("ignored".into()),
        ] {
            let (_sender, receiver) = queue_with_events([
                Ok(key(KeyCode::Char('a'))),
                Ok(boundary),
                Ok(key(KeyCode::Char('b'))),
            ]);
            let mut actual = Vec::new();
            let result = run_batch(&receiver, |event, _| {
                let classified = classify_terminal_event(event.unwrap());
                match classified {
                    ClassifiedTerminalEvent::Application(event) => match *event {
                        AppEvent::Key(key) => {
                            actual.push(key.code);
                            Ok(BatchControl::Continue)
                        }
                        _ => panic!("unexpected application event"),
                    },
                    ClassifiedTerminalEvent::Resize | ClassifiedTerminalEvent::Ignored => {
                        Ok(BatchControl::Boundary)
                    }
                    _ => panic!("unexpected terminal classification"),
                }
            })
            .unwrap();

            assert_eq!(result.processed, 2);
            assert_eq!(actual, [KeyCode::Char('a')]);
            assert_eq!(
                receiver.try_recv().unwrap().unwrap(),
                key(KeyCode::Char('b'))
            );
        }
    }

    #[test]
    fn terminal_read_error_and_reader_closure_are_run_loop_errors() {
        let mut terminal = test_terminal();
        let mut app = App::new();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);

        let mut batch_state = TerminalBatchState::default();
        let read_error = process_terminal_input(
            &mut terminal,
            &mut app,
            &mut executor,
            &mut batch_state,
            Some(Err(io::Error::other("read failed"))),
            true,
        )
        .unwrap_err();
        assert!(
            read_error
                .to_string()
                .contains("failed to read terminal event")
        );

        let closed = process_terminal_input(
            &mut terminal,
            &mut app,
            &mut executor,
            &mut batch_state,
            None,
            true,
        )
        .unwrap_err();
        assert!(closed.to_string().contains("terminal event reader closed"));
    }

    #[test]
    fn terminal_affinity_uses_pre_event_route_and_exact_navigation_action() {
        let mut app = App::new();
        assert_eq!(
            terminal_key_batch_disposition(&app, &KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
            TerminalBatchDisposition::Affinity(TerminalBatchAffinity::SessionsNavigation(
                KeyCode::Down
            ))
        );

        app.navigation.screen = Screen::Chat;
        assert_eq!(
            terminal_key_batch_disposition(
                &app,
                &KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE)
            ),
            TerminalBatchDisposition::Affinity(TerminalBatchAffinity::ChatViewport(
                KeyCode::PageUp
            ))
        );
        assert_eq!(
            terminal_mouse_batch_disposition(
                &app,
                &MouseEvent {
                    kind: MouseEventKind::ScrollUp,
                    column: 0,
                    row: 0,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            TerminalBatchDisposition::Affinity(TerminalBatchAffinity::MouseScroll {
                screen: Screen::Chat,
                kind: MouseEventKind::ScrollUp,
            })
        );

        app.navigation.popup = Popup::SessionSelect;
        assert_eq!(
            terminal_key_batch_disposition(
                &app,
                &KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE)
            ),
            TerminalBatchDisposition::FreshFrame
        );
    }

    #[test]
    fn sessions_opposite_navigation_matches_draw_after_each_event() {
        fn app_at_visible_bottom() -> App {
            let mut app = App::new();
            app.sessions.session_groups = vec![SessionGroup {
                cwd: Some("/workspace".into()),
                sessions: (0..20)
                    .map(|index| SessionSummary {
                        session_id: format!("session-{index}"),
                        ..SessionSummary::default()
                    })
                    .collect(),
                ..SessionGroup::default()
            }];
            app.sessions.session_cursor = 1;
            app
        }

        let events = [key(KeyCode::Down), key(KeyCode::Up)];
        let mut reference = app_at_visible_bottom();
        let mut batched = app_at_visible_bottom();
        draw_test_app(&mut reference, 80, 8);
        draw_test_app(&mut batched, 80, 8);
        assert_eq!(reference.render.start_page_scroll(), 0);

        run_reference_with_draws(&mut reference, &events, 80, 8);
        let batch_sizes = run_batched_with_draws(&mut batched, &events, 80, 8);

        assert_eq!(batch_sizes, [1, 1]);
        assert_eq!(
            batched.sessions.session_cursor,
            reference.sessions.session_cursor
        );
        assert_eq!(
            batched.render.start_page_scroll(),
            reference.render.start_page_scroll()
        );
        assert_eq!(reference.render.start_page_scroll(), 1);
    }

    #[test]
    fn chat_opposite_viewport_navigation_matches_draw_after_each_event() {
        let mut reference = App::new();
        reference.navigation.screen = Screen::Chat;
        populate_chat(&mut reference);
        let mut batched = App::new();
        batched.navigation.screen = Screen::Chat;
        populate_chat(&mut batched);
        for app in [&mut reference, &mut batched] {
            app.render.set_chat_scroll_offset(u16::MAX);
            draw_test_app(app, 40, 10);
        }
        let maximum = reference.render.chat_scroll_offset();
        assert!(maximum > 10);
        assert_eq!(batched.render.chat_scroll_offset(), maximum);

        let events = [key(KeyCode::PageUp), key(KeyCode::PageDown)];
        run_reference_with_draws(&mut reference, &events, 40, 10);
        let batch_sizes = run_batched_with_draws(&mut batched, &events, 40, 10);

        assert_eq!(batch_sizes, [1, 1]);
        assert_eq!(
            batched.render.chat_scroll_offset(),
            reference.render.chat_scroll_offset()
        );
        assert_eq!(reference.render.chat_scroll_offset(), maximum - 10);
    }

    #[test]
    fn delegate_opposite_viewport_navigation_matches_draw_after_each_event() {
        let mut reference = App::new();
        reference.navigation.screen = Screen::Delegate;
        populate_chat(&mut reference);
        let mut batched = App::new();
        batched.navigation.screen = Screen::Delegate;
        populate_chat(&mut batched);
        for app in [&mut reference, &mut batched] {
            app.render.set_chat_scroll_offset(u16::MAX);
            draw_test_app(app, 40, 10);
        }
        let maximum = reference.render.chat_scroll_offset();
        assert!(maximum > 1);
        assert_eq!(batched.render.chat_scroll_offset(), maximum);

        let events = [key(KeyCode::Up), key(KeyCode::Down)];
        run_reference_with_draws(&mut reference, &events, 40, 10);
        let batch_sizes = run_batched_with_draws(&mut batched, &events, 40, 10);

        assert_eq!(batch_sizes, [1, 1]);
        assert_eq!(
            batched.render.chat_scroll_offset(),
            reference.render.chat_scroll_offset()
        );
        assert_eq!(reference.render.chat_scroll_offset(), maximum - 1);
    }

    #[test]
    fn chat_and_delegate_opposite_mouse_scroll_match_draw_after_each_event() {
        for screen in [Screen::Chat, Screen::Delegate] {
            let mut reference = App::new();
            reference.navigation.screen = screen.clone();
            populate_chat(&mut reference);
            let mut batched = App::new();
            batched.navigation.screen = screen.clone();
            populate_chat(&mut batched);
            for app in [&mut reference, &mut batched] {
                app.render.set_chat_scroll_offset(u16::MAX);
                draw_test_app(app, 40, 10);
            }
            let maximum = reference.render.chat_scroll_offset();
            assert!(maximum > 3);
            assert_eq!(batched.render.chat_scroll_offset(), maximum);

            let events = [
                scroll(MouseEventKind::ScrollUp),
                scroll(MouseEventKind::ScrollDown),
            ];
            run_reference_with_draws(&mut reference, &events, 40, 10);
            let batch_sizes = run_batched_with_draws(&mut batched, &events, 40, 10);

            assert_eq!(batch_sizes, [1, 1]);
            assert_eq!(
                batched.render.chat_scroll_offset(),
                reference.render.chat_scroll_offset(),
                "screen {screen:?}"
            );
            assert_eq!(reference.render.chat_scroll_offset(), maximum - 3);
        }
    }

    #[test]
    fn homogeneous_navigation_typing_and_mouse_bursts_coalesce() {
        let repeated = |code| vec![key(code), key(code), key(code)];

        let mut sessions = App::new();
        sessions.sessions.session_groups = vec![SessionGroup {
            cwd: Some("/workspace".into()),
            sessions: (0..5)
                .map(|index| SessionSummary {
                    session_id: format!("session-{index}"),
                    ..SessionSummary::default()
                })
                .collect(),
            ..SessionGroup::default()
        }];
        let result = run_app_batch(&mut sessions, &repeated(KeyCode::Down));
        assert_eq!(result.processed, 3);
        assert!(result.deferred.is_none());

        for (screen, code) in [
            (Screen::Chat, KeyCode::PageUp),
            (Screen::Delegate, KeyCode::Up),
        ] {
            let mut app = App::new();
            app.navigation.screen = screen;
            let result = run_app_batch(&mut app, &repeated(code));
            assert_eq!(result.processed, 3);
            assert!(result.deferred.is_none());
        }

        for screen in [Screen::Chat, Screen::Delegate] {
            let mut app = App::new();
            app.navigation.screen = screen;
            let events = vec![scroll(MouseEventKind::ScrollUp); 3];
            let result = run_app_batch(&mut app, &events);
            assert_eq!(result.processed, 3);
            assert!(result.deferred.is_none());
        }

        for (code, initial_cursor, expected_cursor) in
            [(KeyCode::Right, 0, 3), (KeyCode::Left, 4, 1)]
        {
            let mut composer = App::new();
            composer.navigation.screen = Screen::Chat;
            composer.composer.replace_input("abcd".into());
            composer.composer.input_cursor = initial_cursor;
            let result = run_app_batch(&mut composer, &repeated(code));
            assert_eq!(result.processed, 3);
            assert!(result.deferred.is_none());
            assert_eq!(composer.composer.input_cursor, expected_cursor);
        }

        let mut typing = App::new();
        typing.navigation.screen = Screen::Chat;
        let events = [
            key(KeyCode::Char('a')),
            key(KeyCode::Char('b')),
            key(KeyCode::Char('c')),
        ];
        let result = run_app_batch(&mut typing, &events);
        assert_eq!(result.processed, 3);
        assert!(result.deferred.is_none());
        assert_eq!(typing.composer.input, "abc");
    }

    #[test]
    fn popup_open_draw_then_page_down_uses_published_geometry() {
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;
        app.sessions.session_discovery_in_progress = true;
        app.sessions.session_groups = vec![SessionGroup {
            cwd: Some("/workspace".into()),
            sessions: (0..20)
                .map(|index| SessionSummary {
                    session_id: format!("session-{index}"),
                    ..SessionSummary::default()
                })
                .collect(),
            ..SessionGroup::default()
        }];
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let (_sender, receiver) = queue_with_events([
            Ok(Event::Key(KeyEvent::new(
                KeyCode::Char('x'),
                KeyModifiers::CONTROL,
            ))),
            Ok(key(KeyCode::Char('l'))),
            Ok(key(KeyCode::PageDown)),
        ]);
        let first = Some(receiver.try_recv().unwrap());

        let mut batch_state = TerminalBatchState::default();
        let batch = process_terminal_batch(
            &receiver,
            first,
            |event, is_first| {
                process_terminal_input(
                    &mut terminal,
                    &mut app,
                    &mut executor,
                    &mut batch_state,
                    event,
                    is_first,
                )
            },
            || false,
        )
        .unwrap();
        assert_eq!(batch.processed, 2);
        assert_eq!(app.navigation.popup, Popup::SessionSelect);
        assert_eq!(app.sessions.session_cursor, 0);

        draw_test_app(&mut app, 80, 24);
        let expected_cursor = app
            .render
            .session_popup_page_step()
            .min(app.sessions.visible_popup_items().len() - 1);
        assert!(expected_cursor > 1);
        assert!(matches!(
            process_terminal_input(
                &mut terminal,
                &mut app,
                &mut executor,
                &mut TerminalBatchState::default(),
                Some(receiver.try_recv().unwrap()),
                true,
            )
            .unwrap(),
            BatchControl::Boundary
        ));
        assert_eq!(app.sessions.session_cursor, expected_cursor);
    }

    #[test]
    fn resize_draw_then_composer_up_uses_resized_geometry() {
        let mut app = App::new();
        app.navigation.screen = Screen::Chat;
        app.composer
            .replace_input("abcdefghijklmnopqrstuvwxyz0123456789".repeat(3));
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let (_sender, receiver) =
            queue_with_events([Ok(Event::Resize(24, 12)), Ok(key(KeyCode::Up))]);

        let mut batch_state = TerminalBatchState::default();
        let batch = run_batch(&receiver, |event, is_first| {
            process_terminal_input(
                &mut terminal,
                &mut app,
                &mut executor,
                &mut batch_state,
                event,
                is_first,
            )
        })
        .unwrap();
        assert_eq!(batch.processed, 1);

        draw_test_app(&mut app, 24, 12);
        let line_width = app.render.composer_input_line_width();
        let mut expected = ComposerState::new();
        expected.input = app.composer.input.clone();
        expected.input_cursor = app.composer.input_cursor;
        expected.input_up_visual(line_width, 2);
        assert!(matches!(
            process_terminal_input(
                &mut terminal,
                &mut app,
                &mut executor,
                &mut TerminalBatchState::default(),
                Some(receiver.try_recv().unwrap()),
                true,
            )
            .unwrap(),
            BatchControl::Boundary
        ));
        assert_eq!(app.composer.input_cursor, expected.input_cursor);
        assert_eq!(
            app.composer.input_preferred_col,
            expected.input_preferred_col
        );
    }

    #[test]
    fn popup_open_and_close_transitions_end_terminal_batches() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut executor = EffectExecutor::new(&tx);
        let mut terminal = test_terminal();
        let mut app = App::new();
        app.connection.conn = ConnState::Connected;

        let open = execute_event(
            &mut terminal,
            &mut app,
            &mut executor,
            AppEvent::Key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL)),
        )
        .unwrap();
        assert!(open.ends_terminal_batch);
        assert_eq!(app.navigation.popup, Popup::CommandPalette);

        let close = execute_event(
            &mut terminal,
            &mut app,
            &mut executor,
            AppEvent::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .unwrap();
        assert!(close.ends_terminal_batch);
        assert_eq!(app.navigation.popup, Popup::None);
    }

    fn connection_event(attempt: u32) -> ConnectionManagerEvent {
        ConnectionManagerEvent::State(crate::app::ConnectionEvent::Connecting {
            attempt,
            delay_ms: 10,
        })
    }

    fn pending_terminal_events() -> (TerminalEventSender, TerminalEventReceiver) {
        terminal_event_queue(1)
    }

    #[tokio::test]
    async fn acp_progresses_while_connection_source_stays_ready() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (_sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let (_term_tx, term_events) = pending_terminal_events();
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
                    &term_events,
                )
                .await;
            saw_acp |= matches!(
                scheduled_application(event),
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
        let (_term_tx, term_events) = pending_terminal_events();
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
                    &term_events,
                )
                .await;
            saw_supervisor |= matches!(
                scheduled_application(event),
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
        let (term_tx, term_events) = terminal_event_queue(32);
        for _ in 0..32 {
            term_tx.send(Ok(key(KeyCode::Char('x'))));
        }
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
        tokio::time::timeout(Duration::from_secs(1), async {
            for _ in 0..(NON_TICK_SOURCE_COUNT * 2 + 2) {
                match scheduler
                    .next_event(
                        &mut tick_interval,
                        &mut conn_rx,
                        &mut srv_rx,
                        &mut sup_rx,
                        &term_events,
                    )
                    .await
                {
                    ScheduledEvent::Application(event) => match *event {
                        AppEvent::Connection(_) => saw_connection = true,
                        AppEvent::Acp(_) => saw_acp = true,
                        AppEvent::Supervisor(_) => saw_supervisor = true,
                        AppEvent::Tick => saw_tick = true,
                        _ => {}
                    },
                    ScheduledEvent::Terminal(Some(Ok(Event::Key(_)))) => terminal_events += 1,
                    _ => {}
                }
            }
        })
        .await
        .expect("scheduler stopped making progress");

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
        let (_term_tx, term_events) = pending_terminal_events();
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

        let event = scheduler
            .next_event(
                &mut tick_interval,
                &mut conn_rx,
                &mut srv_rx,
                &mut sup_rx,
                &term_events,
            )
            .await;
        assert!(matches!(
            event,
            ScheduledEvent::Application(event) if matches!(*event, AppEvent::Tick)
        ));
    }

    #[tokio::test]
    async fn overdue_tick_alternates_with_ready_non_tick_work() {
        let (conn_tx, mut conn_rx) = mpsc::unbounded_channel();
        let (_srv_tx, mut srv_rx) = mpsc::unbounded_channel();
        let (_sup_tx, mut sup_rx) = mpsc::unbounded_channel();
        let (_term_tx, term_events) = pending_terminal_events();
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
                    &term_events,
                )
                .await;
            assert_eq!(
                matches!(
                    event,
                    ScheduledEvent::Application(event) if matches!(*event, AppEvent::Tick)
                ),
                expect_tick
            );
        }

        assert_eq!(conn_rx.len(), 2);
    }
}
