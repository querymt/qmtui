use std::{
    collections::VecDeque,
    io,
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{self as std_mpsc, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::Notify;

use crate::application::ExternalEditorOutcome;

use super::editor::open_external_editor;

pub(super) type AppTerminal = Terminal<CrosstermBackend<std::io::Stdout>>;
pub(super) type TerminalEventResult = io::Result<Event>;

const TERMINAL_EVENT_QUEUE_CAPACITY: usize = 128;
const TERMINAL_POLL_INTERVAL: Duration = Duration::from_millis(5);

struct QueueState {
    events: VecDeque<TerminalEventResult>,
    sender_open: bool,
}

struct QueueShared {
    state: Mutex<QueueState>,
    not_empty: Notify,
    not_full: Condvar,
    capacity: usize,
}

pub(super) struct TerminalEventSender {
    shared: Arc<QueueShared>,
}

pub(super) struct TerminalEventReceiver {
    shared: Arc<QueueShared>,
}

pub(super) fn terminal_event_queue(
    capacity: usize,
) -> (TerminalEventSender, TerminalEventReceiver) {
    assert!(
        capacity > 0,
        "terminal event queue capacity must be positive"
    );
    let shared = Arc::new(QueueShared {
        state: Mutex::new(QueueState {
            events: VecDeque::with_capacity(capacity),
            sender_open: true,
        }),
        not_empty: Notify::new(),
        not_full: Condvar::new(),
        capacity,
    });
    (
        TerminalEventSender {
            shared: Arc::clone(&shared),
        },
        TerminalEventReceiver { shared },
    )
}

impl TerminalEventSender {
    fn wait_for_capacity(&self, timeout: Duration) -> bool {
        let state = self.shared.state.lock().expect("terminal queue poisoned");
        if state.events.len() < self.shared.capacity {
            return true;
        }
        let (state, _) = self
            .shared
            .not_full
            .wait_timeout_while(state, timeout, |state| {
                state.events.len() >= self.shared.capacity
            })
            .expect("terminal queue poisoned while waiting for capacity");
        state.events.len() < self.shared.capacity
    }

    pub(super) fn send(&self, event: TerminalEventResult) {
        let mut state = self.shared.state.lock().expect("terminal queue poisoned");
        while state.events.len() >= self.shared.capacity {
            state = self
                .shared
                .not_full
                .wait(state)
                .expect("terminal queue poisoned while sending");
        }
        state.events.push_back(event);
        drop(state);
        self.shared.not_empty.notify_one();
    }
}

impl Drop for TerminalEventSender {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().expect("terminal queue poisoned");
        state.sender_open = false;
        drop(state);
        self.shared.not_empty.notify_one();
    }
}

impl TerminalEventReceiver {
    pub(super) fn try_recv(&self) -> Result<TerminalEventResult, TryRecvError> {
        let mut state = self.shared.state.lock().expect("terminal queue poisoned");
        if let Some(event) = state.events.pop_front() {
            drop(state);
            self.shared.not_full.notify_one();
            return Ok(event);
        }
        if state.sender_open {
            Err(TryRecvError::Empty)
        } else {
            Err(TryRecvError::Disconnected)
        }
    }

    pub(super) async fn recv(&self) -> Option<TerminalEventResult> {
        loop {
            // Register before checking the queue so insertion cannot be lost between check and wait.
            let notified = self.shared.not_empty.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.try_recv() {
                Ok(event) => return Some(event),
                Err(TryRecvError::Disconnected) => return None,
                Err(TryRecvError::Empty) => notified.await,
            }
        }
    }
}

enum ReaderCommand {
    Pause(std_mpsc::SyncSender<()>),
    Resume(std_mpsc::SyncSender<()>),
    Shutdown(std_mpsc::SyncSender<()>),
}

#[derive(Clone)]
pub(super) struct TerminalReaderController {
    command_tx: std_mpsc::Sender<ReaderCommand>,
    queue: Arc<QueueShared>,
}

impl TerminalReaderController {
    pub(super) fn pause(&self) -> anyhow::Result<()> {
        self.send_command(ReaderCommand::Pause)
    }

    pub(super) fn resume(&self) -> anyhow::Result<()> {
        self.send_command(ReaderCommand::Resume)
    }

    fn shutdown(&self) -> anyhow::Result<()> {
        self.send_command(ReaderCommand::Shutdown)
    }

    fn send_command(
        &self,
        command: impl FnOnce(std_mpsc::SyncSender<()>) -> ReaderCommand,
    ) -> anyhow::Result<()> {
        let (ack_tx, ack_rx) = std_mpsc::sync_channel(0);
        self.command_tx
            .send(command(ack_tx))
            .map_err(|_| anyhow::anyhow!("terminal reader control channel closed"))?;
        self.queue.not_full.notify_all();
        ack_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("terminal reader stopped before acknowledging control"))
    }
}

pub(super) fn with_reader_paused<T>(
    reader: Option<&TerminalReaderController>,
    operation: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    if let Some(reader) = reader {
        reader.pause()?;
    }
    let operation_result = operation();
    let resume_result = reader.map(TerminalReaderController::resume).transpose();
    let value = operation_result?;
    resume_result?;
    Ok(value)
}

pub(super) struct TerminalReader {
    events: TerminalEventReceiver,
    controller: TerminalReaderController,
    thread: Option<JoinHandle<()>>,
}

impl TerminalReader {
    pub(super) fn start() -> anyhow::Result<Self> {
        Self::start_with_source_inner(|timeout| {
            if event::poll(timeout)? {
                event::read().map(Some)
            } else {
                Ok(None)
            }
        })
    }

    #[cfg(test)]
    pub(super) fn start_with_source<F>(source: F) -> anyhow::Result<Self>
    where
        F: FnMut(Duration) -> io::Result<Option<Event>> + Send + 'static,
    {
        Self::start_with_source_inner(source)
    }

    fn start_with_source_inner<F>(mut source: F) -> anyhow::Result<Self>
    where
        F: FnMut(Duration) -> io::Result<Option<Event>> + Send + 'static,
    {
        let (event_tx, events) = terminal_event_queue(TERMINAL_EVENT_QUEUE_CAPACITY);
        let (command_tx, command_rx) = std_mpsc::channel();
        let controller = TerminalReaderController {
            command_tx,
            queue: Arc::clone(&events.shared),
        };
        let thread = thread::Builder::new()
            .name("qmtui-terminal-reader".into())
            .spawn(move || reader_thread(&mut source, event_tx, command_rx))?;
        Ok(Self {
            events,
            controller,
            thread: Some(thread),
        })
    }

    pub(super) fn events(&self) -> &TerminalEventReceiver {
        &self.events
    }

    pub(super) fn controller(&self) -> TerminalReaderController {
        self.controller.clone()
    }

    pub(super) fn shutdown(&mut self) -> anyhow::Result<()> {
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        let control_result = self.controller.shutdown();
        let join_result = thread
            .join()
            .map_err(|_| anyhow::anyhow!("terminal reader thread panicked"));
        match (control_result, join_result) {
            (_, Err(error)) => Err(error),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }
}

impl Drop for TerminalReader {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn reader_thread<F>(
    source: &mut F,
    event_tx: TerminalEventSender,
    command_rx: std_mpsc::Receiver<ReaderCommand>,
) where
    F: FnMut(Duration) -> io::Result<Option<Event>>,
{
    let mut paused = false;
    loop {
        let command = if paused {
            match command_rx.recv() {
                Ok(command) => Some(command),
                Err(_) => return,
            }
        } else {
            command_rx.try_recv().ok()
        };
        if let Some(command) = command {
            match command {
                ReaderCommand::Pause(ack) => {
                    paused = true;
                    let _ = ack.send(());
                }
                ReaderCommand::Resume(ack) => {
                    paused = false;
                    let _ = ack.send(());
                }
                ReaderCommand::Shutdown(ack) => {
                    let _ = ack.send(());
                    return;
                }
            }
            continue;
        }
        if paused || !event_tx.wait_for_capacity(TERMINAL_POLL_INTERVAL) {
            continue;
        }

        match source(TERMINAL_POLL_INTERVAL) {
            Ok(Some(event)) => event_tx.send(Ok(event)),
            Ok(None) => {}
            Err(error) => {
                event_tx.send(Err(error));
                return;
            }
        }
    }
}

pub(super) fn enter() -> anyhow::Result<AppTerminal> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

pub(super) fn leave(terminal: &mut AppTerminal) -> anyhow::Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    Ok(())
}

pub(super) fn open_external_editor_with_terminal(
    terminal: &mut AppTerminal,
    initial_text: &str,
) -> anyhow::Result<ExternalEditorOutcome> {
    terminal.show_cursor()?;
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    let outcome = open_external_editor(initial_text);

    enable_raw_mode()?;
    execute!(terminal.backend_mut(), EnterAlternateScreen)?;
    terminal.hide_cursor()?;
    Ok(outcome)
}

pub(super) fn redraw(terminal: &mut AppTerminal) -> anyhow::Result<()> {
    terminal.clear()?;
    terminal.autoresize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    use super::*;

    fn key(character: char) -> Event {
        Event::Key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE))
    }

    #[test]
    fn queue_try_recv_is_nonblocking_and_preserves_fifo() {
        let (sender, receiver) = terminal_event_queue(2);
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        sender.send(Ok(key('a')));
        sender.send(Ok(key('b')));
        assert_eq!(receiver.try_recv().unwrap().unwrap(), key('a'));
        assert_eq!(receiver.try_recv().unwrap().unwrap(), key('b'));
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        drop(sender);
        assert!(matches!(
            receiver.try_recv(),
            Err(TryRecvError::Disconnected)
        ));
    }

    #[tokio::test]
    async fn queue_receive_has_no_lost_wakeup() {
        let (sender, receiver) = terminal_event_queue(1);
        let receive = tokio::spawn(async move { receiver.recv().await });
        tokio::task::yield_now().await;

        sender.send(Ok(key('x')));
        let event = tokio::time::timeout(Duration::from_secs(1), receive)
            .await
            .expect("receiver was not notified")
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(event, key('x'));
    }

    #[tokio::test]
    async fn bounded_queue_applies_backpressure_without_dropping_events() {
        let (sender, receiver) = terminal_event_queue(2);
        let producer = thread::spawn(move || {
            for index in 0..100 {
                sender.send(Ok(key(char::from_u32('a' as u32 + index % 26).unwrap())));
            }
        });

        for index in 0..100 {
            let expected = key(char::from_u32('a' as u32 + index % 26).unwrap());
            assert_eq!(receiver.recv().await.unwrap().unwrap(), expected);
        }
        producer.join().unwrap();
        assert!(receiver.recv().await.is_none());
    }

    fn wait_for_poll_after(polls: &AtomicUsize, previous: usize) {
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while polls.load(Ordering::SeqCst) == previous {
            assert!(
                std::time::Instant::now() < deadline,
                "reader did not resume"
            );
            thread::yield_now();
        }
    }

    fn wait_until_queue_is_full(reader: &TerminalReader) {
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            let queued = reader
                .events
                .shared
                .state
                .lock()
                .expect("terminal queue poisoned")
                .events
                .len();
            if queued == TERMINAL_EVENT_QUEUE_CAPACITY {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "terminal queue did not fill"
            );
            thread::yield_now();
        }
    }

    #[test]
    fn reader_pause_resume_and_shutdown_are_acknowledged() {
        let polls = Arc::new(AtomicUsize::new(0));
        let source_polls = Arc::clone(&polls);
        let mut reader = TerminalReader::start_with_source(move |_| {
            source_polls.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(1));
            Ok(None)
        })
        .unwrap();
        let controller = reader.controller();

        wait_for_poll_after(&polls, 0);
        controller.pause().unwrap();
        let paused_at = polls.load(Ordering::SeqCst);
        thread::sleep(Duration::from_millis(15));
        assert_eq!(polls.load(Ordering::SeqCst), paused_at);

        controller.resume().unwrap();
        wait_for_poll_after(&polls, paused_at);
        reader.shutdown().unwrap();
    }

    #[test]
    fn pause_acknowledgement_follows_already_read_event_enqueue() {
        let (entered_tx, entered_rx) = std_mpsc::sync_channel(0);
        let (release_tx, release_rx) = std_mpsc::sync_channel(0);
        let mut first = true;
        let mut reader = TerminalReader::start_with_source(move |_| {
            if first {
                first = false;
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(Some(key('q')))
            } else {
                thread::sleep(Duration::from_millis(1));
                Ok(None)
            }
        })
        .unwrap();
        entered_rx.recv().unwrap();

        let controller = reader.controller();
        let pause = thread::spawn(move || controller.pause());
        release_tx.send(()).unwrap();
        pause.join().unwrap().unwrap();

        assert_eq!(reader.events().try_recv().unwrap().unwrap(), key('q'));
        reader.shutdown().unwrap();
    }

    #[test]
    fn pause_is_acknowledged_while_terminal_queue_is_full() {
        let mut next = 0usize;
        let (full_tx, full_rx) = std_mpsc::sync_channel(0);
        let mut reader = TerminalReader::start_with_source(move |_| {
            let character = char::from_u32('a' as u32 + (next % 26) as u32).unwrap();
            next += 1;
            if next == TERMINAL_EVENT_QUEUE_CAPACITY {
                full_tx.send(()).unwrap();
            }
            Ok(Some(key(character)))
        })
        .unwrap();
        full_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        wait_until_queue_is_full(&reader);

        let controller = reader.controller();
        let (done_tx, done_rx) = std_mpsc::sync_channel(0);
        let pause = thread::spawn(move || {
            let result = controller.pause();
            done_tx.send(()).unwrap();
            result
        });
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("pause was not acknowledged with a full queue");
        pause.join().unwrap().unwrap();

        for index in 0..TERMINAL_EVENT_QUEUE_CAPACITY {
            let expected = key(char::from_u32('a' as u32 + (index % 26) as u32).unwrap());
            assert_eq!(reader.events().try_recv().unwrap().unwrap(), expected);
        }
        assert!(matches!(
            reader.events().try_recv(),
            Err(TryRecvError::Empty)
        ));
        reader.shutdown().unwrap();
    }

    #[test]
    fn shutdown_is_acknowledged_and_joins_while_terminal_queue_is_full() {
        let mut next = 0usize;
        let (full_tx, full_rx) = std_mpsc::sync_channel(0);
        let reader = TerminalReader::start_with_source(move |_| {
            let character = char::from_u32('a' as u32 + (next % 26) as u32).unwrap();
            next += 1;
            if next == TERMINAL_EVENT_QUEUE_CAPACITY {
                full_tx.send(()).unwrap();
            }
            Ok(Some(key(character)))
        })
        .unwrap();
        full_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        wait_until_queue_is_full(&reader);

        let (done_tx, done_rx) = std_mpsc::sync_channel(0);
        let shutdown = thread::spawn(move || {
            let mut reader = reader;
            let result = reader.shutdown();
            let queued = (0..TERMINAL_EVENT_QUEUE_CAPACITY)
                .map(|_| reader.events().try_recv().unwrap().unwrap())
                .collect::<Vec<_>>();
            done_tx.send((result, queued)).unwrap();
        });
        let (result, queued) = done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("shutdown did not acknowledge and join with a full queue");
        result.unwrap();
        shutdown.join().unwrap();

        let expected = (0..TERMINAL_EVENT_QUEUE_CAPACITY)
            .map(|index| key(char::from_u32('a' as u32 + (index % 26) as u32).unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(queued, expected);
    }

    #[test]
    fn terminal_ownership_seam_resumes_reader_after_success_and_error() {
        let polls = Arc::new(AtomicUsize::new(0));
        let source_polls = Arc::clone(&polls);
        let mut reader = TerminalReader::start_with_source(move |_| {
            source_polls.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(1));
            Ok(None)
        })
        .unwrap();
        let controller = reader.controller();
        wait_for_poll_after(&polls, 0);

        with_reader_paused(Some(&controller), || {
            let paused_at = polls.load(Ordering::SeqCst);
            thread::sleep(Duration::from_millis(10));
            assert_eq!(polls.load(Ordering::SeqCst), paused_at);
            Ok(())
        })
        .unwrap();
        let after_success = polls.load(Ordering::SeqCst);
        wait_for_poll_after(&polls, after_success);

        let result: anyhow::Result<()> = with_reader_paused(Some(&controller), || {
            let paused_at = polls.load(Ordering::SeqCst);
            thread::sleep(Duration::from_millis(10));
            assert_eq!(polls.load(Ordering::SeqCst), paused_at);
            Err(anyhow::anyhow!("editor failed"))
        });
        assert_eq!(result.unwrap_err().to_string(), "editor failed");
        let after_error = polls.load(Ordering::SeqCst);
        wait_for_poll_after(&polls, after_error);

        reader.shutdown().unwrap();
    }
}
