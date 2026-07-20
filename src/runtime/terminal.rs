use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::{app::App, ui};

use super::editor::{apply_external_editor_outcome, open_external_editor};

pub(super) type AppTerminal = Terminal<CrosstermBackend<std::io::Stdout>>;

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
    app: &mut App,
) -> anyhow::Result<()> {
    terminal.show_cursor()?;
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    let result = open_external_editor(&app.input);

    enable_raw_mode()?;
    execute!(terminal.backend_mut(), EnterAlternateScreen)?;
    terminal.hide_cursor()?;
    terminal.clear()?;
    terminal.autoresize()?;
    app.card_cache.invalidate();
    app.streaming_cache.invalidate();
    apply_external_editor_outcome(app, result);
    terminal.draw(|frame| ui::draw(frame, app))?;
    Ok(())
}
