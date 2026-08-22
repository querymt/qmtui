use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::application::ExternalEditorOutcome;

use super::editor::open_external_editor;

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
