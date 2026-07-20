use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::app::{App, LogLevel};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EditorCommand {
    pub(super) program: OsString,
    pub(super) args: Vec<OsString>,
}

pub(super) fn editor_command_from_env(
    env: &[(impl AsRef<str>, Option<impl AsRef<str>>)],
) -> Option<EditorCommand> {
    env.iter().find_map(|(_, value)| {
        value
            .as_ref()
            .and_then(|value| parse_editor_command(value.as_ref().trim()))
    })
}

fn parse_editor_command(value: &str) -> Option<EditorCommand> {
    let parts: Vec<_> = value.split_whitespace().collect();
    let (program, args) = parts.split_first()?;
    Some(EditorCommand {
        program: OsString::from(program),
        args: args.iter().map(OsString::from).collect(),
    })
}

fn system_editor_command() -> Option<EditorCommand> {
    let visual = std::env::var("VISUAL").ok();
    let editor = std::env::var("EDITOR").ok();
    editor_command_from_env(&[("VISUAL", visual.as_deref()), ("EDITOR", editor.as_deref())])
}

fn temp_editor_file_path() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("qmt-tui-editor-{}-{nanos}.md", std::process::id()))
}

fn run_external_editor(command: &EditorCommand, path: &Path) -> anyhow::Result<Option<String>> {
    let status = Command::new(&command.program)
        .args(&command.args)
        .arg(path)
        .status()?;
    if !status.success() {
        return Ok(None);
    }
    Ok(Some(fs::read_to_string(path)?))
}

fn cleanup_temp_editor_file(path: &Path) -> anyhow::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

pub(super) fn open_external_editor(initial_text: &str) -> anyhow::Result<Option<String>> {
    let command = system_editor_command()
        .ok_or_else(|| anyhow::anyhow!("set $VISUAL or $EDITOR to use an external editor"))?;
    let path = temp_editor_file_path();
    fs::write(&path, initial_text)?;
    let result = run_external_editor(&command, &path);
    cleanup_temp_editor_file(&path)?;
    result
}

pub(super) fn apply_external_editor_result(app: &mut App, updated_input: String) {
    app.input = updated_input;
    app.input_cursor = app.input.len();
    app.input_scroll = 0;
    app.refresh_mention_state();
    app.refresh_slash_state();
}

pub(super) fn apply_external_editor_outcome(app: &mut App, result: anyhow::Result<Option<String>>) {
    match result {
        Ok(Some(updated_input)) => {
            apply_external_editor_result(app, updated_input);
            app.set_status(
                LogLevel::Info,
                "editor",
                "loaded prompt from external editor",
            );
        }
        Ok(None) => {
            app.set_status(LogLevel::Info, "editor", "external editor cancelled");
        }
        Err(err) => {
            app.set_status(
                LogLevel::Error,
                "editor",
                format!("external editor failed: {err}"),
            );
        }
    }
}
