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

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use crate::app::App;

    use super::{
        apply_external_editor_outcome, apply_external_editor_result, editor_command_from_env,
    };

    #[test]
    fn editor_command_prefers_visual_over_editor() {
        let env = [("VISUAL", Some("nvim -f")), ("EDITOR", Some("vim"))];
        let cmd = editor_command_from_env(&env).expect("expected editor command");
        assert_eq!(cmd.program, "nvim");
        assert_eq!(cmd.args, vec![OsString::from("-f")]);
    }

    #[test]
    fn editor_command_uses_editor_when_visual_missing() {
        let env = [("VISUAL", None), ("EDITOR", Some("nano"))];
        let cmd = editor_command_from_env(&env).expect("expected editor command");
        assert_eq!(cmd.program, "nano");
        assert!(cmd.args.is_empty());
    }

    #[test]
    fn editor_command_rejects_blank_values() {
        let env = [("VISUAL", Some("   ")), ("EDITOR", Some(""))];
        assert!(editor_command_from_env(&env).is_none());
    }

    #[test]
    fn apply_external_editor_result_updates_input_and_cursor() {
        let mut app = App::new();
        app.input = "old".into();
        app.input_cursor = 1;
        app.input_scroll = 3;

        apply_external_editor_result(&mut app, "new text".into());

        assert_eq!(app.input, "new text");
        assert_eq!(app.input_cursor, "new text".len());
        assert_eq!(app.input_scroll, 0);
    }

    #[test]
    fn apply_external_editor_outcome_updates_input_on_success() {
        let mut app = App::new();
        app.input = "draft".into();

        apply_external_editor_outcome(&mut app, Ok(Some("revised prompt".into())));

        assert_eq!(app.input, "revised prompt");
        assert_eq!(app.input_cursor, "revised prompt".len());
        assert_eq!(app.status, "loaded prompt from external editor");
        assert!(matches!(app.logs.last(), Some(entry) if entry.target == "editor"));
    }

    #[test]
    fn apply_external_editor_outcome_keeps_input_on_cancel() {
        let mut app = App::new();
        app.input = "draft".into();

        apply_external_editor_outcome(&mut app, Ok(None));

        assert_eq!(app.input, "draft");
        assert_eq!(app.status, "external editor cancelled");
    }
}
