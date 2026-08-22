use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::application::ExternalEditorOutcome;

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
        anyhow::bail!("external editor exited with status {status}");
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

pub(super) fn open_external_editor(initial_text: &str) -> ExternalEditorOutcome {
    open_external_editor_with_command(initial_text, system_editor_command())
}

fn open_external_editor_with_command(
    initial_text: &str,
    command: Option<EditorCommand>,
) -> ExternalEditorOutcome {
    let result = (|| {
        let command = command
            .ok_or_else(|| anyhow::anyhow!("set $VISUAL or $EDITOR to use an external editor"))?;
        let path = temp_editor_file_path();
        fs::write(&path, initial_text)?;
        let result = run_external_editor(&command, &path);
        cleanup_temp_editor_file(&path)?;
        result
    })();

    match result {
        Ok(Some(updated_input)) => ExternalEditorOutcome::Completed(updated_input),
        Ok(None) => ExternalEditorOutcome::Cancelled,
        Err(error) => ExternalEditorOutcome::Failed(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::{EditorCommand, editor_command_from_env, open_external_editor_with_command};
    use crate::application::ExternalEditorOutcome;
    use std::ffi::OsString;

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
    fn missing_editor_returns_typed_failure() {
        let outcome = open_external_editor_with_command("draft", None);
        assert!(matches!(
            outcome,
            ExternalEditorOutcome::Failed(message)
                if message == "set $VISUAL or $EDITOR to use an external editor"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn nonzero_editor_exit_returns_typed_failure_with_status() {
        let command = EditorCommand {
            program: OsString::from("sh"),
            args: vec![
                OsString::from("-c"),
                OsString::from("exit 7"),
                OsString::from("sh"),
            ],
        };

        let outcome = open_external_editor_with_command("draft", Some(command));

        assert!(matches!(
            outcome,
            ExternalEditorOutcome::Failed(message)
                if message == "external editor exited with status exit status: 7"
        ));
    }
}
