/// Metadata for a single built-in slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommandDef {
    pub name: &'static str,
    pub description: &'static str,
    /// `true` = conceptually chat-screen-only (informational; used in help display).
    pub chat_only: bool,
}

/// A slash command shown in composer autocomplete.
///
/// Built-in commands stay in [`SLASH_COMMANDS`]. ACP-advertised commands are
/// owned strings because they arrive at runtime via `AvailableCommandsUpdate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommandItem {
    pub name: String,
    pub description: String,
}

impl SlashCommandItem {
    pub(crate) fn from_local(def: &SlashCommandDef) -> Self {
        Self {
            name: def.name.to_string(),
            description: def.description.to_string(),
        }
    }
}

/// Normalize an ACP-advertised command name.
///
/// Agents may send `review` or `/review`. The composer parser can represent any
/// non-empty name that does not contain whitespace or control characters.
pub(crate) fn normalize_command_name(name: &str) -> Option<String> {
    let name = name.strip_prefix('/').unwrap_or(name);
    if is_valid_command_name(name) {
        Some(name.to_string())
    } else {
        None
    }
}

pub(crate) fn is_valid_command_name(name: &str) -> bool {
    !name.is_empty() && !name.chars().any(|c| c.is_whitespace() || c.is_control())
}

pub(crate) fn advertised_description(description: &str, argument_hint: Option<&str>) -> String {
    match argument_hint.map(str::trim).filter(|hint| !hint.is_empty()) {
        Some(hint) => format!("{description} {hint}"),
        None => description.to_string(),
    }
}

pub(crate) fn is_local_command_name(name: &str) -> bool {
    SLASH_COMMANDS
        .iter()
        .any(|command| command.name.eq_ignore_ascii_case(name))
}

/// All registered slash commands, in the order they appear in the autocomplete popup.
pub const SLASH_COMMANDS: &[SlashCommandDef] = &[
    SlashCommandDef {
        name: "model",
        description: "model selector [filter]",
        chat_only: true,
    },
    SlashCommandDef {
        name: "mode",
        description: "switch mode (build, plan)",
        chat_only: true,
    },
    SlashCommandDef {
        name: "review",
        description: "enter review mode",
        chat_only: true,
    },
    SlashCommandDef {
        name: "theme",
        description: "open theme picker",
        chat_only: false,
    },
    SlashCommandDef {
        name: "profile",
        description: "set profile [id or name]",
        chat_only: false,
    },
    SlashCommandDef {
        name: "sessions",
        description: "open session switcher",
        chat_only: false,
    },
    SlashCommandDef {
        name: "delegates",
        description: "list delegate sessions",
        chat_only: true,
    },
    SlashCommandDef {
        name: "new",
        description: "new session",
        chat_only: false,
    },
    SlashCommandDef {
        name: "help",
        description: "show help",
        chat_only: false,
    },
    SlashCommandDef {
        name: "logs",
        description: "open logs popup",
        chat_only: false,
    },
    SlashCommandDef {
        name: "auth",
        description: "provider auth",
        chat_only: false,
    },
    SlashCommandDef {
        name: "fork",
        description: "fork latest turn",
        chat_only: true,
    },
    SlashCommandDef {
        name: "undo",
        description: "undo last turn",
        chat_only: true,
    },
    SlashCommandDef {
        name: "redo",
        description: "redo",
        chat_only: true,
    },
    SlashCommandDef {
        name: "editor",
        description: "open external editor",
        chat_only: true,
    },
    SlashCommandDef {
        name: "cancel",
        description: "cancel active turn",
        chat_only: true,
    },
    SlashCommandDef {
        name: "thinking",
        description: "set thinking (auto, low, med, high, max)",
        chat_only: true,
    },
    SlashCommandDef {
        name: "quit",
        description: "quit",
        chat_only: false,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_command_name_accepts_protocol_names_and_rejects_unsafe_tokens() {
        assert_eq!(
            normalize_command_name("/Explain-Error"),
            Some("Explain-Error".into())
        );
        assert_eq!(normalize_command_name("/help"), Some("help".into()));
        assert_eq!(
            normalize_command_name("project.create"),
            Some("project.create".into())
        );
        assert_eq!(
            normalize_command_name("namespace:command"),
            Some("namespace:command".into())
        );
        assert_eq!(
            normalize_command_name("/tmp/file.txt"),
            Some("tmp/file.txt".into())
        );
        assert_eq!(normalize_command_name("/123"), Some("123".into()));
        assert_eq!(normalize_command_name(""), None);
        assert_eq!(normalize_command_name("/"), None);
        assert_eq!(normalize_command_name("two words"), None);
        assert_eq!(normalize_command_name(" padded"), None);
        assert_eq!(normalize_command_name("line\nbreak"), None);
        assert_eq!(normalize_command_name("nul\0byte"), None);
    }

    #[test]
    fn advertised_description_appends_non_empty_hints() {
        assert_eq!(
            advertised_description("Explain an error", Some("[error]")),
            "Explain an error [error]"
        );
        assert_eq!(advertised_description("Show help", Some("  ")), "Show help");
        assert_eq!(advertised_description("Show help", None), "Show help");
    }

    #[test]
    fn local_command_lookup_is_case_insensitive() {
        assert!(is_local_command_name("Help"));
        assert!(!is_local_command_name("explain-error"));
    }
}
