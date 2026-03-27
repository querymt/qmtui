/// Metadata for a single slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashCommandDef {
    pub name: &'static str,
    pub description: &'static str,
    /// `true` = conceptually chat-screen-only (informational; used in help display).
    pub chat_only: bool,
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
        name: "theme",
        description: "open theme picker",
        chat_only: false,
    },
    SlashCommandDef {
        name: "sessions",
        description: "open session switcher",
        chat_only: false,
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
