use crate::themes_gen::Base16Palette;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Screen {
    Sessions,
    Chat,
    /// Read-only view for delegate child sessions (no input box).
    Delegate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Popup {
    None,
    CommandPalette,
    Mesh,
    MeshInvite,
    MeshInviteQr,
    ModelSelect,
    SessionSelect,
    NewSession,
    ThemeSelect,
    Help,
    Log,
    ProviderAuth,
    ForkTurnSelect,
    ProfileSelect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandPaletteAction {
    OpenMesh,
    AttachRemoteSession,
    CreateRemoteSession,
    CreateMeshInvite,
    ModelSelect,
    SessionSelect,
    DelegateSessions,
    NewSession,
    ThemeSelect,
    Help,
    Log,
    ProviderAuth,
    ForkTurnSelect,
    ProfileSelect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandPaletteCommand {
    pub(crate) title: &'static str,
    pub(crate) description: &'static str,
    pub(crate) shortcut: &'static str,
    pub(crate) action: CommandPaletteAction,
    pub(crate) chat_only: bool,
}

pub(crate) const COMMAND_PALETTE_COMMANDS: &[CommandPaletteCommand] = &[
    CommandPaletteCommand {
        title: "Open Mesh",
        description: "View mesh nodes and remote sessions",
        shortcut: "",
        action: CommandPaletteAction::OpenMesh,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Attach remote session",
        description: "Attach an existing session from a mesh node",
        shortcut: "",
        action: CommandPaletteAction::AttachRemoteSession,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Create remote session",
        description: "Start a new session on a mesh node",
        shortcut: "",
        action: CommandPaletteAction::CreateRemoteSession,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Create mesh invite",
        description: "Generate a mesh invite link and QR code",
        shortcut: "",
        action: CommandPaletteAction::CreateMeshInvite,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Model selector",
        description: "Choose the model for this session or delegates",
        shortcut: "C-x m",
        action: CommandPaletteAction::ModelSelect,
        chat_only: true,
    },
    CommandPaletteCommand {
        title: "Session switcher",
        description: "Browse and load sessions",
        shortcut: "C-x l",
        action: CommandPaletteAction::SessionSelect,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Delegate sessions",
        description: "Browse delegate child sessions",
        shortcut: "",
        action: CommandPaletteAction::DelegateSessions,
        chat_only: true,
    },
    CommandPaletteCommand {
        title: "New session",
        description: "Start a new session in a directory",
        shortcut: "C-x n",
        action: CommandPaletteAction::NewSession,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Theme picker",
        description: "Change the UI theme",
        shortcut: "C-x t",
        action: CommandPaletteAction::ThemeSelect,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Help",
        description: "Show keyboard shortcuts and commands",
        shortcut: "C-x ?",
        action: CommandPaletteAction::Help,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Logs",
        description: "Show in-memory logs",
        shortcut: "Ctrl+l",
        action: CommandPaletteAction::Log,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Provider auth",
        description: "Manage provider authentication",
        shortcut: "C-x a",
        action: CommandPaletteAction::ProviderAuth,
        chat_only: false,
    },
    CommandPaletteCommand {
        title: "Fork selector",
        description: "Choose a turn to fork from",
        shortcut: "C-x f",
        action: CommandPaletteAction::ForkTurnSelect,
        chat_only: true,
    },
    CommandPaletteCommand {
        title: "Profile selector",
        description: "Choose the active profile for new sessions",
        shortcut: "C-x p",
        action: CommandPaletteAction::ProfileSelect,
        chat_only: false,
    },
];

pub(crate) struct NavigationState {
    pub(crate) screen: Screen,
    pub(crate) popup: Popup,
    pub(crate) chord: bool, // true after ctrl+x pressed, waiting for second key
    pub(crate) command_palette_cursor: usize,
    pub(crate) command_palette_filter: String,
    pub(crate) theme_cursor: usize,
    pub(crate) theme_filter: String,
    pub(crate) help_scroll: usize,
}

impl NavigationState {
    pub(crate) fn new() -> Self {
        Self {
            screen: Screen::Sessions,
            popup: Popup::None,
            chord: false,
            command_palette_cursor: 0,
            command_palette_filter: String::new(),
            theme_cursor: 0,
            theme_filter: String::new(),
            help_scroll: 0,
        }
    }

    pub(crate) fn open_command_palette(&mut self) {
        self.popup = Popup::CommandPalette;
        self.command_palette_filter.clear();
        self.command_palette_cursor = 0;
    }

    pub(crate) fn filtered_command_palette_commands(&self) -> Vec<&'static CommandPaletteCommand> {
        let q = self.command_palette_filter.trim().to_lowercase();
        COMMAND_PALETTE_COMMANDS
            .iter()
            .filter(|command| !command.chat_only || matches!(self.screen, Screen::Chat))
            .filter(|command| {
                q.is_empty()
                    || command.title.to_lowercase().contains(&q)
                    || command.description.to_lowercase().contains(&q)
                    || command.shortcut.to_lowercase().contains(&q)
            })
            .collect()
    }

    pub(crate) fn move_command_palette_cursor(&mut self, delta: isize) {
        self.command_palette_cursor = move_wrapping_cursor(
            self.command_palette_cursor,
            self.filtered_command_palette_commands().len(),
            delta,
        );
    }

    pub(crate) fn selected_command_palette_action(&self) -> Option<CommandPaletteAction> {
        self.filtered_command_palette_commands()
            .get(self.command_palette_cursor)
            .map(|command| command.action)
    }

    pub(crate) fn command_palette_filter_insert(&mut self, c: char) {
        self.command_palette_filter.push(c);
        self.command_palette_cursor = 0;
    }

    pub(crate) fn command_palette_filter_backspace(&mut self) {
        self.command_palette_filter.pop();
        self.command_palette_cursor = 0;
    }

    pub(crate) fn open_theme_selector(&mut self, current_index: usize) {
        self.popup = Popup::ThemeSelect;
        self.theme_filter.clear();
        self.theme_cursor = current_index;
    }

    pub(crate) fn theme_matches(&self, id: &str, label: &str) -> bool {
        let q = self.theme_filter.to_lowercase();
        q.is_empty() || label.to_lowercase().contains(&q) || id.to_lowercase().contains(&q)
    }

    pub(crate) fn filtered_themes<'a>(
        &self,
        themes: &'a [Base16Palette],
    ) -> Vec<(usize, &'a Base16Palette)> {
        themes
            .iter()
            .enumerate()
            .filter(|(_, theme)| self.theme_matches(theme.id, theme.label))
            .collect()
    }

    pub(crate) fn selected_theme_index(&self, themes: &[Base16Palette]) -> Option<usize> {
        if self.theme_filter.is_empty() {
            Some(self.theme_cursor)
        } else {
            self.filtered_themes(themes)
                .get(self.theme_cursor)
                .map(|(index, _)| *index)
        }
    }

    pub(crate) fn move_theme_cursor_up(&mut self) {
        self.theme_cursor = self.theme_cursor.saturating_sub(1);
    }

    pub(crate) fn move_theme_cursor_down(&mut self, filtered_len: usize) {
        let max = filtered_len.saturating_sub(1);
        self.theme_cursor = (self.theme_cursor + 1).min(max);
    }

    pub(crate) fn theme_filter_insert(&mut self, c: char) {
        self.theme_filter.push(c);
        self.theme_cursor = 0;
    }

    pub(crate) fn theme_filter_backspace(&mut self) {
        self.theme_filter.pop();
        self.theme_cursor = 0;
    }

    pub(crate) fn open_help(&mut self) {
        self.popup = Popup::Help;
        self.help_scroll = 0;
    }

    pub(crate) fn scroll_help_up(&mut self) {
        self.help_scroll = self.help_scroll.saturating_sub(1);
    }

    pub(crate) fn scroll_help_down(&mut self) {
        self.help_scroll = self.help_scroll.saturating_add(1);
    }
}

fn move_wrapping_cursor(cursor: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        0
    } else {
        (cursor as isize + delta).rem_euclid(len as isize) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command_titles(state: &NavigationState) -> Vec<&'static str> {
        state
            .filtered_command_palette_commands()
            .into_iter()
            .map(|command| command.title)
            .collect()
    }

    #[test]
    fn constructor_uses_exact_defaults() {
        let state = NavigationState::new();

        assert_eq!(state.screen, Screen::Sessions);
        assert_eq!(state.popup, Popup::None);
        assert!(!state.chord);
        assert_eq!(state.command_palette_cursor, 0);
        assert!(state.command_palette_filter.is_empty());
        assert_eq!(state.theme_cursor, 0);
        assert!(state.theme_filter.is_empty());
        assert_eq!(state.help_scroll, 0);
    }

    #[test]
    fn palette_filter_trims_lowercases_and_matches_title() {
        let mut state = NavigationState::new();
        state.command_palette_filter = "  tHeMe PICK  ".into();

        assert_eq!(command_titles(&state), ["Theme picker"]);
    }

    #[test]
    fn palette_filter_matches_description() {
        let mut state = NavigationState::new();
        state.command_palette_filter = "keyboard shortcuts".into();

        assert_eq!(command_titles(&state), ["Help"]);
    }

    #[test]
    fn palette_filter_matches_shortcut() {
        let mut state = NavigationState::new();
        state.screen = Screen::Chat;
        state.command_palette_filter = "C-X F".into();

        assert_eq!(command_titles(&state), ["Fork selector"]);
    }

    #[test]
    fn palette_chat_only_commands_require_chat_screen() {
        let mut state = NavigationState::new();

        state.screen = Screen::Sessions;
        assert!(!command_titles(&state).contains(&"Model selector"));
        state.screen = Screen::Delegate;
        assert!(!command_titles(&state).contains(&"Model selector"));
        state.screen = Screen::Chat;
        assert!(command_titles(&state).contains(&"Model selector"));
        assert!(command_titles(&state).contains(&"Delegate sessions"));
        assert!(command_titles(&state).contains(&"Fork selector"));
    }

    #[test]
    fn opening_palette_replaces_popup_and_resets_only_palette_input() {
        let mut state = NavigationState::new();
        state.popup = Popup::Help;
        state.chord = true;
        state.command_palette_filter = "theme".into();
        state.command_palette_cursor = 4;
        state.theme_filter = "keep".into();
        state.help_scroll = 8;

        state.open_command_palette();

        assert_eq!(state.popup, Popup::CommandPalette);
        assert!(state.chord);
        assert!(state.command_palette_filter.is_empty());
        assert_eq!(state.command_palette_cursor, 0);
        assert_eq!(state.theme_filter, "keep");
        assert_eq!(state.help_scroll, 8);
    }

    #[test]
    fn palette_cursor_wraps_and_selects_filtered_action() {
        let mut state = NavigationState::new();
        state.screen = Screen::Chat;

        state.move_command_palette_cursor(-1);
        assert_eq!(
            state.command_palette_cursor,
            state.filtered_command_palette_commands().len() - 1
        );

        state.command_palette_filter = "theme".into();
        state.command_palette_cursor = 0;
        assert_eq!(
            state.selected_command_palette_action(),
            Some(CommandPaletteAction::ThemeSelect)
        );
        state.move_command_palette_cursor(1);
        assert_eq!(state.command_palette_cursor, 0);
    }

    #[test]
    fn palette_empty_result_resets_cursor_and_has_no_selection() {
        let mut state = NavigationState::new();
        state.command_palette_filter = "no command matches this".into();
        state.command_palette_cursor = 9;

        state.move_command_palette_cursor(1);

        assert_eq!(state.command_palette_cursor, 0);
        assert_eq!(state.selected_command_palette_action(), None);
    }

    #[test]
    fn palette_filter_edits_reset_cursor() {
        let mut state = NavigationState::new();
        state.command_palette_cursor = 5;

        state.command_palette_filter_insert('x');
        assert_eq!(state.command_palette_filter, "x");
        assert_eq!(state.command_palette_cursor, 0);
        state.command_palette_cursor = 3;
        state.command_palette_filter_backspace();
        assert!(state.command_palette_filter.is_empty());
        assert_eq!(state.command_palette_cursor, 0);
    }

    #[test]
    fn opening_theme_selector_uses_supplied_index_and_resets_filter() {
        let mut state = NavigationState::new();
        state.popup = Popup::Help;
        state.theme_filter = "old".into();
        state.theme_cursor = 2;

        state.open_theme_selector(17);

        assert_eq!(state.popup, Popup::ThemeSelect);
        assert!(state.theme_filter.is_empty());
        assert_eq!(state.theme_cursor, 17);
    }

    #[test]
    fn theme_matching_is_case_insensitive_for_id_and_label() {
        let mut state = NavigationState::new();
        state.theme_filter = "PENUMBRA".into();
        assert!(state.theme_matches("other", "Penumbra Dark"));

        state.theme_filter = "BASE16-3024".into();
        assert!(state.theme_matches("base16-3024", "other"));
    }

    #[test]
    fn theme_matching_does_not_trim_query() {
        let mut state = NavigationState::new();
        state.theme_filter = " penumbra ".into();

        assert!(!state.theme_matches("base16-penumbra", "Penumbra"));
    }

    #[test]
    fn theme_cursor_clamps_without_wrapping() {
        let mut state = NavigationState::new();

        state.move_theme_cursor_up();
        assert_eq!(state.theme_cursor, 0);
        state.move_theme_cursor_down(3);
        state.move_theme_cursor_down(3);
        state.move_theme_cursor_down(3);
        assert_eq!(state.theme_cursor, 2);
        state.move_theme_cursor_down(0);
        assert_eq!(state.theme_cursor, 0);
    }

    #[test]
    fn theme_filter_edits_reset_cursor() {
        let mut state = NavigationState::new();
        state.theme_cursor = 12;

        state.theme_filter_insert('x');
        assert_eq!(state.theme_filter, "x");
        assert_eq!(state.theme_cursor, 0);
        state.theme_cursor = 4;
        state.theme_filter_backspace();
        assert!(state.theme_filter.is_empty());
        assert_eq!(state.theme_cursor, 0);
    }

    #[test]
    fn empty_theme_filter_preserves_out_of_range_selection_index() {
        let mut state = NavigationState::new();
        state.theme_cursor = 99;

        assert_eq!(state.selected_theme_index(&[]), Some(99));
    }

    #[test]
    fn nonempty_theme_filter_returns_no_selection_without_match() {
        let mut state = NavigationState::new();
        state.theme_filter = "missing".into();
        state.theme_cursor = 0;

        assert_eq!(state.selected_theme_index(&[]), None);
    }

    #[test]
    fn opening_help_resets_scroll() {
        let mut state = NavigationState::new();
        state.popup = Popup::ThemeSelect;
        state.help_scroll = 14;

        state.open_help();

        assert_eq!(state.popup, Popup::Help);
        assert_eq!(state.help_scroll, 0);
    }

    #[test]
    fn help_scroll_saturates_in_both_directions() {
        let mut state = NavigationState::new();

        state.scroll_help_up();
        assert_eq!(state.help_scroll, 0);
        state.help_scroll = usize::MAX;
        state.scroll_help_down();
        assert_eq!(state.help_scroll, usize::MAX);
    }
}
