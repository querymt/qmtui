mod help;
mod palette;
mod theme;

pub(crate) use help::{HelpInputResult, handle_key as handle_help_key};
pub(crate) use palette::{PaletteInputResult, handle_key as handle_palette_key};
pub(crate) use theme::{ThemeInputResult, handle_key as handle_theme_key};
