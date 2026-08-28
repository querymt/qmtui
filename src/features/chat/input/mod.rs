mod completions;
mod composer;
mod elicitation;

pub(crate) use completions::{CompletionResult, handle_key as handle_completion_key};
pub(crate) use composer::{ComposerKeyResult, handle_key as handle_composer_key};
pub(crate) use elicitation::{ElicitationResponseEffect, handle_key as handle_elicitation_key};
