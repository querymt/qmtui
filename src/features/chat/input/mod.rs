mod completions;
mod composer;
mod coordination;
mod elicitation;

pub(crate) use completions::{CompletionResult, handle_key as handle_completion_key};
pub(crate) use composer::{ComposerKeyResult, handle_key as handle_composer_key};
pub(crate) use coordination::{
    CancelIntent, ChatCommandIntent, ChatInputContext, ChatInputResult, ChatViewportIntent,
    PromptSubmission, build_prompt_submission, chat_command_intent, handle_coordination_key,
};
pub(crate) use elicitation::{ElicitationResponseEffect, handle_key as handle_elicitation_key};
