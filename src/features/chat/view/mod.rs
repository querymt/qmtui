mod cards;
mod completions;
mod composer;
mod elicitation;
pub(crate) mod header;
mod history;
mod screen;
mod streaming;
mod tools;
mod viewport;

pub(crate) use cards::{FinalizedRenderInput, build_finalized_cards};
pub(crate) use completions::{completion_panel_height, draw_completion_panel};
pub(crate) use composer::{draw_input_panel, input_layout_metrics};
pub(crate) use elicitation::{
    draw_popup as draw_elicitation_popup, popup_height as elicitation_popup_height,
};
pub(crate) use history::{ForkPopupInput, draw_fork_turn_popup};
pub(crate) use screen::{ChatScreenInput, draw_chat, draw_delegate_view};
#[cfg(test)]
pub(crate) use streaming::{StreamingRenderInput, build_streaming_card};

#[cfg(test)]
pub(crate) fn build_message_cards(app: &mut crate::app::App) -> &[crate::render_state::Card] {
    build_message_cards_for_width(app, 120)
}

#[cfg(test)]
pub(crate) fn build_message_cards_for_width(
    app: &mut crate::app::App,
    full_width: u16,
) -> &[crate::render_state::Card] {
    let now_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();
    build_message_cards_for_width_at(app, full_width, now_unix_secs)
}

#[cfg(test)]
pub(crate) fn build_message_cards_for_width_at(
    app: &mut crate::app::App,
    full_width: u16,
    now_unix_secs: i64,
) -> &[crate::render_state::Card] {
    let effective_cwd = app.current_session_cwd();
    let input = FinalizedRenderInput {
        session_identity: screen::session_identity(&app.sessions),
        messages: &app.chat.messages,
        delegates: &app.delegates.delegate_entries,
        effective_cwd,
        show_thinking: app.chat.show_thinking,
        full_width,
        theme: crate::render_state::ThemeCacheKey::current_frame(),
        now_unix_secs,
    };
    build_finalized_cards(input, &mut app.render)
}

#[cfg(test)]
pub(crate) fn build_streaming_card_for_test(
    app: &mut crate::app::App,
    full_width: u16,
) -> Option<crate::render_state::Card> {
    let input = StreamingRenderInput {
        session_identity: screen::session_identity(&app.sessions),
        fallback_ordinal: app.chat.messages.len(),
        activity: &app.chat.activity,
        is_turn_active: app.chat.is_turn_active(),
        content: &app.chat.streaming_content,
        content_message_id: app.chat.streaming_content_message_id.as_deref(),
        thinking: &app.chat.streaming_thinking,
        thinking_message_id: app.chat.streaming_thinking_message_id.as_deref(),
        show_thinking: app.chat.show_thinking,
        full_width,
        theme: crate::render_state::ThemeCacheKey::current_frame(),
        tick: app.render.tick,
    };
    build_streaming_card(input, &mut app.render)
}
