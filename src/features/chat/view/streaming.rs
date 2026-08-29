use ratatui::text::{Line, Span};

use crate::domain::activity::ActivityState;
use crate::markdown::{self, CardBlock};
use crate::render_state::{
    Card, CardKind, RenderState, SessionIdentity, StreamKind, StreamingCacheKeyRef, ThemeCacheKey,
};
use crate::theme::Theme;
use crate::view_shared::{SpinnerKind, spinner};

pub(crate) struct StreamingRenderInput<'a> {
    pub(crate) session_identity: SessionIdentity,
    pub(crate) fallback_ordinal: usize,
    pub(crate) activity: &'a ActivityState,
    pub(crate) is_turn_active: bool,
    pub(crate) content: &'a str,
    pub(crate) content_message_id: Option<&'a str>,
    pub(crate) thinking: &'a str,
    pub(crate) thinking_message_id: Option<&'a str>,
    pub(crate) show_thinking: bool,
    pub(crate) full_width: u16,
    pub(crate) theme: ThemeCacheKey,
    pub(crate) tick: u64,
}

pub(crate) fn build_streaming_card(
    input: StreamingRenderInput<'_>,
    render: &mut RenderState,
) -> Option<Card> {
    let activity_text = match input.activity {
        ActivityState::RunningTool { name } => {
            format!("{} tool: {name}", spinner(SpinnerKind::Braille, input.tick))
        }
        ActivityState::Compacting { .. } => {
            format!("{} compacting", spinner(SpinnerKind::Braille, input.tick))
        }
        ActivityState::Streaming => {
            format!("{} streaming", spinner(SpinnerKind::Braille, input.tick))
        }
        _ => format!("{} thinking", spinner(SpinnerKind::Braille, input.tick)),
    };

    let session = render.observe_session(&input.session_identity);
    let has_thinking = input.show_thinking && !input.thinking.is_empty();
    let has_content = !input.content.is_empty();

    if has_thinking || has_content {
        let mut blocks = Vec::new();

        if has_thinking {
            let key = StreamingCacheKeyRef::new(
                &session,
                StreamKind::Thinking,
                input.thinking_message_id,
                input.content_message_id,
                input.fallback_ordinal,
                input.thinking,
                input.full_width,
                input.theme,
                input.show_thinking,
            );
            let mut thinking_blocks = if let Some(cached) = render.streaming_blocks(key) {
                cached.to_vec()
            } else {
                let rendered =
                    markdown::render(input.thinking, Theme::thinking_text(), render.highlighter());
                render.store_streaming_blocks(key, rendered.clone());
                rendered
            };
            markdown::prepend_span_to_first_text(
                &mut thinking_blocks,
                Span::styled("\u{25CF} ", Theme::thinking()),
            );
            blocks.extend(thinking_blocks);
            if has_content {
                blocks.push(CardBlock::Text(Line::default()));
            }
        }

        if has_content {
            let key = StreamingCacheKeyRef::new(
                &session,
                StreamKind::Content,
                input.content_message_id,
                input.thinking_message_id,
                input.fallback_ordinal,
                input.content,
                input.full_width,
                input.theme,
                input.show_thinking,
            );
            let content_blocks = if let Some(cached) = render.streaming_blocks(key) {
                cached.to_vec()
            } else {
                let rendered =
                    markdown::render(input.content, Theme::assistant_text(), render.highlighter());
                render.store_streaming_blocks(key, rendered.clone());
                rendered
            };
            blocks.extend(content_blocks);
        }

        blocks.push(CardBlock::Text(Line::from(Span::styled(
            activity_text,
            Theme::thinking(),
        ))));
        Some(Card::new(CardKind::Streaming, blocks))
    } else if input.is_turn_active {
        Some(Card::new(
            CardKind::Thinking,
            vec![CardBlock::Text(Line::from(Span::styled(
                activity_text,
                Theme::thinking(),
            )))],
        ))
    } else {
        None
    }
}
