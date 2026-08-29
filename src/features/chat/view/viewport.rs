use ratatui::{Frame, layout::Rect, widgets::Block};

use crate::chat_state::ChatState;
use crate::delegates_state::DelegatesState;
use crate::render_state::{Card, RenderState, SessionIdentity, ThemeCacheKey};
use crate::theme::Theme;

use super::streaming::{StreamingRenderInput, build_streaming_card};
use super::{FinalizedRenderInput, build_finalized_cards};

pub(crate) struct MessagesRenderInput<'a> {
    pub(crate) session_identity: SessionIdentity,
    pub(crate) chat: &'a ChatState,
    pub(crate) delegates: &'a DelegatesState,
    pub(crate) effective_cwd: Option<String>,
    pub(crate) theme: ThemeCacheKey,
    pub(crate) now_unix_secs: i64,
}

pub(crate) fn draw_messages(
    frame: &mut Frame,
    area: Rect,
    input: MessagesRenderInput<'_>,
    render: &mut RenderState,
) {
    frame.render_widget(Block::default().style(Theme::base()), area);

    let finalized_input = FinalizedRenderInput {
        session_identity: input.session_identity.clone(),
        messages: &input.chat.messages,
        delegates: &input.delegates.delegate_entries,
        effective_cwd: input.effective_cwd,
        show_thinking: input.chat.show_thinking,
        full_width: area.width,
        theme: input.theme,
        now_unix_secs: input.now_unix_secs,
    };
    build_finalized_cards(finalized_input, render);

    let streaming_input = StreamingRenderInput {
        session_identity: input.session_identity,
        fallback_ordinal: input.chat.messages.len(),
        activity: &input.chat.activity,
        is_turn_active: input.chat.is_turn_active(),
        content: &input.chat.streaming_content,
        content_message_id: input.chat.streaming_content_message_id.as_deref(),
        thinking: &input.chat.streaming_thinking,
        thinking_message_id: input.chat.streaming_thinking_message_id.as_deref(),
        show_thinking: input.chat.show_thinking,
        full_width: area.width,
        theme: input.theme,
        tick: render.tick,
    };
    let streaming_card = build_streaming_card(streaming_input, render);

    let total_height: u16 = render
        .cards()
        .iter()
        .chain(streaming_card.iter())
        .map(|card| card.height(area.width))
        .sum();

    if total_height == 0 && render.cards().is_empty() && streaming_card.is_none() {
        return;
    }

    render.compensate_chat_growth(total_height);
    let scroll = render.clamp_chat_scroll(total_height, area.height);

    let all_cards: Vec<&Card> = render.cards().iter().chain(streaming_card.iter()).collect();

    let mut y = -(scroll as i32);
    for card in &all_cards {
        let card_height = card.height(area.width);
        let card_top = y;
        let card_bottom = y + card_height as i32;

        if card_bottom > 0 && card_top < area.height as i32 {
            let render_y = card_top.max(0) as u16;
            let visible_height = (card_bottom.min(area.height as i32) - render_y as i32) as u16;
            let clip_top = (-card_top).max(0) as u16;

            card.render(
                frame,
                Rect {
                    x: area.x,
                    y: area.y + render_y,
                    width: area.width,
                    height: visible_height.min(card_height),
                },
                clip_top,
            );
        }

        y += card_height as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use crate::domain::activity::{
        ActivityState, DelegateChildState, DelegateEntry, DelegateStats, DelegateStatus,
    };
    use crate::domain::chat::ChatEntry;
    use crate::domain::tool::ToolDetail;
    use crate::features::chat::view::screen::session_identity;
    use crate::render_state::CardKind;

    fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
        buffer.content().iter().map(|cell| cell.symbol()).collect()
    }

    fn user(text: &str, id: &str) -> ChatEntry {
        ChatEntry::User {
            text: text.into(),
            message_id: Some(id.into()),
        }
    }

    fn now_unix_secs() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs() as i64)
            .unwrap_or_default()
    }

    fn render_messages_buffer(app: &mut App, width: u16, height: u16) -> ratatui::buffer::Buffer {
        let effective_cwd = app.current_session_cwd();
        let input = MessagesRenderInput {
            session_identity: session_identity(&app.sessions),
            chat: &app.chat,
            delegates: &app.delegates,
            effective_cwd,
            theme: ThemeCacheKey::current_frame(),
            now_unix_secs: now_unix_secs(),
        };
        let backend = ratatui::backend::TestBackend::new(width.max(1), height.max(1));
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                draw_messages(
                    frame,
                    Rect::new(0, 0, width, height),
                    input,
                    &mut app.render,
                );
            })
            .unwrap();
        terminal.backend().buffer().clone()
    }

    fn build_streaming_card_for_test(app: &mut App, full_width: u16) -> Card {
        let input = StreamingRenderInput {
            session_identity: session_identity(&app.sessions),
            fallback_ordinal: app.chat.messages.len(),
            activity: &app.chat.activity,
            is_turn_active: app.chat.is_turn_active(),
            content: &app.chat.streaming_content,
            content_message_id: app.chat.streaming_content_message_id.as_deref(),
            thinking: &app.chat.streaming_thinking,
            thinking_message_id: app.chat.streaming_thinking_message_id.as_deref(),
            show_thinking: app.chat.show_thinking,
            full_width,
            theme: ThemeCacheKey::current_frame(),
            tick: app.render.tick,
        };
        build_streaming_card(input, &mut app.render).expect("streaming card")
    }

    #[test]
    fn draw_messages_mutates_only_render_state() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());
        app.chat.messages.push(user("hello", "user-1"));
        app.chat.activity = ActivityState::Streaming;
        app.chat.streaming_content = "answer".into();
        app.chat.streaming_content_message_id = Some("assistant-1".into());
        app.chat.streaming_thinking = "plan".into();
        app.chat.streaming_thinking_message_id = Some("assistant-1".into());
        app.delegates.delegate_entries.push(DelegateEntry {
            delegation_id: "delegation-1".into(),
            child_session_id: Some("child-1".into()),
            delegate_tool_call_id: None,
            target_agent_id: Some("coder".into()),
            objective: "Implement the change".into(),
            status: DelegateStatus::InProgress,
            stats: DelegateStats::default(),
            started_at: None,
            ended_at: None,
            child_state: DelegateChildState::None,
        });
        let messages = format!("{:?}", app.chat.messages);
        let activity = app.chat.activity.clone();
        let streaming_content = app.chat.streaming_content.clone();
        let streaming_content_message_id = app.chat.streaming_content_message_id.clone();
        let streaming_thinking = app.chat.streaming_thinking.clone();
        let streaming_thinking_message_id = app.chat.streaming_thinking_message_id.clone();
        let show_thinking = app.chat.show_thinking;
        let session_id = app.sessions.session_id.clone();
        let delegates = app.delegates.delegate_entries.clone();

        let _ = render_messages_buffer(&mut app, 24, 5);

        assert_eq!(format!("{:?}", app.chat.messages), messages);
        assert_eq!(app.chat.activity, activity);
        assert_eq!(app.chat.streaming_content, streaming_content);
        assert_eq!(
            app.chat.streaming_content_message_id,
            streaming_content_message_id
        );
        assert_eq!(app.chat.streaming_thinking, streaming_thinking);
        assert_eq!(
            app.chat.streaming_thinking_message_id,
            streaming_thinking_message_id
        );
        assert_eq!(app.chat.show_thinking, show_thinking);
        assert_eq!(app.sessions.session_id, session_id);
        assert_eq!(app.delegates.delegate_entries, delegates);
        assert!(!app.render.cards().is_empty());
        assert!(app.render.test_chat_previous_total_height() > 0);
    }

    #[test]
    fn draw_messages_keeps_default_viewport_pinned_to_bottom() {
        let mut app = App::new();
        app.chat.messages = (0..4)
            .map(|index| user(&format!("message {index}"), &format!("user-{index}")))
            .collect();

        let buffer = render_messages_buffer(&mut app, 30, 4);

        assert_eq!(app.render.chat_scroll_offset(), 0);
        assert!(buffer_text(&buffer).contains("message 3"));
    }

    #[test]
    fn finalized_growth_compensates_scrolled_viewport_exactly() {
        let mut app = App::new();
        app.chat.messages = (0..3)
            .map(|index| user(&format!("message {index}"), &format!("user-{index}")))
            .collect();
        let _ = render_messages_buffer(&mut app, 30, 4);
        let previous_height = app.render.test_chat_previous_total_height();
        app.render.set_chat_scroll_offset(2);

        app.chat.messages.push(user("new message", "user-3"));
        let _ = render_messages_buffer(&mut app, 30, 4);
        let total_height = app.render.test_chat_previous_total_height();

        assert!(total_height > previous_height);
        assert_eq!(
            app.render.chat_scroll_offset(),
            2 + total_height.saturating_sub(previous_height)
        );
        let max_scroll = total_height.saturating_sub(4);
        assert_eq!(max_scroll - app.render.chat_scroll_offset(), 3);
    }

    #[test]
    fn streaming_growth_includes_wrapping_thinking_and_activity_row() {
        let mut app = App::new();
        app.chat.activity = ActivityState::Streaming;
        app.chat.streaming_content = "short".into();
        let _ = render_messages_buffer(&mut app, 20, 1);
        let previous_height = app.render.test_chat_previous_total_height();
        app.render.set_chat_scroll_offset(1);

        app.chat.streaming_content =
            "a much longer streaming answer that wraps over several terminal rows".into();
        app.chat.streaming_thinking = "thinking also wraps over multiple rows".into();
        let _ = render_messages_buffer(&mut app, 20, 1);
        let total_height = app.render.test_chat_previous_total_height();

        assert!(total_height > previous_height);
        assert_eq!(
            app.render.chat_scroll_offset(),
            1 + total_height.saturating_sub(previous_height)
        );
        let card = build_streaming_card_for_test(&mut app, 20);
        assert_eq!(card.height(20), total_height);
    }

    #[test]
    fn shrink_and_height_resize_clamp_bottom_relative_scroll() {
        let mut app = App::new();
        app.chat.messages = (0..4)
            .map(|index| user(&format!("message {index}"), &format!("user-{index}")))
            .collect();
        let _ = render_messages_buffer(&mut app, 30, 4);
        app.render.set_chat_scroll_offset(7);

        let _ = render_messages_buffer(&mut app, 30, 6);
        assert_eq!(app.render.chat_scroll_offset(), 6);

        app.chat.messages.truncate(2);
        let _ = render_messages_buffer(&mut app, 30, 4);
        assert_eq!(app.render.test_chat_previous_total_height(), 6);
        assert_eq!(app.render.chat_scroll_offset(), 2);
    }

    #[test]
    fn width_changes_compensate_growth_then_clamp_shrink() {
        let mut app = App::new();
        app.chat.messages.push(user(
            "This deliberately long message wraps differently as the viewport width changes.",
            "user-1",
        ));
        let _ = render_messages_buffer(&mut app, 40, 1);
        let wide_height = app.render.test_chat_previous_total_height();
        let wide_identity = app.render.test_card_identity(0);
        app.render.set_chat_scroll_offset(1);

        let _ = render_messages_buffer(&mut app, 18, 1);
        let narrow_height = app.render.test_chat_previous_total_height();
        let narrow_identity = app.render.test_card_identity(0);
        assert!(narrow_height > wide_height);
        assert_eq!(
            app.render.chat_scroll_offset(),
            1 + narrow_height.saturating_sub(wide_height)
        );
        assert_ne!(narrow_identity, wide_identity);

        let _ = render_messages_buffer(&mut app, 40, 1);
        assert_eq!(app.render.test_chat_previous_total_height(), wide_height);
        assert_eq!(
            app.render.chat_scroll_offset(),
            (1 + narrow_height.saturating_sub(wide_height)).min(wide_height.saturating_sub(1))
        );
    }

    #[test]
    fn hidden_thinking_preserves_tool_batch_height_for_viewport() {
        let mut app = App::new();
        app.chat.show_thinking = false;
        app.chat.messages = vec![
            ChatEntry::ToolCall {
                tool_call_id: Some("tool-1".into()),
                name: "read".into(),
                is_error: false,
                detail: ToolDetail::None,
            },
            ChatEntry::Thinking {
                content: "hidden".into(),
                message_id: Some("thinking-1".into()),
            },
            ChatEntry::ToolCall {
                tool_call_id: Some("tool-2".into()),
                name: "write".into(),
                is_error: false,
                detail: ToolDetail::None,
            },
        ];

        let _ = render_messages_buffer(&mut app, 30, 1);

        assert_eq!(app.render.cards().len(), 1);
        assert_eq!(
            app.render.cards()[0].kind,
            CardKind::Tool { compact: false }
        );
        assert_eq!(
            app.render.test_chat_previous_total_height(),
            app.render.cards()[0].height(30)
        );
    }

    #[test]
    fn empty_draw_preserves_stale_viewport_and_tiny_areas_do_not_panic() {
        let mut app = App::new();
        app.render.test_seed_chat_viewport(7, 19);

        let _ = render_messages_buffer(&mut app, 0, 0);
        assert_eq!(app.render.chat_scroll_offset(), 7);
        assert_eq!(app.render.test_chat_previous_total_height(), 19);

        app.chat.messages.push(user("tiny", "user-1"));
        let _ = render_messages_buffer(&mut app, 0, 0);
        let _ = render_messages_buffer(&mut app, 1, 1);
    }
}
