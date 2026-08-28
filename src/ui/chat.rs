use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::Block,
};

use crate::app::App;
use crate::domain::activity::{ActivityState, DelegateEntry, DelegateStatus};
use crate::domain::chat::ChatEntry;
use crate::features::chat::view::{
    FinalizedRenderInput, build_finalized_cards, completion_panel_height, draw_completion_panel,
    draw_elicitation_popup, draw_input_panel, elicitation_popup_height, input_layout_metrics,
};
use crate::markdown;
use crate::render_state::{
    Card, CardKind, RenderState, SessionIdentity, StreamKind, StreamingCacheKeyRef, ThemeCacheKey,
};
use crate::theme::Theme;

use super::draw_header;
use super::start::short_cwd;

// ── Spinner ───────────────────────────────────────────────────────────────────

const BRAILLE_SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpinnerKind {
    Braille,
    Line,
    Dots,
}

const LINE_SPINNER: &[&str] = &["-", "\\", "|", "/"];
const DOTS_SPINNER: &[&str] = &[".  ", ".. ", "..."];

pub(super) fn spinner_frames(kind: SpinnerKind) -> &'static [&'static str] {
    match kind {
        SpinnerKind::Braille => BRAILLE_SPINNER,
        SpinnerKind::Line => LINE_SPINNER,
        SpinnerKind::Dots => DOTS_SPINNER,
    }
}

pub(crate) fn spinner(kind: SpinnerKind, tick: u64) -> &'static str {
    let frames = spinner_frames(kind);
    frames[(tick as usize / 2) % frames.len()]
}

// ── Popup symbols ─────────────────────────────────────────────────────────────
pub(crate) const CHECK_CHECKED: &str = "\u{2611} "; // ☑ ballot box checked   – success/done
pub(crate) const CHECK_FAILED: &str = "\u{2612}"; // ☒ ballot box with X     – failed

// ── Status bar icons ──────────────────────────────────────────────────────────
const ICON_CONTEXT: &str = "\u{1F5AA}"; // 🖪 document      – context token usage
const ICON_TOOLS: &str = "\u{2692}"; // ⚒  tools          – tool call count
pub(crate) const ICON_DELEGATES: &str = "\u{2387}"; // ⎇  alt/fork       – delegation count
pub(crate) const ICON_MULTI_SESSION: &str = "𐬽"; // multi-session recent activity indicator
pub(crate) const ICON_MESH: &str = "\u{1F5A7}"; // mesh nodes (U+1F5A7)

struct MessagesRenderInput<'a> {
    session_identity: SessionIdentity,
    messages: &'a [ChatEntry],
    delegates: &'a [DelegateEntry],
    effective_cwd: Option<String>,
    show_thinking: bool,
    activity: &'a ActivityState,
    is_turn_active: bool,
    streaming_content: &'a str,
    streaming_content_message_id: Option<&'a str>,
    streaming_thinking: &'a str,
    streaming_thinking_message_id: Option<&'a str>,
}

fn render_session_identity(app: &App) -> SessionIdentity {
    let session_id = app.sessions.session_id.clone();
    let remote_node_id = session_id
        .as_deref()
        .and_then(|id| app.sessions.session_remote_node_id(id))
        .map(str::to_string);
    let is_remote = session_id
        .as_deref()
        .is_some_and(|id| app.sessions.is_remote_session_id(id));
    SessionIdentity::new(session_id, remote_node_id, is_remote)
}

fn now_unix_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn build_message_cards(app: &mut App) -> &[Card] {
    build_message_cards_for_width(app, 120)
}

#[cfg(test)]
pub(crate) fn build_message_cards_for_width(app: &mut App, full_width: u16) -> &[Card] {
    build_message_cards_for_width_at(app, full_width, now_unix_secs())
}

#[cfg(test)]
pub(crate) fn build_message_cards_for_width_at(
    app: &mut App,
    full_width: u16,
    now_unix_secs: i64,
) -> &[Card] {
    let session_identity = render_session_identity(app);
    let effective_cwd = app.current_session_cwd();
    let input = FinalizedRenderInput {
        session_identity,
        messages: &app.chat.messages,
        delegates: &app.delegates.delegate_entries,
        effective_cwd,
        show_thinking: app.chat.show_thinking,
        full_width,
        theme: ThemeCacheKey::current_frame(),
        now_unix_secs,
    };
    build_finalized_cards(input, &mut app.render)
}

// ── Shared header builder ─────────────────────────────────────────────────────

fn format_status_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!(
            "{}h{:02}m{:02}s",
            secs / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    }
}

/// Build left + right span vectors for the chat/delegate header bar.
fn build_chat_header_spans(app: &App) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let model_str = match (&app.models.current_provider, &app.models.current_model) {
        (Some(p), Some(m)) => {
            if let Some(node_id) = app.models.current_model_node_id.as_deref() {
                let label = app
                    .models
                    .models
                    .iter()
                    .find(|entry| {
                        crate::models_state::ModelsState::model_entry_matches_node(
                            entry,
                            p,
                            m,
                            Some(node_id),
                        )
                    })
                    .and_then(|e| e.node_label.as_deref())
                    .unwrap_or(node_id);
                format!("{p}/{m}@{label}")
            } else {
                format!("{p}/{m}")
            }
        }
        _ => "no model".into(),
    };
    let sid = app
        .sessions
        .session_id
        .as_deref()
        .map(|s| if s.len() > 8 { &s[..8] } else { s })
        .unwrap_or("???");

    let mut right_spans: Vec<Span<'static>> = Vec::new();

    if let Some(dur) = app.chat.llm_request_elapsed() {
        right_spans.push(Span::styled(
            format!(" {} ", format_status_duration(dur)),
            Theme::status(),
        ));
    }

    if let Some(context_tokens) = app.chat.session_stats.latest_context_tokens
        && context_tokens > 0
        && app.chat.context_limit > 0
    {
        let pct = (context_tokens as f64 / app.chat.context_limit as f64 * 100.0) as u32;
        right_spans.push(Span::styled(
            format!(" {ICON_CONTEXT} {pct}% "),
            Theme::status(),
        ));
    }

    if app.chat.session_stats.total_tool_calls > 0 {
        right_spans.push(Span::styled(
            format!(" {ICON_TOOLS} {} ", app.chat.session_stats.total_tool_calls),
            Theme::status(),
        ));
    }

    if let Some(cost) = app.chat.cumulative_cost
        && cost > 0.0
    {
        right_spans.push(Span::styled(
            format!(" ${cost:.4} "),
            Theme::status_accent(),
        ));
    }

    if !app.delegates.delegate_entries.is_empty() {
        let (mut done, mut has_failed, mut has_running, mut awaiting_input) =
            (0usize, false, false, false);
        for e in &app.delegates.delegate_entries {
            match e.status {
                DelegateStatus::Completed | DelegateStatus::Cancelled => done += 1,
                DelegateStatus::Failed => has_failed = true,
                DelegateStatus::InProgress => has_running = true,
            }
            awaiting_input |= e.awaiting_input();
        }
        let total = app.delegates.delegate_entries.len();
        let style = if has_failed {
            Theme::error_on_dim()
        } else if has_running {
            Theme::status_accent()
        } else {
            Theme::status()
        };
        right_spans.push(Span::styled(
            format!(" {ICON_DELEGATES} {done}/{total} "),
            style,
        ));
        if awaiting_input {
            right_spans.push(Span::styled(" awaiting input ", Theme::mode_badge("plan")));
        }
    }

    let other_active_session_count = app.sessions.other_active_session_count();
    if other_active_session_count > 0 {
        right_spans.push(Span::styled(
            format!(" {ICON_MULTI_SESSION} {other_active_session_count} "),
            Theme::status(),
        ));
    }

    if let Some(span) = super::mesh_header_span(app) {
        right_spans.push(span);
    }

    let effort_label = app.models.reasoning_effort_label().to_string();
    right_spans.push(Span::styled(
        format!(" {model_str}"),
        Theme::status_accent(),
    ));
    right_spans.push(Span::styled(":", Theme::reasoning_effort_sep()));
    right_spans.push(Span::styled(
        format!("{effort_label} "),
        Theme::reasoning_effort_level(),
    ));

    let sid_span = Span::styled(format!(" {sid}"), Theme::status());
    let mut left_spans = vec![sid_span];
    if let Some(cwd) = app.current_session_cwd() {
        left_spans.push(Span::styled(
            format!(":{}", short_cwd(&cwd, 20)),
            Style::default().fg(Theme::ok()).bg(Theme::bg_dim()),
        ));
    }
    left_spans.push(Span::styled(
        format!(" {} ", app.sessions.agent_mode),
        Theme::mode_badge(&app.sessions.agent_mode),
    ));
    if app.delegates.parent_session_id.is_some() {
        left_spans.push(Span::styled(" \u{2b11} child ", Theme::status_accent()));
    }
    let profile_label = app.current_profile_label();
    let active_profile_id = app.profiles.active_profile_id.as_deref();
    let current_profile_id = app.current_session_profile_id();
    let profile_text = if current_profile_id.is_some()
        && active_profile_id.is_some()
        && current_profile_id != active_profile_id
    {
        format!(
            " profile:{} (new:{}) ",
            profile_label,
            app.profiles.active_profile_label()
        )
    } else {
        format!(" profile:{} ", profile_label)
    };
    left_spans.push(Span::styled(profile_text, Theme::status()));

    (left_spans, right_spans)
}

// ── Draw delegate view (read-only child session) ─────────────────────────────

pub(super) fn draw_delegate_view(f: &mut Frame, app: &mut App) {
    let area = f.area();

    let (input_height, input_layout) = input_layout_metrics(&app.composer, &mut app.render, area);
    let elicitation_height = elicitation_popup_height(
        app.chat.elicitation.as_ref(),
        app.chat.elicitation_ui.as_ref(),
        &mut app.render,
        area,
    );

    let chunks = if elicitation_height > 0 {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),                  // header
                Constraint::Min(3),                     // messages
                Constraint::Length(elicitation_height), // elicitation popup
                Constraint::Length(1),                  // input border line
                Constraint::Length(input_height),       // input hint area
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // header
                Constraint::Min(3),    // messages (all remaining space)
            ])
            .split(area)
    };

    let (left_spans, right_spans) = build_chat_header_spans(app);
    draw_header(f, app, chunks[0], left_spans, right_spans);
    let session_identity = render_session_identity(app);
    let effective_cwd = app.current_session_cwd();
    let input = MessagesRenderInput {
        session_identity,
        messages: &app.chat.messages,
        delegates: &app.delegates.delegate_entries,
        effective_cwd,
        show_thinking: app.chat.show_thinking,
        activity: &app.chat.activity,
        is_turn_active: app.chat.is_turn_active(),
        streaming_content: &app.chat.streaming_content,
        streaming_content_message_id: app.chat.streaming_content_message_id.as_deref(),
        streaming_thinking: &app.chat.streaming_thinking,
        streaming_thinking_message_id: app.chat.streaming_thinking_message_id.as_deref(),
    };
    draw_messages(f, chunks[1], input, &mut app.render);

    if elicitation_height > 0 {
        if let (Some(state), Some(ui)) = (&app.chat.elicitation, &app.chat.elicitation_ui) {
            draw_elicitation_popup(f, state, ui, &mut app.render, chunks[2]);
        }
        draw_input_panel(
            f,
            &app.chat,
            &app.sessions.agent_mode,
            spinner(SpinnerKind::Braille, app.render.tick),
            &mut app.render,
            (chunks[3], chunks[4]),
            input_layout,
        );
    }
}

// ── Draw chat screen ──────────────────────────────────────────────────────────

pub(super) fn draw_chat(f: &mut Frame, app: &mut App) {
    let area = f.area();
    // Both slash-completion and @-mention share one panel slot (mutually exclusive).
    let completion_panel_height = completion_panel_height(&app.composer);

    let (input_height, input_layout) = input_layout_metrics(&app.composer, &mut app.render, area);
    let elicitation_height = elicitation_popup_height(
        app.chat.elicitation.as_ref(),
        app.chat.elicitation_ui.as_ref(),
        &mut app.render,
        area,
    );

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),                       // header
            Constraint::Min(3),                          // messages
            Constraint::Length(completion_panel_height), // slash or mention
            Constraint::Length(elicitation_height),      // elicitation popup (0 when inactive)
            Constraint::Length(1),                       // input border line
            Constraint::Length(input_height),            // input (dynamic)
        ])
        .split(area);

    let (left_spans, right_spans) = build_chat_header_spans(app);
    draw_header(f, app, chunks[0], left_spans, right_spans);

    // messages
    let session_identity = render_session_identity(app);
    let effective_cwd = app.current_session_cwd();
    let input = MessagesRenderInput {
        session_identity,
        messages: &app.chat.messages,
        delegates: &app.delegates.delegate_entries,
        effective_cwd,
        show_thinking: app.chat.show_thinking,
        activity: &app.chat.activity,
        is_turn_active: app.chat.is_turn_active(),
        streaming_content: &app.chat.streaming_content,
        streaming_content_message_id: app.chat.streaming_content_message_id.as_deref(),
        streaming_thinking: &app.chat.streaming_thinking,
        streaming_thinking_message_id: app.chat.streaming_thinking_message_id.as_deref(),
    };
    draw_messages(f, chunks[1], input, &mut app.render);

    if completion_panel_height > 0 {
        draw_completion_panel(
            f,
            &app.composer,
            spinner(SpinnerKind::Braille, app.render.tick),
            chunks[2],
        );
    }

    if elicitation_height > 0
        && let (Some(state), Some(ui)) = (&app.chat.elicitation, &app.chat.elicitation_ui)
    {
        draw_elicitation_popup(f, state, ui, &mut app.render, chunks[3]);
    }

    draw_input_panel(
        f,
        &app.chat,
        &app.sessions.agent_mode,
        spinner(SpinnerKind::Braille, app.render.tick),
        &mut app.render,
        (chunks[4], chunks[5]),
        input_layout,
    );
}

// ── Streaming card (transient, not cached) ────────────────────────────────────

/// Build the transient streaming/thinking card (not cached, rebuilt every frame).
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

/// Build the transient card while caching only its markdown blocks.
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
                blocks.push(crate::markdown::CardBlock::Text(Line::default()));
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

        blocks.push(crate::markdown::CardBlock::Text(Line::from(Span::styled(
            activity_text,
            Theme::thinking(),
        ))));
        Some(Card::new(CardKind::Streaming, blocks))
    } else if input.is_turn_active {
        Some(Card::new(
            CardKind::Thinking,
            vec![crate::markdown::CardBlock::Text(Line::from(Span::styled(
                activity_text,
                Theme::thinking(),
            )))],
        ))
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) fn build_streaming_card_for_test(app: &mut App, full_width: u16) -> Option<Card> {
    let input = StreamingRenderInput {
        session_identity: render_session_identity(app),
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
    build_streaming_card(input, &mut app.render)
}

// ── Draw messages area ────────────────────────────────────────────────────────

fn draw_messages(
    f: &mut Frame,
    area: Rect,
    input: MessagesRenderInput<'_>,
    render: &mut RenderState,
) {
    f.render_widget(Block::default().style(Theme::base()), area);

    let theme = ThemeCacheKey::current_frame();
    let finalized_input = FinalizedRenderInput {
        session_identity: input.session_identity.clone(),
        messages: input.messages,
        delegates: input.delegates,
        effective_cwd: input.effective_cwd,
        show_thinking: input.show_thinking,
        full_width: area.width,
        theme,
        now_unix_secs: now_unix_secs(),
    };
    build_finalized_cards(finalized_input, render);

    let streaming_input = StreamingRenderInput {
        session_identity: input.session_identity,
        fallback_ordinal: input.messages.len(),
        activity: input.activity,
        is_turn_active: input.is_turn_active,
        content: input.streaming_content,
        content_message_id: input.streaming_content_message_id,
        thinking: input.streaming_thinking,
        thinking_message_id: input.streaming_thinking_message_id,
        show_thinking: input.show_thinking,
        full_width: area.width,
        theme,
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

    let mut y: i32 = -(scroll as i32);
    for card in &all_cards {
        let card_h = card.height(area.width);
        let card_top = y;
        let card_bottom = y + card_h as i32;

        if card_bottom > 0 && card_top < area.height as i32 {
            let render_y = card_top.max(0) as u16;
            let visible_h = (card_bottom.min(area.height as i32) - render_y as i32) as u16;
            let clip_top = (-card_top).max(0) as u16;

            card.render(
                f,
                Rect {
                    x: area.x,
                    y: area.y + render_y,
                    width: area.width,
                    height: visible_h.min(card_h),
                },
                clip_top,
            );
        }

        y += card_h as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::activity::{DelegateChildState, DelegateStats};
    use crate::domain::tool::ToolDetail;

    fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
        buffer.content().iter().map(|cell| cell.symbol()).collect()
    }

    fn user(text: &str, id: &str) -> ChatEntry {
        ChatEntry::User {
            text: text.into(),
            message_id: Some(id.into()),
        }
    }

    fn render_messages_buffer(app: &mut App, width: u16, height: u16) -> ratatui::buffer::Buffer {
        let session_identity = render_session_identity(app);
        let effective_cwd = app.current_session_cwd();
        let input = MessagesRenderInput {
            session_identity,
            messages: &app.chat.messages,
            delegates: &app.delegates.delegate_entries,
            effective_cwd,
            show_thinking: app.chat.show_thinking,
            activity: &app.chat.activity,
            is_turn_active: app.chat.is_turn_active(),
            streaming_content: &app.chat.streaming_content,
            streaming_content_message_id: app.chat.streaming_content_message_id.as_deref(),
            streaming_thinking: &app.chat.streaming_thinking,
            streaming_thinking_message_id: app.chat.streaming_thinking_message_id.as_deref(),
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

    #[test]
    fn chat_header_shows_current_and_distinct_new_profile_labels() {
        let mut app = App::new();
        app.sessions.session_id = Some("session-1".into());
        app.sessions.agent_id = Some("agent-1".into());
        app.profiles.profiles = vec![
            crate::domain::profile::ProfileInfo {
                id: "current".into(),
                name: "Current".into(),
                ..Default::default()
            },
            crate::domain::profile::ProfileInfo {
                id: "next".into(),
                name: "Next".into(),
                ..Default::default()
            },
        ];
        app.profiles.active_profile_id = Some("next".into());
        app.profiles
            .bind_session_profile("session-1".into(), "current".into());
        let backend = ratatui::backend::TestBackend::new(100, 12);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();

        terminal.draw(|frame| draw_chat(frame, &mut app)).unwrap();

        assert!(buffer_text(terminal.backend().buffer()).contains("profile:Current (new:Next)"));
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
        let card = build_streaming_card_for_test(&mut app, 20).expect("streaming card");
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
