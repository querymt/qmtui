use std::path::Path;

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, List, ListItem, ListState, Padding, Paragraph, Wrap},
};

use crate::app::App;
use crate::chat_state::{ChatState, ElicitationUiState};
use crate::composer_state::ComposerState;
use crate::domain::activity::{ActivityState, DelegateEntry, DelegateStatus, SessionOp};
use crate::domain::chat::{ChatEntry, ElicitationResponseOutcome};
use crate::domain::elicitation::ElicitationState;
use crate::domain::tool::{
    MultiEditSection, ShellOutput, SymbolDiffSection, SymbolReplacement, ToolDetail,
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::input_layout::InputVisualLayout;
use crate::markdown;
use crate::render_state::{
    Card, CardKind, DelegatePresentationKeyRef, FinalizedCardKeyRef, FinalizedMessageKeyRef,
    FinalizedRenderContextKey, RenderState, SessionIdentity, StreamKind, StreamingCacheKeyRef,
    ThemeCacheKey,
};
use crate::theme::Theme;

use super::start::short_cwd;
use super::{INPUT_OVERLINE, draw_header};

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

// ── Elicitation symbols ───────────────────────────────────────────────────────
const OUTCOME_BULLET: &str = "\u{25B8} ";
const RADIO_SELECTED: &str = "\u{25CF} "; // ● filled circle  – single-select active
const RADIO_UNSELECTED: &str = "\u{25CB} "; // ○ empty circle   – single-select inactive
pub(crate) const CHECK_CHECKED: &str = "\u{2611} "; // ☑ ballot box checked   – success/done
pub(crate) const CHECK_FAILED: &str = "\u{2612}"; // ☒ ballot box with X     – failed
const CHECK_UNCHECKED: &str = "\u{2610} "; // ☐ ballot box unchecked – multi-select off

// ── Status bar icons ──────────────────────────────────────────────────────────
const ICON_CONTEXT: &str = "\u{1F5AA}"; // 🖪 document      – context token usage
const ICON_TOOLS: &str = "\u{2692}"; // ⚒  tools          – tool call count
pub(crate) const ICON_DELEGATES: &str = "\u{2387}"; // ⎇  alt/fork       – delegation count
pub(crate) const ICON_MULTI_SESSION: &str = "𐬽"; // multi-session recent activity indicator
pub(crate) const ICON_MESH: &str = "\u{1F5A7}"; // mesh nodes (U+1F5A7)

// ── General text symbols ──────────────────────────────────────────────────────
const ARROW_UP: &str = "\u{2191}"; // ↑ upwards arrow

// ── Build message cards (cached) ──────────────────────────────────────────────

/// Build cards for finalized messages incrementally (cached).
/// Does NOT include the streaming/thinking card — that's built separately.
pub(crate) struct FinalizedRenderInput<'a> {
    pub(crate) session_identity: SessionIdentity,
    pub(crate) messages: &'a [ChatEntry],
    pub(crate) delegates: &'a [DelegateEntry],
    pub(crate) effective_cwd: Option<String>,
    pub(crate) show_thinking: bool,
    pub(crate) full_width: u16,
    pub(crate) theme: ThemeCacheKey,
    pub(crate) now_unix_secs: i64,
}

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

struct PlannedFinalizedCard<'a> {
    key: FinalizedCardKeyRef<'a>,
    source_start: usize,
}

fn flush_tool_plan<'a>(
    planned: &mut Vec<PlannedFinalizedCard<'a>>,
    pending: &mut Vec<FinalizedMessageKeyRef<'a>>,
    source_start: &mut Option<usize>,
) {
    if pending.is_empty() {
        return;
    }
    let compact = matches!(
        planned.last().map(|card| card.key.kind()),
        Some(CardKind::Assistant | CardKind::Streaming | CardKind::Thinking)
    );
    planned.push(PlannedFinalizedCard {
        key: FinalizedCardKeyRef::new(CardKind::Tool { compact }, std::mem::take(pending)),
        source_start: source_start.take().expect("tool plan source"),
    });
}

fn plan_finalized_cards<'a>(input: &'a FinalizedRenderInput<'a>) -> Vec<PlannedFinalizedCard<'a>> {
    let mut planned = Vec::new();
    let mut pending_tools = Vec::new();
    let mut tool_source_start = None;
    let mut delegate_idx = 0;

    for (ordinal, entry) in input.messages.iter().enumerate() {
        let delegate = if let ChatEntry::ToolCall {
            name, tool_call_id, ..
        } = entry
            && name == "delegate"
        {
            let matched = tool_call_id
                .as_deref()
                .and_then(|id| {
                    input
                        .delegates
                        .iter()
                        .find(|entry| entry.delegate_tool_call_id.as_deref() == Some(id))
                })
                .or_else(|| input.delegates.get(delegate_idx));
            delegate_idx += 1;
            Some(DelegatePresentationKeyRef::from_entry(
                matched,
                input.now_unix_secs,
            ))
        } else {
            None
        };
        let message_key = FinalizedMessageKeyRef::new(ordinal, entry, delegate);

        match entry {
            ChatEntry::ToolCall { .. } => {
                tool_source_start.get_or_insert(ordinal);
                pending_tools.push(message_key);
            }
            ChatEntry::Thinking { .. } if !input.show_thinking => {}
            _ => {
                flush_tool_plan(&mut planned, &mut pending_tools, &mut tool_source_start);
                let kind = match entry {
                    ChatEntry::User { .. } => CardKind::User,
                    ChatEntry::Assistant { .. } => CardKind::Assistant,
                    ChatEntry::Thinking { .. } => CardKind::Thinking,
                    ChatEntry::CompactionStart { .. } | ChatEntry::CompactionEnd { .. } => {
                        CardKind::Compaction
                    }
                    ChatEntry::Info(_) => CardKind::Info,
                    ChatEntry::Error(_) => CardKind::Error,
                    ChatEntry::Elicitation { .. } => CardKind::Elicitation,
                    ChatEntry::ToolCall { .. } => unreachable!("tool calls are batched"),
                };
                planned.push(PlannedFinalizedCard {
                    key: FinalizedCardKeyRef::new(kind, vec![message_key]),
                    source_start: ordinal,
                });
            }
        }
    }
    flush_tool_plan(&mut planned, &mut pending_tools, &mut tool_source_start);
    planned
}

pub(crate) fn build_finalized_cards<'a>(
    input: FinalizedRenderInput<'_>,
    render: &'a mut RenderState,
) -> &'a [Card] {
    let session = render.observe_session(&input.session_identity);
    let context = FinalizedRenderContextKey::new(
        session,
        input.full_width,
        input.theme,
        input.show_thinking,
        input.effective_cwd.clone(),
    );
    let planned = plan_finalized_cards(&input);
    let desired: Vec<_> = planned.iter().map(|card| card.key.clone()).collect();
    let mismatch = render.prepare_finalized_cards(context, &desired);

    if let Some(first) = planned.get(mismatch) {
        let prior_kind = render.cards().last().map(|card| card.kind.clone());
        let rebuilt =
            render_card_suffix(&input, render.highlighter(), first.source_start, prior_kind);
        debug_assert_eq!(rebuilt.len(), planned.len() - mismatch);
        for (planned_card, card) in planned[mismatch..].iter().zip(rebuilt) {
            render.push_finalized_card(&planned_card.key, card);
        }
    }
    render.finish_finalized_cards(input.messages.len());
    render.cards()
}

fn render_card_suffix(
    input: &FinalizedRenderInput<'_>,
    highlighter: &crate::highlight::Highlighter,
    start_idx: usize,
    prior_kind: Option<CardKind>,
) -> Vec<Card> {
    let mut cards = Vec::new();
    let mut pending_tools: Vec<Line<'static>> = Vec::new();
    let current_cwd = input.effective_cwd.as_deref();

    let flush_tools = |tools: &mut Vec<Line<'static>>, cards: &mut Vec<Card>| {
        if !tools.is_empty() {
            let lines = std::mem::take(tools);
            let blocks = lines
                .into_iter()
                .map(crate::markdown::CardBlock::Text)
                .collect();
            let previous_kind = cards.last().map(|card| &card.kind).or(prior_kind.as_ref());
            let compact = matches!(
                previous_kind,
                Some(CardKind::Assistant | CardKind::Streaming | CardKind::Thinking)
            );
            cards.push(Card::new(CardKind::Tool { compact }, blocks));
        }
    };

    let delegate_entry_for_tool =
        |tool_call_id: Option<&String>, sequential_idx: usize| -> Option<&DelegateEntry> {
            tool_call_id
                .and_then(|id| {
                    input
                        .delegates
                        .iter()
                        .find(|entry| entry.delegate_tool_call_id.as_deref() == Some(id.as_str()))
                })
                .or_else(|| input.delegates.get(sequential_idx))
        };
    let mut delegate_idx = input.messages[..start_idx]
        .iter()
        .filter(|entry| matches!(entry, ChatEntry::ToolCall { name, .. } if name == "delegate"))
        .count();

    for entry in &input.messages[start_idx..] {
        match entry {
            ChatEntry::User { text, .. } => {
                flush_tools(&mut pending_tools, &mut cards);
                let blocks = markdown::render(text, Theme::user_text(), highlighter);
                cards.push(Card::new(CardKind::User, blocks));
            }
            ChatEntry::Assistant {
                content, thinking, ..
            } => {
                flush_tools(&mut pending_tools, &mut cards);
                let mut blocks = Vec::new();
                if input.show_thinking
                    && let Some(thinking_text) = thinking
                {
                    let mut rendered =
                        markdown::render(thinking_text, Theme::thinking_text(), highlighter);
                    markdown::prepend_span_to_first_text(
                        &mut rendered,
                        Span::styled("\u{25CF} ", Theme::thinking()),
                    );
                    blocks.extend(rendered);
                    blocks.push(crate::markdown::CardBlock::Text(Line::default()));
                }
                blocks.extend(markdown::render(
                    content,
                    Theme::assistant_text(),
                    highlighter,
                ));
                cards.push(Card::new(CardKind::Assistant, blocks));
            }
            ChatEntry::Thinking { content, .. } => {
                if input.show_thinking {
                    flush_tools(&mut pending_tools, &mut cards);
                    let mut blocks = markdown::render(content, Theme::thinking_text(), highlighter);
                    markdown::prepend_span_to_first_text(
                        &mut blocks,
                        Span::styled("\u{25CF} ", Theme::thinking()),
                    );
                    cards.push(Card::new(CardKind::Thinking, blocks));
                }
            }
            ChatEntry::ToolCall {
                name,
                is_error,
                detail,
                tool_call_id,
            } => {
                let style = if *is_error {
                    Theme::tool_error()
                } else {
                    Theme::tool_text()
                };
                let sym = if *is_error { "x" } else { ">" };

                // For delegate tool calls, compute duration and lifecycle from delegate entries.
                let delegate_entry = if name == "delegate" {
                    let entry = delegate_entry_for_tool(tool_call_id.as_ref(), delegate_idx);
                    delegate_idx += 1;
                    entry
                } else {
                    None
                };
                let delegate_duration = delegate_entry
                    .and_then(|entry| {
                        let start = entry.started_at?;
                        let end = entry.ended_at.unwrap_or(input.now_unix_secs);
                        let secs = (end - start).max(0) as u64;
                        Some(if secs < 60 {
                            format!(":{secs}s")
                        } else {
                            format!(":{}m{}s", secs / 60, secs % 60)
                        })
                    })
                    .unwrap_or_default();
                let tool_style = style;
                match detail {
                    ToolDetail::Edit {
                        file,
                        old,
                        new,
                        start_line,
                        ..
                    } => {
                        pending_tools.push(Line::from(vec![
                            Span::styled(format!("{sym} {name} "), style),
                            Span::styled(short_path(file).to_string(), Theme::diff_file()),
                        ]));
                        pending_tools.extend(build_diff_lines(old, new, *start_line));
                    }
                    ToolDetail::MultiEdit {
                        file,
                        edit_count,
                        sections,
                    } => {
                        pending_tools.push(Line::from(vec![
                            Span::styled(format!("{sym} {name} "), style),
                            Span::styled(short_path(file).to_string(), Theme::diff_file()),
                            Span::styled(format!(" ({edit_count} edits)"), Theme::status_accent()),
                        ]));
                        pending_tools.extend(build_multiedit_lines(sections, 6));
                    }
                    ToolDetail::ReplaceSymbolInput { replacements } => {
                        let title = replace_symbol_title(replacements, &[], current_cwd);
                        pending_tools.push(Line::from(vec![
                            Span::styled(format!("{sym} {name} "), style),
                            Span::styled(title, Theme::diff_file()),
                        ]));
                    }
                    ToolDetail::ReplaceSymbolDiff {
                        replacements,
                        sections,
                    } => {
                        let title = replace_symbol_title(replacements, sections, current_cwd);
                        pending_tools.push(Line::from(vec![
                            Span::styled(format!("{sym} {name} "), style),
                            Span::styled(title, Theme::diff_file()),
                        ]));
                        pending_tools.extend(build_symbol_diff_lines(sections, 4));
                    }
                    ToolDetail::Shell {
                        command,
                        arguments,
                        workdir,
                        output,
                    } => {
                        let mut spans = vec![Span::styled(format!("{sym} {name}"), style)];
                        if let Some(workdir) = workdir.as_deref().filter(|s| !s.trim().is_empty()) {
                            spans.push(Span::styled("@", tool_style));
                            spans.push(Span::styled(workdir.to_string(), Theme::diff_file()));
                        }
                        pending_tools.push(Line::from(spans));
                        let command = shell_command_display(command, arguments);
                        pending_tools.extend(build_shell_lines(&command, output.as_ref()));
                    }
                    ToolDetail::ReadTool {
                        path,
                        start_line,
                        end_line,
                    } => {
                        let mut spans = vec![
                            Span::styled(format!("{sym} {name} "), style),
                            Span::styled(short_path(path).to_string(), Theme::diff_file()),
                        ];
                        if let Some(start) = start_line {
                            spans.push(Span::styled(":", tool_style));
                            let range = match end_line {
                                Some(end) if end != start => format!("{start}-{end}"),
                                _ => start.to_string(),
                            };
                            spans.push(Span::styled(range, Theme::status_accent()));
                        }
                        pending_tools.push(Line::from(spans));
                    }
                    ToolDetail::WriteFile { path, content } => {
                        pending_tools.push(Line::from(vec![
                            Span::styled(format!("{sym} {name} "), style),
                            Span::styled(short_path(path).to_string(), Theme::diff_file()),
                        ]));
                        pending_tools.extend(build_write_lines(content));
                    }
                    ToolDetail::Generic { input, result } => {
                        let (summary, output) = match (input.as_deref(), result.as_deref()) {
                            (Some(input), result) => (input, result),
                            (None, Some(result)) => (result, None),
                            (None, None) => ("", None),
                        };
                        push_summary_lines(&mut pending_tools, sym, name, style, summary, output);
                    }
                    ToolDetail::SearchText {
                        pattern,
                        path,
                        include,
                        counts,
                    } => {
                        let location = if !include.is_empty() {
                            include.as_str()
                        } else if !path.is_empty() {
                            short_path(path)
                        } else {
                            "."
                        };
                        let mut summary = format!("\"{pattern}\" {location}");
                        if let Some(counts) = counts {
                            summary.push_str(&format!(
                                " ({} files, {} matches)",
                                counts.files, counts.matches
                            ));
                        }
                        push_summary_lines(&mut pending_tools, sym, name, style, &summary, None);
                    }
                    ToolDetail::Glob { pattern, path } => {
                        let summary = if path.is_empty() {
                            pattern.clone()
                        } else {
                            format!("{pattern} in {}", short_path(path))
                        };
                        push_summary_lines(&mut pending_tools, sym, name, style, &summary, None);
                    }
                    ToolDetail::List { path } => {
                        let summary = if path.is_empty() {
                            "."
                        } else {
                            short_path(path)
                        };
                        push_summary_lines(&mut pending_tools, sym, name, style, summary, None);
                    }
                    ToolDetail::Index { path, metadata } => {
                        let mut summary = if path.is_empty() {
                            ".".to_string()
                        } else if metadata.is_some() {
                            path.clone()
                        } else {
                            short_path(path).to_string()
                        };
                        if let Some(metadata) = metadata {
                            summary.push_str(&format!(
                                " ({}, {} imports, {} functions)",
                                metadata.language, metadata.imports, metadata.functions
                            ));
                        }
                        push_summary_lines(&mut pending_tools, sym, name, style, &summary, None);
                    }
                    ToolDetail::DeleteFile { path } => push_summary_lines(
                        &mut pending_tools,
                        sym,
                        name,
                        style,
                        short_path(path),
                        None,
                    ),
                    ToolDetail::Browse { url } => {
                        let summary = truncate_summary(url, 60);
                        push_summary_lines(&mut pending_tools, sym, name, style, &summary, None);
                    }
                    ToolDetail::Todo { items } => {
                        pending_tools
                            .push(Line::from(Span::styled(format!("{sym} {name}"), style)));
                        for item in items {
                            let check = if item.status == "completed" { "x" } else { " " };
                            pending_tools.push(Line::from(Span::styled(
                                format!("  [{check}] {}", item.content),
                                Theme::diff_file(),
                            )));
                        }
                    }
                    ToolDetail::Delegate {
                        target_agent_id,
                        objective,
                    } => {
                        let objective = truncate_summary(objective, 50);
                        if target_agent_id.is_empty() {
                            push_summary_lines(
                                &mut pending_tools,
                                sym,
                                name,
                                style,
                                &objective,
                                None,
                            );
                        } else {
                            let label = if delegate_duration.is_empty() {
                                format!("({target_agent_id})")
                            } else {
                                format!("({target_agent_id}{delegate_duration})")
                            };
                            let (delegate_status, delegate_status_style) = match delegate_entry {
                                Some(entry) if entry.awaiting_input() => {
                                    ("awaiting input", Theme::mode_badge("plan"))
                                }
                                Some(entry) => match entry.status {
                                    DelegateStatus::InProgress => {
                                        ("running", Theme::status_accent())
                                    }
                                    DelegateStatus::Completed => ("done", Theme::status()),
                                    DelegateStatus::Failed => ("failed", Theme::tool_error()),
                                    DelegateStatus::Cancelled => ("cancelled", Theme::status()),
                                },
                                None => ("queued", Theme::status()),
                            };
                            let mut spans = vec![Span::styled(format!("{sym} {name} "), style)];
                            spans.push(Span::styled(format!("{label} "), Theme::status_accent()));
                            spans.push(Span::styled(
                                format!("{delegate_status} "),
                                delegate_status_style,
                            ));
                            if !objective.is_empty() {
                                spans.push(Span::styled(objective, Theme::diff_file()));
                            }
                            pending_tools.push(Line::from(spans));
                        }
                    }
                    ToolDetail::LanguageQuery { action, uri } => {
                        let summary = format!("{action} {}", short_path(uri));
                        push_summary_lines(&mut pending_tools, sym, name, style, &summary, None);
                    }
                    ToolDetail::Question { .. } => {
                        push_summary_lines(&mut pending_tools, sym, name, style, "asking...", None)
                    }
                    ToolDetail::ApplyPatch { .. } => {
                        push_summary_lines(&mut pending_tools, sym, name, style, "patch", None)
                    }
                    ToolDetail::None => {
                        pending_tools
                            .push(Line::from(Span::styled(format!("{sym} {name}"), style)));
                    }
                }
            }
            ChatEntry::CompactionStart { token_estimate } => {
                flush_tools(&mut pending_tools, &mut cards);
                let token_str = format!("~{} tokens", token_estimate);
                cards.push(Card::new(
                    CardKind::Compaction,
                    vec![
                        crate::markdown::CardBlock::Text(Line::from(vec![
                            Span::styled("[compact] ", Theme::status_accent()),
                            Span::styled(
                                "Summarizing conversation history",
                                Theme::status_accent(),
                            ),
                        ])),
                        crate::markdown::CardBlock::Text(Line::from(Span::styled(
                            format!("  {token_str}"),
                            Theme::status(),
                        ))),
                    ],
                ));
            }
            ChatEntry::CompactionEnd {
                token_estimate,
                summary,
                summary_len,
            } => {
                flush_tools(&mut pending_tools, &mut cards);
                let mut blocks = vec![crate::markdown::CardBlock::Text(Line::from(vec![
                    Span::styled("[compact] ", Theme::status_accent()),
                    Span::styled("Conversation summarized", Theme::status_accent()),
                ]))];
                if let Some(token_estimate) = token_estimate {
                    blocks.push(crate::markdown::CardBlock::Text(Line::from(Span::styled(
                        format!("  ~{} tokens -> {} chars", token_estimate, summary_len),
                        Theme::status(),
                    ))));
                } else {
                    blocks.push(crate::markdown::CardBlock::Text(Line::from(Span::styled(
                        format!("  {} chars", summary_len),
                        Theme::status(),
                    ))));
                }
                blocks.push(crate::markdown::CardBlock::Text(Line::default()));
                blocks.extend(markdown::render(
                    summary,
                    Theme::assistant_text(),
                    highlighter,
                ));
                cards.push(Card::new(CardKind::Compaction, blocks));
            }
            ChatEntry::Info(text) => {
                flush_tools(&mut pending_tools, &mut cards);
                cards.push(Card::new(
                    CardKind::Info,
                    vec![crate::markdown::CardBlock::Text(Line::from(text.clone()))],
                ));
            }
            ChatEntry::Error(text) => {
                flush_tools(&mut pending_tools, &mut cards);
                cards.push(Card::new(
                    CardKind::Error,
                    vec![crate::markdown::CardBlock::Text(Line::from(text.clone()))],
                ));
            }
            ChatEntry::Elicitation {
                message,
                source: _,
                outcome,
                ..
            } => {
                flush_tools(&mut pending_tools, &mut cards);
                let header = crate::markdown::CardBlock::Text(Line::from(vec![
                    Span::styled("[?] ", Theme::status_accent()),
                    Span::styled(message.clone(), Theme::status_accent()),
                ]));
                let mut card_blocks = vec![header];
                match outcome {
                    None => {
                        card_blocks.push(crate::markdown::CardBlock::Text(Line::from(
                            Span::styled("  waiting for response\u{2026}", Theme::thinking()),
                        )));
                    }
                    Some(outcome) => {
                        for (text, style) in elicitation_outcome_lines(outcome) {
                            card_blocks.push(crate::markdown::CardBlock::Text(Line::from(
                                Span::styled(format!("  {text}"), style),
                            )));
                        }
                    }
                }
                cards.push(Card::new(CardKind::Elicitation, card_blocks));
            }
        }
    }

    flush_tools(&mut pending_tools, &mut cards);
    cards
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

fn elicitation_outcome_lines(outcome: &ElicitationResponseOutcome) -> Vec<(String, Style)> {
    let (texts, style) = match outcome {
        ElicitationResponseOutcome::Selected(labels) => (
            labels
                .iter()
                .map(|label| format!("{OUTCOME_BULLET}{label}"))
                .collect::<Vec<_>>()
                .join("\n")
                .lines()
                .map(str::to_string)
                .collect(),
            Theme::info_text(),
        ),
        ElicitationResponseOutcome::Text(text) => (
            text.lines().map(str::to_string).collect(),
            Theme::info_text(),
        ),
        ElicitationResponseOutcome::Boolean(true) => (vec!["Yes".into()], Theme::info_text()),
        ElicitationResponseOutcome::Boolean(false) => (vec!["No".into()], Theme::info_text()),
        ElicitationResponseOutcome::Declined => (vec!["declined".into()], Theme::status()),
        ElicitationResponseOutcome::Cancelled => (vec!["cancelled".into()], Theme::status()),
        ElicitationResponseOutcome::UnsupportedSchema => (
            vec!["unsupported schema - cannot answer in TUI".into()],
            Theme::info_text(),
        ),
        ElicitationResponseOutcome::Responded => (vec!["responded".into()], Theme::info_text()),
    };
    texts.into_iter().map(|text| (text, style)).collect()
}

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
        use crate::domain::activity::DelegateStatus;
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
            &app.composer,
            &app.chat,
            &app.sessions.agent_mode,
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
    let completion_panel_height = if app.composer.slash_state.is_some()
        || app.composer.mention_state.is_some()
        || app.composer.file_index_loading
        || app.composer.file_index_error.is_some()
    {
        6u16
    } else {
        0u16
    };

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
        if app.composer.slash_state.is_some() {
            draw_slash_panel(f, app, chunks[2]);
        } else {
            draw_mention_panel(f, app, chunks[2]);
        }
    }

    if elicitation_height > 0
        && let (Some(state), Some(ui)) = (&app.chat.elicitation, &app.chat.elicitation_ui)
    {
        draw_elicitation_popup(f, state, ui, &mut app.render, chunks[3]);
    }

    draw_input_panel(
        f,
        &app.composer,
        &app.chat,
        &app.sessions.agent_mode,
        &mut app.render,
        (chunks[4], chunks[5]),
        input_layout,
    );
}

fn input_layout_metrics(
    composer: &ComposerState,
    render: &mut RenderState,
    area: Rect,
) -> (u16, InputVisualLayout) {
    // Compute how many visual rows the input text needs when wrapped.
    let input_inner_width = area.width.saturating_sub(4) as usize;
    let prefix_width = 2usize; // "> "
    let input_layout = render.prepare_composer_input_layout(
        &composer.input,
        composer.input_cursor,
        input_inner_width,
        prefix_width,
    );
    let max_input_lines: u16 = 5;
    let input_height = (input_layout.total_rows() as u16).clamp(1, max_input_lines) + 1; // +1 bottom padding

    (input_height, input_layout)
}

fn wrap_elicitation_message(text: &str, width: u16) -> Vec<String> {
    let width = width.max(1) as usize;
    let mut rows = Vec::new();

    for logical_line in text.lines() {
        let mut row = String::new();
        let mut row_width = 0;
        for word in logical_line.split_whitespace() {
            let word_width = UnicodeWidthStr::width(word);
            if !row.is_empty() && row_width + 1 + word_width > width {
                rows.push(std::mem::take(&mut row));
                row_width = 0;
            }

            if word_width > width {
                if !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                    row_width = 0;
                }
                for ch in word.chars() {
                    let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if row_width > 0 && row_width + ch_width > width {
                        rows.push(std::mem::take(&mut row));
                        row_width = 0;
                    }
                    row.push(ch);
                    row_width += ch_width;
                }
            } else {
                if !row.is_empty() {
                    row.push(' ');
                    row_width += 1;
                }
                row.push_str(word);
                row_width += word_width;
            }
        }
        rows.push(row);
    }

    if rows.is_empty() {
        rows.push(String::new());
    }
    rows
}

fn wrapped_text_rows(text: &str, width: u16) -> u16 {
    wrap_elicitation_message(text, width)
        .len()
        .min(u16::MAX as usize) as u16
}

fn elicitation_option_text(label: &str, description: Option<&str>) -> String {
    description
        .map(|description| format!("{label}  {description}"))
        .unwrap_or_else(|| label.to_string())
}

fn elicitation_option_rows(state: &ElicitationState, ui: &ElicitationUiState, width: u16) -> u16 {
    use crate::domain::elicitation::ElicitationFieldKind;
    let schema_rows = match ui
        .current_field_index(state.fields.len())
        .and_then(|index| state.fields.get(index))
        .map(|field| &field.kind)
    {
        Some(ElicitationFieldKind::SingleSelect { options })
        | Some(ElicitationFieldKind::MultiSelect { options }) => options
            .iter()
            .map(|option| {
                wrapped_text_rows(
                    &elicitation_option_text(&option.label, option.description.as_deref()),
                    width,
                )
            })
            .sum(),
        _ => 1,
    };
    let option_count = match ui
        .current_field_index(state.fields.len())
        .and_then(|index| state.fields.get(index))
        .map(|field| &field.kind)
    {
        Some(ElicitationFieldKind::SingleSelect { options })
        | Some(ElicitationFieldKind::MultiSelect { options }) => options.len(),
        _ => 0,
    };
    schema_rows + u16::from(state.allow_custom && option_count > 0)
}

fn elicitation_popup_height(
    state: Option<&ElicitationState>,
    ui: Option<&ElicitationUiState>,
    render: &mut RenderState,
    area: Rect,
) -> u16 {
    if let (Some(state), Some(ui)) = (state, ui) {
        let message_width = area.width.saturating_sub(4).max(1);
        let message_rows = wrapped_text_rows(&state.message, message_width);
        let option_width = area.width.saturating_sub(6).max(1);
        let option_rows = elicitation_option_rows(state, ui, option_width);
        let custom_rows = if ui.custom_active {
            let width = area.width.saturating_sub(4) as usize;
            render
                .prepare_elicitation_custom_layout(&state.custom_input, ui.custom_cursor, width, 2)
                .total_rows() as u16
        } else {
            0
        };
        // Top padding, title, wrapped message, choices/input, and hint, each separated as needed.
        let content_rows = option_rows.max(1) + custom_rows.min(5);
        let custom_spacing = u16::from(ui.custom_active);
        (7 + message_rows.max(1) + content_rows + custom_spacing).min(area.height.saturating_sub(3))
    } else {
        0
    }
}

fn draw_input_panel(
    f: &mut Frame,
    _composer: &ComposerState,
    chat: &ChatState,
    agent_mode: &str,
    render: &mut RenderState,
    areas: (Rect, Rect),
    input_layout: InputVisualLayout,
) {
    let (border_area, input_area) = areas;
    // input border line reflects active session state
    let border_style = match &chat.activity {
        ActivityState::SessionOp(SessionOp::Undo) => Theme::input_border_undo(),
        ActivityState::SessionOp(SessionOp::Redo) => Theme::input_border_redo(),
        _ if chat.cancel_confirm_active() => Theme::input_border_cancel_confirm(),
        ActivityState::Compacting { .. } => Theme::input_border_compacting(),
        ActivityState::Thinking | ActivityState::Streaming | ActivityState::RunningTool { .. } => {
            Theme::input_border_thinking()
        }
        _ if chat.elicitation.is_some() => Theme::input_border_thinking(), // accent while waiting
        _ => Theme::mode_border(agent_mode),
    };
    let border_line =
        Paragraph::new(INPUT_OVERLINE.repeat(border_area.width as usize)).style(border_style);
    f.render_widget(border_line, border_area);

    // input area
    let input_bg = Block::default()
        .padding(Padding::new(2, 2, 0, 1))
        .style(Theme::input());
    let inner = input_bg.inner(input_area);
    f.render_widget(input_bg, input_area);

    let (label_text, label_style) = match &chat.activity {
        ActivityState::SessionOp(SessionOp::Undo) => (
            format!("{} undoing ", spinner(SpinnerKind::Braille, render.tick)),
            Theme::input_undo(),
        ),
        ActivityState::SessionOp(SessionOp::Redo) => (
            format!("{} redoing ", spinner(SpinnerKind::Braille, render.tick)),
            Theme::input_redo(),
        ),
        _ if chat.cancel_confirm_active() => (
            format!(
                "{} Esc again to stop ",
                spinner(SpinnerKind::Braille, render.tick)
            ),
            Theme::input_cancel_confirm(),
        ),
        ActivityState::Compacting { .. }
        | ActivityState::RunningTool { .. }
        | ActivityState::Thinking
        | ActivityState::Streaming => (
            format!("{} ", spinner(SpinnerKind::Braille, render.tick)),
            Theme::input_thinking(),
        ),
        _ if chat.elicitation.is_some() => (
            format!("  answer above {ARROW_UP} "),
            Theme::input_thinking(),
        ),
        _ => ("> ".into(), Theme::mode_border(agent_mode)),
    };
    let input_style = Theme::input();
    let hide_input_contents = chat.should_hide_input_contents();

    if hide_input_contents {
        render.ensure_composer_input_cursor_visible(inner.height, true);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(label_text, label_style))),
            inner,
        );
    } else {
        let layout = input_layout;
        let mut lines: Vec<Line<'static>> = Vec::new();
        for (idx, row) in layout.rows.iter().enumerate() {
            if idx == 0 {
                lines.push(Line::from(vec![
                    Span::styled(label_text.clone(), label_style),
                    Span::styled(row.text.clone(), input_style),
                ]));
            } else {
                lines.push(Line::from(Span::styled(row.text.clone(), input_style)));
            }
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled("", input_style)));
        }

        let visible = inner.height;
        let scroll = render.ensure_composer_input_cursor_visible(visible, false);

        f.render_widget(Paragraph::new(lines).scroll((scroll, 0)), inner);

        let visual_row = (layout.cursor_row as u16).saturating_sub(scroll);
        if visual_row < visible {
            f.set_cursor_position((inner.x + layout.cursor_col as u16, inner.y + visual_row));
        }
    }
}

// ── Elicitation popup ─────────────────────────────────────────────────────────

fn draw_elicitation_popup(
    f: &mut Frame,
    state: &ElicitationState,
    ui: &ElicitationUiState,
    render: &mut RenderState,
    area: Rect,
) {
    use crate::domain::elicitation::ElicitationFieldKind;

    if area.height == 0 || area.width == 0 {
        return;
    }

    f.render_widget(Block::default().style(Theme::popup_bg()), area);
    let inner = Rect {
        x: area.x + 1,
        y: area.y,
        width: area.width.saturating_sub(2),
        height: area.height,
    };
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let mut row = inner.y;
    let max_y = inner.y + inner.height;
    // Leave one row of padding before the header.
    row = row.saturating_add(1).min(max_y);
    if row < max_y {
        let title_style = Theme::status_accent().add_modifier(Modifier::BOLD);
        let title = Line::from(vec![
            Span::styled(OUTCOME_BULLET, title_style),
            Span::styled("Question", title_style),
        ]);
        f.render_widget(
            Paragraph::new(title).style(Theme::popup_bg()),
            Rect::new(inner.x, row, inner.width, 1),
        );
        row += 1;
    }
    // Keep the title visually separate from the question body.
    row = row.saturating_add(1).min(max_y);
    if row < max_y {
        let message_area = Rect::new(inner.x + 2, row, inner.width.saturating_sub(2), max_y - row);
        let message_lines: Vec<Line<'static>> =
            wrap_elicitation_message(&state.message, message_area.width)
                .into_iter()
                .map(|line| Line::from(Span::styled(line, Theme::fg())))
                .collect();
        let message_rows = message_lines.len().max(1) as u16;
        let message = Paragraph::new(message_lines).style(Theme::popup_bg());
        let visible_rows = message_rows.min(message_area.height);
        f.render_widget(
            message,
            Rect::new(
                message_area.x,
                message_area.y,
                message_area.width,
                visible_rows,
            ),
        );
        row += visible_rows;
    }
    // Keep the choices visually separate from the question text.
    row = row.saturating_add(1).min(max_y);

    let Some(field) = ui
        .current_field_index(state.fields.len())
        .and_then(|index| state.fields.get(index))
        .cloned()
    else {
        return;
    };
    match &field.kind {
        ElicitationFieldKind::SingleSelect { options }
        | ElicitationFieldKind::MultiSelect { options } => {
            let is_multi = matches!(&field.kind, ElicitationFieldKind::MultiSelect { .. });
            let selected_vals = state.selected.get(&field.name);
            for (idx, opt) in options.iter().enumerate() {
                if row >= max_y {
                    break;
                }
                let highlighted = !ui.custom_active && idx == ui.option_cursor;
                let is_chosen = if is_multi {
                    matches!(selected_vals, Some(serde_json::Value::Array(arr)) if arr.contains(&opt.value))
                } else {
                    selected_vals == Some(&opt.value)
                };
                let bullet = if is_multi {
                    if is_chosen {
                        CHECK_CHECKED
                    } else {
                        CHECK_UNCHECKED
                    }
                } else if highlighted {
                    RADIO_SELECTED
                } else {
                    RADIO_UNSELECTED
                };
                let style = if highlighted {
                    Theme::status_accent()
                } else {
                    Theme::status()
                };
                let text_width = inner.width.saturating_sub(4).max(1);
                let option_rows = wrapped_text_rows(
                    &elicitation_option_text(&opt.label, opt.description.as_deref()),
                    text_width,
                )
                .min(max_y - row);
                f.render_widget(
                    Paragraph::new(Span::styled(format!("  {bullet}"), style))
                        .style(Theme::popup_bg()),
                    Rect::new(inner.x, row, 4.min(inner.width), option_rows),
                );
                let option_line = Line::from(vec![
                    Span::styled(opt.label.clone(), style),
                    opt.description
                        .as_ref()
                        .map(|desc| Span::styled(format!("  {desc}"), Theme::dim()))
                        .unwrap_or_else(|| Span::raw("")),
                ]);
                f.render_widget(
                    Paragraph::new(option_line)
                        .style(Theme::popup_bg())
                        .wrap(Wrap { trim: true }),
                    Rect::new(inner.x + 4, row, text_width, option_rows),
                );
                row += option_rows;
            }

            if state.allow_custom && !options.is_empty() && row < max_y {
                let highlighted = ui.option_cursor == options.len();
                let style = if highlighted || ui.custom_active {
                    Theme::status_accent()
                } else {
                    Theme::status()
                };
                let bullet = if is_multi {
                    if ui.custom_active {
                        CHECK_CHECKED
                    } else {
                        CHECK_UNCHECKED
                    }
                } else if highlighted {
                    RADIO_SELECTED
                } else {
                    RADIO_UNSELECTED
                };
                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(format!("  {bullet}"), style),
                        Span::styled("Custom answer…", style),
                    ]))
                    .style(Theme::popup_bg()),
                    Rect::new(inner.x, row, inner.width, 1),
                );
                row += 1;
            }

            if ui.custom_active && row < max_y {
                // Keep the custom editor visually separate from the choices.
                row = row.saturating_add(1).min(max_y);
                let layout = render.prepare_elicitation_custom_layout(
                    &state.custom_input,
                    ui.custom_cursor,
                    inner.width.saturating_sub(2) as usize,
                    2,
                );
                let total_rows = layout.total_rows() as u16;
                // Reserve a spacer, the help row, and bottom padding below the editor.
                let visible = total_rows.min(max_y.saturating_sub(row + 3)).min(5);
                let scroll = render.ensure_elicitation_custom_cursor_visible(visible);

                let lines: Vec<Line<'static>> = layout
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(idx, visual_row)| {
                        if idx == 0 {
                            Line::from(vec![
                                Span::styled("> ", Theme::status_accent()),
                                Span::styled(visual_row.text.clone(), Theme::fg()),
                            ])
                        } else {
                            Line::from(Span::styled(visual_row.text.clone(), Theme::fg()))
                        }
                    })
                    .collect();
                f.render_widget(
                    Paragraph::new(lines)
                        .style(Theme::popup_bg())
                        .scroll((scroll, 0)),
                    Rect::new(inner.x + 2, row, inner.width.saturating_sub(2), visible),
                );
                let cursor_row = (layout.cursor_row as u16).saturating_sub(scroll);
                if cursor_row < visible {
                    f.set_cursor_position((
                        inner.x + 2 + layout.cursor_col as u16,
                        row + cursor_row,
                    ));
                }
                row += visible;
            }
        }
        ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
            let placeholder = if matches!(&field.kind, ElicitationFieldKind::NumberInput { .. }) {
                "enter number..."
            } else {
                "enter text..."
            };
            let display = if state.text_input.is_empty() {
                Span::styled(placeholder, Theme::dim())
            } else {
                Span::styled(state.text_input.clone(), Theme::fg())
            };
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("  > ", Theme::status_accent()),
                    display,
                ]))
                .style(Theme::popup_bg()),
                Rect::new(inner.x, row, inner.width, 1),
            );
            row += 1;
        }
        ElicitationFieldKind::BooleanToggle => {
            let value = state
                .selected
                .get(&field.name)
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        if value {
                            format!("  {CHECK_CHECKED}Yes")
                        } else {
                            format!("  {CHECK_UNCHECKED}No")
                        },
                        Theme::status_accent(),
                    ),
                    Span::styled("  (Space to toggle)", Theme::dim()),
                ]))
                .style(Theme::popup_bg()),
                Rect::new(inner.x, row, inner.width, 1),
            );
            row += 1;
        }
    }

    // Keep controls visually separate from the answer area.
    row = row.saturating_add(1).min(max_y);
    if row < max_y {
        let hint = if ui.custom_active {
            " type answer  Shift+Enter newline  Enter submit  Esc back".to_string()
        } else {
            match field.kind {
                ElicitationFieldKind::MultiSelect { .. } => format!(
                    " {0}{1} navigate  Space toggle  Enter submit  Esc decline",
                    super::ARROW_UP,
                    super::ARROW_DOWN
                ),
                ElicitationFieldKind::TextInput | ElicitationFieldKind::NumberInput { .. } => {
                    " type answer  Enter submit  Esc decline".to_string()
                }
                ElicitationFieldKind::BooleanToggle => {
                    " Space toggle  Enter submit  Esc decline".to_string()
                }
                ElicitationFieldKind::SingleSelect { .. } => format!(
                    " {0}{1} navigate  Enter select  Esc decline",
                    super::ARROW_UP,
                    super::ARROW_DOWN
                ),
            }
        };
        f.render_widget(
            Paragraph::new(Span::styled(hint, Theme::dim())).style(Theme::popup_bg()),
            Rect::new(inner.x, row, inner.width, 1),
        );
    }
}

// ── Slash command completion panel ────────────────────────────────────────────

fn draw_slash_panel(f: &mut Frame, app: &App, area: Rect) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let Some(state) = &app.composer.slash_state else {
        return;
    };

    let max_name_len = state
        .results
        .iter()
        .map(|cmd| cmd.name.len())
        .max()
        .unwrap_or(0);

    let items: Vec<ListItem> = state
        .results
        .iter()
        .map(|cmd| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("  /{:<width$}  ", cmd.name, width = max_name_len),
                    Theme::status_accent(),
                ),
                Span::styled(cmd.description, Theme::dim()),
            ]))
        })
        .collect();

    let title = Line::from(vec![
        Span::styled(" /", Theme::status_accent()),
        Span::styled(" commands ", Theme::dim()),
    ]);
    let list = List::new(items)
        .block(Block::default().title(title).style(Theme::popup_bg()))
        .highlight_style(Theme::selected())
        .highlight_symbol("");
    let mut list_state = ListState::default().with_selected(Some(state.selected_index));
    f.render_stateful_widget(list, area, &mut list_state);
}

// ── Mention panel ─────────────────────────────────────────────────────────────

fn draw_mention_panel(f: &mut Frame, app: &App, area: Rect) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let mut items: Vec<ListItem> = Vec::new();
    if app.composer.file_index_loading && app.composer.file_index.is_empty() {
        items.push(ListItem::new(Line::from(vec![Span::styled(
            format!(
                "{} indexing files",
                spinner(SpinnerKind::Braille, app.render.tick)
            ),
            Theme::thinking(),
        )])));
    } else if let Some(error) = &app.composer.file_index_error {
        items.push(ListItem::new(Line::from(vec![Span::styled(
            format!("file index error: {error}"),
            Theme::error_text(),
        )])));
    } else if let Some(mention) = &app.composer.mention_state {
        if mention.results.is_empty() {
            items.push(ListItem::new(Line::from(vec![Span::styled(
                format!("no matches for @{}", mention.query),
                Theme::info_text(),
            )])));
        } else {
            for entry in &mention.results {
                let icon = if entry.is_dir { "[D]" } else { "[F]" };
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(format!("{icon} "), Theme::status()),
                    Span::styled(entry.path.clone(), Theme::input()),
                ])));
            }
        }
    }

    if items.is_empty() {
        return;
    }

    let title = if let Some(mention) = &app.composer.mention_state {
        format!(" @ files - {} ", mention.query)
    } else {
        " @ files ".into()
    };
    let list = List::new(items)
        .block(Block::default().title(title).style(Theme::popup_bg()))
        .highlight_style(Theme::selected())
        .highlight_symbol("");
    let selected = app
        .composer
        .mention_state
        .as_ref()
        .map(|mention| mention.selected_index)
        .filter(|_| !app.composer.file_index_loading);
    let mut state = ListState::default().with_selected(selected);
    f.render_stateful_widget(list, area, &mut state);
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

// ── Diff / write helpers ──────────────────────────────────────────────────────

fn short_path(path: &str) -> &str {
    // show last 2 components
    let mut count = 0;
    for (i, c) in path.char_indices().rev() {
        if c == '/' {
            count += 1;
            if count == 2 {
                return &path[i + 1..];
            }
        }
    }
    path
}

fn strip_cwd(path: &str, cwd: Option<&str>) -> String {
    cwd.and_then(|cwd| Path::new(path).strip_prefix(cwd).ok())
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn truncate_summary(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('\u{2026}');
    out
}

fn push_summary_lines(
    pending_tools: &mut Vec<Line<'static>>,
    sym: &str,
    name: &str,
    style: Style,
    summary: &str,
    output: Option<&str>,
) {
    if let Some(output) = output {
        pending_tools.push(Line::from(vec![
            Span::styled(format!("{sym} {name} "), style),
            Span::styled(summary.to_string(), Theme::diff_file()),
        ]));
        for line in output.lines() {
            pending_tools.push(Line::from(Span::styled(
                format!("  {line}"),
                Theme::tool_output(),
            )));
        }
    } else if summary.contains('\n') {
        pending_tools.push(Line::from(Span::styled(format!("{sym} {name}"), style)));
        for line in summary.lines() {
            pending_tools.push(Line::from(Span::styled(
                format!("  {line}"),
                Theme::diff_file(),
            )));
        }
    } else if matches!(name, "index" | "search_text") && summary.ends_with(')') {
        if let Some((summary, metadata)) = summary.rsplit_once(" (") {
            pending_tools.push(Line::from(vec![
                Span::styled(format!("{sym} {name} "), style),
                Span::styled(summary.to_string(), Theme::diff_file()),
                Span::styled(format!(" ({metadata}"), Theme::status_accent()),
            ]));
        } else {
            pending_tools.push(Line::from(vec![
                Span::styled(format!("{sym} {name} "), style),
                Span::styled(summary.to_string(), Theme::diff_file()),
            ]));
        }
    } else {
        pending_tools.push(Line::from(vec![
            Span::styled(format!("{sym} {name} "), style),
            Span::styled(summary.to_string(), Theme::diff_file()),
        ]));
    }
}

fn replace_symbol_title(
    replacements: &[SymbolReplacement],
    sections: &[SymbolDiffSection],
    cwd: Option<&str>,
) -> String {
    if replacements.is_empty() && !sections.is_empty() {
        let first = &sections[0];
        return if first.symbol.is_empty() {
            first.path.clone()
        } else {
            format!("{} {}", first.path, first.symbol)
        };
    }

    let mut files = replacements
        .iter()
        .filter(|replacement| !replacement.path.is_empty())
        .map(|replacement| strip_cwd(&replacement.path, cwd))
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    match files.as_slice() {
        [] => "symbols".into(),
        [one] => short_path(one).to_string(),
        [first, ..] => format!("{} (+{})", short_path(first), files.len() - 1),
    }
}

fn build_multiedit_lines(sections: &[MultiEditSection], max_sections: usize) -> Vec<Line<'static>> {
    build_sectioned_diff_lines(
        sections
            .iter()
            .map(|section| {
                let header = if section.replace_all {
                    format!("edit {} (all)", section.edit_index)
                } else {
                    format!("edit {}", section.edit_index)
                };
                (
                    header,
                    &section.old,
                    &section.new,
                    section.start_line,
                    Theme::status_accent(),
                )
            })
            .collect::<Vec<_>>(),
        max_sections,
    )
}

fn build_symbol_diff_lines(
    sections: &[SymbolDiffSection],
    max_sections: usize,
) -> Vec<Line<'static>> {
    build_sectioned_diff_lines(
        sections
            .iter()
            .map(|section| {
                (
                    section.symbol.clone(),
                    &section.old,
                    &section.new,
                    section.start_line,
                    Theme::diff_file(),
                )
            })
            .collect::<Vec<_>>(),
        max_sections,
    )
}

fn build_sectioned_diff_lines(
    sections: Vec<(String, &String, &String, Option<usize>, Style)>,
    max_sections: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let preview_count = sections.len().min(max_sections.max(1));

    for (idx, (header, old, new, start_line, header_style)) in
        sections.iter().take(preview_count).enumerate()
    {
        if !header.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("  @@ ", Theme::diff_context()),
                Span::styled(header.clone(), *header_style),
            ]));
        }
        lines.extend(build_diff_lines(old, new, *start_line));
        if idx + 1 < preview_count {
            lines.push(Line::default());
        }
    }

    if sections.len() > preview_count {
        lines.push(Line::from(Span::styled(
            format!(
                "  ... {} more sections collapsed",
                sections.len() - preview_count
            ),
            Theme::diff_context(),
        )));
    }

    lines
}

fn shell_command_display(command: &str, arguments: &[String]) -> String {
    if arguments.is_empty() {
        return command.to_string();
    }
    std::iter::once(command)
        .chain(arguments.iter().map(String::as_str))
        .map(shell_quote_arg)
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote_arg(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_string();
    }
    if arg
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '='))
    {
        return arg.to_string();
    }
    format!("'{}'", arg.replace('\'', "'\\''"))
}

pub(crate) fn build_shell_lines(command: &str, output: Option<&ShellOutput>) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    let mut command_lines = command.lines();
    if let Some(first) = command_lines.next() {
        lines.push(Line::from(vec![
            Span::styled("  ", Theme::tool_output()),
            Span::styled("$", Theme::status_accent()),
            Span::styled(format!(" {first}"), Theme::tool_output()),
        ]));
        for line in command_lines {
            lines.push(Line::from(Span::styled(
                format!("    {line}"),
                Theme::tool_output(),
            )));
        }
    } else {
        lines.push(Line::from(vec![
            Span::styled("  ", Theme::tool_output()),
            Span::styled("$", Theme::status_accent()),
        ]));
    }

    let output_lines = output
        .into_iter()
        .flat_map(|output| output.stdout.lines().chain(output.stderr.lines()))
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if !output_lines.is_empty() {
        let keep = 5;
        let display_hidden_count = output_lines.len().saturating_sub(keep);
        let hidden_line_count = output
            .map(|output| output.preceding_line_count)
            .unwrap_or_default()
            .saturating_add(display_hidden_count);
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "  output tail:",
            Theme::diff_file(),
        )));
        for line in output_lines.into_iter().skip(display_hidden_count) {
            lines.push(Line::from(Span::styled(
                format!("    {line}"),
                Theme::tool_output(),
            )));
        }
        if hidden_line_count > 0 {
            lines.push(Line::from(Span::styled(
                format!("  ... {hidden_line_count} earlier output lines hidden"),
                Theme::diff_context(),
            )));
        }
    }

    lines
}

pub(crate) fn build_diff_lines(
    old: &str,
    new: &str,
    start_line: Option<usize>,
) -> Vec<Line<'static>> {
    use similar::{ChangeTag, TextDiff};

    let mut lines = Vec::new();
    let start = start_line.unwrap_or(1);

    let diff = TextDiff::from_lines(old, new);

    // compute max line number for gutter width
    let old_count = old.lines().count();
    let new_count = new.lines().count();
    let max_line = start + old_count.max(new_count);
    let gw = max_line.to_string().len();

    let mut old_line_idx = 0usize;
    let mut new_line_idx = 0usize;

    // group changes to pair up Delete/Insert for inline highlighting
    let changes: Vec<_> = diff.iter_all_changes().collect();
    let mut i = 0;
    while i < changes.len() {
        let change = &changes[i];
        match change.tag() {
            ChangeTag::Equal => {
                let ln = format!("{:>w$}", start + old_line_idx, w = gw);
                lines.push(Line::from(vec![
                    Span::styled(format!("  {ln} "), Theme::diff_context()),
                    Span::styled(
                        format!("  {}", change.value().trim_end_matches('\n')),
                        Theme::diff_context(),
                    ),
                ]));
                old_line_idx += 1;
                new_line_idx += 1;
                i += 1;
            }
            ChangeTag::Delete => {
                // collect consecutive deletes
                let del_start = i;
                while i < changes.len() && changes[i].tag() == ChangeTag::Delete {
                    i += 1;
                }
                let del_end = i;
                // collect consecutive inserts right after
                let ins_start = i;
                while i < changes.len() && changes[i].tag() == ChangeTag::Insert {
                    i += 1;
                }
                let ins_end = i;

                let dels: Vec<&str> = changes[del_start..del_end]
                    .iter()
                    .map(|c| c.value().trim_end_matches('\n'))
                    .collect();
                let inss: Vec<&str> = changes[ins_start..ins_end]
                    .iter()
                    .map(|c| c.value().trim_end_matches('\n'))
                    .collect();

                // render paired lines with inline char diff
                let paired = dels.len().min(inss.len());
                for j in 0..paired {
                    let (del_spans, ins_spans) = inline_diff(dels[j], inss[j]);

                    let ln_old = format!("{:>w$}", start + old_line_idx, w = gw);
                    let mut del_line = vec![
                        Span::styled(format!("  {ln_old} "), Theme::diff_context()),
                        Span::styled("- ", Theme::diff_removed()),
                    ];
                    del_line.extend(del_spans);
                    lines.push(Line::from(del_line));

                    let ln_new = format!("{:>w$}", start + new_line_idx, w = gw);
                    let mut ins_line = vec![
                        Span::styled(format!("  {ln_new} "), Theme::diff_context()),
                        Span::styled("+ ", Theme::diff_added()),
                    ];
                    ins_line.extend(ins_spans);
                    lines.push(Line::from(ins_line));

                    old_line_idx += 1;
                    new_line_idx += 1;
                }

                // remaining unpaired deletes
                for del in dels.iter().skip(paired) {
                    let ln = format!("{:>w$}", start + old_line_idx, w = gw);
                    lines.push(Line::from(vec![
                        Span::styled(format!("  {ln} "), Theme::diff_context()),
                        Span::styled(format!("- {del}"), Theme::diff_removed()),
                    ]));
                    old_line_idx += 1;
                }
                // remaining unpaired inserts
                for ins in inss.iter().skip(paired) {
                    let ln = format!("{:>w$}", start + new_line_idx, w = gw);
                    lines.push(Line::from(vec![
                        Span::styled(format!("  {ln} "), Theme::diff_context()),
                        Span::styled(format!("+ {ins}"), Theme::diff_added()),
                    ]));
                    new_line_idx += 1;
                }
            }
            ChangeTag::Insert => {
                // standalone insert (not preceded by delete)
                let ln = format!("{:>w$}", start + new_line_idx, w = gw);
                lines.push(Line::from(vec![
                    Span::styled(format!("  {ln} "), Theme::diff_context()),
                    Span::styled(
                        format!("+ {}", change.value().trim_end_matches('\n')),
                        Theme::diff_added(),
                    ),
                ]));
                new_line_idx += 1;
                i += 1;
            }
        }
    }

    lines
}

/// Char-level diff between two lines. Returns (old_spans, new_spans) with
/// changed ranges highlighted.
fn inline_diff(old: &str, new: &str) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    use similar::{ChangeTag, TextDiff};

    let diff = TextDiff::from_words(old, new);

    // build two flat lists: (text, is_highlighted) for old and new
    let mut old_parts: Vec<(String, bool)> = Vec::new();
    let mut new_parts: Vec<(String, bool)> = Vec::new();

    fn push_part(parts: &mut Vec<(String, bool)>, text: &str, highlighted: bool) {
        if let Some(last) = parts.last_mut()
            && last.1 == highlighted
        {
            last.0.push_str(text);
            return;
        }
        parts.push((text.to_string(), highlighted));
    }

    for change in diff.iter_all_changes() {
        let val = change.value();
        match change.tag() {
            ChangeTag::Equal => {
                push_part(&mut old_parts, val, false);
                push_part(&mut new_parts, val, false);
            }
            ChangeTag::Delete => {
                push_part(&mut old_parts, val, true);
            }
            ChangeTag::Insert => {
                push_part(&mut new_parts, val, true);
            }
        }
    }

    let old_spans = old_parts
        .into_iter()
        .map(|(text, hl)| {
            if hl {
                Span::styled(text, Theme::diff_removed_hl())
            } else {
                Span::styled(text, Theme::diff_removed())
            }
        })
        .collect();

    let new_spans = new_parts
        .into_iter()
        .map(|(text, hl)| {
            if hl {
                Span::styled(text, Theme::diff_added_hl())
            } else {
                Span::styled(text, Theme::diff_added())
            }
        })
        .collect();

    (old_spans, new_spans)
}

pub(crate) fn build_write_lines(content: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let total = content.lines().count();
    let gw = total.to_string().len();
    let max_preview = 20;
    for (i, l) in content.lines().enumerate() {
        if i >= max_preview {
            lines.push(Line::from(Span::styled(
                format!(
                    "  {:>w$}   ... ({} more lines)",
                    "",
                    total - max_preview,
                    w = gw
                ),
                Theme::diff_context(),
            )));
            break;
        }
        let ln = format!("{:>w$}", i + 1, w = gw);
        lines.push(Line::from(vec![
            Span::styled(format!("  {ln} "), Theme::diff_context()),
            Span::styled(format!("+ {l}"), Theme::diff_added()),
        ]));
    }
    lines
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
