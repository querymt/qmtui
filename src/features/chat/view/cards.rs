use ratatui::text::{Line, Span};

use crate::domain::activity::{DelegateEntry, DelegateStatus};
use crate::domain::chat::{ChatEntry, ElicitationResponseOutcome};
use crate::highlight::Highlighter;
use crate::markdown::{self, CardBlock};
use crate::render_state::{
    Card, CardKind, DelegatePresentationKeyRef, FinalizedCardKeyRef, FinalizedMessageKeyRef,
    FinalizedRenderContextKey, RenderState, SessionIdentity, ThemeCacheKey,
};
use crate::theme::Theme;

use super::tools::{
    DelegateToolPresentation, DelegateToolStatus, ToolRenderInput, render_tool_lines,
};

const OUTCOME_BULLET: &str = "\u{25B8} ";

/// Narrow immutable inputs for incrementally building finalized chat cards.
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
    highlighter: &Highlighter,
    start_idx: usize,
    prior_kind: Option<CardKind>,
) -> Vec<Card> {
    let mut cards = Vec::new();
    let mut pending_tools: Vec<Line<'static>> = Vec::new();
    let current_cwd = input.effective_cwd.as_deref();

    let flush_tools = |tools: &mut Vec<Line<'static>>, cards: &mut Vec<Card>| {
        if !tools.is_empty() {
            let lines = std::mem::take(tools);
            let blocks = lines.into_iter().map(CardBlock::Text).collect();
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
                    blocks.push(CardBlock::Text(Line::default()));
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
                let delegate = if name == "delegate" {
                    let status = match delegate_entry {
                        Some(entry) if entry.awaiting_input() => DelegateToolStatus::AwaitingInput,
                        Some(entry) => match entry.status {
                            DelegateStatus::InProgress => DelegateToolStatus::Running,
                            DelegateStatus::Completed => DelegateToolStatus::Done,
                            DelegateStatus::Failed => DelegateToolStatus::Failed,
                            DelegateStatus::Cancelled => DelegateToolStatus::Cancelled,
                        },
                        None => DelegateToolStatus::Queued,
                    };
                    Some(DelegateToolPresentation {
                        duration: &delegate_duration,
                        status,
                    })
                } else {
                    None
                };
                pending_tools.extend(render_tool_lines(ToolRenderInput {
                    name,
                    is_error: *is_error,
                    detail,
                    effective_cwd: current_cwd,
                    delegate,
                }));
            }
            ChatEntry::CompactionStart { token_estimate } => {
                flush_tools(&mut pending_tools, &mut cards);
                let token_str = format!("~{} tokens", token_estimate);
                cards.push(Card::new(
                    CardKind::Compaction,
                    vec![
                        CardBlock::Text(Line::from(vec![
                            Span::styled("[compact] ", Theme::status_accent()),
                            Span::styled(
                                "Summarizing conversation history",
                                Theme::status_accent(),
                            ),
                        ])),
                        CardBlock::Text(Line::from(Span::styled(
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
                let mut blocks = vec![CardBlock::Text(Line::from(vec![
                    Span::styled("[compact] ", Theme::status_accent()),
                    Span::styled("Conversation summarized", Theme::status_accent()),
                ]))];
                if let Some(token_estimate) = token_estimate {
                    blocks.push(CardBlock::Text(Line::from(Span::styled(
                        format!("  ~{} tokens -> {} chars", token_estimate, summary_len),
                        Theme::status(),
                    ))));
                } else {
                    blocks.push(CardBlock::Text(Line::from(Span::styled(
                        format!("  {} chars", summary_len),
                        Theme::status(),
                    ))));
                }
                blocks.push(CardBlock::Text(Line::default()));
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
                    vec![CardBlock::Text(Line::from(text.clone()))],
                ));
            }
            ChatEntry::Error(text) => {
                flush_tools(&mut pending_tools, &mut cards);
                cards.push(Card::new(
                    CardKind::Error,
                    vec![CardBlock::Text(Line::from(text.clone()))],
                ));
            }
            ChatEntry::Elicitation {
                message,
                source: _,
                outcome,
                ..
            } => {
                flush_tools(&mut pending_tools, &mut cards);
                let header = CardBlock::Text(Line::from(vec![
                    Span::styled("[?] ", Theme::status_accent()),
                    Span::styled(message.clone(), Theme::status_accent()),
                ]));
                let mut card_blocks = vec![header];
                match outcome {
                    None => {
                        card_blocks.push(CardBlock::Text(Line::from(Span::styled(
                            "  waiting for response\u{2026}",
                            Theme::thinking(),
                        ))));
                    }
                    Some(outcome) => {
                        for (text, style) in elicitation_outcome_lines(outcome) {
                            card_blocks.push(CardBlock::Text(Line::from(Span::styled(
                                format!("  {text}"),
                                style,
                            ))));
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

fn elicitation_outcome_lines(
    outcome: &ElicitationResponseOutcome,
) -> Vec<(String, ratatui::style::Style)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representative_finalized_card_build_uses_narrow_inputs() {
        let messages = [ChatEntry::User {
            text: "hello".into(),
            message_id: Some("user-1".into()),
        }];
        let mut render = RenderState::new();
        let cards = build_finalized_cards(
            FinalizedRenderInput {
                session_identity: SessionIdentity::new(Some("session-1".into()), None, false),
                messages: &messages,
                delegates: &[],
                effective_cwd: None,
                show_thinking: true,
                full_width: 80,
                theme: ThemeCacheKey::new(0, 0),
                now_unix_secs: 0,
            },
            &mut render,
        );

        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].kind, CardKind::User);
    }
}
