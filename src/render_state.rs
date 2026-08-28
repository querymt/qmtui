use std::cell::{Cell, Ref, RefCell};

use ratatui::{
    Frame,
    layout::Rect,
    text::{Line, Span},
    widgets::{Block, Paragraph, Wrap},
};

use crate::domain::activity::{DelegateEntry, DelegateStatus};
use crate::domain::chat::{ChatEntry, ElicitationResponseOutcome};
use crate::domain::tool::ToolDetail;
use crate::highlight::Highlighter;
use crate::markdown::CardBlock;
use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionKind {
    Local,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionIdentity {
    session_id: Option<String>,
    remote_node_id: Option<String>,
    kind: SessionKind,
}

impl SessionIdentity {
    pub(crate) fn new(
        session_id: Option<String>,
        remote_node_id: Option<String>,
        is_remote: bool,
    ) -> Self {
        Self {
            session_id,
            remote_node_id,
            kind: if is_remote {
                SessionKind::Remote
            } else {
                SessionKind::Local
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RenderChange {
    SessionChanged(SessionIdentity),
    FinalizedMessagesChanged,
    StreamingContentChanged,
    StreamingThinkingChanged,
    DelegatePresentationChanged,
    ThemeChanged,
    ExternalEditorReturned,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionCacheKey {
    epoch: u64,
    identity: SessionIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ThemeCacheKey {
    index: usize,
    revision: u64,
}

impl ThemeCacheKey {
    pub(crate) fn new(index: usize, revision: u64) -> Self {
        Self { index, revision }
    }

    pub(crate) fn current_frame() -> Self {
        Self::new(Theme::frame_index(), Theme::render_revision())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FinalizedRenderContextKey {
    session: SessionCacheKey,
    full_width: u16,
    theme: ThemeCacheKey,
    show_thinking: bool,
    effective_cwd: Option<String>,
}

impl FinalizedRenderContextKey {
    pub(crate) fn new(
        session: SessionCacheKey,
        full_width: u16,
        theme: ThemeCacheKey,
        show_thinking: bool,
        effective_cwd: Option<String>,
    ) -> Self {
        Self {
            session,
            full_width,
            theme,
            show_thinking,
            effective_cwd,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DelegatePresentationKey {
    Queued,
    Matched {
        delegation_id: String,
        matched_tool_id: Option<String>,
        status: DelegateStatus,
        awaiting_input: bool,
        started_at: Option<i64>,
        ended_at: Option<i64>,
        duration_secs: Option<u64>,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DelegatePresentationKeyRef<'a> {
    Queued,
    Matched {
        delegation_id: &'a str,
        matched_tool_id: Option<&'a str>,
        status: DelegateStatus,
        awaiting_input: bool,
        started_at: Option<i64>,
        ended_at: Option<i64>,
        duration_secs: Option<u64>,
    },
}

impl<'a> DelegatePresentationKeyRef<'a> {
    pub(crate) fn from_entry(entry: Option<&'a DelegateEntry>, now_unix_secs: i64) -> Self {
        let Some(entry) = entry else {
            return Self::Queued;
        };
        let duration_secs = entry.started_at.map(|start| {
            let end = entry.ended_at.unwrap_or(now_unix_secs);
            (end - start).max(0) as u64
        });
        Self::Matched {
            delegation_id: &entry.delegation_id,
            matched_tool_id: entry.delegate_tool_call_id.as_deref(),
            status: entry.status,
            awaiting_input: entry.awaiting_input(),
            started_at: entry.started_at,
            ended_at: entry.ended_at,
            duration_secs,
        }
    }

    fn to_owned(self) -> DelegatePresentationKey {
        match self {
            Self::Queued => DelegatePresentationKey::Queued,
            Self::Matched {
                delegation_id,
                matched_tool_id,
                status,
                awaiting_input,
                started_at,
                ended_at,
                duration_secs,
            } => DelegatePresentationKey::Matched {
                delegation_id: delegation_id.to_string(),
                matched_tool_id: matched_tool_id.map(str::to_string),
                status,
                awaiting_input,
                started_at,
                ended_at,
                duration_secs,
            },
        }
    }
}

impl DelegatePresentationKey {
    fn matches_ref(&self, other: DelegatePresentationKeyRef<'_>) -> bool {
        match (self, other) {
            (Self::Queued, DelegatePresentationKeyRef::Queued) => true,
            (
                Self::Matched {
                    delegation_id,
                    matched_tool_id,
                    status,
                    awaiting_input,
                    started_at,
                    ended_at,
                    duration_secs,
                },
                DelegatePresentationKeyRef::Matched {
                    delegation_id: other_delegation_id,
                    matched_tool_id: other_tool_id,
                    status: other_status,
                    awaiting_input: other_awaiting_input,
                    started_at: other_started_at,
                    ended_at: other_ended_at,
                    duration_secs: other_duration_secs,
                },
            ) => {
                delegation_id == other_delegation_id
                    && matched_tool_id.as_deref() == other_tool_id
                    && *status == other_status
                    && *awaiting_input == other_awaiting_input
                    && *started_at == other_started_at
                    && *ended_at == other_ended_at
                    && *duration_secs == other_duration_secs
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FinalizedMessageKey {
    User {
        ordinal: usize,
        message_id: Option<String>,
        text: String,
    },
    Assistant {
        ordinal: usize,
        message_id: Option<String>,
        content: String,
        thinking: Option<String>,
    },
    Thinking {
        ordinal: usize,
        message_id: Option<String>,
        content: String,
    },
    ToolCall {
        ordinal: usize,
        tool_call_id: Option<String>,
        name: String,
        is_error: bool,
        detail: Box<ToolDetail>,
        delegate: Option<DelegatePresentationKey>,
    },
    CompactionStart {
        ordinal: usize,
        token_estimate: u32,
    },
    CompactionEnd {
        ordinal: usize,
        token_estimate: Option<u32>,
        summary: String,
        summary_len: u32,
    },
    Info {
        ordinal: usize,
        text: String,
    },
    Error {
        ordinal: usize,
        text: String,
    },
    Elicitation {
        ordinal: usize,
        elicitation_id: String,
        message: String,
        outcome: Option<ElicitationResponseOutcome>,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FinalizedMessageKeyRef<'a> {
    ordinal: usize,
    entry: &'a ChatEntry,
    delegate: Option<DelegatePresentationKeyRef<'a>>,
}

impl<'a> FinalizedMessageKeyRef<'a> {
    pub(crate) fn new(
        ordinal: usize,
        entry: &'a ChatEntry,
        delegate: Option<DelegatePresentationKeyRef<'a>>,
    ) -> Self {
        Self {
            ordinal,
            entry,
            delegate,
        }
    }

    fn to_owned(self) -> FinalizedMessageKey {
        match self.entry {
            ChatEntry::User { text, message_id } => FinalizedMessageKey::User {
                ordinal: self.ordinal,
                message_id: message_id.clone(),
                text: text.clone(),
            },
            ChatEntry::Assistant {
                content,
                thinking,
                message_id,
            } => FinalizedMessageKey::Assistant {
                ordinal: self.ordinal,
                message_id: message_id.clone(),
                content: content.clone(),
                thinking: thinking.clone(),
            },
            ChatEntry::Thinking {
                content,
                message_id,
            } => FinalizedMessageKey::Thinking {
                ordinal: self.ordinal,
                message_id: message_id.clone(),
                content: content.clone(),
            },
            ChatEntry::ToolCall {
                tool_call_id,
                name,
                is_error,
                detail,
            } => FinalizedMessageKey::ToolCall {
                ordinal: self.ordinal,
                tool_call_id: tool_call_id.clone(),
                name: name.clone(),
                is_error: *is_error,
                detail: Box::new(detail.clone()),
                delegate: self.delegate.map(DelegatePresentationKeyRef::to_owned),
            },
            ChatEntry::CompactionStart { token_estimate } => FinalizedMessageKey::CompactionStart {
                ordinal: self.ordinal,
                token_estimate: *token_estimate,
            },
            ChatEntry::CompactionEnd {
                token_estimate,
                summary,
                summary_len,
            } => FinalizedMessageKey::CompactionEnd {
                ordinal: self.ordinal,
                token_estimate: *token_estimate,
                summary: summary.clone(),
                summary_len: *summary_len,
            },
            ChatEntry::Info(text) => FinalizedMessageKey::Info {
                ordinal: self.ordinal,
                text: text.clone(),
            },
            ChatEntry::Error(text) => FinalizedMessageKey::Error {
                ordinal: self.ordinal,
                text: text.clone(),
            },
            ChatEntry::Elicitation {
                elicitation_id,
                message,
                outcome,
                ..
            } => FinalizedMessageKey::Elicitation {
                ordinal: self.ordinal,
                elicitation_id: elicitation_id.clone(),
                message: message.clone(),
                outcome: outcome.clone(),
            },
        }
    }
}

impl FinalizedMessageKey {
    fn matches_ref(&self, other: FinalizedMessageKeyRef<'_>) -> bool {
        match (self, other.entry) {
            (
                Self::User {
                    ordinal,
                    message_id,
                    text,
                },
                ChatEntry::User {
                    text: other_text,
                    message_id: other_id,
                },
            ) => *ordinal == other.ordinal && message_id == other_id && text == other_text,
            (
                Self::Assistant {
                    ordinal,
                    message_id,
                    content,
                    thinking,
                },
                ChatEntry::Assistant {
                    content: other_content,
                    thinking: other_thinking,
                    message_id: other_id,
                },
            ) => {
                *ordinal == other.ordinal
                    && message_id == other_id
                    && content == other_content
                    && thinking == other_thinking
            }
            (
                Self::Thinking {
                    ordinal,
                    message_id,
                    content,
                },
                ChatEntry::Thinking {
                    content: other_content,
                    message_id: other_id,
                },
            ) => *ordinal == other.ordinal && message_id == other_id && content == other_content,
            (
                Self::ToolCall {
                    ordinal,
                    tool_call_id,
                    name,
                    is_error,
                    detail,
                    delegate,
                },
                ChatEntry::ToolCall {
                    tool_call_id: other_id,
                    name: other_name,
                    is_error: other_is_error,
                    detail: other_detail,
                },
            ) => {
                *ordinal == other.ordinal
                    && tool_call_id == other_id
                    && name == other_name
                    && *is_error == *other_is_error
                    && detail.as_ref() == other_detail
                    && match (delegate, other.delegate) {
                        (Some(stored), Some(lookup)) => stored.matches_ref(lookup),
                        (None, None) => true,
                        _ => false,
                    }
            }
            (
                Self::CompactionStart {
                    ordinal,
                    token_estimate,
                },
                ChatEntry::CompactionStart {
                    token_estimate: other_estimate,
                },
            ) => *ordinal == other.ordinal && token_estimate == other_estimate,
            (
                Self::CompactionEnd {
                    ordinal,
                    token_estimate,
                    summary,
                    summary_len,
                },
                ChatEntry::CompactionEnd {
                    token_estimate: other_estimate,
                    summary: other_summary,
                    summary_len: other_len,
                },
            ) => {
                *ordinal == other.ordinal
                    && token_estimate == other_estimate
                    && summary == other_summary
                    && summary_len == other_len
            }
            (Self::Info { ordinal, text }, ChatEntry::Info(other_text)) => {
                *ordinal == other.ordinal && text == other_text
            }
            (Self::Error { ordinal, text }, ChatEntry::Error(other_text)) => {
                *ordinal == other.ordinal && text == other_text
            }
            (
                Self::Elicitation {
                    ordinal,
                    elicitation_id,
                    message,
                    outcome,
                },
                ChatEntry::Elicitation {
                    elicitation_id: other_id,
                    message: other_message,
                    outcome: other_outcome,
                    ..
                },
            ) => {
                *ordinal == other.ordinal
                    && elicitation_id == other_id
                    && message == other_message
                    && outcome == other_outcome
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalizedCardKey {
    kind: CardKind,
    source_keys: Vec<FinalizedMessageKey>,
}

#[derive(Debug, Clone)]
pub(crate) struct FinalizedCardKeyRef<'a> {
    kind: CardKind,
    source_keys: Vec<FinalizedMessageKeyRef<'a>>,
}

impl<'a> FinalizedCardKeyRef<'a> {
    pub(crate) fn new(kind: CardKind, source_keys: Vec<FinalizedMessageKeyRef<'a>>) -> Self {
        Self { kind, source_keys }
    }

    pub(crate) fn kind(&self) -> &CardKind {
        &self.kind
    }

    fn to_owned(&self) -> FinalizedCardKey {
        FinalizedCardKey {
            kind: self.kind.clone(),
            source_keys: self
                .source_keys
                .iter()
                .copied()
                .map(FinalizedMessageKeyRef::to_owned)
                .collect(),
        }
    }
}

impl FinalizedCardKey {
    fn matches_ref(&self, other: &FinalizedCardKeyRef<'_>) -> bool {
        self.kind == other.kind
            && self.source_keys.len() == other.source_keys.len()
            && self
                .source_keys
                .iter()
                .zip(&other.source_keys)
                .all(|(stored, lookup)| stored.matches_ref(*lookup))
    }
}

// Message cards use background color only, without borders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CardKind {
    User,
    Assistant,
    Tool { compact: bool },
    Streaming,
    Thinking,
    Compaction,
    Error,
    Info,
    Elicitation,
}

pub(crate) struct Card {
    #[cfg(test)]
    identity: u64,
    pub(crate) kind: CardKind,
    blocks: Vec<CardBlock>,
    top_pad: u16,
    bottom_pad: u16,
    cached_width: Cell<Option<u16>>,
    cached_lines: RefCell<Vec<Line<'static>>>,
}

impl Card {
    pub(crate) fn new(kind: CardKind, blocks: Vec<CardBlock>) -> Self {
        let (top_pad, bottom_pad): (u16, u16) = match kind {
            CardKind::Tool { compact: true } => (0, 0),
            CardKind::Tool { compact: false } => (1, 0),
            _ => (1, 1),
        };
        Self {
            #[cfg(test)]
            identity: NEXT_CARD_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            kind,
            blocks,
            top_pad,
            bottom_pad,
            cached_width: Cell::new(None),
            cached_lines: RefCell::new(Vec::new()),
        }
    }

    /// Rebuild cached lines if the width has changed, then return a borrow.
    pub(crate) fn lines_for(&self, inner_w: u16) -> Ref<'_, Vec<Line<'static>>> {
        if self.cached_width.get() != Some(inner_w) {
            let mut lines = Vec::new();
            for block in &self.blocks {
                match block {
                    CardBlock::Text(line) => lines.push(line.clone()),
                    CardBlock::Table(table) => lines.extend(table.layout(inner_w as usize)),
                }
            }
            *self.cached_lines.borrow_mut() = lines;
            self.cached_width.set(Some(inner_w));
        }
        self.cached_lines.borrow()
    }

    /// Compute visual height using the full card render width.
    pub(crate) fn height(&self, width: u16) -> u16 {
        let inner_w = width.saturating_sub(4);
        let lines = self.lines_for(inner_w);
        let line_rows: u16 = lines
            .iter()
            .map(|line| {
                let width = line.width();
                let inner_width = inner_w as usize;
                if inner_width == 0 || width == 0 {
                    1
                } else {
                    width.div_ceil(inner_width) as u16
                }
            })
            .sum::<u16>()
            .max(1);
        self.top_pad + line_rows + self.bottom_pad
    }

    pub(crate) fn render(&self, frame: &mut Frame, area: Rect, clip_top: u16) {
        if area.height == 0 || area.width == 0 {
            return;
        }

        let (background_style, text_style) = match self.kind {
            CardKind::User => (Theme::user_card(), Theme::user_text()),
            CardKind::Assistant | CardKind::Streaming => {
                (Theme::assistant_card(), Theme::assistant_text())
            }
            CardKind::Tool { .. } => (Theme::assistant_card(), Theme::tool_text()),
            CardKind::Thinking => (Theme::assistant_card(), Theme::thinking()),
            CardKind::Compaction => (Theme::assistant_card(), Theme::status_accent()),
            CardKind::Error => (Theme::assistant_card(), Theme::error_text()),
            CardKind::Info => (Theme::assistant_card(), Theme::info_text()),
            CardKind::Elicitation => (Theme::assistant_card(), Theme::status_accent()),
        };

        frame.render_widget(Block::default().style(background_style), area);

        let has_top_pad = !matches!(self.kind, CardKind::Tool { compact: true });
        let visible_top_pad = u16::from(clip_top == 0 && has_top_pad);
        let content_skip = clip_top.saturating_sub(1);
        let content_height = area.height.saturating_sub(visible_top_pad);
        if content_height == 0 {
            return;
        }

        let content_area = Rect {
            x: area.x + 2,
            y: area.y + visible_top_pad,
            width: area.width.saturating_sub(4),
            height: content_height,
        };
        let lines = self.lines_for(area.width.saturating_sub(4));
        let styled_lines: Vec<Line<'static>> = lines
            .iter()
            .map(|line| {
                Line::from(
                    line.spans
                        .iter()
                        .map(|span| {
                            let style = if span.style.fg.is_some() {
                                span.style.bg(background_style.bg.unwrap_or(Theme::bg()))
                            } else {
                                text_style
                            };
                            Span::styled(span.content.clone(), style)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        frame.render_widget(
            Paragraph::new(styled_lines)
                .wrap(Wrap { trim: false })
                .scroll((content_skip, 0)),
            content_area,
        );
    }
}

#[cfg(test)]
static NEXT_CARD_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

struct CardCache {
    context: Option<FinalizedRenderContextKey>,
    keys: Vec<FinalizedCardKey>,
    cards: Vec<Card>,
    #[cfg(test)]
    source_entry_count: usize,
}

impl CardCache {
    fn new() -> Self {
        Self {
            context: None,
            keys: Vec::new(),
            cards: Vec::new(),
            #[cfg(test)]
            source_entry_count: 0,
        }
    }

    fn invalidate(&mut self) {
        self.context = None;
        self.keys.clear();
        self.cards.clear();
        #[cfg(test)]
        {
            self.source_entry_count = 0;
        }
    }

    fn first_mismatch(
        &self,
        context: &FinalizedRenderContextKey,
        desired: &[FinalizedCardKeyRef<'_>],
    ) -> usize {
        if self.context.as_ref() != Some(context) {
            return 0;
        }
        let shared_len = self.keys.len().min(desired.len());
        self.keys
            .iter()
            .zip(desired)
            .position(|(stored, lookup)| !stored.matches_ref(lookup))
            .unwrap_or(shared_len)
    }

    fn truncate(&mut self, index: usize) {
        self.keys.truncate(index);
        self.cards.truncate(index);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamKind {
    Content,
    Thinking,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StreamingCacheKey {
    session: SessionCacheKey,
    kind: StreamKind,
    message_id: Option<String>,
    paired_message_id: Option<String>,
    fallback_ordinal: usize,
    content: String,
    full_width: u16,
    theme: ThemeCacheKey,
    show_thinking: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamingCacheKeyRef<'a> {
    session: &'a SessionCacheKey,
    kind: StreamKind,
    message_id: Option<&'a str>,
    paired_message_id: Option<&'a str>,
    fallback_ordinal: usize,
    content: &'a str,
    full_width: u16,
    theme: ThemeCacheKey,
    show_thinking: bool,
}

impl<'a> StreamingCacheKeyRef<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session: &'a SessionCacheKey,
        kind: StreamKind,
        message_id: Option<&'a str>,
        paired_message_id: Option<&'a str>,
        fallback_ordinal: usize,
        content: &'a str,
        full_width: u16,
        theme: ThemeCacheKey,
        show_thinking: bool,
    ) -> Self {
        Self {
            session,
            kind,
            message_id,
            paired_message_id,
            fallback_ordinal,
            content,
            full_width,
            theme,
            show_thinking,
        }
    }

    fn to_owned(self) -> StreamingCacheKey {
        StreamingCacheKey {
            session: self.session.clone(),
            kind: self.kind,
            message_id: self.message_id.map(str::to_string),
            paired_message_id: self.paired_message_id.map(str::to_string),
            fallback_ordinal: self.fallback_ordinal,
            content: self.content.to_string(),
            full_width: self.full_width,
            theme: self.theme,
            show_thinking: self.show_thinking,
        }
    }
}

impl StreamingCacheKey {
    fn matches_ref(&self, other: StreamingCacheKeyRef<'_>) -> bool {
        self.session == *other.session
            && self.kind == other.kind
            && self.message_id.as_deref() == other.message_id
            && self.paired_message_id.as_deref() == other.paired_message_id
            && self.fallback_ordinal == other.fallback_ordinal
            && self.content == other.content
            && self.full_width == other.full_width
            && self.theme == other.theme
            && self.show_thinking == other.show_thinking
    }
}

struct StreamingCacheEntry {
    #[cfg(test)]
    identity: u64,
    key: StreamingCacheKey,
    blocks: Vec<CardBlock>,
}

#[cfg(test)]
static NEXT_STREAMING_CACHE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

struct StreamingCache {
    entry: Option<StreamingCacheEntry>,
}

impl StreamingCache {
    fn new() -> Self {
        Self { entry: None }
    }

    fn get(&self, key: StreamingCacheKeyRef<'_>) -> Option<&[CardBlock]> {
        (!key.content.is_empty())
            .then_some(())
            .and(self.entry.as_ref())
            .filter(|entry| entry.key.matches_ref(key))
            .map(|entry| entry.blocks.as_slice())
    }

    fn store(&mut self, key: StreamingCacheKeyRef<'_>, blocks: Vec<CardBlock>) {
        self.entry = (!key.content.is_empty()).then(|| StreamingCacheEntry {
            #[cfg(test)]
            identity: NEXT_STREAMING_CACHE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            key: key.to_owned(),
            blocks,
        });
    }

    fn invalidate(&mut self) {
        self.entry = None;
    }
}

#[derive(Default)]
struct ChatViewportState {
    scroll_offset: u16,
    previous_total_height: u16,
}

/// Owner for render-local cards, caches, identity, and viewport accounting.
pub(crate) struct RenderState {
    highlighter: Highlighter,
    card_cache: CardCache,
    streaming_cache: StreamingCache,
    streaming_thinking_cache: StreamingCache,
    session_identity: Option<SessionIdentity>,
    session_epoch: u64,
    chat_viewport: ChatViewportState,
    pub(crate) tick: u64,
    #[cfg(test)]
    invalidation_order: Vec<CacheInvalidation>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheInvalidation {
    Finalized,
    Content,
    Thinking,
}

impl RenderState {
    pub(crate) fn new() -> Self {
        Self {
            highlighter: Highlighter::new(),
            card_cache: CardCache::new(),
            streaming_cache: StreamingCache::new(),
            streaming_thinking_cache: StreamingCache::new(),
            session_identity: None,
            session_epoch: 0,
            chat_viewport: ChatViewportState::default(),
            tick: 0,
            #[cfg(test)]
            invalidation_order: Vec::new(),
        }
    }

    pub(crate) fn highlighter(&self) -> &Highlighter {
        &self.highlighter
    }

    pub(crate) fn cards(&self) -> &[Card] {
        &self.card_cache.cards
    }

    pub(crate) fn observe_session(&mut self, identity: &SessionIdentity) -> SessionCacheKey {
        if self.session_identity.as_ref() != Some(identity) {
            self.advance_session_epoch(identity.clone());
        }
        self.session_cache_key(identity)
    }

    pub(crate) fn apply_change(&mut self, change: RenderChange) {
        match change {
            RenderChange::SessionChanged(identity) => {
                self.advance_session_epoch(identity);
                self.reset_chat_viewport();
                self.invalidate_content_cache();
                self.invalidate_thinking_cache();
                self.invalidate_card_cache();
            }
            RenderChange::FinalizedMessagesChanged | RenderChange::DelegatePresentationChanged => {
                self.invalidate_card_cache()
            }
            RenderChange::StreamingContentChanged => self.invalidate_content_cache(),
            RenderChange::StreamingThinkingChanged => self.invalidate_thinking_cache(),
            RenderChange::ThemeChanged => self.invalidate_theme_caches(),
            RenderChange::ExternalEditorReturned => {
                self.invalidate_card_cache();
                self.invalidate_content_cache();
            }
        }
    }

    fn advance_session_epoch(&mut self, identity: SessionIdentity) {
        self.session_identity = Some(identity);
        self.session_epoch = self.session_epoch.saturating_add(1);
    }

    fn session_cache_key(&self, identity: &SessionIdentity) -> SessionCacheKey {
        SessionCacheKey {
            epoch: self.session_epoch,
            identity: identity.clone(),
        }
    }

    pub(crate) fn prepare_finalized_cards(
        &mut self,
        context: FinalizedRenderContextKey,
        desired: &[FinalizedCardKeyRef<'_>],
    ) -> usize {
        let mismatch = self.card_cache.first_mismatch(&context, desired);
        self.card_cache.truncate(mismatch);
        self.card_cache.context = Some(context);
        mismatch
    }

    pub(crate) fn push_finalized_card(&mut self, key: &FinalizedCardKeyRef<'_>, card: Card) {
        self.card_cache.keys.push(key.to_owned());
        self.card_cache.cards.push(card);
    }

    pub(crate) fn finish_finalized_cards(&mut self, source_entry_count: usize) {
        debug_assert_eq!(self.card_cache.keys.len(), self.card_cache.cards.len());
        #[cfg(test)]
        {
            self.card_cache.source_entry_count = source_entry_count;
        }
        #[cfg(not(test))]
        let _ = source_entry_count;
    }

    pub(crate) fn streaming_blocks(&self, key: StreamingCacheKeyRef<'_>) -> Option<&[CardBlock]> {
        match key.kind {
            StreamKind::Content => self.streaming_cache.get(key),
            StreamKind::Thinking => self.streaming_thinking_cache.get(key),
        }
    }

    pub(crate) fn store_streaming_blocks(
        &mut self,
        key: StreamingCacheKeyRef<'_>,
        blocks: Vec<CardBlock>,
    ) {
        match key.kind {
            StreamKind::Content => self.streaming_cache.store(key, blocks),
            StreamKind::Thinking => self.streaming_thinking_cache.store(key, blocks),
        }
    }

    fn invalidate_card_cache(&mut self) {
        self.card_cache.invalidate();
        #[cfg(test)]
        self.invalidation_order.push(CacheInvalidation::Finalized);
    }

    fn invalidate_content_cache(&mut self) {
        self.streaming_cache.invalidate();
        #[cfg(test)]
        self.invalidation_order.push(CacheInvalidation::Content);
    }

    fn invalidate_thinking_cache(&mut self) {
        self.streaming_thinking_cache.invalidate();
        #[cfg(test)]
        self.invalidation_order.push(CacheInvalidation::Thinking);
    }

    fn invalidate_theme_caches(&mut self) {
        self.invalidate_card_cache();
        self.invalidate_content_cache();
        self.invalidate_thinking_cache();
    }

    pub(crate) fn replace_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    pub(crate) fn chat_scroll_offset(&self) -> u16 {
        self.chat_viewport.scroll_offset
    }

    #[cfg(test)]
    pub(crate) fn set_chat_scroll_offset(&mut self, offset: u16) {
        self.chat_viewport.scroll_offset = offset;
    }

    pub(crate) fn scroll_chat_up(&mut self, rows: u16) {
        self.chat_viewport.scroll_offset = self.chat_viewport.scroll_offset.saturating_add(rows);
    }

    pub(crate) fn scroll_chat_down(&mut self, rows: u16) {
        self.chat_viewport.scroll_offset = self.chat_viewport.scroll_offset.saturating_sub(rows);
    }

    pub(crate) fn scroll_chat_to_top(&mut self) {
        self.chat_viewport.scroll_offset = u16::MAX;
    }

    pub(crate) fn scroll_chat_to_bottom(&mut self) {
        self.chat_viewport.scroll_offset = 0;
    }

    pub(crate) fn compensate_chat_growth(&mut self, total_height: u16) {
        let growth = total_height.saturating_sub(self.chat_viewport.previous_total_height);
        if self.chat_viewport.scroll_offset > 0 && growth > 0 {
            self.chat_viewport.scroll_offset =
                self.chat_viewport.scroll_offset.saturating_add(growth);
        }
        self.chat_viewport.previous_total_height = total_height;
    }

    pub(crate) fn clamp_chat_scroll(&mut self, total_height: u16, viewport_height: u16) -> u16 {
        let max_scroll = total_height.saturating_sub(viewport_height);
        self.chat_viewport.scroll_offset = self.chat_viewport.scroll_offset.min(max_scroll);
        max_scroll.saturating_sub(self.chat_viewport.scroll_offset)
    }

    pub(crate) fn reset_chat_viewport(&mut self) {
        self.chat_viewport = ChatViewportState::default();
    }

    #[cfg(test)]
    pub(crate) fn test_chat_previous_total_height(&self) -> u16 {
        self.chat_viewport.previous_total_height
    }

    #[cfg(test)]
    pub(crate) fn test_seed_chat_viewport(&mut self, offset: u16, previous_total_height: u16) {
        self.chat_viewport = ChatViewportState {
            scroll_offset: offset,
            previous_total_height,
        };
    }

    #[cfg(test)]
    pub(crate) fn test_seed_card_cache(&mut self, source_entry_count: usize) {
        self.card_cache.invalidate();
        self.card_cache.source_entry_count = source_entry_count;
        self.card_cache.cards.push(Card::new(
            CardKind::Info,
            vec![CardBlock::Text(Line::from("card sentinel"))],
        ));
    }

    #[cfg(test)]
    pub(crate) fn test_card_source_entry_count(&self) -> usize {
        self.card_cache.source_entry_count
    }

    #[cfg(test)]
    pub(crate) fn test_card_identity(&self, index: usize) -> Option<u64> {
        self.card_cache.cards.get(index).map(|card| card.identity)
    }

    #[cfg(test)]
    pub(crate) fn test_seed_streaming_cache(&mut self, kind: StreamKind) {
        let identity = SessionIdentity::new(Some("sentinel".into()), None, false);
        let session = SessionCacheKey { epoch: 1, identity };
        let key = StreamingCacheKeyRef::new(
            &session,
            kind,
            Some("message"),
            None,
            1,
            "sentinel",
            80,
            ThemeCacheKey::new(0, 0),
            true,
        );
        self.store_streaming_blocks(key, vec![CardBlock::Text(Line::from("stream sentinel"))]);
    }

    #[cfg(test)]
    pub(crate) fn test_streaming_cache_populated(&self, kind: StreamKind) -> bool {
        match kind {
            StreamKind::Content => self.streaming_cache.entry.is_some(),
            StreamKind::Thinking => self.streaming_thinking_cache.entry.is_some(),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_streaming_cache_identity(&self, kind: StreamKind) -> Option<u64> {
        match kind {
            StreamKind::Content => self.streaming_cache.entry.as_ref(),
            StreamKind::Thinking => self.streaming_thinking_cache.entry.as_ref(),
        }
        .map(|entry| entry.identity)
    }

    #[cfg(test)]
    pub(crate) fn test_session_epoch(&self) -> u64 {
        self.session_epoch
    }
}

#[cfg(test)]
mod tests {
    use ratatui::text::Line;

    use super::*;

    fn identity(id: &str, node: Option<&str>, remote: bool) -> SessionIdentity {
        SessionIdentity::new(Some(id.to_string()), node.map(str::to_string), remote)
    }

    fn stream_key<'a>(
        session: &'a SessionCacheKey,
        kind: StreamKind,
        message_id: Option<&'a str>,
        paired_message_id: Option<&'a str>,
        content: &'a str,
    ) -> StreamingCacheKeyRef<'a> {
        StreamingCacheKeyRef::new(
            session,
            kind,
            message_id,
            paired_message_id,
            4,
            content,
            80,
            ThemeCacheKey::new(2, 7),
            true,
        )
    }

    #[test]
    fn constructor_starts_with_empty_private_caches_and_zero_layout_state() {
        let state = RenderState::new();

        assert!(state.cards().is_empty());
        assert_eq!(state.test_card_source_entry_count(), 0);
        assert!(!state.test_streaming_cache_populated(StreamKind::Content));
        assert!(!state.test_streaming_cache_populated(StreamKind::Thinking));
        assert_eq!(state.test_session_epoch(), 0);
        assert_eq!(state.chat_scroll_offset(), 0);
        assert_eq!(state.test_chat_previous_total_height(), 0);
        assert_eq!(state.tick, 0);
    }

    #[test]
    fn session_observation_and_explicit_reload_use_monotonic_epochs() {
        let mut state = RenderState::new();
        let local = identity("same", None, false);
        let remote = identity("same", Some("node-1"), true);

        assert_eq!(state.observe_session(&local).epoch, 1);
        assert_eq!(state.observe_session(&local).epoch, 1);
        state.advance_session_epoch(local.clone());
        assert_eq!(state.observe_session(&local).epoch, 2);
        assert_eq!(state.observe_session(&remote).epoch, 3);
    }

    #[test]
    fn streaming_key_compares_exact_content_ids_kind_and_session() {
        let mut state = RenderState::new();
        let local = identity("session", None, false);
        let session = state.observe_session(&local);
        let content = stream_key(
            &session,
            StreamKind::Content,
            Some("a1"),
            Some("t1"),
            "same",
        );
        state.store_streaming_blocks(content, vec![CardBlock::Text(Line::from("cached"))]);

        assert!(state.streaming_blocks(content).is_some());
        assert!(
            state
                .streaming_blocks(stream_key(
                    &session,
                    StreamKind::Content,
                    Some("a1"),
                    Some("t1"),
                    "size",
                ))
                .is_none(),
            "same-length replacement must miss"
        );
        assert!(
            state
                .streaming_blocks(stream_key(
                    &session,
                    StreamKind::Content,
                    Some("a2"),
                    Some("t1"),
                    "same",
                ))
                .is_none()
        );
        assert!(
            state
                .streaming_blocks(stream_key(
                    &session,
                    StreamKind::Thinking,
                    Some("a1"),
                    Some("t1"),
                    "same",
                ))
                .is_none()
        );
        for changed in [
            StreamingCacheKeyRef {
                full_width: 79,
                ..content
            },
            StreamingCacheKeyRef {
                fallback_ordinal: 5,
                ..content
            },
            StreamingCacheKeyRef {
                theme: ThemeCacheKey::new(2, 8),
                ..content
            },
            StreamingCacheKeyRef {
                show_thinking: false,
                ..content
            },
        ] {
            assert!(state.streaming_blocks(changed).is_none());
        }

        state.advance_session_epoch(local.clone());
        let reloaded = state.observe_session(&local);
        assert!(
            state
                .streaming_blocks(stream_key(
                    &reloaded,
                    StreamKind::Content,
                    Some("a1"),
                    Some("t1"),
                    "same",
                ))
                .is_none()
        );
    }

    #[test]
    fn empty_stream_content_never_hits_or_stores() {
        let mut state = RenderState::new();
        let local = identity("session", None, false);
        let session = state.observe_session(&local);
        let empty = stream_key(&session, StreamKind::Content, None, None, "");

        state.store_streaming_blocks(empty, vec![CardBlock::Text(Line::from("unexpected"))]);
        assert!(state.streaming_blocks(empty).is_none());
        assert!(!state.test_streaming_cache_populated(StreamKind::Content));
    }

    fn seeded_caches() -> RenderState {
        let mut state = RenderState::new();
        state.test_seed_card_cache(2);
        state.test_seed_streaming_cache(StreamKind::Content);
        state.test_seed_streaming_cache(StreamKind::Thinking);
        state
    }

    fn assert_cache_scope(
        state: &RenderState,
        cards: bool,
        content: bool,
        thinking: bool,
        order: &[CacheInvalidation],
    ) {
        assert_eq!(state.test_card_source_entry_count() > 0, cards);
        assert_eq!(
            state.test_streaming_cache_populated(StreamKind::Content),
            content
        );
        assert_eq!(
            state.test_streaming_cache_populated(StreamKind::Thinking),
            thinking
        );
        assert_eq!(state.invalidation_order, order);
    }

    #[test]
    fn session_change_advances_epoch_and_clears_content_thinking_then_cards() {
        let mut state = seeded_caches();
        let session = identity("same", None, false);
        let expected_order = [
            CacheInvalidation::Content,
            CacheInvalidation::Thinking,
            CacheInvalidation::Finalized,
        ];
        state.test_seed_chat_viewport(9, 27);

        state.apply_change(RenderChange::SessionChanged(session.clone()));
        assert_eq!(state.test_session_epoch(), 1);
        assert_eq!(state.chat_scroll_offset(), 0);
        assert_eq!(state.test_chat_previous_total_height(), 0);
        assert_cache_scope(&state, false, false, false, &expected_order);

        state.test_seed_card_cache(2);
        state.test_seed_streaming_cache(StreamKind::Content);
        state.test_seed_streaming_cache(StreamKind::Thinking);
        state.test_seed_chat_viewport(u16::MAX, 12);
        state.invalidation_order.clear();
        state.apply_change(RenderChange::SessionChanged(session));
        assert_eq!(state.test_session_epoch(), 2);
        assert_eq!(state.chat_scroll_offset(), 0);
        assert_eq!(state.test_chat_previous_total_height(), 0);
        assert_cache_scope(&state, false, false, false, &expected_order);
    }

    #[test]
    fn finalized_message_change_clears_only_cards() {
        let mut state = seeded_caches();
        state.apply_change(RenderChange::FinalizedMessagesChanged);
        assert_cache_scope(&state, false, true, true, &[CacheInvalidation::Finalized]);
    }

    #[test]
    fn streaming_content_change_clears_only_content() {
        let mut state = seeded_caches();
        state.apply_change(RenderChange::StreamingContentChanged);
        assert_cache_scope(&state, true, false, true, &[CacheInvalidation::Content]);
    }

    #[test]
    fn streaming_thinking_change_clears_only_thinking() {
        let mut state = seeded_caches();
        state.apply_change(RenderChange::StreamingThinkingChanged);
        assert_cache_scope(&state, true, true, false, &[CacheInvalidation::Thinking]);
    }

    #[test]
    fn delegate_presentation_change_clears_only_cards() {
        let mut state = seeded_caches();
        state.apply_change(RenderChange::DelegatePresentationChanged);
        assert_cache_scope(&state, false, true, true, &[CacheInvalidation::Finalized]);
    }

    #[test]
    fn theme_change_clears_cards_content_then_thinking() {
        let mut state = seeded_caches();
        state.apply_change(RenderChange::ThemeChanged);
        assert_cache_scope(
            &state,
            false,
            false,
            false,
            &[
                CacheInvalidation::Finalized,
                CacheInvalidation::Content,
                CacheInvalidation::Thinking,
            ],
        );
    }

    #[test]
    fn external_editor_return_clears_cards_and_content_but_retains_thinking() {
        let mut state = seeded_caches();
        state.apply_change(RenderChange::ExternalEditorReturned);
        assert_cache_scope(
            &state,
            false,
            false,
            true,
            &[CacheInvalidation::Finalized, CacheInvalidation::Content],
        );
    }

    #[test]
    fn tick_replacement_preserves_wall_clock_derived_value() {
        let mut state = RenderState::new();
        state.replace_tick(17);
        state.replace_tick(3);
        assert_eq!(state.tick, 3);
    }

    #[test]
    fn chat_viewport_movement_saturates_without_touching_render_policy() {
        let mut state = seeded_caches();
        state.compensate_chat_growth(12);
        let card_identity = state.test_card_identity(0);
        let content_identity = state.test_streaming_cache_identity(StreamKind::Content);
        let thinking_identity = state.test_streaming_cache_identity(StreamKind::Thinking);

        state.set_chat_scroll_offset(u16::MAX - 1);
        state.scroll_chat_up(3);
        assert_eq!(state.chat_scroll_offset(), u16::MAX);
        state.scroll_chat_down(u16::MAX);
        assert_eq!(state.chat_scroll_offset(), 0);
        state.scroll_chat_down(1);
        assert_eq!(state.chat_scroll_offset(), 0);
        state.scroll_chat_to_top();
        assert_eq!(state.chat_scroll_offset(), u16::MAX);
        state.scroll_chat_to_bottom();
        assert_eq!(state.chat_scroll_offset(), 0);
        assert_eq!(state.test_chat_previous_total_height(), 12);

        assert_eq!(state.test_card_identity(0), card_identity);
        assert_eq!(
            state.test_streaming_cache_identity(StreamKind::Content),
            content_identity
        );
        assert_eq!(
            state.test_streaming_cache_identity(StreamKind::Thinking),
            thinking_identity
        );
        assert_eq!(state.test_session_epoch(), 0);
        assert!(state.invalidation_order.is_empty());
    }

    #[test]
    fn bottom_pinned_chat_viewport_stays_pinned_during_growth() {
        let mut state = RenderState::new();

        state.compensate_chat_growth(10);
        state.compensate_chat_growth(17);

        assert_eq!(state.chat_scroll_offset(), 0);
        assert_eq!(state.test_chat_previous_total_height(), 17);
        assert_eq!(state.clamp_chat_scroll(17, 5), 12);
        assert_eq!(state.chat_scroll_offset(), 0);
    }

    #[test]
    fn scrolled_chat_growth_preserves_top_origin_slice() {
        let mut state = RenderState::new();
        state.compensate_chat_growth(10);
        state.set_chat_scroll_offset(3);
        assert_eq!(state.clamp_chat_scroll(10, 4), 3);

        state.compensate_chat_growth(15);

        assert_eq!(state.chat_scroll_offset(), 8);
        assert_eq!(state.test_chat_previous_total_height(), 15);
        assert_eq!(state.clamp_chat_scroll(15, 4), 3);
    }

    #[test]
    fn shrink_updates_baseline_then_clamps_chat_scroll() {
        let mut state = RenderState::new();
        state.test_seed_chat_viewport(12, 20);

        state.compensate_chat_growth(8);

        assert_eq!(state.test_chat_previous_total_height(), 8);
        assert_eq!(state.clamp_chat_scroll(8, 5), 0);
        assert_eq!(state.chat_scroll_offset(), 3);
    }

    #[test]
    fn chat_scroll_clamp_handles_empty_short_zero_and_tiny_viewports() {
        let mut state = RenderState::new();
        state.scroll_chat_to_top();
        assert_eq!(state.clamp_chat_scroll(0, 0), 0);
        assert_eq!(state.chat_scroll_offset(), 0);

        state.scroll_chat_to_top();
        assert_eq!(state.clamp_chat_scroll(3, 5), 0);
        assert_eq!(state.chat_scroll_offset(), 0);

        state.set_chat_scroll_offset(2);
        assert_eq!(state.clamp_chat_scroll(5, 0), 3);
        assert_eq!(state.chat_scroll_offset(), 2);

        state.scroll_chat_to_top();
        assert_eq!(state.clamp_chat_scroll(5, 1), 0);
        assert_eq!(state.chat_scroll_offset(), 4);
    }

    #[test]
    fn reset_chat_viewport_clears_offset_and_baseline_only() {
        let mut state = seeded_caches();
        state.test_seed_chat_viewport(7, 19);
        let card_identity = state.test_card_identity(0);
        let content_identity = state.test_streaming_cache_identity(StreamKind::Content);
        let thinking_identity = state.test_streaming_cache_identity(StreamKind::Thinking);

        state.reset_chat_viewport();

        assert_eq!(state.chat_scroll_offset(), 0);
        assert_eq!(state.test_chat_previous_total_height(), 0);
        assert_eq!(state.test_card_identity(0), card_identity);
        assert_eq!(
            state.test_streaming_cache_identity(StreamKind::Content),
            content_identity
        );
        assert_eq!(
            state.test_streaming_cache_identity(StreamKind::Thinking),
            thinking_identity
        );
        assert_eq!(state.test_session_epoch(), 0);
        assert!(state.invalidation_order.is_empty());
    }
}
