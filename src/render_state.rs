use crate::highlight::Highlighter;
use crate::markdown::CardBlock;
use crate::ui::CardCache;

/// Temporary phase 4 owner for render-local state.
pub(crate) struct RenderState {
    pub(crate) highlighter: Highlighter,
    pub(crate) card_cache: CardCache,
    pub(crate) streaming_cache: StreamingCache,
    pub(crate) streaming_thinking_cache: StreamingCache,
    pub(crate) prev_total_height: u16,
    pub(crate) tick: u64,
}

impl RenderState {
    pub(crate) fn new() -> Self {
        Self {
            highlighter: Highlighter::new(),
            card_cache: CardCache::new(),
            streaming_cache: StreamingCache::new(),
            streaming_thinking_cache: StreamingCache::new(),
            prev_total_height: 0,
            tick: 0,
        }
    }

    pub(crate) fn invalidate_card_cache(&mut self) {
        self.card_cache.invalidate();
    }

    pub(crate) fn invalidate_content_cache(&mut self) {
        self.streaming_cache.invalidate();
    }

    pub(crate) fn invalidate_thinking_cache(&mut self) {
        self.streaming_thinking_cache.invalidate();
    }

    pub(crate) fn invalidate_theme_caches(&mut self) {
        self.invalidate_card_cache();
        self.invalidate_content_cache();
        self.invalidate_thinking_cache();
    }

    pub(crate) fn replace_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    pub(crate) fn compensate_scroll_for_growth(
        &mut self,
        total_height: u16,
        scroll_offset: &mut u16,
    ) {
        let growth = total_height.saturating_sub(self.prev_total_height);
        if *scroll_offset > 0 && growth > 0 {
            *scroll_offset = (*scroll_offset).saturating_add(growth);
        }
        self.prev_total_height = total_height;
    }
}

/// Cache for rendered streaming markdown to avoid re-parsing every frame.
/// Invalidated when the corresponding streaming buffer grows or is cleared.
pub(crate) struct StreamingCache {
    rendered_len: usize,
    blocks: Vec<CardBlock>,
}

impl StreamingCache {
    pub(crate) fn new() -> Self {
        Self {
            rendered_len: 0,
            blocks: Vec::new(),
        }
    }

    pub(crate) fn get(&self, content_len: usize) -> Option<&[CardBlock]> {
        if content_len > 0 && content_len == self.rendered_len {
            Some(&self.blocks)
        } else {
            None
        }
    }

    pub(crate) fn store(&mut self, content_len: usize, blocks: Vec<CardBlock>) {
        self.rendered_len = content_len;
        self.blocks = blocks;
    }

    pub(crate) fn invalidate(&mut self) {
        self.rendered_len = 0;
        self.blocks.clear();
    }
}

#[cfg(test)]
mod tests {
    use ratatui::text::Line;

    use super::*;

    fn block(text: &'static str) -> CardBlock {
        CardBlock::Text(Line::from(text))
    }

    fn seed_all_caches(state: &mut RenderState) {
        state.card_cache.processed_messages = 2;
        state.streaming_cache.store(7, vec![block("content")]);
        state
            .streaming_thinking_cache
            .store(8, vec![block("thinking")]);
    }

    #[test]
    fn constructor_uses_exact_six_field_defaults() {
        let RenderState {
            highlighter: _,
            card_cache,
            streaming_cache,
            streaming_thinking_cache,
            prev_total_height,
            tick,
        } = RenderState::new();

        assert!(card_cache.cards.is_empty());
        assert_eq!(card_cache.processed_messages, 0);
        assert!(streaming_cache.get(1).is_none());
        assert!(streaming_thinking_cache.get(1).is_none());
        assert_eq!(prev_total_height, 0);
        assert_eq!(tick, 0);
    }

    #[test]
    fn streaming_cache_hits_only_the_stored_nonzero_length() {
        let mut cache = StreamingCache::new();
        cache.store(7, vec![block("content")]);

        assert!(cache.get(0).is_none());
        assert!(cache.get(6).is_none());
        assert_eq!(cache.get(7).map(<[_]>::len), Some(1));
        assert!(cache.get(8).is_none());
    }

    #[test]
    fn individual_invalidations_clear_only_the_selected_cache() {
        let mut state = RenderState::new();
        seed_all_caches(&mut state);

        state.invalidate_content_cache();
        assert_eq!(state.card_cache.processed_messages, 2);
        assert!(state.streaming_cache.get(7).is_none());
        assert!(state.streaming_thinking_cache.get(8).is_some());

        state.invalidate_thinking_cache();
        assert_eq!(state.card_cache.processed_messages, 2);
        assert!(state.streaming_thinking_cache.get(8).is_none());

        state.invalidate_card_cache();
        assert_eq!(state.card_cache.processed_messages, 0);
    }

    #[test]
    fn theme_invalidation_clears_all_three_caches() {
        let mut state = RenderState::new();
        seed_all_caches(&mut state);

        state.invalidate_theme_caches();

        assert_eq!(state.card_cache.processed_messages, 0);
        assert!(state.streaming_cache.get(7).is_none());
        assert!(state.streaming_thinking_cache.get(8).is_none());
    }

    #[test]
    fn session_switch_invalidation_keeps_streaming_sessions_isolated() {
        let mut state = RenderState::new();
        seed_all_caches(&mut state);

        state.invalidate_content_cache();
        state.invalidate_thinking_cache();
        state.invalidate_card_cache();
        state.streaming_cache.store(4, vec![block("next")]);

        assert_eq!(state.card_cache.processed_messages, 0);
        assert!(state.streaming_cache.get(7).is_none());
        assert!(state.streaming_cache.get(4).is_some());
        assert!(state.streaming_thinking_cache.get(8).is_none());
    }

    #[test]
    fn external_editor_invalidation_retains_thinking_cache() {
        let mut state = RenderState::new();
        seed_all_caches(&mut state);

        state.invalidate_card_cache();
        state.invalidate_content_cache();

        assert_eq!(state.card_cache.processed_messages, 0);
        assert!(state.streaming_cache.get(7).is_none());
        assert!(state.streaming_thinking_cache.get(8).is_some());
    }

    #[test]
    fn delegate_invalidation_clears_only_card_cache() {
        let mut state = RenderState::new();
        seed_all_caches(&mut state);

        state.invalidate_card_cache();

        assert_eq!(state.card_cache.processed_messages, 0);
        assert!(state.streaming_cache.get(7).is_some());
        assert!(state.streaming_thinking_cache.get(8).is_some());
    }

    #[test]
    fn tick_replacement_preserves_wall_clock_derived_value() {
        let mut state = RenderState::new();

        state.replace_tick(17);
        state.replace_tick(3);

        assert_eq!(state.tick, 3);
    }

    #[test]
    fn height_growth_compensates_only_when_scrolled_up() {
        let mut state = RenderState::new();
        let mut scroll_offset = 4;

        state.compensate_scroll_for_growth(10, &mut scroll_offset);
        assert_eq!(scroll_offset, 14);
        assert_eq!(state.prev_total_height, 10);

        state.compensate_scroll_for_growth(7, &mut scroll_offset);
        assert_eq!(scroll_offset, 14);
        assert_eq!(state.prev_total_height, 7);

        scroll_offset = 0;
        state.compensate_scroll_for_growth(12, &mut scroll_offset);
        assert_eq!(scroll_offset, 0);
        assert_eq!(state.prev_total_height, 12);
    }
}
