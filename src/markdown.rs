use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};
use ratatui::{
    style::Style,
    text::{Line, Span},
};

use crate::highlight::Highlighter;
use crate::theme::Theme;
use crate::ui::{MD_BULLET, MD_HRULE_CHAR};

/// A block inside a card — either plain text or a deferred table.
#[derive(Clone)]
pub enum CardBlock {
    Text(Line<'static>),
    Table(TableBlock),
}

/// Deferred table that can be re-laid-out at draw time.
#[derive(Clone)]
pub struct TableBlock {
    pub alignments: Vec<Alignment>,
    pub rows: Vec<Vec<Vec<Span<'static>>>>, // row → cell → spans
    pub base_style: Style,
}

/// Render a markdown string into styled card blocks.
/// `base_style` is applied to plain text (allows caller to set the card bg).
pub fn render(md: &str, base_style: Style, hl: &Highlighter) -> Vec<CardBlock> {
    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let parser = Parser::new_ext(md, opts);

    let mut blocks: Vec<CardBlock> = Vec::new();
    let mut ctx = RenderCtx {
        base_style,
        hl,
        style_stack: vec![base_style],
        current_spans: Vec::new(),
        in_code_block: false,
        code_block_lines: Vec::new(),
        code_lang: None,
        list_depth: 0,
        list_indices: Vec::new(),
        in_heading: false,
        heading_level: 0,
        in_blockquote: false,
        in_table: false,
        in_table_head: false,
        table_alignments: Vec::new(),
        table_rows: Vec::new(),
        current_cell_spans: Vec::new(),
    };

    for event in parser {
        match event {
            Event::Start(tag) => ctx.open_tag(&tag, &mut blocks),
            Event::End(tag) => ctx.close_tag(&tag, &mut blocks),
            Event::Text(text) => ctx.push_text(&text),
            Event::Code(code) => ctx.push_inline_code(&code),
            Event::SoftBreak => ctx.push_text(" "),
            Event::HardBreak => ctx.flush_line(&mut blocks),
            Event::Rule => {
                ctx.flush_line(&mut blocks);
                blocks.push(CardBlock::Text(Line::from(Span::styled(
                    MD_HRULE_CHAR.repeat(40),
                    Theme::md_hr(),
                ))));
            }
            _ => {}
        }
    }

    // flush any remaining spans
    ctx.flush_line(&mut blocks);
    // strip trailing blank lines — card padding handles vertical spacing
    while blocks
        .last()
        .is_some_and(|b| matches!(b, CardBlock::Text(l) if l.spans.is_empty()))
    {
        blocks.pop();
    }
    blocks
}

struct RenderCtx<'a> {
    base_style: Style,
    hl: &'a Highlighter,
    style_stack: Vec<Style>,
    current_spans: Vec<Span<'static>>,
    in_code_block: bool,
    code_block_lines: Vec<String>,
    code_lang: Option<String>,
    list_depth: usize,
    list_indices: Vec<Option<u64>>, // None = unordered, Some(n) = ordered
    in_heading: bool,
    heading_level: u8,
    in_blockquote: bool,
    // table state
    in_table: bool,
    in_table_head: bool,
    table_alignments: Vec<Alignment>,
    table_rows: Vec<Vec<Vec<Span<'static>>>>, // rows → cells → spans
    current_cell_spans: Vec<Span<'static>>,
}

impl RenderCtx<'_> {
    fn current_style(&self) -> Style {
        self.style_stack.last().copied().unwrap_or(self.base_style)
    }

    fn push_style(&mut self, style: Style) {
        // merge with current: new style overrides fields that are set
        let current = self.current_style();
        let merged = Style {
            fg: style.fg.or(current.fg),
            bg: style.bg.or(current.bg),
            underline_color: style.underline_color.or(current.underline_color),
            add_modifier: current.add_modifier | style.add_modifier,
            sub_modifier: current.sub_modifier | style.sub_modifier,
        };
        self.style_stack.push(merged);
    }

    fn pop_style(&mut self) {
        if self.style_stack.len() > 1 {
            self.style_stack.pop();
        }
    }

    fn open_tag(&mut self, tag: &Tag, blocks: &mut Vec<CardBlock>) {
        match tag {
            Tag::Heading { level, .. } => {
                self.in_heading = true;
                self.heading_level = *level as u8;
                self.push_style(Theme::md_heading());
            }
            Tag::Emphasis => self.push_style(Theme::md_italic()),
            Tag::Strong => self.push_style(Theme::md_bold()),
            Tag::Strikethrough => self.push_style(Theme::md_strikethrough()),
            Tag::Link {
                dest_url, title, ..
            } => {
                self.push_style(Theme::md_link_title());
                // store url for close_tag — we'll append it
                // for simplicity we just style the link text
                let _ = (dest_url, title);
            }
            Tag::BlockQuote(_) => {
                self.in_blockquote = true;
                self.push_style(Theme::md_blockquote());
            }
            Tag::CodeBlock(kind) => {
                self.in_code_block = true;
                self.code_block_lines.clear();
                self.code_lang = match kind {
                    pulldown_cmark::CodeBlockKind::Fenced(lang) => {
                        let l = lang.trim().to_string();
                        if l.is_empty() { None } else { Some(l) }
                    }
                    _ => None,
                };
            }
            Tag::List(start) => {
                // Flush any pending spans from the parent item so a nested
                // list starts on its own line rather than being concatenated.
                self.flush_line(blocks);
                self.list_depth += 1;
                self.list_indices.push(*start);
            }
            Tag::Item => {
                // handled in Text via prefix
            }
            Tag::Paragraph => {}
            Tag::Table(alignments) => {
                self.in_table = true;
                self.table_alignments = alignments.clone();
                self.table_rows.clear();
            }
            Tag::TableHead => {
                self.in_table_head = true;
                self.table_rows.push(Vec::new());
            }
            Tag::TableRow => {
                self.table_rows.push(Vec::new());
            }
            Tag::TableCell => {
                self.current_cell_spans.clear();
            }
            _ => {}
        }
    }

    fn close_tag(&mut self, tag: &TagEnd, blocks: &mut Vec<CardBlock>) {
        match tag {
            TagEnd::Heading(_) => {
                // prepend heading marker
                let marker = match self.heading_level {
                    1 => "# ",
                    2 => "## ",
                    3 => "### ",
                    _ => "#### ",
                };
                self.current_spans
                    .insert(0, Span::styled(marker.to_string(), Theme::md_heading()));
                self.in_heading = false;
                self.heading_level = 0;
                self.flush_line(blocks);
                self.pop_style();
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.pop_style();
            }
            TagEnd::Link => {
                self.pop_style();
            }
            TagEnd::BlockQuote(_) => {
                self.in_blockquote = false;
                self.pop_style();
            }
            TagEnd::CodeBlock => {
                self.in_code_block = false;
                // lang label
                if let Some(lang) = &self.code_lang {
                    blocks.push(CardBlock::Text(Line::from(Span::styled(
                        format!("  {lang}"),
                        Theme::md_code_lang(),
                    ))));
                }
                // syntax-highlighted code
                let code = self.code_block_lines.join("\n");
                let highlighted = self.hl.highlight_block(&code, self.code_lang.as_deref());
                blocks.extend(highlighted.into_iter().map(CardBlock::Text));
                blocks.push(CardBlock::Text(Line::from(Span::styled(
                    " ",
                    Theme::md_code_block(),
                ))));
                self.code_block_lines.clear();
                self.code_lang = None;
            }
            TagEnd::List(_) => {
                self.list_depth = self.list_depth.saturating_sub(1);
                self.list_indices.pop();
                // add spacing after top-level lists (like paragraphs do)
                if self.list_depth == 0 {
                    blocks.push(CardBlock::Text(Line::default()));
                }
            }
            TagEnd::Item => {
                self.flush_line(blocks);
            }
            TagEnd::Paragraph => {
                self.flush_line(blocks);
                blocks.push(CardBlock::Text(Line::default()));
            }
            TagEnd::Table => {
                self.flush_table(blocks);
                self.in_table = false;
                self.in_table_head = false;
                self.table_alignments.clear();
                self.table_rows.clear();
            }
            TagEnd::TableHead => {
                self.in_table_head = false;
            }
            TagEnd::TableRow => {}
            TagEnd::TableCell => {
                let spans = std::mem::take(&mut self.current_cell_spans);
                if let Some(row) = self.table_rows.last_mut() {
                    row.push(spans);
                }
            }
            _ => {}
        }
    }

    /// Emit list-bullet / blockquote prefix if this is the first span on
    /// the current line and we are inside a list or blockquote.
    fn ensure_line_prefix(&mut self) {
        // List bullet / number
        if self.current_spans.is_empty() && !self.list_indices.is_empty() {
            let indent = "  ".repeat(self.list_depth.saturating_sub(1));
            let bullet = if let Some(Some(n)) = self.list_indices.last_mut() {
                let b = format!("{indent}{n}. ");
                *n += 1;
                b
            } else {
                format!("{indent}{MD_BULLET}")
            };
            self.current_spans
                .push(Span::styled(bullet, Theme::md_list_bullet()));
        }

        // Blockquote prefix
        if self.in_blockquote && self.current_spans.is_empty() {
            self.current_spans
                .push(Span::styled("  ", Theme::md_blockquote()));
        }
    }

    fn push_text(&mut self, text: &str) {
        if self.in_code_block {
            // collect lines for code block
            for line in text.split('\n') {
                self.code_block_lines.push(line.to_string());
            }
            // the last split element is from a trailing \n — remove empty
            if self.code_block_lines.last().is_some_and(|l| l.is_empty()) {
                self.code_block_lines.pop();
            }
            return;
        }

        // table cell text — collect into cell spans, not current_spans
        if self.in_table {
            let style = if self.in_table_head {
                Theme::md_table_header()
            } else {
                self.current_style()
            };
            self.current_cell_spans
                .push(Span::styled(text.to_string(), style));
            return;
        }

        let style = self.current_style();
        self.ensure_line_prefix();
        self.current_spans
            .push(Span::styled(text.to_string(), style));
    }

    fn push_inline_code(&mut self, code: &str) {
        if self.in_table {
            self.current_cell_spans
                .push(Span::styled(code.to_string(), Theme::md_code_inline()));
            return;
        }
        self.ensure_line_prefix();
        self.current_spans
            .push(Span::styled(code.to_string(), Theme::md_code_inline()));
    }

    /// Push the collected table rows as a deferred TableBlock.
    fn flush_table(&mut self, blocks: &mut Vec<CardBlock>) {
        let num_cols = self.table_alignments.len();
        if num_cols == 0 || self.table_rows.is_empty() {
            return;
        }
        let rows = std::mem::take(&mut self.table_rows);
        blocks.push(CardBlock::Table(TableBlock {
            alignments: self.table_alignments.clone(),
            rows,
            base_style: self.base_style,
        }));
    }

    fn flush_line(&mut self, blocks: &mut Vec<CardBlock>) {
        if !self.current_spans.is_empty() {
            let spans = std::mem::take(&mut self.current_spans);
            blocks.push(CardBlock::Text(Line::from(spans)));
        }
    }
}

// ── Table layout engine ─────────────────────────────────────────────────────

impl TableBlock {
    /// Re-layout this table to fit inside `inner_w` columns.
    pub fn layout(&self, inner_w: usize) -> Vec<Line<'static>> {
        use unicode_width::UnicodeWidthStr;

        let num_cols = self.alignments.len();
        if num_cols == 0 || self.rows.is_empty() {
            return Vec::new();
        }

        const MIN_COL: usize = 3;

        let cell_width = |cell: &[Span<'_>]| -> usize {
            cell.iter()
                .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
                .sum()
        };

        let mut natural_widths = vec![0usize; num_cols];
        for row in &self.rows {
            for (c, cell) in row.iter().enumerate().take(num_cols) {
                natural_widths[c] = natural_widths[c].max(cell_width(cell));
            }
        }
        for w in &mut natural_widths {
            *w = (*w).max(1);
        }

        let chrome = num_cols * 3 + 1;
        let natural_total: usize = natural_widths.iter().sum();

        let col_widths = if natural_total + chrome <= inner_w {
            natural_widths
        } else {
            let budget = inner_w.saturating_sub(chrome);
            let min_needed = num_cols * MIN_COL;
            if budget >= min_needed {
                let total_natural = natural_total.max(1);
                let mut widths: Vec<usize> = natural_widths
                    .iter()
                    .map(|&w| (w * budget / total_natural).max(MIN_COL))
                    .collect();
                let mut used: usize = widths.iter().sum();
                while used < budget {
                    let mut best = 0;
                    let mut best_deficit = 0i64;
                    for (i, &w) in natural_widths.iter().enumerate() {
                        let target = w * budget / total_natural;
                        let deficit = target as i64 - widths[i] as i64;
                        if deficit > best_deficit {
                            best_deficit = deficit;
                            best = i;
                        }
                    }
                    if best_deficit <= 0 {
                        let i = widths
                            .iter()
                            .enumerate()
                            .find(|(_, w)| **w < budget)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        widths[i] += 1;
                    } else {
                        widths[best] += 1;
                    }
                    used += 1;
                }
                widths
            } else {
                vec![MIN_COL; num_cols]
            }
        };

        let border_style = Theme::md_table_border();

        let h_border = |left: char, mid: char, right: char| -> Line<'static> {
            let mut s = String::new();
            for (i, &w) in col_widths.iter().enumerate() {
                if i == 0 {
                    s.push(left);
                } else {
                    s.push(mid);
                }
                for _ in 0..w + 2 {
                    s.push('─');
                }
            }
            s.push(right);
            Line::from(Span::styled(s, border_style))
        };

        let padding = |text_w: usize, col_w: usize, align: Alignment| -> (usize, usize) {
            let gap = col_w.saturating_sub(text_w);
            match align {
                Alignment::Right => (gap, 0),
                Alignment::Center => {
                    let left = gap / 2;
                    (left, gap - left)
                }
                _ => (0, gap),
            }
        };

        let mut result = Vec::new();

        // Top border
        result.push(h_border('┌', '┬', '┐'));

        for (row_idx, row) in self.rows.iter().enumerate() {
            let is_header = row_idx == 0;
            let style = if is_header {
                Theme::md_table_header()
            } else {
                self.base_style
            };

            // Wrap each cell to its column budget
            let mut cell_lines: Vec<Vec<Vec<Span<'static>>>> = Vec::new();
            let mut max_lines = 1;
            for (c, &col_w) in col_widths.iter().enumerate().take(num_cols) {
                let cell = row.get(c).map(|v| v.as_slice()).unwrap_or(&[]);
                let wrapped = if cell.is_empty() {
                    vec![Vec::new()]
                } else {
                    wrap_spans(cell, col_w)
                };
                max_lines = max_lines.max(wrapped.len());
                cell_lines.push(wrapped);
            }

            // Emit each sub-line of this logical row
            for line_idx in 0..max_lines {
                let mut spans: Vec<Span<'static>> = Vec::new();
                for (c, &col_w) in col_widths.iter().enumerate().take(num_cols) {
                    spans.push(Span::styled("│ ".to_string(), border_style));

                    let cell_spans = cell_lines[c].get(line_idx);
                    let text_w = cell_spans
                        .map(|s| {
                            s.iter()
                                .map(|sp| UnicodeWidthStr::width(sp.content.as_ref()))
                                .sum::<usize>()
                        })
                        .unwrap_or(0);
                    let align = self.alignments.get(c).copied().unwrap_or(Alignment::None);
                    let (lpad, rpad) = padding(text_w, col_w, align);

                    if lpad > 0 {
                        spans.push(Span::styled(" ".repeat(lpad), style));
                    }
                    if let Some(cell) = cell_spans {
                        spans.extend(cell.iter().cloned());
                    }
                    if rpad > 0 {
                        spans.push(Span::styled(" ".repeat(rpad), style));
                    }
                    spans.push(Span::styled(" ".to_string(), border_style));
                }
                spans.push(Span::styled("│".to_string(), border_style));
                result.push(Line::from(spans));
            }

            // Separator after header
            if is_header {
                result.push(h_border('├', '┼', '┤'));
            }
        }

        // Bottom border
        result.push(h_border('└', '┴', '┘'));

        // Trailing blank line for spacing (like paragraphs)
        result.push(Line::default());

        result
    }
}

/// Word-wrap a sequence of spans into sub-lines of at most `width` display columns.
fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    use unicode_width::UnicodeWidthStr;

    let width = width.max(1);

    // Flatten to chars with their original styles.
    let mut chars = Vec::new();
    for span in spans {
        let s = span.style;
        for c in span.content.chars() {
            chars.push((c, s));
        }
    }

    if chars.is_empty() {
        return vec![Vec::new()];
    }

    // Collect words (non-whitespace sequences).  Inter-word whitespace is
    // ignored; a single synthetic space is inserted when joining words.
    let mut words: Vec<&[(char, Style)]> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        while i < chars.len() && chars[i].0.is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < chars.len() && !chars[i].0.is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            words.push(&chars[start..i]);
        }
    }

    let mut lines: Vec<Vec<(char, Style)>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    let mut cur_w: usize = 0;

    for word in words {
        let w = word
            .iter()
            .map(|(c, _)| UnicodeWidthStr::width(c.encode_utf8(&mut [0; 4])))
            .sum::<usize>();
        if cur.is_empty() {
            if w <= width {
                cur.extend_from_slice(word);
                cur_w = w;
            } else {
                // Hard-break the word into chunks.
                let mut idx = 0;
                while idx < word.len() {
                    let mut chunk = Vec::new();
                    let mut chunk_w = 0;
                    while idx < word.len() {
                        let cw = UnicodeWidthStr::width(word[idx].0.encode_utf8(&mut [0; 4]));
                        if chunk_w + cw > width {
                            break;
                        }
                        chunk_w += cw;
                        chunk.push(word[idx]);
                        idx += 1;
                    }
                    if chunk.is_empty() {
                        // Single char wider than width — take it anyway.
                        chunk.push(word[idx]);
                        idx += 1;
                    }
                    lines.push(chunk);
                }
            }
        } else if cur_w + 1 + w <= width {
            let space_style = cur.last().map(|(_, s)| *s).unwrap_or_default();
            cur.push((' ', space_style));
            cur.extend_from_slice(word);
            cur_w += 1 + w;
        } else {
            lines.push(std::mem::take(&mut cur));
            cur_w = 0;
            if w <= width {
                cur.extend_from_slice(word);
                cur_w = w;
            } else {
                let mut idx = 0;
                while idx < word.len() {
                    let mut chunk = Vec::new();
                    let mut chunk_w = 0;
                    while idx < word.len() {
                        let cw = UnicodeWidthStr::width(word[idx].0.encode_utf8(&mut [0; 4]));
                        if chunk_w + cw > width {
                            break;
                        }
                        chunk_w += cw;
                        chunk.push(word[idx]);
                        idx += 1;
                    }
                    if chunk.is_empty() {
                        chunk.push(word[idx]);
                        idx += 1;
                    }
                    lines.push(chunk);
                }
            }
        }
    }

    if !cur.is_empty() {
        lines.push(cur);
    }

    if lines.is_empty() {
        lines.push(Vec::new());
    }

    // Merge consecutive chars of the same style into spans.
    lines
        .into_iter()
        .map(|line| {
            let mut spans = Vec::new();
            let mut text = String::new();
            let mut style = None;
            for &(ch, s) in &line {
                if style == Some(s) {
                    text.push(ch);
                } else {
                    if let Some(prev) = style {
                        spans.push(Span::styled(std::mem::take(&mut text), prev));
                    }
                    text.push(ch);
                    style = Some(s);
                }
            }
            if let Some(s) = style {
                spans.push(Span::styled(text, s));
            }
            spans
        })
        .collect()
}

/// Insert `span` at the front of the first `CardBlock::Text` in `blocks`.
/// If there is no text block, push a new one containing only `span`.
pub fn prepend_span_to_first_text(blocks: &mut Vec<CardBlock>, span: Span<'static>) {
    for block in blocks.iter_mut() {
        if let CardBlock::Text(line) = block {
            line.spans.insert(0, span);
            return;
        }
    }
    blocks.push(CardBlock::Text(Line::from(vec![span])));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::highlight::Highlighter;
    use crate::theme::Theme;

    /// Extract the raw text from rendered Lines (ignoring styles).
    fn lines_to_text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    fn render_md(md: &str) -> Vec<Line<'static>> {
        Theme::set_by_index(0);
        Theme::begin_frame();
        let hl = Highlighter::new();
        let base = Style::default();
        let blocks = render(md, base, &hl);
        let mut lines = Vec::new();
        for block in blocks {
            match block {
                CardBlock::Text(line) => lines.push(line),
                CardBlock::Table(table) => lines.extend(table.layout(200)),
            }
        }
        lines
    }

    // ── Basic table structure ──────────────────────────────────────────

    #[test]
    fn table_basic_renders_box_drawing_borders() {
        let md = "| A | B |\n|---|---|\n| 1 | 2 |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        // Should contain box-drawing top border
        assert!(
            text.iter().any(|l| l.contains('┌') && l.contains('┐')),
            "Expected top border with ┌ and ┐, got: {text:#?}"
        );
        // Should contain box-drawing bottom border
        assert!(
            text.iter().any(|l| l.contains('└') && l.contains('┘')),
            "Expected bottom border with └ and ┘, got: {text:#?}"
        );
        // Should contain header separator
        assert!(
            text.iter().any(|l| l.contains('├') && l.contains('┤')),
            "Expected header separator with ├ and ┤, got: {text:#?}"
        );
        // Should contain vertical separators
        assert!(
            text.iter().any(|l| l.contains('│')),
            "Expected vertical separators │, got: {text:#?}"
        );
    }

    #[test]
    fn table_basic_contains_cell_content() {
        let md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let joined = text.join("\n");

        assert!(
            joined.contains("Name"),
            "Missing header 'Name' in:\n{joined}"
        );
        assert!(joined.contains("Age"), "Missing header 'Age' in:\n{joined}");
        assert!(
            joined.contains("Alice"),
            "Missing cell 'Alice' in:\n{joined}"
        );
        assert!(joined.contains("30"), "Missing cell '30' in:\n{joined}");
        assert!(joined.contains("Bob"), "Missing cell 'Bob' in:\n{joined}");
        assert!(joined.contains("25"), "Missing cell '25' in:\n{joined}");
    }

    #[test]
    fn table_correct_line_count() {
        // A 1-header + 2-body-row table should produce:
        //   top border, header row, separator, body row 1, body row 2, bottom border, blank line
        let md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        // Filter out empty trailing lines
        let non_empty: Vec<_> = text.iter().filter(|l| !l.is_empty()).collect();
        assert_eq!(
            non_empty.len(),
            6, // top + header + sep + 2 body rows + bottom
            "Expected 6 non-empty lines, got {}: {non_empty:#?}",
            non_empty.len()
        );
    }

    // ── Alignment ──────────────────────────────────────────────────────

    #[test]
    fn table_left_alignment_pads_right() {
        let md = "| X |\n|:--|\n| hi |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        // Find the body row with "hi"
        let body_row = text
            .iter()
            .find(|l| l.contains("hi"))
            .expect("body row missing");
        // After "hi" there should be trailing spaces before the border
        assert!(
            body_row.contains("hi "),
            "Left-aligned cell should have trailing space padding: {body_row}"
        );
    }

    #[test]
    fn table_right_alignment_pads_left() {
        let md = "| X |\n|--:|\n| hi |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        let body_row = text
            .iter()
            .find(|l| l.contains("hi"))
            .expect("body row missing");
        // Before "hi" there should be leading spaces after the border
        assert!(
            body_row.contains(" hi"),
            "Right-aligned cell should have leading space padding: {body_row}"
        );
    }

    #[test]
    fn table_center_alignment() {
        let md = "| Header |\n|:------:|\n| hi |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        let body_row = text
            .iter()
            .find(|l| l.contains("hi"))
            .expect("body row missing");
        // "hi" should have padding on both sides
        assert!(
            body_row.contains(" hi "),
            "Center-aligned cell should have padding on both sides: {body_row}"
        );
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn table_empty_cells() {
        let md = "| A | B |\n|---|---|\n|   | x |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let joined = text.join("\n");

        assert!(
            joined.contains("x"),
            "Non-empty cell content missing: {joined}"
        );
        // Should still render correct structure
        assert!(
            text.iter().any(|l| l.contains('┌')),
            "Table borders missing for table with empty cells"
        );
    }

    #[test]
    fn table_single_column() {
        let md = "| Solo |\n|------|\n| val |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let joined = text.join("\n");

        assert!(joined.contains("Solo"), "Header missing");
        assert!(joined.contains("val"), "Body missing");
        assert!(
            text.iter().any(|l| l.contains('┌') && l.contains('┐')),
            "Single-column table should have borders"
        );
        // Single column: no ┬ or ┼ (only top-left and top-right)
        assert!(
            !text.iter().any(|l| l.contains('┬')),
            "Single-column table should NOT have ┬ junction"
        );
    }

    #[test]
    fn table_inline_code_in_cell() {
        let md = "| Code |\n|------|\n| `foo` |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let joined = text.join("\n");

        assert!(
            joined.contains("foo"),
            "Inline code content should appear in table: {joined}"
        );
    }

    #[test]
    fn table_bold_in_cell() {
        let md = "| Styled |\n|--------|\n| **bold** |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let joined = text.join("\n");

        assert!(
            joined.contains("bold"),
            "Bold text should appear in table cell: {joined}"
        );
    }

    #[test]
    fn table_does_not_break_surrounding_content() {
        let md = "Before\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nAfter\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let joined = text.join("\n");

        assert!(joined.contains("Before"), "Text before table missing");
        assert!(joined.contains("After"), "Text after table missing");
        assert!(joined.contains('┌'), "Table borders missing");
    }

    // ── Header styling ─────────────────────────────────────────────────

    #[test]
    fn table_header_has_bold_modifier() {
        let md = "| Head |\n|------|\n| body |\n";
        let lines = render_md(md);

        // Find the line containing "Head" (the header row)
        let header_line = lines
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref().contains("Head")))
            .expect("Header line not found");

        let head_span = header_line
            .spans
            .iter()
            .find(|s| s.content.as_ref().contains("Head"))
            .expect("Header span not found");

        assert!(
            head_span
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::BOLD),
            "Header cell should be bold, got style: {:?}",
            head_span.style
        );
    }

    // ── Column width calculation ───────────────────────────────────────

    // ── List bullet rendering ────────────────────────────────────────────

    #[test]
    fn list_item_starting_with_inline_code_has_bullet() {
        let md = "- `foo` bar\n- `baz` qux\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        // Every non-empty line should contain the bullet character
        let bullet = "\u{2022}"; // •
        let items_with_bullet: Vec<_> = text.iter().filter(|l| l.contains(bullet)).collect();
        assert_eq!(
            items_with_bullet.len(),
            2,
            "Expected 2 lines with bullet, got {}: {text:#?}",
            items_with_bullet.len()
        );
        // First item should contain bullet followed by inline code content
        assert!(
            text.iter().any(|l| l.contains(bullet) && l.contains("foo")),
            "First list item should have bullet and 'foo': {text:#?}"
        );
        assert!(
            text.iter().any(|l| l.contains(bullet) && l.contains("baz")),
            "Second list item should have bullet and 'baz': {text:#?}"
        );
    }

    #[test]
    fn ordered_list_item_starting_with_inline_code_has_number() {
        let md = "1. `alpha` first\n2. `beta` second\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        assert!(
            text.iter().any(|l| l.contains("1.") && l.contains("alpha")),
            "First ordered item should have '1.' and 'alpha': {text:#?}"
        );
        assert!(
            text.iter().any(|l| l.contains("2.") && l.contains("beta")),
            "Second ordered item should have '2.' and 'beta': {text:#?}"
        );
    }

    #[test]
    fn nested_list_first_child_not_concatenated_to_parent() {
        // Reproduces the bug where the first nested item gets glued to the
        // parent line instead of appearing on its own indented line.
        let md = "\
- parent text:
  - `child1`
  - `child2`
";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let bullet = "\u{2022}"; // •

        // The parent line should NOT contain child content
        let parent = text
            .iter()
            .find(|l| l.contains("parent text"))
            .expect("should have a parent line");
        assert!(
            !parent.contains("child1"),
            "Parent line must not contain first child text, got: {parent:?}"
        );

        // child1 should be on its own line with a bullet
        assert!(
            text.iter()
                .any(|l| l.contains(bullet) && l.contains("child1")),
            "child1 should be on its own bulleted line: {text:#?}"
        );
        // child2 should be on its own line with a bullet
        assert!(
            text.iter()
                .any(|l| l.contains(bullet) && l.contains("child2")),
            "child2 should be on its own bulleted line: {text:#?}"
        );
    }

    #[test]
    fn nested_list_with_inline_code_real_world() {
        // Real-world case from the bug report: parent text followed by
        // nested list where sub-items start with inline code.
        let md = "\
- `/rest/settings` remains accessible and leaks config:
  - `settingsMode: \"public\"`
  - `authMethod: \"email\"`
  - `authCookie.secure: false`
- `authCookie.secure: false` is important.
";
        let lines = render_md(md);
        let text = lines_to_text(&lines);
        let bullet = "\u{2022}"; // •

        // Parent line must not contain settingsMode
        let parent = text
            .iter()
            .find(|l| l.contains("leaks config"))
            .expect("should have a parent line");
        assert!(
            !parent.contains("settingsMode"),
            "Parent line must not contain first nested item, got: {parent:?}"
        );

        // Each nested item should have its own bulleted line
        for needle in &["settingsMode", "authMethod", "authCookie.secure: false"] {
            assert!(
                text.iter()
                    .any(|l| l.contains(bullet) && l.contains(needle)),
                "{needle} should appear on its own bulleted line: {text:#?}"
            );
        }

        // The standalone item about authCookie should also have a bullet
        assert!(
            text.iter()
                .any(|l| l.contains(bullet) && l.contains("is important")),
            "Standalone item should have bullet: {text:#?}"
        );
    }

    // ── Column width calculation ───────────────────────────────────────

    #[test]
    fn table_columns_pad_to_widest_cell() {
        let md = "| A | B |\n|---|---|\n| short | x |\n| a very long cell | y |\n";
        let lines = render_md(md);
        let text = lines_to_text(&lines);

        // The top border should be wide enough for "a very long cell"
        let top = text.iter().find(|l| l.contains('┌')).expect("top border");
        let bottom = text
            .iter()
            .find(|l| l.contains('└'))
            .expect("bottom border");
        // Top and bottom borders should have the same length
        assert_eq!(
            top.len(),
            bottom.len(),
            "Top and bottom borders should be same width"
        );
    }

    // ── Width-aware layout tests ───────────────────────────────────────

    #[test]
    fn table_wraps_when_narrower_than_natural() {
        let md = "| Name | Description |\n|------|-------------|\n| Alice | A very long description that exceeds budget |\n";
        let blocks = render(md, Style::default(), &Highlighter::new());
        let table = match &blocks[0] {
            CardBlock::Table(t) => t,
            _ => panic!("Expected a table block"),
        };

        let inner_w = 30;
        let lines = table.layout(inner_w);
        for line in &lines {
            let w = line
                .spans
                .iter()
                .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
                .sum::<usize>();
            assert!(
                w <= inner_w,
                "Line width {w} exceeds inner_w {inner_w}: {line:?}"
            );
        }
        // Borders should be present
        assert!(
            lines
                .iter()
                .any(|l| l.spans.iter().any(|s| s.content.contains('┌')))
        );
        assert!(
            lines
                .iter()
                .any(|l| l.spans.iter().any(|s| s.content.contains('└')))
        );
        assert!(
            lines
                .iter()
                .any(|l| l.spans.iter().any(|s| s.content.contains('├')))
        );
    }

    #[test]
    fn table_natural_width_preserved_when_fits() {
        let md = "| A | B |\n|---|---|\n| 1 | 2 |\n";
        let blocks = render(md, Style::default(), &Highlighter::new());
        let table = match &blocks[0] {
            CardBlock::Table(t) => t,
            _ => panic!("Expected a table block"),
        };

        let inner_w = 80;
        let lines = table.layout(inner_w);
        let text: Vec<String> = lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        // At a wide width the header row should still be a single line
        let header = text
            .iter()
            .find(|l| l.contains("A") && l.contains("B"))
            .expect("header row");
        assert!(
            header.contains("│ A │ B │"),
            "Expected natural-width header, got: {header}"
        );
    }

    #[test]
    fn table_relayout_on_width_change() {
        let md = "| Name | Value |\n|------|-------|\n| LongWordHere | abcdef |\n";
        let blocks = render(md, Style::default(), &Highlighter::new());
        let table = match &blocks[0] {
            CardBlock::Table(t) => t,
            _ => panic!("Expected a table block"),
        };

        let wide = table.layout(80);
        let narrow = table.layout(20);

        assert_ne!(
            wide.len(),
            narrow.len(),
            "Narrow layout should produce more lines due to wrapping"
        );

        for lines in [&wide, &narrow] {
            for line in lines.iter() {
                let w = line
                    .spans
                    .iter()
                    .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
                    .sum::<usize>();
                let limit = if std::ptr::eq(lines, &wide) { 80 } else { 20 };
                assert!(w <= limit, "Line width {w} exceeds limit {limit}: {line:?}");
            }
        }
    }
}
