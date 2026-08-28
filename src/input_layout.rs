use unicode_width::UnicodeWidthChar;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InputVisualRow {
    pub(crate) text: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) columns: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InputVisualLayout {
    pub(crate) rows: Vec<InputVisualRow>,
    pub(crate) cursor_row: usize,
    pub(crate) cursor_col: usize,
    pub(crate) cursor_text_col: usize,
}

impl InputVisualLayout {
    pub(crate) fn total_rows(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn cursor_offset_for_row_col(&self, row: usize, preferred_col: usize) -> usize {
        let Some(row) = self.rows.get(row) else {
            return 0;
        };
        let mut best = row.end;
        for (col, offset) in &row.columns {
            if *col <= preferred_col {
                best = *offset;
            } else {
                break;
            }
        }
        best
    }
}

pub(crate) fn build_input_visual_layout(
    input: &str,
    input_cursor: usize,
    line_width: usize,
    prefix_width: usize,
) -> InputVisualLayout {
    let line_width = line_width.max(1);
    let mut rows: Vec<InputVisualRow> = Vec::new();
    let mut row_text = String::new();
    let mut row_columns = vec![(0usize, 0usize)];
    let mut row_start = 0usize;
    let mut row_end = 0usize;
    let mut col = prefix_width;
    let mut text_col = 0usize;
    let mut cursor_row = 0usize;
    let mut cursor_col = prefix_width;
    let mut cursor_text_col = 0usize;
    let mut cursor_found = input_cursor == 0;

    let row_prefix_width = |rows_len: usize| if rows_len == 0 { prefix_width } else { 0 };

    if cursor_found {
        cursor_row = 0;
        cursor_col = prefix_width;
        cursor_text_col = 0;
    }

    let finish_row = |rows: &mut Vec<InputVisualRow>,
                      row_text: &mut String,
                      row_columns: &mut Vec<(usize, usize)>,
                      row_start: &mut usize,
                      row_end: &mut usize,
                      next_start: usize| {
        rows.push(InputVisualRow {
            text: std::mem::take(row_text),
            start: *row_start,
            end: *row_end,
            columns: std::mem::take(row_columns),
        });
        *row_columns = vec![(0, next_start)];
        *row_start = next_start;
        *row_end = next_start;
    };

    for (byte_idx, ch) in input.char_indices() {
        if !cursor_found && byte_idx == input_cursor {
            cursor_row = rows.len();
            cursor_col = col;
            cursor_text_col = text_col;
            cursor_found = true;
        }

        if ch == '\n' {
            row_end = byte_idx;
            finish_row(
                &mut rows,
                &mut row_text,
                &mut row_columns,
                &mut row_start,
                &mut row_end,
                byte_idx + ch.len_utf8(),
            );
            col = row_prefix_width(rows.len());
            text_col = 0;
            continue;
        }

        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if ch_width > 0 && col + ch_width > line_width {
            finish_row(
                &mut rows,
                &mut row_text,
                &mut row_columns,
                &mut row_start,
                &mut row_end,
                byte_idx,
            );
            col = row_prefix_width(rows.len());
            text_col = 0;
            if !cursor_found && byte_idx == input_cursor {
                cursor_row = rows.len();
                cursor_col = col;
                cursor_text_col = 0;
                cursor_found = true;
            }
        }

        row_text.push(ch);
        col += ch_width;
        text_col += ch_width;
        row_end = byte_idx + ch.len_utf8();
        row_columns.push((text_col, row_end));
    }

    if !cursor_found && input_cursor == input.len() {
        cursor_row = rows.len();
        cursor_col = col;
        cursor_text_col = text_col;
    }

    rows.push(InputVisualRow {
        text: row_text,
        start: row_start,
        end: row_end,
        columns: row_columns,
    });

    InputVisualLayout {
        rows,
        cursor_row,
        cursor_col,
        cursor_text_col,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_texts(layout: &InputVisualLayout) -> Vec<&str> {
        layout.rows.iter().map(|row| row.text.as_str()).collect()
    }

    #[test]
    fn unicode_widths_combining_scalars_and_utf8_cursor_offsets_are_preserved() {
        let input = "a界e\u{301}b";
        let cursor = "a界e\u{301}".len();
        let layout = build_input_visual_layout(input, cursor, 4, 1);

        assert_eq!(row_texts(&layout), vec!["a界", "e\u{301}b"]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (1, 1));
        assert_eq!(layout.cursor_text_col, 1);
        assert_eq!(
            layout.rows[1].columns,
            vec![(0, "a界".len()), (1, 5), (1, cursor), (2, input.len())]
        );
    }

    #[test]
    fn hard_consecutive_and_trailing_newlines_keep_empty_rows_and_byte_boundaries() {
        let input = "a\n\n";
        let on_first_newline = build_input_visual_layout(input, 1, 8, 2);
        let after_first_newline = build_input_visual_layout(input, 2, 8, 2);
        let at_end = build_input_visual_layout(input, input.len(), 8, 2);

        assert_eq!(row_texts(&at_end), vec!["a", "", ""]);
        assert_eq!(
            (on_first_newline.cursor_row, on_first_newline.cursor_col),
            (0, 3)
        );
        assert_eq!(
            (
                after_first_newline.cursor_row,
                after_first_newline.cursor_col
            ),
            (1, 0)
        );
        assert_eq!((at_end.cursor_row, at_end.cursor_col), (2, 0));
        assert_eq!((at_end.rows[2].start, at_end.rows[2].end), (3, 3));
    }

    #[test]
    fn cursor_at_wrap_trigger_stays_on_previous_row_end() {
        let layout = build_input_visual_layout("abcd", 2, 4, 2);

        assert_eq!(row_texts(&layout), vec!["ab", "cd"]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (0, 4));
        assert_eq!(layout.cursor_text_col, 2);
    }

    #[test]
    fn zero_and_one_width_preserve_prefix_only_first_row() {
        for width in [0, 1] {
            let layout = build_input_visual_layout("a", 1, width, 2);
            assert_eq!(row_texts(&layout), vec!["", "a"]);
            assert_eq!((layout.cursor_row, layout.cursor_col), (1, 1));
            assert_eq!((layout.rows[0].start, layout.rows[0].end), (0, 0));
        }
    }

    #[test]
    fn character_wider_than_line_wraps_once_then_is_inserted() {
        let layout = build_input_visual_layout("界", "界".len(), 1, 0);

        assert_eq!(row_texts(&layout), vec!["", "界"]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (1, 2));
    }

    #[test]
    fn empty_input_always_returns_one_prefixed_row() {
        let layout = build_input_visual_layout("", 0, 0, 2);

        assert_eq!(layout.total_rows(), 1);
        assert_eq!(row_texts(&layout), vec![""]);
        assert_eq!((layout.cursor_row, layout.cursor_col), (0, 2));
    }

    #[test]
    fn target_column_uses_last_display_column_not_exceeding_preference() {
        let layout = build_input_visual_layout("a界b", 0, 20, 2);

        assert_eq!(layout.cursor_offset_for_row_col(0, 0), 0);
        assert_eq!(layout.cursor_offset_for_row_col(0, 2), 1);
        assert_eq!(layout.cursor_offset_for_row_col(0, 3), "a界".len());
        assert_eq!(
            layout.cursor_offset_for_row_col(0, usize::MAX),
            "a界b".len()
        );
        assert_eq!(layout.cursor_offset_for_row_col(1, 0), 0);
    }
}
