use std::path::Path;

use ratatui::{
    style::Style,
    text::{Line, Span},
};

use crate::domain::tool::{
    MultiEditSection, ShellOutput, SymbolDiffSection, SymbolReplacement, ToolDetail,
};
use crate::theme::Theme;

#[derive(Clone, Copy)]
pub(super) enum DelegateToolStatus {
    Queued,
    AwaitingInput,
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy)]
pub(super) struct DelegateToolPresentation<'a> {
    pub(super) duration: &'a str,
    pub(super) status: DelegateToolStatus,
}

pub(super) struct ToolRenderInput<'a> {
    pub(super) name: &'a str,
    pub(super) is_error: bool,
    pub(super) detail: &'a ToolDetail,
    pub(super) effective_cwd: Option<&'a str>,
    pub(super) delegate: Option<DelegateToolPresentation<'a>>,
}

pub(super) fn render_tool_lines(input: ToolRenderInput<'_>) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let style = if input.is_error {
        Theme::tool_error()
    } else {
        Theme::tool_text()
    };
    let sym = if input.is_error { "x" } else { ">" };
    let tool_style = style;

    match input.detail {
        ToolDetail::Edit {
            file,
            old,
            new,
            start_line,
            ..
        } => {
            lines.push(Line::from(vec![
                Span::styled(format!("{sym} {} ", input.name), style),
                Span::styled(short_path(file).to_string(), Theme::diff_file()),
            ]));
            lines.extend(build_diff_lines(old, new, *start_line));
        }
        ToolDetail::MultiEdit {
            file,
            edit_count,
            sections,
        } => {
            lines.push(Line::from(vec![
                Span::styled(format!("{sym} {} ", input.name), style),
                Span::styled(short_path(file).to_string(), Theme::diff_file()),
                Span::styled(format!(" ({edit_count} edits)"), Theme::status_accent()),
            ]));
            lines.extend(build_multiedit_lines(sections, 6));
        }
        ToolDetail::ReplaceSymbolInput { replacements } => {
            let title = replace_symbol_title(replacements, &[], input.effective_cwd);
            lines.push(Line::from(vec![
                Span::styled(format!("{sym} {} ", input.name), style),
                Span::styled(title, Theme::diff_file()),
            ]));
        }
        ToolDetail::ReplaceSymbolDiff {
            replacements,
            sections,
        } => {
            let title = replace_symbol_title(replacements, sections, input.effective_cwd);
            lines.push(Line::from(vec![
                Span::styled(format!("{sym} {} ", input.name), style),
                Span::styled(title, Theme::diff_file()),
            ]));
            lines.extend(build_symbol_diff_lines(sections, 4));
        }
        ToolDetail::Shell {
            command,
            arguments,
            workdir,
            output,
        } => {
            let mut spans = vec![Span::styled(format!("{sym} {}", input.name), style)];
            if let Some(workdir) = workdir.as_deref().filter(|s| !s.trim().is_empty()) {
                spans.push(Span::styled("@", tool_style));
                spans.push(Span::styled(workdir.to_string(), Theme::diff_file()));
            }
            lines.push(Line::from(spans));
            let command = shell_command_display(command, arguments);
            lines.extend(build_shell_lines(&command, output.as_ref()));
        }
        ToolDetail::ReadTool {
            path,
            start_line,
            end_line,
        } => {
            let mut spans = vec![
                Span::styled(format!("{sym} {} ", input.name), style),
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
            lines.push(Line::from(spans));
        }
        ToolDetail::WriteFile { path, content } => {
            lines.push(Line::from(vec![
                Span::styled(format!("{sym} {} ", input.name), style),
                Span::styled(short_path(path).to_string(), Theme::diff_file()),
            ]));
            lines.extend(build_write_lines(content));
        }
        ToolDetail::Generic {
            input: value,
            result,
        } => {
            let (summary, output) = match (value.as_deref(), result.as_deref()) {
                (Some(value), result) => (value, result),
                (None, Some(result)) => (result, None),
                (None, None) => ("", None),
            };
            push_summary_lines(&mut lines, sym, input.name, style, summary, output);
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
            push_summary_lines(&mut lines, sym, input.name, style, &summary, None);
        }
        ToolDetail::Glob { pattern, path } => {
            let summary = if path.is_empty() {
                pattern.clone()
            } else {
                format!("{pattern} in {}", short_path(path))
            };
            push_summary_lines(&mut lines, sym, input.name, style, &summary, None);
        }
        ToolDetail::List { path } => {
            let summary = if path.is_empty() {
                "."
            } else {
                short_path(path)
            };
            push_summary_lines(&mut lines, sym, input.name, style, summary, None);
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
            push_summary_lines(&mut lines, sym, input.name, style, &summary, None);
        }
        ToolDetail::DeleteFile { path } => {
            push_summary_lines(&mut lines, sym, input.name, style, short_path(path), None)
        }
        ToolDetail::Browse { url } => {
            let summary = truncate_summary(url, 60);
            push_summary_lines(&mut lines, sym, input.name, style, &summary, None);
        }
        ToolDetail::Todo { items } => {
            lines.push(Line::from(Span::styled(
                format!("{sym} {}", input.name),
                style,
            )));
            for item in items {
                let check = if item.status == "completed" { "x" } else { " " };
                lines.push(Line::from(Span::styled(
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
                push_summary_lines(&mut lines, sym, input.name, style, &objective, None);
            } else {
                let delegate = input.delegate.unwrap_or(DelegateToolPresentation {
                    duration: "",
                    status: DelegateToolStatus::Queued,
                });
                let label = if delegate.duration.is_empty() {
                    format!("({target_agent_id})")
                } else {
                    format!("({target_agent_id}{})", delegate.duration)
                };
                let (delegate_status, delegate_status_style) = match delegate.status {
                    DelegateToolStatus::AwaitingInput => {
                        ("awaiting input", Theme::mode_badge("plan"))
                    }
                    DelegateToolStatus::Running => ("running", Theme::status_accent()),
                    DelegateToolStatus::Done => ("done", Theme::status()),
                    DelegateToolStatus::Failed => ("failed", Theme::tool_error()),
                    DelegateToolStatus::Cancelled => ("cancelled", Theme::status()),
                    DelegateToolStatus::Queued => ("queued", Theme::status()),
                };
                let mut spans = vec![Span::styled(format!("{sym} {} ", input.name), style)];
                spans.push(Span::styled(format!("{label} "), Theme::status_accent()));
                spans.push(Span::styled(
                    format!("{delegate_status} "),
                    delegate_status_style,
                ));
                if !objective.is_empty() {
                    spans.push(Span::styled(objective, Theme::diff_file()));
                }
                lines.push(Line::from(spans));
            }
        }
        ToolDetail::LanguageQuery { action, uri } => {
            let summary = format!("{action} {}", short_path(uri));
            push_summary_lines(&mut lines, sym, input.name, style, &summary, None);
        }
        ToolDetail::Question { .. } => {
            push_summary_lines(&mut lines, sym, input.name, style, "asking...", None)
        }
        ToolDetail::ApplyPatch { .. } => {
            push_summary_lines(&mut lines, sym, input.name, style, "patch", None)
        }
        ToolDetail::None => lines.push(Line::from(Span::styled(
            format!("{sym} {}", input.name),
            style,
        ))),
    }

    lines
}

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

fn build_shell_lines(command: &str, output: Option<&ShellOutput>) -> Vec<Line<'static>> {
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

fn build_diff_lines(old: &str, new: &str, start_line: Option<usize>) -> Vec<Line<'static>> {
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

fn build_write_lines(content: &str) -> Vec<Line<'static>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::tool::{IndexMetadata, SearchResultCounts};

    fn render(name: &str, detail: &ToolDetail) -> Vec<Line<'static>> {
        render_tool_lines(ToolRenderInput {
            name,
            is_error: false,
            detail,
            effective_cwd: None,
            delegate: None,
        })
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    #[test]
    fn shell_renderer_shows_stderr_without_stdout() {
        let detail = ToolDetail::Shell {
            command: "failing".into(),
            arguments: Vec::new(),
            workdir: None,
            output: Some(ShellOutput {
                stdout: String::new(),
                stderr: "permission denied".into(),
                preceding_line_count: 0,
            }),
        };
        let text = render("shell", &detail)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>();

        assert!(text.iter().any(|line| line.contains("permission denied")));
    }

    #[test]
    fn multiedit_renderer_uses_compact_sequential_section_labels() {
        let detail = ToolDetail::MultiEdit {
            file: "src/lib.rs".into(),
            edit_count: 2,
            sections: vec![
                MultiEditSection {
                    edit_index: 1,
                    replace_all: false,
                    old: "one".into(),
                    new: "ONE".into(),
                    start_line: None,
                },
                MultiEditSection {
                    edit_index: 2,
                    replace_all: true,
                    old: "two".into(),
                    new: "TWO".into(),
                    start_line: None,
                },
            ],
        };
        let text = render("multiedit", &detail)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("(2 edits)"));
        assert!(text.contains("edit 1"));
        assert!(text.contains("edit 2 (all)"));
        assert!(!text.contains("edit 3"));
    }

    #[test]
    fn semantic_tool_renderer_accepts_narrow_inputs() {
        let detail = ToolDetail::Generic {
            input: Some("request".into()),
            result: Some("response".into()),
        };
        let lines = render("generic", &detail);
        let text = lines.iter().map(line_text).collect::<Vec<_>>();

        assert_eq!(text, ["> generic request", "  response"]);
    }

    #[test]
    fn index_tool_call_metadata_suffix_uses_accent_color() {
        let detail = ToolDetail::Index {
            path: "src/(generated)/main.rs".into(),
            metadata: Some(IndexMetadata {
                language: "rust".into(),
                imports: 2,
                functions: 1,
            }),
        };
        let lines = render("index", &detail);
        let line = &lines[0];

        assert_eq!(line.spans.len(), 3);
        assert_eq!(line.spans[1].content, "src/(generated)/main.rs");
        assert_eq!(line.spans[1].style.fg, Theme::diff_file().fg);
        assert_eq!(line.spans[2].content, " (rust, 2 imports, 1 functions)");
        assert_eq!(line.spans[2].style.fg, Theme::status_accent().fg);
    }

    #[test]
    fn search_text_tool_call_counts_suffix_uses_accent_color() {
        let detail = ToolDetail::SearchText {
            pattern: "needle".into(),
            path: String::new(),
            include: "*.rs".into(),
            counts: Some(SearchResultCounts {
                files: 5,
                matches: 28,
            }),
        };
        let lines = render("search_text", &detail);
        let line = &lines[0];

        assert_eq!(line.spans.len(), 3);
        assert_eq!(line.spans[1].content, "\"needle\" *.rs");
        assert_eq!(line.spans[1].style.fg, Theme::diff_file().fg);
        assert_eq!(line.spans[2].content, " (5 files, 28 matches)");
        assert_eq!(line.spans[2].style.fg, Theme::status_accent().fg);
    }

    #[test]
    fn read_tool_call_range_uses_split_styles() {
        let detail = ToolDetail::ReadTool {
            path: "src/acp_state.rs".into(),
            start_line: Some(2135),
            end_line: Some(2205),
        };
        let lines = render("read_tool", &detail);
        let line = &lines[0];

        assert_eq!(line_text(line), "> read_tool src/acp_state.rs:2135-2205");
        assert_eq!(line.spans.len(), 4);
        assert_eq!(line.spans[1].style.fg, Theme::diff_file().fg);
        assert_eq!(line.spans[2].content, ":");
        assert_eq!(line.spans[2].style.fg, Theme::tool_text().fg);
        assert_eq!(line.spans[3].style.fg, Theme::status_accent().fg);
    }

    #[test]
    fn tool_summary_utf8_truncation_is_render_owned_and_char_safe() {
        let browse_url = format!("https://example.test/{}tail", "界".repeat(50));
        let objective = format!("{}tail", "界".repeat(50));
        let expected_url = format!("{}…", browse_url.chars().take(59).collect::<String>());
        let expected_objective = format!("{}…", objective.chars().take(49).collect::<String>());
        let browse = ToolDetail::Browse { url: browse_url };
        let delegate = ToolDetail::Delegate {
            target_agent_id: String::new(),
            objective,
        };

        assert_eq!(
            [
                line_text(&render("browse", &browse)[0]),
                line_text(&render("delegate", &delegate)[0]),
            ],
            [
                format!("> browse {expected_url}"),
                format!("> delegate {expected_objective}"),
            ]
        );
    }
}
