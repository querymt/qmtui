use serde_json::Value;

use crate::app::{ChatEntry, DiffPreviewSection, ShellOutputTail, ToolDetail};
use crate::ui::{
    build_diff_lines, build_sectioned_diff_lines, build_shell_lines, build_write_lines,
};

const DEFAULT_READ_TOOL_LIMIT: u64 = 2000;

pub(crate) fn parse_tool_detail(
    tool_name: &str,
    arguments: Option<&Value>,
    cwd: Option<&str>,
) -> ToolDetail {
    let Some(args) = arguments else {
        return ToolDetail::None;
    };
    let obj = normalize_args(args);

    match tool_name {
        "shell" => {
            let command = shell_command_display(&obj);
            let workdir = string_field(&obj, "workdir");
            let cached_lines = build_shell_lines(&command, workdir.as_deref(), None);
            ToolDetail::Shell {
                command,
                workdir,
                output_tail: None,
                cached_lines,
            }
        }
        "read_tool" => {
            let path = string_field(&obj, "path").unwrap_or_default();
            let offset = obj.get("offset").and_then(Value::as_u64).unwrap_or(0);
            let limit = obj
                .get("limit")
                .and_then(Value::as_u64)
                .filter(|limit| *limit > 0)
                .unwrap_or(DEFAULT_READ_TOOL_LIMIT);
            ToolDetail::ReadTool {
                path,
                start_line: Some(offset.saturating_add(1)),
                end_line: Some(offset.saturating_add(limit)),
            }
        }
        "write_file" => {
            let path = string_field(&obj, "path").unwrap_or_default();
            let content = string_field(&obj, "content").unwrap_or_default();
            ToolDetail::WriteFile {
                path,
                cached_lines: build_write_lines(&content),
                content,
            }
        }
        "edit" => {
            let file = string_field(&obj, "filePath")
                .or_else(|| string_field(&obj, "file_path"))
                .unwrap_or_default();
            let old = string_field(&obj, "oldString")
                .or_else(|| string_field(&obj, "old_string"))
                .unwrap_or_default();
            let new = string_field(&obj, "newString")
                .or_else(|| string_field(&obj, "new_string"))
                .unwrap_or_default();
            ToolDetail::Edit {
                file,
                cached_lines: build_diff_lines(&old, &new, None),
                old,
                new,
                start_line: None,
            }
        }
        "multiedit" => {
            let file = string_field(&obj, "filePath")
                .or_else(|| string_field(&obj, "file_path"))
                .unwrap_or_default();
            let sections = obj
                .get("edits")
                .and_then(Value::as_array)
                .map(|edits| {
                    edits
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, edit)| {
                            let old = string_field(edit, "oldString")
                                .or_else(|| string_field(edit, "old_string"))?;
                            let new = string_field(edit, "newString")
                                .or_else(|| string_field(edit, "new_string"))?;
                            let suffix = if edit
                                .get("replaceAll")
                                .or_else(|| edit.get("replace_all"))
                                .and_then(Value::as_bool)
                                .unwrap_or(false)
                            {
                                " (all)"
                            } else {
                                ""
                            };
                            Some(DiffPreviewSection {
                                header: format!("edit {}{}", idx + 1, suffix),
                                old,
                                new,
                                start_line: None,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            ToolDetail::MultiEdit {
                file,
                edit_count: sections.len(),
                cached_lines: build_sectioned_diff_lines(&sections, 6),
                sections,
            }
        }
        "search_text" => {
            let pattern = string_field(&obj, "pattern").unwrap_or_default();
            let path = string_field(&obj, "path").unwrap_or_default();
            let include = string_field(&obj, "include").unwrap_or_default();
            let location = if !include.is_empty() {
                include
            } else if !path.is_empty() {
                short_path(&path).to_string()
            } else {
                ".".into()
            };
            ToolDetail::Summary(format!("\"{}\" {}", pattern, location))
        }
        "glob" => summary_path_arg(&obj, "pattern", "path"),
        "ls" | "index" => ToolDetail::Summary(
            string_field(&obj, "path")
                .map(|path| short_path(&path).to_string())
                .unwrap_or_else(|| ".".into()),
        ),
        "delete_file" => ToolDetail::Summary(
            string_field(&obj, "path")
                .map(|path| short_path(&path).to_string())
                .unwrap_or_default(),
        ),
        "browse" | "web_fetch" => ToolDetail::Summary(
            string_field(&obj, "url")
                .map(|url| truncate_summary(&url, 60))
                .unwrap_or_default(),
        ),
        "todowrite" => todo_summary(&obj),
        "delegate" => delegate_summary(&obj),
        "language_query" => {
            let action = string_field(&obj, "action").unwrap_or_default();
            let uri = string_field(&obj, "uri").unwrap_or_default();
            ToolDetail::Summary(format!("{} {}", action, short_path(&uri)))
        }
        "question" => ToolDetail::Summary("asking...".into()),
        "apply_patch" => ToolDetail::Summary("patch".into()),
        "replace_symbol" => ToolDetail::Summary(replace_symbol_title(&obj, cwd)),
        _ => ToolDetail::None,
    }
}

pub(crate) fn reconcile_tool_call_start(
    messages: &mut [ChatEntry],
    tool_call_id: Option<&str>,
    tool_name: &str,
    start_detail: ToolDetail,
) -> bool {
    let Some(id) = tool_call_id else { return false };
    let fallback_name = format!("{tool_name} (failed)");
    for entry in messages.iter_mut().rev() {
        if let ChatEntry::ToolCall {
            tool_call_id: Some(existing),
            name,
            detail,
            ..
        } = entry
        {
            if existing != id {
                continue;
            }
            if name == &fallback_name {
                *name = tool_name.to_string();
            }
            if matches!(detail, ToolDetail::None) && !matches!(start_detail, ToolDetail::None) {
                *detail = start_detail;
            }
            return true;
        }
    }
    false
}

pub(crate) fn update_tool_detail(
    messages: &mut [ChatEntry],
    tool_call_id: Option<&str>,
    result: &str,
) -> bool {
    let Some(id) = tool_call_id else { return false };
    let parsed = serde_json::from_str::<Value>(result).ok();
    let result_text = tool_result_text(parsed.as_ref(), result);

    for entry in messages.iter_mut().rev() {
        let ChatEntry::ToolCall {
            tool_call_id: Some(existing),
            name,
            detail,
            ..
        } = entry
        else {
            continue;
        };
        if existing != id {
            continue;
        }

        match detail {
            ToolDetail::Shell {
                command,
                workdir,
                output_tail,
                cached_lines,
            } if name.starts_with("shell") => {
                if let Some(tail) = shell_output_tail_from_result(parsed.as_ref(), result) {
                    *output_tail = Some(tail);
                    *cached_lines =
                        build_shell_lines(command, workdir.as_deref(), output_tail.as_ref());
                }
            }
            ToolDetail::ReadTool {
                start_line,
                end_line,
                ..
            } if name == "read_tool" => {
                if let Some((first, last)) = read_tool_result_line_range(&result_text) {
                    *start_line = Some(first);
                    *end_line = Some(last);
                }
            }
            ToolDetail::Edit {
                old,
                new,
                start_line,
                cached_lines,
                ..
            } => {
                if let Some(start) = compact_receipt_old_starts(&result_text).into_iter().next() {
                    *start_line = Some(start);
                    *cached_lines = build_diff_lines(old, new, Some(start));
                }
            }
            ToolDetail::MultiEdit {
                sections,
                cached_lines,
                ..
            } => {
                let starts = compact_receipt_old_starts(&result_text);
                if !starts.is_empty() {
                    for (section, start) in sections.iter_mut().zip(starts) {
                        section.start_line = Some(start);
                    }
                    *cached_lines = build_sectioned_diff_lines(sections, 6);
                }
            }
            _ => {}
        }
        return true;
    }
    false
}

pub(crate) fn mark_tool_call_failed(
    messages: &mut [ChatEntry],
    tool_call_id: Option<&str>,
    tool_name: &str,
) -> bool {
    let Some(id) = tool_call_id else { return false };
    for entry in messages.iter_mut().rev() {
        if let ChatEntry::ToolCall {
            tool_call_id: Some(existing),
            name,
            is_error,
            ..
        } = entry
            && existing == id
            && name == tool_name
            && !*is_error
        {
            *is_error = true;
            return true;
        }
    }
    false
}

fn normalize_args(args: &Value) -> Value {
    args.as_str()
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .unwrap_or_else(|| args.clone())
}

fn string_field(obj: &Value, key: &str) -> Option<String> {
    obj.get(key).and_then(|v| match v {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => Some(other.to_string()),
    })
}

fn shell_command_display(obj: &Value) -> String {
    let command = string_field(obj, "command").unwrap_or_default();
    let Some(args) = obj.get("args").and_then(Value::as_array) else {
        return command;
    };
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(shell_quote_arg(&command));
    parts.extend(args.iter().filter_map(Value::as_str).map(shell_quote_arg));
    if parts.len() == 1 {
        command
    } else {
        parts.join(" ")
    }
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

fn shell_output_tail_from_result(parsed: Option<&Value>, raw: &str) -> Option<ShellOutputTail> {
    let (stdout, stderr) = parsed
        .and_then(|obj| {
            Some((
                obj.get("stdout")?.as_str().unwrap_or_default().to_string(),
                obj.get("stderr")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ))
        })
        .unwrap_or_else(|| (raw.to_string(), String::new()));
    let lines = stdout
        .lines()
        .chain(stderr.lines())
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return None;
    }
    let keep = 5;
    let hidden_line_count = lines.len().saturating_sub(keep);
    Some(ShellOutputTail {
        lines: lines.into_iter().skip(hidden_line_count).collect(),
        hidden_line_count,
    })
}

fn tool_result_text(parsed: Option<&Value>, raw: &str) -> String {
    parsed
        .map(content_to_string)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| raw.to_string())
}

fn content_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Object(obj) => {
            if let Some(text) = obj.get("text").and_then(Value::as_str) {
                return text.to_string();
            }
            if let Some(content) = obj.get("content") {
                return content_to_string(content);
            }
            serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
        }
        Value::Array(arr) => arr
            .iter()
            .map(content_to_string)
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        _ => v.to_string(),
    }
}

fn read_tool_result_line_range(text: &str) -> Option<(u64, u64)> {
    let mut first = None;
    let mut last = None;
    for line in text.lines() {
        let Some((num, _)) = line.split_once('|') else {
            continue;
        };
        let Ok(n) = num.trim().parse::<u64>() else {
            continue;
        };
        first.get_or_insert(n);
        last = Some(n);
    }
    first.zip(last)
}

fn compact_receipt_old_starts(text: &str) -> Vec<usize> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix("old_start=")?;
            rest.split_whitespace().next()?.parse().ok()
        })
        .collect()
}

fn summary_path_arg(obj: &Value, main_key: &str, path_key: &str) -> ToolDetail {
    let main = string_field(obj, main_key).unwrap_or_default();
    let path = string_field(obj, path_key).unwrap_or_default();
    if path.is_empty() {
        ToolDetail::Summary(main)
    } else {
        ToolDetail::Summary(format!("{} in {}", main, short_path(&path)))
    }
}

fn todo_summary(obj: &Value) -> ToolDetail {
    let Some(todos) = obj.get("todos").and_then(Value::as_array) else {
        return ToolDetail::None;
    };
    let lines = todos
        .iter()
        .filter_map(|todo| {
            let content = todo.get("content").and_then(Value::as_str)?;
            let status = todo
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending");
            let check = if status == "completed" { "x" } else { " " };
            Some(format!("[{check}] {content}"))
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        ToolDetail::None
    } else {
        ToolDetail::Summary(lines.join("\n"))
    }
}

fn delegate_summary(obj: &Value) -> ToolDetail {
    let agent = string_field(obj, "target_agent_id").unwrap_or_default();
    let objective = string_field(obj, "objective").unwrap_or_default();
    let objective = truncate_summary(&objective, 50);
    if agent.is_empty() {
        ToolDetail::Summary(objective)
    } else {
        ToolDetail::Summary(format!("({agent}) {objective}"))
    }
}

fn replace_symbol_title(obj: &Value, cwd: Option<&str>) -> String {
    let Some(replacements) = obj.get("replacements").and_then(Value::as_array) else {
        return "symbols".into();
    };
    let mut files = replacements
        .iter()
        .filter_map(|replacement| string_field(replacement, "path"))
        .map(|path| strip_cwd(&path, cwd))
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    match files.as_slice() {
        [] => "symbols".into(),
        [one] => short_path(one).to_string(),
        [first, ..] => format!("{} (+{})", short_path(first), files.len() - 1),
    }
}

fn strip_cwd(path: &str, cwd: Option<&str>) -> String {
    cwd.and_then(|cwd| path.strip_prefix(cwd))
        .map(|path| path.trim_start_matches('/').to_string())
        .unwrap_or_else(|| path.to_string())
}

fn short_path(path: &str) -> &str {
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

fn truncate_summary(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}
