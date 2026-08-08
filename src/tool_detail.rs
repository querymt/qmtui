use std::path::Path;

use serde_json::Value;

use crate::domain::chat::ChatEntry;
use crate::domain::tool::{DiffPreviewSection, ShellOutputTail, ToolDetail};

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
        "shell" => ToolDetail::Shell {
            command: shell_command_display(&obj),
            workdir: string_field(&obj, "workdir"),
            output_tail: None,
        },
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
            ToolDetail::WriteFile { path, content }
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
            let is_failed_fallback = name == &fallback_name;
            if is_failed_fallback {
                *name = tool_name.to_string();
            }
            if (is_failed_fallback || matches!(detail, ToolDetail::None))
                && !matches!(start_detail, ToolDetail::None)
            {
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
            ToolDetail::Shell { output_tail, .. } if name.starts_with("shell") => {
                if let Some(tail) = shell_output_tail_from_result(parsed.as_ref(), result) {
                    *output_tail = Some(tail);
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
            ToolDetail::Edit { start_line, .. } => {
                let json_start = parsed
                    .as_ref()
                    .and_then(|value| value.get("startLineOld"))
                    .and_then(Value::as_u64)
                    .map(|start| start as usize);
                if let Some(start) = json_start
                    .or_else(|| compact_receipt_old_starts(&result_text).into_iter().next())
                {
                    *start_line = Some(start);
                }
            }
            ToolDetail::MultiEdit { sections, .. } => {
                let starts = compact_receipt_old_starts(&result_text);
                if !starts.is_empty() {
                    for (section, start) in sections.iter_mut().zip(starts) {
                        section.start_line = Some(start);
                    }
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
    let fallback_name = format!("{tool_name} (failed)");
    for entry in messages.iter_mut().rev() {
        if let ChatEntry::ToolCall {
            tool_call_id: Some(existing),
            name,
            is_error,
            ..
        } = entry
            && existing == id
            && (name == tool_name || name == &fallback_name)
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
            if let Some(rest) = line.strip_prefix("old_start=") {
                return rest.split_whitespace().next()?.parse().ok();
            }
            let old_part = line
                .strip_prefix("H ")?
                .split_whitespace()
                .find(|part| part.starts_with("old="))?;
            old_part
                .strip_prefix("old=")?
                .split(',')
                .next()?
                .parse()
                .ok()
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
    cwd.and_then(|cwd| Path::new(path).strip_prefix(cwd).ok())
        .map(|path| path.to_string_lossy().into_owned())
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegate_tool_shows_agent_and_objective() {
        let args = serde_json::json!({
            "target_agent_id": "coder",
            "objective": "List the contents of /tmp"
        });
        let detail = parse_tool_detail("delegate", Some(&args), None);
        match detail {
            ToolDetail::Summary(s) => {
                assert!(s.contains("coder"), "must contain agent name, got: {s}");
                assert!(
                    s.contains("List the contents"),
                    "must contain objective, got: {s}"
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn delegate_tool_without_agent_shows_objective_only() {
        let args = serde_json::json!({
            "objective": "Do something"
        });
        let detail = parse_tool_detail("delegate", Some(&args), None);
        match detail {
            ToolDetail::Summary(s) => {
                assert!(
                    s.contains("Do something"),
                    "must contain objective, got: {s}"
                );
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    fn assert_summary_truncates_with_ellipsis(tool_name: &str, args: serde_json::Value) {
        let detail = parse_tool_detail(tool_name, Some(&args), None);

        match detail {
            ToolDetail::Summary(s) => {
                assert!(s.ends_with(crate::ui::ELLIPSIS), "got: {s}");
                assert!(s.is_char_boundary(s.len()), "got: {s}");
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn shell_tool_preserves_full_utf8_command() {
        let command =
            "cat > check/kimi.md << 'EOF'\n# Review: feat/profiles\n\n## 🔴 Critical / High";
        let args = serde_json::json!({ "command": command });
        let detail = parse_tool_detail("shell", Some(&args), None);

        match detail {
            ToolDetail::Shell { command: got, .. } => assert_eq!(got, command),
            other => panic!("expected Shell, got: {other:?}"),
        }
    }

    #[test]
    fn shell_tool_formats_arguments_like_the_production_card() {
        let detail = parse_tool_detail(
            "shell",
            Some(&serde_json::json!({
                "command": "echo",
                "args": ["hello world", "", "safe"]
            })),
            None,
        );

        assert!(matches!(
            detail,
            ToolDetail::Shell { command, .. } if command == "echo 'hello world' '' safe"
        ));
    }

    #[test]
    fn shell_tool_uses_only_explicit_workdir() {
        let explicit = parse_tool_detail(
            "shell",
            Some(&serde_json::json!({ "command": "pwd", "workdir": "/workspace/project" })),
            Some("/session/cwd"),
        );
        match explicit {
            ToolDetail::Shell { workdir, .. } => {
                assert_eq!(workdir.as_deref(), Some("/workspace/project"))
            }
            other => panic!("expected Shell, got: {other:?}"),
        }

        let implicit = parse_tool_detail(
            "shell",
            Some(&serde_json::json!({ "command": "pwd" })),
            Some("/session/cwd"),
        );
        match implicit {
            ToolDetail::Shell { workdir, .. } => assert_eq!(workdir, None),
            other => panic!("expected Shell, got: {other:?}"),
        }
    }

    #[test]
    fn read_tool_uses_one_based_inclusive_display_range() {
        let detail = parse_tool_detail(
            "read_tool",
            Some(&serde_json::json!({
                "path": "src/acp_state.rs",
                "offset": 2134,
                "limit": 71,
            })),
            None,
        );

        match detail {
            ToolDetail::ReadTool {
                path,
                start_line,
                end_line,
            } => {
                assert_eq!(path, "src/acp_state.rs");
                assert_eq!(start_line, Some(2135));
                assert_eq!(end_line, Some(2205));
            }
            other => panic!("expected ReadTool, got: {other:?}"),
        }
    }

    #[test]
    fn read_tool_defaults_missing_offset_and_limit_for_display_range() {
        let detail = parse_tool_detail(
            "read_tool",
            Some(&serde_json::json!({
                "path": "apps/portal/priv/static/assets/css/app.css",
                "limit": 300,
            })),
            None,
        );

        match detail {
            ToolDetail::ReadTool {
                start_line,
                end_line,
                ..
            } => {
                assert_eq!(start_line, Some(1));
                assert_eq!(end_line, Some(300));
            }
            other => panic!("expected ReadTool, got: {other:?}"),
        }

        let default_limit = parse_tool_detail(
            "read_tool",
            Some(&serde_json::json!({ "path": "README.md" })),
            None,
        );
        match default_limit {
            ToolDetail::ReadTool {
                start_line,
                end_line,
                ..
            } => {
                assert_eq!(start_line, Some(1));
                assert_eq!(end_line, Some(DEFAULT_READ_TOOL_LIMIT));
            }
            other => panic!("expected ReadTool, got: {other:?}"),
        }
    }

    #[test]
    fn read_tool_parses_json_string_arguments_for_display_range() {
        let args = serde_json::Value::String(
            r#"{"path":"apps/admin/lib/admin_web/components/layouts/root.html.heex","root":".","limit":220}"#
                .into(),
        );
        let detail = parse_tool_detail("read_tool", Some(&args), None);

        match detail {
            ToolDetail::ReadTool {
                path,
                start_line,
                end_line,
            } => {
                assert_eq!(
                    path,
                    "apps/admin/lib/admin_web/components/layouts/root.html.heex"
                );
                assert_eq!(start_line, Some(1));
                assert_eq!(end_line, Some(220));
            }
            other => panic!("expected ReadTool, got: {other:?}"),
        }
    }

    #[test]
    fn shell_tool_result_keeps_last_five_output_lines() {
        let detail = parse_tool_detail(
            "shell",
            Some(&serde_json::json!({ "command": "cargo test" })),
            None,
        );
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("shell-tail".into()),
            name: "shell".into(),
            is_error: false,
            detail,
        }];
        let result = serde_json::json!({
            "exit_code": 0,
            "stdout": "out1\nout2\nout3\nout4\nout5\nout6\n",
            "stderr": "err1\nerr2\n",
        })
        .to_string();

        assert!(update_tool_detail(
            &mut messages,
            Some("shell-tail"),
            &result
        ));
        match &messages[0] {
            ChatEntry::ToolCall {
                detail:
                    ToolDetail::Shell {
                        output_tail: Some(tail),
                        ..
                    },
                ..
            } => {
                assert_eq!(tail.hidden_line_count, 3);
                assert_eq!(tail.lines, ["out4", "out5", "out6", "err1", "err2"]);
            }
            other => panic!("expected shell tail, got: {other:?}"),
        }
    }

    #[test]
    fn read_tool_result_refines_range_to_actual_line_numbers() {
        let detail = parse_tool_detail(
            "read_tool",
            Some(&serde_json::json!({
                "path": "apps/portal/lib/portal_web/components/layouts/root.html.heex",
                "limit": 260,
            })),
            None,
        );
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("read-lines".into()),
            name: "read_tool".into(),
            is_error: false,
            detail,
        }];
        let result = "<path>/projects/querymt-org/service/apps/portal/lib/portal_web/components/layouts/root.html.heex</path>\n<type>file</type>\n<content>\n00001| <!DOCTYPE html>\n00002| <html lang=\"en\">\n00027| </html>\n\n(End of file - total 27 lines)\n</content>";

        assert!(update_tool_detail(
            &mut messages,
            Some("read-lines"),
            result
        ));
        match &messages[0] {
            ChatEntry::ToolCall {
                detail:
                    ToolDetail::ReadTool {
                        start_line,
                        end_line,
                        ..
                    },
                ..
            } => {
                assert_eq!(*start_line, Some(1));
                assert_eq!(*end_line, Some(27));
            }
            other => panic!("expected read_tool detail, got: {other:?}"),
        }
    }

    #[test]
    fn read_tool_result_refines_range_from_json_text_content() {
        let detail = parse_tool_detail(
            "read_tool",
            Some(&serde_json::json!({
                "path": "apps/portal/lib/portal_web/components/core_components.ex",
                "offset": 40,
                "limit": 390,
            })),
            None,
        );
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("read-json-lines".into()),
            name: "read_tool".into(),
            is_error: false,
            detail,
        }];
        let result = serde_json::json!([{
            "type": "text",
            "text": "<path>/repo/core_components.ex</path>\n<type>file</type>\n<content>\n00041|   attr :id, :string\n00042|   slot :inner_block\n00430| end\n\n(File has more lines. Use 'offset' parameter to read beyond line 430)\n</content>"
        }])
        .to_string();

        assert!(update_tool_detail(
            &mut messages,
            Some("read-json-lines"),
            &result
        ));
        match &messages[0] {
            ChatEntry::ToolCall {
                detail:
                    ToolDetail::ReadTool {
                        start_line,
                        end_line,
                        ..
                    },
                ..
            } => {
                assert_eq!(*start_line, Some(41));
                assert_eq!(*end_line, Some(430));
            }
            other => panic!("expected read_tool detail, got: {other:?}"),
        }
    }

    #[test]
    fn edit_tool_result_uses_start_line_old() {
        let detail = parse_tool_detail(
            "edit",
            Some(&serde_json::json!({
                "filePath": "src/lib.rs",
                "oldString": "before\n",
                "newString": "after\n",
            })),
            None,
        );
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("edit-json-start".into()),
            name: "edit".into(),
            is_error: false,
            detail,
        }];

        assert!(update_tool_detail(
            &mut messages,
            Some("edit-json-start"),
            r#"{"startLineOld": 42}"#,
        ));
        match &messages[0] {
            ChatEntry::ToolCall {
                detail: ToolDetail::Edit { start_line, .. },
                ..
            } => assert_eq!(*start_line, Some(42)),
            other => panic!("expected Edit detail, got: {other:?}"),
        }
    }

    #[test]
    fn multiedit_tool_result_updates_preview_line_numbers() {
        let detail = parse_tool_detail(
            "multiedit",
            Some(&serde_json::json!({
                "filePath": "src/lib.rs",
                "edits": [
                    { "oldString": "aaa\n", "newString": "bbb\n" },
                    { "oldString": "ccc\n", "newString": "ddd\n" }
                ]
            })),
            None,
        );
        let mut messages = vec![ChatEntry::ToolCall {
            tool_call_id: Some("multi-lines".into()),
            name: "multiedit".into(),
            is_error: false,
            detail,
        }];
        let result = "OK paths=1 edits=2 added=2 deleted=2\nP src/lib.rs\nH replace old=10,1 new=10,1 +1 -1\nH replace old=20,1 new=20,1 +1 -1";

        assert!(update_tool_detail(
            &mut messages,
            Some("multi-lines"),
            result
        ));
        match &messages[0] {
            ChatEntry::ToolCall {
                detail: ToolDetail::MultiEdit { sections, .. },
                ..
            } => {
                assert_eq!(sections[0].start_line, Some(10));
                assert_eq!(sections[1].start_line, Some(20));
            }
            other => panic!("expected multiedit detail, got: {other:?}"),
        }
    }

    #[test]
    fn replace_symbol_uses_pure_summary_for_missing_and_empty_replacements() {
        let missing = parse_tool_detail("replace_symbol", Some(&serde_json::json!({})), None);
        assert!(matches!(missing, ToolDetail::Summary(summary) if summary == "symbols"));

        let empty = parse_tool_detail(
            "replace_symbol",
            Some(&serde_json::json!({ "replacements": [] })),
            None,
        );
        assert!(matches!(empty, ToolDetail::Summary(summary) if summary == "symbols"));
    }

    #[test]
    fn strip_cwd_requires_a_path_component_boundary() {
        assert_eq!(
            strip_cwd("/workspace-other/src/lib.rs", Some("/workspace")),
            "/workspace-other/src/lib.rs"
        );
    }

    #[test]
    fn replace_symbol_uses_multi_path_summary() {
        let detail = parse_tool_detail(
            "replace_symbol",
            Some(&serde_json::json!({
                "replacements": [
                    {
                        "path": "/workspace/src/one.rs",
                        "symbol": "one",
                        "newText": "fn one() {}"
                    },
                    {
                        "path": "/workspace/src/two.rs",
                        "symbol": "two",
                        "newText": "fn two() {}"
                    }
                ]
            })),
            Some("/workspace"),
        );

        assert!(matches!(
            detail,
            ToolDetail::Summary(summary) if summary == "src/one.rs (+1)"
        ));
    }

    #[test]
    fn todowrite_shows_completed_and_pending_items() {
        let detail = parse_tool_detail(
            "todowrite",
            Some(&serde_json::json!({"todos": [
                {"content": "done", "status": "completed"},
                {"content": "next", "status": "pending"},
            ]})),
            None,
        );
        assert!(matches!(
            detail,
            ToolDetail::Summary(summary) if summary == "[x] done\n[ ] next"
        ));
    }

    #[test]
    fn browse_tool_truncates_utf8_url_on_char_boundary() {
        let url = "https://example.com/docs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa🔴/page";
        let args = serde_json::json!({ "url": url });

        assert_summary_truncates_with_ellipsis("browse", args);
    }

    #[test]
    fn delegate_tool_truncates_utf8_objective_on_char_boundary() {
        let objective = "Review this objective with multibyte marker aaaaa🔴 done";
        assert!(!objective.is_char_boundary(50));
        let args = serde_json::json!({ "objective": objective });

        assert_summary_truncates_with_ellipsis("delegate", args);
    }

    #[test]
    fn index_tool_shows_short_path_from_arguments() {
        let args = serde_json::json!({"path": "/home/user/project/src/main.rs"});
        let detail = parse_tool_detail("index", Some(&args), None);
        match detail {
            ToolDetail::Summary(s) => {
                assert_eq!(s, "src/main.rs");
            }
            other => panic!("expected Summary, got: {other:?}"),
        }
    }

    #[test]
    fn failed_tool_end_marks_existing_tool_in_place() {
        let mut messages = vec![
            ChatEntry::ToolCall {
                tool_call_id: Some("tool-1".into()),
                name: "shell".into(),
                is_error: false,
                detail: ToolDetail::Summary("echo ok".into()),
            },
            ChatEntry::Assistant {
                content: "done".into(),
                thinking: None,
                message_id: None,
            },
        ];

        assert!(mark_tool_call_failed(
            &mut messages,
            Some("tool-1"),
            "shell"
        ));

        assert_eq!(messages.len(), 2, "must not append a stale failed badge");
        match &messages[0] {
            ChatEntry::ToolCall { name, is_error, .. } => {
                assert_eq!(name, "shell");
                assert!(*is_error);
            }
            other => panic!("expected ToolCall, got: {other:?}"),
        }
        assert!(matches!(messages[1], ChatEntry::Assistant { .. }));
    }

    #[test]
    fn tool_start_reconciles_failed_fallback_in_place() {
        let mut messages = vec![
            ChatEntry::Assistant {
                content: "before".into(),
                thinking: None,
                message_id: None,
            },
            ChatEntry::ToolCall {
                tool_call_id: Some("missing-start".into()),
                name: "shell (failed)".into(),
                is_error: true,
                detail: ToolDetail::None,
            },
        ];
        let detail = parse_tool_detail(
            "shell",
            Some(&serde_json::json!({
                "command": "cargo test tool_detail_tests"
            })),
            None,
        );

        assert!(reconcile_tool_call_start(
            &mut messages,
            Some("missing-start"),
            "shell",
            detail
        ));
        assert_eq!(
            messages.len(),
            2,
            "start must not append a second tool entry"
        );
        match &messages[1] {
            ChatEntry::ToolCall {
                name,
                is_error,
                detail,
                ..
            } => {
                assert_eq!(name, "shell");
                assert!(*is_error);
                assert!(matches!(
                    detail,
                    ToolDetail::Shell { command, .. } if command == "cargo test tool_detail_tests"
                ));
            }
            other => panic!("expected ToolCall, got: {other:?}"),
        }
    }
}
