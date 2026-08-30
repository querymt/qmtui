use serde_json::Value;

use crate::domain::chat::ChatEntry;
use crate::domain::tool::{MultiEditSection, ShellOutput, SymbolReplacement, TodoItem, ToolDetail};

const DEFAULT_READ_TOOL_LIMIT: u64 = 2000;

pub(crate) fn parse_tool_detail(tool_name: &str, arguments: Option<&Value>) -> ToolDetail {
    let Some(args) = arguments else {
        return ToolDetail::None;
    };
    let obj = normalize_args(args);

    match tool_name {
        "shell" => ToolDetail::Shell {
            command: string_field(&obj, "command").unwrap_or_default(),
            arguments: obj
                .get("args")
                .and_then(Value::as_array)
                .map(|args| {
                    args.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            workdir: string_field(&obj, "workdir"),
            output: None,
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
        "write_file" => ToolDetail::WriteFile {
            path: string_field(&obj, "path").unwrap_or_default(),
            content: string_field(&obj, "content").unwrap_or_default(),
        },
        "edit" => ToolDetail::Edit {
            file: string_field(&obj, "filePath")
                .or_else(|| string_field(&obj, "file_path"))
                .unwrap_or_default(),
            old: string_field(&obj, "oldString")
                .or_else(|| string_field(&obj, "old_string"))
                .unwrap_or_default(),
            new: string_field(&obj, "newString")
                .or_else(|| string_field(&obj, "new_string"))
                .unwrap_or_default(),
            replace_all: obj
                .get("replaceAll")
                .or_else(|| obj.get("replace_all"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            start_line: None,
        },
        "multiedit" => {
            let sections = obj
                .get("edits")
                .and_then(Value::as_array)
                .map(|edits| {
                    edits
                        .iter()
                        .enumerate()
                        .filter_map(|(index, edit)| {
                            Some(MultiEditSection {
                                edit_index: index + 1,
                                replace_all: edit
                                    .get("replaceAll")
                                    .or_else(|| edit.get("replace_all"))
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false),
                                old: string_field(edit, "oldString")
                                    .or_else(|| string_field(edit, "old_string"))?,
                                new: string_field(edit, "newString")
                                    .or_else(|| string_field(edit, "new_string"))?,
                                start_line: None,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            ToolDetail::MultiEdit {
                file: string_field(&obj, "filePath")
                    .or_else(|| string_field(&obj, "file_path"))
                    .unwrap_or_default(),
                edit_count: sections.len(),
                sections,
            }
        }
        "search_text" => ToolDetail::SearchText {
            pattern: string_field(&obj, "pattern").unwrap_or_default(),
            path: string_field(&obj, "path").unwrap_or_default(),
            include: string_field(&obj, "include").unwrap_or_default(),
            counts: None,
        },
        "glob" => ToolDetail::Glob {
            pattern: string_field(&obj, "pattern").unwrap_or_default(),
            path: string_field(&obj, "path").unwrap_or_default(),
        },
        "ls" => ToolDetail::List {
            path: string_field(&obj, "path").unwrap_or_default(),
        },
        "index" => ToolDetail::Index {
            path: string_field(&obj, "path").unwrap_or_default(),
            metadata: None,
        },
        "delete_file" => ToolDetail::DeleteFile {
            path: string_field(&obj, "path").unwrap_or_default(),
        },
        "browse" | "web_fetch" => ToolDetail::Browse {
            url: string_field(&obj, "url").unwrap_or_default(),
        },
        "todowrite" => todo_detail(&obj),
        "delegate" => ToolDetail::Delegate {
            target_agent_id: string_field(&obj, "target_agent_id").unwrap_or_default(),
            objective: string_field(&obj, "objective").unwrap_or_default(),
        },
        "language_query" => ToolDetail::LanguageQuery {
            action: string_field(&obj, "action").unwrap_or_default(),
            uri: string_field(&obj, "uri").unwrap_or_default(),
        },
        "question" => ToolDetail::Question {
            prompt: string_field(&obj, "prompt").unwrap_or_default(),
        },
        "apply_patch" => ToolDetail::ApplyPatch {
            patch: string_field(&obj, "patch").unwrap_or_default(),
        },
        "replace_symbol" => ToolDetail::ReplaceSymbolInput {
            replacements: symbol_replacements(&obj),
        },
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
            ToolDetail::Shell { output, .. } if name.starts_with("shell") => {
                if let Some(result_output) = shell_output_from_result(parsed.as_ref(), result) {
                    *output = Some(result_output);
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

fn shell_output_from_result(parsed: Option<&Value>, raw: &str) -> Option<ShellOutput> {
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
    if stdout.is_empty() && stderr.is_empty() {
        None
    } else {
        Some(ShellOutput {
            stdout,
            stderr,
            preceding_line_count: 0,
        })
    }
}

fn todo_detail(obj: &Value) -> ToolDetail {
    let Some(todos) = obj.get("todos").and_then(Value::as_array) else {
        return ToolDetail::None;
    };
    let items = todos
        .iter()
        .filter_map(|todo| {
            Some(TodoItem {
                content: todo.get("content")?.as_str()?.to_string(),
                status: todo
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("pending")
                    .to_string(),
            })
        })
        .collect::<Vec<_>>();
    if items.is_empty() {
        ToolDetail::None
    } else {
        ToolDetail::Todo { items }
    }
}

fn symbol_replacements(obj: &Value) -> Vec<SymbolReplacement> {
    obj.get("replacements")
        .and_then(Value::as_array)
        .map(|replacements| {
            replacements
                .iter()
                .map(|replacement| SymbolReplacement {
                    path: string_field(replacement, "path").unwrap_or_default(),
                    symbol: string_field(replacement, "symbol").unwrap_or_default(),
                    new_text: string_field(replacement, "newText")
                        .or_else(|| string_field(replacement, "new_text"))
                        .unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(tool_name: &str, arguments: serde_json::Value) -> ToolDetail {
        parse_tool_detail(tool_name, Some(&arguments))
    }

    fn tool_call(id: &str, name: &str, detail: ToolDetail) -> ChatEntry {
        ChatEntry::ToolCall {
            tool_call_id: Some(id.into()),
            name: name.into(),
            is_error: false,
            detail,
        }
    }

    #[test]
    fn delegate_and_browse_preserve_raw_untruncated_values() {
        let objective = format!("{}tail", "界".repeat(50));
        assert_eq!(
            parse(
                "delegate",
                serde_json::json!({
                    "target_agent_id": "coder",
                    "objective": objective,
                }),
            ),
            ToolDetail::Delegate {
                target_agent_id: "coder".into(),
                objective: objective.clone(),
            }
        );

        let url = format!("https://example.test/{}tail", "界".repeat(50));
        assert_eq!(
            parse("browse", serde_json::json!({ "url": url })),
            ToolDetail::Browse { url: url.clone() }
        );
        assert!(objective.chars().count() > 50);
        assert!(url.chars().count() > 60);
    }

    #[test]
    fn shell_preserves_command_arguments_workdir_and_full_raw_output() {
        let detail = parse(
            "shell",
            serde_json::json!({
                "command": "printf",
                "args": ["hello world", "it's", "界", "", 7],
                "workdir": "/tmp/工作",
            }),
        );
        assert_eq!(
            detail,
            ToolDetail::Shell {
                command: "printf".into(),
                arguments: vec!["hello world".into(), "it's".into(), "界".into(), "".into()],
                workdir: Some("/tmp/工作".into()),
                output: None,
            }
        );

        let mut messages = vec![tool_call("shell-1", "shell", detail)];
        let result = serde_json::json!({
            "stdout": "out1  \n\n out2\nout3\nout4\nout5\nout6\n",
            "stderr": "\nerr1  \n错误\n",
        })
        .to_string();
        assert!(update_tool_detail(&mut messages, Some("shell-1"), &result));
        assert!(matches!(
            &messages[0],
            ChatEntry::ToolCall {
                detail: ToolDetail::Shell { output: Some(output), .. },
                ..
            } if output.stdout == "out1  \n\n out2\nout3\nout4\nout5\nout6\n"
                && output.stderr == "\nerr1  \n错误\n"
        ));
    }

    #[test]
    fn read_tool_preserves_raw_path_and_semantic_one_based_range() {
        let detail = parse(
            "read_tool",
            serde_json::json!({
                "path": "/workspace/project/src/acp_state.rs",
                "offset": 2134,
                "limit": 71,
            }),
        );
        assert_eq!(
            detail,
            ToolDetail::ReadTool {
                path: "/workspace/project/src/acp_state.rs".into(),
                start_line: Some(2135),
                end_line: Some(2205),
            }
        );

        for (arguments, expected_end) in [
            (
                serde_json::json!({ "path": "README.md" }),
                DEFAULT_READ_TOOL_LIMIT,
            ),
            (
                serde_json::json!({ "path": "README.md", "limit": 0 }),
                DEFAULT_READ_TOOL_LIMIT,
            ),
            (
                serde_json::json!({ "path": "README.md", "limit": 300 }),
                300,
            ),
        ] {
            assert!(matches!(
                parse("read_tool", arguments),
                ToolDetail::ReadTool { start_line: Some(1), end_line: Some(end), .. }
                    if end == expected_end
            ));
        }
    }

    #[test]
    fn read_results_refine_semantic_line_ranges() {
        let detail = parse(
            "read_tool",
            serde_json::json!({ "path": "src/lib.rs", "limit": 260 }),
        );
        let mut messages = vec![tool_call("read-1", "read_tool", detail)];
        let result = "<content>\n00041| first\n00042| second\n00430| last\n</content>";
        assert!(update_tool_detail(&mut messages, Some("read-1"), result));
        assert!(matches!(
            &messages[0],
            ChatEntry::ToolCall {
                detail: ToolDetail::ReadTool {
                    start_line: Some(41),
                    end_line: Some(430),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn edit_and_multiedit_preserve_raw_text_flags_indices_and_line_metadata() {
        let edit = parse(
            "edit",
            serde_json::json!({
                "file_path": "/workspace/src/lib.rs",
                "old_string": "before\n",
                "new_string": "after\n",
                "replace_all": true,
            }),
        );
        assert_eq!(
            edit,
            ToolDetail::Edit {
                file: "/workspace/src/lib.rs".into(),
                old: "before\n".into(),
                new: "after\n".into(),
                replace_all: true,
                start_line: None,
            }
        );

        let mut edit_messages = vec![tool_call("edit-1", "edit", edit)];
        assert!(update_tool_detail(
            &mut edit_messages,
            Some("edit-1"),
            r#"{"startLineOld":42}"#,
        ));
        assert!(matches!(
            &edit_messages[0],
            ChatEntry::ToolCall {
                detail: ToolDetail::Edit {
                    start_line: Some(42),
                    ..
                },
                ..
            }
        ));

        let multiedit = parse(
            "multiedit",
            serde_json::json!({
                "filePath": "/workspace/src/lib.rs",
                "edits": [
                    { "oldString": "one\n", "newString": "ONE\n", "replaceAll": true },
                    { "old_string": "two\n", "new_string": "TWO\n", "replace_all": false },
                    { "oldString": "missing new value" },
                ],
            }),
        );
        let ToolDetail::MultiEdit {
            file,
            edit_count,
            sections,
        } = &multiedit
        else {
            panic!("expected MultiEdit, got {multiedit:?}");
        };
        assert_eq!(file, "/workspace/src/lib.rs");
        assert_eq!(*edit_count, 2);
        assert_eq!(
            sections,
            &[
                MultiEditSection {
                    edit_index: 1,
                    replace_all: true,
                    old: "one\n".into(),
                    new: "ONE\n".into(),
                    start_line: None,
                },
                MultiEditSection {
                    edit_index: 2,
                    replace_all: false,
                    old: "two\n".into(),
                    new: "TWO\n".into(),
                    start_line: None,
                },
            ]
        );

        let mut multiedit_messages = vec![tool_call("multi-1", "multiedit", multiedit)];
        let receipt = "old_start=10 old_lines=1\nold_start=20 old_lines=1";
        assert!(update_tool_detail(
            &mut multiedit_messages,
            Some("multi-1"),
            receipt,
        ));
        assert!(matches!(
            &multiedit_messages[0],
            ChatEntry::ToolCall {
                detail: ToolDetail::MultiEdit { sections, .. },
                ..
            } if sections.iter().map(|section| section.start_line).collect::<Vec<_>>()
                == [Some(10), Some(20)]
        ));
    }

    #[test]
    fn structured_tool_families_keep_raw_protocol_values() {
        assert_eq!(
            parse(
                "search_text",
                serde_json::json!({
                    "pattern": "needle",
                    "path": "/workspace/project/src/lib.rs",
                    "include": "*.rs",
                }),
            ),
            ToolDetail::SearchText {
                pattern: "needle".into(),
                path: "/workspace/project/src/lib.rs".into(),
                include: "*.rs".into(),
                counts: None,
            }
        );
        assert_eq!(
            parse(
                "glob",
                serde_json::json!({ "pattern": "*.rs", "path": "/workspace/project/src" }),
            ),
            ToolDetail::Glob {
                pattern: "*.rs".into(),
                path: "/workspace/project/src".into(),
            }
        );
        assert_eq!(
            parse(
                "ls",
                serde_json::json!({ "path": "/workspace/project/src" })
            ),
            ToolDetail::List {
                path: "/workspace/project/src".into(),
            }
        );
        assert_eq!(
            parse(
                "index",
                serde_json::json!({ "path": "/workspace/project/src/main.rs" }),
            ),
            ToolDetail::Index {
                path: "/workspace/project/src/main.rs".into(),
                metadata: None,
            }
        );
        assert_eq!(
            parse(
                "delete_file",
                serde_json::json!({ "path": "/workspace/project/tmp/out.txt" }),
            ),
            ToolDetail::DeleteFile {
                path: "/workspace/project/tmp/out.txt".into(),
            }
        );
        assert_eq!(
            parse(
                "language_query",
                serde_json::json!({
                    "action": "definition",
                    "uri": "file:///workspace/project/src/lib.rs",
                }),
            ),
            ToolDetail::LanguageQuery {
                action: "definition".into(),
                uri: "file:///workspace/project/src/lib.rs".into(),
            }
        );
        assert_eq!(
            parse("question", serde_json::json!({ "prompt": "Continue?" })),
            ToolDetail::Question {
                prompt: "Continue?".into(),
            }
        );
        assert_eq!(
            parse("apply_patch", serde_json::json!({ "patch": "raw patch" })),
            ToolDetail::ApplyPatch {
                patch: "raw patch".into(),
            }
        );
    }

    #[test]
    fn todo_and_replace_symbol_preserve_semantic_items_and_replacements() {
        assert_eq!(
            parse(
                "todowrite",
                serde_json::json!({ "todos": [
                    { "content": "done", "status": "completed" },
                    { "content": "next", "status": "pending" },
                    { "content": "working", "status": "in_progress" },
                ] }),
            ),
            ToolDetail::Todo {
                items: vec![
                    TodoItem {
                        content: "done".into(),
                        status: "completed".into()
                    },
                    TodoItem {
                        content: "next".into(),
                        status: "pending".into()
                    },
                    TodoItem {
                        content: "working".into(),
                        status: "in_progress".into()
                    },
                ],
            }
        );
        assert_eq!(
            parse(
                "replace_symbol",
                serde_json::json!({ "replacements": [
                    {
                        "path": "/workspace/src/one.rs",
                        "symbol": "one",
                        "newText": "fn one() {}",
                    },
                    {
                        "path": "/workspace/src/two.rs",
                        "symbol": "two",
                        "newText": "fn two() {}",
                    },
                ] }),
            ),
            ToolDetail::ReplaceSymbolInput {
                replacements: vec![
                    SymbolReplacement {
                        path: "/workspace/src/one.rs".into(),
                        symbol: "one".into(),
                        new_text: "fn one() {}".into(),
                    },
                    SymbolReplacement {
                        path: "/workspace/src/two.rs".into(),
                        symbol: "two".into(),
                        new_text: "fn two() {}".into(),
                    },
                ],
            }
        );
        assert!(matches!(
            parse("todowrite", serde_json::json!({ "todos": [] })),
            ToolDetail::None
        ));
    }

    #[test]
    fn delegate_without_agent_preserves_objective_only() {
        assert_eq!(
            parse(
                "delegate",
                serde_json::json!({ "objective": "Do something" }),
            ),
            ToolDetail::Delegate {
                target_agent_id: String::new(),
                objective: "Do something".into(),
            }
        );
    }

    #[test]
    fn shell_preserves_full_utf8_multiline_command() {
        let command =
            "cat > check/kimi.md << 'EOF'\n# Review: feat/profiles\n\n## 🔴 Critical / High";
        assert!(matches!(
            parse("shell", serde_json::json!({ "command": command })),
            ToolDetail::Shell { command: parsed, arguments, .. }
                if parsed == command && arguments.is_empty()
        ));
    }

    #[test]
    fn shell_keeps_arguments_distinct_without_display_quoting() {
        assert!(matches!(
            parse(
                "shell",
                serde_json::json!({
                    "command": "echo",
                    "args": ["hello world", "", "safe"],
                }),
            ),
            ToolDetail::Shell { command, arguments, .. }
                if command == "echo" && arguments == ["hello world", "", "safe"]
        ));
    }

    #[test]
    fn shell_uses_only_explicit_workdir() {
        assert!(matches!(
            parse(
                "shell",
                serde_json::json!({ "command": "pwd", "workdir": "/workspace/project" }),
            ),
            ToolDetail::Shell { workdir: Some(workdir), .. }
                if workdir == "/workspace/project"
        ));
        assert!(matches!(
            parse("shell", serde_json::json!({ "command": "pwd" })),
            ToolDetail::Shell { workdir: None, .. }
        ));
    }

    #[test]
    fn read_tool_parses_json_string_arguments() {
        let arguments = Value::String(
            r#"{"path":"apps/admin/lib/admin_web/components/layouts/root.html.heex","limit":220}"#
                .into(),
        );
        assert_eq!(
            parse_tool_detail("read_tool", Some(&arguments)),
            ToolDetail::ReadTool {
                path: "apps/admin/lib/admin_web/components/layouts/root.html.heex".into(),
                start_line: Some(1),
                end_line: Some(220),
            }
        );
    }

    #[test]
    fn read_tool_result_refines_range_from_json_text_content() {
        let detail = parse(
            "read_tool",
            serde_json::json!({ "path": "src/lib.rs", "offset": 40, "limit": 390 }),
        );
        let mut messages = vec![tool_call("read-json", "read_tool", detail)];
        let result = serde_json::json!([{
            "type": "text",
            "text": "<content>\n00041| first\n00042| second\n00430| last\n</content>",
        }])
        .to_string();

        assert!(update_tool_detail(
            &mut messages,
            Some("read-json"),
            &result,
        ));
        assert!(matches!(
            &messages[0],
            ChatEntry::ToolCall {
                detail: ToolDetail::ReadTool {
                    start_line: Some(41),
                    end_line: Some(430),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn edit_result_uses_compact_old_start_receipt() {
        let detail = parse(
            "edit",
            serde_json::json!({
                "filePath": "src/lib.rs",
                "oldString": "before\n",
                "newString": "after\n",
            }),
        );
        let mut messages = vec![tool_call("edit-compact", "edit", detail)];

        assert!(update_tool_detail(
            &mut messages,
            Some("edit-compact"),
            "old_start=73 old_lines=1 new_start=73 new_lines=1",
        ));
        assert!(matches!(
            &messages[0],
            ChatEntry::ToolCall {
                detail: ToolDetail::Edit {
                    start_line: Some(73),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn multiedit_result_uses_hunk_old_start_metadata() {
        let detail = parse(
            "multiedit",
            serde_json::json!({
                "filePath": "src/lib.rs",
                "edits": [
                    { "oldString": "aaa\n", "newString": "bbb\n" },
                    { "oldString": "ccc\n", "newString": "ddd\n" },
                ],
            }),
        );
        let mut messages = vec![tool_call("multi-hunks", "multiedit", detail)];
        let result = "H replace old=10,1 new=10,1 +1 -1\nH replace old=20,1 new=20,1 +1 -1";

        assert!(update_tool_detail(
            &mut messages,
            Some("multi-hunks"),
            result,
        ));
        assert!(matches!(
            &messages[0],
            ChatEntry::ToolCall {
                detail: ToolDetail::MultiEdit { sections, .. },
                ..
            } if sections.iter().map(|section| section.start_line).collect::<Vec<_>>()
                == [Some(10), Some(20)]
        ));
    }

    #[test]
    fn replace_symbol_missing_and_empty_inputs_remain_semantic() {
        for arguments in [
            serde_json::json!({}),
            serde_json::json!({ "replacements": [] }),
        ] {
            assert_eq!(
                parse("replace_symbol", arguments),
                ToolDetail::ReplaceSymbolInput {
                    replacements: Vec::new(),
                }
            );
        }
    }

    #[test]
    fn replace_symbol_duplicate_paths_are_not_display_deduplicated() {
        let detail = parse(
            "replace_symbol",
            serde_json::json!({ "replacements": [
                { "path": "/workspace/src/lib.rs", "symbol": "run", "newText": "run" },
                { "path": "/workspace/src/lib.rs", "symbol": "stop", "newText": "stop" },
            ] }),
        );
        assert!(matches!(
            detail,
            ToolDetail::ReplaceSymbolInput { replacements }
                if replacements.len() == 2
                    && replacements[0].path == "/workspace/src/lib.rs"
                    && replacements[1].path == "/workspace/src/lib.rs"
        ));
    }

    #[test]
    fn todowrite_missing_and_empty_inputs_produce_no_detail() {
        assert!(matches!(
            parse("todowrite", serde_json::json!({})),
            ToolDetail::None
        ));
        assert!(matches!(
            parse("todowrite", serde_json::json!({ "todos": [] })),
            ToolDetail::None
        ));
    }

    #[test]
    fn browse_keeps_utf8_url_without_parser_ellipsis() {
        let url = format!("https://example.test/{}tail", "界".repeat(50));
        assert!(matches!(
            parse("browse", serde_json::json!({ "url": url })),
            ToolDetail::Browse { url: parsed } if parsed == url
        ));
    }

    #[test]
    fn delegate_keeps_utf8_objective_without_parser_ellipsis() {
        let objective = format!("{}tail", "界".repeat(50));
        assert!(matches!(
            parse(
                "delegate",
                serde_json::json!({
                    "target_agent_id": "coder",
                    "objective": objective,
                }),
            ),
            ToolDetail::Delegate { target_agent_id, objective: parsed }
                if target_agent_id == "coder" && parsed == objective
        ));
    }

    #[test]
    fn index_keeps_full_unshortened_path() {
        let path = "/home/user/project/src/main.rs";
        assert_eq!(
            parse("index", serde_json::json!({ "path": path })),
            ToolDetail::Index {
                path: path.into(),
                metadata: None,
            }
        );
    }

    #[test]
    fn write_file_preserves_raw_unicode_path_and_content() {
        assert_eq!(
            parse(
                "write_file",
                serde_json::json!({
                    "path": "/workspace/src/界.rs",
                    "content": "fn 界() {\n    println!(\"ok\");\n}\n",
                }),
            ),
            ToolDetail::WriteFile {
                path: "/workspace/src/界.rs".into(),
                content: "fn 界() {\n    println!(\"ok\");\n}\n".into(),
            }
        );
    }

    #[test]
    fn unknown_and_missing_arguments_produce_no_detail() {
        assert!(matches!(
            parse("unknown_tool", serde_json::json!({ "raw": true })),
            ToolDetail::None
        ));
        assert!(matches!(parse_tool_detail("shell", None), ToolDetail::None));
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
        let detail = parse(
            "shell",
            serde_json::json!({ "command": "cargo test tool_detail_tests" }),
        );
        assert!(reconcile_tool_call_start(
            &mut messages,
            Some("missing-start"),
            "shell",
            detail,
        ));
        assert_eq!(messages.len(), 2);
        assert!(matches!(
            &messages[1],
            ChatEntry::ToolCall {
                name,
                is_error: true,
                detail: ToolDetail::Shell { command, .. },
                ..
            } if name == "shell" && command == "cargo test tool_detail_tests"
        ));
    }
}
