use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct UndoStackFrameDto {
    pub message_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UndoResultDto {
    pub success: bool,
    pub message_id: Option<String>,
    #[serde(default)]
    pub reverted_files: Vec<String>,
    pub message: Option<String>,
    #[serde(default)]
    pub undo_stack: Vec<UndoStackFrameDto>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RedoResultDto {
    pub success: bool,
    pub message: Option<String>,
    #[serde(default)]
    pub undo_stack: Vec<UndoStackFrameDto>,
}

#[cfg(test)]
mod tests {
    use super::{RedoResultDto, UndoResultDto};
    use serde_json::json;

    #[test]
    fn undo_result_deserializes_success_fields_and_stack_order() {
        let result: UndoResultDto = serde_json::from_value(json!({
            "success": true,
            "message_id": "message-2",
            "reverted_files": ["src/a.rs", "src/b.rs"],
            "message": "undone",
            "undo_stack": [
                { "message_id": "message-1" },
                { "message_id": "message-2" }
            ],
            "unknown_field": "ignored"
        }))
        .unwrap();

        assert!(result.success);
        assert_eq!(result.message_id.as_deref(), Some("message-2"));
        assert_eq!(result.reverted_files, ["src/a.rs", "src/b.rs"]);
        assert_eq!(result.message.as_deref(), Some("undone"));
        assert_eq!(result.undo_stack[0].message_id, "message-1");
        assert_eq!(result.undo_stack[1].message_id, "message-2");
    }

    #[test]
    fn undo_result_deserializes_failure_with_default_collections() {
        let result: UndoResultDto = serde_json::from_value(json!({
            "success": false,
            "message_id": null,
            "message": "undo rejected"
        }))
        .unwrap();

        assert!(!result.success);
        assert!(result.reverted_files.is_empty());
        assert!(result.undo_stack.is_empty());
        assert_eq!(result.message.as_deref(), Some("undo rejected"));
    }

    #[test]
    fn redo_result_deserializes_success_fields_and_stack_order() {
        let result: RedoResultDto = serde_json::from_value(json!({
            "success": true,
            "message": "redone",
            "undo_stack": [
                { "message_id": "message-1" },
                { "message_id": "message-2" }
            ]
        }))
        .unwrap();

        assert!(result.success);
        assert_eq!(result.message.as_deref(), Some("redone"));
        assert_eq!(result.undo_stack[0].message_id, "message-1");
        assert_eq!(result.undo_stack[1].message_id, "message-2");
    }

    #[test]
    fn redo_result_deserializes_failure_with_default_stack() {
        let result: RedoResultDto = serde_json::from_value(json!({
            "success": false,
            "message": "redo rejected"
        }))
        .unwrap();

        assert!(!result.success);
        assert!(result.undo_stack.is_empty());
        assert_eq!(result.message.as_deref(), Some("redo rejected"));
    }
}
