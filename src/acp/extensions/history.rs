use agent_client_protocol as acp_sdk;
use serde_json::json;

use crate::domain::session::{RedoResult, UndoResult, UndoStackSnapshot};
use crate::protocol::session::{RedoResultDto, UndoResultDto, UndoStackFrameDto};

use super::{call, payload};
use crate::acp::connection::AcpConnection;

pub(super) async fn undo<C: AcpConnection>(
    connection: &C,
    session_id: String,
    message_id: String,
) -> Result<Option<UndoResult>, acp_sdk::Error> {
    let response = call(
        connection,
        "querymt/session/undo",
        json!({ "session_id": session_id, "message_id": message_id }),
    )
    .await?;
    Ok(
        serde_json::from_value::<UndoResultDto>(payload(&response).clone())
            .ok()
            .map(undo_from_wire),
    )
}

pub(super) async fn redo<C: AcpConnection>(
    connection: &C,
    session_id: String,
) -> Result<Option<RedoResult>, acp_sdk::Error> {
    let response = call(
        connection,
        "querymt/session/redo",
        json!({ "session_id": session_id }),
    )
    .await?;
    Ok(
        serde_json::from_value::<RedoResultDto>(payload(&response).clone())
            .ok()
            .map(redo_from_wire),
    )
}

pub(super) async fn stack<C: AcpConnection>(
    connection: &C,
    session_id: &str,
) -> Result<UndoStackSnapshot, acp_sdk::Error> {
    let response = call(
        connection,
        "querymt/session/undoStack",
        json!({ "session_id": session_id }),
    )
    .await?;
    let frames = payload(&response)
        .get("undo_stack")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| serde_json::from_value::<UndoStackFrameDto>(item.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    Ok(stack_from_wire(frames))
}

fn stack_from_wire(frames: Vec<UndoStackFrameDto>) -> UndoStackSnapshot {
    UndoStackSnapshot {
        message_ids: frames.into_iter().map(|frame| frame.message_id).collect(),
    }
}

fn undo_from_wire(result: UndoResultDto) -> UndoResult {
    let stack = stack_from_wire(result.undo_stack);
    if result.success {
        UndoResult::Applied {
            target_message_id: result.message_id,
            reverted_files: result.reverted_files,
            message: result.message,
            stack,
        }
    } else {
        UndoResult::Rejected {
            target_message_id: result.message_id,
            message: result.message,
            stack,
        }
    }
}

fn redo_from_wire(result: RedoResultDto) -> RedoResult {
    let stack = stack_from_wire(result.undo_stack);
    if result.success {
        RedoResult::Applied {
            message: result.message,
            stack,
        }
    } else {
        RedoResult::Rejected {
            message: result.message,
            stack,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn history_decoding_is_tolerant_and_stack_filters_bad_entries() {
        let response = json!({ "undo_stack": [
            { "message_id": "one" },
            { "bad": true },
            { "message_id": "two" }
        ] });
        let frames = payload(&response)
            .get("undo_stack")
            .and_then(serde_json::Value::as_array)
            .expect("stack")
            .iter()
            .filter_map(|value| serde_json::from_value::<UndoStackFrameDto>(value.clone()).ok())
            .collect();
        assert_eq!(stack_from_wire(frames).message_ids, ["one", "two"]);
        assert!(serde_json::from_value::<UndoResultDto>(json!({})).is_err());
    }
}
