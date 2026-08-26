use agent_client_protocol::{self as acp_sdk, schema::v1 as acp};
use serde_json::{Value, json};

use crate::acp_state::AcpAppEvent;
use crate::domain::session::ForkResult;

use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::extensions::history;

pub(super) async fn fork<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    message_id: String,
) -> Result<(), acp_sdk::Error> {
    let Some(session_id) = ctx.state.current_session_id().await else {
        ctx.events.send(AcpAppEvent::ForkResult(ForkResult::Failed {
            source_session_id: None,
            message: Some("cannot fork before a session is loaded".to_string()),
        }));
        return Ok(());
    };
    let request = acp::ForkSessionRequest::new(session_id.clone(), ctx.state.default_cwd())
        .meta(fork_meta(&message_id));
    match ctx.connection.request(request).await {
        Ok(response) => ctx
            .events
            .send(AcpAppEvent::ForkResult(ForkResult::Succeeded {
                source_session_id: Some(session_id),
                forked_session_id: Some(response.session_id.to_string()),
                message: None,
            })),
        Err(err) => ctx.events.send(AcpAppEvent::ForkResult(ForkResult::Failed {
            source_session_id: Some(session_id),
            message: Some(format!("acp fork failed: {err:?}")),
        })),
    }
    Ok(())
}

pub(super) async fn undo<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    message_id: String,
) -> Result<(), acp_sdk::Error> {
    let Some(session_id) = ctx.state.current_session_id().await else {
        ctx.events.error("cannot undo before a session is loaded");
        return Ok(());
    };
    if let Some(result) = history::undo(ctx.connection, session_id, message_id).await? {
        ctx.events.send(AcpAppEvent::UndoResult(result));
    }
    Ok(())
}

pub(super) async fn redo<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
) -> Result<(), acp_sdk::Error> {
    let Some(session_id) = ctx.state.current_session_id().await else {
        ctx.events.error("cannot redo before a session is loaded");
        return Ok(());
    };
    if let Some(result) = history::redo(ctx.connection, session_id).await? {
        ctx.events.send(AcpAppEvent::RedoResult(result));
    }
    Ok(())
}

fn fork_meta(message_id: &str) -> serde_json::Map<String, Value> {
    let mut meta = serde_json::Map::new();
    meta.insert("querymt".to_string(), json!({ "message_id": message_id }));
    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_metadata_keeps_the_querymt_boundary_hint() {
        assert_eq!(fork_meta("m1")["querymt"]["message_id"], "m1");
    }
}
