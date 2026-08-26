use std::collections::BTreeMap;
use std::path::PathBuf;

#[cfg(test)]
use std::path::Path;

use agent_client_protocol::{self as acp_sdk, schema::v1 as acp};
use serde_json::{Value, json};

use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};
use crate::command::{PromptBlock, SessionListRequest};
use crate::domain::session::{SessionGroup, SessionListPage, SessionSummary};

use super::super::configuration;
use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::extensions::history;
use super::super::replay;

pub(super) async fn list<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    request: SessionListRequest,
    cursor: Option<String>,
) {
    let mut acp_request = acp::ListSessionsRequest::new().cursor(cursor);
    if let Some(cwd) = request.cwd() {
        acp_request = acp_request.cwd(PathBuf::from(cwd));
    }
    match ctx.connection.request(acp_request).await {
        Ok(response) => ctx.events.send(AcpAppEvent::SessionList {
            page: list_page(&request, response),
            request,
        }),
        Err(err) => ctx.events.send(AcpAppEvent::SessionListFailed {
            request,
            message: format!("acp session/list failed: {err:?}"),
        }),
    }
}

pub(super) async fn new<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    cwd: Option<String>,
    profile_id: Option<String>,
) -> Result<(), acp_sdk::Error> {
    let mut request = acp::NewSessionRequest::new(
        cwd.map(PathBuf::from)
            .unwrap_or_else(|| ctx.state.default_cwd()),
    );
    if let Some(profile_id) = profile_id.as_deref() {
        request = request.meta(profile_meta(profile_id));
    }
    let response = ctx.connection.request(request).await?;
    let session_id = response.session_id.to_string();
    ctx.state.set_current_session_id(session_id.clone()).await;
    let identity = ctx.state.agent_identity().await;
    ctx.events.send(AcpAppEvent::SessionCreated {
        agent_id: identity.id,
        session_id,
        profile_id,
    });
    if let Some(options) = response.config_options {
        configuration::apply(ctx.state, ctx.events, options).await;
    }
    Ok(())
}

pub(super) async fn load<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    session_id: String,
    cwd: Option<String>,
) -> Result<(), acp_sdk::Error> {
    ctx.state.set_current_session_id(session_id.clone()).await;
    ctx.state.replay.begin(&session_id).await;
    let request = acp::LoadSessionRequest::new(
        session_id.clone(),
        load_cwd(cwd.as_deref(), ctx.state.default_cwd()),
    );
    let response = match ctx.connection.request(request).await {
        Ok(response) => response,
        Err(err) => {
            ctx.state.replay.abort(&session_id).await;
            return Err(err);
        }
    };
    let buffered = ctx.state.replay.start_completion(&session_id).await;
    let response_value = serde_json::to_value(&response).unwrap_or(Value::Null);
    let snapshot_updates = replay::snapshot_updates(&response_value);
    let delegation_updates = replay::delegation_updates(&response_value);
    let provider_change = replay::provider_change(&response_value);
    let profile_id = response
        .config_options
        .as_ref()
        .and_then(|options| configuration::profile_id(options));
    let identity = ctx.state.agent_identity().await;

    ctx.events.send(AcpAppEvent::SessionLoaded {
        session_id: session_id.clone(),
        agent_id: identity.id,
        profile_id,
    });
    send_replay(
        ctx,
        &session_id,
        replay::merge_snapshot_stats(buffered, snapshot_updates),
    );
    while let Some(tail) = ctx.state.replay.drain_completion(&session_id).await {
        send_replay(ctx, &session_id, tail);
    }
    if !delegation_updates.is_empty() {
        ctx.events.send(AcpAppEvent::DelegationReplay {
            session_id: session_id.clone(),
            updates: delegation_updates,
        });
    }
    if let Some(change) = provider_change {
        ctx.events.send(replay::provider_change_event(change));
    }
    if let Some(options) = response.config_options {
        configuration::apply(ctx.state, ctx.events, options).await;
    }
    if let Ok(stack) = history::stack(ctx.connection, &session_id).await {
        ctx.events.send(AcpAppEvent::UndoStack(stack));
    }
    Ok(())
}

fn send_replay<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    session_id: &str,
    updates: Vec<AcpSessionUpdate>,
) {
    if !updates.is_empty() {
        ctx.events.send(AcpAppEvent::SessionReplay {
            session_id: session_id.to_string(),
            updates,
        });
    }
}

pub(super) async fn prompt<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    prompt: Vec<PromptBlock>,
    local_id: String,
) -> Result<(), acp_sdk::Error> {
    let Some(session_id) = ctx.state.current_session_id().await else {
        ctx.events.error("cannot prompt before a session is loaded");
        return Ok(());
    };
    ctx.events
        .session_update(&session_id, AcpSessionUpdate::TurnStarted);
    let connection = ctx.connection.clone();
    let state = ctx.state.clone();
    let events = ctx.events.clone();
    ctx.connection.spawn(async move {
        let request = acp::PromptRequest::new(session_id.clone(), prompt_blocks(prompt));
        match connection.request(request).await {
            Ok(response) => {
                if let Some(update) = state.assistants.flush(&session_id).await {
                    events.session_update(&session_id, update);
                }
                if matches!(response.stop_reason, acp::StopReason::Cancelled) {
                    events.session_update(&session_id, AcpSessionUpdate::Cancelled);
                } else {
                    events.session_update(
                        &session_id,
                        AcpSessionUpdate::Finished {
                            finish_reason: format!("{:?}", response.stop_reason),
                        },
                    );
                }
            }
            Err(err) => events.send(AcpAppEvent::PromptFailed {
                local_id,
                message: format!("acp prompt failed: {err:?}"),
            }),
        }
        Ok(())
    })
}

pub(super) async fn cancel<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
) -> Result<(), acp_sdk::Error> {
    let Some(session_id) = ctx.state.current_session_id().await else {
        ctx.events
            .info("acp", "session/cancel skipped: no active session");
        return Ok(());
    };
    ctx.connection
        .notify(acp::CancelNotification::new(session_id.clone()))?;
    ctx.events
        .info("acp", format!("sent session/cancel for {session_id}"));
    Ok(())
}

pub(super) async fn delete<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    session_id: String,
) -> Result<(), acp_sdk::Error> {
    ctx.connection
        .request(acp::DeleteSessionRequest::new(session_id))
        .await?;
    Ok(())
}

fn prompt_blocks(blocks: Vec<PromptBlock>) -> Vec<acp::ContentBlock> {
    blocks
        .into_iter()
        .map(|block| match block {
            PromptBlock::Text { text } => acp::ContentBlock::Text(acp::TextContent::new(text)),
            PromptBlock::ResourceLink { name, uri } => {
                acp::ContentBlock::ResourceLink(acp::ResourceLink::new(name, uri))
            }
        })
        .collect()
}

fn profile_meta(profile_id: &str) -> serde_json::Map<String, Value> {
    let mut meta = serde_json::Map::new();
    meta.insert("querymt".to_string(), json!({ "profile_id": profile_id }));
    meta
}

fn load_cwd(cwd: Option<&str>, default_cwd: PathBuf) -> PathBuf {
    cwd.and_then(|cwd| (!cwd.trim().is_empty()).then(|| PathBuf::from(cwd)))
        .unwrap_or(default_cwd)
}

fn list_page(request: &SessionListRequest, response: acp::ListSessionsResponse) -> SessionListPage {
    let mut groups: BTreeMap<Option<String>, Vec<SessionSummary>> = BTreeMap::new();
    let response_cursor = response.next_cursor.map(|cursor| cursor.to_string());
    for session in response.sessions {
        let cwd = session.cwd.to_string_lossy().to_string();
        let group_key = request
            .cwd()
            .map(str::to_string)
            .or_else(|| (!cwd.is_empty()).then_some(cwd.clone()));
        groups.entry(group_key).or_default().push(SessionSummary {
            session_id: session.session_id.to_string(),
            name: session.title.clone(),
            title: session.title,
            cwd: (!cwd.is_empty()).then_some(cwd),
            created_at: None,
            updated_at: session.updated_at,
            parent_session_id: None,
            fork_origin: None,
            session_kind: None,
            has_children: false,
            fork_count: 0,
            children: Vec::new(),
            children_next_cursor: None,
            children_total_count: None,
            node: None,
            node_id: None,
            attached: None,
            runtime_state: None,
        });
    }
    if let Some(cwd) = request.cwd() {
        groups.entry(Some(cwd.to_string())).or_default();
    }
    let workspace_cursor = request
        .cwd()
        .is_some()
        .then_some(response_cursor.clone())
        .flatten();
    SessionListPage {
        groups: groups
            .into_iter()
            .map(|(cwd, sessions)| SessionGroup {
                cwd,
                latest_activity: sessions
                    .first()
                    .and_then(|session| session.updated_at.clone()),
                total_count: None,
                next_cursor: workspace_cursor.clone(),
                sessions,
            })
            .collect(),
        next_cursor: response_cursor,
        total_count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_blocks_preserve_text_and_resource_links() {
        let blocks = prompt_blocks(vec![
            PromptBlock::Text {
                text: "inspect".into(),
            },
            PromptBlock::ResourceLink {
                name: "main.rs".into(),
                uri: "file:///repo/main.rs".into(),
            },
        ]);
        assert!(matches!(
            blocks.as_slice(),
            [acp::ContentBlock::Text(text), acp::ContentBlock::ResourceLink(link)]
                if text.text == "inspect" && link.uri == "file:///repo/main.rs"
        ));
    }

    #[test]
    fn root_and_workspace_pages_preserve_cursor_asymmetry() {
        let response = || {
            acp::ListSessionsResponse::new(vec![acp::SessionInfo::new(
                acp::SessionId::from("s1"),
                Path::new("/repo"),
            )])
            .next_cursor(Some("next".to_string()))
        };
        let root = list_page(&SessionListRequest::Discovery, response());
        assert_eq!(root.next_cursor.as_deref(), Some("next"));
        assert!(root.groups[0].next_cursor.is_none());
        let workspace = list_page(
            &SessionListRequest::WorkspaceContinuation {
                cwd: "/repo".into(),
            },
            response(),
        );
        assert_eq!(workspace.groups[0].next_cursor.as_deref(), Some("next"));
    }
}
