use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};

use agent_client_protocol::{
    self as acp_sdk, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, schema::v1 as acp,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use super::commands;
use super::connection::{AcpConnection, internal_error};
use super::context::CommandContext;
use super::events::EventSink;
use super::inbound;
use super::runtime::RuntimeState;
use super::transport::jsonrpc::Peer;
use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};
use crate::app::App;
use crate::application::{self, AppEvent};
use crate::command::{Command, SessionListRequest};
use crate::domain::chat::ChatEntry;
use crate::runtime_events::ServerChannelMsg;

#[derive(Debug, Clone, PartialEq)]
struct RecordedMessage {
    method: String,
    params: Value,
}

#[derive(Clone, Default)]
struct RecordingConnection {
    messages: Arc<Mutex<Vec<RecordedMessage>>>,
    responses: Arc<Mutex<VecDeque<Result<Value, String>>>>,
}

impl RecordingConnection {
    fn with_responses(responses: impl IntoIterator<Item = Value>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(
                responses.into_iter().map(Ok).collect::<VecDeque<_>>(),
            )),
            ..Self::default()
        }
    }

    fn messages(&self) -> Vec<RecordedMessage> {
        self.messages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl AcpConnection for RecordingConnection {
    async fn request<R>(&self, request: R) -> Result<R::Response, acp_sdk::Error>
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static,
    {
        let message = request.to_untyped_message()?;
        self.messages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(RecordedMessage {
                method: message.method.clone(),
                params: message.params,
            });
        let response = self
            .responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or_else(|| Ok(json!({})))
            .map_err(internal_error)?;
        R::Response::from_value(&message.method, response)
    }

    fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
    where
        N: JsonRpcNotification + Send + Sync + 'static,
    {
        let message = notification.to_untyped_message()?;
        self.messages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(RecordedMessage {
                method: message.method,
                params: message.params,
            });
        Ok(())
    }

    fn spawn(
        &self,
        future: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
    ) -> Result<(), acp_sdk::Error> {
        tokio::spawn(async move {
            let _ = future.await;
        });
        Ok(())
    }
}

fn harness(
    connection: &RecordingConnection,
) -> (
    Arc<RuntimeState>,
    EventSink,
    mpsc::UnboundedReceiver<ServerChannelMsg>,
) {
    let state = Arc::new(RuntimeState::new(Some("/launch".into())));
    let (tx, rx) = mpsc::unbounded_channel();
    let events = EventSink::new(tx);
    let _ = connection;
    (state, events, rx)
}

fn context<'a>(
    connection: &'a RecordingConnection,
    state: &'a Arc<RuntimeState>,
    events: &'a EventSink,
) -> CommandContext<'a, RecordingConnection> {
    CommandContext {
        connection,
        state,
        events,
    }
}

#[tokio::test]
async fn router_keeps_noop_and_exact_unsupported_semantics() {
    let connection = RecordingConnection::default();
    let (state, events, mut rx) = harness(&connection);
    commands::dispatch(
        context(&connection, &state, &events),
        Command::SubscribeSession {
            session_id: "session".into(),
            agent_id: Some("agent".into()),
        },
    )
    .await
    .expect("subscribe no-op");
    assert!(connection.messages().is_empty());
    assert!(rx.try_recv().is_err());

    commands::dispatch(context(&connection, &state, &events), Command::GetFileIndex)
        .await
        .expect("file index unsupported");
    assert!(matches!(
        rx.try_recv().expect("file index error"),
        ServerChannelMsg::Acp(AcpAppEvent::Error { message })
            if message == "file mentions are not exposed in the ACP subset yet"
    ));

    let unsupported = Command::ClearApiToken {
        provider: "openai".into(),
    };
    commands::dispatch(context(&connection, &state, &events), unsupported.clone())
        .await
        .expect("api token unsupported");
    assert!(matches!(
        rx.try_recv().expect("unsupported error"),
        ServerChannelMsg::Acp(AcpAppEvent::Error { message })
            if message == format!("unsupported in the current ACP subset: {unsupported:?}")
    ));
    assert!(connection.messages().is_empty());
}

#[tokio::test]
async fn extension_router_pins_methods_underscore_prefix_and_explicit_nulls() {
    let connection = RecordingConnection::with_responses([
        json!({
            "session_id": "parent",
            "agent_id": "coder",
            "model": null
        }),
        json!({ "providers": [] }),
        json!({ "node_id": "node", "sessions": [], "total_count": 0 }),
        json!({
            "invite_id": "invite",
            "url": "qmt://invite",
            "expires_at": 1,
            "max_uses": 1
        }),
    ]);
    let (state, events, _rx) = harness(&connection);
    let commands = [
        Command::SetDelegateModel {
            session_id: "parent".into(),
            agent_id: "coder".into(),
            model_id: None,
            node_id: None,
        },
        Command::ListAuthProviders,
        Command::ListRemoteSessions {
            node_id: "node".into(),
            offset: 0,
            limit: 50,
        },
        Command::CreateMeshInvite {
            mesh_name: None,
            ttl: None,
            max_uses: None,
        },
    ];
    for command in commands {
        commands::dispatch(context(&connection, &state, &events), command)
            .await
            .expect("extension command");
    }

    let messages = connection.messages();
    assert_eq!(
        messages
            .iter()
            .map(|message| message.method.as_str())
            .collect::<Vec<_>>(),
        [
            "_querymt/session/setDelegateModel",
            "_querymt/auth/status",
            "_querymt/remote/sessions",
            "_querymt/mesh/createInvite",
        ]
    );
    assert_eq!(messages[0].params["model_id"], Value::Null);
    assert_eq!(messages[0].params["node_id"], Value::Null);
    assert_eq!(
        messages[2].params,
        json!({ "node_id": "node", "offset": 0, "limit": 50 })
    );
    assert_eq!(
        messages[3].params,
        json!({ "mesh_name": null, "ttl": null, "max_uses": null })
    );
}

#[tokio::test]
async fn standard_router_pins_list_and_cancel_wire_shapes() {
    let connection = RecordingConnection::with_responses([json!({
        "sessions": [],
        "nextCursor": null
    })]);
    let (state, events, _rx) = harness(&connection);
    commands::dispatch(
        context(&connection, &state, &events),
        Command::ListSessions {
            request: SessionListRequest::WorkspaceFirstPage {
                cwd: "/repo".into(),
            },
            cursor: Some("cursor".into()),
        },
    )
    .await
    .expect("list sessions");
    state.set_current_session_id("session").await;
    commands::dispatch(
        context(&connection, &state, &events),
        Command::CancelSession,
    )
    .await
    .expect("cancel");

    let messages = connection.messages();
    assert_eq!(messages[0].method, "session/list");
    assert_eq!(messages[0].params["cwd"], "/repo");
    assert_eq!(messages[0].params["cursor"], "cursor");
    assert_eq!(messages[1].method, "session/cancel");
    assert_eq!(messages[1].params["sessionId"], "session");
}

#[tokio::test]
async fn load_emits_loaded_replay_delegation_provider_and_stack_in_order() {
    let load_response = json!({
        "_meta": {
            "querymt/sessionLoadSnapshot.v1": {
                "audit": { "events": [
                    {
                        "kind": {
                            "type": "prompt_received",
                            "data": { "content": "hello", "message_id": "u1" }
                        }
                    },
                    {
                        "kind": {
                            "type": "provider_changed",
                            "data": {
                                "provider": "openai",
                                "model": "gpt-5",
                                "context_limit": 100
                            }
                        }
                    }
                ] },
                "delegationUpdates": [{
                    "version": 1,
                    "sessionId": "session",
                    "delegationId": "d1",
                    "state": "requested",
                    "targetAgentId": "coder",
                    "objective": "implement",
                    "requestedAt": 1,
                    "updatedAt": 1
                }]
            }
        }
    });
    let connection = RecordingConnection::with_responses([
        load_response,
        json!({ "undo_stack": [{ "message_id": "u1" }] }),
    ]);
    let (state, events, mut rx) = harness(&connection);
    commands::dispatch(
        context(&connection, &state, &events),
        Command::LoadSession {
            session_id: "session".into(),
            cwd: Some("/repo".into()),
        },
    )
    .await
    .expect("load");

    let emitted = (0..5)
        .map(|_| rx.try_recv().expect("ordered load event"))
        .collect::<Vec<_>>();
    assert!(matches!(
        &emitted[0],
        ServerChannelMsg::Acp(AcpAppEvent::SessionLoaded { session_id, .. })
            if session_id == "session"
    ));
    assert!(matches!(
        &emitted[1],
        ServerChannelMsg::Acp(AcpAppEvent::SessionReplay { updates, .. })
            if matches!(&updates[0], AcpSessionUpdate::UserMessage { message_id: Some(id), .. } if id == "u1")
    ));
    assert!(matches!(
        &emitted[2],
        ServerChannelMsg::Acp(AcpAppEvent::DelegationReplay { updates, .. })
            if updates.len() == 1
    ));
    assert!(matches!(
        &emitted[3],
        ServerChannelMsg::Acp(AcpAppEvent::ProviderChanged { provider, model, .. })
            if provider == "openai" && model == "gpt-5"
    ));
    assert!(matches!(
        &emitted[4],
        ServerChannelMsg::Acp(AcpAppEvent::UndoStack(stack))
            if stack.message_ids == ["u1"]
    ));
    assert!(rx.try_recv().is_err());
    assert_eq!(
        connection
            .messages()
            .iter()
            .map(|message| message.method.as_str())
            .collect::<Vec<_>>(),
        ["session/load", "_querymt/session/undoStack"]
    );
}

#[tokio::test]
async fn session_notification_reaches_application_reducer_coordination() {
    let connection = RecordingConnection::default();
    let (state, events, mut rx) = harness(&connection);
    let mut app = App::new();
    app.sessions.session_id = Some("session".into());
    let local_id = app.push_pending_prompt("hello".into());
    app.render.test_seed_card_cache(7);
    let original_logs = app.diagnostics.logs.clone();
    let original_log_cursor = app.diagnostics.log_cursor;
    let original_log_filter = app.diagnostics.log_filter.clone();
    let original_log_level_filter = app.diagnostics.log_level_filter;
    let original_status = app.diagnostics.status.clone();

    inbound::session_notification(
        &state,
        &events,
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::UserMessageChunk(
                acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new("hello")))
                    .message_id(Some(acp::MessageId::from("u1"))),
            ),
        ),
    )
    .await;

    let event = match rx.try_recv().expect("session notification event") {
        ServerChannelMsg::Acp(event) => AppEvent::Acp(event),
    };
    let effects = application::update(&mut app, event);

    assert!(app.sessions.session_activity.contains_key("session"));
    assert!(matches!(
        app.chat.messages.as_slice(),
        [ChatEntry::User {
            text,
            message_id: Some(message_id),
        }] if text == "hello" && message_id == "u1" && message_id != &local_id
    ));
    assert_eq!(app.chat.undoable_turns.len(), 1);
    assert_eq!(app.chat.undoable_turns[0].message_id, "u1");
    assert_eq!(app.render.test_card_source_entry_count(), 0);
    assert_eq!(app.diagnostics.logs, original_logs);
    assert_eq!(app.diagnostics.log_cursor, original_log_cursor);
    assert_eq!(app.diagnostics.log_filter, original_log_filter);
    assert_eq!(app.diagnostics.log_level_filter, original_log_level_filter);
    assert_eq!(app.diagnostics.status, original_status);
    assert_eq!(effects, Vec::new());
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn stdio_and_websocket_standard_inbound_share_normalized_events() {
    let notification = acp::SessionNotification::new(
        "session",
        acp::SessionUpdate::UserMessageChunk(
            acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new("hello")))
                .message_id(Some(acp::MessageId::from("u1"))),
        ),
    );
    let (stdio_tx, mut stdio_rx) = mpsc::unbounded_channel();
    let stdio_events = EventSink::new(stdio_tx);
    let stdio_state = Arc::new(RuntimeState::new(None));
    inbound::session_notification(&stdio_state, &stdio_events, notification.clone()).await;

    let (wire_tx, _wire_rx) = mpsc::unbounded_channel::<Message>();
    let peer = Peer::new(wire_tx);
    let connection = RecordingConnection::default();
    let (ws_tx, mut ws_rx) = mpsc::unbounded_channel();
    let ws_events = EventSink::new(ws_tx);
    let ws_state = Arc::new(RuntimeState::new(None));
    let text = serde_json::to_string(&json!({
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": serde_json::to_value(notification).expect("notification params")
    }))
    .expect("websocket envelope");
    inbound::websocket_text(&peer, &connection, &ws_state, &ws_events, &text)
        .await
        .expect("websocket inbound");

    let extract = |message| match message {
        ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate {
            session_id,
            update:
                AcpSessionUpdate::UserMessage {
                    content,
                    message_id,
                },
            is_replay,
        }) => (session_id, content, message_id, is_replay),
        other => panic!("unexpected event: {other:?}"),
    };
    assert_eq!(
        extract(stdio_rx.try_recv().expect("stdio event")),
        extract(ws_rx.try_recv().expect("websocket event"))
    );
}
