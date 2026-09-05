use std::collections::VecDeque;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::{
    self as acp_sdk, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, schema::v1 as acp,
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use super::commands;
use super::connection::{AcpConnection, internal_error};
use super::context::CommandContext;
use super::elicitation::PendingResponse;
use super::events::EventSink;
use super::inbound;
use super::runtime::RuntimeState;
use super::transport::jsonrpc::Peer;
use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};
use crate::app::App;
use crate::application::{self, AppEvent};
use crate::command::{Command, PromptBlock, SessionListRequest};
use crate::domain::auth::{AuthMethod, OAuthFlowKind, OAuthResultStatus, OAuthStatus};
use crate::domain::chat::ChatEntry;
use crate::domain::session::{ForkResult, RedoResult, UndoResult, UndoStackSnapshot};
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
    commands::dispatch(context(&connection, &state, &events), unsupported)
        .await
        .expect("api token unsupported");
    assert!(matches!(
        rx.try_recv().expect("unsupported error"),
        ServerChannelMsg::Acp(AcpAppEvent::Error { message })
            if message == "unsupported in the current ACP subset: ClearApiToken"
    ));

    let api_key = "sentinel-secret-api-key";
    let unsupported = Command::SetApiToken {
        provider: "openai".into(),
        api_key: api_key.into(),
    };
    assert!(!format!("{unsupported:?}").contains(api_key));
    commands::dispatch(context(&connection, &state, &events), unsupported)
        .await
        .expect("set api token unsupported");
    let ServerChannelMsg::Acp(AcpAppEvent::Error { message }) =
        rx.try_recv().expect("unsupported error")
    else {
        panic!("expected unsupported error");
    };
    assert_eq!(
        message,
        "unsupported in the current ACP subset: SetApiToken"
    );
    assert!(!message.contains(api_key));
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
        json!({
            "providers": [{
                "provider": "openai",
                "display_name": "OpenAI",
                "oauth_status": "connected",
                "has_stored_api_key": false,
                "has_env_api_key": true,
                "env_var_name": "OPENAI_API_KEY",
                "supports_oauth": true,
                "preferred_method": "oauth"
            }]
        }),
        json!({
            "node_id": "node",
            "sessions": [{
                "id": "remote",
                "node_id": "node",
                "node_label": "Remote Node",
                "title": "Remote Session",
                "cwd": "/remote",
                "updated_at": "now",
                "profile_id": "fast",
                "model_id": "gpt-5"
            }],
            "next_offset": 50,
            "total_count": 51
        }),
        json!({
            "invite_id": "invite",
            "url": "qmt://invite",
            "qr_code": "QR",
            "expires_at": 1,
            "max_uses": 1,
            "mesh_name": "Team"
        }),
    ]);
    let (state, events, mut rx) = harness(&connection);
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

    assert_eq!(
        connection.messages(),
        vec![
            RecordedMessage {
                method: "_querymt/session/setDelegateModel".into(),
                params: json!({
                    "session_id": "parent",
                    "agent_id": "coder",
                    "model_id": null,
                    "node_id": null
                }),
            },
            RecordedMessage {
                method: "_querymt/auth/status".into(),
                params: json!({}),
            },
            RecordedMessage {
                method: "_querymt/remote/sessions".into(),
                params: json!({ "node_id": "node", "offset": 0, "limit": 50 }),
            },
            RecordedMessage {
                method: "_querymt/mesh/createInvite".into(),
                params: json!({ "mesh_name": null, "ttl": null, "max_uses": null }),
            },
        ]
    );

    assert!(matches!(
        rx.try_recv().expect("delegate model event"),
        ServerChannelMsg::Acp(AcpAppEvent::DelegateModelSet {
            session_id,
            agent_id,
            model: None,
        }) if session_id == "parent" && agent_id == "coder"
    ));
    assert!(matches!(
        rx.try_recv().expect("auth providers event"),
        ServerChannelMsg::Acp(AcpAppEvent::AuthProviders(providers))
            if providers.len() == 1
                && providers[0].provider == "openai"
                && providers[0].display_name == "OpenAI"
                && providers[0].oauth_status == Some(OAuthStatus::Connected)
                && !providers[0].has_stored_api_key
                && providers[0].has_env_api_key
                && providers[0].env_var_name.as_deref() == Some("OPENAI_API_KEY")
                && providers[0].supports_oauth
                && providers[0].preferred_method == Some(AuthMethod::OAuth)
    ));
    assert!(matches!(
        rx.try_recv().expect("remote sessions event"),
        ServerChannelMsg::Acp(AcpAppEvent::RemoteSessions(list))
            if list.node_id == "node"
                && list.next_offset == Some(50)
                && list.total_count == 51
                && list.sessions.len() == 1
                && list.sessions[0].id == "remote"
                && list.sessions[0].node_id == "node"
                && list.sessions[0].node_label.as_deref() == Some("Remote Node")
                && list.sessions[0].title.as_deref() == Some("Remote Session")
                && list.sessions[0].cwd.as_deref() == Some("/remote")
                && list.sessions[0].updated_at.as_deref() == Some("now")
                && list.sessions[0].profile_id.as_deref() == Some("fast")
                && list.sessions[0].model_id.as_deref() == Some("gpt-5")
    ));
    assert!(matches!(
        rx.try_recv().expect("mesh invite event"),
        ServerChannelMsg::Acp(AcpAppEvent::MeshInviteCreated(invite))
            if invite.invite_id == "invite"
                && invite.url == "qmt://invite"
                && invite.qr_code.as_deref() == Some("QR")
                && invite.expires_at == 1
                && invite.max_uses == 1
                && invite.mesh_name.as_deref() == Some("Team")
    ));
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn initialize_pins_client_fields_gated_follow_ups_and_event_order() {
    let connection = RecordingConnection::with_responses([
        json!({
            "protocolVersion": 1,
            "agentCapabilities": {},
            "agentInfo": { "name": "querymt-agent", "title": "QueryMT Agent", "version": "1" }
        }),
        json!({
            "methods": ["querymt/mesh/nodes", "querymt/profiles"],
            "features": { "profiles": true }
        }),
        json!({ "nodes": [{ "id": "node-1", "label": "Remote" }] }),
        json!({
            "profiles": [{ "id": "fast", "name": "Fast" }],
            "active_profile_id": "fast"
        }),
    ]);
    let (state, events, mut rx) = harness(&connection);

    commands::dispatch(context(&connection, &state, &events), Command::Init)
        .await
        .expect("initialize");

    let messages = connection.messages();
    assert_eq!(
        messages
            .iter()
            .map(|message| message.method.as_str())
            .collect::<Vec<_>>(),
        [
            "initialize",
            "_querymt/capabilities",
            "_querymt/mesh/nodes",
            "_querymt/profiles",
        ]
    );
    assert_eq!(
        messages[0].params,
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": { "readTextFile": false, "writeTextFile": false },
                "terminal": false,
                "auth": { "terminal": false },
                "elicitation": { "form": {} }
            },
            "clientInfo": { "name": "qmtui", "version": env!("CARGO_PKG_VERSION") }
        })
    );
    assert_eq!(messages[1].params, json!({}));
    assert_eq!(messages[2].params, json!({}));
    assert_eq!(messages[3].params, json!({}));

    assert!(matches!(
        rx.try_recv().expect("initialized"),
        ServerChannelMsg::Acp(AcpAppEvent::Initialized {
            agent_id,
            agent_name,
            profiles,
            active_profile_id: None,
            agent_mode: Some(mode),
            reasoning_effort: Some(None),
        }) if agent_id == "querymt-agent"
            && agent_name == "QueryMT Agent"
            && profiles.is_empty()
            && mode == "build"
    ));
    assert!(matches!(
        rx.try_recv().expect("capabilities"),
        ServerChannelMsg::Acp(AcpAppEvent::ControlCapabilities(value))
            if value["methods"][0] == "querymt/mesh/nodes"
    ));
    assert!(matches!(
        rx.try_recv().expect("mesh nodes"),
        ServerChannelMsg::Acp(AcpAppEvent::MeshNodes(nodes))
            if nodes.nodes[0].id == "node-1"
    ));
    assert!(matches!(
        rx.try_recv().expect("profiles"),
        ServerChannelMsg::Acp(AcpAppEvent::Profiles { profiles, active_profile_id })
            if profiles[0].id == "fast" && active_profile_id.as_deref() == Some("fast")
    ));
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn initialize_omits_unadvertised_follow_ups_and_emits_exact_empty_profiles() {
    let connection = RecordingConnection::with_responses([
        json!({
            "protocolVersion": 1,
            "agentCapabilities": {},
            "agentInfo": { "name": "querymt-agent", "title": "QueryMT Agent", "version": "1" }
        }),
        json!({ "methods": [], "features": { "profiles": false } }),
    ]);
    let (state, events, mut rx) = harness(&connection);

    commands::dispatch(context(&connection, &state, &events), Command::Init)
        .await
        .expect("initialize without optional capabilities");

    let messages = connection.messages();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].method, "initialize");
    assert_eq!(
        messages[1],
        RecordedMessage {
            method: "_querymt/capabilities".into(),
            params: json!({}),
        }
    );
    assert!(messages.iter().all(|message| {
        message.method != "_querymt/mesh/nodes" && message.method != "_querymt/profiles"
    }));
    assert!(matches!(
        rx.try_recv().expect("initialized"),
        ServerChannelMsg::Acp(AcpAppEvent::Initialized {
            agent_id,
            agent_name,
            profiles,
            active_profile_id: None,
            agent_mode: Some(mode),
            reasoning_effort: Some(None),
        }) if agent_id == "querymt-agent"
            && agent_name == "QueryMT Agent"
            && profiles.is_empty()
            && mode == "build"
    ));
    assert!(matches!(
        rx.try_recv().expect("capabilities"),
        ServerChannelMsg::Acp(AcpAppEvent::ControlCapabilities(value))
            if value == json!({ "methods": [], "features": { "profiles": false } })
    ));
    assert!(matches!(
        rx.try_recv().expect("empty profiles"),
        ServerChannelMsg::Acp(AcpAppEvent::Profiles {
            profiles,
            active_profile_id: None,
        }) if profiles.is_empty()
    ));
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn standard_session_router_pins_new_prompt_delete_payloads_and_events() {
    let connection = RecordingConnection::with_responses([
        json!({ "sessionId": "session-1" }),
        json!({ "stopReason": "end_turn" }),
        json!({}),
    ]);
    let (state, events, mut rx) = harness(&connection);

    commands::dispatch(
        context(&connection, &state, &events),
        Command::NewSession {
            cwd: Some("/repo".into()),
            profile_id: Some("fast".into()),
        },
    )
    .await
    .expect("new session");
    commands::dispatch(
        context(&connection, &state, &events),
        Command::Prompt {
            prompt: vec![
                PromptBlock::Text {
                    text: "hello".into(),
                },
                PromptBlock::ResourceLink {
                    name: "guide".into(),
                    uri: "file:///repo/guide.md".into(),
                },
            ],
            local_id: "local-1".into(),
        },
    )
    .await
    .expect("prompt dispatch");

    let created = rx.recv().await.expect("created");
    let started = rx.recv().await.expect("turn started");
    let finished = tokio::time::timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("prompt completion timeout")
        .expect("finished");
    assert!(matches!(
        created,
        ServerChannelMsg::Acp(AcpAppEvent::SessionCreated {
            agent_id,
            session_id,
            profile_id: Some(profile_id),
        }) if agent_id == "querymt" && session_id == "session-1" && profile_id == "fast"
    ));
    assert!(matches!(
        started,
        ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate {
            session_id,
            update: AcpSessionUpdate::TurnStarted,
            is_replay: false,
        }) if session_id == "session-1"
    ));
    assert!(matches!(
        finished,
        ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate {
            session_id,
            update: AcpSessionUpdate::Finished { finish_reason },
            is_replay: false,
        }) if session_id == "session-1" && finish_reason == "EndTurn"
    ));

    commands::dispatch(
        context(&connection, &state, &events),
        Command::DeleteSession {
            session_id: "old-session".into(),
        },
    )
    .await
    .expect("delete session");

    let messages = connection.messages();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].method, "session/new");
    assert_eq!(
        messages[0].params,
        json!({
            "cwd": "/repo",
            "mcpServers": [],
            "_meta": { "querymt": { "profile_id": "fast" } }
        })
    );
    assert_eq!(messages[1].method, "session/prompt");
    assert_eq!(
        messages[1].params,
        json!({
            "sessionId": "session-1",
            "prompt": [
                { "type": "text", "text": "hello" },
                { "type": "resource_link", "name": "guide", "uri": "file:///repo/guide.md" }
            ]
        })
    );
    assert_eq!(
        messages[2],
        RecordedMessage {
            method: "session/delete".into(),
            params: json!({ "sessionId": "old-session" }),
        }
    );
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn catalog_router_pins_config_model_refresh_payloads_and_events() {
    let option = json!({
        "id": "mode",
        "name": "Mode",
        "type": "select",
        "currentValue": "plan",
        "options": [{ "value": "plan", "name": "Plan" }]
    });
    let connection = RecordingConnection::with_responses([
        json!({ "configOptions": [option.clone()] }),
        json!({ "configOptions": [] }),
        json!({ "configOptions": [] }),
        json!({
            "models": [{
                "id": "openai/gpt-5",
                "label": "GPT-5",
                "provider": "openai",
                "model": "gpt-5"
            }]
        }),
    ]);
    let (state, events, mut rx) = harness(&connection);
    state.set_current_session_id("session-1").await;

    for command in [
        Command::SetAgentMode {
            mode: "plan".into(),
        },
        Command::SetReasoningEffort {
            reasoning_effort: "high".into(),
        },
        Command::SetSessionModel {
            session_id: "session-1".into(),
            model_id: "openai/gpt-5".into(),
            node_id: Some("node-1".into()),
        },
        Command::ListAllModels { refresh: true },
    ] {
        commands::dispatch(context(&connection, &state, &events), command)
            .await
            .expect("catalog command");
    }

    let messages = connection.messages();
    assert_eq!(
        messages
            .iter()
            .map(|message| message.method.as_str())
            .collect::<Vec<_>>(),
        [
            "session/set_config_option",
            "session/set_config_option",
            "session/set_config_option",
            "_querymt/refreshModels",
        ]
    );
    assert_eq!(
        messages[0].params,
        json!({ "sessionId": "session-1", "configId": "mode", "value": "plan" })
    );
    assert_eq!(
        messages[1].params,
        json!({ "sessionId": "session-1", "configId": "reasoning_effort", "value": "high" })
    );
    assert_eq!(messages[2].params["sessionId"], "session-1");
    assert_eq!(messages[2].params["configId"], "model");
    assert_eq!(messages[2].params["value"], "openai/gpt-5");
    assert_eq!(
        messages[2].params["_meta"]["querymt"]["modelEntry"]["node_id"],
        "node-1"
    );
    assert_eq!(messages[3].params, json!({ "wait_for_completion": true }));

    assert!(matches!(
        rx.try_recv().expect("mode"),
        ServerChannelMsg::Acp(AcpAppEvent::AgentMode { mode }) if mode == "plan"
    ));
    assert!(matches!(
        rx.try_recv().expect("model log"),
        ServerChannelMsg::Acp(AcpAppEvent::InfoLog { target: "acp", message })
            if message.contains("model=gpt-5")
                && message.contains("id=openai/gpt-5")
                && message.contains("node=node-1")
    ));
    assert!(matches!(
        rx.try_recv().expect("selected provider"),
        ServerChannelMsg::Acp(AcpAppEvent::ProviderChanged { provider, model, .. })
            if provider == "openai" && model == "gpt-5"
    ));
    assert!(matches!(
        rx.try_recv().expect("models"),
        ServerChannelMsg::Acp(AcpAppEvent::Models { models, .. })
            if models[0].id == "openai/gpt-5"
    ));
    assert!(matches!(
        rx.try_recv().expect("default provider"),
        ServerChannelMsg::Acp(AcpAppEvent::ProviderChanged { provider, model, .. })
            if provider == "openai" && model == "gpt-5"
    ));
    assert!(rx.try_recv().is_err());
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
async fn extension_families_pin_auth_history_mesh_profile_methods_payloads_and_events() {
    let connection = RecordingConnection::with_responses([
        json!({
            "flow_id": "flow-1",
            "provider": "openai",
            "authorization_url": "https://example.test/authorize",
            "flow_kind": "redirect_code"
        }),
        json!({ "provider": "openai", "success": true, "message": "connected" }),
        json!({ "provider": "openai", "success": false, "message": "disconnect failed" }),
        json!({
            "success": true,
            "message_id": "u1",
            "reverted_files": ["src/lib.rs"],
            "message": "undone",
            "undo_stack": [{ "message_id": "u1" }]
        }),
        json!({ "success": true, "message": "redone", "undo_stack": [] }),
        json!({ "sessionId": "forked" }),
        json!({
            "enabled": true,
            "peer_id": "peer-1",
            "transport": "relay",
            "known_peer_count": 1,
            "has_invite_store": true,
            "has_mesh_state_store": false,
            "scopes": [{ "kind": "team", "id": "scope-1" }]
        }),
        json!({
            "nodes": [{
                "id": "node-1",
                "label": "Remote",
                "capabilities": ["sessions"],
                "active_sessions": 2,
                "transport": "relay",
                "last_seen_at": "now"
            }]
        }),
        json!({
            "session_id": "remote-1",
            "node_id": "node-1",
            "attached": true,
            "config_options": [{ "id": "mode" }],
            "snapshot": { "cursor": 1 }
        }),
        json!({
            "session_id": "remote-2",
            "node_id": "node-1",
            "attached": false,
            "config_options": [],
            "snapshot": null
        }),
        json!({
            "profiles": [{
                "id": "fast",
                "name": "Fast",
                "description": "Fast profile",
                "tags": ["quick"],
                "source": "local",
                "config_kind": "inline",
                "fingerprint": "abc"
            }],
            "active_profile_id": "fast"
        }),
        json!({
            "profile_id": "fast",
            "agents": [{
                "id": "coder",
                "name": "Coder",
                "description": "Writes code",
                "capabilities": ["edit"]
            }]
        }),
    ]);
    let (state, events, mut rx) = harness(&connection);
    state.set_current_session_id("session-1").await;

    for command in [
        Command::StartOAuthLogin {
            provider: "openai".into(),
        },
        Command::CompleteOAuthLogin {
            flow_id: "flow-1".into(),
            response: "code-1".into(),
        },
        Command::DisconnectOAuth {
            provider: "openai".into(),
        },
        Command::Undo {
            message_id: "u1".into(),
        },
        Command::Redo,
        Command::ForkSession {
            message_id: "u1".into(),
        },
        Command::ListRemoteNodes,
        Command::CreateRemoteSession {
            node_id: "node-1".into(),
            cwd: None,
        },
        Command::AttachRemoteSession {
            node_id: "node-1".into(),
            session_id: "remote-2".into(),
        },
        Command::ListProfiles,
        Command::ListProfileAgents {
            profile_id: "fast".into(),
        },
    ] {
        commands::dispatch(context(&connection, &state, &events), command)
            .await
            .expect("extension command");
    }

    assert_eq!(
        connection.messages(),
        vec![
            RecordedMessage {
                method: "_querymt/auth/start".into(),
                params: json!({ "provider": "openai" }),
            },
            RecordedMessage {
                method: "_querymt/auth/complete".into(),
                params: json!({ "flow_id": "flow-1", "response": "code-1" }),
            },
            RecordedMessage {
                method: "_querymt/auth/logout".into(),
                params: json!({ "provider": "openai" }),
            },
            RecordedMessage {
                method: "_querymt/session/undo".into(),
                params: json!({ "session_id": "session-1", "message_id": "u1" }),
            },
            RecordedMessage {
                method: "_querymt/session/redo".into(),
                params: json!({ "session_id": "session-1" }),
            },
            RecordedMessage {
                method: "session/fork".into(),
                params: json!({
                    "sessionId": "session-1",
                    "cwd": "/launch",
                    "_meta": { "querymt": { "message_id": "u1" } }
                }),
            },
            RecordedMessage {
                method: "_querymt/mesh/status".into(),
                params: json!({}),
            },
            RecordedMessage {
                method: "_querymt/mesh/nodes".into(),
                params: json!({}),
            },
            RecordedMessage {
                method: "_querymt/remote/createSession".into(),
                params: json!({ "node_id": "node-1", "cwd": null, "attach": true }),
            },
            RecordedMessage {
                method: "_querymt/remote/attachSession".into(),
                params: json!({ "node_id": "node-1", "session_id": "remote-2" }),
            },
            RecordedMessage {
                method: "_querymt/profiles".into(),
                params: json!({}),
            },
            RecordedMessage {
                method: "_querymt/profile/agents".into(),
                params: json!({ "profile_id": "fast" }),
            },
        ]
    );

    let emitted = std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>();
    assert_eq!(emitted.len(), 12);
    assert!(matches!(
        &emitted[0],
        ServerChannelMsg::Acp(AcpAppEvent::OAuthFlowStarted(flow))
            if flow.flow_id == "flow-1"
                && flow.provider == "openai"
                && flow.authorization_url == "https://example.test/authorize"
                && flow.flow_kind == OAuthFlowKind::RedirectCode
    ));
    assert!(matches!(
        &emitted[1],
        ServerChannelMsg::Acp(AcpAppEvent::OAuthResult(result))
            if result.provider == "openai"
                && result.status == OAuthResultStatus::Success
                && result.message == "connected"
    ));
    assert!(matches!(
        &emitted[2],
        ServerChannelMsg::Acp(AcpAppEvent::OAuthResult(result))
            if result.provider == "openai"
                && result.status == OAuthResultStatus::Failure
                && result.message == "disconnect failed"
    ));
    assert!(matches!(
        &emitted[3],
        ServerChannelMsg::Acp(AcpAppEvent::UndoResult(result))
            if result == &UndoResult::Applied {
                target_message_id: Some("u1".into()),
                reverted_files: vec!["src/lib.rs".into()],
                message: Some("undone".into()),
                stack: UndoStackSnapshot { message_ids: vec!["u1".into()] },
            }
    ));
    assert!(matches!(
        &emitted[4],
        ServerChannelMsg::Acp(AcpAppEvent::RedoResult(result))
            if result == &RedoResult::Applied {
                message: Some("redone".into()),
                stack: UndoStackSnapshot::default(),
            }
    ));
    assert!(matches!(
        &emitted[5],
        ServerChannelMsg::Acp(AcpAppEvent::ForkResult(result))
            if result == &ForkResult::Succeeded {
                source_session_id: Some("session-1".into()),
                forked_session_id: Some("forked".into()),
                message: None,
            }
    ));
    assert!(matches!(
        &emitted[6],
        ServerChannelMsg::Acp(AcpAppEvent::MeshStatus(status))
            if status.enabled
                && status.peer_id.as_deref() == Some("peer-1")
                && status.transport.as_deref() == Some("relay")
                && status.known_peer_count == 1
                && status.has_invite_store
                && !status.has_mesh_state_store
                && status.scopes.len() == 1
                && status.scopes[0].kind == "team"
                && status.scopes[0].id == "scope-1"
    ));
    assert!(matches!(
        &emitted[7],
        ServerChannelMsg::Acp(AcpAppEvent::MeshNodes(nodes))
            if nodes.nodes.len() == 1
                && nodes.nodes[0].id == "node-1"
                && nodes.nodes[0].label == "Remote"
                && nodes.nodes[0].capabilities == ["sessions"]
                && nodes.nodes[0].active_sessions == 2
                && nodes.nodes[0].transport == "relay"
                && nodes.nodes[0].last_seen_at.as_deref() == Some("now")
    ));
    assert!(matches!(
        &emitted[8],
        ServerChannelMsg::Acp(AcpAppEvent::RemoteSessionAttached(info))
            if info.session_id == "remote-1"
                && info.node_id == "node-1"
                && info.attached
                && info.config_options == [json!({ "id": "mode" })]
                && info.snapshot == Some(json!({ "cursor": 1 }))
    ));
    assert!(matches!(
        &emitted[9],
        ServerChannelMsg::Acp(AcpAppEvent::RemoteSessionAttached(info))
            if info.session_id == "remote-2"
                && info.node_id == "node-1"
                && !info.attached
                && info.config_options.is_empty()
                && info.snapshot.is_none()
    ));
    assert!(matches!(
        &emitted[10],
        ServerChannelMsg::Acp(AcpAppEvent::Profiles {
            profiles,
            active_profile_id,
        }) if profiles.len() == 1
            && profiles[0].id == "fast"
            && profiles[0].name == "Fast"
            && profiles[0].description.as_deref() == Some("Fast profile")
            && profiles[0].tags == ["quick"]
            && profiles[0].source.as_deref() == Some("local")
            && profiles[0].config_kind.as_deref() == Some("inline")
            && profiles[0].fingerprint.as_deref() == Some("abc")
            && active_profile_id.as_deref() == Some("fast")
    ));
    assert!(matches!(
        &emitted[11],
        ServerChannelMsg::Acp(AcpAppEvent::ProfileAgents { profile_id, agents })
            if profile_id == "fast"
                && agents.len() == 1
                && agents[0].id == "coder"
                && agents[0].name == "Coder"
                && agents[0].description.as_deref() == Some("Writes code")
                && agents[0].capabilities == ["edit"]
    ));
}

#[tokio::test]
async fn extension_response_strictness_and_tolerance_are_preserved() {
    let strict = RecordingConnection::with_responses([json!({})]);
    let (strict_state, strict_events, mut strict_rx) = harness(&strict);
    commands::dispatch(
        context(&strict, &strict_state, &strict_events),
        Command::ListProfiles,
    )
    .await
    .expect("profile errors are reported as availability events");
    assert!(matches!(
        strict_rx.try_recv().expect("strict profile error"),
        ServerChannelMsg::Acp(AcpAppEvent::InfoLog { target: "profiles", message })
            if message.starts_with("profile catalog unavailable: ")
    ));

    let tolerant = RecordingConnection::with_responses([json!({ "malformed": true })]);
    let (tolerant_state, tolerant_events, mut tolerant_rx) = harness(&tolerant);
    commands::dispatch(
        context(&tolerant, &tolerant_state, &tolerant_events),
        Command::StartOAuthLogin {
            provider: "openai".into(),
        },
    )
    .await
    .expect("malformed optional auth result is tolerated");
    assert!(tolerant_rx.try_recv().is_err());
}

#[tokio::test]
async fn elicitation_response_dispatches_the_registered_websocket_shape_once() {
    let connection = RecordingConnection::default();
    let (state, events, _rx) = harness(&connection);
    let (wire_tx, mut wire_rx) = mpsc::unbounded_channel();
    state
        .elicitations
        .insert(
            "e1".into(),
            PendingResponse::WebSocketResponse {
                peer: Peer::new(wire_tx),
                id: json!(41),
            },
        )
        .await;

    commands::dispatch(
        context(&connection, &state, &events),
        Command::ElicitationResponse {
            elicitation_id: "e1".into(),
            action: "accept".into(),
            content: Some(json!({ "selection": "yes" })),
        },
    )
    .await
    .expect("elicitation response");
    let Message::Text(text) = wire_rx.try_recv().expect("wire response") else {
        panic!("text response");
    };
    assert_eq!(
        serde_json::from_str::<Value>(&text).expect("response JSON"),
        json!({
            "jsonrpc": "2.0",
            "id": 41,
            "result": { "action": "accept", "content": { "selection": "yes" } }
        })
    );
    assert!(wire_rx.try_recv().is_err());
    assert!(connection.messages().is_empty());
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
    let messages = connection.messages();
    assert_eq!(
        messages
            .iter()
            .map(|message| message.method.as_str())
            .collect::<Vec<_>>(),
        ["session/load", "_querymt/session/undoStack"]
    );
    assert_eq!(
        messages[0].params,
        json!({ "sessionId": "session", "cwd": "/repo", "mcpServers": [] })
    );
    assert_eq!(messages[1].params, json!({ "session_id": "session" }));
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

fn normalized_event(message: ServerChannelMsg) -> Value {
    match message {
        ServerChannelMsg::Acp(AcpAppEvent::SessionUpdate {
            session_id,
            update,
            is_replay,
        }) => {
            let update = match update {
                AcpSessionUpdate::AssistantContentDelta {
                    content,
                    message_id,
                } => {
                    json!({ "kind": "assistant", "content": content, "messageId": message_id })
                }
                AcpSessionUpdate::AssistantThinkingDelta {
                    content,
                    message_id,
                } => {
                    json!({ "kind": "thought", "content": content, "messageId": message_id })
                }
                AcpSessionUpdate::AssistantMessage {
                    content,
                    thinking,
                    message_id,
                } => json!({
                    "kind": "assistantMessage", "content": content,
                    "thinking": thinking, "messageId": message_id
                }),
                AcpSessionUpdate::ToolCallStart {
                    tool_call_id,
                    name,
                    arguments,
                } => json!({
                    "kind": "toolStart", "toolCallId": tool_call_id,
                    "name": name, "arguments": arguments
                }),
                AcpSessionUpdate::ToolCallEnd {
                    tool_call_id,
                    name,
                    is_error,
                    result,
                } => json!({
                    "kind": "toolEnd", "toolCallId": tool_call_id,
                    "name": name, "isError": is_error, "result": result
                }),
                AcpSessionUpdate::UsageUpdate {
                    used,
                    size,
                    cost_usd,
                } => json!({
                    "kind": "usage", "used": used, "size": size, "costUsd": cost_usd
                }),
                other => panic!("unexpected normalized session update: {other:?}"),
            };
            json!({
                "event": "sessionUpdate", "sessionId": session_id,
                "update": update, "isReplay": is_replay
            })
        }
        ServerChannelMsg::Acp(AcpAppEvent::AgentMode { mode }) => {
            json!({ "event": "agentMode", "mode": mode })
        }
        ServerChannelMsg::Acp(AcpAppEvent::ReasoningEffort { reasoning_effort }) => {
            json!({ "event": "reasoningEffort", "value": reasoning_effort })
        }
        other => panic!("unexpected normalized event: {other:?}"),
    }
}

#[tokio::test]
async fn stdio_and_websocket_standard_inbound_share_normalized_events() {
    let mode_option = acp::SessionConfigOption::select(
        "reasoning_effort",
        "Reasoning effort",
        "high",
        vec![acp::SessionConfigSelectOption::new("high", "High")],
    );
    let notifications = vec![
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::AgentThoughtChunk(
                acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new("think")))
                    .message_id(Some(acp::MessageId::from("a1"))),
            ),
        ),
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::AgentMessageChunk(
                acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new("answer")))
                    .message_id(Some(acp::MessageId::from("a1"))),
            ),
        ),
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::ToolCall(
                acp::ToolCall::new("tool-1", "Run shell").raw_input(json!({ "cmd": "pwd" })),
            ),
        ),
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(
                "tool-1",
                acp::ToolCallUpdateFields::new()
                    .title("Run shell".to_string())
                    .status(acp::ToolCallStatus::Completed)
                    .raw_output(json!("/repo")),
            )),
        ),
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::CurrentModeUpdate(acp::CurrentModeUpdate::new("plan")),
        ),
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::ConfigOptionUpdate(acp::ConfigOptionUpdate::new(vec![mode_option])),
        ),
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::UsageUpdate(
                acp::UsageUpdate::new(8, 16).cost(acp::Cost::new(0.5, "usd")),
            ),
        ),
        acp::SessionNotification::new(
            "session",
            acp::SessionUpdate::SessionInfoUpdate(acp::SessionInfoUpdate::new().title("ignored")),
        ),
    ];
    let (stdio_tx, mut stdio_rx) = mpsc::unbounded_channel();
    let stdio_events = EventSink::new(stdio_tx);
    let stdio_state = Arc::new(RuntimeState::new(None));
    let (wire_tx, _wire_rx) = mpsc::unbounded_channel::<Message>();
    let peer = Peer::new(wire_tx);
    let connection = RecordingConnection::default();
    let (ws_tx, mut ws_rx) = mpsc::unbounded_channel();
    let ws_events = EventSink::new(ws_tx);
    let ws_state = Arc::new(RuntimeState::new(None));

    for notification in notifications {
        inbound::session_notification(&stdio_state, &stdio_events, notification.clone()).await;
        let text = serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": serde_json::to_value(notification).expect("notification params")
        }))
        .expect("websocket envelope");
        inbound::websocket_text(&peer, &connection, &ws_state, &ws_events, &text)
            .await
            .expect("websocket inbound");
    }

    let stdio = std::iter::from_fn(|| stdio_rx.try_recv().ok())
        .map(normalized_event)
        .collect::<Vec<_>>();
    let websocket = std::iter::from_fn(|| ws_rx.try_recv().ok())
        .map(normalized_event)
        .collect::<Vec<_>>();
    assert_eq!(stdio, websocket);
    assert_eq!(
        stdio,
        vec![
            json!({ "event": "sessionUpdate", "sessionId": "session", "update": {
                "kind": "thought", "content": "think", "messageId": "a1"
            }, "isReplay": false }),
            json!({ "event": "sessionUpdate", "sessionId": "session", "update": {
                "kind": "assistant", "content": "answer", "messageId": "a1"
            }, "isReplay": false }),
            json!({ "event": "sessionUpdate", "sessionId": "session", "update": {
                "kind": "assistantMessage", "content": "answer", "thinking": "think", "messageId": "a1"
            }, "isReplay": false }),
            json!({ "event": "sessionUpdate", "sessionId": "session", "update": {
                "kind": "toolStart", "toolCallId": "tool-1", "name": "shell",
                "arguments": { "cmd": "pwd" }
            }, "isReplay": false }),
            json!({ "event": "sessionUpdate", "sessionId": "session", "update": {
                "kind": "toolEnd", "toolCallId": "tool-1", "name": "shell",
                "isError": false, "result": "/repo"
            }, "isReplay": false }),
            json!({ "event": "agentMode", "mode": "plan" }),
            json!({ "event": "reasoningEffort", "value": "high" }),
            json!({ "event": "sessionUpdate", "sessionId": "session", "update": {
                "kind": "usage", "used": 8, "size": 16, "costUsd": 0.5
            }, "isReplay": false }),
        ]
    );
}
