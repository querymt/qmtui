use std::future::Future;
use std::sync::Arc;

use agent_client_protocol::{
    self as acp_sdk, AcpAgent, Agent, Client, ConnectionTo, JsonRpcNotification, JsonRpcRequest,
    UntypedMessage, schema::v1 as acp,
};
use tokio::sync::mpsc;

use crate::app::ConnectionEvent;
use crate::command::Command;
use crate::runtime_events::{ConnectionManagerEvent, ServerChannelMsg};

use super::super::commands;
use super::super::connection::AcpConnection;
use super::super::context::CommandContext;
use super::super::elicitation::{self, PendingResponse};
use super::super::events::EventSink;
use super::super::inbound;
use super::super::notification;
use super::super::runtime::RuntimeState;

impl AcpConnection for ConnectionTo<Agent> {
    async fn request<R>(&self, request: R) -> Result<R::Response, acp_sdk::Error>
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static,
    {
        self.send_request(request).block_task().await
    }

    fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
    where
        N: JsonRpcNotification + Send + Sync + 'static,
    {
        self.send_notification(notification)
    }

    fn spawn(
        &self,
        future: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
    ) -> Result<(), acp_sdk::Error> {
        self.spawn(future)
    }
}

pub(in crate::acp) async fn run(
    argv: Vec<String>,
    cmd_rx: &mut mpsc::UnboundedReceiver<Command>,
    srv_tx: mpsc::UnboundedSender<ServerChannelMsg>,
    conn_tx: mpsc::UnboundedSender<ConnectionManagerEvent>,
    launch_cwd: Option<String>,
) -> Result<(), acp_sdk::Error> {
    let agent = AcpAgent::from_args(argv)?;
    let state = Arc::new(RuntimeState::new(launch_cwd));
    let events = EventSink::new(srv_tx);

    Client
        .builder()
        .on_receive_notification(
            {
                let state = state.clone();
                let events = events.clone();
                async move |notification: acp::SessionNotification, _cx| {
                    inbound::session_notification(&state, &events, notification).await;
                    Ok(())
                }
            },
            acp_sdk::on_receive_notification!(),
        )
        .on_receive_notification(
            {
                let events = events.clone();
                async move |notification: UntypedMessage, cx| {
                    let (method, params) = notification.clone().into_parts();
                    if inbound::extension_notification(&method)
                        == Some(inbound::ExtensionNotification::Delegation)
                    {
                        inbound::delegation_notification(&events, params);
                        Ok(acp_sdk::Handled::Yes)
                    } else {
                        Ok(acp_sdk::Handled::No {
                            message: (notification, cx),
                            retry: false,
                        })
                    }
                }
            },
            acp_sdk::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: acp::RequestPermissionRequest, responder, _cx| {
                responder.respond(elicitation::permission_response(&request))
            },
            acp_sdk::on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = state.clone();
                let events = events.clone();
                async move |request: acp::CreateElicitationRequest, responder, _cx| {
                    register_elicitation(&state, &events, request, responder).await;
                    Ok(())
                }
            },
            acp_sdk::on_receive_request!(),
        )
        .connect_with(agent, |connection: ConnectionTo<Agent>| async move {
            let _ = conn_tx.send(ConnectionManagerEvent::State(ConnectionEvent::Connected));
            while let Some(command) = cmd_rx.recv().await {
                let context = CommandContext {
                    connection: &connection,
                    state: &state,
                    events: &events,
                };
                if let Err(err) = commands::dispatch(context, command).await {
                    events.error(format!("acp request failed: {err:?}"));
                }
            }
            state.elicitations.clear().await;
            state.assistants.clear().await;
            Ok(())
        })
        .await
}

async fn register_elicitation(
    state: &Arc<RuntimeState>,
    events: &EventSink,
    request: acp::CreateElicitationRequest,
    responder: acp_sdk::Responder<acp::CreateElicitationResponse>,
) {
    let session_id = elicitation::request_session_id(&request);
    notification::flush_assistant(state, events, &session_id).await;
    let responder_id = responder.id();
    let elicitation_id = responder_id
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| responder_id.to_string());
    let update = elicitation::requested_update(elicitation_id.clone(), request, "acp");
    state
        .elicitations
        .insert(elicitation_id, PendingResponse::Sdk(responder))
        .await;
    events.session_update(&session_id, update);
}
