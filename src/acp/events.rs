use tokio::sync::mpsc;

use crate::acp_state::{AcpAppEvent, AcpSessionUpdate};
use crate::runtime_events::ServerChannelMsg;

#[derive(Clone)]
pub(super) struct EventSink {
    tx: mpsc::UnboundedSender<ServerChannelMsg>,
}

impl EventSink {
    pub(super) fn new(tx: mpsc::UnboundedSender<ServerChannelMsg>) -> Self {
        Self { tx }
    }

    pub(super) fn send(&self, event: AcpAppEvent) {
        let _ = self.tx.send(ServerChannelMsg::Acp(event));
    }

    pub(super) fn error(&self, message: impl Into<String>) {
        self.send(AcpAppEvent::Error {
            message: message.into(),
        });
    }

    pub(super) fn info(&self, target: &'static str, message: impl Into<String>) {
        self.send(AcpAppEvent::InfoLog {
            target,
            message: message.into(),
        });
    }

    pub(super) fn session_update(&self, session_id: &str, update: AcpSessionUpdate) {
        self.send(AcpAppEvent::SessionUpdate {
            session_id: session_id.to_string(),
            update,
            is_replay: false,
        });
    }
}
