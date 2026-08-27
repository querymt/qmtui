use crate::acp_state::AcpAppEvent;
use crate::app::ConnectionEvent;

#[derive(Debug)]
pub(crate) enum ConnectionManagerEvent {
    State(ConnectionEvent),
}

#[derive(Debug)]
pub(crate) enum ServerChannelMsg {
    Acp(AcpAppEvent),
}
