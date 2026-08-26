use std::sync::Arc;

use super::connection::AcpConnection;
use super::events::EventSink;
use super::runtime::RuntimeState;

pub(super) struct CommandContext<'a, C: AcpConnection> {
    pub(super) connection: &'a C,
    pub(super) state: &'a Arc<RuntimeState>,
    pub(super) events: &'a EventSink,
}

impl<'a, C: AcpConnection> Copy for CommandContext<'a, C> {}

impl<C: AcpConnection> Clone for CommandContext<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}
