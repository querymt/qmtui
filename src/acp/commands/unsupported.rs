use crate::command::Command;

use super::super::connection::AcpConnection;
use super::super::context::CommandContext;

pub(super) fn file_index<C: AcpConnection>(ctx: CommandContext<'_, C>) {
    ctx.events
        .error("file mentions are not exposed in the acp subset yet");
}

pub(super) fn command<C: AcpConnection>(ctx: CommandContext<'_, C>, command: &Command) {
    ctx.events.error(format!(
        "unsupported in the current acp subset: {command:?}"
    ));
}
