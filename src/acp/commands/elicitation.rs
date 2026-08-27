use super::super::connection::AcpConnection;
use super::super::context::CommandContext;

pub(super) async fn respond<C: AcpConnection>(
    ctx: CommandContext<'_, C>,
    elicitation_id: String,
    action: String,
    content: Option<serde_json::Value>,
) {
    ctx.state
        .elicitations
        .respond(&elicitation_id, &action, content)
        .await;
}
