use agent_client_protocol::{self as acp_sdk, UntypedMessage};
use serde_json::Value;

use super::connection::AcpConnection;

pub(super) mod auth;
pub(super) mod capabilities;
pub(super) mod delegation;
pub(super) mod history;
pub(super) mod mesh;
pub(super) mod models;
pub(super) mod profiles;

pub(super) async fn call<C: AcpConnection>(
    connection: &C,
    method: &str,
    params: Value,
) -> Result<Value, acp_sdk::Error> {
    let wire_method = format!("_{method}");
    connection
        .request(UntypedMessage::new(&wire_method, params)?)
        .await
}

pub(super) fn payload(response: &Value) -> &Value {
    response.get("data").unwrap_or(response)
}
