use agent_client_protocol as acp_sdk;
use serde_json::{Value, json};

use super::call;
use crate::acp::connection::AcpConnection;

pub(in crate::acp) async fn get<C: AcpConnection>(connection: &C) -> Result<Value, acp_sdk::Error> {
    call(connection, "querymt/capabilities", json!({})).await
}
