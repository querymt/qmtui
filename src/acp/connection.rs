use std::future::Future;

use agent_client_protocol::{self as acp_sdk, JsonRpcNotification, JsonRpcRequest};

pub(super) trait AcpConnection: Clone + Send + Sync + 'static {
    fn request<R>(
        &self,
        request: R,
    ) -> impl Future<Output = Result<R::Response, acp_sdk::Error>> + Send
    where
        R: JsonRpcRequest + Send + Sync + 'static,
        R::Response: Send + 'static;

    fn notify<N>(&self, notification: N) -> Result<(), acp_sdk::Error>
    where
        N: JsonRpcNotification + Send + Sync + 'static;

    fn spawn(
        &self,
        fut: impl Future<Output = Result<(), acp_sdk::Error>> + Send + 'static,
    ) -> Result<(), acp_sdk::Error>;
}

pub(super) fn internal_error(message: impl ToString) -> acp_sdk::Error {
    acp_sdk::Error::internal_error().data(message.to_string())
}
