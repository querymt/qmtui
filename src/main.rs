#[tokio::main]
async fn main() -> anyhow::Result<()> {
    qmtui::runtime::run().await
}
