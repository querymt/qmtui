# Examples

## qmtacp

`qmtacp` is a small ACP WebSocket client for probing an agent with the same ACP v1 protocol types used by qmtui. It advertises only its minimal probe-client capabilities and connects to `ws://127.0.0.1:3030/ws` by default.

The client is intentionally non-interactive: permission requests select a reject option when available, otherwise they are cancelled. Prompts that require tool authorization can therefore fail; use qmtui when interactive approval is needed.

Run the default initialize + new-session handshake:

```sh
cargo run --example qmtacp
```

Use another endpoint or command:

```sh
cargo run --example qmtacp -- --url ws://127.0.0.1:3030/ws list
cargo run --example qmtacp -- new --cwd /path/to/project --profile default
cargo run --example qmtacp -- load SESSION_ID
cargo run --example qmtacp -- prompt SESSION_ID "Hello"
```

Run `cargo run --example qmtacp -- --help` for all options and raw JSON-RPC calls.
