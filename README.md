# qmtui

`qmtui` is the Rust terminal interface for QueryMT agents. It connects to an
Agent Client Protocol (ACP) endpoint, translates terminal and ACP activity into
application events, updates feature-owned state, executes explicit effects, and
renders the result with Ratatui.

## Architecture

The binary entry point is intentionally small: `src/main.rs` installs Tokio and
calls `qmtui::runtime::run()`. The library is organized around these layers:

| Area | Responsibility |
| --- | --- |
| `src/domain/` | Semantic data shared by state and features; no terminal, UI, transport, channel, filesystem, or process dependencies. |
| `src/*_state.rs` | Feature-owned semantic state and reducer operations. `src/render_state.rs` separately owns layout metrics, render caches, and Ratatui primitives. |
| `src/features/*/input/` | Feature input interpretation. These modules may consume Crossterm key types but do not perform I/O or know about rendering and ACP transport. |
| `src/features/*/view/` | Feature rendering with Ratatui. Views receive state or narrow render inputs and do not know about ACP transport. |
| `src/application.rs` | The `AppEvent`, `Effect`, and `update` boundary for application transitions. |
| `src/handlers.rs` | Root input routing and cross-feature composition over the feature input modules. |
| `src/acp/` | ACP connections, transport, protocol translation, replay, and command dispatch. This subsystem emits application-facing ACP events and has no UI dependency. |
| `src/runtime/` | Process composition: endpoint selection, channels, terminal lifecycle, event scheduling, persistence, clipboard/editor work, and effect execution. |
| `src/ui/mod.rs` | Root screen and popup composition through `ui::draw`; detailed rendering remains in feature view modules. |
| `src/app.rs` | Crate-local composition root joining the twelve feature/state owners plus `should_quit`. |

`App` fields are `pub(crate)` deliberately. Routing, persistence, rendering,
and runtime composition need to combine multiple owners, while public getters
for every field would broaden rather than improve the internal boundary.
Reducers are pure with respect to external side effects: they update in-memory
state and return `Effect` values for the runtime to execute.

## Event And Effect Flow

The application follows one production event path:

```text
Crossterm key/mouse       ACP connection       runtime/connection/supervisor
        |                       |                           |
        +-----------------------+---------------------------+
                                v
                 runtime::event_loop -> AppEvent
                                |
                                v
                     application::update
                       /              \
              handlers + feature      ACP/state reducers
                    input modules      and coordination
                       \              /
                         state changes
                         + Vec<Effect>
                                |
                                v
                    runtime::EffectExecutor
       command channel / persistence / clipboard / editor / terminal / quit
                                |
                      RuntimeEvent feedback
                                |
                       application::update
                                |
                                v
             ui::draw -> feature views -> RenderState
```

`runtime::event_loop::run_loop` schedules terminal, ACP, connection, supervisor,
and tick events without letting a permanently ready source starve the others.
It draws the current state, sends each event through `application::update`, and
passes returned effects to `runtime::EffectExecutor::execute`. Runtime work that
needs application feedback is converted to `RuntimeEvent` and re-enters the
same update path rather than mutating state from a background adapter.

ACP input follows the same boundary. `acp::inbound` decodes WebSocket or shared
connection input, `acp::notification::translate` converts SDK notifications to
application-facing updates, and `acp::notification::apply` handles replay and
stream ordering before emitting `AcpAppEvent`. The runtime event loop wraps that
as `AppEvent::Acp`; ACP code never renders or imports `crate::ui`.

## Boundary Rules

The architectural guidelines are:

- Domain modules stay free of UI/terminal crates, Tokio synchronization,
  filesystem/process access, ACP transport, and protocol DTO imports.
- Top-level semantic `*_state.rs` modules stay free of Ratatui, Crossterm,
  channels, I/O, process spawning, and ACP transport.
- Feature input may use Crossterm but not Ratatui, Tokio synchronization, I/O,
  process spawning, or ACP transport. Feature views and root UI may use Ratatui
  but not ACP transport.
- ACP may use Tokio, JSON-RPC, SDK, and transport types, but not `crate::ui`.
- Removed migration surfaces stay removed: legacy server-message symbols,
  diagnostic forwarding methods on `App`, root UI temporary exports, broad
  `use crate::handlers::*`, and crate-wide dead-code suppression.
- `src/main.rs` remains a thin call to `qmtui::runtime::run()`, and the twelve
  `App` owners plus `should_quit` remain `pub(crate)`.

Intentional exceptions are narrow and documented:

- `src/render_state.rs`, feature views, and `src/ui/` may use Ratatui because
  they own render caches, layout primitives, or rendering.
- Feature input may use Crossterm key types. Test modules may use `use super::*`.
- ACP owns its Tokio, JSON-RPC, SDK, and transport dependencies. Runtime,
  configuration, session, and persistence adapters may own channels, files,
  subprocesses, terminal operations, and other side effects.
- `src/themes_gen.rs` retains its generated crate-level `allow(dead_code)`;
  targeted item-level allowances remain valid for retained decoded fields.
- The `cfg(test)` App-to-render adapters in `src/features/chat/view/mod.rs` are
  retained for root cache/session integration tests.
- `TuiConfig::delegate_models` remains a read-only migration field, and hidden
  `--acp-websocket` remains a compatibility alias for `--ws`.

## Testing

Tests live with the code that owns the contract. Domain and state modules test
semantic transitions; feature input modules test key-to-intent behavior; feature
view modules use Ratatui test backends for layout and rendering; ACP modules test
translation, command, replay, JSON-RPC, and transport contracts; and application,
handler, runtime, and root UI tests cover composition and event/effect ordering.

Run the same primary gates used by CI:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets --all-features
cargo test --all-targets --all-features
```

Deterministic tests do not replace environmental validation. Live ACP/backend
behavior, an external compatible WebSocket service, subprocess stdio framing,
real terminal interaction, system clipboard integration, and complete
interactive UI smoke require suitable external services or a manual environment
and are not claimed by the automated suite.
