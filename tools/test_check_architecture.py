#!/usr/bin/env python3
"""Self-tests for the qmtui architecture checker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_architecture import APP_OWNER_FIELDS, check_repository


APP_SOURCE = """\
pub struct App {
%s
    pub(crate) should_quit: bool,
}

impl App {
    pub fn new() -> Self { todo!() }
}
""" % "".join(f"    pub(crate) {name}: (),\n" for name in APP_OWNER_FIELDS)

MAIN_SOURCE = """\
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    qmtui::runtime::run().await
}
"""


class ArchitectureCheckerTests(unittest.TestCase):
    def fixture(self, files: dict[str, str] | None = None) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        contents = {
            "src/app.rs": APP_SOURCE,
            "src/main.rs": MAIN_SOURCE,
        }
        contents.update(files or {})
        for relative, source in contents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return root

    def messages(self, files: dict[str, str] | None = None) -> list[str]:
        return [str(violation) for violation in check_repository(self.fixture(files))]

    def assert_fails(self, path: str, source: str, expected: str) -> None:
        messages = self.messages({path: source})
        self.assertTrue(
            any(expected in message for message in messages),
            f"expected {expected!r} in {messages!r}",
        )

    def assert_passes(self, files: dict[str, str] | None = None) -> None:
        self.assertEqual(self.messages(files), [])

    def test_domain_rejects_forbidden_dependencies(self) -> None:
        cases = {
            "crate::ui": ("use crate::ui::draw;\n", "crate::ui"),
            "Ratatui": ("use ratatui::Frame;\n", "Ratatui"),
            "Crossterm": ("use crossterm::event::KeyCode;\n", "Crossterm"),
            "Tokio": ("use tokio::{sync::mpsc, time};\n", "Tokio synchronization"),
            "filesystem": ("use std::{fs, time::Duration};\n", "filesystem"),
            "process": ("use std::process::{Command, Stdio};\n", "processes"),
            "ACP transport": (
                "use crate::acp::transport::jsonrpc::Peer;\n",
                "ACP transport",
            ),
            "protocol DTO": ("use crate::protocol::SessionDto;\n", "protocol DTOs"),
        }
        for name, (source, expected) in cases.items():
            with self.subTest(name=name):
                self.assert_fails("src/domain/example.rs", source, expected)

    def test_domain_allows_semantic_imports_and_ignores_comments_and_strings(self) -> None:
        self.assert_passes(
            {
                "src/domain/example.rs": """\
use crate::domain::tool::ToolDetail;
// use crate::ui::draw;
const EXAMPLE: &str = "ratatui::Frame";
struct Example(ToolDetail);
"""
            }
        )

    def test_acp_rejects_ui_but_allows_transport_tokio_and_sdk(self) -> None:
        self.assert_fails("src/acp/example.rs", "use crate::ui::draw;\n", "crate::ui")
        self.assert_passes(
            {
                "src/acp/example.rs": """\
use super::transport::jsonrpc::Peer;
use tokio::sync::mpsc;
use agent_client_protocol::JsonRpcMessage;
"""
            }
        )

    def test_semantic_state_rejects_ui_runtime_and_process_dependencies(self) -> None:
        cases = {
            "Ratatui": ("use ratatui::Frame;\n", "Ratatui"),
            "Crossterm": ("use crossterm::event::KeyEvent;\n", "Crossterm"),
            "Tokio channel": ("use tokio::sync::mpsc;\n", "Tokio synchronization"),
            "filesystem": ("use std::fs;\n", "filesystem"),
            "process import": ("use std::process::Command;\n", "processes"),
            "Command::new": ("fn run() { Command::new(\"x\"); }\n", "spawn processes"),
            "ACP transport": ("use super::transport::Peer;\n", "ACP transport"),
        }
        for name, (source, expected) in cases.items():
            with self.subTest(name=name):
                self.assert_fails("src/chat_state.rs", source, expected)

    def test_render_state_is_the_ratatui_state_exception(self) -> None:
        self.assert_passes({"src/render_state.rs": "use ratatui::text::Line;\n"})

    def test_feature_input_allows_crossterm_but_rejects_runtime_dependencies(self) -> None:
        self.assert_passes(
            {"src/features/chat/input/composer.rs": "use crossterm::event::KeyEvent;\n"}
        )
        cases = {
            "Ratatui": ("use ratatui::Frame;\n", "Ratatui"),
            "Tokio": ("use tokio::sync::oneshot;\n", "Tokio synchronization"),
            "filesystem": ("use std::fs::File;\n", "filesystem"),
            "process": ("use std::process::Command;\n", "processes"),
            "transport": ("use crate::acp::transport::Peer;\n", "ACP transport"),
        }
        for name, (source, expected) in cases.items():
            with self.subTest(name=name):
                self.assert_fails("src/features/chat/input/composer.rs", source, expected)

    def test_renderer_allows_ratatui_and_test_app_adapter_but_rejects_transport(self) -> None:
        self.assert_passes(
            {
                "src/features/chat/view/mod.rs": """\
use ratatui::Frame;
#[cfg(test)]
fn draw_for_test(_app: &mut crate::app::App) {}
"""
            }
        )
        for path in ("src/features/chat/view/screen.rs", "src/ui/mod.rs"):
            with self.subTest(path=path):
                self.assert_fails(
                    path,
                    "use crate::acp::transport::jsonrpc::Peer;\n",
                    "ACP transport",
                )

    def test_wildcard_policy_rejects_handlers_but_allows_test_super(self) -> None:
        self.assert_fails(
            "src/example.rs",
            "use crate::handlers::*;\n",
            "cross-module wildcard",
        )
        self.assert_passes(
            {"src/example.rs": "#[cfg(test)]\nmod tests { use super::*; }\n"}
        )

    def test_ui_mod_rejects_temporary_export_only_at_root_ui(self) -> None:
        declaration = "pub(crate) use owner::draw;\n"
        self.assert_fails("src/ui/mod.rs", declaration, "temporary pub(crate) use")
        self.assert_passes({"src/features/chat/view/mod.rs": declaration})

    def test_dead_code_policy_allows_generated_and_item_level_allowances(self) -> None:
        self.assert_passes(
            {
                "src/themes_gen.rs": "#![allow(unused_imports, dead_code)]\n",
                "src/domain/example.rs": """\
#[allow(dead_code)]
struct RetainedDto;
#[cfg_attr(test, allow(dead_code))]
struct ConditionalRetainedDto;
const EXAMPLE: &str = "#![allow(dead_code)]";
// #![cfg_attr(test, allow(dead_code))]
""",
            }
        )

        self.assert_fails(
            "src/lib.rs",
            "#![allow(dead_code)]\n",
            "crate-level allow(dead_code)",
        )

    def test_dead_code_policy_rejects_combined_crate_allowance(self) -> None:
        self.assert_fails(
            "src/lib.rs",
            """\
#![allow(
    unused_imports,
    dead_code,
)]
""",
            "crate-level allow(dead_code)",
        )

    def test_dead_code_policy_rejects_conditional_crate_allowance(self) -> None:
        self.assert_fails(
            "src/lib.rs",
            """\
#![cfg_attr(
    any(test, feature = "example"),
    allow(unused_imports, dead_code),
)]
""",
            "crate-level allow(dead_code)",
        )

    def test_removed_app_forward_is_scoped_to_impl_app(self) -> None:
        app_with_forward = APP_SOURCE + "\nimpl App { fn push_log(&mut self) {} }\n"
        self.assert_fails("src/app.rs", app_with_forward, "App forwarding method push_log")
        self.assert_passes(
            {
                "src/diagnostics.rs": """\
struct DiagnosticsState;
impl DiagnosticsState { fn push_log(&mut self) {} }
"""
            }
        )

    def test_every_removed_app_forward_is_rejected(self) -> None:
        for method in (
            "push_log",
            "set_status",
            "filtered_logs",
            "cycle_log_level_filter",
        ):
            with self.subTest(method=method):
                source = APP_SOURCE + f"\nimpl App {{ fn {method}(&self) {{}} }}\n"
                self.assert_fails("src/app.rs", source, method)

    def test_legacy_file_and_identifiers_are_rejected(self) -> None:
        self.assert_fails("src/server_msg.rs", "pub struct Other;\n", "server_msg module")
        for symbol in ("server_msg", "RawServerMsg", "handle_server_msg"):
            with self.subTest(symbol=symbol):
                self.assert_fails(
                    "src/example.rs",
                    f"fn example() {{ let {symbol} = 1; }}\n",
                    "legacy server message symbol",
                )

    def test_stale_phase_11_comments_are_rejected(self) -> None:
        self.assert_fails(
            "src/example.rs",
            "// Phase 11 temporary compatibility forwarding layer.\n",
            "stale Phase 11",
        )

    def test_app_owner_fields_and_should_quit_must_be_pub_crate(self) -> None:
        self.assert_passes()
        public_app = APP_SOURCE.replace(
            "pub(crate) navigation: ()", "pub navigation: ()"
        ).replace("pub(crate) should_quit: bool", "pub should_quit: bool")
        messages = self.messages({"src/app.rs": public_app})
        self.assertTrue(any("App.navigation must be pub(crate)" in item for item in messages))
        self.assertTrue(any("App.should_quit must be pub(crate)" in item for item in messages))

    def test_thin_main_passes_and_rejects_modules_features_and_missing_run(self) -> None:
        self.assert_passes()
        invalid = """\
mod features;
use qmtui::features::chat;
fn main() {}
"""
        messages = self.messages({"src/main.rs": invalid})
        self.assertTrue(any("must not declare modules" in item for item in messages))
        self.assertTrue(any("must not import feature modules" in item for item in messages))
        self.assertTrue(any("must call qmtui::runtime::run()" in item for item in messages))

    def test_violations_report_source_line(self) -> None:
        messages = self.messages(
            {"src/domain/example.rs": "use crate::domain::tool::ToolDetail;\nuse ratatui::Frame;\n"}
        )
        self.assertTrue(any(item.startswith("src/domain/example.rs:2:") for item in messages))


if __name__ == "__main__":
    unittest.main()
