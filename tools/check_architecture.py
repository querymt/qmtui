#!/usr/bin/env python3
"""Enforce qmtui's lightweight, source-level architecture boundaries."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

APP_OWNER_FIELDS = (
    "navigation",
    "sessions",
    "delegates",
    "chat",
    "composer",
    "profiles",
    "models",
    "diagnostics",
    "mesh",
    "connection",
    "render",
    "auth",
)


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclass(frozen=True)
class Import:
    paths: tuple[tuple[str, ...], ...]
    position: int


@dataclass(frozen=True)
class Comment:
    text: str
    position: int


def _mask_non_code(source: str) -> tuple[str, list[Comment]]:
    """Mask comments and literals while preserving offsets and newlines."""
    chars = list(source)
    comments: list[Comment] = []
    length = len(source)
    index = 0

    def mask(start: int, end: int) -> None:
        for offset in range(start, end):
            if chars[offset] != "\n":
                chars[offset] = " "

    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end == -1:
                end = length
            comments.append(Comment(source[index:end], index))
            mask(index, end)
            index = end
            continue

        if source.startswith("/*", index):
            start = index
            depth = 1
            index += 2
            while index < length and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            comments.append(Comment(source[start:index], start))
            mask(start, index)
            continue

        raw = re.match(r'(?:br|r)(?P<hashes>#{0,255})"', source[index:])
        if raw:
            start = index
            delimiter = '"' + raw.group("hashes")
            content_start = index + raw.end()
            end_at = source.find(delimiter, content_start)
            index = length if end_at == -1 else end_at + len(delimiter)
            mask(start, index)
            continue

        quote_start = index + 1 if source.startswith('b"', index) else index
        if quote_start < length and source[quote_start] == '"':
            start = index
            index = quote_start + 1
            while index < length:
                if source[index] == "\\":
                    index += 2
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            mask(start, min(index, length))
            continue

        char_literal = re.match(r"'(?:\\.|[^'\\\n])'", source[index:])
        if char_literal:
            end = index + char_literal.end()
            mask(index, end)
            index = end
            continue

        index += 1

    return "".join(chars), comments


def _line_number(source: str, position: int) -> int:
    return source.count("\n", 0, position) + 1


def _expand_use_tree(use_tree: str) -> tuple[tuple[str, ...], ...]:
    tokens = re.findall(r"[A-Za-z_]\w*|::|[{},*]", use_tree)

    def parse(index: int, prefix: tuple[str, ...]) -> tuple[list[tuple[str, ...]], int]:
        paths: list[tuple[str, ...]] = []
        while index < len(tokens):
            while index < len(tokens) and tokens[index] in {",", "::"}:
                index += 1
            if index >= len(tokens) or tokens[index] == "}":
                return paths, index + (index < len(tokens))

            components: list[str] = []
            while index < len(tokens) and tokens[index] not in {"{", "}", ","}:
                token = tokens[index]
                if token == "as":
                    index += 2
                    break
                if token != "::":
                    components.append(token)
                index += 1

            clean = tuple(component for component in components if component != "self")
            current = prefix + clean
            if index < len(tokens) and tokens[index] == "{":
                nested, index = parse(index + 1, current)
                paths.extend(nested)
            elif current:
                paths.append(current)

            if index < len(tokens) and tokens[index] == "}":
                return paths, index + 1
            if index < len(tokens) and tokens[index] == ",":
                index += 1

        return paths, index

    paths, _ = parse(0, ())
    return tuple(paths)


def _imports(code: str) -> list[Import]:
    imports: list[Import] = []
    for match in re.finditer(r"\buse\s+([^;]+);", code, re.DOTALL):
        imports.append(Import(_expand_use_tree(match.group(1)), match.start()))
    return imports


def _starts_with(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def _first_import(
    imports: Iterable[Import], predicate: Callable[[tuple[str, ...]], bool]
) -> int | None:
    return next(
        (item.position for item in imports if any(predicate(path) for path in item.paths)),
        None,
    )


def _first_code_match(code: str, pattern: str) -> int | None:
    match = re.search(pattern, code, re.MULTILINE)
    return None if match is None else match.start()


def _first_dependency(
    code: str,
    imports: list[Import],
    import_predicate: Callable[[tuple[str, ...]], bool],
    reference_pattern: str,
) -> int | None:
    positions = [
        position
        for position in (
            _first_import(imports, import_predicate),
            _first_code_match(code, reference_pattern),
        )
        if position is not None
    ]
    return min(positions) if positions else None


def _is_feature_kind(relative: Path, kind: str) -> bool:
    parts = relative.parts
    return len(parts) >= 3 and parts[0] == "src" and parts[1] == "features" and (
        kind in parts[2:-1] or relative.name == f"{kind}.rs"
    )


def _matching_delimiter(code: str, opening: int, left: str, right: str) -> int | None:
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == left:
            depth += 1
        elif code[index] == right:
            depth -= 1
            if depth == 0:
                return index
    return None


def _matching_brace(code: str, opening: int) -> int | None:
    return _matching_delimiter(code, opening, "{", "}")


def _crate_dead_code_allowance(code: str) -> int | None:
    for attribute in re.finditer(r"#[ \t\r\n]*![ \t\r\n]*\[", code):
        attribute_opening = attribute.end() - 1
        attribute_closing = _matching_delimiter(code, attribute_opening, "[", "]")
        if attribute_closing is None:
            continue

        body_start = attribute_opening + 1
        body = code[body_start:attribute_closing]
        for allowance in re.finditer(r"\ballow\s*\(", body):
            allowance_opening = body_start + allowance.end() - 1
            allowance_closing = _matching_delimiter(code, allowance_opening, "(", ")")
            if allowance_closing is not None and re.search(
                r"\bdead_code\b", code[allowance_opening + 1 : allowance_closing]
            ):
                return attribute.start()
    return None


def _check_app(relative: Path, source: str, code: str) -> list[Violation]:
    violations: list[Violation] = []
    struct_match = re.search(r"\bstruct\s+App\s*\{", code)
    if struct_match is None:
        return [Violation(relative.as_posix(), 1, "src/app.rs must define struct App")]

    opening = code.find("{", struct_match.start())
    closing = _matching_brace(code, opening)
    if closing is None:
        return [Violation(relative.as_posix(), _line_number(source, opening), "App has no closing brace")]

    body = code[opening + 1 : closing]
    fields: dict[str, tuple[str, int]] = {}
    field_pattern = re.compile(
        r"(?m)^\s*(?P<visibility>pub(?:\s*\([^)]*\))?\s+)?"
        r"(?P<name>[A-Za-z_]\w*)\s*:"
    )
    for match in field_pattern.finditer(body):
        visibility = re.sub(r"\s+", "", (match.group("visibility") or "").strip())
        fields[match.group("name")] = (visibility, opening + 1 + match.start())

    for name in APP_OWNER_FIELDS + ("should_quit",):
        if name not in fields:
            violations.append(
                Violation(relative.as_posix(), _line_number(source, opening), f"App.{name} must exist and be pub(crate)")
            )
            continue
        visibility, position = fields[name]
        if visibility != "pub(crate)":
            violations.append(
                Violation(
                    relative.as_posix(),
                    _line_number(source, position),
                    f"App.{name} must be pub(crate), not {visibility or 'private'}",
                )
            )

    forbidden_methods = {
        "push_log",
        "set_status",
        "filtered_logs",
        "cycle_log_level_filter",
    }
    for impl_match in re.finditer(r"\bimpl\s+App\s*\{", code):
        impl_opening = code.find("{", impl_match.start())
        impl_closing = _matching_brace(code, impl_opening)
        if impl_closing is None:
            continue
        impl_body = code[impl_opening + 1 : impl_closing]
        for method in re.finditer(r"\bfn\s+([A-Za-z_]\w*)", impl_body):
            before = impl_body[: method.start()]
            depth = 1 + before.count("{") - before.count("}")
            name = method.group(1)
            if depth == 1 and name in forbidden_methods:
                position = impl_opening + 1 + method.start()
                violations.append(
                    Violation(
                        relative.as_posix(),
                        _line_number(source, position),
                        f"removed App forwarding method {name} must not be restored",
                    )
                )
    return violations


def _check_main(relative: Path, source: str, code: str, imports: list[Import]) -> list[Violation]:
    violations: list[Violation] = []
    module = re.search(r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+", code)
    if module:
        violations.append(
            Violation(relative.as_posix(), _line_number(source, module.start()), "main.rs must not declare modules")
        )

    feature_import = _first_import(
        imports,
        lambda path: "features" in path and path[0] in {"crate", "qmtui"},
    )
    if feature_import is not None:
        violations.append(
            Violation(
                relative.as_posix(),
                _line_number(source, feature_import),
                "main.rs must not import feature modules",
            )
        )

    if not re.search(r"\bqmtui\s*::\s*runtime\s*::\s*run\s*\(\s*\)", code):
        violations.append(
            Violation(relative.as_posix(), 1, "main.rs must call qmtui::runtime::run()")
        )
    return violations


def check_repository(root: Path) -> list[Violation]:
    root = root.resolve()
    src_root = root / "src"
    violations: list[Violation] = []

    legacy_file = src_root / "server_msg.rs"
    if legacy_file.exists():
        violations.append(Violation("src/server_msg.rs", 1, "legacy server_msg module must remain removed"))

    rust_files = sorted(src_root.rglob("*.rs")) if src_root.is_dir() else []
    for path in rust_files:
        relative = path.relative_to(root)
        source = path.read_text(encoding="utf-8")
        code, comments = _mask_non_code(source)
        imports = _imports(code)
        rel = relative.as_posix()

        def add(position: int | None, message: str) -> None:
            if position is not None:
                violations.append(Violation(rel, _line_number(source, position), message))

        dead_code = _crate_dead_code_allowance(code)
        if dead_code is not None and rel != "src/themes_gen.rs":
            add(dead_code, "crate-level allow(dead_code) is only permitted in src/themes_gen.rs")

        legacy = re.search(r"\b(?:server_msg|RawServerMsg|handle_server_msg)\b", code)
        add(legacy.start() if legacy else None, "legacy server message symbol must remain removed")

        wildcard = _first_import(imports, lambda item: item == ("crate", "handlers", "*"))
        add(wildcard, "cross-module wildcard import use crate::handlers::* is forbidden")

        for comment in comments:
            if re.search(r"(?i)phase\s*11", comment.text) and re.search(
                r"(?i)compatib|forward(?:ing)?(?:\s+layer)?|migration|temporary", comment.text
            ):
                add(comment.position, "stale Phase 11 compatibility/forward-layer comment must be removed")

        is_domain = relative.parts[:2] == ("src", "domain")
        is_acp = relative.parts[:2] == ("src", "acp")
        is_state = (
            relative.parent == Path("src")
            and relative.name.endswith("_state.rs")
            and relative.name != "render_state.rs"
        )
        is_input = _is_feature_kind(relative, "input")
        is_view = _is_feature_kind(relative, "view")
        is_ui = relative.parts[:2] == ("src", "ui")

        crate_ui = lambda item: _starts_with(item, ("crate", "ui"))
        ratatui = lambda item: item[:1] == ("ratatui",)
        crossterm = lambda item: item[:1] == ("crossterm",)
        tokio_sync = lambda item: _starts_with(item, ("tokio", "sync"))
        std_fs = lambda item: _starts_with(item, ("std", "fs"))
        std_process = lambda item: _starts_with(item, ("std", "process"))
        acp_transport = lambda item: _starts_with(item, ("crate", "acp", "transport")) or _starts_with(
            item, ("super", "transport")
        )
        protocol = lambda item: _starts_with(item, ("crate", "protocol"))

        if is_domain:
            checks = (
                (crate_ui, r"\bcrate\s*::\s*ui\b", "domain code must not depend on crate::ui"),
                (ratatui, r"\bratatui\s*::", "domain code must not depend on Ratatui"),
                (crossterm, r"\bcrossterm\s*::", "domain code must not depend on Crossterm"),
                (tokio_sync, r"\btokio\s*::\s*sync\b", "domain code must not depend on Tokio synchronization"),
                (std_fs, r"\bstd\s*::\s*fs\b", "domain code must not access the filesystem"),
                (std_process, r"\bstd\s*::\s*process\b", "domain code must not access processes"),
                (acp_transport, r"\b(?:crate\s*::\s*acp\s*::|super\s*::\s*)transport\b", "domain code must not depend on ACP transport"),
                (protocol, r"\bcrate\s*::\s*protocol\b", "domain code must not import protocol DTOs"),
            )
            for predicate, pattern, message in checks:
                add(_first_dependency(code, imports, predicate, pattern), message)

        if is_acp:
            add(
                _first_dependency(code, imports, crate_ui, r"\bcrate\s*::\s*ui\b"),
                "ACP code must not depend on crate::ui",
            )

        if is_state:
            checks = (
                (ratatui, r"\bratatui\s*::", "semantic state must not depend on Ratatui"),
                (crossterm, r"\bcrossterm\s*::", "semantic state must not depend on Crossterm"),
                (tokio_sync, r"\btokio\s*::\s*sync\b", "semantic state must not depend on Tokio synchronization"),
                (std_fs, r"\bstd\s*::\s*fs\b", "semantic state must not access the filesystem"),
                (std_process, r"\bstd\s*::\s*process\b", "semantic state must not access processes"),
                (acp_transport, r"\b(?:crate\s*::\s*acp\s*::|super\s*::\s*)transport\b", "semantic state must not depend on ACP transport"),
            )
            for predicate, pattern, message in checks:
                add(_first_dependency(code, imports, predicate, pattern), message)
            add(_first_code_match(code, r"\bCommand\s*::\s*new\s*\("), "semantic state must not spawn processes")

        if is_input:
            checks = (
                (ratatui, r"\bratatui\s*::", "feature input must not depend on Ratatui"),
                (tokio_sync, r"\btokio\s*::\s*sync\b", "feature input must not depend on Tokio synchronization"),
                (std_fs, r"\bstd\s*::\s*fs\b", "feature input must not access the filesystem"),
                (std_process, r"\bstd\s*::\s*process\b", "feature input must not access processes"),
                (acp_transport, r"\b(?:crate\s*::\s*acp\s*::|super\s*::\s*)transport\b", "feature input must not depend on ACP transport"),
            )
            for predicate, pattern, message in checks:
                add(_first_dependency(code, imports, predicate, pattern), message)
            add(_first_code_match(code, r"\bCommand\s*::\s*new\s*\("), "feature input must not spawn processes")

        if is_view or is_ui:
            add(
                _first_dependency(
                    code,
                    imports,
                    acp_transport,
                    r"\b(?:crate\s*::\s*acp\s*::|super\s*::\s*)transport\b",
                ),
                "renderers must not depend on ACP transport",
            )

        if rel == "src/ui/mod.rs":
            temporary_export = re.search(
                r"(?m)^\s*pub\s*\(\s*crate\s*\)\s+use\b", code
            )
            add(
                temporary_export.start() if temporary_export else None,
                "src/ui/mod.rs must not contain temporary pub(crate) use exports",
            )

        if rel == "src/app.rs":
            violations.extend(_check_app(relative, source, code))
        elif rel == "src/main.rs":
            violations.extend(_check_main(relative, source, code, imports))

    required = (("src/app.rs", "required App composition root is missing"), ("src/main.rs", "required thin entry point is missing"))
    for relative, message in required:
        if not (root / relative).is_file():
            violations.append(Violation(relative, 1, message))

    return sorted(set(violations))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = check_repository(root)
    if violations:
        print("Architecture checks failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print(f"Architecture checks passed ({len(list((root / 'src').rglob('*.rs')))} Rust files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
