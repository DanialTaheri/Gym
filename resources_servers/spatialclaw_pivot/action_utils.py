# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonicalize the executable action in a SpatialClaw assistant message."""

from __future__ import annotations

import ast
import re
from typing import Any


_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def visible_action_text(value: Any) -> str:
    text = str(value or "")
    text = _THINK.sub("", text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return ast.unparse(node)
    return ast.unparse(node)


def _canonical_expr(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(
            {
                "name": _call_name(node.func),
                "args": [_canonical_expr(value) for value in node.args],
                "keywords": {
                    str(value.arg): _canonical_expr(value.value)
                    for value in node.keywords
                    if value.arg is not None
                },
            }
        )
        self.generic_visit(node)

    # Calls in these bodies are definitions, not actions executed by the
    # assistant's top-level SpatialClaw program. Defaults and decorators could
    # execute, but SpatialClaw actions do not use them; ignoring the complete
    # definition is safer than rewarding a call hidden in an unused body.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None


def _parse_modules(text: str) -> list[ast.Module]:
    blocks = _CODE_FENCE.findall(text)
    candidates = blocks or [text]
    modules: list[ast.Module] = []
    for candidate in candidates:
        try:
            modules.append(ast.parse(candidate.strip()))
        except SyntaxError:
            continue
    return modules


def canonicalize_spatialclaw_action(value: Any) -> dict[str, Any] | None:
    """Return a formatting-independent action description, or ``None``.

    SpatialClaw's main policy emits Python rather than Responses function-call
    objects. Pivot rewards therefore compare the ordered call structure. A
    terminal ``ReturnAnswer`` is represented separately so the normal
    SpatialClaw answer scorers can be used.
    """

    text = visible_action_text(value)
    if not text:
        return None
    has_code_fence = bool(_CODE_FENCE.search(text))
    modules = _parse_modules(text)
    if not modules:
        return None
    collector = _CallCollector()
    for module in modules:
        collector.visit(module)

    return_calls = [
        call for call in collector.calls if call["name"].split(".")[-1] == "ReturnAnswer"
    ]
    if return_calls:
        # A valid SpatialClaw terminal program may prepare evidence, print a
        # diagnostic, and then submit one literal ReturnAnswer as its final
        # top-level statement. The corrected SFT trajectories use this form.
        # Reject an early/nested answer or any program that continues after it.
        if len(return_calls) != 1:
            return None
        statements = [statement for module in modules for statement in module.body]
        if not statements:
            return None
        final_statement = statements[-1]
        if not (
            isinstance(final_statement, ast.Expr)
            and isinstance(final_statement.value, ast.Call)
            and _call_name(final_statement.value.func).split(".")[-1]
            == "ReturnAnswer"
        ):
            return None
        final_call = final_statement.value
        if not final_call.args:
            return None
        try:
            answer = str(ast.literal_eval(final_call.args[0]))
        except (ValueError, TypeError):
            answer = ast.unparse(final_call.args[0])
        return {"type": "final_answer", "answer": answer, "scoring_mode": "auto"}

    # Some validated SFT turns intentionally execute only assignments/imports
    # or comments. Preserve those as an empty coarse operation sequence. A
    # code fence is required so arbitrary parseable prose/identifiers cannot
    # receive credit as a no-op Python action.
    if not collector.calls and not has_code_fence:
        return None
    return {"type": "python_calls", "calls": collector.calls}
