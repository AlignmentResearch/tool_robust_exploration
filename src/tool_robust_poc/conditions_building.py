"""Condition-specific message assembly for judge robustness experiments.

The assembler turns a ConditionBlueprint + payload dict into API-ready
(messages, extra_api_kwargs). No condition-name switching — the blueprint
IS the full specification.
"""

from __future__ import annotations

import json

from tool_robust_poc.core_types import ConditionBlueprint, ToolBinding


def assemble_messages(
    blueprint: ConditionBlueprint,
    payload: dict[str, str],
) -> tuple[list[dict], dict]:
    """Turn a blueprint + payload into API-ready messages.

    Template placeholders (``{name}``) in ``system_text`` and ``user_text``
    are filled from *payload*, excluding keys claimed by tool bindings.
    Tool-bound values go into tool result messages instead.

    Returns:
        (messages, extra_api_kwargs) ready for chat completion API.
    """
    # 1. Determine which payload keys are bound to tools
    tool_bound_keys = {tb.binds_to for tb in blueprint.tool_bindings}

    # 2. Build inline substitutions (payload minus tool-bound values)
    inline_payload = {k: v for k, v in payload.items() if k not in tool_bound_keys}

    # 3. Render templates
    user_content = blueprint.user_text.format(**inline_payload)

    system_content = None
    if blueprint.system_text is not None:
        system_content = blueprint.system_text.format(**inline_payload)

    # 4. Assemble messages
    messages: list[dict] = []
    extra_kwargs: dict = {}

    if not blueprint.tool_bindings:
        # Inline condition (baseline or multi_msg style)
        if system_content is None:
            messages = [{"role": "user", "content": user_content}]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
    else:
        # Tool condition — always includes tool schemas
        if system_content is not None:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        # Simulated assistant tool calls + tool results
        tool_calls = []
        tool_messages = []
        for i, binding in enumerate(blueprint.tool_bindings):
            call_id = f"{binding.tool_name}-{i + 1}"
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": binding.tool_name,
                    "arguments": json.dumps(dict(binding.tool_arguments)),
                },
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": payload[binding.binds_to],
            })

        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        messages.extend(tool_messages)

        extra_kwargs["tools"] = _auto_tool_schemas(blueprint.tool_bindings)

    if blueprint.tool_choice is not None:
        extra_kwargs["tool_choice"] = blueprint.tool_choice

    return messages, extra_kwargs


def _auto_tool_schemas(bindings: list[ToolBinding]) -> list[dict]:
    """Generate tool schemas from tool bindings."""
    schemas = []
    for binding in bindings:
        # Build parameter properties from tool_arguments keys
        properties: dict = {}
        for key, value in binding.tool_arguments.items():
            prop: dict = {"description": f"The {key} parameter."}
            if isinstance(value, int):
                prop["type"] = "integer"
            elif isinstance(value, str):
                prop["type"] = "string"
            else:
                prop["type"] = "string"
            properties[key] = prop

        schema: dict = {
            "type": "function",
            "function": {
                "name": binding.tool_name,
                "description": binding.description or f"Retrieves {binding.binds_to}.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(binding.tool_arguments.keys()),
                    "additionalProperties": False,
                },
            },
        }
        schemas.append(schema)
    return schemas
