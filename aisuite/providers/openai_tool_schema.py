"""Project function parameters onto the OpenAI Responses schema subset."""

import json
from dataclasses import dataclass
from itertools import combinations


class _Unsupported(ValueError):
    """The original argument contract cannot be represented losslessly."""


_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_LISTS = ("oneOf", "anyOf", "allOf", "prefixItems")
_SINGLES = (
    "items",
    "additionalProperties",
    "contains",
    "if",
    "then",
    "else",
    "not",
    "propertyNames",
    "additionalItems",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_ANNOTATIONS = {"description", "title", "default", "examples", "$comment"}
_REFERENCES = {
    "$schema",
    "$ref",
    "$id",
    "$defs",
    "definitions",
    "$anchor",
    "$dynamicRef",
    "$dynamicAnchor",
    "$recursiveRef",
    "$recursiveAnchor",
    "id",
}


def _types(schema):
    kinds = schema.get("type", [])
    kinds = {kinds} if isinstance(kinds, str) else set(kinds)
    if "number" in kinds:
        kinds.add("integer")
    return kinds


def _tags(schema):
    if not isinstance(schema, dict):
        return set()
    values = [schema["const"]] if "const" in schema else schema.get("enum", [])
    return set(values) if all(isinstance(value, str) for value in values) else set()


def _disjoint(a, b, parent_type=None):
    left, right = _types(a), _types(b)
    if left and right and left.isdisjoint(right):
        return True
    if left == right == {"array"}:
        return a.get("maxItems", float("inf")) < b.get("minItems", 0) or b.get(
            "maxItems", float("inf")
        ) < a.get("minItems", 0)
    if parent_type != "object" and not left == right == {"object"}:
        return False
    ap, bp = a.get("properties", {}), b.get("properties", {})
    ar, br = set(a.get("required", [])), set(b.get("required", []))
    if (
        a.get("additionalProperties") is False
        and not a.get("patternProperties")
        and br - ap.keys()
        or b.get("additionalProperties") is False
        and not b.get("patternProperties")
        and ar - bp.keys()
    ):
        return True
    for key in ar & br & ap.keys() & bp.keys():
        x, y = _tags(ap[key]), _tags(bp[key])
        if x and y and x.isdisjoint(y):
            return True
    return False


def _intersect(base, guard):
    """Merge a closed action/object branch with its shared field definitions."""
    if not isinstance(base, dict) or not isinstance(guard, dict):
        raise _Unsupported("boolean schema in object intersection")
    result = dict(base)
    if "properties" in guard:
        left, right = base.get("properties", {}), guard["properties"]
        if (
            guard.get("additionalProperties") is not False
            or base.get("additionalProperties") is not False
            or "patternProperties" in base
            or "patternProperties" in guard
        ):
            raise _Unsupported("open object intersection")
        result["properties"] = {
            key: _intersect(left[key], value)
            for key, value in right.items()
            if key in left
        }
    for key, value in guard.items():
        if key == "properties":
            continue
        if key == "required":
            result[key] = list(dict.fromkeys([*base.get(key, []), *value]))
        elif key == "enum" and key in base:
            if not all(isinstance(tag, str) for tag in [*base[key], *value]):
                raise _Unsupported("non-string enum intersection")
            result[key] = [entry for entry in value if entry in base[key]]
            if not result[key]:
                raise _Unsupported("empty enum intersection")
        elif key in {"minimum", "minLength", "minItems"} and key in base:
            result[key] = max(base[key], value)
        elif key in {"maximum", "maxLength", "maxItems"} and key in base:
            result[key] = min(base[key], value)
        elif (
            key not in base
            or key in _ANNOTATIONS
            or json.dumps(base[key], sort_keys=True)
            == json.dumps(value, sort_keys=True)
        ):
            result[key] = value
        else:
            raise _Unsupported(f"incompatible {key} constraints")
    return result


def _object_union(node, choices):
    """Expand single-key XORs and tagged branches without broadening either."""
    base = {key: value for key, value in node.items() if key != "oneOf"}
    properties = base["properties"]
    if "patternProperties" in base:
        return None
    if all(
        set(branch) == {"required"} and len(branch["required"]) == 1
        for branch in choices
    ):
        keys = [branch["required"][0] for branch in choices]
        if len(set(keys)) != len(keys) or not set(keys) <= properties.keys():
            raise _Unsupported("ambiguous exclusive fields")
        branches = []
        for key in keys:
            excluded = set(keys) - {key}
            if excluded.intersection(base.get("required", [])):
                continue
            fields = {key: properties[key]}
            fields.update((k, v) for k, v in properties.items() if k not in excluded)
            branches.append(
                {
                    **base,
                    "properties": fields,
                    "required": list(dict.fromkeys([*base.get("required", []), key])),
                }
            )
        return branches
    if all(
        b.get("additionalProperties") is False and "properties" in b for b in choices
    ):
        branches = [_intersect(base, branch) for branch in choices]
        if all(_disjoint(a, b) for a, b in combinations(branches, 2)):
            # Put a discriminator before shared fields when generating the object.
            for key in base.get("required", []):
                tags = [_tags(b["properties"].get(key)) for b in branches]
                if all(tags) and all(a.isdisjoint(b) for a, b in combinations(tags, 2)):
                    for b in branches:
                        b["properties"] = {key: b["properties"][key], **b["properties"]}
                    break
            return branches
    return None


def _lower(node, status):
    if not isinstance(node, dict):
        return node
    if _REFERENCES.intersection(node):
        status["references"] = True
    result = dict(node)
    if "oneOf" in node:
        status["found"] = True
        options = node["oneOf"]
        choices = (
            options
            if isinstance(options, list) and all(isinstance(b, dict) for b in options)
            else []
        )
        branches = None
        if (
            choices
            and node.get("type") == "object"
            and node.get("additionalProperties") is False
            and "properties" in node
        ):
            try:
                branches = _object_union(node, choices)
            except _Unsupported:
                pass
        if branches:
            result = {
                **{k: v for k, v in node.items() if k in _ANNOTATIONS},
                "anyOf": branches,
            }
        elif (
            choices
            and "anyOf" not in node
            and all(
                _disjoint(a, b, node.get("type")) for a, b in combinations(choices, 2)
            )
        ):
            result["anyOf"] = result.pop("oneOf")
        else:
            status["exact"] = False
    for key in _MAPS:
        if key in result:
            result[key] = {
                name: _lower(sub, status) for name, sub in result[key].items()
            }
    for key in _LISTS:
        if key in result:
            result[key] = [_lower(sub, status) for sub in result[key]]
    for key in _SINGLES:
        if key in result:
            sub = result[key]
            result[key] = (
                [_lower(s, status) for s in sub]
                if isinstance(sub, list)
                else _lower(sub, status)
            )
    return result


_STRICT_KEYS = {
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "anyOf",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
}


def _accepts_null(schema):
    if "anyOf" in schema:
        return any(_accepts_null(option) for option in schema["anyOf"])
    return "null" in _types(schema) and ("enum" not in schema or None in schema["enum"])


def _strict(schema, path, omit, nullable):
    if not isinstance(schema, dict) or schema.keys() - _STRICT_KEYS - _ANNOTATIONS:
        raise _Unsupported("unsupported schema keyword or boolean schema")
    result = {
        k: v for k, v in schema.items() if k not in {"default", "examples", "$comment"}
    }
    if "const" in result:
        constant = result.pop("const")
        if "enum" in result:
            raise _Unsupported("combined const and enum")
        result["enum"] = [constant]
    if any(isinstance(v, (dict, list)) for v in result.get("enum", [])):
        raise _Unsupported("structured enum value")
    if "anyOf" in result:
        if result.keys() - {"anyOf", "description", "title"}:
            raise _Unsupported("union with sibling constraints")
        options = []
        for sub in result["anyOf"]:
            option = _strict(sub, path, omit, nullable)
            if option.keys() == {"anyOf"}:
                options.extend(option["anyOf"])
            else:
                options.append(option)
        result["anyOf"] = options
        return result
    if not result.get("type") and _tags(result):
        result["type"] = "string"
    declared = result.get("type")
    if not declared:
        raise _Unsupported("untyped value")
    if isinstance(declared, list) and len(set(declared) - {"null"}) > 1:
        raise _Unsupported("multiple non-null types")
    if _accepts_null(result):
        nullable.add(path)
    kinds = _types(result)
    if "object" in kinds:
        if result.get("additionalProperties") is not False:
            raise _Unsupported("object accepts arbitrary keys")
        fields = result.get("properties", {})
        required = set(result.get("required", []))
        if required - fields.keys():
            raise _Unsupported("required keys outside a closed object")
        properties = {}
        for key, sub in fields.items():
            at = (*path, key)
            field = _strict(sub, at, omit, nullable)
            if key not in required:
                omit.add(at)
                if "anyOf" in field:
                    field = {**field, "anyOf": [*field["anyOf"], {"type": "null"}]}
                else:
                    field = {"anyOf": [field, {"type": "null"}]}
            properties[key] = field
        result.update(properties=properties, required=list(properties))
    if "array" in kinds:
        result["items"] = _strict(result.get("items"), (*path, None), omit, nullable)
    return result


@dataclass(frozen=True)
class ArgumentPlan:
    wrapped: bool
    omit_null: frozenset

    def decode(self, arguments):
        try:
            value = json.loads(arguments)
        except (ValueError, TypeError):
            return arguments
        if not isinstance(value, dict):
            return arguments
        if self.wrapped:
            if value.keys() != {"request"} or not isinstance(value["request"], dict):
                return arguments
            value = value["request"]

        def restore(item, path):
            if isinstance(item, dict):
                return {
                    k: restore(v, (*path, k))
                    for k, v in item.items()
                    if v is not None or (*path, k) not in self.omit_null
                }
            if isinstance(item, list):
                return [restore(v, (*path, None)) for v in item]
            return item

        return json.dumps(restore(value, ()), ensure_ascii=False, separators=(",", ":"))


def project_function(tool):
    """Return the wire tool and its request-local argument decoder, if needed."""
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        return tool, None
    status = {"found": False, "exact": True, "references": False}
    lowered = _lower(parameters, status)
    if not status["found"]:
        return tool, None
    # Moving a referenced subschema into a branch can change its JSON Pointer or base URI.
    if status["references"]:
        lowered = parameters
        status["exact"] = False
    fallback = lowered
    if parameters.get("type") == "object" and "type" not in fallback:
        fallback = {"type": "object", **fallback}
    if tool.get("strict") is False:
        return {**tool, "parameters": fallback}, None
    try:
        if not status["exact"]:
            raise _Unsupported("overlapping or referenced oneOf alternatives")
        omit, nullable = set(), set()
        wire = _strict(lowered, (), omit, nullable)
        if omit & nullable:
            raise _Unsupported("omission and explicit null share a field")
        wrapped = "anyOf" in wire
        if wrapped:
            wire = {
                "type": "object",
                "properties": {"request": wire},
                "required": ["request"],
                "additionalProperties": False,
            }
        if wire.get("type") != "object":
            raise _Unsupported("function parameters must be an object")
    except _Unsupported as exc:
        if tool.get("strict") is True:
            raise ValueError(
                f"Cannot represent strict function {tool.get('name')!r}: {exc}"
            ) from exc
        return {**tool, "parameters": fallback, "strict": False}, None
    plan = ArgumentPlan(wrapped, frozenset(omit)) if wrapped or omit else None
    return {**tool, "parameters": wire, "strict": True}, plan
