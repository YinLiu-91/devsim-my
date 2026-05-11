from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
from typing import Iterable


@dataclass(frozen=True)
class CaseSpec:
    name: str
    script: str
    script_args: tuple[str, ...]
    working_dir: str
    depends: tuple[str, ...] = ()


_ADD_TEST_RE = re.compile(r"^add_test\((?P<name>\S+)\s+(?P<body>.+)\)$")
_SET_DEPENDS_RE = re.compile(
    r'^set_tests_properties\((?P<name>"[^"]+"|\S+)\s+PROPERTIES\s+DEPENDS\s+(?P<deps>.+)\)$'
)
_IF_ZERO_RE = re.compile(r"IF\s*\(\s*0\s*\).*?ENDIF\s*\(\s*0\s*\)", flags=re.IGNORECASE | re.DOTALL)


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token


def _parse_add_test_line(line: str) -> CaseSpec | None:
    m = _ADD_TEST_RE.match(line.strip())
    if not m:
        return None
    name = m.group("name")
    body = m.group("body")
    tokens = shlex.split(body)
    if "--args" not in tokens:
        return None
    args_idx = tokens.index("--args")
    # test command is only relevant when it executes python script through --args
    if args_idx + 1 >= len(tokens):
        return None

    script = tokens[args_idx + 1]
    if not script.endswith(".py"):
        return None

    script_args: list[str] = []
    i = args_idx + 2
    while i < len(tokens) and not tokens[i].startswith("--"):
        script_args.append(tokens[i])
        i += 1

    working_dir = "/devsim/testing"
    if "--working" in tokens:
        widx = tokens.index("--working")
        if widx + 1 < len(tokens):
            working_dir = tokens[widx + 1]

    return CaseSpec(
        name=_strip_quotes(name),
        script=script,
        script_args=tuple(script_args),
        working_dir=working_dir,
    )


def _parse_depends_line(line: str) -> tuple[str, tuple[str, ...]] | None:
    m = _SET_DEPENDS_RE.match(line.strip())
    if not m:
        return None
    name = _strip_quotes(m.group("name"))
    deps_tokens = shlex.split(m.group("deps"))
    deps = tuple(_strip_quotes(d) for d in deps_tokens)
    return name, deps


def _expand_token(token: str, vars_map: dict[str, tuple[str, ...]]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        vals = vars_map.get(key, ())
        if not vals:
            return ""
        return vals[0]

    return re.sub(r"\$\{([^}]+)\}", repl, token)


def _expand_tokens(tokens: list[str], vars_map: dict[str, tuple[str, ...]]) -> list[str]:
    expanded: list[str] = []
    for t in tokens:
        m = re.fullmatch(r"\$\{([^}]+)\}", t)
        if m:
            expanded.extend(vars_map.get(m.group(1), ()))
        else:
            expanded.append(_expand_token(t, vars_map))
    return expanded


def _extract_commands(cmake_text: str) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    i = 0
    n = len(cmake_text)
    while i < n:
        if not (cmake_text[i].isalpha() or cmake_text[i] == "_"):
            i += 1
            continue
        j = i + 1
        while j < n and (cmake_text[j].isalnum() or cmake_text[j] == "_"):
            j += 1
        name = cmake_text[i:j]
        k = j
        while k < n and cmake_text[k].isspace():
            k += 1
        if k >= n or cmake_text[k] != "(":
            i = j
            continue
        depth = 1
        k += 1
        body_start = k
        while k < n and depth > 0:
            if cmake_text[k] == "(":
                depth += 1
            elif cmake_text[k] == ")":
                depth -= 1
            k += 1
        if depth == 0:
            commands.append((name.lower(), cmake_text[body_start : k - 1].strip()))
            i = k
        else:
            break
    return commands


def _collect_cmake_cases(
    commands: list[tuple[str, str]],
    vars_map: dict[str, tuple[str, ...]],
    start: int = 0,
    end: int | None = None,
) -> tuple[list[CaseSpec], dict[str, tuple[str, ...]]]:
    if end is None:
        end = len(commands)
    cases: list[CaseSpec] = []
    dep_map: dict[str, tuple[str, ...]] = {}
    i = start
    while i < end:
        cmd, body = commands[i]
        if cmd == "set":
            tokens = shlex.split(body)
            if tokens:
                vars_map[tokens[0]] = tuple(_expand_tokens(tokens[1:], vars_map))
            i += 1
            continue
        if cmd == "add_test":
            raw = f"add_test({body})"
            expanded = _expand_token(raw, vars_map)
            case = _parse_add_test_line(expanded)
            if case:
                cases.append(case)
            i += 1
            continue
        if cmd == "set_tests_properties":
            raw = f"set_tests_properties({body})"
            expanded = _expand_token(raw, vars_map)
            dep_entry = _parse_depends_line(expanded)
            if dep_entry:
                dep_map[dep_entry[0]] = dep_entry[1]
            i += 1
            continue
        if cmd == "foreach":
            tokens = _expand_tokens(shlex.split(body), vars_map)
            if not tokens:
                i += 1
                continue
            loop_var = tokens[0]
            loop_items = tokens[1:]
            depth = 1
            j = i + 1
            while j < end and depth > 0:
                if commands[j][0] == "foreach":
                    depth += 1
                elif commands[j][0] == "endforeach":
                    depth -= 1
                j += 1
            body_end = j - 1
            for item in loop_items:
                vars_map[loop_var] = (item,)
                loop_cases, loop_deps = _collect_cmake_cases(commands, vars_map, i + 1, body_end)
                cases.extend(loop_cases)
                dep_map.update(loop_deps)
            i = j
            continue
        i += 1
    return cases, dep_map


def _load_cases_from_ctest(ctest_file: Path) -> list[CaseSpec]:
    lines = ctest_file.read_text(encoding="utf-8").splitlines()
    cases: list[CaseSpec] = []
    dep_map: dict[str, tuple[str, ...]] = {}
    for line in lines:
        line = line.strip()
        if line.startswith("add_test("):
            case = _parse_add_test_line(line)
            if case:
                cases.append(case)
        elif line.startswith("set_tests_properties("):
            dep_entry = _parse_depends_line(line)
            if dep_entry:
                dep_map[dep_entry[0]] = dep_entry[1]
    case_names = {c.name for c in cases}
    return [
        CaseSpec(
            name=c.name,
            script=c.script,
            script_args=c.script_args,
            working_dir=c.working_dir,
            depends=tuple(d for d in dep_map.get(c.name, ()) if d in case_names),
        )
        for c in cases
    ]


def load_cases(cmake_file: Path) -> list[CaseSpec]:
    if cmake_file.name == "CTestTestfile.cmake":
        return _load_cases_from_ctest(cmake_file)

    text = cmake_file.read_text(encoding="utf-8")
    text = _IF_ZERO_RE.sub("", text)
    commands = _extract_commands(text)
    repo_root = str(cmake_file.resolve().parents[1])
    vars_map: dict[str, tuple[str, ...]] = {
        "PROJECT_SOURCE_DIR": (repo_root,),
        "PROJECT_BINARY_DIR": (repo_root,),
    }
    cases, dep_map = _collect_cmake_cases(commands, vars_map)

    case_names = {c.name for c in cases}
    enriched_cases: list[CaseSpec] = []
    for c in cases:
        deps = tuple(d for d in dep_map.get(c.name, ()) if d in case_names)
        enriched_cases.append(
            CaseSpec(
                name=c.name,
                script=c.script,
                script_args=c.script_args,
                working_dir=c.working_dir,
                depends=deps,
            )
        )
    return enriched_cases


def filter_cases(cases: Iterable[CaseSpec], pattern: str | None) -> list[CaseSpec]:
    if not pattern:
        return list(cases)
    cre = re.compile(pattern)
    return [c for c in cases if cre.search(c.name)]
