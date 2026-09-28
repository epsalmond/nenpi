"""Colorful, text-only renderers for quota diagnosis reports.

The JSON report remains the source of truth. These functions only format
already-computed data and deliberately avoid reading transcript contents.
"""

from __future__ import annotations

import re
import textwrap
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


def _view_args(args: Any) -> SimpleNamespace:
    """Supply the few common CLI options without mutating the caller's args."""
    return SimpleNamespace(
        no_color=getattr(args, "no_color", False),
        color=getattr(args, "color", "auto"),
        ascii=getattr(args, "ascii", False),
        width=getattr(args, "width", None),
    )


def _wrap(text: Any, width: int, indent: str = "") -> list[str]:
    value = str(text or "")
    if not value:
        return []
    return textwrap.wrap(
        value,
        width=max(20, width - len(indent)),
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [indent]


def _tokens(value: Any) -> str:
    from .drain import format_tokens

    try:
        return format_tokens(int(round(float(value or 0))))
    except (TypeError, ValueError, OverflowError):
        return "unknown"


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _percent(value: Any) -> str:
    try:
        return "%.1f%%" % (100.0 * float(value))
    except (TypeError, ValueError, OverflowError):
        return "unknown"


def _session_label(value: Any) -> str:
    if not value:
        return ""
    from .drain import short_id

    return short_id(str(value))


def _wrap_styled(lines: list[str], text: Any, width: int, paint, style: str,
                 indent: str = "") -> None:
    lines.extend(paint(line, style) for line in _wrap(text, width, indent))


def _wrap_parts(lines: list[str], parts: Sequence[tuple[str, str]], width: int,
                paint, indent: str = "") -> None:
    """Wrap plain text first, preserving explicit semantic spans across lines.

    ANSI escapes never enter width calculations. Role boundaries survive wraps;
    the same content is used for colored and plain output.
    """
    parts = [(re.sub(r"\s", " ", value), role) for value, role in parts]
    text = "".join(value for value, _ in parts)
    spans = []
    end = 0
    for value, role in parts:
        spans.append((end, end + len(value), role))
        end += len(value)
    cursor = 0
    for line in _wrap(text, width, indent):
        body = line[len(indent):]
        start = text.find(body, cursor)
        end = start + len(body)
        rendered = indent
        for lo, hi, role in spans:
            lo, hi = max(lo, start), min(hi, end)
            if lo < hi:
                rendered += paint(text[lo:hi], role)
        lines.append(rendered)
        cursor = end


def _field(lines: list[str], label: str, value: str, width: int, paint,
           role: str = "data", indent: str = "") -> None:
    _wrap_parts(lines, [(label, "label"), (value, role)], width, paint, indent)


def _finish(lines: list[str], view: Any) -> str:
    result = "\n".join(lines)
    if getattr(view, "ascii", False):
        for glyph, replacement in (
            ("·", "/"), ("×", "x"), ("≤", "<="), ("≥", ">="),
            ("—", "--"), ("–", "-"), ("→", "->"), ("↳", "->"),
            ("…", "..."), ("−", "-"), ("¹", "[1]"), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
        ):
            result = result.replace(glyph, replacement)
    return result


def _scope_label(scope: Mapping[str, Any]) -> str:
    parts = []
    harness = scope.get("harness")
    if harness:
        parts.append(str(harness).upper())
    since, until = scope.get("since"), scope.get("until")
    if since and until:
        parts.append("%s to %s" % (since, until))
    elif since:
        parts.append("since %s" % since)
    elif until:
        parts.append("through %s" % until)
    else:
        parts.append("all available time")
    if scope.get("session_id"):
        parts.append("session " + _session_label(scope["session_id"]))
    if scope.get("prompt") is not None:
        parts.append("prompt %s" % scope["prompt"])
    return " · ".join(parts)


def _table(lines, headers, rows, width, paint, indent="   "):
    """Codex layout: padded cells, bold headers and a dim heavy separator."""
    if not rows:
        return
    rows = [[(paint.glyphs(str(value)), role) for value, role in row] for row in rows]
    def column_sizes():
        return [max(len(header), *(len(line) for row in rows
                                   for line in row[i][0].splitlines()))
                for i, header in enumerate(headers)]
    sizes = column_sizes()
    gap = 2
    def required_width():
        return sum(sizes) + (gap + 2) * (len(sizes) - 1) + 2 + len(indent)
    if required_width() > width and "Context" in headers:
        column = headers.index("Context")
        for row in rows:
            value, role = row[column]
            row[column] = (value.replace(" (", "\n("), role)
        sizes = column_sizes()
        if required_width() > width:
            gap = 1
    if required_width() > width:
        for row in rows:
            _wrap_styled(lines, row[0][0], width, paint, "label", indent)
            for header, (value, role) in zip(headers[1:], row[1:]):
                _field(lines, header + ": ", str(value).replace("\n", " "), width, paint, role, indent + "  ")
        return
    numeric = {"Count", "Input", "Cached", "Uncached", "Usage*", "Calls", "Repetition*", "Repeated", "Projects", "Reduction*",
               "Cache write*", "Break-even turn*", "Post-Shake Turns", "Saved / cost*"}
    def row_text(cells):
        pieces = []
        for index, (header, (value, role), size) in enumerate(zip(headers, cells, sizes)):
            value = str(value)
            aligned = value.rjust(size) if index > 0 and header in numeric else value.ljust(size)
            pieces.append(" " + paint(aligned, role) + " ")
        return indent + (" " * gap).join(pieces).rstrip()
    lines.append(row_text([(h, "table_header") for h in headers]))
    bar = "-" if paint.ascii_only else "━"
    lines.append(indent + paint((" " * gap).join(bar * (n + 2) for n in sizes), "detail"))
    for row in rows:
        for line_index in range(max(len(value.splitlines()) for value, _ in row)):
            lines.append(row_text([(value.splitlines()[line_index] if line_index < len(value.splitlines()) else "", role)
                                   for value, role in row]))


def _change(before, after):
    return "%s → %s" % (_tokens(before) if before is not None else "?",
                         _tokens(after) if after is not None else "?")


def _render_shakes(lines, events, width, paint):
    if not events:
        return
    all_subagents = all(e.get("role") == "subagent" for e in events)
    title = "Shake · subagents of this prompt" if all_subagents else "Shake · agents in this prompt"
    lines.extend(("", paint("   " + title, "heading")))
    rows = []
    reasons = set()
    for event in sorted(events, key=lambda e: (str(e.get("thread_id")), e.get("timestamp", 0))):
        payback = event.get("payback") or {}
        estimated = payback.get("status") == "estimated"
        calls = event.get("subsequent_response_count", 0)
        reduction = event.get("reduction_tokens")
        if estimated:
            break_even = str(payback["break_even_calls"])
            if payback.get("cache_baseline") == "presumed_expired":
                break_even += " (cold)"
            rewrite = _tokens(payback["extra_uncached_tokens"])
            net = payback["net_read_equivalent_tokens_saved"]
            net_text = ("-" if net < 0 else "") + _tokens(abs(net))
            net_role = "improvement" if net > 0 else "cost" if net < 0 else "data"
        else:
            break_even, rewrite, net_text, net_role = "—", "?", "—", "detail"
            if payback.get("status") == "cold_cache_baseline_unknown":
                break_even = "cold resume"
                before_u, after_u = event.get("before_uncached_input_tokens"), event.get("after_uncached_input_tokens")
                if before_u is not None and after_u is not None:
                    rewrite = _tokens(max(0, after_u - before_u))
                reasons.add("cold resume: pre-idle cache is not a valid cost baseline")
            else:
                reasons.add(str(payback.get("status") or "missing_usage").replace("_", " "))
        name = event.get("agent_name")
        if not name:
            name = "main" if event.get("role") == "root" else "subagent" if event.get("role") == "subagent" else "unknown"
        agent = "%s %s" % (_session_label(event.get("thread_id")), name)
        rows.append([
            (agent, "label"),
            (_change(event.get("before_context_tokens"), event.get("after_context_tokens"))
             + ((" (%s%s)" % ("−" if reduction > 0 else "+" if reduction < 0 else "",
                                _tokens(abs(reduction)))) if reduction is not None else ""), "data"),
            (rewrite, "data"),
            (break_even, "total" if estimated and payback.get("break_even_reached") else "data"),
            (str(calls), "cost" if estimated and not payback.get("break_even_reached") else "data"),
            (net_text, net_role),
        ])
    _table(lines, ["Subagent" if all_subagents else "Agent", "Context", "Cache write*",
                   "Break-even turn*", "Post-Shake Turns", "Saved / cost*"], rows, width, paint)
    _wrap_styled(lines, "* Cache write: extra uncached input. Net savings/cost: cached-read-equivalent tokens; "
                 "constant reduction. Warm: one rewrite penalty. Cold: first avoided read is uncached, no rewrite penalty. "
                 "Turns end at the next Shake or prompt end.",
                 width, paint, "detail", "   ")
    if reasons:
        _wrap_styled(lines, "Unavailable: " + "; ".join(sorted(reasons)) + ".", width, paint, "detail", "   ")


def render_auto(report: Mapping[str, Any], args: Any) -> str:
    """A compact workload table and Shake payback chart; full evidence stays in JSON."""
    from .drain import make_painter, terminal_width

    view = _view_args(args)
    paint = make_painter(view)
    width = terminal_width(view)
    lines = [paint("Nenpi auto", "heading")]
    scope = report.get("scope") or {}
    _wrap_styled(lines, _scope_label({k: scope[k] for k in ("harness", "since", "until") if k in scope}),
                 width, paint, "detail")
    findings = report.get("findings") or []
    summary = report.get("summary") or {}
    if len(findings) != 1 and summary.get("responses") is not None:
        _wrap_styled(lines, "%s calls · %s input · %s output" % (
            summary["responses"], _tokens(summary.get("input_tokens")),
            _tokens(summary.get("output_tokens"))), width, paint, "total")
    if not findings:
        lines.append(paint("No activity in this scope.", "summary"))
        _append_next(lines, report.get("next") or [], width, paint)
        return _finish(lines, view)

    all_shakes = (report.get("shake") or {}).get("events") or []
    seen_commands = set()
    for rank, finding in enumerate(findings, 1):
        values = finding.get("values") or {}
        session = finding.get("short_id") or _session_label(finding.get("session_id"))
        location = "%s · %s · prompt %s" % (finding.get("cwd") or "—", session, finding.get("prompt", "?"))
        lines.extend(("", paint("%d. %s" % (rank, location), "heading")))
        if values:
            responses = _integer(values.get("responses"))
            small = _integer(values.get("small_result_responses"))
            rows = [[("All calls", "label"), (str(responses), "data"),
                     (_tokens(values.get("input_tokens")), "total"),
                     (_tokens(values.get("cached_input_tokens")), "data"),
                     (_tokens(values.get("uncached_input_tokens")), "data"),
                     ("100%" if values.get("weighted_units") else "—", "total")]]
            if small:
                rows.append([("Tiny output → large read", "label"), (str(small), "data"),
                             (_tokens(values.get("small_result_input_tokens")), "total"),
                             (_tokens(values.get("small_result_cached_input_tokens")), "data"),
                             (_tokens(values.get("small_result_uncached_input_tokens")), "data"),
                             (_percent(values["small_result_weighted_share"])
                              if values.get("small_result_weighted_share") is not None else "—", "total")])
            _table(lines, ["Calls", "Count", "Input", "Cached", "Uncached", "Usage*"], rows, width, paint)
            if small:
                threshold = report.get("thresholds") or {}
                _wrap_styled(lines, "Tiny output: ≤%s tool-result tokens before a ≥%s context read. Longest run: %s calls." % (
                    _tokens(_integer(threshold.get("small_result_chars"), 1024) / 4),
                    _tokens(threshold.get("minimum_context_tokens", 32768)),
                    values.get("small_result_max_streak", "?")), width, paint, "detail", "   ")
                _wrap_styled(lines, "Tool-output size; use --explain to identify polling.",
                             width, paint, "detail", "   ")
            missing = _integer(values.get("unweighted_responses"))
            if missing:
                _wrap_styled(lines, "%s calls lack model weights." % missing, width, paint, "warning", "   ")
        else:
            _wrap_styled(lines, finding.get("title"), width, paint, "summary", "   ")
            for evidence in (finding.get("evidence") or [])[:2]:
                _wrap_styled(lines, evidence, width, paint, "detail", "   ")

        events = [e for e in all_shakes if e.get("session_id") == finding.get("session_id")
                  and e.get("prompt") == finding.get("prompt")]
        _render_shakes(lines, events, width, paint)
        _render_estimated_shakes(lines, [e for e in report.get("estimated_shake_savings", [])
            if e["harness"] == finding.get("harness") and e["session_id"] == finding["session_id"]
            and e["prompt"] == finding["prompt"]], width, paint)
        candidate = finding.get("shake_candidate")
        if candidate:
            _field(lines, "Preview /shake on continuation: ",
                   "%s (%s context)" % (candidate.get("thread_id"), _tokens(candidate.get("last_context_tokens"))),
                   width, paint, "instruction", "   ")
        # One useful drilldown per finding; role/thread drilldowns remain in JSON.
        next_steps = [entry for entry in finding.get("next") or [] if entry.get("cmd") != "/shake"]
        if next_steps:
            _append_next(lines, [{"cmd": next_steps[0].get("cmd")}], width, paint,
                         indent="   ", seen=seen_commands)
    if any(f.get("values") for f in findings):
        lines.append("")
        _wrap_styled(lines, "* Weighted usage share within this prompt. Details: --json.",
                     width, paint, "detail")
    return _finish(lines, view)


def _command_text(command, paint):
    """Highlight Nenpi-generated shell words; preserve the exact copyable command.

    This is intentionally a small lexer for generated commands, not a shell
    parser. Palette roles mirror Codex's Catppuccin shell syntax defaults.
    """
    words = re.finditer(r"\s+|'[^']*'|\"(?:\\.|[^\"])*\"|[^\s'\"]+", command)
    pieces = []
    first = True
    for match in words:
        word = match.group()
        if word.isspace():
            pieces.append(word)
        elif first:
            pieces.append(paint(word, "command"))
            first = False
        elif word.startswith(("'", '\"')):
            pieces.append(paint(word, "command_string"))
        elif word.startswith("-"):
            prefix = "--" if word.startswith("--") else "-"
            pieces.append(paint(prefix, "command_punctuation") + paint(word[len(prefix):], "command_option"))
        else:
            pieces.append(paint(word, "command_argument"))
    return "".join(pieces)


def _append_next(lines: list[str], entries: Sequence[Mapping[str, Any]], width: int,
                 paint, indent: str = "", seen: set[str] | None = None) -> None:
    for entry in entries:
        command = entry.get("cmd")
        why = entry.get("why")
        if command:
            command = str(command)
            if seen is not None and command in seen:
                continue
            if seen is not None:
                seen.add(command)
            # Keep runnable commands on one logical line even when this exceeds
            # the report width; terminals can soft-wrap without changing paste.
            lines.extend(("", indent + _command_text(command, paint)))
            if why:
                _wrap_styled(lines, str(why), width, paint, "detail", indent + "      ")
        elif why:
            _wrap_styled(lines, str(why), width, paint, "detail", indent)


def _activity_row(lines: list[str], row: Mapping[str, Any], rank: int,
                  width: int, paint, denominator: float, outer: bool = False,
                  indent: str = "") -> None:
    label = row.get("tool_family") if outer else row.get("activity")
    label = label or row.get("activity") or row.get("tool") or "unknown activity"
    lines.append(paint(indent + "%d. %s" % (rank, label), "label"))

    calls = _integer(row.get("tool_calls", row.get("calls")))
    result_chars = _integer(row.get("result_chars"))
    responses = _integer(row.get("model_responses"))
    root_calls = row.get("root_calls")
    subagent_calls = row.get("subagent_calls")
    calls_label = "%d %s" % (calls, "call" if calls == 1 else "calls")
    if root_calls is not None or subagent_calls is not None:
        calls_label += " (%s root / %s subagent)" % (
            root_calls if root_calls is not None else "?",
            subagent_calls if subagent_calls is not None else "?",
        )
    _wrap_styled(
        lines,
        "%s · result text %s chars (~%s tokens estimated) · %d matched %s"
        % (calls_label, format(result_chars, ","),
           _tokens(row.get("est_result_tokens") if row.get("est_result_tokens") is not None
                   else result_chars / 4), responses,
           "response" if responses == 1 else "responses"),
        width, paint, "data", indent + "   ",
    )

    cached = _tokens(row.get("cached_input_tokens"))
    uncached = _tokens(row.get("uncached_input_tokens"))
    input_total = _integer(row.get("cached_input_tokens")) + _integer(row.get("uncached_input_tokens"))
    output_tokens = row.get("output_tokens")
    try:
        units = float(row.get("weighted_units", 0) or 0)
        unit_text = "%.2f weighted units" % units
        if denominator > 0:
            unit_text += " (%s of full matched weighted usage)" % _percent(units / denominator)
    except (TypeError, ValueError, OverflowError):
        unit_text = "weighted usage unknown"
    _wrap_parts(lines, [
        ("input processed ", "label"), (_tokens(input_total), "total"),
        (" (%s cached / %s uncached)" % (cached, uncached), "data"),
        (" · model output ", "label"),
        (_tokens(output_tokens) if output_tokens is not None else "unknown", "data"),
        (" · ", "label"), (unit_text, "total"),
    ], width, paint, indent + "   ")
    if row.get("unmatched_calls"):
        _wrap_styled(lines, "%s calls have no unique usage match" % row["unmatched_calls"],
                     width, paint, "warning", indent + "   ")
    if row.get("unweighted_responses"):
        _wrap_styled(lines, "%s responses lack model weights; input is still counted" % row["unweighted_responses"],
                     width, paint, "warning", indent + "   ")


def render_activity(activity: Mapping[str, Any], args: Any) -> str:
    """Render the outer result-size and response-usage bridge for tool activity.

    Family totals are the primary rows; static nested activities appear beneath
    each family so result volume and response-associated use can be assessed
    together without treating association as causation.
    """
    from .drain import make_painter, terminal_width

    view = _view_args(args)
    paint = make_painter(view)
    width = terminal_width(view)
    top = getattr(args, "top", None)
    limit = _integer(top, 0) if top is not None else 0
    lines = [paint("Activity and associated model usage", "heading")]

    total_calls = _integer(activity.get("tool_calls"))
    explained_calls = _integer(activity.get("explained_calls"))
    responses = _integer(activity.get("model_responses"))
    unmatched = _integer(activity.get("unmatched_calls"))
    matched_units = float(activity.get("weighted_units", 0) or 0)
    scope_responses = activity.get("scope_model_responses")
    coverage = activity.get("coverage_model_responses")
    usage_coverage = activity.get("coverage_weighted_usage")
    _wrap_styled(
        lines,
        "%d selected tool calls · %d parsed into activities · %d matched model responses · %d unpriced calls"
        % (total_calls, explained_calls, responses, unmatched),
        width, paint, "summary",
    )
    scope_chars = activity.get("scope_result_chars", activity.get("result_chars"))
    if scope_chars is not None:
        chars = _integer(scope_chars)
        estimated = activity.get("est_result_tokens")
        _wrap_styled(
            lines,
            "all selected tool results: %s chars (~%s tokens estimated at 4 chars/token)"
            % (format(chars, ","), _tokens(estimated if estimated is not None else chars / 4)),
            width, paint, "data",
        )
    explained_chars = activity.get("explained_result_chars")
    unexplained_chars = activity.get("unexplained_result_chars")
    if explained_chars is not None or unexplained_chars is not None:
        _wrap_styled(
            lines,
            "result characters in mapped calls: %s; from calls without a source/call match: %s"
            % (format(_integer(explained_chars), ","), format(_integer(unexplained_chars), ",")),
            width, paint, "data",
        )
    if scope_responses is not None:
        coverage_text = "response coverage %s / %s" % (responses, scope_responses)
        if coverage is not None:
            coverage_text += " (%s)" % _percent(coverage)
        if usage_coverage is not None:
            coverage_text += "; matched weighted-usage coverage %s" % _percent(usage_coverage)
        _wrap_styled(lines, coverage_text, width, paint, "detail")
    scope_units = activity.get("scope_weighted_units")
    if scope_units is not None:
        scope_text = "Shares below use %.2f matched weighted units" % matched_units
        if usage_coverage is not None:
            scope_text += " of %.2f scoped units (%s coverage)" % (
                _number(scope_units), _percent(usage_coverage))
        scope_text += "; shared responses are counted once."
    else:
        scope_text = "Matched responses total %.2f weighted units; shared responses are counted once." % matched_units
    _wrap_styled(lines, scope_text, width, paint, "data")
    scope_unweighted = activity.get("scope_unweighted_responses")
    matched_unweighted = activity.get("matched_unweighted_responses")
    if scope_unweighted is not None:
        _wrap_styled(
            lines,
            "%s of %s scoped responses lack model weights; %s matched responses are unweighted."
            % (scope_unweighted, scope_responses if scope_responses is not None else "unknown",
               matched_unweighted if matched_unweighted is not None else "unknown"),
            width, paint, "data",
        )
    if activity.get("unpriced_tool_call_responses"):
        _wrap_styled(lines, "%s tool-call batches had no unique usage record." % activity["unpriced_tool_call_responses"],
                     width, paint, "warning")

    family_rows = activity.get("tool_families") or []
    groups = activity.get("groups") or []
    if family_rows:
        lines.extend(("", paint("Tool families and their activity", "heading")))
        _wrap_styled(
            lines,
            "Families and nested activities include result size plus response-associated usage. "
            "Shares use the full matched weighted-usage total; nested rows are included in their family total.",
            width, paint, "detail",
        )
        children = {}
        for group in groups:
            children.setdefault(str(group.get("tool_family") or ""), []).append(group)
        shown = family_rows[:limit] if limit else family_rows
        for rank, family in enumerate(shown, 1):
            family_name = str(family.get("tool_family") or family.get("activity") or "unknown")
            _activity_row(lines, family, rank, width, paint, matched_units, outer=True)
            nested = sorted(
                children.get(family_name, []),
                key=lambda row: (_number(row.get("weighted_units")),
                                 _integer(row.get("tool_calls"))), reverse=True,
            )
            if len(nested) == 1 and (
                str(nested[0].get("activity")) == family_name
                or (family_name == "unlinked tool calls"
                    and nested[0].get("activity") == "tool activity unlinked")
            ):
                continue
            shown_nested = nested[:limit] if limit else nested
            for child_rank, row in enumerate(shown_nested, 1):
                _activity_row(lines, row, child_rank, width, paint, matched_units,
                              outer=False, indent="      ")
    elif groups:
        # Older payloads do not have the exclusive outer-family key yet.
        lines.extend(("", paint("Activity by matched response usage", "heading")))
        shown = groups[:limit] if limit else groups
        for rank, row in enumerate(shown, 1):
            _activity_row(lines, row, rank, width, paint, matched_units, outer=False)

    if activity.get("poll_only_calls"):
        _wrap_styled(
            lines,
            "%s calls reference only process polls or clock checks; longer waits may reduce model responses."
            % activity["poll_only_calls"],
            width, paint, "warning",
        )
    if activity.get("note"):
        _wrap_styled(lines, activity["note"], width, paint, "detail")
    return _finish(lines, view)


def _render_estimated_shakes(lines, estimates, width, paint):
    if not estimates:
        return
    lines.extend(("", paint("   Estimated Shake savings", "heading")))
    rows = []
    for item in estimates:
        net = item["net_read_equivalent_tokens_saved"]
        rows.append([
            (_session_label(item["thread_id"]), "label"),
            (_tokens(item["reduction_tokens"]), "data"),
            (_tokens(item["cache_write_tokens"]), "data"),
            (str(item["break_even_calls"]) if item["break_even_calls"] is not None else "not reached", "data"),
            (str(item["post_shake_calls"]), "data" if item["break_even_calls"] else "cost"),
            (("-" if net < 0 else "") + _tokens(abs(net)), "improvement" if net > 0 else "cost"),
        ])
    _table(lines, ["Thread", "Reduction*", "Cache write*", "Break-even turn*", "Post-Shake Turns", "Saved / cost*"], rows, width, paint)
    _wrap_styled(lines, "* Old tool outputs → 120-token stubs; protect 16K recent tokens. Best single intervention per thread/prompt. Net savings in cached-read-equivalent tokens; repetition savings are separate.", width, paint, "detail", "   ")


def render_activities(report, args):
    from .drain import make_painter, terminal_width
    view = _view_args(args)
    paint, width = make_painter(view), terminal_width(view)
    scope = report["scope"]
    total = report["summary"]
    lines = []
    _wrap_styled(lines, "%s · %s input · %s calls" % (
        _scope_label(scope), _tokens(total["input_tokens"]), format(total["responses"], ",")), width, paint, "summary")
    for key, label in (("project", "Projects"), ("exclude_project", "Excluding")):
        if scope.get(key):
            _wrap_styled(lines, label + ": " + ", ".join(scope[key]), width, paint, "detail")
    labels = {"Reading/searching code": "Reads/searches", "Coordinating agents": "Agent coordination",
              "Editing code": "Edits", "Other / unknown": "Other", "Mixed activity": "Mixed",
              "Waiting for processes": "Process waits", "Waiting for agents": "Agent waits",
              "Waiting for CI": "CI checks"}
    def label(bucket):
        return labels.get(bucket["activity"], bucket["activity"])
    lines.append("")
    rows = [[(label(b), "label"), (_tokens(b["input_tokens"]), "total"),
             (_tokens(b["repeated_input_tokens"]) if b["repeated_input_tokens"] else "—",
              "cost" if b["repeated_input_tokens"] else "detail")] for b in report["activities"]]
    _table(lines, ["Activity", "Input", "Repeated"], rows, width, paint, indent="")
    hidden = report["total_activities"] - len(rows)
    if hidden:
        lines.append(paint("+%s more (--top 0)" % hidden, "detail"))
    if not rows:
        _wrap_styled(lines, "No matching activity. Try --since 7d or nenpi config.", width, paint, "summary")
    opportunities = report.get("repetition", [])
    if opportunities and not scope.get("activity"):
        lines.append("")
        _table(lines, ["Repeated activity", "Input", "Calls"], [
            [(label(b), "label"), (_tokens(b["repeated_input_tokens"]), "cost"),
             (format(b["repeated_responses"], ","), "data")] for b in opportunities], width, paint, indent="")
        lines.append(_command_text(opportunities[0]["inspect"], paint))
    elif not scope.get("activity") and report["activities"]:
        lines.append(_command_text(report["activities"][0]["inspect"], paint))
    if scope.get("activity"):
        for bucket in report["activities"]:
            groups = bucket.get("operation_groups", [])
            if groups:
                lines.append("")
                _table(lines, ["Operations", "Input", "Calls"], [
                    [(g["operation"], "label"), (_tokens(g["input_tokens"]), "total"),
                     (format(g["responses"], ","), "data")] for g in groups[:5]], width, paint, indent="")
                if len(groups) > 5:
                    lines.append(paint("+%s operation groups (--json)" % (len(groups) - 5), "detail"))
            for location in bucket["locations"][:5]:
                lines.append("")
                metric = "repeated_input_tokens" if bucket["repeated_responses"] else "input_tokens"
                _wrap_styled(lines, "%s · %s · prompt %s · %s %sinput" % (
                    location["project"], location["harness"], location["prompt"],
                    _tokens(location[metric]), "repeated " if metric == "repeated_input_tokens" else ""), width, paint, "label")
                lines.append(_command_text(location["inspect"], paint))
    estimates = [e for e in report.get("estimated_shake_savings", []) if e["net_read_equivalent_tokens_saved"] > 0]
    if estimates:
        estimate = max(estimates, key=lambda e: e["net_read_equivalent_tokens_saved"])
        lines.append("")
        _wrap_parts(lines, [("Shake · " + estimate["project"] + " · ", "label"),
            ("~" + _tokens(estimate["net_read_equivalent_tokens_saved"]) + " net read-equivalent saved", "improvement")], width, paint)
        lines.append(_command_text(estimate["inspect"], paint))
    return _finish(lines, view)
