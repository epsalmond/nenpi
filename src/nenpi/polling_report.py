"""What the polling/recipe advisory would have flagged, and what it cost.

Classification is `command_classification.classify` over the same normalized
responses `activities` uses; cost is each flagged response's own input and
weighted units. Nothing here leaves the machine or persists.
"""
from collections import defaultdict
import json
import statistics

from .command_classification import RECIPES, THRESHOLD, WINDOW_STEPS, Step, classify

SWEEP = (2, 3, 4, 5)
MECHANICAL_MIN_RESPONSES = 6
MECHANICAL_SMALL_RESULT_TOKENS = 500


def _empty():
    return dict(responses=0, input_tokens=0, cached_input_tokens=0, output_tokens=0, weighted_units=0.0)


def _add(row, r):
    row["responses"] += 1
    row["input_tokens"] += r.context
    row["cached_input_tokens"] += r.cached
    row["output_tokens"] += r.output
    row["weighted_units"] += r.usage


def _sequences(responses):
    """Steps per thread and prompt in transcript order; unknown threads are left out."""
    groups = defaultdict(list)
    for r in responses:
        if r.thread:
            groups[(r.harness, r.session, r.thread, r.prompt)].append(r)
    for key, rows in groups.items():
        rows.sort(key=lambda r: (r.timestamp, r.response_id))
        owners = [(r, op.step or Step()) for r in rows for op in r.operations]
        yield key, rows, owners


def flag_responses(responses, threshold=THRESHOLD, window_steps=WINDOW_STEPS):
    """One record per response the advisory would have fired on."""
    flagged = []
    for _key, _rows, owners in _sequences(responses):
        verdicts = classify([step for _r, step in owners], threshold, window_steps)
        by_response = {}
        for (r, step), verdict in zip(owners, verdicts):
            if not (verdict.polling or verdict.recipe_id):
                continue
            record = by_response.get(id(r))
            if record is None:
                activity = next((op.activity for op in r.operations if op.step is step), "")
                record = by_response[id(r)] = dict(response=r, kind="none", recipe_id=None,
                    label=step.label or activity or "unknown", pattern=step.source if verdict.recipe_id and not verdict.polling else step.signature)
            if verdict.kind == "pure_poll" or (verdict.kind == "watch" and record["kind"] == "none"):
                record["kind"] = verdict.kind
            record["recipe_id"] = record["recipe_id"] or verdict.recipe_id
        flagged.extend(by_response.values())
    return flagged


def _totals(flagged, responses):
    total = _empty()
    for r in responses:
        _add(total, r)
    summary = dict(_empty(), scope=total)
    by_kind = {kind: _empty() for kind in ("pure_poll", "watch", "recipe_only")}
    by_recipe = {recipe: _empty() for recipe in RECIPES}
    for f in flagged:
        r = f["response"]
        _add(summary, r)
        _add(by_kind[f["kind"] if f["kind"] != "none" else "recipe_only"], r)
        if f["recipe_id"]:
            _add(by_recipe[f["recipe_id"]], r)
    for key in ("responses", "input_tokens", "weighted_units"):
        summary[key + "_share"] = summary[key] / total[key] if total[key] else None
    return summary, by_kind, by_recipe


def _patterns(flagged, top):
    rows = {}
    for f in flagged:
        r = f["response"]
        key = (f["label"], f["recipe_id"])
        row = rows.setdefault(key, dict(_empty(), pattern=f["label"], recipe_id=f["recipe_id"],
            pure_poll=0, watch=0, recipe_only=0, sessions=set(), runs=set()))
        _add(row, r)
        row[f["kind"] if f["kind"] != "none" else "recipe_only"] += 1
        row["sessions"].add((r.harness, r.session))
        row["runs"].add((r.harness, r.session, r.thread, r.prompt, f["pattern"]))
    ranked = sorted(rows.values(), key=lambda row: (-row["input_tokens"], row["pattern"]))
    for row in ranked:
        row["sessions"] = len(row["sessions"])
        row["runs"] = len(row["runs"])
    return ranked[:top] if top else ranked


def _advisory(flagged):
    """If one advisory per (session, pattern) were obeyed, later flags are the ceiling saved."""
    runs = defaultdict(list)
    for f in flagged:
        r = f["response"]
        runs[(r.harness, r.session, f["label"], f["pattern"])].append(f)
    counts = sorted(len(v) for v in runs.values())
    after = _empty()
    for rows in runs.values():
        for f in sorted(rows, key=lambda f: f["response"].timestamp)[1:]:
            _add(after, f["response"])
    return dict(session_patterns=len(counts),
        flags_per_pattern=dict(median=statistics.median(counts) if counts else 0,
            p90=counts[int(0.9 * (len(counts) - 1))] if counts else 0, max=counts[-1] if counts else 0),
        after_first_advisory=after)


def _by_session(flagged, top):
    sessions = {}
    for f in flagged:
        r = f["response"]
        row = sessions.setdefault((r.harness, r.session), dict(_empty(), harness=r.harness, account=r.account,
            session_id=r.session, project=r.project, polling_responses=0, recipe_responses=0, prompts={}))
        _add(row, r)
        row["polling_responses"] += int(f["kind"] != "none")
        row["recipe_responses"] += int(bool(f["recipe_id"]))
        prompt = row["prompts"].setdefault(r.prompt, dict(_empty(), prompt=r.prompt, polling_responses=0,
            recipe_responses=0, recipes={}))
        _add(prompt, r)
        prompt["polling_responses"] += int(f["kind"] != "none")
        prompt["recipe_responses"] += int(bool(f["recipe_id"]))
        if f["recipe_id"]:
            prompt["recipes"][f["recipe_id"]] = prompt["recipes"].get(f["recipe_id"], 0) + 1
    ranked = sorted(sessions.values(), key=lambda row: (-row["input_tokens"], row["session_id"]))
    for row in ranked:
        row["prompts"] = sorted(row["prompts"].values(), key=lambda p: (-p["input_tokens"], p["prompt"]))
    return ranked[:top] if top else ranked


def mechanical_runs(responses, top, window_steps=WINDOW_STEPS):
    """Offline discovery only, never part of the hook contract.

    A run is consecutive responses in one thread and prompt with no write, each
    with linked tool calls and small results. Score rewards long runs of
    distinct commands, the shape of a procedure a script could own.
    """
    found = []
    for _key, rows, _owners in _sequences(responses):
        run = []
        for r in rows + [None]:
            steps = [op.step or Step() for op in r.operations] if r else []
            small = r is not None and steps and sum(t for _ts, t in r.results) <= MECHANICAL_SMALL_RESULT_TOKENS
            if r is not None and small and not any(s.write for s in steps):
                run.append(r)
                continue
            if len(run) >= MECHANICAL_MIN_RESPONSES:
                found.append(run)
            run = []
    rows = []
    for run in found:
        steps = [op.step or Step() for r in run for op in r.operations]
        signatures = [s.signature for s in steps if s.signature]
        distinct = len(set(signatures)) / len(signatures) if signatures else 0.0
        labels = []
        for r in run:
            for op in r.operations:
                label = (op.step.label if op.step else "") or op.activity
                if not labels or labels[-1] != label:
                    labels.append(label)
        row = dict(_empty(), harness=run[0].harness, session_id=run[0].session, prompt=run[0].prompt,
            score=round(len(run) * distinct, 2), distinct_ratio=round(distinct, 2), families=labels[:8])
        for r in run:
            _add(row, r)
        rows.append(row)
    rows.sort(key=lambda row: (-row["score"] * row["input_tokens"], row["session_id"]))
    return rows[:top] if top else rows


def build(responses, threshold=THRESHOLD, window_steps=WINDOW_STEPS, top=10, sweep=True):
    flagged = flag_responses(responses, threshold, window_steps)
    summary, by_kind, by_recipe = _totals(flagged, responses)
    report = dict(threshold=threshold, window_steps=window_steps, summary=summary,
        by_kind=by_kind, by_recipe=by_recipe, top_patterns=_patterns(flagged, top),
        advisory_once_per_session_pattern=_advisory(flagged), sessions=_by_session(flagged, top))
    if sweep:
        report["threshold_sweep"] = []
        for value in SWEEP:
            rows = flagged if value == threshold else flag_responses(responses, value, window_steps)
            row = _totals(rows, responses)[0]
            report["threshold_sweep"].append(dict(threshold=value, responses=row["responses"],
                input_tokens=row["input_tokens"], weighted_units=row["weighted_units"],
                pure_poll=sum(1 for f in rows if f["kind"] == "pure_poll")))
    return report


def build_report(analysis, args):
    from .activity_model import normalize
    from .activity_report import scoped_command
    responses = normalize(analysis)
    report = build(responses, args.threshold, args.window_steps, args.top)
    report["mechanical_runs"] = dict(note="offline discovery only; not part of the hook contract",
        runs=mechanical_runs(responses, args.top, args.window_steps))
    report.update(schema_version=1, scope=dict(since=args.since, until=args.until, harness=args.harness),
        recipes=RECIPES)
    report["next"] = [dict(cmd=scoped_command(analysis, "polling", "--json"), why="Full JSON with sessions and prompts"),
        dict(cmd=scoped_command(analysis, "activities"), why="Rank all activities")]
    for row in report["sessions"]:
        from . import drain as d
        row["inspect"] = scoped_command(analysis, "auto", "--harness", row["harness"],
            "--session", d.pick_id(row["session_id"], d.analysis_session_ids(analysis)))
    return report


def _tokens(value):
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= size:
            return "%.1f%s" % (value / size, unit)
    return str(int(value))


def _share(value):
    return "—" if value is None else "%.1f%%" % (value * 100)


def render(report):
    s = report["summary"]
    scope = report["scope"]
    lines = ["Polling/recipe advisory: what it would have flagged (since %s, threshold %d, window %d steps)" % (
        scope["since"] or "start", report["threshold"], report["window_steps"]),
        "  %d responses (%s of %d), input %s (%s), %.2f weighted units (%s)" % (
            s["responses"], _share(s["responses_share"]), s["scope"]["responses"], _tokens(s["input_tokens"]),
            _share(s["input_tokens_share"]), s["weighted_units"], _share(s["weighted_units_share"])), ""]
    lines.append("  %-22s %9s %10s %10s" % ("by kind", "responses", "input", "units"))
    for name, row in report["by_kind"].items():
        lines.append("  %-22s %9d %10s %10.2f" % (name, row["responses"], _tokens(row["input_tokens"]), row["weighted_units"]))
    for name, row in report["by_recipe"].items():
        lines.append("  %-22s %9d %10s %10.2f" % (name, row["responses"], _tokens(row["input_tokens"]), row["weighted_units"]))
    lines += ["", "  top patterns by input"]
    for rank, row in enumerate(report["top_patterns"], 1):
        lines.append("  %2d. %-28s %-20s %6d resp (%d pure/%d watch/%d recipe) %8s  %.2f units  %d sessions" % (
            rank, row["pattern"][:28], row["recipe_id"] or "-", row["responses"], row["pure_poll"], row["watch"],
            row["recipe_only"], _tokens(row["input_tokens"]), row["weighted_units"], row["sessions"]))
    advisory = report["advisory_once_per_session_pattern"]
    after = advisory["after_first_advisory"]
    lines += ["", "  one advisory per (session, pattern): %d patterns, flags each median %s / p90 %s / max %s;"
              " later flags %d responses, input %s, %.2f units" % (
                  advisory["session_patterns"], advisory["flags_per_pattern"]["median"],
                  advisory["flags_per_pattern"]["p90"], advisory["flags_per_pattern"]["max"],
                  after["responses"], _tokens(after["input_tokens"]), after["weighted_units"])]
    if report.get("threshold_sweep"):
        lines.append("  threshold sweep: " + ", ".join("%d → %d resp / %s" % (
            row["threshold"], row["responses"], _tokens(row["input_tokens"])) for row in report["threshold_sweep"]))
    lines += ["", "Next:"] + ["  %s  # %s" % (n["cmd"], n["why"]) for n in report["next"]]
    return "\n".join(lines)


def command_polling(args):
    from . import drain as d
    analysis = d.prepare(args)
    report = build_report(analysis, args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 0
