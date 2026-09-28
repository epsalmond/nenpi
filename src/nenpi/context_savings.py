"""Shared hypothetical tool-output reduction policy and cache payback model."""
from collections import defaultdict, deque

PROTECT_TOKENS = 16_000
STUB_TOKENS = 120
MIN_OUTPUT_TOKENS = 512
POLICY = {
    "name": "old-tool-outputs-v1", "protect_recent_tokens": PROTECT_TOKENS,
    "replacement_tokens": STUB_TOKENS, "minimum_output_tokens": MIN_OUTPUT_TOKENS,
    "selection": "best single intervention per thread/prompt",
    "horizon": "next context decrease, prompt boundary, or end of observed calls",
    "claude_unknown_cache_duration": "1h (conservative)",
}


def rates(response):
    p = response.prices
    read = p.get("cache_read", p.get("cached_input", 0))
    write = p.get(response.cache_write_kind or "cache_write_1h", p.get("input", 0)) if response.harness == "claude" else p.get("input", 0)
    if response.harness == "claude" and response.cache_write_kind == "cache_write_unknown":
        write = max(p.get("cache_write_5m", 0), p.get("cache_write_1h", 0))
    return read, write


def estimate_savings(responses, applied=()):
    """Estimate only outputs seen entering a continuously growing context.

    Context growth past each output establishes its protected-tail age. Any
    decrease clears history, rather than assuming a compacted output survived.
    The cache rewrite starts at the earliest changed output, preserving prefix.
    """
    excluded = {(e.get("session_id"), e.get("thread_id"), e.get("prompt")) for e in applied}
    streams = defaultdict(list)
    for r in responses:
        if not r.thread or (r.harness == "codex" and (r.session, r.thread, r.prompt) in excluded):
            continue
        streams[(r.harness, r.account, r.session, r.thread, r.prompt)].append(r)
    estimates = []
    for key, stream in streams.items():
        segments = []
        for r in stream:
            if not segments or r.reset:
                segments.append([])
            segments[-1].append(r)
        best = None
        for segment in segments:
            if any(not r.prices for r in segment):
                continue
            suffix_rates = [0.0] * (len(segment) + 1)
            for i in range(len(segment) - 1, -1, -1):
                suffix_rates[i] = suffix_rates[i + 1] + rates(segment[i])[0]
            pending = []
            retained = deque()
            removed = 0
            earliest = None
            for i, r in enumerate(segment):
                for ts, size, prefix, end in pending:
                    if ts < r.timestamp:
                        retained.append((size, prefix, end))
                pending = [item for item in pending if item[0] >= r.timestamp]
                while retained and r.context - retained[0][2] >= PROTECT_TOKENS:
                    size, prefix, _ = retained.popleft()
                    if size >= MIN_OUTPUT_TOKENS:
                        removed += size - STUB_TOKENS
                        earliest = prefix if earliest is None else min(earliest, prefix)
                reduction = min(removed, max(0, r.context - PROTECT_TOKENS))
                if reduction and earliest is not None:
                    read, write = rates(r)
                    if read > 0:
                        cold = r.cached == 0
                        rebuild = 0 if cold else max(0, min(r.context - reduction, r.cached - reduction) - earliest)
                        penalty = rebuild * max(0, write - read) / 1e6
                        bonus = reduction * (write - read) / 1e6 if cold else 0
                        net = reduction * suffix_rates[i] / 1e6 + bonus - penalty
                        if best is None or net > best["net_units_saved"]:
                            balance, break_even = bonus - penalty, None
                            for turn, later in enumerate(segment[i:], 1):
                                balance += reduction * rates(later)[0] / 1e6
                                if balance >= 0:
                                    break_even = turn
                                    break
                            best = dict(
                                harness=key[0], account=key[1], session_id=key[2], thread_id=key[3], prompt=key[4],
                                project=r.project, timestamp=r.timestamp, context_before=r.context,
                                reduction_tokens=reduction, context_after=r.context - reduction,
                                cache_write_tokens=rebuild, cache_baseline="cold" if cold else "warm",
                                cache_write_kind=r.cache_write_kind or ("cache_write_1h" if r.harness == "claude" else "input"),
                                post_shake_calls=len(segment) - i, break_even_calls=break_even,
                                read_tokens_saved=reduction * (len(segment) - i), net_units_saved=net,
                                net_read_equivalent_tokens_saved=net * 1e6 / read,
                                policy=POLICY["name"],
                            )
                # Upper-bound insertion position includes assistant output and
                # all results in this batch before protecting the recent tail.
                end = r.context + r.output + sum(size for _, size in r.results)
                for ts, size in r.results:
                    pending.append((ts, size, r.context + r.output, end))
        if best:
            estimates.append(best)
    return sorted(estimates, key=lambda e: (-e["read_tokens_saved"], e["session_id"], e["thread_id"]))
