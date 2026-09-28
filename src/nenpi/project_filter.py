"""Project selection after quota attribution, without caching full working paths."""
import hashlib
import fnmatch
import json
import os


def matches(path, pattern):
    pattern = os.path.normpath(os.path.expanduser(pattern))
    if not os.path.isabs(pattern) and '/' not in pattern:
        return fnmatch.fnmatchcase(os.path.basename(path), pattern)
    path = os.path.normpath(path)
    return fnmatch.fnmatchcase(path, pattern) or path.startswith(pattern.rstrip('/') + '/')


def source_cwd(source):
    # Metadata is near the beginning; never walk large tool-output bodies.
    try:
        with open(source, encoding='utf-8') as stream:
            for _ in range(100):
                line = stream.readline(256 * 1024)
                if not line or not line.endswith('\n'):
                    break
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                payload = record.get('payload') if record.get('type') == 'session_meta' else record
                cwd = payload.get('cwd') if isinstance(payload, dict) else None
                if isinstance(cwd, str) and cwd:
                    return cwd
    except OSError:
        pass
    return ''


def filter_analysis(analysis):
    from . import drain as d
    include = getattr(analysis.args, 'project', []) or []
    exclude = getattr(analysis.args, 'exclude_project', []) or []
    if not include and not exclude:
        return
    scan = analysis.scan
    paths = {}
    for entry, harness, source in scan._tool_sources:
        if source:
            cwd = source_cwd(source)
            for sid, summary in entry.sessions.items():
                key = (harness, sid)
                # Prefer the source matching the session's chosen working directory.
                if key not in paths or (cwd and hashlib.sha1(cwd.encode()).hexdigest()[:12] == scan.sessions.get(key, summary).cwd_hash):
                    paths[key] = cwd
    keep = {key for key, summary in scan.sessions.items()
            if (not include or any(matches(paths.get(key) or summary.cwd, p) for p in include))
            and not any(matches(paths.get(key) or summary.cwd, p) for p in exclude)}
    for name in ('sessions', 'boundaries', 'prompt_labels', 'compactions', 'session_accounts'):
        setattr(scan, name, {k: v for k, v in getattr(scan, name).items() if k in keep})
    scan.events = {h: [e for e in rows if (h, e[d.EVENT_SESSION]) in keep] for h, rows in scan.events.items()}
    scan.filter_tools(lambda h, rows: [r for r in rows if (h, r[d.TOOL_SESSION]) in keep])
    analysis.prompts = {k: v for k, v in analysis.prompts.items() if k in keep}
