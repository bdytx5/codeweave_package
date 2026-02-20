"""Local JSONL logging for weave calls.

Patches the batch processor's processor_fn directly, which is the single
point all calls go through regardless of whether the project uses the legacy
start/end batch path or the newer calls_complete path.

Usage:
    import weave
    from weave.trace_server_bindings.jsonl_logging_trace_server import attach_jsonl_logger

    client = weave.init("my-project")
    log_path = attach_jsonl_logger(client)   # returns Path of the log file
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from weave.trace_server_bindings.models import CompleteBatchItem, EndBatchItem, StartBatchItem


# ---------------------------------------------------------------------------
# Global git path override
# ---------------------------------------------------------------------------

def set_git_path(path: str) -> None:
    """Set the git repo directory for snapshotting. Takes priority over
    the WEAVE_GIT_PATH environment variable. Can be called before or after weave.init().
    """
    import os as _os
    _os.environ['WEAVE_GIT_PATH'] = path


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: str) -> str:
    """Run a git command, return stdout stripped. Returns '' on failure."""
    try:
        return subprocess.check_output(
            ["git"] + args, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _capture_git_state(run_id: str) -> dict[str, Any]:
    """Capture git state at run time without touching the working tree.

    If the working tree is dirty, creates a snapshot commit on a ref
    named refs/cdweave/<run_id> — the working tree is NOT modified.
    The snapshot commit SHA can be used later to restore the exact code state
    with: git checkout <snapshot_commit> -- .

    Returns a dict with:
        git_repo_root     — absolute path to repo root (None if not a git repo)
        git_commit        — current HEAD SHA (short)
        git_dirty         — True if there were uncommitted changes
        git_snapshot_sha  — full SHA of the snapshot commit (if dirty), else None
    """
    import os
    cwd = os.environ.get('WEAVE_GIT_PATH')
    if not cwd:
        return {"git_repo_root": None, "git_commit": None, "git_dirty": False, "git_snapshot_sha": None}

    repo_root = _git(["rev-parse", "--show-toplevel"], cwd)
    if not repo_root:
        return {"git_repo_root": None, "git_commit": None, "git_dirty": False, "git_snapshot_sha": None}

    commit = _git(["rev-parse", "--short", "HEAD"], repo_root)

    # Check for any uncommitted changes
    status = _git(["status", "--porcelain"], repo_root)
    dirty = bool(status)

    snapshot_sha = None
    if dirty:
        # Write the current index + working tree into a tree object without
        # touching the working tree or the index permanently.
        # 1. Write a temporary index that includes all working-tree changes
        import tempfile, os as _os
        tmp_index = tempfile.mktemp(prefix=".cdweave_idx_", dir=_os.path.join(repo_root, ".git"))
        env = {**_os.environ, "GIT_INDEX_FILE": tmp_index}
        try:
            import subprocess
            # Copy current index into temp index (skip if no HEAD yet)
            if _git(["rev-parse", "HEAD"], repo_root):
                subprocess.check_call(
                    ["git", "read-tree", "HEAD"],
                    cwd=repo_root, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            # Add all working tree changes (tracked + untracked) into temp index
            subprocess.check_call(
                ["git", "add", "-A"],
                cwd=repo_root, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Write tree from temp index
            tree_sha = subprocess.check_output(
                ["git", "write-tree"],
                cwd=repo_root, env=env, text=True
            ).strip()
            # Create a commit object (does NOT move HEAD or any branch)
            head_sha = _git(["rev-parse", "HEAD"], repo_root)
            msg = f"cdweave snapshot: {run_id}"
            # If no HEAD yet (empty repo), create root commit with no parent
            commit_tree_args = ["git", "commit-tree", tree_sha, "-m", msg]
            if head_sha:
                commit_tree_args += ["-p", head_sha]
            snapshot_sha = subprocess.check_output(
                commit_tree_args, cwd=repo_root, text=True
            ).strip()
            # Store it in a named ref so it's reachable / not GC'd
            ref = f"refs/cdweave/{run_id}"
            _git(["update-ref", ref, snapshot_sha], repo_root)
        except Exception:
            snapshot_sha = None
        finally:
            for f in (tmp_index, tmp_index + ".lock"):
                try:
                    _os.unlink(f)
                except Exception:
                    pass

    return {
        "git_repo_root": repo_root,
        "git_commit": commit,
        "git_dirty": dirty,
        "git_snapshot_sha": snapshot_sha,
    }


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _default_log_path(project: str | None = None) -> Path:
    import os
    ts = time.strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    # Sanitise project name for use as a directory name
    safe_project = "".join(c if c.isalnum() or c in "-_." else "_" for c in (project or "default"))
    runs_dir = Path(os.path.expanduser("~")) / ".cache" / "codeweave" / safe_project
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / f"{ts}_{uid}.jsonl"


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


def _dt_to_ts(dt: Any) -> float | None:
    if dt is None:
        return None
    try:
        return dt.timestamp()
    except Exception:
        return None


def _find_remote_server(server: Any) -> Any:
    """Walk middleware wrappers to find the actual RemoteHTTPTraceServer."""
    if type(server).__name__ in ("RemoteHTTPTraceServer", "StainlessRemoteHTTPTraceServer"):
        return server
    next_server = getattr(server, "_next_trace_server", None)
    if next_server is not None:
        return _find_remote_server(next_server)
    return None


def _wandb_url(op_name: str | None, call_id: str | None) -> str | None:
    if not op_name or not call_id:
        return None
    try:
        parts = op_name.removeprefix("weave:///").split("/")
        entity, project = parts[0], parts[1]
        return f"https://wandb.ai/{entity}/{project}/r/call/{call_id}"
    except (IndexError, AttributeError):
        return None


def _source_for_op_name(op_name: str | None) -> dict[str, Any]:
    """Extract short name from op_name URI, find the matching live op, read its source attrs."""
    if not op_name:
        return {}
    short = op_name.split("/")[-1].split(":")[0]
    import sys
    for module in list(sys.modules.values()):
        try:
            obj = getattr(module, short, None)
        except Exception:
            continue
        if obj is not None and getattr(obj, "_source_file", None) is not None:
            name = getattr(getattr(obj, "resolve_fn", obj), "__name__", short)
            return {
                "function": name,
                "source_file": obj._source_file,
                "source_line_start": obj._source_line_start,
                "source_line_end": obj._source_line_end,
            }
    return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def attach_jsonl_logger(
    client: Any,
    log_path: str | Path | None = None,
) -> Path:
    """Patch the weave client to also log every completed call to a local JSONL file.

    Args:
        client:     The WeaveClient returned by weave.init().
        log_path:   Where to write logs. Defaults to runs/<timestamp>_<uid>.jsonl.

    Returns:
        The Path of the log file being written to.
    """
    project = getattr(client, "project", None)
    resolved_path = Path(log_path) if log_path else _default_log_path(project)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive run_id from the log filename
    run_id = resolved_path.stem

    # Capture git state — creates a snapshot commit if tree is dirty (does NOT touch working tree)
    git_info = _capture_git_state(run_id)

    remote_server = _find_remote_server(client.server)
    if remote_server is None:
        raise RuntimeError(
            "Could not find RemoteHTTPTraceServer. "
            "Call attach_jsonl_logger right after weave.init()."
        )

    proc = remote_server.call_processor
    if proc is None:
        raise RuntimeError("No call_processor found on the remote server.")

    original_fn = proc.processor_fn
    _pending: dict[str, dict[str, Any]] = {}

    def _append(entry: dict[str, Any]) -> None:
        with resolved_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, **git_info}) + "\n")

    def patched_fn(batch: list[Any], **kwargs: Any) -> None:
        # Pass 1: stash all starts first (ends may arrive before starts in the same batch)
        for item in batch:
            if isinstance(item, StartBatchItem):
                s = item.req.start
                if s.id:
                    _pending[s.id] = {
                        "op_name": s.op_name,
                        "timestamp_start": _dt_to_ts(s.started_at),
                        "inputs": _safe_json(s.inputs),
                        "parent_id": s.parent_id,
                        "trace_id": s.trace_id,
                        "attributes": _safe_json(s.attributes),
                    }

        # Pass 2: write complete entries
        for item in batch:
            if isinstance(item, CompleteBatchItem):
                call = item.req
                ts_start = _dt_to_ts(call.started_at)
                ts_end = _dt_to_ts(call.ended_at)
                _append({
                    "call_id": call.id,
                    "op_name": call.op_name,
                    "wandb_url": _wandb_url(call.op_name, call.id),
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                    "duration_s": (ts_end - ts_start) if ts_end and ts_start else None,
                    "inputs": _safe_json(call.inputs),
                    "output": _safe_json(call.output),
                    "error": call.exception,
                    "parent_id": call.parent_id,
                    "trace_id": call.trace_id,
                    "attributes": _safe_json(call.attributes),
                    "summary": _safe_json(dict(call.summary) if call.summary else {}),
                    **_source_for_op_name(call.op_name),
                })

            elif isinstance(item, EndBatchItem):
                e = item.req.end
                pending = _pending.pop(e.id, {})
                ts_start = pending.get("timestamp_start")
                ts_end = _dt_to_ts(e.ended_at)
                op_name = pending.get("op_name")
                _append({
                    "call_id": e.id,
                    "op_name": op_name,
                    "wandb_url": _wandb_url(op_name, e.id),
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                    "duration_s": (ts_end - ts_start) if ts_end and ts_start else None,
                    "inputs": pending.get("inputs"),
                    "output": _safe_json(e.output),
                    "error": e.exception,
                    "parent_id": pending.get("parent_id"),
                    "trace_id": pending.get("trace_id"),
                    "attributes": pending.get("attributes"),
                    "summary": _safe_json(dict(e.summary) if e.summary else {}),
                    **_source_for_op_name(op_name),
                })

        original_fn(batch, **kwargs)

    proc.processor_fn = patched_fn
    return resolved_path
