"""Known-answer regression tests for issue #1910.

#1910: "SessionStart hook injects expired/superseded memories unlabeled".

Memanto's lifecycle is status-based (``status`` / ``expired_at`` /
``expired_by``); the old ``expires_at`` / ``ttl_seconds`` TTL pair was removed
(memory_write_service.py _REMOVED_SCHEMA_FIELDS). So "expired" below means
``status == "expired"`` as stamped by ``set_lifecycle``, not a TTL deadline.

Needs no API key: the Moorcheh backend is replaced by an in-memory fake that
honours the ``#memory_type:`` exact filter, like tests/test_memory_read_multi_type.py.

Drop into memanto/tests/ and run:  pytest tests/test_issue_1910_expired_injection.py
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memanto.app.services.memory_read_service import MemoryReadService

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = (
    REPO_ROOT / "memanto" / "cli" / "connect" / "assets" / "hooks" / "session_start.py"
)

SYNC_QUERY_TYPES = ["instruction", "preference", "goal"]

# Hook tests run a POSIX sh stub named `memanto`; Windows cannot run it.
POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="fake memanto stub is a POSIX sh script"
)


# --------------------------------------------------------------------------- #
# Fake backend
# --------------------------------------------------------------------------- #
def _row(
    mid,
    content,
    *,
    mtype="preference",
    status="active",
    score=0.9,
    expired_at=None,
    updated="2026-09-01T00:00:00Z",
    legacy=False,
):
    row = {
        "id": mid,
        "text": f"[{mtype.upper()}] {mid}\n\n{content}",
        "memory_type": mtype,
        "agent_id": "agent-1",
        "actor_id": "agent-1",
        "source": "user",
        "confidence": 0.95,
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": updated,
        "score": score,
    }
    if not legacy:  # records written before the lifecycle field carry no status
        row["status"] = status
    if expired_at:
        row["expired_at"] = expired_at
        row["expired_by"] = "manual"
    return row


class _Search:
    def __init__(self, rows):
        self.rows = rows

    def query(self, **kw):
        wanted = re.findall(r"#memory_type:([A-Za-z0-9_.-]+)", kw["query"])
        hits = [r for r in self.rows if all(r["memory_type"] == w for w in wanted)]
        hits.sort(key=lambda r: r["score"], reverse=True)
        return {"results": hits[: kw["top_k"]], "execution_time": 0.0}


class _Client:
    def __init__(self, rows):
        self.similarity_search = _Search(rows)


def _recall(rows, status="active", types=None):
    svc = MemoryReadService(_Client(rows))
    out = svc.search_memories(
        query="user preferences",
        agent_id="agent-1",
        type=types or ["preference"],
        min_confidence=0.8,
        min_similarity_score=0.15,
        status=status,
        limit=10,
    )
    return [m["id"] for m in out["results"]], out["results"]


# --------------------------------------------------------------------------- #
# 1-7: known-answer cases on the read path used by `memanto memory sync`
# --------------------------------------------------------------------------- #
def test_01_expired_only_returns_nothing():
    ids, _ = _recall(
        [
            _row(
                "A",
                "Deploy on Fridays",
                status="expired",
                expired_at="2026-09-10T00:00:00Z",
            )
        ]
    )
    assert ids == []


def test_02_expired_plus_replacement_returns_only_replacement():
    ids, _ = _recall(
        [
            _row(
                "A",
                "Deploy on Fridays",
                status="expired",
                expired_at="2026-09-10T00:00:00Z",
                score=0.95,
            ),
            _row("B", "Deploy on Tuesdays", score=0.90),
        ]
    )
    assert ids == ["B"]


def test_03_reworded_replacement_ranks_lower_but_still_wins():
    # The stale version is the better semantic match; it must still be dropped.
    ids, _ = _recall(
        [
            _row(
                "A",
                "Always use tabs",
                status="expired",
                expired_at="2026-09-10T00:00:00Z",
                score=0.99,
            ),
            _row("B", "Indentation: 4 spaces, never tab characters", score=0.41),
        ]
    )
    assert ids == ["B"]


def test_04_multiple_versions_only_newest_active_survives():
    ids, _ = _recall(
        [
            _row(
                "v1",
                "Budget is 5k",
                status="expired",
                expired_at="2026-08-10T00:00:00Z",
                score=0.97,
            ),
            _row(
                "v2",
                "Budget is 8k",
                status="expired",
                expired_at="2026-09-01T00:00:00Z",
                score=0.96,
            ),
            _row("v3", "Budget is 12k", score=0.80),
        ]
    )
    assert ids == ["v3"]


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-09-30T23:59:59+14:00",  # stamp that is still "in the future" for UTC-12 readers
        "2026-09-30T00:00:00-12:00",
        "2026-09-30T12:00:00Z",  # "exactly now"-style stamp
    ],
)
def test_05_timezone_edge_expiry_is_status_not_clock(stamp):
    # Expiry is a stamped state, so the timestamp's offset must not matter.
    ids, _ = _recall(
        [
            _row(
                "A",
                "Office is in Berlin",
                status="expired",
                expired_at=stamp,
                score=0.95,
            ),
            _row("B", "Office is in Lisbon", score=0.90),
        ]
    )
    assert ids == ["B"]


def test_06_legacy_record_without_status_counts_as_active():
    ids, _ = _recall([_row("L", "Prefers dark mode", legacy=True)])
    assert ids == ["L"]


def test_07_multi_type_union_like_session_sync_drops_expired():
    rows = [
        _row(
            "i-old",
            "Run tests with nose",
            mtype="instruction",
            status="expired",
            expired_at="2026-09-01T00:00:00Z",
            score=0.97,
        ),
        _row("i-new", "Run tests with pytest", mtype="instruction", score=0.85),
        _row(
            "g-old",
            "Ship v1 by August",
            mtype="goal",
            status="expired",
            expired_at="2026-09-01T00:00:00Z",
            score=0.93,
        ),
        _row("g-new", "Ship v1 by October", mtype="goal", score=0.80),
    ]
    ids, _ = _recall(rows, types=SYNC_QUERY_TYPES)
    assert set(ids) == {"i-new", "g-new"}


def test_08_status_all_keeps_both_but_every_row_carries_its_status():
    # Documents the contract _filter_by_status relies on: callers that ask for
    # "all" must be able to label. Any renderer fed by status="all" has to use this.
    _, rows = _recall(
        [
            _row(
                "A",
                "Deploy on Fridays",
                status="expired",
                expired_at="2026-09-10T00:00:00Z",
            ),
            _row("B", "Deploy on Tuesdays"),
        ],
        status="all",
    )
    assert {r["id"]: r["status"] for r in rows} == {"A": "expired", "B": "active"}


# --------------------------------------------------------------------------- #
# 9: end to end through the CLI command the SessionStart hook runs
# --------------------------------------------------------------------------- #
def test_09_memory_sync_injects_replacement_never_expired(tmp_path):
    from typer.testing import CliRunner

    from memanto.cli.main import app

    rows = [
        _row(
            "A",
            "Deploy on Fridays",
            mtype="instruction",
            status="expired",
            expired_at="2026-09-10T00:00:00Z",
            score=0.95,
        ),
        _row("B", "Deploy on Tuesdays", mtype="instruction", score=0.90),
    ]
    svc = MemoryReadService(_Client(rows))
    client = MagicMock()
    calls = []

    def recall(**kw):
        calls.append(kw)
        res = svc.search_memories(
            query=kw["query"],
            agent_id=kw["agent_id"],
            type=kw.get("type"),
            min_confidence=kw.get("min_confidence"),
            min_similarity_score=kw.get("min_similarity"),
            status=kw.get("status", "all"),
            limit=kw.get("limit", 10),
        )
        return {"memories": res["results"]}

    client.recall.side_effect = recall
    captured = {}

    def fake_inject(project_dir, text, **_):
        captured["text"] = text
        return {"updated": ["CLAUDE.md"]}

    with (
        patch("memanto.cli.commands.memory_mgmt.get_client", return_value=client),
        patch("memanto.cli.commands.memory_mgmt.config_manager") as cfg,
        patch(
            "memanto.cli.connect.updater.inject_dynamic_memories",
            side_effect=fake_inject,
        ),
    ):
        cfg.get_active_session.return_value = ("agent-1", "tok")
        result = CliRunner().invoke(
            app, ["memory", "sync", "--project-dir", str(tmp_path)]
        )

    assert result.exit_code == 0, result.stdout
    assert calls and calls[0]["status"] == "active"  # the #2017 behaviour, now locked
    assert "Deploy on Tuesdays" in captured["text"]
    assert "Deploy on Fridays" not in captured["text"]
    assert not (tmp_path / "MEMORY.md").exists()  # sync no longer writes MEMORY.md


# --------------------------------------------------------------------------- #
# 10: the SessionStart hook must report what `memory sync` recalled, not a
#     stale MEMORY.md (sync stopped writing that file in #2017)
# --------------------------------------------------------------------------- #
def _run_hook(tmp_path, sync_stdout):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "MEMORY.md").write_text(  # stale pre-#2017 file built with status='all'
        "# Memory — agent-1\n\n> Total memories: **7**  \n\n"
        "### deploy-day\n\n> Deploy on Fridays\n\n*Status: active*\n\n"
        "### deploy-day-v2\n\n> Deploy on Tuesdays\n\n*Status: active*\n",
        encoding="utf-8",
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "memanto"  # stands in for `memanto memory sync`
    fake.write_text(f"#!/bin/sh\necho '{sync_stdout}'\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "TMPDIR": str(tmp_path),
    }
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"cwd": str(project), "session_id": "t10"}),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    ).stdout


@POSIX_ONLY
def test_10_hook_does_not_advertise_stale_memory_md(tmp_path):
    out = _run_hook(tmp_path, "OK Recalled 2 dynamic memories!")
    assert "Read MEMORY.md" not in out, out
    assert "7 memories" not in out, out  # count must not come from the stale file
    assert "2 active memories" in out, out  # count comes from the sync output


@POSIX_ONLY
def test_10b_hook_reports_empty_sync(tmp_path):
    out = _run_hook(
        tmp_path, "No active dynamic memories found. Cleared dynamic sections (if any)."
    )
    assert "Read MEMORY.md" not in out, out
    assert "no active standing memories" in out, out
