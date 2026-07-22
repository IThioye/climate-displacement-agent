"""Flask interface for running and observing the evidence agent."""

from __future__ import annotations

import hmac
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from agent import run as run_agent


DB_PATH = Path(os.getenv("AGENT_LOG_DB", ROOT / "data" / "processed" / "agent_runs.db"))
QUESTIONS_PATH = ROOT / "evaluation" / "interface_questions.json"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
AGENT_RUN_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                question TEXT NOT NULL,
                region TEXT NOT NULL DEFAULT '',
                country TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                outcome TEXT,
                answer TEXT,
                critic TEXT,
                sources_json TEXT NOT NULL DEFAULT '[]',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                events_json TEXT NOT NULL DEFAULT '[]',
                error TEXT
            )
        """)


def create_log(run_id: str, question: str, region: str, country: str) -> None:
    with db_connection() as connection:
        connection.execute(
            "INSERT INTO runs (id, created_at, question, region, country, status) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, utc_now(), question, region, country, "queued"),
        )


def update_log(run_id: str, job: dict) -> None:
    result = job.get("result") or {}
    with db_connection() as connection:
        connection.execute(
            """UPDATE runs SET finished_at=?, status=?, outcome=?, answer=?, critic=?,
               sources_json=?, metrics_json=?, events_json=?, error=? WHERE id=?""",
            (
                utc_now(), job["status"], result.get("outcome"), result.get("answer"),
                result.get("critic"), json.dumps(result.get("sources", [])),
                json.dumps(result.get("metrics", {})), json.dumps(job["events"]),
                job.get("error"), run_id,
            ),
        )


def add_event(
    run_id: str,
    stage: str,
    message: str,
    admin_message: str | None = None,
) -> None:
    event = {
        "time": utc_now(), "stage": stage, "message": message,
        "admin_message": admin_message or message,
    }
    with JOBS_LOCK:
        if run_id in JOBS:
            JOBS[run_id]["events"].append(event)


def execute_run(run_id: str, question: str, region: str, country: str) -> None:
    try:
        add_event(
            run_id, "queue", "Your request is waiting to start.",
            "Run queued; AGENT_RUN_LOCK serializes model-backed execution.",
        )
        with AGENT_RUN_LOCK:
            with JOBS_LOCK:
                JOBS[run_id]["status"] = "running"
            add_event(
                run_id, "queue", "Your request is now being processed.",
                "Execution lock acquired; starting agent pipeline.",
            )
            result = run_agent(
                question, region=region, country=country,
                event_callback=lambda stage, message, admin_message=None: add_event(
                    run_id, stage, message, admin_message
                ),
            )
            with JOBS_LOCK:
                JOBS[run_id]["result"] = result
                JOBS[run_id]["status"] = "completed"
    except Exception as exc:
        add_event(
            run_id, "error", "The request could not be completed safely.",
            f"Run failed safely: {type(exc).__name__}: {exc}",
        )
        with JOBS_LOCK:
            JOBS[run_id]["status"] = "failed"
            JOBS[run_id]["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with JOBS_LOCK:
            snapshot = dict(JOBS[run_id])
            snapshot["events"] = list(JOBS[run_id]["events"])
        update_log(run_id, snapshot)


def admin_required(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        username = os.getenv("ADMIN_USERNAME", "admin")
        password = os.getenv("ADMIN_PASSWORD", "change-me")
        auth = request.authorization
        valid = (
            auth is not None
            and hmac.compare_digest(auth.username or "", username)
            and hmac.compare_digest(auth.password or "", password)
        )
        if not valid:
            return Response(
                "Admin authentication required.", 401,
                {"WWW-Authenticate": 'Basic realm="Agent administration"'},
            )
        return function(*args, **kwargs)
    return wrapped


def dashboard_data() -> dict:
    with db_connection() as connection:
        summary = connection.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                   SUM(CASE WHEN outcome='blocked' THEN 1 ELSE 0 END) AS blocked
            FROM runs
        """).fetchone()
        recent = connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

    latencies, costs, tool_totals, publisher_totals = [], [], {}, {}
    normalized_runs = []
    for row in recent:
        item = dict(row)
        metrics = json.loads(item.pop("metrics_json") or "{}")
        sources = json.loads(item.pop("sources_json") or "[]")
        item["metrics"] = metrics
        item["sources"] = sources
        item.pop("events_json", None)
        if "latency_s" in metrics:
            latencies.append(float(metrics["latency_s"]))
        if "estimated_cost_usd" in metrics:
            costs.append(float(metrics["estimated_cost_usd"]))
        for tool, count in metrics.get("tool_calls", {}).items():
            tool_totals[tool] = tool_totals.get(tool, 0) + int(count)
        for source in sources:
            publisher = source.get("publisher", "Unknown")
            publisher_totals[publisher] = publisher_totals.get(publisher, 0) + 1
        normalized_runs.append(item)
    return {
        "summary": dict(summary),
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "avg_cost": sum(costs) / len(costs) if costs else 0,
        "tool_totals": tool_totals,
        "publisher_totals": publisher_totals,
        "recent": normalized_runs,
    }


def create_app(testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config.update(TESTING=testing, MAX_CONTENT_LENGTH=32_000)
    initialize_database()

    @app.get("/")
    def index():
        questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        return render_template("index.html", questions=questions)

    @app.post("/api/runs")
    def start_run():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        region = str(payload.get("region", "")).strip()[:100]
        country = str(payload.get("country", "")).strip()[:100]
        if not question:
            return jsonify({"error": "Question is required."}), 400
        if len(question) > 8_000:
            return jsonify({"error": "Question exceeds the 8,000 character limit."}), 400
        run_id = uuid.uuid4().hex
        job = {"id": run_id, "status": "queued", "events": [], "result": None, "error": None}
        with JOBS_LOCK:
            JOBS[run_id] = job
        create_log(run_id, question, region, country)
        thread = threading.Thread(
            target=execute_run, args=(run_id, question, region, country), daemon=True,
            name=f"agent-run-{run_id[:8]}",
        )
        thread.start()
        return jsonify({"run_id": run_id, "status": "queued"}), 202

    @app.get("/api/runs/<run_id>")
    def run_status(run_id: str):
        with JOBS_LOCK:
            job = JOBS.get(run_id)
            if job:
                return jsonify({
                    "id": run_id, "status": job["status"],
                    "events": list(job["events"]), "result": job["result"], "error": job["error"],
                })
        with db_connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            abort(404)
        item = dict(row)
        return jsonify({
            "id": run_id, "status": item["status"],
            "events": json.loads(item["events_json"]),
            "result": {
                "answer": item["answer"], "critic": item["critic"],
                "sources": json.loads(item["sources_json"]),
                "metrics": json.loads(item["metrics_json"]), "outcome": item["outcome"],
            },
            "error": item["error"],
        })

    @app.get("/admin")
    @admin_required
    def admin():
        return render_template("admin.html", **dashboard_data())

    @app.get("/admin/runs/<run_id>")
    @admin_required
    def admin_run(run_id: str):
        with db_connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            abort(404)
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json") or "[]")
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        item["events"] = json.loads(item.pop("events_json") or "[]")
        return render_template("run_detail.html", run=item)

    @app.get("/admin/logs.json")
    @admin_required
    def admin_logs():
        with db_connection() as connection:
            rows = [dict(row) for row in connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT 500"
            ).fetchall()]
        return jsonify(rows)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "agent_version": "1.0.0"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        threaded=True,
    )
