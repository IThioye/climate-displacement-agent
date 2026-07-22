"""Flask route, asynchronous job, logging, and admin authentication tests."""

from __future__ import annotations

import base64
import time

import app as webapp


def authorization(username="admin", password="test-password") -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def fake_agent(question, region="", country="", event_callback=None):
    event_callback("security", "The request passed the safety check.", "L1 passed in interface test.")
    event_callback(
        "retrieval", "The most relevant passages were selected.",
        "BM25+dense+RRF test path selected two passages.",
    )
    return {
        "answer": f"Test answer for: {question}",
        "critic": "VERDICT: PASS",
        "sources": [{
            "source_id": "test", "publisher": "IDMC", "year": 2025,
            "page": 2, "url": "https://example.test/report.pdf",
        }],
        "metrics": {
            "latency_s": 0.01, "estimated_cost_usd": 0,
            "tool_calls": {"search_evidence": 1},
        },
        "outcome": "completed",
    }


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-password")
    monkeypatch.setattr(webapp, "DB_PATH", tmp_path / "runs.db")
    monkeypatch.setattr(webapp, "run_agent", fake_agent)
    webapp.JOBS.clear()
    return webapp.create_app(testing=True).test_client()


def test_index_and_health(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    assert client.get("/health").get_json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert b"Questions to test" in page.data
    assert b"Region filter" not in page.data
    assert b"Run measurements" not in page.data


def test_async_run_is_logged_and_visible_to_admin(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    response = client.post("/api/runs", json={"question": "Test displacement evidence"})
    assert response.status_code == 202
    run_id = response.get_json()["run_id"]

    deadline = time.time() + 3
    payload = None
    while time.time() < deadline:
        payload = client.get(f"/api/runs/{run_id}").get_json()
        if payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)
    assert payload["status"] == "completed"
    assert payload["result"]["outcome"] == "completed"
    assert len(payload["events"]) >= 4

    assert client.get("/admin").status_code == 401
    admin = client.get("/admin", headers=authorization())
    assert admin.status_code == 200
    assert b"Test displacement evidence" in admin.data
    detail = client.get(f"/admin/runs/{run_id}", headers=authorization())
    assert detail.status_code == 200
    assert b"VERDICT: PASS" in detail.data
    assert b"BM25+dense+RRF" in detail.data


def test_empty_question_is_rejected(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    response = client.post("/api/runs", json={"question": "  "})
    assert response.status_code == 400
