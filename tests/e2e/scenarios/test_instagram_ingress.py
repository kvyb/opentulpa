from __future__ import annotations

from uuid import uuid4

import pytest
from harness.runner import E2EHarness

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.ingress]


def test_live_instagram_ingress_read_smoke(e2e_harness: E2EHarness) -> None:
    service = e2e_harness.composio_service
    result = service.get_instagram_conversation(
        customer_id="cust_e2e_ingress",
        conversation_id="conv_e2e_1",
        connected_account_id="acct_e2e_1",
    )
    assert result["ok"] is True
    assert result["summary"]["conversation_id"] == "conv_e2e_1"
    assert result["summary"]["latest_message_id"]

    report = e2e_harness.write_status_report(
        scenario="live_instagram_ingress_read_smoke",
        ok=True,
        details={
            "conversation_id": result["summary"]["conversation_id"],
            "latest_message_id": result["summary"]["latest_message_id"],
        },
    )
    assert report.exists()


def test_live_instagram_ingress_extract_and_local_sink(e2e_harness: E2EHarness) -> None:
    customer_id = "cust_e2e_ingress"
    csv_relative_path = f"tulpa_stuff/e2e/live_instagram_ingress_{uuid4().hex[:8]}.csv"

    upsert = e2e_harness.upsert_instagram_workflow(
        customer_id=customer_id,
        name="E2E IG Ingress Extract",
        conversation_id="conv_e2e_1",
        connected_account_id="acct_e2e_1",
        required_fields=["name", "phone", "date", "time", "party_size"],
        csv_relative_path=csv_relative_path,
    )
    assert upsert["status_code"] == 200
    workflow = upsert["payload"]["workflow"]

    run = e2e_harness.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])
    assert run["status_code"] == 200
    payload = run["payload"]
    assert payload["ok"] is True
    assert int(payload["processed_conversations"]) >= 1
    assert int(payload.get("matched_conversations") or 0) >= 1
    assert isinstance(payload.get("results"), list) and payload["results"]

    csv_path = e2e_harness.project_root / csv_relative_path
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Alex Rivera" in csv_text
    assert "+1 415 555 1234" in csv_text
    assert any(item.get("method") == "get_instagram_conversation" for item in e2e_harness.composio_service.calls)

    report = e2e_harness.write_status_report(
        scenario="live_instagram_ingress_extract_and_local_sink",
        ok=True,
        details={
            "workflow_id": workflow["workflow_id"],
            "processed_conversations": payload["processed_conversations"],
            "matched_conversations": payload.get("matched_conversations"),
            "csv_path": str(csv_path),
        },
    )
    assert report.exists()
