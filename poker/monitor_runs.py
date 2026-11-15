"""Read-only live run snapshot; update the shared ledger without launching or stopping anything."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ac2.sdk import Client
from vals_tools.score import score_rows, summarize


def main():
    client = Client(project="vals-hackathon")
    path = Path("poker/run_ledger.json")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    observed = datetime.now(timezone.utc).isoformat()
    summaries = []
    for run in ledger["runs"]:
        if run["id"].startswith("eval_"):
            info = client.eval.get(run["id"])
            run["status"] = info.status
            run["results"] = summarize(score_rows(client, eval_id=run["id"]))
            summaries.append({"id": run["id"], "status": info.status, "scored_games": run["results"]["games"]})
        else:
            info = client.train.get(run["id"])
            run["status"] = info.status
            run["checkpoints"] = [{"id": c.id, "step": c.iter_num, "created_at": c.created_at} for c in client.train.checkpoints(run["id"])]
            run["step_metrics"] = client.train.get_step_metrics(train_job_id=run["id"])
            summaries.append({"id": run["id"], "status": info.status, "checkpoints": run["checkpoints"], "metric_phases": list(run["step_metrics"])})
        run["observed_at"] = observed
    ledger["datasets"] = [{"id": d.id, "name": d.name, "task_count": d.task_count, "updated_at": d.updated_at}
                          for d in client.datasets.list()]
    ledger["observed_at"] = observed
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    output = Path("out/runs/live_snapshot.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"observed_at": observed, "runs": summaries}, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
