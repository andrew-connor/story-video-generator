"""
Tracks API costs across the pipeline and writes to a running log file.
Costs are written immediately after each API call, not batched at the end.
"""

import json
from datetime import datetime
from pathlib import Path


def safe_write_local(write_fn, filepath: str = None):
    """Write with PermissionError retry — local copy to avoid circular import."""
    while True:
        try:
            write_fn()
            return
        except PermissionError:
            label = filepath or "a file"
            print(f"\n  ⚠️  Access denied writing to {label}. Please close it.")
            retry = input("  Press Enter to retry, or type 'skip' to skip: ").strip().lower()
            if retry == "skip":
                return

LOG_FILE = Path("pipeline_cost_log.json")


class CostTracker:
    def __init__(self, word_directory: str, pricing_updated: bool = True, pricing_warning: str = None):
        self.word_directory = word_directory
        self.pricing_updated = pricing_updated
        self.pricing_warning = pricing_warning
        self.steps = []
        self.total_cost = 0.0
        self._run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initialize the run entry in the log immediately
        self._write_to_log()

    def add(self, step: str, cost: float, detail: str = ""):
        """Record a cost and write to log immediately."""
        self.steps.append({
            "step": step,
            "cost": cost,
            "detail": detail,
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        self.total_cost += cost
        print(f"  💰 {step}: ${cost:.4f}{' — ' + detail if detail else ''}")
        self._write_to_log()

    def _write_to_log(self):
        """Write current state to the cost log file immediately."""
        # Load existing log or start fresh
        if LOG_FILE.exists():
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except Exception:
                log_data = {"runs": []}
        else:
            log_data = {"runs": []}

        # Find or create this run's entry
        run_entry = None
        for run in log_data["runs"]:
            if run.get("run_timestamp") == self._run_timestamp and run.get("word_directory") == self.word_directory:
                run_entry = run
                break

        if run_entry is None:
            run_entry = {
                "run_timestamp": self._run_timestamp,
                "word_directory": self.word_directory,
                "steps": [],
                "total_cost": 0.0,
                "pricing_updated": self.pricing_updated,
                "pricing_warning": self.pricing_warning,
                "status": "in_progress",
            }
            log_data["runs"].append(run_entry)

        # Update the entry
        run_entry["steps"] = self.steps
        run_entry["total_cost"] = round(self.total_cost, 6)

        # Recalculate cumulative total
        log_data["cumulative_total"] = round(
            sum(r.get("total_cost", 0) for r in log_data["runs"]), 6
        )

        safe_write_local(lambda: open(LOG_FILE, "w", encoding="utf-8").write(__import__("json").dumps(log_data, indent=2)), str(LOG_FILE))

    def finalize(self, status: str = "complete"):
        """Mark the run as complete or failed in the log."""
        if LOG_FILE.exists():
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
                for run in log_data["runs"]:
                    if run.get("run_timestamp") == self._run_timestamp and run.get("word_directory") == self.word_directory:
                        run["status"] = status
                        run["ended"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        break
                safe_write_local(lambda: open(LOG_FILE, "w", encoding="utf-8").write(__import__("json").dumps(log_data, indent=2)), str(LOG_FILE))
            except Exception:
                pass

    def print_total(self):
        """Print the full cost summary."""
        print(f"\n--- Cost Summary ---")
        for step in self.steps:
            print(f"  {step['step']}: ${step['cost']:.4f}{' — ' + step['detail'] if step['detail'] else ''}")
        print(f"  {'─' * 35}")
        print(f"  Total: ${self.total_cost:.4f}")
        if not self.pricing_updated:
            print(f"\n  ⚠️  PRICING WARNING: {self.pricing_warning}")

    def save_to_log(self):
        """Finalize the log entry and print cumulative total."""
        self.finalize(status="complete")
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            print(f"\n  Cost log updated: {LOG_FILE}")
            print(f"  Cumulative total across all runs: ${log_data.get('cumulative_total', 0):.4f}")
