import json
import threading
from pathlib import Path
from loguru import logger

class WorkflowState:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_file = state_dir / "state.json"
        self._lock = threading.Lock()
        self._state = {"steps": {}, "cancelled": False}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            self._state = json.loads(self.state_file.read_text())

    def mark_success(self, step_id: str, outputs: dict):
        with self._lock:
            self._state["steps"][step_id] = {"status": "success", "outputs": outputs}
            self._write()

    def mark_failed(self, step_id: str, error: str):
        with self._lock:
            self._state["steps"][step_id] = {"status": "failed", "error": error}
            self._write()

    def get_status(self, step_id: str):
        return self._state["steps"].get(step_id, {}).get("status", "pending")

    def get_outputs(self, step_id: str):
        return self._state["steps"].get(step_id, {}).get("outputs", {})

    def set_cancelled(self, flag: bool):
        with self._lock:
            self._state["cancelled"] = flag
            self._write()

    def is_cancelled(self) -> bool:
        return self._state.get("cancelled", False)

    def _write(self):
        self.state_file.write_text(json.dumps(self._state, indent=2))

class Node:
    def __init__(self, step_id: str, run_fn, inputs: list, outputs: list):
        self.step_id = step_id
        self.run_fn = run_fn
        self.inputs = inputs
        self.outputs = outputs

class Workflow:
    def __init__(self, nodes: list[Node], state: WorkflowState):
        self.nodes = nodes
        self.state = state

    def execute(self):
        for node in self.nodes:
            status = self.state.get_status(node.step_id)
            if status == "success":
                logger.info(f"Skip {node.step_id}: already successful.")
                continue
            if self.state.is_cancelled():
                logger.warning("Execution cancelled. Stopping workflow.")
                break
            try:
                logger.info(f"Run {node.step_id}")
                outputs = node.run_fn(self.state)
                self.state.mark_success(node.step_id, outputs)
                logger.info(f"Success {node.step_id}")
            except Exception as e:
                logger.exception(f"Failed {node.step_id}: {e}")
                self.state.mark_failed(node.step_id, str(e))
                break  # Stop on failure, allow resume later
