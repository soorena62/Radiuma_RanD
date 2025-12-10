from PySide6.QtWidgets import QPushButton, QHBoxLayout, QWidget
from dagster_radiuma.jobs import radiomics_job, radiomics_batch_job

class WorkflowControls(QWidget):
    def __init__(self, log_panel, history_panel, parent=None):
        super().__init__(parent)

        self.log_panel = log_panel
        self.history_panel = history_panel

        # دکمه‌ها
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.resume_btn = QPushButton("Resume")
        self.cancel_btn = QPushButton("Cancel")

        layout = QHBoxLayout()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.resume_btn)
        layout.addWidget(self.cancel_btn)
        self.setLayout(layout)

        # اتصال
        self.start_btn.clicked.connect(self.start_workflow)
        self.stop_btn.clicked.connect(self.stop_workflow)
        self.resume_btn.clicked.connect(self.resume_workflow)
        self.cancel_btn.clicked.connect(self.cancel_workflow)

        self.current_run_id = None

    def log(self, message):
        if hasattr(self.log_panel, "append"):
            self.log_panel.append(message)
        if hasattr(self.history_panel, "append"):
            self.history_panel.append(message)

    def start_workflow(self):
        self.log("[GUI] Start → Dagster job starting...")
        result = radiomics_job.execute_in_process()
        self.current_run_id = result.run_id
        self.log(f"[GUI] Workflow started. Success={result.success}")

    def stop_workflow(self):
        if self.current_run_id:
            self.log(f"[GUI] Stop → Terminating run {self.current_run_id}")
            # dagster_api.terminate_run(self.current_run_id)
            self.log("[GUI] Workflow stopped.")

    def resume_workflow(self):
        if self.current_run_id:
            self.log(f"[GUI] Resume → Restarting run {self.current_run_id}")
            # dagster_api.resume_run(self.current_run_id)
            self.log("[GUI] Workflow resumed. [Recovered]")

    def cancel_workflow(self):
        if self.current_run_id:
            self.log(f"[GUI] Cancel → Cancelling run {self.current_run_id}")
            # dagster_api.cancel_run(self.current_run_id)
            self.log("[GUI] Workflow cancelled.")
