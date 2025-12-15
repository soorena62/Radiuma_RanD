import sys
import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QTextEdit, QPushButton, QWidget, QVBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor

from NodeGraphQt import NodeGraph
from NodeGraphQt.errors import NodeRegistrationError

from dagster import DagsterInstance, execute_job
import yaml
from dagster_radiuma.jobs import radiomics_job, radiomics_batch_job

from app.gui.node_catalog import (
    NodeCatalog,
    ImageReaderNode,
    ImageWriterNode,
    ImageRegistrationNode,
    ImageFilterNode,
    ImageFusionNode,
    FeatureExtractionNode
)

from app.gui.gui_controls import WorkflowControls


class VisualEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiuma Workflow Editor")
        self.resize(1200, 800)

        # Graph
        self.graph = NodeGraph()

        # Register nodes with unique alias: identifier + ClassName
        def reg(cls):
            key = f"{cls.__identifier__}.{cls.__name__}"
            try:
                self.graph.register_node(cls, alias=key)
            except NodeRegistrationError:
                pass

        reg(ImageReaderNode)
        reg(ImageWriterNode)
        reg(ImageRegistrationNode)
        reg(ImageFilterNode)
        reg(ImageFusionNode)
        reg(FeatureExtractionNode)

        # Viewer
        self.viewer = self.graph.widget

        # Node catalog
        self.catalog = NodeCatalog(self.graph)

        # Log panel
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("Logs will appear here...")

        # Workflow controls (Start/Stop/Resume/Cancel)
        self.controls = WorkflowControls(self.log_panel, self.log_panel)
        # اتصال متدهای کنترل به Dagster
        self.controls.start_btn.clicked.connect(self.start_workflow)
        self.controls.stop_btn.clicked.connect(self.stop_workflow)
        self.controls.resume_btn.clicked.connect(self.resume_workflow)
        self.controls.cancel_btn.clicked.connect(self.cancel_workflow)

        # Left panel layout
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.catalog)
        left_layout.addWidget(self.controls)

        # Splitters
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(left_panel)
        top_splitter.addWidget(self.viewer)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.log_panel)
        main_splitter.setSizes([600, 200])

        self.setCentralWidget(main_splitter)

        # Dagster instance
        self.instance = DagsterInstance.ephemeral()
        self.current_run_id = None

        # Initial log
        self.log("Radiuma Workflow Editor started.", "INFO")
        self.log("Custom nodes registered with alias: identifier + ClassName.", "INFO")

    def log(self, message, level="INFO"):
        fmt = QTextCharFormat()
        if level == "INFO":
            fmt.setForeground(QColor("green"))
        elif level == "WARNING":
            fmt.setForeground(QColor("orange"))
        elif level == "ERROR":
            fmt.setForeground(QColor("red"))
        else:
            fmt.setForeground(QColor("black"))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.log_panel.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"[{timestamp}] [{level}] {message}\n", fmt)
        self.log_panel.setTextCursor(cursor)

    # --- کنترل اجرای Dagster ---
    import yaml

    def start_workflow(self):
        # Log start of workflow from GUI
        self.log("[GUI] Start → Dagster job starting...", "INFO")

        try:
            # Load configuration from configs.yaml
            with open("configs.yaml", "r", encoding="utf-8") as f:
                run_config = yaml.safe_load(f)

            # Check which job type is defined in the config
            if "radiomics_batch_job" in run_config:
                # Batch mode → run multiple cases automatically
                self.log("[GUI] Batch mode detected → Running Batch Workflow", "INFO")
                result = radiomics_batch_job.execute_in_process(
                    run_config=run_config["radiomics_batch_job"],
                    instance=self.instance
                )
            elif "radiomics_job" in run_config:
                # Single case mode → run one case only
                self.log("[GUI] Single case detected → Running Single Workflow", "INFO")
                result = radiomics_job.execute_in_process(
                    run_config=run_config["radiomics_job"],
                    instance=self.instance
                )
            else:
                # Config file does not contain required job section
                self.log("[GUI] Config file missing required job section.", "ERROR")
                return

            # Save run_id and log success/failure
            self.current_run_id = result.run_id
            self.log(f"[GUI] Workflow finished. run_id={self.current_run_id} Success={result.success}", "INFO")

        except Exception as e:
            # Log any unexpected errors
            self.log(f"[GUI] Workflow error: {e}", "ERROR")

    def stop_workflow(self):
        if self.current_run_id:
            self.instance.cancel_run(self.current_run_id)
            self.log(f"[GUI] Workflow stopped. run_id={self.current_run_id}", "WARNING")

    def resume_workflow(self):
        if self.current_run_id:
            self.instance.resume_run(self.current_run_id)
            self.log(f"[GUI] Workflow resumed. run_id={self.current_run_id}", "INFO")

    def cancel_workflow(self):
        if self.current_run_id:
            self.instance.cancel_run(self.current_run_id)
            self.log(f"[GUI] Workflow cancelled. run_id={self.current_run_id}", "ERROR")


def main():
    app = QApplication(sys.argv)
    window = VisualEditor()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
