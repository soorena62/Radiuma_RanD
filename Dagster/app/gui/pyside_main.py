import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from app.engine.dagster_runner import run_workflow_with_config
# Write Your Code Here:

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiuma Mini - Workflow Runner")
        layout = QVBoxLayout(self)

        self.status = QLabel("Idle")
        self.btn_select_image = QPushButton("Select Image File")
        self.btn_select_mask = QPushButton("Select Mask File")
        self.btn_run = QPushButton("Run Workflow")

        layout.addWidget(self.status)
        layout.addWidget(self.btn_select_image)
        layout.addWidget(self.btn_select_mask)
        layout.addWidget(self.btn_run)

        self.image_file = None
        self.mask_file = None

        self.btn_select_image.clicked.connect(self.select_image)
        self.btn_select_mask.clicked.connect(self.select_mask)
        self.btn_run.clicked.connect(self.run_workflow)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "data/images", "NIfTI files (*.nii.gz)")
        if file_path:
            self.image_file = Path(file_path).name
            self.status.setText(f"Selected image: {self.image_file}")

    def select_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Mask", "data/masks", "NIfTI files (*.nii.gz)")
        if file_path:
            self.mask_file = Path(file_path).name
            self.status.setText(f"Selected mask: {self.mask_file}")

    def run_workflow(self):
        if not self.image_file or not self.mask_file:
            self.status.setText("Please select both image and mask files!")
            return
        self.status.setText("Running workflow...")
        result = run_workflow_with_config(self.image_file, self.mask_file)
        if result.success:
            self.status.setText("Workflow completed successfully!")
        else:
            self.status.setText("Workflow failed!")

def run_gui():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(400, 200)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
