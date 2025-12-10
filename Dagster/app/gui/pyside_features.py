import sys
import json
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem

class FeatureViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiomics Features Viewer")
        layout = QVBoxLayout(self)

        self.status = QLabel("Load features.json to view results")
        self.table = QTableWidget()
        self.btn_load = QPushButton("Load Features")

        layout.addWidget(self.status)
        layout.addWidget(self.table)
        layout.addWidget(self.btn_load)

        self.btn_load.clicked.connect(self.load_features)

    def load_features(self):
        features_path = Path("artifacts/features.json")
        if not features_path.exists():
            self.status.setText("No features.json found! Run workflow first.")
            return

        try:
            features = json.loads(features_path.read_text())
        except Exception as e:
            self.status.setText(f"Error reading features.json: {e}")
            return

        # Displaying features in a table
        self.table.setRowCount(len(features))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Feature", "Value"])

        for i, (key, value) in enumerate(features.items()):
            self.table.setItem(i, 0, QTableWidgetItem(key))
            self.table.setItem(i, 1, QTableWidgetItem(str(value)))

        self.status.setText("Features loaded successfully!")

def run_gui():
    app = QApplication(sys.argv)
    viewer = FeatureViewer()
    viewer.resize(500, 400)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
