from PySide6.QtWidgets import (
    QMessageBox, QGraphicsRectItem, QGraphicsTextItem,
    QPushButton, QGraphicsProxyWidget, QTableWidget,
    QTableWidgetItem, QDialog, QVBoxLayout
)
from PySide6.QtGui import QBrush, QColor
from pathlib import Path
import napari
import json
from .port_item import PortItem  # Import PortItem with type support

class NodeWidget(QGraphicsRectItem):
    def __init__(self, title: str, inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]):
        """
        inputs/outputs: list of (label, port_type)
        Example:
            inputs=[("Image", "Image"), ("Mask", "Mask")]
            outputs=[("Features", "Features")]
        """
        super().__init__(0, 0, 160, 90)
        self.setBrush(QBrush(QColor("#e0e0e0")))
        self.title = QGraphicsTextItem(title, self)
        self.title.setPos(10, 5)

        self.status_text = QGraphicsTextItem("", self)
        self.status_text.setPos(10, 40)

        # Create input ports
        self.input_ports = []
        y_offset = 20
        for label, port_type in inputs:
            port = PortItem(label, port_type, is_input=True, parent=self)
            port.setPos(0, y_offset)
            self.input_ports.append(port)
            y_offset += 20

        # Create output ports
        self.output_ports = []
        y_offset = 20
        for label, port_type in outputs:
            port = PortItem(label, port_type, is_input=False, parent=self)
            port.setPos(150, y_offset)
            self.output_ports.append(port)
            y_offset += 20

        # Output Display Button
        self.btn_view = QPushButton("View Output")
        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(self.btn_view)
        proxy.setPos(10, 60)
        self.btn_view.clicked.connect(self.view_output)

    def set_status(self, status: str):
        # Change node color and status text based on execution state
        if status == "running":
            self.setBrush(QBrush(QColor("yellow")))
            self.status_text.setPlainText("⏳ Running")
        elif status == "success":
            self.setBrush(QBrush(QColor("lightgreen")))
            self.status_text.setPlainText("✔ Success")
        elif status == "failed":
            self.setBrush(QBrush(QColor("red")))
            self.status_text.setPlainText("❌ Failed")
        else:
            self.setBrush(QBrush(QColor("#e0e0e0")))
            self.status_text.setPlainText("")

    def view_output(self):
        node_type = self.title.toPlainText()

        # Image Reader Or Image Filter → Processed Image
        if node_type in ["Image Reader", "Image Filter"]:
            file_path = Path("artifacts/processed_image.nii.gz")
            if file_path.exists():
                viewer = napari.Viewer()
                viewer.open(str(file_path))
            else:
                QMessageBox.warning(None, "Output not found", f"No output image for {node_type}")

        # Mask Reader → Mask
        elif node_type == "Mask Reader":
            file_path = Path("artifacts/mask.nii.gz")
            if file_path.exists():
                viewer = napari.Viewer()
                viewer.open(str(file_path))
            else:
                QMessageBox.warning(None, "Output not found", "No mask file found")

        # Image Registration → Registered image
        elif node_type == "Image Registration":
            file_path = Path("artifacts/registered_image.nii.gz")
            if file_path.exists():
                viewer = napari.Viewer()
                viewer.open(str(file_path))
            else:
                QMessageBox.warning(None, "Output not found", "No registered image found")

        # Image Fusion → Composite Image
        elif node_type == "Image Fusion":
            file_path = Path("artifacts/fused_image.nii.gz")
            if file_path.exists():
                viewer = napari.Viewer()
                viewer.open(str(file_path))
            else:
                QMessageBox.warning(None, "Output not found", "No fused image found")

        # Image Extraction Or PySERA Extract → Features table
        elif node_type in ["Image Extraction", "PySERA Extract"]:
            file_path = Path("artifacts/features.json")
            if file_path.exists():
                features = json.loads(file_path.read_text())
                dialog = QDialog()
                dialog.setWindowTitle("Radiomics Features")
                layout = QVBoxLayout(dialog)
                table = QTableWidget()
                table.setRowCount(len(features))
                table.setColumnCount(2)
                table.setHorizontalHeaderLabels(["Feature", "Value"])
                for i, (key, value) in enumerate(features.items()):
                    table.setItem(i, 0, QTableWidgetItem(key))
                    table.setItem(i, 1, QTableWidgetItem(str(value)))
                layout.addWidget(table)
                dialog.exec()
            else:
                QMessageBox.warning(None, "Output not found", "No features.json found")

        # Writer → Final Output File
        elif node_type in ["Writer", "Image Writer"]:
            file_path = Path("artifacts/output.nii.gz")
            if file_path.exists():
                QMessageBox.information(None, "Writer Output", f"Output file saved: {file_path}")
            else:
                QMessageBox.warning(None, "Output not found", "No output file found")
