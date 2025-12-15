from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
from PySide6.QtGui import QBrush, QColor

class PortItem(QGraphicsEllipseItem):
    def __init__(self, label: str, port_type: str, is_input: bool, parent=None):
        super().__init__(-5, -5, 10, 10, parent)
        # Color based on input/output
        self.setBrush(QBrush(QColor("green") if is_input else QColor("blue")))
        self.label = QGraphicsTextItem(label, parent)
        self.label.setDefaultTextColor(QColor("black"))
        self.label.setPos(10 if is_input else -50, -7)
        self.is_input = is_input
        # Port type (e.g. "Image", "Mask", "Features")
        self.port_type = port_type
