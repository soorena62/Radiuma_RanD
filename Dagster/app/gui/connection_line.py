from PySide6.QtWidgets import QGraphicsLineItem, QMessageBox
from PySide6.QtGui import QPen, QColor

class ConnectionLine(QGraphicsLineItem):
    def __init__(self, source_port, target_port):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        self.update_position()

        # Validate connection types
        if self.source_port.port_type != self.target_port.port_type:
            # Invalid connection → red line + warning
            self.setPen(QPen(QColor("red"), 2))
            QMessageBox.warning(None, "Invalid Connection",
                                f"Cannot connect {self.source_port.port_type} → {self.target_port.port_type}")
        else:
            # Valid connection → black line
            self.setPen(QPen(QColor("blue"), 2))

    def update_position(self):
        src_pos = self.source_port.scenePos()
        tgt_pos = self.target_port.scenePos()
        self.setLine(src_pos.x(), src_pos.y(), tgt_pos.x(), tgt_pos.y())
