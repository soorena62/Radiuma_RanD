from PySide6.QtWidgets import QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtCore import QPointF
from app.gui.node_widget import NodeWidget


class Port(QGraphicsEllipseItem):
    """An input/output port that is placed next to each node."""
    def __init__(self, parent_node: NodeWidget, kind: str, offset_x: float, offset_y: float):
        super().__init__(-5, -5, 10, 10, parent_node)
        self.parent_node = parent_node
        self.kind = kind  # "input" Or "output"
        self.setBrush(QBrush(QColor("#2e7d32") if kind == "input" else QColor("#1565c0")))
        self.setPos(parent_node.rect().x() + offset_x, parent_node.rect().y() + offset_y)
        self.setZValue(1.0)


class ConnectionLine(QGraphicsLineItem):
    """Connection between two ports."""
    def __init__(self, source_port: Port, target_port: Port):
        super().__init__()
        self.source_port = source_port
        self.target_port = target_port
        pen = QPen(QColor("#555"))
        pen.setWidth(2)
        self.setPen(pen)
        self.update_positions()

    def update_positions(self):
        src = self.source_port.mapToScene(QPointF(0, 0))
        tgt = self.target_port.mapToScene(QPointF(0, 0))
        self.setLine(src.x(), src.y(), tgt.x(), tgt.y())


class GraphScene(QGraphicsScene):
    """Graph scene with nodes, ports, and connections. Only connected nodes are exported."""
    def __init__(self):
        super().__init__()
        self.connections: list[ConnectionLine] = []
        # Manual connection mode: You click on an outgoing port, then click on the destination incoming port.
        self._pending_source_port: Port | None = None

    # ---------- Helper tool for adding default nodes ----------
    def add_default_medical_nodes(self):
        """Adds five medical nodes and installs their ports."""
        # Create nodes
        reader = NodeWidget("Image Reader", [], ["output"])
        registration = NodeWidget("Image Registration", ["input"], ["output"])
        fusion = NodeWidget("Image Fusion", ["input1", "input2"], ["output"])
        extraction = NodeWidget("Image Extraction", ["input"], ["output"])
        writer = NodeWidget("Image Writer", ["input"], [])

        # Initial placement
        reader.setPos(50, 50)
        registration.setPos(250, 50)
        fusion.setPos(450, 50)
        extraction.setPos(650, 50)
        writer.setPos(850, 50)

        # Add to scene
        for n in (reader, registration, fusion, extraction, writer):
            self.addItem(n)

        # Install ports for each node (a simple I/O is enough)
        # Reader: Output only
        reader.output_port = Port(reader, "output", 150, 45)

        # Registration: Input + Output
        registration.input_port = Port(registration, "input", 0, 45)
        registration.output_port = Port(registration, "output", 150, 45)

        # Fusion: Two inputs + one output
        fusion.input_port_1 = Port(fusion, "input", 0, 30)
        fusion.input_port_2 = Port(fusion, "input", 0, 60)
        fusion.output_port = Port(fusion, "output", 150, 45)

        # Extraction: Input + Output
        extraction.input_port = Port(extraction, "input", 0, 45)
        extraction.output_port = Port(extraction, "output", 150, 45)

        # Writer: Input only
        writer.input_port = Port(writer, "input", 0, 45)

        # Register the port click handle for manual connection
        self._install_port_handlers([reader, registration, fusion, extraction, writer])

    def _install_port_handlers(self, nodes: list[NodeWidget]):
        """Mouse handles for ports so the user can make connections."""
        all_ports: list[Port] = []
        for n in nodes:
            for attr in dir(n):
                if attr.endswith("port"):
                    p = getattr(n, attr)
                    if isinstance(p, Port):
                        all_ports.append(p)

        for port in all_ports:
            port.mousePressEvent = lambda event, p=port: self._on_port_clicked(p)

    def _on_port_clicked(self, port: Port):
        """Ports click logic: output first, then input; connection is made."""
        if port.kind == "output":
            # Start connection
            self._pending_source_port = port
        elif port.kind == "input" and self._pending_source_port is not None:
            # Connection completion
            line = ConnectionLine(self._pending_source_port, port)
            self.addItem(line)
            self.connections.append(line)
            # Update positions when nodes move
            self._pending_source_port = None

    # ----------Export/Load Graph ----------
    def export_graph(self):
        """Exports only nodes involved in connections."""
        data = {"nodes": [], "connections": []}
        used_nodes = set()

        for conn in self.connections:
            src_node = conn.source_port.parent_node
            tgt_node = conn.target_port.parent_node
            used_nodes.add(src_node)
            used_nodes.add(tgt_node)
            data["connections"].append({
                "source": src_node.title.toPlainText(),
                "target": tgt_node.title.toPlainText()
            })

        for node in used_nodes:
            data["nodes"].append({
                "id": node.title.toPlainText(),  # Persistent ID based on title
                "type": node.title.toPlainText(),
                "pos": [node.pos().x(), node.pos().y()]
            })

        return data

    def save_graph(self, filename: str):
        import json
        from pathlib import Path
        Path(filename).write_text(json.dumps(self.export_graph(), indent=2))

    def load_graph(self, filename: str):
        """Rebuild ports and connections from saved file.
        Note: Here we assume that the default nodes are present on the scene; we only draw the connections.
        """
        import json
        from pathlib import Path
        if not Path(filename).exists():
            return
        data = json.loads(Path(filename).read_text())

        # Create a mapping from title → node in the scene
        title_to_node = {}
        for item in self.items():
            if isinstance(item, NodeWidget):
                title_to_node[item.title.toPlainText()] = item

        # Rebuild connections
        self._rebuild_ports_if_missing(title_to_node)
        for conn in data.get("connections", []):
            src = title_to_node.get(conn["source"])
            tgt = title_to_node.get(conn["target"])
            if not src or not tgt:
                continue

           # Select default input/output port based on node name
            src_port = getattr(src, "output_port", None)
            tgt_port = getattr(tgt, "input_port", None)

           # Fusion has two inputs; if the destination is Fusion and the first input is busy, take the second input
            if tgt.title.toPlainText() == "Image Fusion":
                # If there is no previous connection to input_port_1, connect to it; otherwise connect to input_port_2
                candidates = [tgt.input_port_1, tgt.input_port_2]
                tgt_port = candidates[0]
                for c in self.connections:
                    if c.target_port is candidates[0]:
                        tgt_port = candidates[1]
                        break

            if src_port and tgt_port:
                line = ConnectionLine(src_port, tgt_port)
                self.addItem(line)
                self.connections.append(line)

    def _rebuild_ports_if_missing(self, title_to_node: dict):
        """If the node ports haven't been created yet for some reason, we'll create them here."""
        for title, n in title_to_node.items():
            # Reader
            if title == "Image Reader" and not hasattr(n, "output_port"):
                n.output_port = Port(n, "output", 150, 45)
            # Registration
            if title == "Image Registration":
                if not hasattr(n, "input_port"):
                    n.input_port = Port(n, "input", 0, 45)
                if not hasattr(n, "output_port"):
                    n.output_port = Port(n, "output", 150, 45)
            # Fusion
            if title == "Image Fusion":
                if not hasattr(n, "input_port_1"):
                    n.input_port_1 = Port(n, "input", 0, 30)
                if not hasattr(n, "input_port_2"):
                    n.input_port_2 = Port(n, "input", 0, 60)
                if not hasattr(n, "output_port"):
                    n.output_port = Port(n, "output", 150, 45)
            # Extraction
            if title == "Image Extraction":
                if not hasattr(n, "input_port"):
                    n.input_port = Port(n, "input", 0, 45)
                if not hasattr(n, "output_port"):
                    n.output_port = Port(n, "output", 150, 45)
            # Writer
            if title == "Image Writer" and not hasattr(n, "input_port"):
                n.input_port = Port(n, "input", 0, 45)