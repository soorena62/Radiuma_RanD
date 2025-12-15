import SimpleITK as sitk
import numpy as np
import pandas as pd
import pysera
from NodeGraphQt import BaseNode
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QTreeWidget, QTreeWidgetItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon


# -------------------------------
# Helpers for safe input resolution
# -------------------------------

def _resolve_array_from_input(node: BaseNode, port_index: int, upstream_attr_name: str):
    """
    Try to get a NumPy array from node.get_input(port_index).
    If it's a Port (NodeGraphQt Port), traverse to the connected upstream node
    and read the given upstream_attr_name as a fallback payload.
    """
    val = node.get_input(port_index)
    # direct array
    if isinstance(val, np.ndarray):
        return val
    # try to traverse port connections
    try:
        if hasattr(val, 'connected_ports'):
            ports = val.connected_ports()
            if ports:
                upstream_node = ports[0].node()
                upstream_val = getattr(upstream_node, upstream_attr_name, None)
                if isinstance(upstream_val, np.ndarray):
                    return upstream_val
    except Exception:
        pass
    return None


def _resolve_features_from_input(node: BaseNode, port_index: int, upstream_attr_name: str):
    """
    Resolve features payload from input. Accepts pandas.DataFrame, dict, list, or str.
    If input returns a Port, read upstream node attribute.
    """
    val = node.get_input(port_index)
    # direct usable types
    if hasattr(val, "to_csv"):  # pandas DataFrame
        return val
    if isinstance(val, (dict, list, str)):
        return val
    # fallback via upstream
    try:
        if hasattr(val, 'connected_ports'):
            ports = val.connected_ports()
            if ports:
                upstream_node = ports[0].node()
                upstream_val = getattr(upstream_node, upstream_attr_name, None)
                if hasattr(upstream_val, "to_csv") or isinstance(upstream_val, (dict, list, str)):
                    return upstream_val
    except Exception:
        pass
    return None


# -------------------------------
# Node Definitions
# -------------------------------

class ImageReaderNode(BaseNode):
    __identifier__ = 'image.io'
    NODE_NAME = 'Image Reader'

    def __init__(self):
        super().__init__()
        self.add_output('image')
        self.add_output('mask')
        # store payloads for upstream fallback resolution
        self._image_array = None
        self._mask_array = None

    def run(self):
        try:
            image_path = r"C:/Users/Omen16/Documents/Radiuma_Mini/data/images/CT_AVM.nii.gz"
            mask_path  = r"C:/Users/Omen16/Documents/Radiuma_Mini/data/masks/CT_AVM_mask.nii.gz"

            image = sitk.ReadImage(image_path)
            mask  = sitk.ReadImage(mask_path)

            img_arr = sitk.GetArrayFromImage(image)
            msk_arr = sitk.GetArrayFromImage(mask)

            # save locally and on ports
            self._image_array = img_arr
            self._mask_array = msk_arr

            print("Reader: loaded image & mask.")
            self.set_output(0, img_arr)
            self.set_output(1, msk_arr)
        except Exception as e:
            print("Reader error:", e)


class ImageWriterNode(BaseNode):
    __identifier__ = 'image.io'
    NODE_NAME = 'Image Writer'

    def __init__(self):
        super().__init__()
        self.add_input('image')
        self.add_input('features')

    def run(self):
        # resolve image array robustly (if input is a Port, traverse upstream)
        image_array = _resolve_array_from_input(self, 0, '_image_array')
        features = _resolve_features_from_input(self, 1, '_features')

        # image save
        if isinstance(image_array, np.ndarray) and image_array.size > 0:
            try:
                sitk_image = sitk.GetImageFromArray(image_array)
                sitk.WriteImage(sitk_image, "output_image.nii.gz")
                print("Writer: image saved.")
            except Exception as e:
                print("Writer image save error:", e)
        else:
            print("Writer: no image to save.")

        # features save
        if features is not None:
            try:
                if hasattr(features, "to_csv"):
                    features.to_csv("features.csv", index=False)
                elif isinstance(features, dict):
                    pd.DataFrame([features]).to_csv("features.csv", index=False)
                elif isinstance(features, list):
                    pd.DataFrame(features).to_csv("features.csv", index=False)
                else:
                    pd.DataFrame([{"features": str(features)}]).to_csv("features.csv", index=False)
                print("Writer: features saved.")
            except Exception as e:
                print("Writer features save error:", e)
        else:
            print("Writer: no features to save.")


class ImageFilterNode(BaseNode):
    __identifier__ = 'image.proc'
    NODE_NAME = 'Image Filter'

    def __init__(self):
        super().__init__()
        self.add_input('image')
        self.add_output('filtered_image')
        self._filtered_array = None

    def run(self):
        # resolve input image array robustly
        image_array = _resolve_array_from_input(self, 0, '_image_array')
        if not isinstance(image_array, np.ndarray) or image_array.size == 0:
            print("Filter: no valid image.")
            return
        try:
            image = sitk.GetImageFromArray(image_array)
            filtered = sitk.SmoothingRecursiveGaussian(image, sigma=2.0)
            filtered_arr = sitk.GetArrayFromImage(filtered)

            self._filtered_array = filtered_arr
            self.set_output(0, filtered_arr)
            print("Filter: applied Gaussian smoothing.")
        except Exception as e:
            print("Filter error:", e)


class ImageRegistrationNode(BaseNode):
    __identifier__ = 'image.proc'
    NODE_NAME = 'Image Registration'

    def __init__(self):
        super().__init__()
        self.add_input('fixed_image')
        self.add_input('moving_image')
        self.add_output('registered_image')
        self._registered_array = None

    def run(self):
        fixed_array = _resolve_array_from_input(self, 0, '_image_array')
        moving_array = _resolve_array_from_input(self, 1, '_image_array')

        if not isinstance(fixed_array, np.ndarray) or not isinstance(moving_array, np.ndarray):
            print("Registration: invalid inputs.")
            return

        try:
            fixed = sitk.GetImageFromArray(fixed_array)
            moving = sitk.GetImageFromArray(moving_array)

            registration = sitk.ImageRegistrationMethod()
            registration.SetMetricAsMeanSquares()
            registration.SetOptimizerAsGradientDescent(
                learningRate=1.0,
                numberOfIterations=50
            )
            registration.SetInterpolator(sitk.sitkLinear)

            initial_transform = sitk.CenteredTransformInitializer(
                fixed,
                moving,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration.SetInitialTransform(initial_transform, inPlace=False)

            final_transform = registration.Execute(fixed, moving)
            registered = sitk.Resample(
                moving, fixed, final_transform,
                sitk.sitkLinear, 0.0, moving.GetPixelID()
            )

            reg_arr = sitk.GetArrayFromImage(registered)
            self._registered_array = reg_arr
            self.set_output(0, reg_arr)
            print("Registration: complete.")
        except Exception as e:
            print("Registration error:", e)


class ImageFusionNode(BaseNode):
    __identifier__ = 'image.proc'
    NODE_NAME = 'Image Fusion'

    def __init__(self):
        super().__init__()
        self.add_input('image_A')
        self.add_input('image_B')
        self.add_output('fused_image')
        self._fused_array = None

    def run(self):
        img_a = _resolve_array_from_input(self, 0, '_image_array')
        img_b = _resolve_array_from_input(self, 1, '_image_array')

        if not isinstance(img_a, np.ndarray) or not isinstance(img_b, np.ndarray):
            print("Fusion: invalid inputs.")
            return

        try:
            fused = (img_a.astype(np.float32) * 0.5 + img_b.astype(np.float32) * 0.5)
            self._fused_array = fused
            self.set_output(0, fused)
            print("Fusion: complete.")
        except Exception as e:
            print("Fusion error:", e)


class FeatureExtractionNode(BaseNode):
    __identifier__ = 'analysis'
    NODE_NAME = 'Feature Extraction'

    def __init__(self):
        super().__init__()
        self.add_input('image')
        self.add_input('mask')
        self.add_output('features')
        self._features = None

    def run(self):
        image_array = _resolve_array_from_input(self, 0, '_image_array')
        mask_array  = _resolve_array_from_input(self, 1, '_mask_array')

        if not isinstance(image_array, np.ndarray) or not isinstance(mask_array, np.ndarray):
            print("Extraction: invalid image/mask.")
            return

        try:
            result = pysera.process_batch(
                image_input=image_array,
                mask_input=mask_array,
                output_path="./results",
                categories="all",
                dimensions="3d",
                apply_preprocessing=True
            )
            features = result.get("features_extracted")
            self._features = features
            self.set_output(0, features)

            # End of processing report
            print(f"Extraction finished. Success={result.get('success')}, "
                  f"Features={len(features) if features is not None else 0}, "
                  f"Time={result.get('processing_time')}s")

        except Exception as e:
            print("PySERA error:", e)


# -------------------------------
# Node Catalog Widget
# -------------------------------

class NodeCatalog(QWidget):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.node_count = 0

        layout = QVBoxLayout(self)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search nodes...")
        layout.addWidget(self.search_bar)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        layout.addWidget(self.tree)

        self.categories = {
            "Image I/O": [
                (ImageReaderNode, "icons/camera.png"),
                (ImageWriterNode, "icons/save.png")
            ],
            "Image Processing": [
                (ImageFilterNode, "icons/filter.png"),
                (ImageRegistrationNode, "icons/transform.png"),
                (ImageFusionNode, "icons/fusion.png")
            ],
            "Feature Extraction": [
                (FeatureExtractionNode, "icons/chart.png")
            ]
        }

        self.populate_tree()
        self.search_bar.textChanged.connect(self.filter_nodes)
        self.tree.itemDoubleClicked.connect(self.add_node)

    def populate_tree(self):
        self.tree.clear()
        for category, nodes in self.categories.items():
            cat_item = QTreeWidgetItem([category])
            self.tree.addTopLevelItem(cat_item)
            for node_cls, icon_path in nodes:
                node_item = QTreeWidgetItem([node_cls.NODE_NAME])
                node_item.setData(0, Qt.UserRole, node_cls)
                node_item.setIcon(0, QIcon(icon_path))
                cat_item.addChild(node_item)

    def filter_nodes(self, text):
        text = text.lower()
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            visible_cat = False
            for j in range(cat_item.childCount()):
                node_item = cat_item.child(j)
                node_name = node_item.text(0).lower()
                is_match = text in node_name
                node_item.setHidden(not is_match)
                if is_match:
                    visible_cat = True
            cat_item.setHidden(not visible_cat)

    def add_node(self, item, column):
        node_cls = item.data(0, Qt.UserRole)
        if node_cls:
            node = node_cls()
            # grid placement to avoid overlap
            spacing_x, spacing_y = 200, 120
            x = spacing_x * (self.node_count % 4)
            y = spacing_y * (self.node_count // 4)
            self.graph.add_node(node, pos=(x, y))
            self.node_count += 1
