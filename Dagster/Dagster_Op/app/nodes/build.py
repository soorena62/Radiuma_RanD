from app.engine.core import Node
from app.nodes.imaging import image_reader_fn, image_registration_fn, image_filter_fn, image_writer_fn

def build_nodes():
    return [
        Node("image_reader", image_reader_fn, inputs=[], outputs=["image_path"]),
        Node("image_registration", image_registration_fn, inputs=["image_path"], outputs=["registered_path"]),
        Node("image_filter", image_filter_fn, inputs=["registered_path"], outputs=["filtered_path"]),
        Node("image_writer", image_writer_fn, inputs=["filtered_path"], outputs=["final_output_path"]),
    ]





# from app.engine.core import Node
# from app.nodes.imaging import image_reader_fn, image_registration_fn, image_filter_fn, image_writer_fn
# from app.features.pysera_node import pysera_extract_fn

# def build_nodes():
#     return [
#         Node("image_reader", image_reader_fn, [], ["image_path"]),
#         Node("image_registration", image_registration_fn, ["image_path"], ["registered_path"]),
#         Node("image_filter", image_filter_fn, ["registered_path"], ["filtered_path"]),
#         Node("pysera_extract", pysera_extract_fn, ["filtered_path"], ["features_path"]),
#         Node("image_writer", image_writer_fn, ["filtered_path"], ["final_output_path"]),
#     ]