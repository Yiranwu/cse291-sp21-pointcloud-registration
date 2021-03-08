import numpy as np
from PIL import Image
from utils.file_utils import get_data_files

def process(image_data_dir):
    rgb_files, depth_files, label_files, meta_files = get_data_files(image_data_dir,
                                                                     target_levels=(1, 2))

    for i, object_id in enumerate(meta["object_ids"]):
        name = Path(meta_files[file_idx]).name[:-9] + f"_{i}.npz"

        mask = label == object_id
        points_object = points_viewer[mask]
        colors_object = rgb[mask]