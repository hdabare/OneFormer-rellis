from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2

# import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_outback_dicts(img_dir, split_filename):
    split_file = os.path.join(img_dir, split_filename)
    with open(split_file) as f:
        data = f.readlines()

    dataset_dicts = []
    for idx, v in enumerate(data):
        record = {}
        
        img_filename = os.path.join(img_dir, "image", v.strip() + ".png")
        ann_filename = os.path.join(img_dir, "annotation", v.strip() + "_orig.png")
        height, width = cv2.imread(img_filename).shape[:2]
        
        record["file_name"] = img_filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["sem_seg_file_name"] = ann_filename
      
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("outback_" + d, lambda d=d: get_outback_dicts("/home/hdabare/GANav-offroad/data/outback/", d + ".txt"))
    MetadataCatalog.get("outback_" + d).set(stuff_classes=[
        "BACKGROUND", 
        "GRASS_TREE", 
        "POLE", 
        "TREE", 
        "LEAVES", 
        "FENCE", 
        "LOG", 
        "GRASS", 
        "ROAD_SIGN", 
        "SMALL", 
        "GRAVEL", 
        "GROUND", 
        "HORIZON", 
        "ROOTS", 
        "SKY", 
        "DELINEATOR", 
        "ROCK", 
        "ROAD"
        ],
            ignore_label=0,
            thing_dataset_id_to_contiguous_id={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17})

MetadataCatalog.get("outback_val").set(evaluator_type="sem_seg")
DatasetCatalog.register("outback_test", lambda d=d: get_outback_dicts("/home/hdabare/GANav-offroad/data/outback/", "test.txt"))
MetadataCatalog.get("outback_test").set(stuff_classes=[  "BACKGROUND", 
        "GRASS_TREE", 
        "POLE", 
        "TREE", 
        "LEAVES", 
        "FENCE", 
        "LOG", 
        "GRASS", 
        "ROAD_SIGN", 
        "SMALL", 
        "GRAVEL", 
        "GROUND", 
        "HORIZON", 
        "ROOTS", 
        "SKY", 
        "DELINEATOR", 
        "ROCK", 
        "ROAD"],
        ignore_label=0,
        thing_dataset_id_to_contiguous_id={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17})
MetadataCatalog.get("outback_test").set(evaluator_type="sem_seg")