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

[[0, 0, 0, 0], #BACKGROUND 0 0
[140, 255, 25, 255], #GRASS_TREE 1 5
[140, 25, 255, 255], #POLE 2 5
[255, 197, 25, 255], #TREE 3 5
[25, 255, 82, 255], #LEAVES 4 1
[25, 82, 255, 255], #FENCE_NET 5 5
[255, 25, 197, 255], #LOG 6 5 
[255, 111, 25, 255], #GRASS 7 2
[226, 255, 25, 255], #ROAD_SIGN 8 5 
[54, 255, 25, 255], #SMALL_BRANCH 9 3
[0, 0, 0, 255], #GRASS 7 2
[25, 255, 168, 255], #GRAVEL 10 2
[25, 168, 255, 255], #GROUND 11 1
[54, 25, 255, 255], #HORIZON 12 0
[226, 25, 255, 255], # ROOTS 13 5
[255, 25, 111, 255], #SKY 14 0
[255, 68, 25, 255], #DELINEATOR 15 5
[255, 154, 25, 255] #ROCK 16 5
                 ]

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