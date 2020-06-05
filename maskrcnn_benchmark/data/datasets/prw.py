import os
import cv2
import torch.utils.data
from PIL import Image
import pandas as pd
import numpy as np
import json
import torch


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg

class PRWDataset(torch.utils.data.Dataset):

    CLASSES = ("__background__ ", 'person') ###

    def __init__(self, root, ann_file, split, transforms=None):

        self.gallery_size = cfg.TEST.GALLERY_SIZE
        self.root = root
        self.anno = ann_file
        self.split = split
        self.transforms = transforms

        self.anno_dir = 'data/prw/annotations'
        self.train_DF = 'data/prw/SIPN_annotation/trainAllDF.csv'
        self.test_DF = 'data/prw/SIPN_annotation/testAllDF.csv'
        self.query_DF = 'data/prw/SIPN_annotation/queryDF.csv'
        self.gallery = 'data/prw/SIPN_annotation/q_to_g{}DF.csv'.format(self.gallery_size)
        self.demo = False
        if self.demo:
            self.pid = 'pid_0.csv'
            self.pid_file = os.path.join(self.anno_dir, 'pids', self.pid)
            query_box = pd.read_csv(self.pid_file)
            imname = query_box['imname']
            self.ids = np.array(imname.squeeze()).tolist() # 's15533.jpg'
        else:
            with open(self.anno) as json_anno:
                anno_dict = json.load(json_anno)
            self.ids = [img['file_name'] for img in anno_dict['images']]

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PRWDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

        if self.split == 'train' or self.split == 'val':
            self.all_boxes = pd.read_csv(self.train_DF)
        if self.split == 'test':
            self.all_boxes = pd.read_csv(self.test_DF)
        if self.split == 'query':
            self.all_boxes = pd.read_csv(self.query_DF)
    # as you would do normally

    def __getitem__(self, index):
        # load the image as a PIL Image
        img_id = self.ids[index]
        im_path = os.path.join(self.root, img_id)
        img = Image.open(im_path).convert("RGB")

        target = self.get_groundtruth(index)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        img_num = img_id.split('.')[0][5:11]
        if self.split == 'query':
            boxes_df = self.all_boxes.query('@index')
            boxes_x1 = boxes_df['x1'].copy().astype(np.float32)
            boxes_y1 = boxes_df['y1'].copy().astype(np.float32)
            boxes_x2 = boxes_df['del_x'].copy() + boxes_x1
            boxes_y2 = boxes_df['del_y'].copy() + boxes_y1
            pid = boxes_df['pid'].copy()
            boxes = [[boxes_x1, boxes_y1, boxes_x2, boxes_y2, pid]]
        else:
            boxes_df = self.all_boxes.query('imname==@img_id')
            boxes = boxes_df.loc[:, 'x1': 'pid'].copy()
            #boxes = boxes_df.copy()
            boxes.loc[:, 'del_x'] += boxes.loc[:, 'x1']
            boxes.loc[:, 'del_y'] += boxes.loc[:, 'y1']
            boxes = boxes.values.astype(np.float32)
            boxes = boxes.tolist()
        anno = self._preprocess_annotation(img_num, boxes)

        orig_shape = self.get_img_info(index)

        (width, height) = orig_shape['width'], orig_shape['height']
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("img", anno["img_name"])
        target.add_field("labels", anno["labels"])
        target.add_field("pid", anno["pid"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, img, target):
        boxes = []
        gt_classes = []
        pid = []
        img_name = []
        name = 'person'
        difficult = False
        difficult_boxes = []
        for obj in target:
            bndbox = tuple(
                map(
                    int,
                    [
                        obj[0],
                        obj[1],
                        obj[2],
                        obj[3],
                    ],
                )
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            pid.append(int(obj[-1]))
            img_name.append(int(img)) ###
            difficult_boxes.append(difficult)

        res = {
            "img_name": torch.tensor(img_name, dtype=torch.int32),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "pid": torch.tensor(pid, dtype=torch.int32),
            "difficult": torch.tensor(difficult_boxes),
            #"im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_id = self.ids[index]
        im_path = os.path.join(self.root, img_id)
        img = cv2.imread(im_path).astype(np.float32)
        orig_shape = img.shape

        return {"height": orig_shape[0], "width": orig_shape[1]}

    def map_class_id_to_class_name(self, class_id):
        return PRWDataset.CLASSES[class_id]

