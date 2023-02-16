""" 
Create captioning file for the dataset with:
1). training ground truth,
2). testing ground truth,
3). validation ground truth - kids set

"""
import os
import json
import argparse
import numpy as np
import pandas as pd


# Create caption set in MSCOCO format
def create_coco_struct(gt_data, image_id, sentence_id, imgs, f_path, split):
    for img in gt_data.keys():
        sentence = gt_data[img]
        img_caption = {'sentenceids' : [sentence_id], 'imgid': image_id,
                       'filepath' : f_path, 'filename': img,
                       'sentences' : [{'tokens': sentence.split(), 
                                       'raw': sentence, 'imgid': image_id,
                                       'sentid': sentence_id}],
                       'split' : split}
        imgs.append(img_caption)
        sentence_id += 1
        image_id += 1
    return imgs, image_id, sentence_id

def create_caption_labels(train, val, out):    
    gt_train = json.load(open(train, 'r'))
    gt_val = json.load(open(val, 'r'))
    # Initialize imgid, senid and list of img_captions
    sentence_id, image_id = 0, 0
    imgs = []
    # Training set load
    train_path = train[:train.rfind('/')]
    imgs, img_id, sentence_id = create_coco_struct(gt_train, image_id, sentence_id, imgs, train_path,'train')
    print("Number of training samples: {}".format(len(imgs)))
    # Validation set load
    val_path = val[:val.rfind('/')]
    imgs, img_id, sentence_id = create_coco_struct(gt_val, image_id, sentence_id, imgs, val_path, 'val')
    print("Number after val samples: {}".format(len(imgs)))
    # Aggregate everything into one .json file
    caption_out = {'images' : imgs, 'dataset' : 'Number'}
    f_out = open(out, 'w+')
    json.dump(caption_out, f_out)

##### Parsing arguments #####
parser = argparse.ArgumentParser(description='Process command line optional inputs')
parser.add_argument('--train_gt', '-tr', help="Gt path to training set.")
parser.add_argument('--val_gt', '-vl', help="Gt path to validation set(kids gt) .")
parser.add_argument('-out',  '-o',help="Caption label json file.")
args = parser.parse_args()
create_caption_labels(args.train_gt, args.val_gt, args.out)
