import os
import json
import glob
import argparse
import numpy as np
import pandas as pd

import pdb

# Decide how many digits there are in a number given the label
def get_digit_num(label):
    if "thousand" in label:
        return 4
    elif "hundred" in label:
        return 3
    elif "teen" in label or "ty" in label:
        return 2
    else:
        return 1

# Return attention regions for each file name given image name, prediciton, 
# label and model name for each word
def get_raw_att_file(prediction, label, im_att_files):
    word_region_att = {}
    #if type(prediction) == str:
    for i in range(len(prediction)):
        # Load attention map
        word = prediction[i]
        if word != "<end>" and word != "<start>":
            att_file = [f for f in im_att_files if word in f][0]
            att_map = np.load(att_file)
            # Get num of digits            
            num_digit = get_digit_num(label)
            # Attention region
            word_region_att[i] = get_region_scores(att_map, num_digit)
    return word_region_att

        
# Given raw attention map return attention score for right as 
# well as left region
def get_region_scores(att_map, num_digit):
    # attention regions output
    att_region_scores = {}
    r, c = att_map.shape
    regions = ['left', 'right']
    atts = [np.sum(att_map[:, :int(c/2)]) / np.sum(att_map), 
            np.sum(att_map[:, int(c/2):]) / np.sum(att_map)]
    
    for i in range(len(regions)):
        region, att = regions[i], atts[i]
        att_region_scores[region] = att
    return att_region_scores


# Generate attention region result for each file, return a pd.DF object including:
# Label, Prediction, Word_i_attention_region (where i is nth word in label)
# Since we don't care whether an image is foil or target convert 1 row in res to 2 rows
# in att_region
def get_attention_region_single(res):
    # Get all attention files for the model
    # att_files = glob.glob("{}/*npy".format(res))
    att_files = glob.glob("{}/images_raw/test_{}/*npy".format(res, args.type))
    
    # pdb.set_trace()

    # Row and cols in the ouput
    word_region_col = [["Word_{}_Left_score".format(i), "Word_{}_Left_score".format(i)] for i in range(1, 7)]
    word_region_col = [t for tp in word_region_col for t in tp]
    out_cols = ["Image", "Label", "Prediction"] + word_region_col
    rows = []
    for im in sorted(gt.keys()):
        # retrieve attention files related to current images
        im_att_files = [f for f in att_files if f[f.rfind('/')+1:].startswith(im)]
        label = gt[im]
        prediction = [f[f.rfind('/')+1:][f[f.rfind('/')+1:].find('_')+1: f[f.rfind('/')+1:].rfind('_')] for f in im_att_files]
                
        # In the format of {word_i : {left : score, right : score}}
        att_region_scores = get_raw_att_file(prediction, label, im_att_files)
        row = [im, label, " ".join(prediction)]
        for word in att_region_scores.keys():
            row += [att_region_scores[word]['left'], att_region_scores[word]['right']]
        row += ['N/A' for i in range(len(out_cols) - len(row))]
        rows.append(row)
    att_region_df = pd.DataFrame(rows, columns=out_cols)
    return att_region_df


# For each model, generate a csv file containing results for attention region
def get_attention_region(raw_att_dir):
    for i in range(1, args.num_of_models+1):
        model = "model_{}".format(i)
        " For new/old set"
        #raw_att_path = os.path.join(raw_att_dir, "{}/{}".format(model, args.type))
        #res = os.path.join(args.out, "att_{}_{}.csv".format(args.type, model))
        "For new test set"
        
        raw_att_path = os.path.join(raw_att_dir, model)
        # TODO: update it later, not needed
        raw_att_path += "/drive/MyDrive/Colab/Github/Learning-generative-principles-of-a-symbol-system/Training_and_testing_items"

        res = os.path.join(args.out, "res_{}.csv".format(model))
        # Get proabilities for attention regions
        att_region_df = get_attention_region_single(raw_att_path)
        print("Saving to {}".format(res))
        att_region_df.to_csv(res)



parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption.')
parser.add_argument('--gt', '-g', help='Ground truch json file.')    
# parser.add_argument('--attention_region_map', '-a', help='Attention region map file.')
parser.add_argument('--raw_attention', '-ra', help="Path to raw attention map directories.")
parser.add_argument('--num_of_models', '-n', type=int, help="Number of models.")
parser.add_argument('--type', '-t', help="Whether it's new or old.")
parser.add_argument('--out', '-o', help="Output directory where csv files will be saved.")
args = parser.parse_args()
# Load attention region map
# att_region_map = json.load(open(args.attention_region_map))
gt= json.load(open(args.gt))
get_attention_region(args.raw_attention)
