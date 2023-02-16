import os, sys
import argparse
import numpy as np
import csv, json
import cv2

"""
This script is used to create dataset for number recognition task. This can be used for test csv files
Command line input:
1). input_file: the pre-existing dataset instructions
2). out: directory where the dataset (imgs and labels) go
"""
"Modified for caption training with 'and' and no - before hundreds/thousands."

# Import and save label map as .json file using given file (for up to 5 digit numbers)
''' Only need --out_dir from command line input (This function ONLY need to be run ONCE) '''
def import_label_map(out_dir, label_map='label_map.csv'):
    print("Reading label map for numbers up to five digits from {}..".format(label_map))
    f_csv = open(label_map, newline='')
    reader = csv.reader(f_csv)
    label_map_res = { row[0] : row[1] for row in reader if 'label' not in row }
    f_csv.close()
    print("Saving to {}".format(os.path.join(out_dir, 'label_map.json')))
    f_json = open(os.path.join(out_dir, 'label_map.json'), 'w+')
    json.dump(label_map_res, f_json)
    f_json.close()

# Load dataset instruction from a given file (data_points format: [ num, label, freq ]
def load_dataset(input_file, is_test):
    print("Loading dataset from {}".format(input_file))
    with open(input_file) as f:
        reader = csv.reader(f)
        # Create an array of data points with given frequency
        dataset_instr = []
        if is_test is None:
            dataset_instr = [ row[1:3] for row in reader if 'visual' not in row for i in range(int(row[3])) ]
        else:    
            for row in reader:
                if 'image_name' not in row:
                    first_num = row[2:4]
                    second_num = row[4:6]
                    dataset_instr.append(first_num)
                    dataset_instr.append(second_num)
    f.close()
    return dataset_instr

def correctString(s):
    sp_ch={'-':' ', ',':'', '.':''}
    for k,v in sp_ch.items():
        s=s.replace(k,v)
    return s.lower()
# Generate images and text labels using loaded insturction
def generate_dataset(dataset_instr, out_dir):
    # Desired size of imgs in the dataset, all images will be reshaped to this
    data_dict = {}
    unique={}
    w, h = 224, 224
    for i in range(len(dataset_instr)):
        num, label = dataset_instr[i][0], correctString(dataset_instr[i][1])
        if label not in unique:
            unique[label]=True
        # Font type, scale to its original size and thickness
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 2, 2
        # txt_w - txt_size[0], txt_h - txt_size[1]
        txt_size = cv2.getTextSize(num, font, font_scale, thickness)[0]
        # Decide the size of the image from font size (im_w = 4 * txt_w, im_h = 3 * txt_h)
        #IMG_HEIGHT, IMG_WIDTH = 3 * txt_size[1], 4 * txt_size[0]
        IMG_HEIGHT, IMG_WIDTH = txt_size[1], txt_size[0]
        # Create an all-white blank image
        img = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8) * 255
        # Position of txt
        txt_x, txt_y = int(IMG_WIDTH / 2 - txt_size[0] / 2), int(IMG_HEIGHT / 2 + txt_size[1] / 2)
        cv2.putText(img, num, (txt_x, txt_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)        
        out_img = cv2.resize(img, (w, h))
        img_name = "{}.png".format(i)
        # Save dataset
        data_dict[img_name] = label
        #print("Saving {} for label {}".format(os.path.join(out_dir, img_name), label))
        cv2.imwrite(os.path.join(out_dir, img_name), out_img)
    # Save data dict
    data_dump=str([k for k in unique])
    with open(os.path.join(out_dir, 'num_labels.json'), 'w+') as fp:
        json.dump(data_dict, fp)
        fp.close()
    with open(os.path.join(out_dir, 'unique.json'), 'w+') as fp:
        fp.write(data_dump)
        fp.close()
    return data_dict

# Generate image and ground truth
def generate_gt(label_map, num_label, out_dir):
    # Number of elements on the label vector
    num_of_elements = len(label_map.keys())
    num_label_vec = {}
    # Create label vectors based on the num_labels
    for im_num in num_label:
        label_vec = np.zeros((num_of_elements, 1))
        label_tokens = num_label[im_num].split()
        # Fill in ones at given label position in the label vector
        for t in label_tokens:
            label_vec[int(label_map[t]), 0] = 1        

        num_label_vec[im_num] = label_vec
    # Save num_label_vectors as npy file 
    np.save(os.path.join(out_dir, 'num_label_gt.npy'),  num_label_vec)
    # Load ground truth this way
    loaded_dict = np.load(os.path.join(out_dir, 'num_label_gt.npy'))


##### Parsing arguments #####
parser = argparse.ArgumentParser(description='Process command line optional inputs.')
parser.add_argument('--input_file', '-i', help="Instrution file if want to create certain numbers.")
parser.add_argument('--out_dir',  '-o',help="Output directory.")
parser.add_argument('--test_set',  '-t',help="Generate for test set.")
#parser.add_argument('--label_map', '-l', help="Path to label map.")
args = parser.parse_args()
input_file, out, is_test = args.input_file, args.out_dir, args.test_set
# ONLY Run ONCE to generate label map ( one : 0, two : 1, ...) where the keys are the 1 position 
# label vector (Done)
#import_label_map(out)
##### Main #####
if input_file != None:
    dataset_instr = load_dataset(input_file, is_test)
    print(dataset_instr)
    num_label = generate_dataset(dataset_instr, out)
    #label_map = json.load(open(args.label_map))
    #generate_gt(label_map, num_label, out)
    
