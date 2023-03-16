"""
This script is adpted from:
 https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse

from cv2 import imread
from cv2 import resize as imresize
from nltk.metrics.distance import edit_distance
from models import Encoder, DecoderWithAttention
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=1):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)
    # Keep track of the raw score for each predicted words
    img_scores = []
    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img).to(device)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        # Hidden and cell state
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        # Probability for each token
        scores = decoder.fc(h)  # (s, vocab_size)
        #scores = F.log_softmax(scores, dim=1)
        scores = F.softmax(scores, dim=1) #scores.shape is (1,26) for this problem.
        #scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
        img_scores.append(scores.cpu().detach().numpy())
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)
        
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha, alpha[prev_word_inds].unsqueeze(1)],dim=1)  # (s, step+1, enc_image_size, enc_image_size)
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly
        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    return img_scores, seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True, model='model_1'):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    # Keep track of words that have been saved & number of occurrences
    # in case of multiple word in the same number
    #word_saved = {}
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
        "Save the value of attention map"
        current_word = words[t]
        image_info = image_path[image_path.find("test_")+len("test_"):]
        # which set and type (new or old) is the image
        image_type = image_info[:image_info.rfind('/')]
        image_name = image_info[image_info.rfind('/') + 1 : ]
        # Create directory to save raw attention map, visualization and trial results
        sub_dirs = model + '/' + image_type
        out_att_map = os.path.join(args.out_att_map,f"{model}")
        #out_vis = os.path.join(args.out_vis, sub_dirs)    # No need for final
        try:
            os.makedirs(out_att_map)
            #os.makedirs(out_vis)    # No need for final
        except Exception as e:
            pass
        if current_word not in ['<start>', '<end>']:
            # Save raw attention map files
            att_map_file = "{}_{}_{}.npy".format(image_name, current_word, t)
            np.save(os.path.join(out_att_map, att_map_file), alpha)

    # Save attention visualization on original image
    #plt.savefig(os.path.join(out_vis, image_name))
    #plt.close()
    return ' '.join(words[1:-1])


# Evaluate caption results for each trial, create output and accuracy
# Load trials into a hash
def load_trials(trials):
    img_target = {}
    for index, row in trials.iterrows():
        word_target, word_foil = row["word_target"], row["word_foil"]
        target, foil = row["image_name"].split(' ')
        img_target[target] = word_target
        img_target[foil] = word_foil
    return img_target

# Evaluation for all the trials (first item in image_name is always target and the second is foil)
"Compare raw scores"
def eval_trials(trial, img_scores, img_caption, gt, model="model_1"):
    # Load word map (word to index)
    word_map = json.load(open(args.word_map, 'r'))
    # Set information and create directories
    #set_info = args.img_dir[ args.img_dir.find("eval/")+len("eval/"): ]
    #set_name = set_info[ : set_info.find('/')]
    #set_type = set_info[set_info.find('/') + 1 : ]    
    set_type = args.set_type
    # Create directory to save trial output
    #trial_res_out = os.path.join(args.out_trial, set_name)
    trial_res_out = args.out_trial
    trial_prob_out = os.path.join(args.out_prob, model)
                                  #"{}/{}".format(set_name, 
                                  #               model))
    res_name = "res_{}_{}.csv".format(set_type, model)
    try:
        os.makedirs(trial_prob_out)
        os.makedirs(trial_res_out)
    except:
        pass
        
    # Load ground truth and trial info for testing set
    trials = pd.read_csv(trial)    
    gt_caption = json.load(open(gt, 'r'))
    # Result dataframe
    res_df = pd.DataFrame(columns=["desired_label", "target_image", "target_model_generated_label", 
                                   "foil_image", "foil_word", "foil_model_generated_label", "correct"], 
                          index=trials.index.values.tolist())
    sorted_word_map = list(sorted(word_map.items(), 
                                  key = lambda kv:(kv[1], kv[0])))
    # Load trials into desired format (img : target_number)
    img_target = load_trials(trials)
    # Whichever one image's score has the highest value at the raw scores
    correct = 0
    # Get result
    for index, row in trials.iterrows():
        target, foil = row["image_name"].split(' ')
        word_target = img_target[target]
        "Used to fill out the csv file"
        # Get net output for target and foil image
        target_res, target_label = img_caption[target], gt_caption[target]
        foil_res, foil_label = img_caption[foil], gt_caption[foil]

        "Save probabilities"
        # Target probs
        target_prob_df = pd.DataFrame(index=[word_map[0] for word_map in sorted_word_map])
        target_probs = img_scores[target]
        for i in range(len(target_probs)):
            target_prob_df["Time_{}".format(i)] = target_probs[i][0]    # Raw probs
                                                   
        # Foil probs
        foil_prob_df = pd.DataFrame(index=[word_map[0] for word_map in sorted_word_map])
        foil_probs = img_scores[foil]
        for i in range(len(foil_probs)):
            foil_prob_df["Time_{}".format(i)] = foil_probs[i][0]    # Raw probs

        # Probability output
        "Foil and target separate"
        prob_name = os.path.join(trial_prob_out, 
                                 "prob_{}_trial{}.csv".format(set_type, index))                                                            
        pd.concat([target_prob_df, foil_prob_df],
                  keys=[target_label, foil_label]).to_csv(prob_name)

        "Deciding whether a trial is correct or not basing on the probability"
        is_correct = evaluate_single_trial(word_target, 
                                           word_map,
                                           target_probs, 
                                           foil_probs)
        res_df.loc[index] = [target_label, target, target_res,
                             foil, foil_label, foil_res, is_correct]
        
    # Accuracy and output of evaluation results
    print("Saving result to {}".format(os.path.join(trial_res_out, res_name)))    
    res_df.to_csv(os.path.join(trial_res_out, res_name))
    # Compute the accuracy for the test set
    total = res_df.count()["target_image"]
    correct = res_df[res_df["correct"]==1].count()["target_image"]
    incorrect = res_df[res_df["correct"]==0].count()["target_image"]
    print("{}'s accuracy for {} is {}".format(model, set_type, correct/total))
    return "{}, {}, {} \n".format(model, set_type, correct/total)



# Evaluate if a trial is correct given correct label, wordmap, 
# foil_prob and target prob 
def evaluate_single_trial(word_target, word_map,
                          target_prob, foil_prob):
    words = word_target.split(' ')
    num_words = len(words)
    # Keep track of normalized by mu and std probs 
    # for foil and target at desired index
    t_probs, f_probs = [], []
    for i in range(num_words):
        if words[i] in word_map.keys():
            word_idx = word_map[words[i]]
        else:
            word_idx = word_map["<unk>"]
        # Compute normalized log probability (Ignore <end> token)
        if i < len(target_prob) and np.argmax(target_prob[i])!=24:
            t_prob = (target_prob[i][0, word_idx] - np.mean(target_prob[i])) / np.std(target_prob[i])
            #t_prob = target_prob[i][0, word_idx]    # Use raw probs
            t_probs.append(t_prob)
        if i < len(foil_prob) and np.argmax(foil_prob[i])!=24:
            f_prob = (foil_prob[i][0, word_idx] - np.mean(foil_prob[i])) / np.std(foil_prob[i])
            #f_prob = foil_prob[i][0, word_idx]    # Use raw probs
            f_probs.append(f_prob)
    # Compare normalized mean probability
    return np.mean(t_probs) > np.mean(f_probs)
    

"Using edit distance"
def eval_trials_edit_distance(trial, img_caption, gt, model="model_1"):
    # Set information and create directories
    #set_info = args.img_dir[ args.img_dir.find("eval/")+len("eval/"): ]
    #set_name = set_info[ : set_info.find('/')]
    #set_type = set_info[set_info.find('/') + 1 : ]
    set_type = args.set_type

    # Create directory to save trial output
    #trial_res_out = os.path.join(args.out_trial, set_name)
    trial_res_out = args.out_trial
    res_name = "res_{}_{}.csv".format(set_type, model)
    try:
        os.makedirs(trial_res_out)
    except:
        pass

    # Load ground truth and trial info for testing set
    trials = pd.read_csv(trial)    
    gt_caption = json.load(open(gt, 'r'))
    # Result dataframe
    res_df = pd.DataFrame(columns=[ "desired_label", "target_image", "target_model_generated_label", 
                                    "dis_target_predicted", "foil_image", "foil_word", 
                                    "foil_model_generated_label", "dis_foil_predicted", "correct"], 
                          index=trials.index.values.tolist())
    # Load trials into desired format (img : target_number)
    img_target = load_trials(trials)
    # A trial is counted as correct if (in order):
    # 1). one of the number is correctly predicted 
    # 2). the edit_distance between target is less than foil
    correct = 0
    # Get result
    for index, row in trials.iterrows():
        target, foil = row["image_name"].split(' ')
        word_target = img_target[target]
        # Get net output for target and foil image
        target_res, target_label = img_caption[target], gt_caption[target]
        foil_res, foil_label = img_caption[foil], gt_caption[foil]
        # Compute target
        # A trial is counted as correct if and only if following conditions are met:
        # -- dis_target_predicted or foil_predict_correct == 0
        # else:
        # 
        dis_foil_predicted = edit_distance(word_target, foil_res)
        dis_target_predicted = edit_distance(word_target, target_res)
        # Record foil_predict_correct for evaluation (0 is correct prediciton)
        foil_predict_correct = edit_distance(foil_label, foil_res)
        is_correct = 0

        # Negation
        if not foil_predict_correct or not dis_target_predicted:
            is_correct = 1
        else:
            if dis_target_predicted < dis_foil_predicted:
                is_correct = 1

        res_df.loc[index] = [target_label, target, target_res, dis_target_predicted, 
                             foil, foil_label, foil_res, dis_foil_predicted, is_correct]

    print("Saving result to {}".format(os.path.join(trial_res_out, res_name)))
    res_df.to_csv(os.path.join(trial_res_out, res_name))
    # Compute the accuracy for the test set
    total = res_df.count()["target_image"]
    correct = res_df[res_df["correct"]==1].count()["target_image"]
    incorrect = res_df[res_df["correct"]==0].count()["target_image"]    
    print("{}'s accuracy for {} is {} edit".format(model, set_type, correct/total))
    return "{}, {}, {} \n".format(model, set_type, correct/total)


# Command line arguments    
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--model_dir', '-m', help='path to model root directory')
parser.add_argument('--word_map', '-wm', help='path to word map JSON')
parser.add_argument('--beam_size', '-b', default=1, type=int, help='beam size for beam search')
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
''' Create caption for images in a folder (modified by violetxi) '''
parser.add_argument('--set_type', '-st', help="old or new")
parser.add_argument('--num_models', '-n', type=int, help="Number of models used for evaluation")
parser.add_argument('--img_dir', '-i', help='path to image directory')
parser.add_argument('--out_vis', '-ov', help='Output directory to save visualization')
parser.add_argument('--out_att_map', '-oa', help="Output directory to save raw attention map")
parser.add_argument('--gt', '-g', help='Ground truth for the testing set.')
parser.add_argument('--trial', '-t', help="Path to where tiral file is.")
parser.add_argument('--out_trial', '-ot', help='Output directory to save trial results')
parser.add_argument('--out_prob', '-op', help='Output directory to save raw probabilities')
parser.add_argument('--res_acc', '-r', help="Accuracy of each model on old/new testing set")
parser.add_argument('--eval_type', '-et', help="Evaluation type: prob or edit")
args = parser.parse_args()
# model name
model_file = 'BEST_checkpoint_coco_1_cap_per_img_1_min_word_freq.pth.tar'
#model_file = 'checkpoint_coco_1_cap_per_img_1_min_word_freq.pth.tar'

if __name__ == '__main__':
    # File to save overall accuracy of a model
    if not args.eval_type == "attention":
        res_acc_f = open(args.res_acc, 'a')
    for n in range(1, args.num_models+1):
        # Load word map (word2ix)
        with open(args.word_map, 'r') as j:
            word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
        "For post-test situation loads n trained models"
        cur_model = "model_" + str(n)
        model_path = os.path.join(args.model_dir, os.path.join(cur_model, model_file))
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
    
        # Create captions for all images in the directory and visulization
        img_caption = {}
        imgs = [f for f in os.listdir(args.img_dir) if f.endswith('png') or f.endswith('jpg')]
        img_scores = {}
        for img in imgs:
            img_full = os.path.join(args.img_dir, img)
            # Encode, decode with attention and beam search
            scores, seq, alphas = caption_image_beam_search(encoder, decoder, img_full, word_map, args.beam_size)
            # Update im score information
            img_scores[img] = scores
            alphas = torch.FloatTensor(alphas)
            # Visualize caption and attention of best sequence
            caption = visualize_att(img_full, seq, alphas, rev_word_map, args.smooth, model=cur_model)
            img_caption[img] = caption
            
        # Get model accuracy and save it into a txt file
        if args.eval_type == 'edit':
            model_acc = eval_trials_edit_distance(args.trial, img_caption, args.gt, model=cur_model)
            res_acc_f.write(model_acc)
        elif args.eval_type == 'prob':    # Probability
            model_acc = eval_trials(args.trial, img_scores, img_caption, args.gt, model=cur_model)
            res_acc_f.write(model_acc)
        else:
            print("Edit distance and probability not computed.")
    
    if not args.eval_type == "attention":
        res_acc_f.close()
