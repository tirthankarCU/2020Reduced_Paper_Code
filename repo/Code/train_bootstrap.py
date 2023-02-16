"""
https:
//github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py
"""


from math import e
import os
import sys
import time
import argparse
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import json
import math


# I/O parameters for data and model (parser from command line)
parser = argparse.ArgumentParser(description="Show, attend and tell training bootstrapping")
parser.add_argument('--model', '-m', help="Path where model is saved")
parser.add_argument('--data_dir', '-d', help="Path where data files are saved by create_input_files.py")
parser.add_argument('--num_models', '-n', type=int, help="Number of models that needs to be trained", default=20)
parser.add_argument('--num_epochs', '-e', type=int, help="Max number of epochs that model needs to be trained until", default=100)
parser.add_argument('--batch_size', '-b', type=int, help="Batch size", default=64)
parser.add_argument('--cl_initial_percentage', '-cip', type=float, help='Initial lesson data percentage in *_cl (currriculum learning) type', default=1)
parser.add_argument('--cl_max_epochs_per_lesson', '-cme', type=int, help='Max epochs per lesson when curriculum learning is used', default=10)
parser.add_argument('--cl_change_percent_lesson', '-ccp', type=float, help='Change in percentage per lesson', default=1.9)

LEARN_DEFAULT, LEARN_FRONT_CL, LEARN_BACK_CL, LEARN_RANDOM_CL = "default", "front_cl", "back_cl", "random_cl"
parser.add_argument('--learning_type', '-l', help='Learning type possible values: "default", "front_cl", "back_cl", "random_cl"', default='default', choices=[LEARN_DEFAULT, LEARN_FRONT_CL, LEARN_BACK_CL, LEARN_RANDOM_CL])

args = parser.parse_args()

# Data parameters
data_folder = args.data_dir  # folder with data files saved by create_input_files.py
data_name = 'coco_1_cap_per_img_1_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = args.batch_size
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
lowest_loss_val = float('Inf')
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

def get_cl_lessons():
    return 1 + math.ceil(math.log((1/args.cl_initial_percentage), args.cl_change_percent_lesson))

def get_max_epochs():
    if args.learning_type == LEARN_DEFAULT:
        return args.num_epochs
    return get_cl_lessons() * args.cl_max_epochs_per_lesson

start_epoch = 0
max_epochs = get_max_epochs()  # number of epochs to train for (if early stopping is not triggered)
def current_lesson(epoch):
    return int(epoch/args.cl_max_epochs_per_lesson) + 1
    
def get_lesson_size(epoch, data_size):
    c_lesson = current_lesson(epoch) - 1
    current_percent = max(args.cl_initial_percentage * math.pow(args.cl_change_percent_lesson, c_lesson), 1)
    return math.ceil(data_size * current_percent)

def shuffle_dataset():
    return args.learning_type == LEARN_DEFAULT or args.learning_type == LEARN_RANDOM_CL

def reverse_dataset():
    return args.learning_type == LEARN_BACK_CL

# Save checkpoint
def save_checkpoint_with_dir(model_dir, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                             bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}    
    filename = 'checkpoint_' + data_name + '.pth.tar'

    if not os.path.exists(model_dir):
        # Create a new directory because it does not exist
        os.makedirs(model_dir)
        
    torch.save(state, os.path.join(model_dir, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(model_dir, 'BEST_' + filename))


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, lowest_loss_val

    if args.learning_type != LEARN_DEFAULT:
        print('Running curriculum learning strategy: ' + args.learning_type)

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    
    # Initialize Metrics container for all the models
    losses_train = []
    top5_accuracy_train = []

    losses_val = []
    top5_accuracy_val = []
    bleu4_score_val = []


    # Train N models and save them to each directory
    for n in range(1, args.num_models+1):
        # Directory where the model will be saved
        model_out = os.path.join(args.model, "model_{}".format(n))
        model_out_untrained = os.path.join(args.model, "untrained/model_{}".format(n))
        # Adding the checkpoint loading so that model can resume training from any step.
        # checkpoint = os.path.join(args.model, "model_{}/checkpoint_{}.pth.tar".format(n, data_name))
        try:
            os.mkdir(model_out)
            os.mkdir(model_out_untrained)
        except Exception as e:
            pass
        # Initialize / load checkpoint
        if checkpoint is None:
            decoder = DecoderWithAttention(attention_dim=attention_dim,
                                           embed_dim=emb_dim,
                                           decoder_dim=decoder_dim,
                                           vocab_size=len(word_map),
                                           dropout=dropout)
            decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                 lr=decoder_lr)
            encoder = Encoder()
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        else:
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            best_bleu4 = checkpoint['bleu-4']
            decoder = checkpoint['decoder']
            decoder_optimizer = checkpoint['decoder_optimizer']
            encoder = checkpoint['encoder']
            encoder_optimizer = checkpoint['encoder_optimizer']
            if fine_tune_encoder is True and encoder_optimizer is None:
                encoder.fine_tune(fine_tune_encoder)
                encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                     lr=encoder_lr)

        # Move to GPU, if available
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        
        # # # Save the untrained model
        save_checkpoint_with_dir(model_out_untrained, data_name, 0, 0, 
                            encoder, decoder, encoder_optimizer,
                            decoder_optimizer, 0, True)

        # Loss function
        criterion = nn.CrossEntropyLoss().to(device)

        # Custom dataloaders
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std =[0.229, 0.224, 0.225])
        # data_folder = {DATA_DIR}/images_processed/
        # data_name = 'coco_1_cap_per_img_1_min_word_freq' -> base name shared by data files
        train_loader = torch.utils.data.DataLoader(
            CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]), reverse=reverse_dataset()),
            batch_size=batch_size, shuffle=shuffle_dataset(), num_workers=workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
            batch_size=batch_size, shuffle=shuffle_dataset(), num_workers=workers, pin_memory=True)

        #Initializing metrics tracking for each epoch
        losses_per_epoch_train = []
        top5_accuracy_per_epoch_train = []

        losses_per_epoch_val = []
        top5_accuracy_per_epoch_val = []
        bleu4_score_per_epoch_val = []

        # Epochs
        for epoch in range(start_epoch, max_epochs):
            # Decay learning rate if there is no improvement for 20 consecutive epochs
            # and terminate training after 50 consecutive epochs
            if epochs_since_improvement == 50:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, 0.8)

            # One epoch's training
            recent_loss, top5_accuracy = train(train_loader=train_loader,
                  encoder=encoder,
                  decoder=decoder,
                  criterion=criterion,
                  encoder_optimizer=encoder_optimizer,
                  decoder_optimizer=decoder_optimizer,
                  epoch=epoch)

            # Update training metrics for plot
            losses_per_epoch_train.append(recent_loss)
            top5_accuracy_per_epoch_train.append(top5_accuracy)

            # One epoch's validation
            recent_loss, top5_accuracy, recent_bleu4 = validate(val_loader=val_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion)

            # Update validation metrics for plot
            losses_per_epoch_val.append(recent_loss)
            top5_accuracy_per_epoch_val.append(top5_accuracy)
            bleu4_score_per_epoch_val.append(recent_bleu4)

            # Check if there was an improvement using bleu
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            # Check if there was an improvement using loss
            #is_best = recent_loss < lowest_loss_val
            #lowest_loss_val = min(recent_loss, lowest_loss_val)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            #save_checkpoint_with_dir(model_out, data_name, epoch, epochs_since_improvement, 
            #                         encoder, decoder, encoder_optimizer,
            #                         decoder_optimizer, lowest_loss_val, is_best)
            save_checkpoint_with_dir(model_out, data_name, epoch, epochs_since_improvement, 
                                     encoder, decoder, encoder_optimizer,
                                     decoder_optimizer, recent_bleu4, is_best)
        # Delete encoder&decoder objects and reset memory
        del decoder
        del encoder
        torch.cuda.empty_cache()
        # Reset epochs since improvement to 0 for a new round of training
        epochs_since_improvement = 0
        best_bleu4, start_epoch = 0, 0
        check_point = None
        fine_tune_encoder = False
    
        # Update the global metrics
        losses_train.append(losses_per_epoch_train)
        top5_accuracy_train.append(top5_accuracy_per_epoch_train)

        losses_val.append(losses_per_epoch_val)
        top5_accuracy_val.append(top5_accuracy_per_epoch_val)
        bleu4_score_val.append(bleu4_score_per_epoch_val)

    # Save metrics data
    all_metrics_data = {
        "training_loss": losses_train,
        "top5_training_accuracy": top5_accuracy_train,
        "validation_loss": losses_val,
        "top5_validation_accuracy": top5_accuracy_val,
        "bleu_score": bleu4_score_val
    }
    with open(os.path.join(args.model, "all_metrics_data.json"), "w") as outfile:
        json.dump(all_metrics_data, outfile)
    
            
def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    lesson_size = get_lesson_size(epoch, len(train_loader))

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):

        if i == lesson_size:
            break

        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}/{1}] Batch: [{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Lesson: [{cl}/{tcl}]'.format(epoch, max_epochs, i, len(train_loader),
                                            batch_time=batch_time,
                                            data_time=data_time, loss=losses,
                                            top5=top5accs, cl=current_lesson(epoch), tcl=get_cl_lessons()))
                  

    return losses.avg, top5accs.avg

def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        allcaps = allcaps.to(device)
        
        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, top5=top5accs))

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return losses.avg, top5accs.avg,  bleu4



if __name__ == '__main__':        
    main()
