# relative path imports
import sys
import json
import numpy as np

sys.path.insert(0, './models/')

from data_loader_captions import DataProvision
import os

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Sampler

#from torchsummary import summary
from tqdm import tqdm

# dirty fix to train model
from temporal_proposals import Temporal_Proposal_Architecture as architecture
from caption_module import Caption_Module_Architecture as caption_architecture

from opt import default_options


torch.autograd.set_detect_anomaly(True)

sys.path.insert(0, './densevid_eval-master')

from evaluator_old import *

class WithoutReplacementRandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.randperm(len(self.data_source))[:self.num_samples]
        return iter(samples)

    def __len__(self):
        return self.num_samples


def fetch_dataloader(types, opt, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    for split in ['train', 'val_1', 'val_2', 'test']:

        if split in types:
            #path = os.path.join(data_dir, split)
            # use the train_transformer if training data, else use eval_transformer without random flip

            vid_dataset = DataProvision(opt, split)
            if split == 'train':
                if params['use_sampler'] == True:
                    vid_dataset_sampler = WithoutReplacementRandomSampler(vid_dataset, num_samples=params['num_vids_per_epoch'])
                else:
                    vid_dataset_sampler = None
                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=False,
                                num_workers=params['num_workers'], sampler = vid_dataset_sampler)
            else:
                if params['use_sampler'] == True:
                    vid_dataset_sampler = WithoutReplacementRandomSampler(vid_dataset, num_samples=params['num_vids_per_epoch'])
                else:
                    vid_dataset_sampler = None
                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=False,
                                num_workers=params['num_workers'], sampler = vid_dataset_sampler)
                # dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=False,
                #                 num_workers=params['num_workers'],
                #                 pin_memory=params['cuda'])

            dataloaders[split] = dl
    return dataloaders, vid_dataset

# taken from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

def lastindex(lst, value):
    lst.reverse()
    i = lst.index(value)
    lst.reverse()
    return len(lst) - i - 1

def getKey(item):
    return item['score']

def getJointKey(item):
    return item['proposal_score'] + 5.*item['sentence_confidence']

def remove_repeated_word(sentence):

    sen_idx = [0]
    for i in range(1, len(sentence)):
        if sentence[i] != sentence[i-1]:
            sen_idx.append(i)

    sentence = [sentence[idx] for idx in sen_idx]
    return sentence

"""
Generate batch data and corresponding mask data for the input
"""
def process_batch_data(batch_data, max_length):
    dim = batch_data[0].shape[1]

    out_batch_data = torch.zeros(len(batch_data), max_length, dim, dtype=torch.float)
    out_batch_data_mask = torch.zeros(len(batch_data), max_length, dtype=torch.float)

    for i, data in enumerate(batch_data):
        effective_len = min(max_length, data.shape[0])
        out_batch_data[i, :effective_len, :] = data[:effective_len]
        out_batch_data_mask[i, :effective_len] = 1.

    out_batch_data = torch.FloatTensor(out_batch_data)
    out_batch_data_mask = torch.FloatTensor(out_batch_data_mask)

    return out_batch_data, out_batch_data_mask


def evaluation_metric_greedy(proposal_model, caption_model, options, data_loader, vid_dl):

    print('Evaluating caption scores ...')

    word2ix = options['vocab']
    ix2word = {ix:word for word,ix in word2ix.items()}
    
    # output json data for evaluation
    out_data = {}
    out_data['version'] = 'VERSION 1.0'
    out_data['external_data'] = {'used':False, 'details': ''}
    out_data['results'] = {}
    results = {}
    
    count = 0
    batch_size = options['eval_batch_size']    # default batch size to evaluate
    assert batch_size == 1
    
    # how we got @1000 metrics
    eval_num = batch_size*len(data_loader)
    print('Will evaluate %d samples'%eval_num)

    # get the 1000 val_id's, might be better to get them from the dataloader itself since it does return them
    ## Just sample 1000 val_ids from dataloader call
    val_ids = vid_dl.get_ids()[:eval_num] # dont need these
    anchors = vid_dl.get_anchors() # we need these
    localizaitons = vid_dl.get_localization()
    
    for i, data in enumerate(tqdm(data_loader, file=sys.stdout)):

        data_dict = data


        print('\nProcessed %d-th batch \n'%count)
        vid = data_dict["video_id"][0]
        print('video id: %s'%vid)

        
        vid_fw = data_dict["video_feat_fw"].to(device)
        vid_bw = data_dict["video_feat_bw"].to(device)

        #proposal_score_fw, proposal_score_bw, rnn_outputs_fw, rnn_outputs_bw = sess.run([proposal_outputs['proposal_score_fw'], proposal_outputs['proposal_score_bw'], proposal_outputs['rnn_outputs_fw'], proposal_outputs['rnn_outputs_bw']], feed_dict={proposal_inputs['video_feat_fw']:batch_data['video_feat_fw'], proposal_inputs['video_feat_bw']:batch_data['video_feat_bw']})
        proposal_score_fw, proposal_score_bw, rnn_outputs_fw, rnn_outputs_bw = model(vid_fw, vid_bw)
        feat_len = data_dict['video_feat_fw'][0].shape[0]
        
        duration = localizaitons['val_2'][vid]['duration']
        
        '''calculate final score by summarizing forward score and backward score
        '''
        proposal_score = torch.zeros(feat_len, options['num_anchors'])
        proposal_infos = []

        
        for i in range(feat_len):
            pre_start = -1.
            for j in range(options['num_anchors']):
                forward_score = proposal_score_fw[i,j]
                # calculate time stamp
                end = (float(i+1)/feat_len)*duration
                start = end-anchors[j]
                start = max(0., start)

                if start == pre_start:
                    continue

                # backward
                end_bw = duration - start
                i_bw = min(int(round((end_bw/duration)*feat_len)-1), feat_len-1)
                i_bw = max(i_bw, 0)
                backward_score = proposal_score_bw[i_bw,j]

                proposal_score[i,j] = forward_score*backward_score

                hidden_feat_fw = rnn_outputs_fw[i]
                hidden_feat_bw = rnn_outputs_bw[i_bw]
                    
                # change this to id's of start and end
                #TODO: look into the start id in the dataloader to see if they correspond
                proposal_feats = vid_fw[0][feat_len-1-i_bw:i+1]
                proposal_infos.append({'timestamp':[start, end], 'score': proposal_score[i,j], 'event_hidden_feat_fw': hidden_feat_fw, 'event_hidden_feat_bw': hidden_feat_bw, 'proposal_feats': proposal_feats})
                            
                pre_start = start
        
        # add the largest proposal
        hidden_feat_fw = rnn_outputs_fw[feat_len-1]
        hidden_feat_bw = rnn_outputs_bw[feat_len-1]
        
        proposal_feats = vid_fw[0]
        proposal_infos.append({'timestamp':[0., duration], 'score': 1., 'event_hidden_feat_fw': hidden_feat_fw, 'event_hidden_feat_bw': hidden_feat_bw, 'proposal_feats': proposal_feats})

        proposal_infos = sorted(proposal_infos, key=getKey, reverse=True)

        # consider adding threshold for score at 0.5
        proposal_infos = proposal_infos[:options['max_proposal_num']]

        print('Number of proposals: %d'%len(proposal_infos))

        # get hidden feats and proposal feats for the captioning network, this is still 1 vid
        event_hidden_feats_fw = [item['event_hidden_feat_fw'].cpu().detach().numpy() for item in proposal_infos]
        event_hidden_feats_bw = [item['event_hidden_feat_bw'].cpu().detach().numpy() for item in proposal_infos]
        proposal_feats = [item['proposal_feats'] for item in proposal_infos]

        
        event_hidden_feats_fw = torch.FloatTensor(event_hidden_feats_fw)
        event_hidden_feats_bw = torch.FloatTensor(event_hidden_feats_bw)
        proposal_feats, _ = process_batch_data(proposal_feats, options['max_proposal_len'])

        print(proposal_feats.shape, event_hidden_feats_fw.shape, event_hidden_feats_bw.shape)
        # quit()

        sentences, word_confidences = model_caption.caption_eval(proposal_feats.to(device), event_hidden_feats_fw.to(device), event_hidden_feats_bw.to(device))
        
        print(sentences.shape)
        sentences = [[ix2word[i] for i in ids.cpu().detach().numpy()] for ids in sentences]
        sentences = [sentence[1:] for sentence in sentences]
        # print(len(sentences))
        # print(feat_len)
        # quit()
        # remove <END> word
        #TODO: add clause if multiple end in sentence only remove last one, currently removes the first one
        #TODO: can calso consider removing all end words and everything after the last end word just like unknown below
        out_sentences = []
        sentence_confidences = []
        for i, sentence in enumerate(sentences):
            end_id = options['caption_seq_len']
            if '<END>' in sentence:
                end_id = sentence.index('<END>')
                # removed the last end keyword so as to not cut out too many words from the sentence.
                #end_id = lastindex(sentence, '<END>')
                sentence = sentence[:end_id]

            sentence = [x for x in sentence if x != '<UNK>']
            #sentence = [x for x in sentence if x != '<END>']
            sentence = remove_repeated_word(sentence)
            sentence_length = len(sentence)
            sentence = ' '.join(sentence)
            sentence = sentence.replace(' ,', ',')
            # sentence = sentence.replace('<END>', '')
            out_sentences.append(sentence)

            if sentence_length <= 4:
                sentence_confidence = -1000.   # a very low score for very short sentence 
            else:
                sentence_confidence = float(np.mean(word_confidences[i, 1:end_id]))  # use np.mean instead of np.sum to avoid favoring short sentences
            sentence_confidences.append(sentence_confidence)

        
        print('Output sentences: ')
        for out_sentence in out_sentences:
            print(out_sentence)

        result = [{'timestamp': proposal['timestamp'], 'proposal_score': float(proposal['score']), 'sentence': out_sentences[i], 'sentence_confidence': float(sentence_confidences[i])} for i, proposal in enumerate(proposal_infos)]

        # jointly ranking by proposal score and sentence confidence
        result = sorted(result, key=getJointKey, reverse=True)
                    
        results[vid] = result

        count += 1

        if count >= eval_num:
            break

    out_data['results'] = results
    
    resFile = 'results/%d/temp_results.json'%options['train_id']
    root_folder = os.path.dirname(resFile)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    print('Saving result json file ...')
    with open(resFile, 'w') as fid:
        json.dump(out_data, fid)
    
    # Call evaluator
    
    resFile = 'results/%d/temp_results.json'%options['train_id']
    root_folder = os.path.dirname(resFile)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    evaluator = ANETcaptions(ground_truth_filenames=['densevid_eval-master/data/val_2.json'],
                             prediction_filename=resFile,
                             tious=options['tiou_measure'],
                             max_proposals=options['max_proposal_num'],
                             verbose=True)
    evaluator.evaluate()

    # Output the results
    for i, tiou in enumerate(options['tiou_measure']):
        print('-' * 80)
        print('tIoU: %.2f'%tiou)
        print('-' * 80)
        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            print('| %s: %2.4f'%(metric, 100*score))

    # Print the averages
    print('-' * 80)
    print('Average across all tIoUs')
    print('-' * 80)
    avg_scores = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        avg_score = 100 * sum(score) / float(len(score))
        avg_scores[metric] = avg_score
    
    # print output evaluation scores
    fid = open('results/%d/score_history.txt'%options['train_id'], 'a')
    for metric, score in avg_scores.items():
        print('%s: %.4f'%(metric, score))
        # also write to a temp file
        fid.write('%s: %.4f\n'%(metric, score))
    fid.write('\n')
    fid.close()

    combined_score = avg_scores['METEOR']
    
    return avg_scores, combined_score


if __name__ == "__main__":


    torch.autograd.set_detect_anomaly(True)
    #manage_dataset_and_models()
    cuda_flag = torch.cuda.is_available()
    print(cuda_flag)
    # read in params based on model from json file
    params = {
        "num_vids_per_epoch": 4200,
        "batch_size": 1,
        "epochs": 1,
        "num_workers": 1,
        "dataset_name": 'activitynet_captions',
        "use_sampler": False
    }

    options = default_options()

    # use GPU if available
    params["cuda"] = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(818976)
    if params['cuda']: torch.cuda.manual_seed(818976)
    dataloaders, vid_dl = fetch_dataloader(['val_2'], options, params)
    print(dataloaders)
    train_dl = dataloaders['val_2']

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = architecture(options)
    PATH = options['caption_checkpoint']
    checkpoint = torch.load(PATH, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.float()
    print("model to float")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("model device", str(device), str(torch.cuda.device_count()))
    model = model.to(device)
    print("model to gpu")
    model.train()

    model_caption = caption_architecture(options, device)
    model_caption.load_state_dict(checkpoint['model_caption_state_dict'])

    model_caption = model_caption.float()
    print("model_caption to float")

    model_caption = model_caption.to(device)
    print("model_caption to gpu")
    model_caption.eval()


    avg_scores, combined_score = evaluation_metric_greedy(model, model_caption, options, train_dl, vid_dl)
