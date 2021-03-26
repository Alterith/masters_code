from comet_ml import Experiment

sys.path.insert(0, './models/')

# relative path imports
import sys
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from data_provider_captions import DataProvision
import os

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import torch.nn.init as init

from focal_loss import FocalLoss

from tqdm import tqdm

# dirty fix to train model
from temporal_proposals import Temporal_Proposal_Architecture as architecture
from caption_module import Caption_Module_Architecture as caption_architecture
arch_name = "Captions"

from opt import default_options

import linecache

torch.autograd.set_detect_anomaly(True)



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
    for split in ['train', 'val_1', 'val_2']:

        if split in types:

            vid_dataset = DataProvision(opt, split)
            if split == 'train':
                if params['use_sampler'] == True:
                    vid_dataset_sampler = RandomSampler(vid_dataset, replacement=True, num_samples=params['num_vids_per_epoch'])
                else:
                    vid_dataset_sampler = None

                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=True,
                                num_workers=params['num_workers'], sampler = vid_dataset_sampler)
            else:
                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=True,
                                num_workers=params['num_workers'],
                                pin_memory=params['cuda'])

            dataloaders[split] = dl
    return dataloaders

# taken from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        init.torch.nn.init.uniform_(m.weight.data, -0.08, 0.08)


def calc_proposal_weights(target_tensor):
    """
    calculates the binary weights for a given tensor for each dimension along axis 0
    Args:
        target_tensor: (torch.Tensor) is the tensor for which we calculate the binary weights [batch, num_points]
    Returns:
        pos_weights: (torch.Tensor) contains the pos_class weights for bcewithlogitsloss [1, num_points]
    """
    pos_weights = None
    gt_unique_bins = None
    gt_cat_counts = None
    for i in range(0, target_tensor.shape[1]):
        gt_unique, gt_counts = torch.unique(target_tensor[:,i], return_counts=True)

        # if 1 isnt in a cluster append it with 0 count to the counts and bins
        if not 1 in gt_unique:
            gt_unique = torch.cat((gt_unique, torch.Tensor([1])), dim = 0)
            gt_counts = torch.cat((gt_counts, torch.Tensor([0])), dim = 0)

        # if 0 isnt in a cluster append it with 0 count to the counts and bins
        if not 0 in gt_unique:
            gt_unique = torch.cat((torch.Tensor([0]), gt_unique), dim = 0)
            gt_counts = torch.cat((torch.Tensor([0]), gt_counts), dim = 0)

        if i == 0:
            gt_unique_bins = gt_unique
            gt_cat_counts = gt_counts.unsqueeze(0)
        else:
            gt_counts = gt_counts.unsqueeze(0)
            gt_cat_counts = torch.cat((gt_cat_counts, gt_counts), dim=0)

    # add epsilon for non-zero division
    gt_cat_counts = gt_cat_counts.float()
    gt_cat_counts[:,1] += 1e-3
    # calculate positive weights neg_total/pos_total
    pos_weights = torch.div(gt_cat_counts[:,0], gt_cat_counts[:,1])

    return pos_weights.cpu().detach().numpy()



def calc_caption_weights(target_tensor, target_mask, end_id, options):
    """
    calculates the weights for a given tensor for input into crossentropy loss
    Args:
        target_tensor: (torch.Tensor) is the tensor for which we calculate the weights [1,1,num_features, 30]
        target_mask: (torch.Tensor) is the tensor for which we calculate target tensor mask [1,1,num_features, 30]
        end_id: (torch.Tensor) is the tensor which contains the end id's for captions
        options: (dict) are the default options used
    Returns:
        weights: (torch.Tensor) contains the class weights for crossentropyloss [1, len(vocab)]
    """
    # create torch tensor with len of vocab and fill with initial value
    weights = torch.empty(options['vocab_size']).fill_(0.25)
    unique_words = None
    target_masked = []
    for i in range(0, end_id.shape[0]):

        target_masked.append(target_tensor[0][0][end_id][i][torch.nonzero(target_mask[0][0][end_id][i]).view(-1)][1:].tolist())

    # get the unique words after mask is applied
    flattened_target_masked = [item for sublist in target_masked for item in sublist]
    flattened_target_masked = np.asarray(flattened_target_masked)
    
    unique_words, unique_counts = torch.unique(flattened_target_masked, return_counts=True)

    unique_weights = [float(sum(unique_counts))/float(len(unique_counts) * cnt) for cnt in unique_counts]

    # assign weights to the unique words
    for idx in unique_words:
        weights[idx] = unique_weights[idx]

    return weights.cpu().detach().numpy()


def val_model(model_temporal, model_caption, val_loader):

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, file=sys.stdout)):
            data_dict = data

            vid_fw = data_dict["video_feat_fw"].to(device)
            vid_bw = data_dict["video_feat_bw"].to(device)
            pos_weight = None


            pos_weight_fw = calc_proposal_weights(data_dict["proposal_fw"][0])
            pos_weight_bw = calc_proposal_weights(data_dict["proposal_bw"][0])
            pos_weight = pos_weight_fw + pos_weight_bw
            pos_weight = pos_weight / 2.0
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
                
                

            output_fw, output_bw, hs_fw, hs_bw = model(vid_fw, vid_bw)


            proposal_loss = criterion(output_fw, data_dict["proposal_fw"][0].type(torch.float).to(device))

            proposal_loss += criterion(output_bw, data_dict["proposal_bw"][0].type(torch.float).to(device))
            
            temporal_loss += proposal_loss.item()
            
            
            sentences, fc_word_scores, start_id, end_id = model_caption(vid_fw, hs_fw, hs_bw, data_dict["proposal_caption_fw"], data_dict["proposal_caption_bw"])
            caption_weights = None
            
            try:
                caption_weights = calc_caption_weights(data_dict["caption"], data_dict["caption_mask"], end_id, options)
                caption_weights = torch.FloatTensor(caption_weights)
            except Exception as e:
                caption_weights = torch.ones(options['vocab_size'])
                
            caption_weights = caption_weights.to(device)
                
            criterion_caption = FocalLoss(alpha = caption_weights, gamma = 5., reduction="none")

            #loss = criterion_caption(fc_word_scores[0], data_dict["caption"][0][0][end_id][0].type(torch.long).to(device))
            capt_loss = torch.sum(criterion_caption(fc_word_scores[0], data_dict["caption"][0][0][end_id][0].type(torch.long).to(device))*data_dict["caption_mask"][0][0][end_id][0].type(torch.float32).to(device))


            for capt_idx in range(1, fc_word_scores.shape[0]):

                capt_loss += torch.sum(criterion_caption(fc_word_scores[capt_idx], data_dict["caption"][0][0][end_id][capt_idx].type(torch.long).to(device))*data_dict["caption_mask"][0][0][end_id][capt_idx].type(torch.float32).to(device))
            # subtract number of start words from the total as its loss is 0 
            capt_loss = capt_loss / float(data_dict["caption_mask"].sum() - fc_word_scores.shape[0] + 1e-8)

            print(capt_loss)
            
            caption_loss += capt_loss.item()
            print(caption_loss)
            
            total_loss = proposal_loss + options['weight_caption']*capt_loss
            
    model.train()
    return float(total_loss)/float(len(val_loader))



if __name__ == "__main__":


    experiment = Experiment(api_key="GEXoVCGs96Ut6MDUDKWqvcWGw", project_name="general", workspace="alterith")
    
    experiment.set_name("C3D_16_Captions_CHPC_TEMP_ADAM_FL_WEIGHT_2_FIX_08_02_2021")

    torch.autograd.set_detect_anomaly(True)
    #manage_dataset_and_models()
    cuda_flag = torch.cuda.is_available()
    print(cuda_flag)
    # read in params based on model from json file
    params = {
        "num_vids_per_epoch": 80000,
        "batch_size": 1,
        "epochs": 100,
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
    dataloaders = fetch_dataloader(['train'], options, params)
    print(dataloaders)
    train_dl = dataloaders['train']

    # hopeful optimization

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = architecture(options)
    model.apply(weight_init)

    model = model.float()
    print("model to float")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("model device", str(device), str(torch.cuda.device_count()))
    model = model.to(device)
    print("model to gpu")
    model.train()

    model_caption = caption_architecture(options, device)
    model_caption.apply(weight_init)

    model_caption = model_caption.float()
    print("model_caption to float")

    model_caption = model_caption.to(device)
    print("model_caption to gpu")
    model_caption.train()


    word_embedding = {n for n, m in model_caption.named_modules() if isinstance(m, torch.nn.Embedding)}

    word_embedding_param_names = {n for n, _ in model_caption.named_parameters() if n.rsplit('.', 1)[0] in word_embedding}

    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters()], 'weight_decay':1e-6},
        {'params': [p for n, p in model_caption.named_parameters() if n not in word_embedding_param_names], 'weight_decay':1e-6},
        {'params': [p for n, p in model_caption.named_parameters() if n in word_embedding_param_names], 'weight_decay': 0.0}
    ], lr=1e-3)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_bak = checkpoint['epoch'] + 1

    
    experiment.log_text("MOBILENET captions")
    
    val_loss_old = 0.0
    val_loss = 0.0
    step_count = 0

    with experiment.train():
        for epoch in range(0, params['epochs']):
            if step_count <= 3:
                temporal_loss = 0.0
                caption_loss = 0.0
                running_loss = 0.0
                total_batches = 0.0

                for i, data in enumerate(tqdm(train_dl, file=sys.stdout)):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    data_dict = data

                    vid_fw = data_dict["video_feat_fw"].to(device)
                    vid_bw = data_dict["video_feat_bw"].to(device)
                    pos_weight = None

                    with torch.no_grad():
                        pos_weight_fw = calc_proposal_weights(data_dict["proposal_fw"][0])
                        pos_weight_bw = calc_proposal_weights(data_dict["proposal_bw"][0])
                        pos_weight = pos_weight_fw + pos_weight_bw
                        pos_weight = pos_weight / 2.0
                        pos_weight = torch.from_numpy(pos_weight)
                        pos_weight = pos_weight.to(device)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
                        
                        

                    output_fw, output_bw, hs_fw, hs_bw = model(vid_fw, vid_bw)


                    proposal_loss = criterion(output_fw, data_dict["proposal_fw"][0].type(torch.float).to(device))

                    proposal_loss += criterion(output_bw, data_dict["proposal_bw"][0].type(torch.float).to(device))
                    
                    temporal_loss += proposal_loss.item() 


                    sentences, fc_word_scores, start_id, end_id = model_caption(vid_fw, hs_fw, hs_bw, data_dict["proposal_caption_fw"], data_dict["proposal_caption_bw"])
                    caption_weights = None
                    with torch.no_grad():
                        try:
                            caption_weights = calc_caption_weights_2(data_dict["caption"], data_dict["caption_mask"], end_id, options)
                            caption_weights = torch.FloatTensor(caption_weights)
                        except Exception as e:
                            caption_weights = torch.ones(options['vocab_size'])
                            
                        caption_weights = caption_weights.to(device)
                        
                    criterion_caption = FocalLoss(alpha = caption_weights, gamma = 5., reduction="none")


                    capt_loss = torch.sum(criterion_caption(fc_word_scores[0], data_dict["caption"][0][0][end_id][0].type(torch.long).to(device))*data_dict["caption_mask"][0][0][end_id][0].type(torch.float32).to(device))


                    for capt_idx in range(1, fc_word_scores.shape[0]):
 
                        capt_loss += torch.sum(criterion_caption(fc_word_scores[capt_idx], data_dict["caption"][0][0][end_id][capt_idx].type(torch.long).to(device))*data_dict["caption_mask"][0][0][end_id][capt_idx].type(torch.float32).to(device))
                    # subtract number of start words from the total as its loss is 0 
                    capt_loss = capt_loss / float(data_dict["caption_mask"].sum() - fc_word_scores.shape[0] + 1e-8)

                    print(capt_loss)
                    
                    caption_loss += capt_loss.item()
                    print(caption_loss)
                    
                    total_loss = proposal_loss + options['weight_caption']*capt_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options['clip_gradient_norm'])
                    torch.nn.utils.clip_grad_norm_(model_caption.parameters(), options['clip_gradient_norm'])
                    optimizer.step()

                    N,C = output_fw.shape



                    running_loss += total_loss.item()
                    #torch.cuda.empty_cache()

                div = len(train_dl)


                experiment.log_metric('Epoch', epoch)
                experiment.log_metric('Running_loss', running_loss/float(div))
                experiment.log_metric('Temporal_loss', temporal_loss/float(div))
                experiment.log_metric('Caption_loss', caption_loss/float(div))

                torch.save({
                    'model_state_dict': model.state_dict(),
                }, "full_caption_C3D_16_"+str(epoch)+".pt")
                experiment.log_asset("full_caption_C3D_16_"+str(epoch)+".pt")
                
            if epoch % 5 == 0:
            
                val_loss_old = val_loss
                val_loss = val_model(model, val_loader)
                if val_loss - val_loss_old < 1e-4:
                    step_count += 1
