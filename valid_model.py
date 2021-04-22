# relative path imports
import sys
# relative model imports
sys.path.insert(0, './models/')

# generic imports
from comet_ml import Experiment

from valid_loader import video_dataset
import os
# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from torchsummary import summary
from tqdm import tqdm

from C3D import C3D_Architecture as architecture_c3d
from C3D_16 import C3D_Architecture as architecture_c3d_16
from C3D_Attn_8 import C3D_Architecture as architecture_c3d_a
from C3D_Attn_16 import C3D_Architecture as architecture_c3d_a_16
from mobileNetV3_3D_Attn import MobileNetV3 as architecture_mobilenet_a
from mobileNetV3_3D import MobileNetV3 as architecture_mobilenet

from opt import default_options

torch.autograd.set_detect_anomaly(True)


def fetch_dataloader(data_dir, types, params):
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
    for split in ['train', 'val', 'test']:

        if split in types:
            #path = os.path.join(data_dir, split)
            # use the train_transformer if training data, else use eval_transformer without random flip

            vid_dataset = video_dataset(data_dir, split, temporal_depth=params['temporal_depth'], patch_width=params['patch_width'], patch_height=params['patch_height'], dataset_name = params['dataset_name'])
            if split == 'train':
                if params['use_sampler'] == True:
                    vid_dataset_sampler = RandomSampler(vid_dataset, replacement=True, num_samples=params['num_vids_per_epoch'])
                else:
                    vid_dataset_sampler = None

                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=False,
                                num_workers=params['num_workers'], sampler = vid_dataset_sampler)
            else:
                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=True,
                                num_workers=params['num_workers'],
                                pin_memory=params['cuda'])

            dataloaders[split] = dl
    return dataloaders

# taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Hyper_params

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    #manage_dataset_and_models()
    cuda_flag = torch.cuda.is_available()
    # read in params based on model from json file
    params = {
        "batch_size": 1,
        "num_workers": 1,
        "temporal_depth": options['temporal_depth'],
        "patch_width": 112,
        "patch_height": 112,
        "dataset_name": 'kinetics_368',
        "use_sampler": False
    }
    
    options = default_options(sys.argv[1])
    
    arch_name = options["arch_name"]

    experiment = Experiment(api_key=options["experiment_key"], project_name=options["project_name"], workspace=options["experiment_workspace"])
    experiment.set_name(options["experiment_name"])
    experiment.log_parameters(params)
    # use GPU if available
    params["cuda"] = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(818976)
    if params['cuda']: torch.cuda.manual_seed(818976)
    dataloaders = fetch_dataloader(
        #"/media/alterith/SR6/Data/kinetics_600/"
        options["kinetics_data_path"]
        ,['test'], params)
    val_dl = dataloaders['test']

    # hopeful optimization
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with experiment.validate():
        with torch.no_grad():
            model = None
            if arch_name == "c3d_8":
                model = architecture_c3d()
            elif arch_name == "c3d_16":
                model = architecture_c3d_16()
            elif arch_name == "c3d_attn_8":
                model = architecture_c3d_a()
            elif arch_name == "c3d_attn_16":
                model = architecture_c3d_a_16()
            elif arch_name == "mobilenet":
                model = architecture_mobilenet()
            elif arch_name == "mobilenet_attn":
                model = architecture_mobilenet_a()
            PATH = options["classifier_checkpoint"]
            checkpoint = torch.load(PATH, map_location="cuda:1")
            model.load_state_dict(checkpoint["model_state_dict"])


            model = model.float()
 
            model.eval()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            total_batches = 0.0
            acc_1 = 0.0
            acc_5 = 0.0
            total_acc_1 = 0.0
            total_acc_5 = 0.0
            for i, data in enumerate(tqdm(val_dl, file=sys.stdout)):

                frames, label = data
                
                # print(frames.shape)

                # print(name)

                output = None

                num_clips = frames.shape[1]

                for j in range(0, num_clips):
                
                    a = frames[:, j, :, :, :, :].to(device)

                    if j == 0:

                        output = model(a)

                    else:
                        output += model(a)
                # frames = frames.to(device)
                # output = model(frames)
                label = label.to(device)

                output = output / float(num_clips)

                acc = accuracy(output, label, (1, 5))
                acc_1 = acc_1 + acc[0]
                acc_5 = acc_5 + acc[1]
                total_acc_1 += acc[0]
                total_acc_5 += acc[1]
                #print("accuracy, %f, %f" %(acc[0], acc[1]))
                #print(loss.item())
                total_batches = total_batches + 1.0

                # running loss intervals
                div = 125
                if i == 0:
                    div = 1
                if i % div == 0 and i != 0:    # print every 125 mini-batches
                    print('[%d] acc 1: %.20f acc 5: %.20f]' %(i, float(acc_1)/float(total_batches), float(acc_5)/float(total_batches)))
                    experiment.log_metric('Acc@1', float(acc_1)/float(total_batches))
                    experiment.log_metric('Acc@5', float(acc_5)/float(total_batches))
                    total_batches = 0.0
                    acc_1 = 0.0
                    acc_5 = 0.0

            experiment.log_metric('Final_Acc@1', float(total_acc_1)/float(len(val_dl)))
            experiment.log_metric('Final_Acc@5', float(total_acc_5)/float(len(val_dl)))



