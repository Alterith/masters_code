# relative path imports
import sys
# relative model imports
sys.path.insert(0, './models/')

# generic imports
from comet_ml import Experiment

from data_loader_classifier import video_dataset
import os
# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import StepLR


from torchsummary import summary
from tqdm import tqdm


from C3D import C3D_Architecture as architecture_c3d
from C3D_16 import C3D_Architecture as architecture_c3d_16
from C3D_Attn_8 import C3D_Architecture as architecture_c3d_a
from C3D_Attn_16 import C3D_Architecture as architecture_c3d_a_16
from mobileNetV3_3D_Attn import MobileNetV3 as architecture_mobilenet_a
from mobileNetV3_3D import MobileNetV3 as architecture_mobilenet

from opt import default_options


#from C3D_OLD import C3D_Architecture as old_architecture

class WithReplacementRandomSampler(Sampler):
    """Samples elements randomly, with replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.LongTensor(len(self.data_source))
        samples.random_(0, len(self.data_source))
        return iter(samples)

    def __len__(self):
        return len(self.data_source)

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
                    vid_dataset_sampler = WithReplacementRandomSampler(vid_dataset)

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
def val_model(model, val_loader):

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, file=sys.stdout)):
            a, b = data
            a = a.to(device)

            b = b.to(device)

            outputs = model(a)

            loss = criterion(outputs, b)
            running_loss += loss.item()
    model.train()
    return float(running_loss)/float(len(val_loader))

# Hyper_params

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    cuda_flag = torch.cuda.is_available()
    # read in params based on model from json file
    params = {
        "batch_size": options['batch_size'],
        "num_workers": options['batch_size'],
        "temporal_depth": options['temporal_depth'],
        "epochs": 1000,
        "learning_rate": 1e-2,
        "patch_width": options['patch_width'],
        "patch_height": options['patch_height'],
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
    torch.manual_seed(options["random_seed"])
    if params['cuda']: torch.cuda.manual_seed(options["random_seed"])
    dataloaders = fetch_dataloader(
        options["kinetics_data_path"]
        ,['train, val'], params)
    train_dl = dataloaders['train']
    train_dl = dataloaders['val']


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
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

    checkpoint = torch.load(PATH, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9, nesterov=True)

    scheduler_step = StepLR(optimizer, step_size = 1, gamma = 0.1)

    
    val_loss_old = 0.0
    val_loss = 0.0
    step_count = 0
    
    with experiment.train():
    
        for epoch in range(0, params['epochs']):
        
            if step_count <= 3:
            
                expected_loss = 0.0
                running_loss = 0.0
                total_batches = 0.0
                acc_1 = 0.0
                acc_5 = 0.0
                acc_10 = 0.0
                acc_20 = 0.0
                acc_50 = 0.0
                
                for i, data in enumerate(tqdm(train_dl, file=sys.stdout)):

                    optimizer.zero_grad()

                    a, b = data
 
                    a = a.to(device)
           
                    b = b.to(device)
            
                    outputs = model(a)
                 
                    loss = criterion(outputs, b)
                    
       
                    
      
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 30.)
                    optimizer.step()
        
                    acc = accuracy(outputs, b, (1, 5, 10, 20, 50))
                    acc_1 = acc_1 + acc[0]
                    acc_5 = acc_5 + acc[1]
                    acc_10 = acc_10 + acc[2]
                    acc_20 = acc_20 + acc[3]
                    acc_50 = acc_50 + acc[4]
  
                    expected_loss += loss.item()
                    running_loss += loss.item()
                    total_batches = total_batches + 1.0

                    # running loss intervals
                    div = 125
                    if i == 0:
                        div = 1
                    if i % div == 0 and i != 0:    # print every 1024 mini-batches

                        print('[%d, %d] E[loss]: %.20f loss: %.20f acc 1: %.20f acc 5: %.20f]' %(epoch, i, float(expected_loss)/(float(i+1)) , float(running_loss)/float(div), float(acc_1)/float(total_batches), float(acc_5)/float(total_batches)))
                        experiment.log_metric('Epoch', epoch)
                        experiment.log_metric('Training_iteration', i)
                        experiment.log_metric('E[loss]', float(expected_loss)/float(i+1))
                        experiment.log_metric('Running_loss', float(running_loss)/float(div))
                        experiment.log_metric('Acc@1', float(acc_1)/float(total_batches))
                        experiment.log_metric('Acc@5', float(acc_5)/float(total_batches))
                        experiment.log_metric('Acc@10', float(acc_10)/float(total_batches))
                        experiment.log_metric('Acc@20', float(acc_20)/float(total_batches))
                        experiment.log_metric('Acc@50', float(acc_50)/float(total_batches))
                        running_loss = 0.0
                        acc_1 = 0.0
                        acc_5 = 0.0
                        acc_10 = 0.0
                        acc_20 = 0.0
                        acc_50 = 0.0
                        total_batches = 0.0

            
            print("Saving Epoch")
            
            torch.save({
                'model_state_dict': model.state_dict()
            }, "./models/"+arch_name.upper()+".pt")
            experiment.log_asset("./models/"+arch_name.upper()+".pt")
            
            if epoch % 5 == 0:
            
                val_loss_old = val_loss
                val_loss = val_model(model, val_loader)
                if val_loss - val_loss_old < 1e-3:
                    scheduler_step.step()
                    step_count += 1

        print("End here")

