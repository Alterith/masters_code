import warnings
warnings.filterwarnings("ignore")

# relative path imports
import sys
# relative model imports
sys.path.insert(0, '../models/c3d/attn/')
sys.path.insert(0, '../models/c3d/normal/')
sys.path.insert(0, '../models/r2p1d/attn/')
sys.path.insert(0, '../models/r2p1d/normal/')

# generic imports

from full_vid_data_loader_decord import video_dataset
import os
import h5py
# torch imports
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from tqdm import tqdm
import gc

# dirty fix to train model
#from C3D_500_feats_16 import C3D_Architecture as architecture
from C3D_Attn import C3D_Architecture as architecture
arch_name = "C3D_ATTN"

torch.autograd.set_detect_anomaly(True)

def fetch_dataloader(data_dir, types, params, stride_idx):
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
    for split in ['train', 'val', 'val_1', 'val_2', 'test']:

        if split in types:
            #path = os.path.join(data_dir, split)
            # use the train_transformer if training data, else use eval_transformer without random flip

            vid_dataset = video_dataset(data_dir, split, temporal_depth=params['temporal_depth'], patch_width=params['patch_width'], patch_height=params['patch_height'], dataset_name = params['dataset_name'], stride=params['stride'], stride_idx=stride_idx)
            if split == 'train':
                if params['use_sampler'] == True:
                    vid_dataset_sampler = RandomSampler(vid_dataset, replacement=True, num_samples=params['num_vids_per_epoch'])
                else:
                    vid_dataset_sampler = None

                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=False,
                                num_workers=params['num_workers'], sampler = vid_dataset_sampler)
            else:
                dl = DataLoader(vid_dataset, batch_size=params['batch_size'], shuffle=False,
                                num_workers=params['num_workers'],
                                pin_memory=params['cuda'])

            dataloaders[split] = dl
    return dataloaders



# Hyper_params

## TODO: for loop through model, raad up on how to validate model n validation set, use model.eval?

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    #manage_dataset_and_models()
    cuda_flag = torch.cuda.is_available()
    # read in params based on model from json file
    params = {
        "num_vids_per_epoch": 1,
        "batch_size": 1,
        "num_workers": 1,
        "temporal_depth": 16,
        "patch_width": 112,
        "patch_height": 112,
        "dataset_name": 'activitynet_captions',
        "use_sampler": False,
        "stride": 100
    }


    # use GPU if available
    params["cuda"] = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(818976)
    if params['cuda']: torch.cuda.manual_seed(818976)
    dataloaders = fetch_dataloader(
        #"/media/alterith/SR6/Data/kinetics_600/"
        #"/media/alterith/Elements/activitynet_captions/activitynet_captions/"
        "/media/alterith/Elements/activitynet_captions/activitynet_captions/"
        ,['val_2'], params, stride_idx=int(sys.argv[2]))
    print(dataloaders)
    val_dl = dataloaders['val_2']

    # hopeful optimization
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    model = architecture(num_classes=368)
    #PATH = "C3D_69_kraken.pt"
    PATH= "C3D_Attn_368_sgd_18.pt"
    checkpoint = torch.load(PATH, map_location="cuda:0")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.float()
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dl, file=sys.stdout)):
            gc.collect()

            if i >= 0:

                

                frames, name = data

                # quit()
                
                print(frames.shape)

                if frames.shape[1] <= 450:
                    name = name[0].split("/")[-1][:-4]
                    name = name.encode("ascii", "ignore")
                    print(name)
                    for j in range(0, frames.shape[1]):
                    
                        a = frames[:, j, :, :, :, :].to(device)

                        prev_layer_output = model(a)

                        # print(output.shape)
                        # print(prev_layer_output.shape)
                        # quit()

                        prev_layer_output = prev_layer_output.squeeze(-1)

                        if i == 0 and j == 0 and int(sys.argv[1]) == 1:
                            #h = h5py.File('4096_feats_C3D_69_stride_16_val_1.hdf5', "a")
                            h = h5py.File(sys.argv[3], "a")

                            # specify none to extend indefinately
                            h.create_dataset('feats', data=prev_layer_output.cpu().detach().numpy(), maxshape=(None, 1280), chunks=True)

                            dt = h5py.special_dtype(vlen=str)
                            h.create_dataset('name', data=[[name]], dtype=dt, maxshape=(None, 25), chunks=True)
                            
                            h.close()

                        else:
                            #h = h5py.File('4096_feats_C3D_69_stride_16_val_1.hdf5', "a")
                            h = h5py.File(sys.argv[3], "a")

                            h["feats"].resize((h["feats"].shape[0] + 1), axis = 0)

                            # place data at end
                            h["feats"][-1:] = prev_layer_output.cpu().detach().numpy()

                            h["name"].resize((h["name"].shape[0] + 1), axis = 0)

                            # place data at end
                            h["name"][-1:] = [[name]]
                            h.close()
                        a = a.cpu()
                        del a
                
                    frames = frames.cpu()
                    del frames
                    #h.close()  
