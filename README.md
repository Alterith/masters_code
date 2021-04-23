# Dense_Video_Captioning_Feature_Extraction_Model_Choice
# DenseVideoCaptioning

Pytorch Implementation of my masters code. The impact of encoded features on dense video captioning performance.

### Data Preparation

Please download the Kinetics-600 labels that can be found at [Kinetics](https://deepmind.com/research/open-source/kinetics)

Thereafter use the "masters_code/dataset/Kinetics/kinetics_368_classes.csv" file to extract the video IDs associated only with the 368 classes associated with the kinetics 368 dataset. Thereafter use the [kinetics-downloader](https://github.com/piaxar/kinetics-downloader) repo to download the videos and place them in train and val folders in the "masters_code/dataset/Kinetics/" folder respectively.

Our Caption video features are already generated and provided below as such there is no need to redownload the activitynet captions dataset.

Please download ActivityNet Captions annotation data from the website [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/).

We used the preprocessed data given by Jingwen Wang in their "Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning" paper code which was obtained from [here](https://github.com/JaywongWang/DenseVideoCaptioning). Please follow their steps and folder structure. 

### Hyper Parameters
The hyper parameters used are included in the opt.py file adapted from [Jingwen Wang's Dense video captioning implementation](https://github.com/JaywongWang/DenseVideoCaptioning)

### Training
Train the classification models by using the script train_classifier.py

Train the caption models by using the script train_captions.py
First pretrain the temporal action proposal module then jointly train it with the caption module 

### Prediction
To run inference on the classification model run valid_model.py
To run inference on the caption model run eval.py

### Pre-trained Models and features

The pretrained models and features can be found at the following link: To be provided at a later date when we have access to a cloud platform with enough storage space.

### Environment

The conda environment used during the course of this dissertation is provided in this repository in the msc_env.yml file.

