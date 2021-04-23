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
<!---
The configuration (from my experiments) is given in opt.py, including model setup, training options, and testing options. You may want to set max_proposal_num=1000 if saving valiation time is not the first priority.
--->
### Training
<!---
Train dense-captioning model using the script train.py.
First pre-train the proposal module (you may need to slightly modify the code to support batch size of 32, using batch size of 1 could lead to unsatisfactory performance). The pretrained proposal model can be found in https://drive.google.com/drive/folders/1IeKkuY3ApYe_QpFjarweRb2MTJKTCOLa. Then train the whole dense-captioning model by setting train_proposal=True and train_caption=True. To understand the proposal module, I refer you to the original [SST](http://openaccess.thecvf.com/content_cvpr_2017/papers/Buch_SST_Single-Stream_Temporal_CVPR_2017_paper.pdf) paper and also my tensorflow [implementation](https://github.com/JaywongWang/SST-Tensorflow) of SST.
--->
### Prediction
<!---
Follow the script test.py to make proposal predictions and to evaluate the predictions. Use max_proposal_num=1000 to generate .json test file and then use script "python2 evaluate.py -s [json_file] -ppv 100" to evaluate the performance (the joint ranking requres to drop items that are less confident).
--->

### Pre-trained Models and features

The pretrained models and features can be found at the following link: To be provided at a later date when we have access to a cloud platform with enough storage space.

### Environment

The conda environment used during the course of this dissertation is provided in this repository in the msc_env.yml file.

