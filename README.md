# Dense_Video_Captioning_Feature_Extraction_Model_Choice
# DenseVideoCaptioning

Pytorch Implementation of my masters code. The impact of encoded features on dense video captioning performance.

### Data Preparation
<!---
Please download annotation data and C3D features from the website [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/). The ActivityNet C3D features with stride of 64 frames (used in my paper) can be found in [https://drive.google.com/open?id=1UquwlUXibq-RERE8UO4_vSTf5IX67JhW](https://drive.google.com/open?id=1UquwlUXibq-RERE8UO4_vSTf5IX67JhW).
Please follow the script dataset/ActivityNet_Captions/preprocess/anchors/get_anchors.py to obtain clustered anchors and their pos/neg weights (for handling imbalance class problem). I already put the generated files in dataset/ActivityNet_Captions/preprocess/anchors/.
Please follow the script dataset/ActivityNet_Captions/preprocess/build_vocab.py to build word dictionary and to build train/val/test encoded sentence data.
--->
### Hyper Parameters
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

