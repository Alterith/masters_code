"""
Configurations: including model configuration and hyper parameter setting
"""

from collections import OrderedDict
import json
import os
# consider inputs to the function so we can change the feature data paths easily
def default_options(base_arch = 'c3d_8'):

    valid_arch = {'c3d_8', 'c3d_16', 'c3d_attn_8', 'c3d_attn_16', 'mobilenet', 'mobilenet_attn'}
    # enforce that the correct arch name is chosen
    if base_arch not in valid_arch:
        raise ValueError("base_arch must be one of %r." % valid_arch)

    options = OrderedDict()

    ### DATA
    options['classification_dataset_name'] = "kinetics_368"
    options['kinetics_data_path'] = './dataset/Kinetics'
    options['kinetics_368_class_list'] = os.path.join(options['kinetics_data_path'], 'kinetics_368_classes.csv')
    options['kinetics_368_class_order'] = os.path.join(options['kinetics_data_path'], 'kinectics_368_order.txt')

    options['base_arch'] = base_arch
    options['feature_data_path_train'] = './caption_features/'+options['base_arch']+'/500_feats_'+options['base_arch']+'_stride_64_train.hdf5'
    options['feature_data_path_val_1'] = './caption_features/'+options['base_arch']+'/500_feats_'+options['base_arch']+'_stride_64_val_1.hdf5'
    options['feature_data_path_val_2'] = './caption_features/'+options['base_arch']+'/500_feats_'+options['base_arch']+'_stride_64_val_2.hdf5'

    options['localization_data_path'] = './dataset/ActivityNet_Captions'
    options['caption_data_root'] = './dataset/ActivityNet_Captions/preprocess'
    options['vocab_file'] = os.path.join(options['caption_data_root'], 'word2id.json')
    options['vocab'] = json.load(open(options['vocab_file']))   # dictionary: word to word_id
    options['vocab_size'] = len(options['vocab'])               # number of words
    options['vocab_counts'] = os.path.join(options['caption_data_root'], 'vocabulary.txt')

    
    options['random_seed'] = 818976         # random seed

    ### DATA CREATION SCRIPTS
    #TODO: use if statement for 8 or 16 frame config, use sys args for out file name
    options['Generate_Classifier_Feats'] = './caption_feats_generation_scripts/'
    options['Generate_PCA_Feats'] = './caption_feats_generation_scripts/'
    options['Reduce_PCA_Feats'] = './caption_feats_generation_scripts/'

    ### MODEL PATHS
    ## CLASSIFICATION MODEL PATHS
    options['classifier_checkpoint_path'] = './classification_model_checkpoints'

    options['classifier_checkpoint'] = os.path.join(options['classifier_checkpoint_path'], options['base_arch'].upper() + '.pt')
    

    options['load_classifier_checkpoint'] = True

    ## CAPTION MODEL PATHS
    options['caption_checkpoint_path'] = './caption_model_checkpoints'
    options['caption_prefix'] = 'CAPTION_'
    options['caption_checkpoint'] = os.path.join(options['caption_checkpoint_path'], options['caption_prefix'] + options['base_arch'].upper())

    ### MODEL CONFIG
    options['patch_width'] = 112            # video patch width
    options['patch_height'] = 112           # video patch height
                                            # video temporal depth
    if options['base_arch'] == 'c3d_8' or options['base_arch'] == 'c3d_attn_8':
        options['temporal_depth'] = 8
    else:
        options['temporal_depth'] = 16

    options['video_feat_dim'] = 500         # dim of image feature
    options['encoded_video_feat_dim'] = 512 # should be equal to rnn size
    options['word_embed_size'] = 512        # size of word embedding
    options['caption_seq_len'] = 30         # maximu length of a sentence
    options['num_rnn_layers'] = 2           # number of RNN layers, only used for captioning network
    options['rnn_size'] = 512               # hidden neuron size
    options['rnn_drop'] = 0.3               # rnn dropout
    options['num_anchors'] = 120            # number of anchors  
    options['max_proposal_len'] = 110       # max length of proposal allowed, used to construct a fixed length tensor for all proposals from one video
    options['attention_hidden_size'] = 512  # size of hidden neuron for the attention hidden layer

    
    ### OPTIMIZATION
    options['optimizer'] = 'adam'           # 'adam','rmsprop','sgd_nestreov_momentum'
    options['momentum'] =  0.9              # only valid when solver is set to momentum optimizer
    options['batch_size'] = 16               # set to 1 to avoid different proposals problem, note that current implementation only supports batch_size=1
    options['eval_batch_size'] = 1
    options['loss_eval_num'] = 1000         # maximum evaluation batch number for loss
    options['metric_eval_num'] = 1000       # evaluation batch number for metric
    options['learning_rate'] = 1e-3         # initial learning rate
    options['lr_decay_factor'] = 0.01       # learning rate decay factor
    options['n_epoch_to_decay'] = list(range(20,80,20))[::-1]
    #TODO: Clean up training script for this
    options['auto_lr_decay'] = True         # whether automatically decay learning rate based on val loss or evaluation score (only when evaluation_metric is True)
    options['n_eval_observe'] = 10          # if after 5 evaluations, the val loss is still not lower, go back to change learning rate 
    options['min_lr'] = 1e-5                # minimum learning rate allowed
    options['weight_decay'] = 1e-6          # regularization strength
    options['init_scale'] = 0.08            # the init scale for uniform, here for initializing word embedding matrix
    options['max_epochs'] = 100             # maximum epochs
    options['init_epoch'] = 0               # initial epoch (useful when starting from last checkpoint)
    options['n_eval_per_epoch'] = 1         # number of evaluations per epoch
    options['eval_init'] = True             # evaluate the initialized model
    options['shuffle'] = True
    options['train_use_sampler'] = False
    options['val_use_sampler'] = True

    options['clip_gradient_norm'] = 100.     # threshold to clip gradients: avoid gradient exploding problem; set to -1 to avoid gradient clipping
    options['weight_proposal'] = 1.0        # contribution weight of proposal module
    options['weight_caption'] = 5.0         # contribution weight of captioning module
    options['proposal_tiou_threshold'] = 0.5   # tiou threshold to positive samples, when changed, calculate class weights for positive/negative class again
    options['caption_tiou_threshold'] = 0.8    # tiou threshold to select high-iou proposals to feed in the captioning module
    options['predict_score_threshold'] = 0.5 # score threshold to select proposals at test time
    options['train_classifier'] = True      # whether to train variables of proposal module
    options['train_proposal'] = True        # whether to train variables of proposal module
    options['train_caption'] = True         # whether to train variables of captioning module
    options['evaluate_metric'] = True       # whether refer to evalutaion metric (CIDEr, METEOR, ...) for optimization


    ### INFERENCE
    options['tiou_measure'] = [0.3, 0.5, 0.7, 0.9]

    return options
