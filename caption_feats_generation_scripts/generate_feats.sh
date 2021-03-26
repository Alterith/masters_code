#!/bin/bash
source /home/alterith/anaconda3/bin/activate dbhugwan_msc_env_kraken
cd /home/alterith/Documents/Dense_Video_Captioning_Feature_Extraction_Model_Choice
base_file_name="classifier_name_feats"
ext=".h5py"
file_name="$base_file_name$ext"
for i in {0..90};
    do
        echo $i
        #python generate_4k_feats.py 1 $i $file_name
        if [ "$i" -eq "0" ]
        then
            #echo "0"
            python generate_classifier_feats.py 1 $i $file_name
        else
            #echo "1"
            python generate_classifier_feats.py 0 $i $file_name
        fi
    done
