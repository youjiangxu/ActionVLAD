# cd ..
#LD_PRELOAD=/usr/lib/libtcmalloc.so.4 \
python \
  train_image_classifier.py \
  --batch_size 4 \
  --gpus 0,1,2,3 \
  --frames_per_video 25 \
  --iter_size 3 \
  --checkpoint_path models/PreTrained/ActionVLAD-pretrained/ucf101/rgb/split1.ckpt \
  --checkpoint_style v2_withStream \
  --train_dir models/Experiments/001_VGG_RGB_UCF_seqvlad_action_pretrained_f25_stage1 \
  --dataset_list_dir data/ucf101/train_test_lists/ \
  --dataset_dir data/ucf101/frames \
  --dataset_name ucf101 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 8 \
  --num_preprocessing_threads 8 \
  --learning_rate 0.01 \
  --optimizer adam \
  --opt_epsilon 1e-4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling seqvlad \
  --netvlad_initCenters 64 \
  --pooled_dropout 0.5 \
  --num_steps_per_decay 3000 \
  --learning_rate_decay_factor 0.1 \
  --clip_gradients 5 \
  --num_streams 1 \
  --trainable_scopes stream0/SeqVLAD,stream0/classifier \
  --checkpoint_exclude_scopes stream0/classifier,stream0/SeqVLAD \
  --train_image_size 224 \
  --weight_decay 4e-5 \
  --split_id 1 \
  --max_number_of_steps 6000
