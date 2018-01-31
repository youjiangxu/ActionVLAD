# cd ../
#LD_PRELOAD=/usr/lib/libtcmalloc.so.4 \
$(which python) \
  train_image_classifier.py \
  --batch_size 4 \
  --gpus 0,1,2,3 \
  --frames_per_video 20 \
  --iter_size 4 \
  --checkpoint_path models/Experiments/001_VGG_RGB_UCF_seqvlad_stage1_f25 \
  --checkpoint_style v2_withStream \
  --train_dir models/Experiments/001_VGG_RGB_UCF_seqvlad_stage2_f25 \
  --dataset_list_dir data/ucf101/train_test_lists/ \
  --dataset_dir data/ucf101/frames \
  --dataset_name ucf101 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 8 \
  --num_preprocessing_threads 8 \
  --learning_rate 0.0001 \
  --optimizer adam \
  --opt_epsilon 1e-4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling seqvlad \
  --netvlad_initCenters 64 \
  --num_steps_per_decay 10000 \
  --learning_rate_decay_factor 0.1 \
  --clip_gradients 5 \
  --num_streams 1 \
  --trainable_scopes stream0/classifier,stream0/SeqVLAD,stream0/vgg_16/conv5 \
  --train_image_size 224 \
  --weight_decay 4e-5 \
  --split_id 1 \
  --max_number_of_steps 15000
