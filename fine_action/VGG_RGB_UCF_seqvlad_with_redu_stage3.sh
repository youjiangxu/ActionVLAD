# cd ..
#LD_PRELOAD=/usr/lib/libtcmalloc.so.4 \
python \
  train_image_classifier.py \
  --batch_size 4 \
  --gpus 0,1,2,3 \
  --frames_per_video 10 \
  --iter_size 2 \
  --checkpoint_path models/Experiments/VGG_RGB_UCF_seqvlad_with_redu_pretrained_stage2_1 \
  --checkpoint_style v2_withStream \
  --train_dir models/Experiments/VGG_RGB_UCF_seqvlad_with_redu_pretrained_stage3_3 \
  --dataset_list_dir data/ucf101/train_test_lists/ \
  --dataset_dir data/ucf101/frames \
  --dataset_name ucf101 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --learning_rate 0.0001 \
  --optimizer adam \
  --opt_epsilon 1e-4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling seqvlad-with-redu \
  --redu_dim 256 \
  --is_step True \
  --netvlad_initCenters models/Experiments/001_VGG_RGB_HMDB_netvlad_stage1/Features/imnet_conv5_kmeans64.pkl \
  --pooled_dropout 0.8 \
  --num_steps_per_decay 3000 \
  --learning_rate_decay_factor 0.1 \
  --clip_gradients 1 \
  --num_streams 1 \
  --trainable_scopes stream0/SeqVLAD,stream0/classifier,stream0/vgg_16/conv5 \
  --train_image_size 224 \
  --weight_decay 4e-5 \
  --split_id 1 \
  --max_number_of_steps 8000
#  --checkpoint_style v2_withStream \
