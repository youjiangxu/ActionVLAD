# cd ..
#LD_PRELOAD=/usr/lib/libtcmalloc.so.4 \
python \
  train_image_classifier.py \
  --batch_size 4 \
  --gpus 2,3 \
  --frames_per_video 10 \
  --iter_size 4 \
  --checkpoint_path models/Experiments/Incpetion_RGB_HMDB_seqvlad_stage1 \
  --train_dir models/Experiments/Inception_RGB_HMDB_seqvlad_stage2 \
  --checkpoint_style v2_withStream \
  --dataset_list_dir data/hmdb51/train_test_lists/ \
  --dataset_dir data/hmdb51/frames \
  --dataset_name hmdb51 \
  --dataset_split_name train \
  --model_name inception_v3 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --learning_rate 0.0001 \
  --optimizer adam \
  --opt_epsilon 1e-4 \
  --preprocessing_name inception \
  --bgr_flip True \
  --pooling seqvlad-with-redu \
  --redu_dim 1024 \
  --is_step True \
  --netvlad_initCenters models/Experiments/001_VGG_RGB_HMDB_netvlad_stage1/Features/imnet_conv5_kmeans64.pkl \
  --pooled_dropout 0.5 \
  --num_steps_per_decay 5000 \
  --learning_rate_decay_factor 0.1 \
  --clip_gradients 5 \
  --num_streams 1 \
  --trainable_scopes stream0/SeqVLAD,stream0/classifier,stream0/InceptionV3/Mixed_7c \
  --train_image_size 224 \
  --weight_decay 4e-5 \
  --split_id 1 \
  --max_number_of_steps 8000
