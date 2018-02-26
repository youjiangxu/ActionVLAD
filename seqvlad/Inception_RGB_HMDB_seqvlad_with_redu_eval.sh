#cd ../
#LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
python \
  eval_image_classifier.py \
  --gpus 2 \
  --batch_size 1 \
  --frames_per_video 10 \
  --checkpoint_path models/Experiments/Inception_RGB_HMDB_seqvlad_stage2 \
  --dataset_dir data/hmdb51/frames \
  --dataset_list_dir data/hmdb51/train_test_lists \
  --dataset_name hmdb51 \
  --dataset_split_name test \
  --model_name inception_v3 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name inception \
  --bgr_flip True \
  --pooling seqvlad-with-redu \
  --redu_dim 1024 \
  --is_step False \
  --netvlad_initCenters 64 \
  --classifier_type linear \
  --ncrops 5
