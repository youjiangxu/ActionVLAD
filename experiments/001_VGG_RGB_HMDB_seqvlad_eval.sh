#cd ../
#LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
python \
  eval_image_classifier.py \
  --gpus 1 \
  --batch_size 1 \
  --frames_per_video 25 \
  --checkpoint_path models/Experiments/001_VGG_RGB_HMDB_seqvlad_action_pretrained_f25_stage2 \
  --dataset_dir data/hmdb51/frames \
  --dataset_list_dir data/hmdb51/train_test_lists \
  --dataset_name hmdb51 \
  --dataset_split_name test \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling seqvlad \
  --netvlad_initCenters 64 \
  --classifier_type linear \
  --ncrops 5
