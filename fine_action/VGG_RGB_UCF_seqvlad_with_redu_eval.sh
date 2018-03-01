#cd ../
#LD_PRELOAD=/usr/lib64/libtcmalloc.so.4 \
python \
  eval_image_classifier.py \
  --gpus 2 \
  --batch_size 1 \
  --frames_per_video 10 \
  --checkpoint_path models/Experiments/VGG_RGB_UCF_seqvlad_with_redu_stage2 \
  --dataset_dir data/ucf101/frames \
  --dataset_list_dir data/ucf101/train_test_lists \
  --dataset_name ucf101 \
  --dataset_split_name test \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling seqvlad-with-redu \
  --redu_dim 256 \
  --netvlad_initCenters 64 \
  --classifier_type linear \
  --ncrops 5
