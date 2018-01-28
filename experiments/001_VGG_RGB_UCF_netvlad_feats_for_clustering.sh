# note, can use the any split model as upto conv5 for RGB (on HMDB) it is basically imagenet pretrained
cd ../
LD_PRELOAD=/usr/lib/libtcmalloc.so.4 \
  $(which python) \
  eval_image_classifier.py \
  --gpus 0 \
  --batch_size 32 \
  --frames_per_video 1 \
  --max_num_batches 100 \
  --checkpoint_path models/PreTrained/imagenet-trained-CUHK/vgg_16_action_rgb_pretrain_uptoConv5.ckpt \
  --dataset_dir /data/UCF-101-frames \
  --dataset_list_dir data/ucf101/train_test_lists/ \
  --dataset_name ucf101 \
  --dataset_split_name train \
  --model_name vgg_16 \
  --modality rgb \
  --num_readers 4 \
  --num_preprocessing_threads 4 \
  --preprocessing_name vgg_ucf \
  --bgr_flip True \
  --pooling None \
  --store_feat stream0/vgg_16/conv5/conv5_3 \
  --store_feat_path /home/sensetime/usr/local/code/ActionVLAD/models/experiments/001_VGG_RGB_UCF_netvlad_stage1/Features/imnet_conv5.h5 \
  --force_random_shuffle True \
  --num_streams 1 \
  --split_id 1
