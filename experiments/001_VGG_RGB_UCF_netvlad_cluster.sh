#cd ../
nice -n 19 python vlad_utils/cluster_feats.py \
  -i models/experiments/001_VGG_RGB_UCF_netvlad_stage1/Features/imnet_conv5.h5 \
  -o models/experiments/001_VGG_RGB_UCF_netvlad_stage1/Features/imnet_conv5_kmeans64.pkl \
  -k 64 \
  -j 8 \
  -n stream0/vgg_16/conv5/conv5_3 \
  --nfeats 5000
