mkdir /data/adtalk_fc/
mkdir /data/adtalk_att/
python scripts/prepro_labels.py --input_json data/dataset.json --output_json data/adtalk.json --output_h5 data/adtalk
python scripts/prepro_feats.py --input_json data/dataset.json --output_dir data/adtalk --images_root data/ --model resnet152

mkdir /data/adtalk_fc/data/
mkdir /data/adtalk_att/data/
mv /data/adtalk_fc/* /data/adtalk_fc/data/
mv /data/adtalk_att/* /data/adtalk_att/data/

python train.py --id topdown --caption_model topdown --input_json data/adtalk.json --input_fc_dir data/adtalk_fc --input_att_dir data/adtalk_att \
--input_label_h5 data/adtalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 \
--checkpoint_path log_topdown --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30