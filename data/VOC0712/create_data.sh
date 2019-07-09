#!/usr/bin/env bash
redo=1
data_root_dir="/home/jade/Data/HandGesture"
dataset_name="Hand_Gesture"
mapfile="/home/jade/Desktop/RefineDet-master/data/VOC0712/hand_gesture.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train test
do
  python /home/jade/Desktop/RefineDet-master/scripts/create_annoset.py  --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$dataset_name/ImageSets/Main/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
