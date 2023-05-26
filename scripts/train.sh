# shellcheck disable=SC2164
cd /media/mayson/SamsungSSD/github/sensorfusion
export PYTHONPATH=$PWD/:$PYTHONPATH
python ./tools/train.py \
  ./projects/configs/bevformer/bevformer_small.py
#./tools/dist_train.sh ./projects/configs/redformer/bevformer_tiny.py 1
