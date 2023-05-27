# shellcheck disable=SC2164
cd /media/mayson/SamsungSSD/github/sensorfusion
export PYTHONPATH=$PWD/:$PYTHONPATH
python ./tools/train.py \
  ./projects/configs/bevformer/redformer_small.py
