# shellcheck disable=SC2164
cd /home/can/Desktop/research/REDFormer  ## go to the REDFormer dir
export PYTHONPATH=$PWD/:$PYTHONPATH
python ./tools/train.py \
  ./projects/configs/redformer/redformer_small.py
