# shellcheck disable=SC2164
cd /home/can/Desktop/research/REDFormer  ## go to the REDFormer dir
export PYTHONPATH=$PWD/:$PYTHONPATH
python ./tools/test.py \
./projects/configs/redformer/redformer_small.py \
ckpts/redformer.pth \
--eval bbox
