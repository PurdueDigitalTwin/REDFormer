# shellcheck disable=SC2164
cd ~/Desktop/research/sensorfusion
export PYTHONPATH=$PWD/:$PYTHONPATH
python ./tools/test.py \
./projects/configs/bevformer/bevformer_small.py \
ckpts/raw_model/bevformer_small_epoch_24.pth \
--eval bbox
