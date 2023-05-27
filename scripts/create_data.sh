# shellcheck disable=SC2164
cd /home/can/Desktop/research/REDformer  ## go to the REDFormer dir
python tools/create_data.py nuscenes \
  --root-path data/nuscenes/full/ \
  --out-dir data/nuscenes/full/ \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus data/nuscenes/full/
