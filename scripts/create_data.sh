# shellcheck disable=SC2164
cd /home/mayson/GitHub/sensorfusion
python tools/create_data.py nuscenes \
  --root-path data/nuscenes/full/ \
  --out-dir data/nuscenes/annotations/ \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus data/nuscenes/

#cd /home/can/Desktop/research/sensorfusion
#python tools/create_data.py nuscenes \
#  --root-path /media/can/samsungssd/nuscenes/full \
#  --out-dir /media/can/samsungssd/nuscenes/full \
#  --extra-tag nuscenes \
#  --version v1.0 \
#  --canbus /media/can/samsungssd/nuscenes/full \

#cd /home/can/Desktop/research/sensorfusion
#python tools/create_data.py nuscenes \
#  --root-path /home/can/Documents/mini/v1.0-mini \
#  --out-dir /home/can/Documents/mini/v1.0-mini \
#  --extra-tag nuscenes \
#  --version v1.0-mini \
#  --canbus /media/can/samsungssd/nuscenes/full \
