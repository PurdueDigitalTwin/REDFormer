import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Generate decompression commands')
parser.add_argument(
    '--download_dir',
    type=str,
    default='/media/can/samsungssd/nuscenes/archive/'
)
parser.add_argument(
    '--dest_dir',
    type=str,
    default='/media/can/samsungssd/nuscenes/full'
)
parser.add_argument(
    '--decompress_commands',
    type=str,
    default='/media/can/samsungssd/nuscenes/decompress_nuscenes.sh'
)
parser.add_argument('--prefix',
                    type=str,
                    default="",
                    # default="#!/bin/bash\n"
                    #         "# FILENAME: decompress_nuscenes.sh\n"
                    #         "#SBATCH -A ziran-i \n"
                    #         "#SBATCH --nodes=1 --gpus-per-node=1 \n"
                    #         "#SBATCH --time=12:00:00 \n"
                    #         "#SBATCH --job-name decompression\n"
                    #         "cd \n"
                    )

args = parser.parse_args()


def nuscenes_data_decompression(
        download_dir,
        dest_dir,
        decompress_commands,
        prefix,
):
    download_path = Path(download_dir)
    dest_path = Path(dest_dir)
    tgz_files = []

    for fname in download_path.iterdir():
        if not str(fname).endswith('.tgz'):
            continue
        tgz_files.append(str(fname))

    print(tgz_files)
    with open(decompress_commands, "w") as f:
        f.writelines(prefix)
        for fname in tgz_files:
            f.writelines(f"tar -zxvf {fname} -C {dest_dir}\n")


if __name__ == "__main__":
    nuscenes_data_decompression(
        args.download_dir,
        args.dest_dir,
        args.decompress_commands,
        args.prefix
    )
