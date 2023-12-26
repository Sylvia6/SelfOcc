set -e
source .venv2/bin/activate
# valid
python tools/plus_data/download_plus.py\
    --lat=31.4720272\
    --lon=120.6019952\
    --start_date="2023-11-06"\
    --end_date="2023-11-07"\
    --start_hour=1\
    --end_hour=8\
    --vehicle_pattern="pdb-l4e-b" \
    --do_snip=True\
    --snip_root="/mnt/bigfile_2/mingyao.li/nerf_data/selfocc_tiny/valid"
echo "valid data downloaded"

# train
python tools/plus_data/download_plus.py\
    --lat=31.4720272\
    --lon=120.6019952\
    --start_date="2023-11-01"\
    --end_date="2023-11-05"\
    --start_hour=1\
    --end_hour=8\
    --vehicle_pattern="pdb-l4e-b" \
    --do_snip=True\
    --snip_root="/mnt/bigfile_2/mingyao.li/nerf_data/selfocc_tiny/train"
echo "train data downloaded"


