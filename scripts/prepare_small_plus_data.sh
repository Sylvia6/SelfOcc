set -e
source .venv2/bin/activate
# train
python tools/plus_data/download_plus_data.py\
    --lat=31.4720272\
    --lon=120.6019952\
    --start_date="2023-09-01"\
    --end_date="2023-10-31"\
    --start_hour=1\
    --end_hour=8\
    --vehicle_pattern="pdb" \
    --do_snip=True\
    --snip_root="/mnt/intel/jupyterhub/mingyao.li/nerf_data/selfocc/train"
echo "train data downloaded"

# valid
python tools/plus_data/download_plus_data.py\
    --lat=31.4720272\
    --lon=120.6019952\
    --start_date="2023-11-01"\
    --end_date="2023-11-15"\
    --start_hour=1\
    --end_hour=8\
    --vehicle_pattern="pdb" \
    --do_snip=True\
    --snip_root="/mnt/intel/jupyterhub/mingyao.li/nerf_data/selfocc/valid"
echo "valid data downloaded"