set -e
source .venv2/bin/activate
# train
python tools/plus_data/download_plus.py\
    --lat=31.4720272\
    --lon=120.6019952\
    --start_date="2023-07-01"\
    --end_date="2023-08-31"\
    --start_hour=1\
    --end_hour=8\
    --vehicle_pattern="pdb-l4e-b" \
    --do_snip=True\
    --snip_root="/mnt/bigfile_2/mingyao.li/nerf_data/selfocc_small/train"
echo "train data downloaded"


#python tools/plus_data/download_plus.py\
#    --lat=31.4720272\
#    --lon=120.6019952\
#    --start_date="2023-09-01"\
#    --end_date="2023-10-31"\
#    --start_hour=1\
#    --end_hour=8\
#    --vehicle_pattern="pdb-l4e-b" \
#    --do_snip=True\
#    --snip_root="/mnt/bigfile_2/mingyao.li/nerf_data/selfocc_small/train"
#echo "train data downloaded"
#
## valid
#python tools/plus_data/download_plus.py\
#    --lat=31.4720272\
#    --lon=120.6019952\
#    --start_date="2023-11-01"\
#    --end_date="2023-11-15"\
#    --start_hour=1\
#    --end_hour=8\
#    --vehicle_pattern="pdb-l4e-b" \
#    --do_snip=True\
#    --snip_root="/mnt/bigfile_2/mingyao.li/nerf_data/selfocc_small/valid"
#echo "valid data downloaded"
