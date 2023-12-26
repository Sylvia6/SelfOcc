
## Prepare env
```
drive --volume /mnt:/mnt:ro 
```
```
./tools/plus_data/plus_env/setup.sh
```
## Snip
**from server**
```
python tools/plus_data/download_plus.py  --lat 31.4720272 --lon 120.6019952  --vehicle_pattern "pdb" --do_snip=True
```
**from local (by http)**
```
python tools/plus_data/download_plus.py  --lat 31.4720272 --lon 120.6019952  --vehicle_pattern "pdb" --do_snip=True --snip_from_nas=False
```
then get snips at data/plus/
