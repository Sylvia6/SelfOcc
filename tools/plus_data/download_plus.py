import os
import logging
import sys
import math
import pandas as pd
import fire
import requests
import time
from clickhouse_driver import Client

BAGDB_URL = "https://bagdb.pluscn.cn:28443"

logging.basicConfig(
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)])


def get_client():  # void -> Client
    clickhouse_host = "172.16.101.71"
    clickhouse_user = "plus_viewer"
    clickhouse_pwd = "ex4u1balSAeR68uC"
    clickhouse_dbname = "bagdb"
    return Client(host=clickhouse_host,user=clickhouse_user,password=clickhouse_pwd,database=clickhouse_dbname)


class Snipper:
    def __init__(self, bag_snip_path, click_client, i_am_on_server=False):
        self.BAG_SNIP_PATH = bag_snip_path
        self.client = click_client
        self.i_am_on_server = i_am_on_server

    # copy from https://github-cn.plus.ai/PlusAI/CenterTrack/blob/master/data_mining/jira_bag_snipper/snip_by_label.py#L158-L172
    def get_bag_snip_name(self, start_t, end_t, bag_path):
        bag_name = os.path.basename(bag_path)
        bag_name_no_ext = os.path.splitext(bag_name)[0]
        bag_ext = os.path.splitext(bag_name)[1]
        snip_name = bag_name_no_ext + "_" + str(start_t) + "to" + str(end_t) + bag_ext
        snip_path = os.path.join(self.BAG_SNIP_PATH, snip_name)
        return snip_name, snip_path

    def nas_snip_bag(self, start_t, end_t, start_time, bag_path):
        """
        start_t: start time of the snip relative to the start of the bag
        start_time: start_time of the bag
        """
        import fastbag
        st = 0
        if start_t < 86400:
            st = start_time
        _, snip_path = self.get_bag_snip_name(start_t, end_t, bag_path)
        logging.info(str((bag_path, st+start_t, st+end_t, snip_path)))
        s = fastbag.snipper.Snipper()
        try:
            s.run(bag_path, snip_path, start=st+start_t, end=st+end_t,
                do_vacuum=False, do_hash=False, do_check_version=False)
        except Exception as e:
            logging.warning(str(e))
    
    @staticmethod
    def download(url, file_path):
        logging.info("start saving to {}".format(file_path))
        with requests.get(url, stream=True) as r:
            if r.ok:
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                    time.sleep(1)
            else:  # HTTP status code 4XX/5XX
                logging.error("Download {} failed: status code {}\n{}".format(
                    os.path.basename(file_path),
                    r.status_code,
                    r.text))

        return file_path

    def download_snip_bag(self, start_t, end_t, bag_path):
        _, snip_path = self.get_bag_snip_name(start_t, end_t, bag_path)
        url = \
                "{bagdb_url}/bagdb/snips/fastsnips?" \
                "fastbag_path={bagdb_url}/raw{bag_path}&" \
                "start={start_time}&" \
                "end={end_time}".format(bagdb_url=BAGDB_URL,
                                         bag_path=bag_path,
                                         start_time=start_t,
                                         end_time=end_t)
        self.download(url, snip_path)

    def snip(self, bag_id, start_ts, end_ts, **kwargs):
        df = get_client().query_dataframe(
            "select fastbag_path, start_time from bags where id={}".format(bag_id))
        if len(df) != 1:
            logging.error("fastbag_path of bag_id {} has {} rows".format(bag_id, len(df)))
        bag_path = df.iloc[0].fastbag_path
        st = start_ts.timestamp() - df.iloc[0].start_time
        et = end_ts.timestamp() - df.iloc[0].start_time
        st = max(int(math.floor(st)), 0)
        et = int(math.ceil(et))
        if self.i_am_on_server:
            self.nas_snip_bag(st, et, df.iloc[0].start_time, bag_path)
        else:
            logging.warning("this is ok when we call small amount of snips, please do not use this function for massive bag snips downloading")
            self.download_snip_bag(st, et, bag_path)

    def __call__(self, snip_infos):
        for snip_info in snip_infos:
            self.snip(**snip_info)


def main(
    lon,
    lat,
    radius=100,
    vehicle_pattern="pdb-l4e",
    start_date="2023-11-01",
    end_date="2023-11-15",
    start_hour=0, # hour
    end_hour=24, # hour
    do_snip=False,
    snip_from_nas=True,
    snip_root="./data/plus",
    ):
    # 1. find snips
    """
    We will use table "bag_positions" with format like:
    |---------------ts--------|------------------bag_name------------|---bag_id---|------longitude------|-------latitude------|
    | 2023-11-01 00:00:01.000 | 20231031T231934_pdb-l4e-b0007_17.bag | 1788667 | 121.20201704532118 | 31.233041747920176 |
    | 2023-11-01 00:00:01.000 | 20231031T231934_pdb-l4e-b0007_17.bag | 1788667 | 121.20201692042787 | 31.233038453214856 |
    """

    sql = """
        select DISTINCT ON (ts, bag_id) *
        from bag_positions
        where ts between '{}' and '{}'
        and toHour(ts) >= {} and toHour(ts) <= {}
        and greatCircleDistance(longitude, latitude, {}, {})<{}
    """.format(start_date, end_date, start_hour, end_hour, lon, lat, radius)
    if vehicle_pattern:
        sql += "    and bag_name like '%{}%'".format(vehicle_pattern)
    logging.info(sql)
    client = get_client()
    df = get_client().query_dataframe(sql)
    if len(df) == 0:
        logging.error("no data")
        return False
    logging.info("get {} rows".format(len(df)))

    # parse data
    snip_infos = []
    for i, idf in iter(df.groupby("bag_id")):
        bag_name = idf.iloc[0].bag_name
        # althougth it rarely happend, but a trip may enter the same area mutiple times
        # so we split the trip if the difference between a trajectory is larger than 30 seconds
        idf = idf.sort_values("ts").reset_index(drop=True)
        split = idf.ts.diff() > pd.Timedelta(seconds=30)
        split_idces = idf[split].index.tolist() # this is nearly alway to be []
        split_idces.append(len(idf))
        start = 0
        for split_idx in split_idces:
            snip_infos.append({"bag_name": bag_name,
             "bag_id": i,
             "start_ts": idf.iloc[start].ts,
             "end_ts": idf.iloc[split_idx-1].ts})
            start = split_idx
    logging.info("get {} snips".format(len(snip_infos)))

    # 2. Do the snip
    if not do_snip:
        logging.info(snip_infos)
        return True
    
    snip_path = "{}/lon{}_lat{}_radius{}_vehicle_pattern{}_start_date{}_end_date{}_start_hour{}_end_hour{}".format(snip_root, lon, lat, radius, vehicle_pattern, start_date, end_date, start_hour, end_hour)
    if not os.path.exists(snip_path):
        os.makedirs(snip_path)
    s = Snipper(snip_path, client, snip_from_nas)
    s(snip_infos)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
