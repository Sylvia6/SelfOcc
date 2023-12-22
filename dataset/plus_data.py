import os, numpy as np, random, math
from . import OPENOCC_DATASET
import json, cv2

def cam_pose_to_nerf(cam_pose, gl=True):  
    """
        plus_data的camera为opencv坐标系
        
        emernerf的数据坐标输入为opencv坐标，nerfstudio的数据坐标输入为opengl坐标
        
        nerf_world与eqdc坐标系一样
        
        < opencv / colmap convention >                --->>>     < opengl / NeRF convention >                    --->>>   < world convention >
        facing [+z] direction, x right, y downwards   --->>>    facing [-z] direction, x right, y upwards        --->>>  facing [+x] direction, z upwards, y left
                    z                                              y ↑                                                      z ↑    x
                   ↗                                                 |                                                        |   ↗ 
                  /                                                  |                                                        |  /
                 /                                                   |                                                        | /
                o------> x                                           o------> x                                    y ←--------o
                |                                                   /
                |                                                  /
                |                                               z ↙
                ↓ 
                y
    """
    if gl:
        opencv2opengl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0],[0,0,0,1]]).astype(float)
        opengl2opencv = np.linalg.inv(opencv2opengl)
        opengl2world = np.array([[0,0,-1,0], [-1,0,0,0], [0,1,0,0],[0,0,0,1]]).astype(float)
        gl_pose = opencv2opengl @ cam_pose @ opengl2opencv  
        world_pose = opengl2world @ gl_pose
    else:
        opencv2world = np.array([[0,0,1,0], [-1,0,0,0], [0,-1,0,0],[0,0,0,1]]).astype(float)
        world_pose = opencv2world @ cam_pose
    return world_pose

@OPENOCC_DATASET.register_module()
class PlusOneFrame:
    def __init__(
        self, 
        data_path: str = "/mnt/intel/jupyterhub/lu.li/plus_data/20230319T090808_pdb-l4e-b0007_6_871to931",
        num_cams: int = 2,
    ):
        self.data_path = data_path
        self.num_cams = num_cams
        self.load_data()
    
    def load_data(self):
        data_file = os.path.join(self.data_path, "transforms.json")
        assert os.path.exists(data_file)
        with open(data_file) as f:
            data = json.load(f)        
        self.frames = data["frames"]

    def __len__(self):
        return len(self.frames) // self.num_cams

    def __getitem__(self, idx):
        """
        return: 
            imgs: List[np.array[H,W,3]]  with N imgs
            metas: 
               timestamp: float
               n_imgs: int  ;should be N
               cam2imu: np.array[4,4] 
               imu2world: np.array[4,4]
               cam2pixel:np.array[N, 3, 4],
        """
        frames = self.frames[idx * self.num_cams: (idx+1) * self.num_cams]
        imgs = [
            cv2.imread(os.path.join(self.data_path, f["file_path"]))
            for f in frames
        ]

        meta = {}
        meta["timestamp"] = frames[0]["timestamp"]    # stereo
        meta["n_imgs"] = self.num_cams
        meta["cam2imu"] = np.array(frames[0]["c2imu"])
        meta["imu2world"] = np.array(frames[0]["imu2w_vio"])
        meta["cam2pixel"] = np.array([f["intr"] for f in frames])
        # note: check world / camera coord  using  "cam_pose_to_nerf" 

        return imgs, meta



if __name__ == "__main__":
    data = PlusOneFrame()
    import pdb;pdb.set_trace() 