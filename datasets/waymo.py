import os
import numpy as np
import yaml
from munch import Munch
import MinkowskiEngine as ME
import pdb

from .custom import CustomDataset


class WaymoDataset(CustomDataset):

    def __init__(self,
                 data_path,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 3.],
                 min_volume_space=[-50., -50., -5.],
                 out_shape=[768,768,32],
                 min_coordinate=[-384,-384,-11],
                 voxel_size=0.2,
                 beam=64,
                 fov=[-17.6, 2.4],
                 training=False,
                 use_sparse_aug=False,
                 positive_num=1,
                 beam_sampling=[0.3, 0.7],
                 ):
        
        super(WaymoDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           out_shape, min_coordinate, voxel_size, beam, fov, training, 
                                           use_sparse_aug, positive_num, beam_sampling)

        with open(label_mapping, 'r') as f:
            waymo = yaml.safe_load(f)
        with open(label_mapping.replace('waymo', 'waymo_split'), 'r') as f:
            waymo_split = yaml.safe_load(f)    
        
        if self.imageset == 'train':
            split = waymo_split['train']
        elif self.imageset == 'val':
            split = waymo_split['valid']
        elif self.imageset == 'test':
            split = waymo_split['test']
        
        self.learning_map_common = waymo['learning_map_common']
        self.learning_map = waymo['learning_map']
        self.ignore_label = ignore_label

        # # ignore -100
        # for k, v in self.learning_map.items():
        #     if v == 0: self.learning_map[k] = -100

        self.im_idx = []
        self.im_names = []

        for i_folder in split:
            folder_name = f"file_{int(i_folder):04d}"  # 如 file_0798
            folder_path = os.path.join(data_path, folder_name)
            if not os.path.exists(folder_path):
                print (f"⚠️ 文件夹缺失: {folder_path}")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith('.bin'):
                    full_path = os.path.join(folder_path, fname)
                    self.im_idx.append(full_path)
                    self.im_names.append(fname)

    
    def __len__(self):
        # print(f'length of waymo: {len(self.im_idx)}')
        return len(self.im_idx)
    
    def __getitem__(self, index):
        scan_id = self.im_idx[index]
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1,4))
        print("raw points:", raw_data.shape[0])
        label_root= "/hkfs/work/workspace/scratch/bx6695-exp/google/label"
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            # annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',dtype=np.uint32).reshape((-1, 1))
            # annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            # frame_name = self.im_names[index].split('file_')[-1]  # 变成 '000_frame_000024_point_cloud.bin'
            name_parts = self.im_names[index].split('_')
            file_id = name_parts[1]     # '026'
            frame_id = name_parts[3]    # '000069'
            # file_id = self.im_names[index].split('_')[0]  # '000'
            # frame_id = self.im_names[index].split('_')[2]  # '000024'
            label_name = f"frame_{frame_id}_semantic_labels.npy"  # 'frame_000024_semantic_labels.npy'
            label_subdir = f"file_{int(file_id):04d}" 
            label_path = os.path.join(label_root, label_subdir, label_name)
            labels = np.load(label_path)

            annotated_data = np.vectorize(lambda x: self.learning_map.get(int(x), self.ignore_label))(labels)
            annotated_data = np.vectorize(lambda x: self.learning_map_common.get(int(x), self.ignore_label))(annotated_data)
            
            # print("最终 annotated_data（前20）：", annotated_data[:20])
            # print(np.unique(annotated_data))
            annotated_data = np.expand_dims(annotated_data, axis=1)
            
        ref = np.tanh(raw_data[:,3])
        data = (raw_data[:, :3], ref, annotated_data.astype(np.int32))
        # print(f"[DEBUG] Fetching sample {index} from {self.__class__.__name__}")
        return self.getitem(index, scan_id, data)
        
    # same collate function as CustomDataset
    
if __name__ == '__main__':
    cfg_txt = open('configs/config.yaml', 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    dataset = WaymoDataset(**cfg.dataset_Waymo, training=True,
                           use_sparse_aug=cfg.generalization_params.use_sparse_aug,
                           positive_num=cfg.generalization_params.positive_num,
                           beam_sampling=cfg.generalization_params.beam_sampling,
                           )
    
    # save training set
    from utils.utils import get_semcolor_common
    from utils.ply_vis import write_ply
    from tqdm import tqdm
    
    ### visualize
    while True:
        (scan_id, coord, feat, semantic_label, idx) = dataset.__getitem__(0)
        pdb.set_trace()
        
        name_list=['orig', 'aug1', 'aug2', 'aug3', 'aug4']
        for i in range(len(coord)):
            vis_color = get_semcolor_common(semantic_label[i])
            vis_xyz = coord[i].float().cpu().numpy()
            os.makedirs('sample', exist_ok=True)
            write_ply(f'sample/waymo_sample_{name_list[i]}_gg.ply', [vis_xyz, vis_color], ['x','y','z','red','green','blue'])
    
# python -m datasets.waymo