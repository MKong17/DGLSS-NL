import os
import numpy as np
import yaml
from munch import Munch
import MinkowskiEngine as ME
import random
from datasets.utils import polarmix
from datasets.custom import CustomDataset
import pdb
from tqdm import tqdm

class KITTIDataset(CustomDataset):

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_label=-100,
                 label_mapping=None,
                 max_volume_space=[50., 50., 2.],
                 min_volume_space=[-50., -50., -4.],
                 out_shape=[512,512,32],
                 min_coordinate=[-256,-256,-21],
                 voxel_size=0.2,
                 beam=64,
                 fov=[-23.6, 3.2],
                 training=False,
                 use_sparse_aug=False,
                 positive_num=1,
                 beam_sampling=[0.3, 0.7],
                 ):
        super(KITTIDataset, self).__init__(data_path, ignore_label, label_mapping, max_volume_space, min_volume_space,
                                           out_shape, min_coordinate, voxel_size, beam, fov, training, 
                                           use_sparse_aug, positive_num, beam_sampling)
        with open(label_mapping, 'r') as f:
            semkittiyaml = yaml.safe_load(f)
        if self.imageset == 'train':
            split = semkittiyaml['split']['train']
        elif self.imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif self.imageset == 'test':
            split = semkittiyaml['split']['test']
            
        self.learning_map = semkittiyaml['learning_map_common']
        self.learning_map_inv = semkittiyaml['learning_map_inv']
        self.label_path = label_path
        
        # ignore -100
        # for k, v in self.learning_map.items():
        #     if v == 0: self.learning_map[k] = -100
        
        self.im_idx = []
        for i_folder in split:
            self.im_idx += self.absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(sefl):
        return len(sefl.im_idx)

   
    def read_lidar_scan(self, index):
        scan_id = self.im_idx[index]

        with open(self.im_idx[index], 'rb') as f:
            raw_data  = np.fromfile(f, dtype=np.float32).reshape(-1, 4) # block = raw_data () open the coordinates and features -> (x, y, z), (ref)
            # raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4)) # read the data from every files
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == 'train':
            label_path = os.path.join(self.label_path, f'label_{index}.npy')
            annotated_data = np.load(label_path)
            annotated_data = np.expand_dims(annotated_data, axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))            
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data = (raw_data[:, :3], raw_data[:, 3][:,None], annotated_data.astype(np.int32))
        return index, scan_id, data

          
    def __getitem__(self, index):        
        scan_id = self.im_idx[index] # every files pathes according to the index        
        instance_classes = [0, 1, 2, 3, 4, 5, 6, 7]
        Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]

        index, scan_id, data = self.read_lidar_scan(index)
        block = np.concatenate([data[0], data[1]], axis=1)    
        labels = data[2]

        if self.imageset == 'train':
            # read another lidar scan
            index_2 = np.random.randint(len(self.im_idx))
            _, _, data2 = self.read_lidar_scan(index_2)
            pts2 = np.concatenate([data2[0], data2[1]], axis=1)  
            labels2 = data2[2]
            alpha = (np.random.randint(2) * 2 - 1) * np.pi
            beta = alpha + np.pi

            data_aug = polarmix(block, labels, pts2, labels2,
                            alpha=alpha, beta=beta,
                            instance_classes=instance_classes,
                            Omega=Omega)# the number of outputs will be significantly increased; 
                                        # here we ensure the dimension of outputs is the same as weak augmentation by random picking
    
            oringinal_item = self.getitem(index, scan_id, data)
            aug_item = self.getitem(index, scan_id, data_aug)

            return oringinal_item, aug_item   
        else:
            return self.getitem(index, scan_id, data)

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_folder = os.path.dirname(script_dir)
    config_path = os.path.join(config_folder, 'configs', 'config_kitti.yaml')

    cfg_txt = open(config_path, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    dataset = KITTIDataset(**cfg.dataset_SemKITTI, training=True, 
                           positive_num=cfg.generalization_params.positive_num,
                           beam_sampling=cfg.generalization_params.beam_sampling)

    # def get_noisy_labels(annotated_data):

    #     labels = annotated_data
    #     noisy_labels = np.empty((len(annotated_data), 1), dtype=np.int32)
    #     for idx, label in enumerate(labels):

    #         if random.randint(1,100) <= 5:
    #             noisy_label = np.random.randint(1,11)
    #             while noisy_label == label:
    #                 noisy_label = np.random.randint(1,11)
    #             noisy_labels[idx] = noisy_label

    #         else:
    #             noisy_labels[idx] = label

    #     new_annotated_data = noisy_labels

    #     return new_annotated_data

    ### visualize
    while True:
        (scan_id, coord, feat, semantic_label, idx) = dataset.__getitem__(0)
        pdb.set_trace()
        from utils.utils import get_semcolor_common
        from utils.ply_vis import write_ply
 
        for i in range(len(coord)):
            vis_color = get_semcolor_common(semantic_label[i])
            vis_xyz = coord[i].float().cpu().numpy()
            os.makedirs('sample', exist_ok=True)
            pdb.set_trace()
            write_ply(f'sample/kitti_2.ply', [vis_xyz, vis_color], ['x','y','z','red','green','blue'])

    #follow used for validate single label and noisy label

    # (scan_id, coord, feat, semantic_label, idx), index_2 = dataset.__getitem__(0)
    # if isinstance(semantic_label, list) and len(semantic_label) == 1:
    #     semantic_label = semantic_label[0]

    # print(type(semantic_label))
    # print(len(semantic_label))
    # print(type(semantic_label[0]))
    # labels = semantic_label.cpu().numpy()
    # #os.makedirs('sample', exist_ok=True)
    # np.save(f'labels_special.npy', labels)

    # print(type(semantic_label))
    # print(len(semantic_label))
    # print(type(semantic_label[0]))
    # labels = semantic_label.cpu().numpy()

    # noisy_label = get_noisy_labels(labels)
    # np.save(f'labels_noisy.npy', noisy_label)

# pre_process for getting noisy labels

    # for index in tqdm(range(len(dataset)), desc="Loading files"):
    #     (scan_id, coord, feat, semantic_label, index) = dataset.__getitem__(index)
        
    #     if isinstance(semantic_label, list) and len(semantic_label) == 1:
    #         semantic_label = semantic_label[0]

    #     print(type(semantic_label))
    #     print(len(semantic_label))

    #     labels = semantic_label.cpu().numpy()
    #     os.makedirs('without_noise', exist_ok=True)
    #     np.save(f'without_noise/label_{index}.npy', labels)

    #     print(type(semantic_label))
    #     print(len(semantic_label))

    #     noisy_label = get_noisy_labels(labels)
    #     os.makedirs('noise', exist_ok=True)
    #     np.save(f'noise/label_{index}.npy', noisy_label)

    #     print(type(noisy_label))
    #     print(len(noisy_label))