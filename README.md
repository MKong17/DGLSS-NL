# DGLSS-NL

The is the official Pytorch implementation of our work "Exploring Single Domain Generalization of LiDAR-based Semantic Segmentation under Imperfect Labels".


### [Paper](https://arxiv.org/abs/2510.09035)

[Weitong Kong*],[Zichao Zeng],[Di Wen],[Jiale Wei],[Kunyu Peng],[June Moh Goo],[Jan Boehm],[Rainer Stiefelhagen]
(* The first two authors contributed equally. In alphabetical order.)

## Abstract
Accurate perception is critical for vehicle safety, with LiDAR as a key enabler in autonomous driving. To ensure robust performance across environments, sensor types, and weather conditions without costly re-annotation, domain generalization in LiDAR-based 3D semantic segmentation is essential. However, LiDAR annotations are often noisy due to sensor imperfections, occlusions, and human errors. Such noise degrades segmentation accuracy and is further amplified under domain shifts, threatening system reliability. While noisy-label learning is well-studied in images, its extension to 3D LiDAR segmentation under domain generalization remains largely unexplored, as the sparse and irregular structure of point clouds limits direct use of 2D methods. To address this gap, we introduce the novel task Domain Generalization for LiDAR Semantic Segmentation under Noisy Labels (DGLSS-NL) and establish the first benchmark by adapting three representative noisy-label learning strategies from image classification to 3D segmentation. However, we find that existing noisy-label learning approaches adapt poorly to LiDAR data. We therefore propose DuNe, a dual-view framework with strong and weak branches that enforce feature-level consistency and apply cross-entropy loss based on confidence-aware filtering of predictions. Our approach shows state-of-the-art performance by achieving 56.86% mIoU on SemanticKITTI, 42.28% on nuScenes, and 52.58% on SemanticPOSS under 10% symmetric label noise, with an overall Arithmetic Mean (AM) of 49.57% and Harmonic Mean (HM) of 48.50%, thereby demonstrating robust domain generalization in DGLSS-NL tasks. The code is available on our project page.

## Requirements
The code has been tested with Python 3.8, CUDA 11.1, pytorch 1.8.0, and pytorch-lightning 1.6.5.

Make a virtual environment using conda, and install the following packages.

* python
* pytorch
* pytorch-lightning
* [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
* easydict
* munch
* PyYAML
* scikit-learn
* numba

## Datasets
We use SemanticKITTI, nuScenes-lidarseg, Waymo, and SemanticPOSS.

### SemanticKITTI
Download the dataset from [here](http://www.semantic-kitti.org/) and prepare the dataset directory as follows.
~~~
path_to_SemanticKITTI
    |- sequences
        |- 00/
            |- labels
                |- 000000.label
                |_ ...
            |- velodyne
                |- 000000.bin
                |_ ...
            |- calib.txt
            |- poses.txt
            |- times.txt
        |_ ...
~~~

### nuScenes-lidarseg
Download the dataset from [here](https://www.nuscenes.org/nuscenes#overview) and prepare the dataset directory as follows.
~~~
path_to_nuScenes
    |- lidarseg
        |- v1.0-{mini, test, trainval}
            |- xxxx_lidarseg.bin
            |_ ...
    |- samples
        |- LIDAR_TOP
            |- xxxx.pcd.bin
            |_ ...
    |- sweeps
    |- v1.0-{mini, test, trainval}
        |- ***.json
        |_ ...
    |- nuscenes_infos_{train, val, test}.pkl
~~~

### Waymo
Download the dataset from [here](https://waymo.com/open/data/perception/) and prepare the dataset directory as follows.
~~~
path_to_Waymo
    |- 0000/
        |- labels
            |- 000000.label
            |_ ...
        |- velodyne
            |- 000000.bin
            |_ ...
    |_ 0999/
~~~

### SemanticPOSS
Download the dataset from [here](http://www.poss.pku.edu.cn/semanticposs.html) and prepare the dataset directory as follows.
~~~
path_to_POSS
    |- sequences
        |- 00/
            |- labels
                |- 000000.label
                |_ ...
            |- velodyne
                |- 000000.bin
                |_ ...
            |- tag
                |- 000000.tag
                |_ ...
            |- calib.txt
            |- poses.txt
            |- times.txt
        |_ ...
~~~

## Train
Run the following command for training:
~~~
python main.py --logdir='name for current experiment log directory' --config='path to config file for source dataset'
~~~
Specify the checkpoint path if you want to resume or load pretrained weights in config_{dataset_name}.yaml (train_params.resume_ckpt or pretrained_ckpt_path)

## Test
Run the following command for testing:
~~~
python main.py --logdir='name for current experiment test log directory' --test --config='path to config file for source dataset'
~~~
Specify the path of checkpoint to test in config_{dataset_name}.yaml (test_params.ckpt_path)

## Reference
If you use our work, please cite us!

~~~
@inproceedings{kim2023single,
  title={Single Domain Generalization for LiDAR Semantic Segmentation},
  author={Kim, Hyeonseong and Kang, Yoonsu and Oh, Changgyoon and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17587--17598},
  year={2023}
}
~~~

