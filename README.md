# DS-Net: A dedicated approach for collapsed building detection from post-event airborne point clouds

<!-- ![banner](figures/result.jpeg) -->
<p align='center'>
<img src="figures/result.jpeg" alt="result" width="500"/>

## Abstract
Collapsed buildings should be detected immediately after earthquakes for humanitarian assistance and postdisaster recovery. Automatic collapsed building detection using deep learning has recently become increasingly popular because of its superior ability to obtain discriminative feature representations. Among various types of data, airborne 3D point clouds are especially useful for detecting collapsed buildings as they precisely record the height information of buildings. However, existing methods are based on the universal point cloud analysis technology that does not explicitly consider the nature of building damage. In this study, we propose Damage-Sensitive Network (DS-Net), a dedicated approach for collapsed building detection. The core of DS-Net is Laplacian Unit (LU), a simple yet effective module for 3D point clouds designed to enhance the feature representation of the damaged part to facilitate collapsed building detection. We perform extensive experiments and demonstrate that DS-Net achieves superior performance compared with existing methods. In particular, a detailed comparison of DS-Net with PointNet++, the standard network on which DS-Net’s design is based, found that DS-Net provides an 8.3% gain in precision, 3.0% gain in recall, and 6.4% gain in IoU over PointNet++ in detecting collapsed buildings. Moreover, it is verified that the detection performance can be further enhanced with increased computational resources. Qualitative analyses reveal that DS-Net excels at detecting damage manifested as roof deformations, debris, and inclinations. In addition, DS-Net produces smoother predictions with sharper boundaries compared to the baseline due to the adaptive nature of LUs. Furthermore, a visual explanation analysis based on Grad-CAM is performed to analyze how DSNet understands building damage. The result suggests that DS-Net can accurately locate varieties of building damage.

## Paper
You can download our paper from [here](https://www.sciencedirect.com/science/article/pii/S1569843222003387). 

More comprehensive explanations and experiments can be found in the first author's [Thesis](https://t2r2.star.titech.ac.jp/rrws/file/CTT100885768/ATD100000413/).

## Setup
The code is tested using Python 3.11 and CUDA 11.8. 

### Create and activate a virtual environment
- python3 -m venv ~/venv/DS-Net  
- source  ~/venv/DS-Net/bin/activate  

### Install necessary packages
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
- pip3 install h5py pyyaml tensorboardx scipy plyfile 

### Install [pointops](https://github.com/POSTECH-CVLab/point-transformer)
- cd scripts/lib/pointops
- python3 setup.py install

## Dataset
Download the dataset from [here](https://drive.google.com/file/d/14RKHsmBdTRNrecDXCDgprgmaC6wRrGyL/view?usp=sharing) and put it under the ```dataset```. 

Please check [here](https://github.com/martianxiu/DS-Net/tree/main/dataset) for the details about the dataset. 

## Training
To train the DS-Net: 
```
sh tool/train.sh your_experiment_name config_name
```

## Testing
To test the DS-Net:
```
sh tool/test.sh your_experiment_name config_name
```

## Acknowledement 
This repo is based on/inspried by many great works including but not limited to:  
[Point Transformer](https://github.com/POSTECH-CVLab/point-transformer) and [KPConv](https://github.com/HuguesTHOMAS/KPConv).  

## Citation
If you find our work useful in your research, please consider citing:
```
@article{xiu2023ds,
  title={DS-Net: A dedicated approach for collapsed building detection from post-event airborne point clouds},
  author={Xiu, Haoyi and Liu, Xin and Wang, Weimin and Kim, Kyoung-Sook and Shinohara, Takayuki and Chang, Qiong and Matsuoka, Masashi},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={116},
  pages={103150},
  year={2023},
  publisher={Elsevier}
}
```
```
@article{xiu2020collapsed,
  title={Collapsed building detection using 3d point clouds and deep learning},
  author={Xiu, Haoyi and Shinohara, Takayuki and Matsuoka, Masashi and Inoguchi, Munenari and Kawabe, Ken and Horie, Kei},
  journal={Remote Sensing},
  volume={12},
  number={24},
  pages={4057},
  year={2020},
  publisher={MDPI}
}
```
