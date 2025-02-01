# [TOG/SIGGRAPH 2024] Eulerian-Lagrangian Fluid Simulation on Particle Flow Maps

Our paper and video results can be found at our [project website](https://zjw49246.github.io/projects/pfm/).

## Installation

1. create conda environment

```bash
conda create -n pfm python=3.8.16
conda activate pfm
```

2. install dependencies
```bash
pip install -r requirements.txt
```

## Execution

### 2D

1. edit hyperparameters in `2D/hyperparameters.py`
2. run simulation

```bash
cd 2D
python run_2D.py
```

3. results can be found in `logs/[exp_name]/`

### 3D

1. edit hyperparameters in `3D/hyperparameters.py`
2. run simulation

```bash
cd 3D
python run_3D.py
```

3. results can be found in `logs/[exp_name]/`

## Bibliography
If you find this repository or our paper useful for your work, please consider citing it as follows:

```
@article{zhou2024eulerian,
  title={Eulerian-Lagrangian Fluid Simulation on Particle Flow Maps},
  author={Zhou, Junwei and Chen, Duowen and Deng, Molin and Deng, Yitong and Sun, Yuchen and Wang, Sinan and Xiong, Shiying and Zhu, Bo},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={4},
  pages={1--20},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```


