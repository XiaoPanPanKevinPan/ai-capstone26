# AI Capstone HW3
Student: 112550169 潘仰祐

# TL;DR
I run it on Debian 13 (trixie) with Python 3.9.25, and the "habitat" conda environment created according to HW0. I didn't modify `navigator.py` (since there is no TODO in it), while the others follow the spec.

## 1. Preparation
The folder is expected to have such a structure:
```
.
├── hw0/                            # just like hw0's repo
│   ├── habitat-lab/
│   └── replica_v1/
└── hw3/
    ├── main.py
    ├── map_processor.py
    ├── navigator.py
    └── semantic_3d_pointcloud/     # come from hw3's repo
```

### 2. Environment Setup
It is suggested to use the environemnt created in [HW0](https://github.com/HCIS-Lab/ai-capstone26/tree/main/hw0). Activate it by running

```
conda activate habitat
```

The main dependencies are 
- opencv-python==4.8.0.76
- numpy==1.26.4

## 3. Execution

1. Just execute `python3 main.py` in the conda(python) virtual environment.
2. Select a start point at the UI.
    - Colorful parts are obstacles, and the gray parts are the inflation from obstacles. 
    - Please notice, due to the cut of ground, the white parts near gray may not be traversable, and a clicking there may cause you fly to the second floor.
3. Select a goal object by its class, which will randomly choose a point with the corresponding sementic color.
4. An enhanced RRT algorithm will be executed to find a path from the start point to the goal point. The path will be visualized on the map. Press any key to continue
    - <span style="background-color: rgb(150, 200, 255); color: black">Light Blue</span> lines and <span style="background-color: rgb(0, 255, 255); color: black">Cyan</span> points are the RRT tree.
    - <span style="background-color: rgb(0, 0, 255); color: black">Blue</span> lines and <span style="background-color: rgb(175, 255, 175); color: black">Light Green</span> points are the RRT path adjusted to leave obstacles.
    - <span style="background-color: rgb(255, 255, 0); color: black">Yellow</span> lines and <span style="background-color: rgb(175, 255, 175); color: black">Light Green</span> points are the path simplified from last path.
    - <span style="background-color: rgb(255, 0, 0); color: black">Red</span> lines and <span style="background-color: rgb(0, 255, 0); color: black">Green</span> points are the final path, which is the last path adjusted again to leave obstacles.
    - The <span style="background-color: black; color: white">Black</span> big dot are the goal points.

5. A 3D simulator will take you tour the path and reach the goal point.

    