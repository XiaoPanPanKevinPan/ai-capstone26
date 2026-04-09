# AI Capstone HW2
Student: 112550169 潘仰祐

# TL;DR
I run it on Debian 13 (trixie) with Python 3.9.25, and the "habitat" conda environment created according to HW0. I didn't modify `load.py` (since there is no TODO in it), while the parameters for `reconstruct.py` follows the homework's spec.

## 1. Preparation
The replica dataset, you can use the same one in `hw0`.

### 2. Environment Setup (Critical)
To avoid **Segmentation Faults** and library conflicts 

- **Python Version**: 3.9 or 3.10 is recommended.
- **NumPy Version**: You **MUST** use `numpy==1.26.4`. (Do NOT use 2.0+).

It is suggested to use the environemnt created in [HW0](https://github.com/HCIS-Lab/ai-capstone26/tree/main/hw0). Activate it by running

```
conda activate habitat
```


## Phase 1: Data Collection

Use the Habitat environment to collect RGB-D images and ground truth poses.

### Command

```bash
# Example: Navigating and saving data for the first floor
python load.py -f 1
```

## Phase 2: 3D Reconstruction
Switch to the reconstruction environment before running the following commands.
⚠️ Requirements:
- open3d
- numpy==1.26.4

### Standard Version (Open3D ICP)
Use Open3D's built-in ICP algorithm for reconstruction.
```bash
# Reconstruct Floor 1
python reconstruct.py -f 1

# Reconstruct Floor 2
python reconstruct.py -f 2
```

### Bonus Version (Custom ICP Implementation)
I have implemented my own ICP algorithm, use the my_icp option.
```bash
# Run reconstruction with your own ICP
python reconstruct.py -f 1 -v my_icp
```
