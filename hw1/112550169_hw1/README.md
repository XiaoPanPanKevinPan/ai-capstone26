# AI Capstone HW1
Student: 112550169 潘仰祐

# How to run
I run it on Debian 13 (trixie) with Python 3.13.5, numpy 2.4.3, and opencv-python 4.13.0.92. As long as the packages and python version are not too old, it should be fine. You can create a virtual environment and install the dependencies by running the following commands:

```
python3 -m venv /tmp/venv       # create a virtual environment
source /tmp/venv/bin/activate   # enter the virtual environment

# then install dependencies
pip3 install numpy opencv-python

# then execute the code
python3 ./bev2front.py
```

It should also be fine if you have installed the conda environment according to HW0. Under this circumstances, you can just use the following commands:

```
conda activate habitat
python3 ./bev2front.py
```




