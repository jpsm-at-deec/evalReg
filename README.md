# evalReg

### A
```
mkdir </path/to/working/space>
cd </path/to/working/space>
git clone https://github.com/jpsm-at-deec/evalReg.git
python3 -m venv </path/to/new/virtual/environment>
pip install --upgrade pip
pip install -r requirements.txt
```

### B
```
cd </path/to/working/space>/evalReg
source </path/to/new/virtual/environment>/bin/activate
python test_transform_manager.py
```
![screenshot](data/screenshot.png)

### D
![drawing-1](data/drawing-1.png)

### E
```
cd aux
python gen_aruco.py
```
![aruco](aux/out.png)

### F
```
cd aux
python realsense_recorder.py --record_imgs
```


