## Get Started

#### 1. put the track1 folder into 'data/'

```
mkdir data   
cd data
download the data in google drive [data](https://drive.google.com/drive/folders/1MFlPn5CvLjYhYkQowW5QT6o6AJaO-QD3?usp=sharing) and unzip it, then put all the folder in the zip in the current foloder 
```

#### 2. filter the invalid rectangle of MSTC

```
cd tools
python filter.py
```

#### 3. utilize trajectory feature fusion
```
cd tools
python trajectory_fusion.py
```

#### 4. clustering
```
cd tools
python cluster.py
```

#### 5. generate track1.py
```
cd tools
python gen_res.py
```
