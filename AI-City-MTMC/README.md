## Get Started

#### 1. put the track1 folder into 'data/'

```
cd data   
mkdir track1  
cp -r path/to/track1/test/ track1/
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
