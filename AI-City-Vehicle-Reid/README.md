## Get Started

#### 1. put the track1 folder into 'data/'

```
cd data   
mkdir track1  
cp -r path/to/track1/test/ track1/
```

#### 2. training models

```
cd train
sh run.sh
```

#### 3. tset
```
cd test
sh run.sh
python gen_track2.py
```

