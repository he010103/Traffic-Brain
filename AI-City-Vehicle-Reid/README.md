## Get Started

#### 1. put the track2 folder into 'data/'

```
cd data     
cp -r path/to/track2/image_test/ .
cp -r path/to/track2/image_query/ .
cd ../test/data
cp -r path/to/track2/test_track.txt .
```

#### 2. training models

```
cd train
sh run.sh
```

#### 3. test
```
cd test
sh run.sh
python gen_track2.py
```

