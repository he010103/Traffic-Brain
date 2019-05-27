## Get Started

#### 1. put the track2 folder into 'data/'

```
cd data     
cp -r path/to/track2/image_test/ .
cp -r path/to/track2/image_query/ .
download the data in google drive [data](https://drive.google.com/drive/folders/1MFlPn5CvLjYhYkQowW5QT6o6AJaO-QD3?usp=sharing)
put the data folder ../test/data
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

