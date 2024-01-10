# Engilish
*  **Theory** : [https://wikidocs.net/226345](https://wikidocs.net/226345) <br>
*  **Implementation** : [https://wikidocs.net/226346](https://wikidocs.net/226346)

# 한글
*  **Theory** : [https://wikidocs.net/225900](https://wikidocs.net/225900) <br>
*  **Implementation** : [https://wikidocs.net/226044](https://wikidocs.net/226044)

This repository is folked from [https://github.com/yjh0410/RT-ODLab](https://github.com/yjh0410/RT-ODLab).
At this repository, simplification and explanation and will be tested at Colab Environment.

# YOLOX2:

|   Model  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOX2-N | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX2-T | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX2-S | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX2-M | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX2-L | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX2-X | 8xb16 |  640  |                        |                   |                   |                    |  |
<!-- | YOLOX2-S | 8xb16 |  640  |          42.0          |        60.2       |        27.6       |          9.2       | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox2_s_coco.pth) | -->

- For training, we train YOLOX2 series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the YOLOX.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64,.
- For learning rate scheduler, we use Linear decay scheduler.


## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_F_10_Pytorch_Yolox2.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Demo
### Detect with Image
```Shell
# Detect with Image

# ! python demo.py --mode image \
#                  --path_to_img /content/dataset/demo/images/ \
#                  --cuda \
#                  -m yolox2_s \
#                  --weight /content/yolox2_s_coco.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --show

# See /content/det_results/demos/image
```

### Detect with Video
```Shell
# Detect with Video

# ! python demo.py --mode video \
#                  --path_to_vid /content/dataset/demo/videos/street.mp4 \
#                  --cuda \
#                  -m yolox2_s \
#                  --weight /content/yolox2_s_coco.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --gif
                 # --show
# See /content/det_results/demos/video Download and check the results
```

### Detect with Camera
```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo.py --mode camera \
#                  --cuda \
#                  -m yolox2_s \
#                  --weight /content/yolox2_s_coco.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --gif
                 # --show
```

## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
# ! wget http://images.cocodataset.org/zips/val2017.zip
# ! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip


# ! unzip train2017.zip  -d dataset/COCO
# ! unzip val2017.zip  -d dataset/COCO
# ! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

# ! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

# ! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

# clear_output()
```

## Test YOLOX2
Taking testing YOLOX2-S on COCO-val as the example,
```Shell
# Test YOLOx2
# ! python test.py --cuda \
#                  -d coco \
#                  --data_path /content/dataset \
#                  -m yolox2_s \
#                  --weight /content/yolox2_s_coco.pth \
#                  -size 640 \
#                  -vt 0.4
                 # --show
# See /content/det_results/coco/yolox2
```

## Evaluate YOLOX2
Taking evaluating YOLOX2-S on COCO-val as the example,
```Shell
# Evaluate YOLOx2
# ! python eval.py --cuda \
#                  -d coco-val \
#                  --data_path /content/dataset \
#                  --weight /content/yolox2_s_coco.pth \
#                  -m yolox2_s
```

# Training test
## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```

## Train YOLOX2
### Single GPU
Taking training YOLOX2-n on COCO as the example,
```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox2_n \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
#  yolox2_n, yolox2_t, yolox2_s, yolox2_m, yolox2_l, yolox2_x
```

```
# cannot train yolox2_t
# ! python train.py --cuda \
#                   -d voc \
#                   --data_path /content/dataset \
#                   -m yolox2_t \
#                   -bs 16 \
#                   --max_epoch 5 \
#                   --wp_epoch 1 \
#                   --eval_epoch 5 \
#                   --fp16 \
#                   --ema \
#                   --multi_scale
#  yolox2_n, yolox2_t, yolox2_s, yolox2_m, yolox2_l, yolox2_x
```

```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox2_s \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
#  yolox2_n, yolox2_t, yolox2_s, yolox2_m, yolox2_l, yolox2_x
```

```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox2_m \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```
# T4 GPU 14.7G

! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox2_l \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
#  yolox2_n, yolox2_t, yolox2_s, yolox2_m, yolox2_l, yolox2_x
```

```
# T4 GPU 14.0 G

! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox2_x \
                  -bs 8 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
#  yolox2_n, yolox2_t, yolox2_s, yolox2_m, yolox2_l, yolox2_x
```


### Multi GPU
Taking training YOLOX2-S on COCO as the example,
```Shell
# Cannot test at Colab-Pro + environment

# ! python -m torch.distributed.run --nproc_per_node=8 train.py \
#                                   --cuda \
#                                   -dist \
#                                   -d voc \
#                                   --data_path /content/dataset \
#                                   -m yolox2_s \
#                                   -bs 128 \
#                                   -size 640 \
#                                   --wp_epoch 3 \
#                                   --max_epoch 300 \
#                                   --eval_epoch 10 \
#                                   --no_aug_epoch 20 \
#                                   --ema \
#                                   --fp16 \
#                                   --sybn \
#                                   --multi_scale \
#                                   --save_folder weights/
```


