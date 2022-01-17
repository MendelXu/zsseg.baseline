This repo is for our paper [A Simple Baseline for Zero-shot Semantic Segmentation with Pre-trained Vision-language Model](https://arxiv.org/pdf/2112.14757.pdf). It is based on the official repo of [MaskFormer](https://github.com/facebookresearch/MaskFormer).

![](resources/proposal.png)
```
@article{xu2021ss,
  title={End-to-End Semi-Supervised Object Detection with Soft Teacher},
  author={Xu, Mengde and Zhang, Zheng and Hu, Han and Wang, Jianfeng and Wang, Lijuan and Wei, Fangyun and Bai, Xiang and Liu, Zicheng},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Guideline
- ### Enviroment
     ```bash
     torch==1.8.0
     torchvision==0.9.0
     detectron2==0.6 #Following https://detectron2.readthedocs.io/en/latest/tutorials/install.html to install it and some required packages
     mmcv==1.3.14
     ```
     FurtherMore, install the modified clip package.
     ```bash
     cd third_party/CLIP
     python -m pip install -Ue .
     ```
- ### Data Preparation
  In our experiments, four datasets are used. For Cityscapes and ADE20k, follow the tutorial in [MaskFormer](https://github.com/facebookresearch/MaskFormer).
- For COCO Stuff 164k:
  - Download data from the offical dataset website and extract it like below.
     ```bash
     Datasets/
          coco/
               #http://images.cocodataset.org/zips/train2017.zip
               train2017/ 
               #http://images.cocodataset.org/zips/val2017.zip
               val2017/   
               #http://images.cocodataset.org/annotations/annotations_trainval2017.zip
               annotations/ 
               #http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
               stuffthingmaps/ 
     ```
  - Format the data to detecttron2 style and split it into Seen (Base) subset and Unseen (Novel) subset.
     ```bash
     python datasets/prepare_coco_stuff_164k_sem_seg.py datasets/coco

     python tools/mask_cls_collect.py datasets/coco/stuffthingmaps_detectron2/train2017_base datasets/coco/stuffthingmaps_detectron2/train2017_base_label_count.pkl
     
     python tools/mask_cls_collect.py datasets/coco/stuffthingmaps_detectron2/val2017 datasets/coco/stuffthingmaps_detectron2/val2017_label_count.pkl
     ```   
- For Pascal VOC 11k:
  - Download data from the offical dataset website and extract it like below.
  ```bash
  datasets/
     VOC2012/
          #http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
          JPEGImages/
          val.txt
          #http://home.bharathh.info/pubs/codes/SBD/download.html
          SegmentationClassAug/
          #https://gist.githubusercontent.com/sun11/2dbda6b31acc7c6292d14a872d0c90b7/raw/5f5a5270089239ef2f6b65b1cc55208355b5acca/trainaug.txt
          train.txt
          
  ```
  - Format the data to detecttron2 style and split it into Seen (Base) subset and Unseen (Novel) subset.
  ```bash
  python datasets/prepare_voc_sem_seg.py datasets/VOC2012

  python tools/mask_cls_collect.py datasets/VOC2012/annotations_detectron2/train datasets/VOC2012/annotations_detectron2/train_base_label_count.json

  python tools/mask_cls_collect.py datasets/VOC2012/annotations_detectron2/val datasets/VOC2012/annotations_detectron2/val_label_count.json
  ```
- ### Training and Evaluation

  Before training and evaluation, see the tutorial in detectron2. For example, to training a zero shot semantic segmentation model on COCO Stuff:
  
- Training with manually designed prompts:
  ```
  python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_single_prompt_bs32_60k.yaml
  ```
- Training with learned prompts:
  ```bash
  # Training prompts
  python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_proposal_classification_learn_prompt_bs32_10k.yaml --num-gpus 8 
  # Training seg model
  python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 8 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS}
  ```
  Note: the prompts training will be affected by the random seed. It is better to run it multiple times.

  For evaluation, add `--eval-only` flag to the traing command.
- Trained Model
  
  :smile: Coming soon.
