
<img src="https://fiverr-res.cloudinary.com/images/q_auto,f_auto/gigs/185668043/original/8de09be1d5623082611e6733021943e7908c8891/do-photoshop-background-removal.png" alt="Italian Trulli" width="1000" height="550" style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;">


Type : Group Project
Group : Daniel Mendoza, Régis Schulze, Vincent Rolin
Duration : 2 weeks

# Background_removal_project 

In this group project we explored current possibilites to remove the background of images using deep learning models, also called image matting.

# Dataset

For this project we used the DUTS dataset (http://saliencydetection.net/duts/) to improve on pretrained models using transfer learning.

# Approach

There're three main approaches: 
- With a trimap and pretrained model  
- Without a trimap -> ModNet model
- Deep Image Matting

# Trimap

trimap_creation.ipynb is a jupiter notebook used to create trimaps given an image folder

# ModNet

- Explenation of the repo can be found on https://github.com/ZHKKKe/MODNet
- We made image matting predictions using the pretrained MODNet model(https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR) following Google - - Collab notebook : https://colab.research.google.com/drive/1GANpbKT06aEFiW-Ssx0DQnnEADcXwQG6?usp=sharing#scrollTo=JOmYOHKfgQ5Y
for_metrics.ipynb is a jupiter notebook used to compare the predictions from the Google Collab and the ground truth(MASK in DUTS folder).

# Deep Image Matting

Based on following github repo : https://github.com/adumrewal/imageMatting
Image Matting. Given an image, the code in this project can separate its foreground and background components.
This repository is to reproduce [Deep image matting](https://arxiv.org/abs/1703.03872) and is a modification to the codes used by foamliu (https://github.com/foamliu/Deep-Image-Matting-PyTorch).
