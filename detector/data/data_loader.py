import os

import numpy as np
import tensorflow as tf
import albumentations as A
import cv2 
import pandas as pd

def augmentation(image,bboxes,classes_ids):
  transform = A.Compose(
    [
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.Blur(blur_limit=4, p=0.5),
    A.RandomCrop(200,200,p=0.5),
    A.IAAPerspective(p=0.5),
    A.Resize(420,420,p=1.0)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['classes_ids']),
   )
  transformed = transform(image=image, bboxes=bboxes, classes_ids=classes_ids)
  return transformed['image'],transformed['bboxes'],transformed['classes_ids']

def tf_augmentation(image,bboxes,classes_ids):
  [image,bboxes,classes_ids]=tf.numpy_function(augmentation, [image,bboxes,classes_ids],[tf.float32,tf.float64,tf.float64] )
  return image,bboxes,classes_ids

def get_bboxes_ids(image_name):
  annot = pd.read_csv("data/raw/annotations.csv")
  classes = {'helmet':0,'head':1}
  data = annot.loc[annot['filename'] == image_name]
  bboxes = data[['xmin','ymin','xmax','ymax']].to_numpy()
  classes_ids = data['class'].to_numpy()
  classes_ids = np.vectorize(classes.get)(classes_ids)
  return bboxes,classes_ids

def images_generator():
  while True:
    images_path="data/raw/images"
    image_names=os.listdir(images_path)
    image_nb = np.random.randint(len(image_names))
    image_name = image_names[image_nb]
    bboxes,classes_ids = get_bboxes_ids(image_name)
    image_path = images_path +'/' + image_name
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yield image,bboxes,classes_ids

def get_dataset(cf, mode="train"):
    # TODO: create your dataset using tf.data.Dataset
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_generator(
    images_generator,
    output_types=(tf.float32,tf.float64,tf.float64))
    dataset = dataset.map(tf_augmentation,num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    #TODO: Split the dataset to train and validation datasets
    return dataset