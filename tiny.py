import pandas as pd
import argparse
import os.path
from distutils.dir_util import copy_tree
from shutil import copy

parser = argparse.ArgumentParser(description='Process Class Subset of Tiny Imagenet (max 200).')
parser.add_argument("-r", default="", help="root directory for the dataset")
parser.add_argument("-labels", default="wnids.txt", help="labels path")
parser.add_argument("-val_annotations", default="val_annotations.txt", help="validation set valAnn path")
parser.add_argument("-n", type=int, default=20, help='number of class.')
parser.add_argument("-c", type=bool, default=True, help='copy images in new folder.')

args = parser.parse_args()

pathLabels = os.path.join(args.r,args.labels)
destDir = os.path.join(args.r,"tinyImagenet_"+str(args.n))
if not os.path.exists(destDir):
    os.mkdir(destDir)
srcTrainDir = os.path.join(args.r, "train")
destTrainDir = os.path.join(args.r,"tinyImagenet_"+str(args.n),"train")
if not os.path.exists(destTrainDir):
    os.mkdir(destTrainDir)
srcValDir = os.path.join(args.r, "val")
destValDir = os.path.join(args.r,"tinyImagenet_"+str(args.n),"val")
if not os.path.exists(destValDir):
    os.mkdir(destValDir)

labels = pd.read_table(pathLabels, names = ['labels'], nrows=args.n)
labels.to_csv("labels_"+str(args.n)+".txt", index=False)

path_val_annotations = os.path.join(srcValDir, args.val_annotations)
column_names = ['name', 'label', 'b0', 'b1', 'b2', 'b3']
valAnn = pd.read_table(path_val_annotations, names=column_names)
valAnn = valAnn[valAnn["label"].isin(labels['labels'])]
valAnn.to_csv(os.path.join(destValDir,"val_annotations_"+str(args.n)+".txt"), sep = '\t', index=False)

frames = []
for label in labels['labels']:
    pathBoxes = os.path.join(srcTrainDir,label, label+"_boxes.txt")
    frames.append(pd.read_table(pathBoxes, names = ['name', 'b0', 'b1', 'b2', 'b3']))
    frames[-1]['label'] = label

trainAnn = pd.concat(frames, ignore_index=True)[column_names]
trainAnn.to_csv(os.path.join(destTrainDir,"train_annotations_"+str(args.n)+".txt"), sep='\t', index=False)

if args.c:
    for label in labels['labels']:
        trainImagesDir = os.path.join(srcTrainDir,label,"images")
        if os.path.exists(trainImagesDir):
            dst = os.path.join(destTrainDir,"images")
            copy_tree(trainImagesDir,dst)
            # copy_tree(trainImagesDir,destination)

    for valImage in valAnn['name']:
        src = os.path.join(srcValDir,"images",valImage)
        dest = os.path.join(destValDir, 'images')
        if not os.path.exists(dest):
            os.mkdir(dest)
        copy(src,dest)