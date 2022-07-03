import pandas as pd
import argparse
import os.path
from distutils.dir_util import copy_tree
from shutil import copy, make_archive

def exctractLabels(n, dest=None, path_labels= "wnids.txt", path_classes="words.txt", ):
    labels = pd.read_table(path_labels, names = ['labels'], nrows=n)
    classes = pd.read_table(path_classes, names = ['labels', 'classes'])
    classes = classes[classes['labels'].isin(labels['labels'])]
    if dest:
        if not os.path.exists(dest):
            os.mkdir(dest)
        labels.to_csv(os.path.join(dest, "labels_"+str(n)+".txt"), index=False)
        classes.to_csv(os.path.join(dest,"classes_"+str(n)+".txt"), index=False, sep='\t')
    return labels['labels'].to_list(), classes['classes'].to_list()

def extractTrainData(labels, src, dest=None, copyImgs=False):
    frames = []
    for label in labels:
        pathBoxes = os.path.join(src,label, label+"_boxes.txt")
        frames.append(pd.read_table(pathBoxes, names = ['name', 'b0', 'b1', 'b2', 'b3']))
        frames[-1]['label'] = label
        if copyImgs and dest:
            if not os.path.exists(dest):
                os.mkdir(dest)
            trainImagesDir = os.path.join(src,label,"images")
            if os.path.exists(trainImagesDir):
                destImgs = os.path.join(dest,"images")
                copy_tree(trainImagesDir,destImgs)
    column_names = ['name', 'label', 'b0', 'b1', 'b2', 'b3']
    trainAnn = pd.concat(frames, ignore_index=True)[column_names]
    if dest:
        if not os.path.exists(dest):
            os.mkdir(dest)
        trainAnn.to_csv(os.path.join(dest,"train_annotations_"+str(len(labels))+".txt"), sep='\t', index=False)
    return trainAnn

def extractValData(labels, src, dest=None, copyImgs=False):
    column_names = ['name', 'label', 'b0', 'b1', 'b2', 'b3']
    valAnn = pd.read_table(os.path.join(src,'val_annotations.txt'), names=column_names)
    valAnn = valAnn[valAnn["label"].isin(labels)]
    if dest:
        if not os.path.exists(dest):
            os.mkdir(dest)
        if copyImgs:
            for valImage in valAnn['name']:
                srcImg = os.path.join(src,"images",valImage)
                destImg = os.path.join(dest, 'images')
                if not os.path.exists(destImg):
                    os.mkdir(os.path.join(dest, 'images'))
                copy(srcImg,destImg)    
        valAnn.to_csv(os.path.join(dest,"val_annotations_"+str(len(labels))+".txt"), sep = '\t', index=False)
    return valAnn

parser = argparse.ArgumentParser(description='Process Class Subset of Tiny Imagenet (max 200).')
parser.add_argument("-r", default="", help="root directory for the dataset")
parser.add_argument("-l", default="wnids.txt", help="labels filename")
parser.add_argument("-c", default="words.txt", help="classes filename")
parser.add_argument("-d", default="tinyImagenet", help="destination directory")
parser.add_argument("-n", type=int, default=10, help='number of class.')
parser.add_argument("-cp", type=bool, default=True, help='copy images in new folder.')
# parser.add_argument("-z", type=bool, default=True, help='zip the output.')

args = parser.parse_args()
dest = args.d + str(args.n)

labels, classes = exctractLabels(args.n, dest)
extractTrainData(labels, os.path.join(args.r, 'train'), os.path.join(dest, 'train'), args.cp)
extractValData(labels, os.path.join(args.r, 'val'), os.path.join(dest, 'val'), args.cp)