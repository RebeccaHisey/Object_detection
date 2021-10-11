import os
import cv2
import sys
import numpy
import tensorflow
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from keras_frcnn import resnet as nn
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

''' 
Faster Region-based CNN implementation for 3DSlicer - Deep Learn Live

Originally developed by Rebecca Hisey for the Laboratory of Percutaneous Surgery, Queens University, Kingston, ON

Model description: 
    Faster RCNN is an object detection network that uses resnet50 as a base network in addition to a region proposal network and a classifier.
    ResNet implementation is pulled directly from keras.applications and is pretrained with ImageNet weights.
    The network takes a 3-channel image and will return either an image with bounding boxes drawn, or a string with the name and location of the 
    bounding boxes.
'''

class Faster_RCNN():
    def __init__(self):
        self.verbose = True

        self.network = 'resnet50'

        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False
        self.anchor_box_scales = [84, 253, 164, 396, 263]
        self.anchor_box_ratios = [[6, 5], [7, 6], [8, 5], [1, 1], [3, 1]]
        self.im_size = 600

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 4

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
    
        self.baseNetwork = None
        self.regionProposalNetwork = None
        self.classifierNetwork = None
        self.model_all = None
        self.class_mapping = {}
    
    def loadModel(self,modelFolder):
        self.loadBaseModel(modelFolder)
        self.loadRegionProposalNetwork(modelFolder)
        self.loadClassifierNetwork(modelFolder)
        self.loadClassMapping(modelFolder)

    def loadBaseModel(self,modelFolder):
        structureFileName = 'base.json'
        weightsFileName = 'base.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
            self.baseNetwork = model_from_json(JSONModel)
        self.baseNetwork.load_weights(os.path.join(modelFolder, weightsFileName))
            
    def loadRegionProposalNetwork(self,modelFolder):
        structureFileName = 'rpn.json'
        weightsFileName = 'rpn.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
            self.regionProposalNetwork = model_from_json(JSONModel)
        self.regionProposalNetwork.load_weights(os.path.join(modelFolder, weightsFileName))
                
    def loadClassifierNetwork(self,modelFolder):
        structureFileName = 'classifier.json'
        weightsFileName = 'classifier.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
            self.classifierNetwork = model_from_json(JSONModel)
        self.classifierNetwork.load_weights(os.path.join(modelFolder, weightsFileName))
            
    def loadClassMapping(modelFolder):
        with open(os.path.join(modelFolder,"labels.txt"),'r') as f:
            labels = f.read()
            labels = self.labels.split(sep="\n")
        self.class_mapping = dict(zip([i for i in range(len(labels))],labels))

    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224))
        resized = numpy.expand_dims(resized, axis=0)
        toolClassification = self.cnnModel.predict(numpy.array(resized))
        labelIndex = numpy.argmax(toolClassification)
        label = self.labels[labelIndex]
        networkOutput = str(label) + str(toolClassification)
        return networkOutput

    def createModel(self,num_classes):
        input_shape_img = (None, None, 3)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(None, 4))

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.anchor_box_scales) * len(self.anchor_box_ratios)
        rpn = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(shared_layers, roi_input, self.num_rois, nb_classes=num_classes, trainable=True)

        self.model_rpn = Model(img_input, rpn[:2])
        self.model_classifier = Model([img_input, roi_input], classifier)

        # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
        self.model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        return(self.model_rpn,self.model_classifier,self.model_all)

    def saveModel(self,trainedModel,saveLocation):
        JSONmodel = trainedModel.to_json()
        structureFileName = 'frcnn.json'
        with open(os.path.join(saveLocation,structureFileName),"w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)
