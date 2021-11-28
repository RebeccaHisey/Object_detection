import os
import cv2
import sys
import numpy
import pickle
import tensorflow
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import keras_frcnn.roi_helpers as roi_helpers
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
    

        self.model_rpn = None
        self.model_classifier = None
        self.class_mapping = {}
    
    def loadModel(self,modelFolder,model_name):
        with open(os.path.join(modelFolder,"config.pickle"), 'rb') as f_in:
            self.C = pickle.load(f_in)
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False

        if self.C.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif self.C.network == 'vgg':
            import keras_frcnn.vgg as nn

        self.class_mapping = self.C.class_mapping

        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)

        self.class_mapping = {v: k for k, v in self.class_mapping.items()}
        self.C.num_rois = int(self.num_rois)

        if self.C.network == 'resnet50':
            num_features = 1024
        elif self.C.network == 'vgg':
            num_features = 512

        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.class_mapping),
                                   trainable=True)

        self.model_rpn = Model(img_input, rpn_layers)

        self.model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(modelFolder))
        self.loadRegionProposalNetwork(modelFolder)
        self.loadClassifierNetwork(modelFolder)

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')
            
    def loadRegionProposalNetwork(self,modelFolder):
        weightsFileName = 'frcnn.hdf5'
        modelFolder = modelFolder.replace("'","")
        self.model_rpn.load_weights(os.path.join(modelFolder, weightsFileName), by_name=True)
                
    def loadClassifierNetwork(self,modelFolder):
        weightsFileName = 'frcnn.hdf5'
        modelFolder = modelFolder.replace("'", "")
        self.model_classifier.load_weights(os.path.join(modelFolder, weightsFileName), by_name=True)
            
    def loadClassMapping(self,modelFolder):
        with open(os.path.join(modelFolder,"labels.txt"),'r') as f:
            labels = f.read()
            labels = labels.split(sep="\n")
        self.class_mapping = dict(zip([i for i in range(len(labels))],labels))
        self.num_classes = len(labels)

    def format_img_size(self,img):
        """ formats the image size based on config """
        img_min_side = float(self.C.im_size)
        (height, width, _) = img.shape

        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    def format_img_channels(self,img):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(numpy.float32)
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor
        img = numpy.transpose(img, (2, 0, 1))
        img = numpy.expand_dims(img, axis=0)
        return img

    def format_img(self,img):
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size(img)
        img = self.format_img_channels(img)
        return img, ratio

    def get_real_coordinates(self,ratio, x1, y1, x2, y2):

        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2, real_y2)

    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_threshold = 0.8
        allConfs = {}

        X, ratio = self.format_img(image)

        X = numpy.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        networkOutput = []

        for jk in range(R.shape[0] // self.C.num_rois + 1):
            ROIs = numpy.expand_dims(R[self.C.num_rois * jk:self.C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
                ROIs_padded = numpy.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if numpy.max(P_cls[0, ii, :]) < bbox_threshold or numpy.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = self.class_mapping[numpy.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                    allConfs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = numpy.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [self.C.rpn_stride * x, self.C.rpn_stride * y, self.C.rpn_stride * (x + w), self.C.rpn_stride * (y + h)])
                probs[cls_name].append(numpy.max(P_cls[0, ii, :]))
                allConfs[cls_name].append(P_cls[0, ii, :])

        for key in bboxes:
            bbox = numpy.array(bboxes[key])

            new_boxes, new_probs, new_allConfs = roi_helpers.non_max_suppression_fast(bbox, numpy.array(probs[key]),
                                                                                      allConfs[key], overlap_thresh=0.5)
            curr_bestProb = 0
            for jk in range(new_boxes.shape[0]):
                if new_probs[jk] > curr_bestProb and new_probs[jk] > 0.1:
                    curr_bestProb = new_probs[jk]
                    (x1, y1, x2, y2) = new_boxes[jk, :]

                    (real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
                    networkOutput.append({"class":key,"xmin":real_x1,"xmax":real_x2,"ymin":real_y1,"ymax":real_y2})

        return str(networkOutput)

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
