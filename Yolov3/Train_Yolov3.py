#! /usr/bin/env python
import tensorflow as tf
import argparse
import math
import os
import numpy as np
import json
import pandas
from voc import parse_voc_annotation, parse_voc_annotation_deepLearnLive
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
tfVersion = tf.__version__
tfVersion = int(tfVersion[0])
if tfVersion >1:
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ProgbarLogger
    from tensorflow.keras.optimizers import Adam
else:
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
    from keras.models import load_model
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model


FLAGS = None

'''config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)'''

class Train_Yolov3:
    def loadData(self,fold,set):
        entries = self.dataCSVFile.loc[(self.dataCSVFile["Fold"] == fold) & (self.dataCSVFile["Set"] == set)]
        textFile = os.path.join("{}_Fold_{}".format(self.saveLocation, fold), set + ".txt")
        labelFile = os.path.join("{}_Fold_{}".format(self.saveLocation, fold),  "classes.txt")
        if not os.path.exists(textFile):
            self.writeDataToTextFile(entries,textFile,labelFile)
        return textFile,labelFile

    def writeDataToTextFile(self,datacsv,textFile,labelFile):
        labels = []
        trainLines = []
        for i in datacsv.index:
            newLine = ''
            filePath = os.path.join(datacsv["Folder"][i], datacsv["FileName"][i])
            newLine += filePath
            boundingBoxes = str(datacsv[self.labelName][i])
            strBoundBox = boundingBoxes.replace(" ", "")
            strBoundBox = strBoundBox.replace("'", "")
            strBoundBox = strBoundBox.replace("[", "")
            strBoundBox = strBoundBox.replace("]", "")
            boundingBoxes = []
            if strBoundBox != "":
                listBoundBox = strBoundBox.split("},{")
                for boundingBox in listBoundBox:
                    boundingBox = boundingBox.replace("{", "")
                    boundingBox = boundingBox.replace("}", "")
                    keyEntryPairs = boundingBox.split(",")
                    boundingBoxDict = {}
                    for pair in keyEntryPairs:
                        key, entry = pair.split(":")
                        if entry.isnumeric():
                            boundingBoxDict[key] = int(entry)
                        else:
                            boundingBoxDict[key] = entry
                    x1 = boundingBoxDict["xmin"]
                    x2 = boundingBoxDict["xmax"]
                    y1 = boundingBoxDict["ymin"]
                    y2 = boundingBoxDict["ymax"]
                    xmin = min(x1, x2)
                    xmax = max(x1, x2)
                    ymin = min(y1, y2)
                    ymax = max(y1, y2)
                    if boundingBoxDict["class"] != "nothing":
                        #classnum = [i for i in range(len(labels)) if labels[i] == boundingBoxDict["class"]][0]
                        if not boundingBoxDict["class"] in labels:
                            labels.append(boundingBoxDict["class"])
                        bboxStr = " {},{},{},{},{}".format(boundingBoxDict["class"], xmin, xmax, ymin, ymax)
                        newLine += bboxStr
                newLine += '\n'
                trainLines.append(newLine)

        trainLines[-1] = trainLines[-1].replace('\n', '')
        labels = sorted(labels)
        for i in range(len(trainLines)):
            matchingLabels = []
            matchingLabelLengths = []
            matchingLabelIndexes = []
            for j in range(len(labels)):
                labelName = labels[j]
                if labelName in trainLines[i]:
                    matchingLabels.append(labelName)
                    matchingLabelLengths.append(len(labelName))
                    matchingLabelIndexes.append(j)
            while len(matchingLabels) != 0:
                longestLength = max(matchingLabelLengths)
                longestLengthIndex = matchingLabelLengths.index(longestLength)
                longestLabel = matchingLabels[longestLengthIndex]
                longestLabelIndex = matchingLabelIndexes[longestLengthIndex]
                matchingLabels.remove(longestLabel)
                matchingLabelLengths.remove(longestLength)
                matchingLabelIndexes.remove(longestLabelIndex)
                trainLines[i] = trainLines[i].replace(longestLabel,str(longestLabelIndex))

        with open(textFile, "w") as f:
            f.writelines(trainLines)

        labels = [i+"\n" for i in labels]
        labels[-1] = labels[-1].replace("\n","")

        if not os.path.exists(labelFile):
            with open(labelFile,"w") as f:
                f.writelines(labels)

    def create_training_instances(self,
            trainTextFile,
            train_cache,
            validTextFile,
            valid_cache,
            testTextFile,
            labelFile,
    ):
        # parse labels
        with open(os.path.join(labelFile), "r") as f:
            labels = f.readlines()
        labels = sorted([x.replace("\n", "") for x in labels])
        print(labels)
        # parse annotations of the training set
        train_ints, train_labels = parse_voc_annotation_deepLearnLive(trainTextFile, train_cache, labels)
        test_ints,_ = parse_voc_annotation_deepLearnLive(testTextFile,None,labels)
        # parse annotations of the validation set, if any, otherwise split the training set
        if os.path.exists(validTextFile):
            valid_ints, valid_labels = parse_voc_annotation_deepLearnLive(validTextFile, valid_cache, labels)
        else:
            print("valid_annot_folder not exists. Splitting the training set.")

            train_valid_split = int(0.8 * len(train_ints))
            np.random.seed(0)
            np.random.shuffle(train_ints)
            np.random.seed()

            valid_ints = train_ints[train_valid_split:]
            train_ints = train_ints[:train_valid_split]

        # compare the seen labels with the given labels in config.json
        if len(labels) > 0:
            overlap_labels = set(labels).intersection(set(train_labels.keys()))

            print('Seen labels: \t' + str(train_labels) + '\n')
            print('Given labels: \t' + str(labels))

            # return None, None, None if some given label is not in the dataset
            if len(overlap_labels) < len(labels):
                print('Some labels have no annotations! Please revise the list of labels in the config.json.')
                return None, None, None
        else:
            print('No labels are provided. Train on all seen labels.')
            print(train_labels)
            labels = train_labels.keys()

        max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

        return train_ints, valid_ints, test_ints,sorted(labels), max_box_per_image, train_labels


    def create_callbacks(self,saved_weights_name, tensorboard_logs, model_to_save):
        makedirs(tensorboard_logs)

        early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=7,
            mode='min',
            verbose=1
        )
        checkpoint = CustomModelCheckpoint(
            model_to_save=model_to_save,
            filepath=saved_weights_name,  # + '{epoch:02d}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            period=1
        )
        reduce_on_plateau = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2,
            verbose=1,
            mode='min',
            epsilon=0.01,
            cooldown=0,
            min_lr=0
        )
        tensorboard = CustomTensorBoard(
            log_dir=tensorboard_logs,
            write_graph=True,
            write_images=True,
            update_freq="batch"
        )
        progBar = ProgbarLogger(count_mode='steps')
        return [early_stop, checkpoint, reduce_on_plateau, tensorboard]


    def create_model(self,
            nb_class,
            anchors,
            max_box_per_image,
            max_grid, batch_size,
            warmup_batches,
            ignore_thresh,
            multi_gpu,
            saved_weights_name,
            lr,
            grid_scales,
            obj_scale,
            noobj_scale,
            xywh_scale,
            class_scale
    ):
        if multi_gpu > 1:
            with tf.device('/cpu:0'):
                template_model, infer_model = create_yolov3_model(
                    nb_class=nb_class,
                    anchors=anchors,
                    max_box_per_image=max_box_per_image,
                    max_grid=max_grid,
                    batch_size=batch_size // multi_gpu,
                    warmup_batches=warmup_batches,
                    ignore_thresh=ignore_thresh,
                    grid_scales=grid_scales,
                    obj_scale=obj_scale,
                    noobj_scale=noobj_scale,
                    xywh_scale=xywh_scale,
                    class_scale=class_scale
                )
        else:
            template_model, infer_model = create_yolov3_model(
                nb_class=nb_class,
                anchors=anchors,
                max_box_per_image=max_box_per_image,
                max_grid=max_grid,
                batch_size=batch_size,
                warmup_batches=warmup_batches,
                ignore_thresh=ignore_thresh,
                grid_scales=grid_scales,
                obj_scale=obj_scale,
                noobj_scale=noobj_scale,
                xywh_scale=xywh_scale,
                class_scale=class_scale
            )

            # load the pretrained weight if exists, otherwise load the backend weight only
        template_model.build((416, 416, 3))
        if os.path.exists(saved_weights_name):
            print("\nLoading pretrained weights.\n")
            template_model.load_weights(saved_weights_name, by_name=True)
        else:
            currentDir = os.path.dirname(os.path.realpath(__file__))
            template_model.load_weights(os.path.join(currentDir, "backend.h5"), by_name=True)

        if multi_gpu > 1:
            train_model = multi_gpu_model(template_model, gpus=multi_gpu)
        else:
            train_model = template_model

        optimizer = Adam(lr=lr, clipnorm=0.001)
        train_model.compile(loss=dummy_loss, optimizer=optimizer)

        infer_model.build((416,416,3))
        return train_model, infer_model


    def train(self):
        self.saveLocation = FLAGS.save_location
        self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
        self.numFolds = self.dataCSVFile["Fold"].max() + 1
        self.labelName = "Tool bounding box"
        config_path = FLAGS.conf
        balanceData = False

        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())

        ###############################
        #   Parse the annotations
        ###############################

        for fold in range(0,self.numFolds):
            foldDir = self.saveLocation + "_Fold_" + str(fold)
            if not os.path.exists(foldDir):
                os.mkdir(foldDir)
            trainTextFile,labelFile = self.loadData(fold,"Train")
            valTextFile, _ = self.loadData(fold,"Validation")
            testTextFile, _ = self.loadData(fold,"Test")
            train_ints, valid_ints, test_ints, labels, max_box_per_image, class_counts = self.create_training_instances(
                trainTextFile,
                config['train']['cache_name'],
                valTextFile,
                config['valid']['cache_name'],
                testTextFile,
                labelFile
            )
            print('\nTraining on: \t' + str(labels) + '\n')

            ###############################
            #   Create the generators
            ###############################
            train_generator = BatchGenerator(
                instances=train_ints,
                anchors=config['model']['anchors'],
                labels=labels,
                downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
                max_box_per_image=max_box_per_image,
                batch_size=config['train']['batch_size'],
                min_net_size=config['model']['min_input_size'],
                max_net_size=config['model']['max_input_size'],
                shuffle=True,
                jitter=0.3,
                norm=normalize,
                modeName="Train"
            )

            valid_generator = BatchGenerator(
                instances=valid_ints,
                anchors=config['model']['anchors'],
                labels=labels,
                downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
                max_box_per_image=max_box_per_image,
                batch_size=config['train']['batch_size'],
                min_net_size=config['model']['min_input_size'],
                max_net_size=config['model']['max_input_size'],
                shuffle=False,
                jitter=0.0,
                norm=normalize,
                modeName="Test"
            )

            test_generator = BatchGenerator(
                instances=test_ints,
                anchors=config['model']['anchors'],
                labels=labels,
                downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
                max_box_per_image=max_box_per_image,
                batch_size=config['train']['batch_size'],
                min_net_size=config['model']['min_input_size'],
                max_net_size=config['model']['max_input_size'],
                shuffle=False,
                jitter=0.0,
                norm=normalize,
                modeName="Test"
            )


            ###############################
            #   Create the model
            ###############################
            if os.path.exists(config['train']['saved_weights_name']):
                config['train']['warmup_epochs'] = 0
            warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

            os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
            multi_gpu = len(config['train']['gpus'].split(','))

            train_model, infer_model = self.create_model(
                nb_class=len(labels),
                anchors=config['model']['anchors'],
                max_box_per_image=max_box_per_image,
                max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
                batch_size=config['train']['batch_size'],
                warmup_batches=warmup_batches,
                ignore_thresh=config['train']['ignore_thresh'],
                multi_gpu=multi_gpu,
                saved_weights_name=os.path.join(foldDir, config['train']['saved_weights_name']),
                lr=config['train']['learning_rate'],
                grid_scales=config['train']['grid_scales'],
                obj_scale=config['train']['obj_scale'],
                noobj_scale=config['train']['noobj_scale'],
                xywh_scale=config['train']['xywh_scale'],
                class_scale=config['train']['class_scale'],
            )

            ###############################
            #   Kick off the training
            ###############################
            callbacks = self.create_callbacks(os.path.join(foldDir, config['train']['saved_weights_name']),
                                         os.path.join(foldDir,config['train']['tensorboard_dir']), infer_model)

            train_model.fit(
                x                = train_generator,
                validation_data  = valid_generator,
                steps_per_epoch  = len(train_generator),# * config['train']['train_times'],
                epochs           = config['train']['nb_epochs'] + config['train']['warmup_epochs'],
                verbose          = 2,
                callbacks        = callbacks,
                workers          = 4,
                max_queue_size   = 8
            )

            # make a GPU version of infer_model for evaluation
            if multi_gpu > 0:
                if tfVersion >1:
                    infer_model.load_weights(os.path.join(foldDir, config['train']['saved_weights_name']), by_name=True)
                else:
                    load_model(os.path.join(foldDir, config['train']['saved_weights_name']))

            ###############################
            #   Run the evaluation
            ###############################
            # compute mAP for all the classes
            average_precisions_val = evaluate(infer_model, valid_generator,save_path=foldDir,mode="Validation")
            valMapLines = []
            for label, average_precision in average_precisions_val.items():
                valMapLines.append(labels[label] + ': {:.4f}\n'.format(average_precision))
            valMapLines.append('\nValidation mAP: {:.4f}'.format(sum(average_precisions_val.values()) / len(average_precisions_val)))

            with open(os.path.join(foldDir,"validation_mAPs.txt"),"w") as f:
                f.writelines(valMapLines)

            average_precisions_test = evaluate(infer_model, test_generator,save_path=foldDir,mode="Test")
            testMapLines = []
            # print the score
            for label, average_precision in average_precisions_test.items():
                testMapLines.append(labels[label] + ': {:.4f}\n'.format(average_precision))
            testMapLines.append('\nTest mAP: {:.4f}'.format(sum(average_precisions_test.values()) / len(average_precisions_test)))

            with open(os.path.join(foldDir,"test_mAPs.txt"),"w") as f:
                f.writelines(testMapLines)

if __name__ == '__main__':
    currentDir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    parser.add_argument('-c',
        '--conf',
        type=str,
        default=os.path.join(currentDir,"config.json"),
        help='path to configuration file'
    )
    parser.add_argument(
        '--data_csv_file',
        type=str,
        default='',
        help='Path to the csv file containing locations for all data used in training'
    )
    parser.add_argument(
        '--save_location',
        type=str,
        default='',
        help='Directory where you would like the trained model to be saved'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tm = Train_Yolov3()
    tm.train()
