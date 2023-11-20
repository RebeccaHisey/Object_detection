import os
import cv2
import pandas
import numpy
import yaml
import argparse
import gc
from YOLOv8 import YOLOv8


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument(
        '--save_location',
        type=str,
        default='',
        help='Name of the directory where the models and results will be saved'
    )
    parser.add_argument(
        '--data_csv_file',
        type=str,
        default='',
        help='Path to the csv file containing locations for all data used in training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='type of output your model generates'
    )
    # Optim
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help='Epochs to wait for no observable improvement for early stopping of training'
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=1e-3,
        help='Epochs to wait for no observable improvement for early stopping of training'
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help='Epochs to wait for no observable improvement for early stopping of training'
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help='Size of input images as integer'
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help='Number of worker threads for data loading'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing'
    )
    parser.add_argument(
        '--include_blank',
        type=bool,
        default=False,
        help='Include images that have no labels for training'
    )
    parser.add_argument(
        '--balance',
        type=bool,
        default=True,
        help='Balance samples for training'
    )
    parser.add_argument(
        '--output_mode',
        type=str,
        default='detect',
        help='Which type of head to use for the YOLO model. One of: detect,segment'
    )
    parser.add_argument(
        '--label_name',
        type=str,
        default='Tissue',
        help='Name of the dataframe column containing labels'
    )
    return parser

def xyxy_to_yolo(img_size,bbox):
    x_centre = ((int(bbox["xmin"]) + int(bbox["xmax"])) / 2) / img_size[1]
    y_centre = ((int(bbox["ymin"]) + int(bbox["ymax"])) / 2) / img_size[0]
    width = ((int(bbox["xmax"]) - int(bbox["xmin"]))) / img_size[1]
    height = ((int(bbox["ymax"]) - int(bbox["ymin"]))) / img_size[0]
    return x_centre, y_centre, width, height

def getMaxClassCounts(data):
    class_counts = {}
    maxCount = 0
    for i in data.index:
        bboxes = eval(data["Tool bounding box"][i])
        for bbox in bboxes:
            if not bbox["class"] in class_counts:
                class_counts[bbox["class"]] = 1
            else:
                class_counts[bbox["class"]] += 1
                if class_counts[bbox["class"]] > maxCount:
                    maxCount = class_counts[bbox["class"]]
    return class_counts,maxCount

def determineNumDuplicatesNeeded(class_counts,max_count):
    numDuplicates = {}
    for key in class_counts:
        total_num = class_counts[key]
        fraction_max = round(max_count / total_num)
        numDuplicates[key] = fraction_max
    return numDuplicates

def writeLabelTextFiles(data, labelName, foldDir,inverted_class_mapping,include_blank=True,balance = True):
    linesToWrite = []
    class_counts, maxCount = getMaxClassCounts(data)
    foundCounts = dict(zip([key for key in class_counts],[0 for key in class_counts]))
    print(class_counts)
    if data["Set"][data.index[0]] == "Train" and balance:
        numDuplicates = determineNumDuplicatesNeeded(class_counts,maxCount)
    else:
        numDuplicates = dict(zip([key for key in class_counts],[1 for key in class_counts]))
    print(numDuplicates)
    for i in data.index:
        if (i- min(data.index)) %1000 == 0:
            print("\tparsed {}/{} samples".format(i- min(data.index),len(data.index)))
        filePath = os.path.join(data["Folder"][i],data["FileName"][i])
        bboxes = eval(data[labelName][i])
        classNames = [bbox["class"] for bbox in bboxes]
        if len(classNames)>0:
            maxDuplicates = max([numDuplicates[class_name] for class_name in classNames])
            for class_name in classNames:
                foundCounts[class_name]+=maxDuplicates
        else:
            maxDuplicates = 1
        if len(bboxes) == 0 and ((data["Set"][i]=="Train" and include_blank) or data["Set"][i] != "Train"):
            for j in range(maxDuplicates):
                linesToWrite.append("{}\n".format(filePath))
        elif len(bboxes) > 0:
            for j in range(maxDuplicates):
                linesToWrite.append("{}\n".format(filePath))
            file,imgExtension = filePath.split(".",-1)
            labelFilePath = file+".txt"
            img = cv2.imread(filePath)
            img_shape = img.shape
            line = ""
            for bbox in bboxes:
                x_centre, y_centre, width, height = xyxy_to_yolo(img_shape,bbox)
                class_name = inverted_class_mapping[bbox["class"]]
                line += "{} {} {} {} {}\n".format(class_name,x_centre,y_centre,width,height)
            with open(labelFilePath, "w") as f:
                f.write(line)
    print(foundCounts)
    fileName = "{}.txt".format(data["Set"][data.index[0]])
    filePath = os.path.join(foldDir,fileName)
    with open(filePath,"w") as f:
        f.writelines(linesToWrite)

def writeSegmentationLabelToTextFile(data,foldDir,labelName):
    linesToWrite = []
    for i in trainingCSV.index:
        segName = data[labelName][i]
        imgName = data["FileName"][i]
        folder_path = data["Folder"][i]
        # Read in segmentation and image
        if os.path.exists(os.path.join(folder_path, segName)):
            linesToWrite.append(os.path.join(folder_path,imgName))
            seg = cv2.imread(os.path.join(folder_path, segName), cv2.IMREAD_GRAYSCALE)

            # Convert labelmap to coordinates
            coords = labelmap_to_contour(seg)

            # Create and write to text file
            fileName,fileExtension = imgName.split(".")
            textFileName = "{}.txt".format(fileName)
            textFilePath = os.path.join(folder_path,textFileName)
            with open(textFilePath, 'w') as textFile:
                textFile.write(str(coords))
    fileName = "{}.txt".format(data["Set"][data.index[0]])
    filePath = os.path.join(foldDir, fileName)
    with open(filePath, "w") as f:
        f.writelines(linesToWrite)


def labelmap_to_contour(labelmap):
    contour_coordinates_list = []

    present_classes = np.unique(labelmap)
    img_shape = labelmap.shape

    for class_label in present_classes:

        if class_label == 0:
            continue
        # Create a binary mask for the current class
        class_mask = np.uint8(labelmap == class_label)

        # Find contours in the binary mask
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, 8, cv2.CV_32S)
        sizes = np.array(stats[:, -1])
        maxSize = np.max(sizes)
        currentIndex = 0

        # Loop through all components except background
        for i in range(num_components):
            if sizes[i] != maxSize:
                newImage = np.zeros(class_mask.shape)
                newImage[labels == i] = 1
                newImage = newImage.astype("uint8")

                contours, _ = cv2.findContours(newImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                maxindex = np.argmax([contours[i].shape[0] for i in range(len(contours))])
                contours = np.asarray(contours[maxindex])

                contour_string = "{}"
                for contour in contours:
                    for point in contour:
                        x = point[0] / img_shape[1]
                        y = point[1] / img_shape[0]
                        contour_string += " {} {}"
                contour_string+="\n"
                contour_coordinates_list.append(contour_coordinates)
    return contour_coordinates_list

def invert_class_mapping(class_mapping):
    inverted_mapping = {}
    for key in class_mapping:
        inverted_mapping[class_mapping[key]] = key
    print(inverted_mapping)
    return inverted_mapping

def removeCache(data_dir):
    cache_path = os.path.dirname(data_dir)
    cache_name = os.path.basename(data_dir)
    cache_name = cache_name+".cache"
    print("removing existing cache: {}".format(os.path.join(cache_path,cache_name)))
    if os.path.exists(os.path.join(cache_path,cache_name)):
        os.remove(os.path.join(cache_path,cache_name))

def prepareData(datacsv, labelName, fold, class_mapping, foldDir,include_blank,mode="detect"):
    config = {}
    sets = ["Train","Validation","Test"]
    inverted_class_mapping = invert_class_mapping(class_mapping)
    for learning_set in sets:
        print("Parsing {} data".format(learning_set.lower()))
        data = datacsv.loc[(datacsv["Fold"] == fold) & (datacsv["Set"] == learning_set)]
        removeCache(data["Folder"][data.index[0]])
        sample_file_path = os.path.join(foldDir,"{}.txt".format(learning_set))
        if not os.path.exists(sample_file_path):
            if mode == 'detect':
                writeLabelTextFiles(data, labelName, foldDir,inverted_class_mapping,include_blank)
            elif mode== 'segment':
                writeSegmentationLabelToTextFile(data,foldDir,labelName)
        if learning_set == "Validation":
            config["val"] = sample_file_path
        else:
            config[learning_set.lower()] = sample_file_path
    config["names"] = class_mapping
    with open(os.path.join(foldDir,"data.yaml"),"w") as f:
        yaml.dump(config,f)
    return os.path.join(foldDir,"data.yaml")

def getClassMapping(datacsv):
    class_names = []
    data = datacsv.loc[datacsv["Fold"]==0]
    for i in data.index:
        bboxes = eval(data["Tool bounding box"][i])
        if len(bboxes) > 0:
            for bbox in bboxes:
                if not bbox["class"] in class_names:
                    class_names.append(bbox["class"])
    class_names = sorted(class_names)
    class_mapping = dict(zip([i for i in range(len(class_names))],class_names))
    return class_mapping

def saveMetrics(metrics,class_mapping,foldDir,mode):
    if mode=='detect':
        class_indexes = metrics.box.ap_class_index
        maps = metrics.box.all_ap
    elif mode == 'segment':
        class_indexes = metrics.seg.ap_class_index
        maps = metrics.seg.all_ap
    linesTo_write = []
    linesTo_write.append("mAP 50:\n")
    for i in range(len(class_indexes)):
        class_name = class_mapping[class_indexes[i]]
        map50 = maps[i][0]
        linesTo_write.append("{}: {}\n".format(class_name,map50))
    linesTo_write.append("\nmAP 95:\n")
    for i in range(len(class_indexes)):
        class_name = class_mapping[class_indexes[i]]
        map95 = maps[i][-1]
        linesTo_write.append("{}: {}\n".format(class_name,map95))
    with open(os.path.join(foldDir,"test_maps.txt"),"w") as f:
        f.writelines(linesTo_write)

def main(args):
    #gpu = torch.device(args.device)
    dataCSVFile = pandas.read_csv(args.data_csv_file)
    config = {}
    class_mapping = getClassMapping(dataCSVFile)
    config["class_mapping"] = class_mapping
    config['mode'] = args.mode
    numFolds = dataCSVFile["Fold"].max() + 1
    for fold in range(0, numFolds):
        foldDir = args.save_location + "_Fold_" + str(fold)
        if not os.path.exists(foldDir):
            os.mkdir(foldDir)
        with open(os.path.join(foldDir,"config.yaml"),"w") as f:
            yaml.dump(class_mapping,f)
        dataPath = prepareData(dataCSVFile, args.label_name, fold,class_mapping,foldDir,args.include_blank)
        yolo = YOLOv8(mode)
        if not os.path.exists(os.path.join(foldDir,"train/weights/best.pt")):
            model = yolo.createModel()
            model.train(data=dataPath,
                        epochs = args.epochs,
                        patience = args.patience,
                        lr0 = args.lr0,
                        lrf = args.lrf,
                        batch = args.batch_size,
                        device=args.device,
                        workers = args.workers,
                        verbose=True,
                        cache=False,
                        project=foldDir,
                        exist_ok=True)

        yolo.loadModel(foldDir)
        model = yolo.model
        metrics = model.val(split="test",
                            device=args.device,
                            iou=0.45,
                            conf=0.25)
        saveMetrics(metrics,class_mapping,foldDir)
        del metrics
        del model
        del yolo
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOv8 training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)