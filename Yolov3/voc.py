import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle

def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(os.listdir(ann_dir)):
            img = {'object':[]}

            try:
                tree = ET.parse(ann_dir + ann)
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)
                continue
            
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
                            
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels


def parse_voc_annotation_deepLearnLive(datasetTextFile, cache_name, labels=[]):
    fake_cache = "C:/doesnotexist"
    if os.path.exists(fake_cache):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        with open(datasetTextFile,"rb") as dataFile:
            for line in dataFile.readlines():
                strLine = str(line)
                img = {'object': []}
                strLine = strLine.replace('\n',"")
                strLine = strLine.replace("'","")
                try:
                    filepath, bboxes = strLine.split(" ",1)
                    bboxes = bboxes.split(" ")
                    img['filename'] = filepath
                    if 'bD' in filepath:
                        img['filename'] = filepath.replace('bD','D')
                    # img['width'] = int(width)
                    # img['height'] = int(height)
                    for bbox in bboxes:
                        classnum, xmin, xmax, ymin, ymax = bbox.split(",")
                        ymax = ymax.replace("\\r\\n", '')
                        obj = {}
                        try:
                            obj['name'] = labels[int(classnum)]
                        except:
                            print(classnum)
                            print(bbox)

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                        obj['xmin'] = int(xmin)
                        obj['ymin'] = int(ymin)
                        obj['xmax'] = int(xmax)
                        obj['ymax'] = int(ymax)

                    if len(img['object']) > 0:
                        all_insts += [img]
                except:
                    print("Skipping file: {}\nFile has no bounding boxes".format(strLine))
        '''cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)'''

    return all_insts, seen_labels