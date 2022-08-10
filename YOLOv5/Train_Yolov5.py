# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import argparse
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import pandas

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)
from utils.loggers import Loggers
# from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class TrainYolov5():

    def __init__(self):
        opt = self.parse_opt(True)
        self.save_location, self.epochs, self.batch_size, self.weights, self.single_cls, self.evolve, self.data, self.cfg, self.resume, self.noval, self.nosave, self.workers, self.freeze, self.size= \
            Path(opt.save_location), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data_csv_file, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.img_size

    def getSetVideos(self, setName, csv, fold):
        """returns the list of videos being used for each set in the fold

        Args:
            setName (str): one of "Test", "Train", "Validation"
            csv (dataframe): dataframe of the master csv containing all the data
            fold (str): fold number #TODO: check if it is str or int

        Returns:
            array: contains all the video folders used for the set
        """
        entries = csv.loc[(csv["Fold"] == fold) & (csv["Set"] == setName)]
        folders = entries["Folder"].unique()
        return folders.tolist()

    def loadData(self, fold,dataCSVFile):
        """reads the master csv and directs the data processessing

        Args:
            fold (str): fold number
            set (str): one of "Test", "Train", "Validation"

        Returns:
            _type_: _description_
        """
        if not os.path.exists(self.save_location):
            os.path.mkdirs(self.save_location)
        dataCSVFile = pandas.read_csv(dataCSVFile)
        classMapping = self.writeDataToTextFile(dataCSVFile)
        self.writeYaml(classMapping, dataCSVFile, fold)

    def writeYaml(self, classMapping,csv, fold):
        """writes the data.yaml file for the fold

        Args:
            classMapping (dict): contains all the class names as values and their numerical indices as keys
            csv (dataframe): the master csv containing all the bounding box data
            fold (str): fold number
        """
        self.outFilePath = os.path.join(self.save_location, "data.yaml")
        outFile = os.path.join(self.outFilePath)
        trainFolders = self.getSetVideos("Train", csv, fold)
        valFolders = self.getSetVideos("Validation", csv, fold)
        testFolders = self.getSetVideos("Test", csv, fold)
        classNames = []
        for i in classMapping.keys():
            classNames.append(i)
        numClasses = len(classMapping)
        
        lines = [f"train: {str(trainFolders)}\n",f"val: {str(valFolders)}\n",f"test: {str(testFolders)}\n",f"nc: {numClasses}\n",f"names: {classNames}\n"]

        with open(outFile, 'w') as yamlFile:
            for line in lines:
                yamlFile.write(line)


    def writeDataToTextFile(self, datacsv):
        """converts the data from DLL format into individual txt files for each directory

        Args:
            datacsv (dataframe): master csv containing labels and filepaths
        """
        classMapping = dict()
        classCount = -1

        for i in datacsv.index:
            lines = []

            # file path of the desired txt file will be the folder plus the filename with .txt extension
            txtFilePath = os.path.join(datacsv['Folder'][i], (datacsv['FileName'][i].strip(".jpg") + ".txt"))

            boundingBoxes = str(datacsv["Tool bounding box"][i])
            # turn the list of bounding boxes into a string
            strBoundBox = boundingBoxes.replace(" ", "")
            strBoundBox = strBoundBox.replace("'", "")
            strBoundBox = strBoundBox.replace("[", "")
            strBoundBox = strBoundBox.replace("]", "")

            boundingBoxes = []

            # make sure the image has associated annotations
            if strBoundBox != "":

                # convert the string bounding box into the invdividual labels
                listBoundBox = strBoundBox.split("},{")

                # iterate through each individual bounding box associated with the image
                for boundingBox in listBoundBox:

                    # convert the dictionary into a string
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

                    # convert the coordinates to YOLO form
                    x1 = boundingBoxDict["xmin"]
                    x2 = boundingBoxDict["xmax"]
                    y1 = boundingBoxDict["ymin"]
                    y2 = boundingBoxDict["ymax"]
                    bbox = [x1,x2,y1,y2]
                    for i in range(0,len(bbox)):
                        if bbox[i] == '-1':
                            bbox[i] = 0

                    x,y,w,h = self.convertCoordinates(bbox)
                    
                    className = boundingBoxDict["class"]
                    if className != "nothing":
                        if not className in classMapping.keys():
                            # add classes to class mapping
                            classCount += 1
                            classMapping[className] = classCount

                        # put together all the bounding box data in YOLO form
                        classIndex = classMapping[boundingBoxDict["class"]]
                        lines.append(f'{classIndex} {x} {y} {w} {h}')

            # do not overwrite any existing text files
            if not os.path.exists(txtFilePath):
                with open((txtFilePath), 'w') as yoloFile:
                    for line in lines:
                        yoloFile.write("%s\n" % line)
            else:
                pass
        
        # return the number of classes
        return classMapping

    def convertCoordinates(self,box):
        """
        converts coordinates from (x1,y1,x2,y2) to YOLO format
        box: (x1,y1,x2,y2)
        """
        for i in range(0,len(box)):
            if int(box[i])<0:
                box[i] = 0
        dw = 1.0/self.size[0]
        dh = 1.0/self.size[1]
        x = (box[0]+box[1])/2.0
        y = (box[2]+box[3])/2.0
        w = box[1]-box[0]
        h = box[3]-box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)
        
    def train(self, hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary

        self.loadData(self.fold, opt.data_csv_file)
        self.data = self.outFilePath
        callbacks.run('on_pretrain_routine_start')

        # Directories
        w = self.save_location / 'weights'  # weights dir
        (w.parent if self.evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

        # Save run settings
        if not self.evolve:
            with open(self.save_location / 'hyp.yaml', 'w') as f:
                yaml.safe_dump(hyp, f, sort_keys=False)
            with open(self.save_location / 'opt.yaml', 'w') as f:
                yaml.safe_dump(vars(opt), f, sort_keys=False)

        # Loggers
        data_dict = None
        if RANK in {-1, 0}:
            loggers = Loggers(self.save_location, self.weights, opt, hyp, LOGGER)  # loggers instance
            # if loggers.wandb:
            #     data_dict = loggers.wandb.data_dict
            #     if resume:
            #         weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

        # Config
        plots = not self.evolve and not opt.noplots  # create plots
        cuda = device.type != 'cpu'
        init_seeds(opt.seed + 1 + RANK, deterministic=True)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = data_dict or check_dataset(self.data)  # check if None
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if self.single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if self.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {self.data}'  # check
        is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

        # Model
        check_suffix(self.weights, '.pt')  # check weights
        pretrained = str(self.weights).endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                self.weights = attempt_download(self.weights)  # download if not found locally
            ckpt = torch.load(self.weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model = Model(self.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            exclude = ['anchor'] if (self.cfg or hyp.get('anchors')) and not self.resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {self.weights}')  # report
        else:
            model = Model(self.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        amp = check_amp(model)  # check AMP

        # Freeze
        self.freeze = [f'model.{x}.' for x in (self.freeze if len(self.freeze) > 1 else range(self.freeze[0]))]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in self.freeze):
                LOGGER.info(f'freezing {k}')
                v.requires_grad = False

        # Image size
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # Batch size
        if RANK == -1 and self.batch_size == -1:  # single-GPU only, estimate best batch size
            self.batch_size = check_train_batch_size(model, imgsz, amp)
            loggers.on_params_update({"batch_size": self.batch_size})

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / self.batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= self.batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
        optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

        # Scheduler
        if opt.cos_lr:
            lf = one_cycle(1, hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        else:
            lf = lambda x: (1 - x / self.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(model) if RANK in {-1, 0} else None

        # Resume
        start_epoch, best_fitness = 0, 0.0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            if self.resume:
                assert start_epoch > 0, f'{self.weights} training to {self.epochs} epochs is finished, nothing to resume.'
            if self.epochs < start_epoch:
                LOGGER.info(f"{self.weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, csd

        # DP mode
        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if opt.sync_bn and cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        train_loader, dataset = create_dataloader(train_path,
                                                imgsz,
                                                self.batch_size // WORLD_SIZE,
                                                gs,
                                                self.single_cls,
                                                hyp=hyp,
                                                augment=True,
                                                cache=None if opt.cache == 'val' else opt.cache,
                                                rect=opt.rect,
                                                rank=LOCAL_RANK,
                                                workers=self.workers,
                                                image_weights=opt.image_weights,
                                                quad=opt.quad,
                                                prefix=colorstr('train: '),
                                                shuffle=True)
        mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
        nb = len(train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {self.data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in {-1, 0}:
            val_loader = create_dataloader(val_path,
                                        imgsz,
                                        self.batch_size // WORLD_SIZE * 2,
                                        gs,
                                        self.single_cls,
                                        hyp=hyp,
                                        cache=None if self.noval else opt.cache,
                                        rect=True,
                                        rank=-1,
                                        workers=self.workers * 2,
                                        pad=0.5,
                                        prefix=colorstr('val: '))[0]

            if not self.resume:
                labels = np.concatenate(dataset.labels, 0)
                # c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if plots:
                    plot_labels(labels, names, self.save_location)

                # Anchors
                if not opt.noautoanchor:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end')

        # DDP mode
        if cuda and RANK != -1:
            model = smart_DDP(model)

        # Model attributes
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        stopper, stop = EarlyStopping(patience=opt.patience), False
        compute_loss = ComputeLoss(model)  # init loss class
        callbacks.run('on_train_start')
        LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_location)}\n"
                    f'Starting training for {self.epochs} epochs...')
        for epoch in range(start_epoch, self.epochs):  # epoch ------------------------------------------------------------------
            callbacks.run('on_train_epoch_start')
            model.train()

            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(3, device=device)  # mean losses
            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)
            LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
            if RANK in {-1, 0}:
                pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                callbacks.run('on_train_batch_start')
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / self.batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni - last_opt_step >= accumulate:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                        (f'{self.epochs}/{self.epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots)
                    if callbacks.stop_training:
                        return
                # end batch ------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()

            if RANK in {-1, 0}:
                # mAP
                callbacks.run('on_train_epoch_end', epoch=epoch)
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or stopper.possible_stop
                if not self.noval or final_epoch:  # Calculate mAP
                    results, maps, _ = val.run(data_dict,
                                            batch_size=self.batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=ema.ema,
                                            single_cls=self.single_cls,
                                            dataloader=val_loader,
                                            save_location=self.save_location,
                                            plots=False,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                stop = stopper(epoch=epoch, fitness=fi)  # early stop check
                if fi > best_fitness:
                    best_fitness = fi
                log_vals = list(mloss) + list(results) + lr
                callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

                # Save model
                if (not self.nosave) or (final_epoch and not self.evolve):  # if save
                    ckpt = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        # 'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        torch.save(ckpt, best)
                    if opt.save_period > 0 and epoch % opt.save_period == 0:
                        torch.save(ckpt, w / f'epoch{epoch}.pt')
                    del ckpt
                    callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # EarlyStopping
            if RANK != -1:  # if DDP training
                broadcast_list = [stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    stop = broadcast_list[0]
            if stop:
                break  # must break all DDP ranks

            # end epoch ----------------------------------------------------------------------------------------------------
        # end training -----------------------------------------------------------------------------------------------------
        if RANK in {-1, 0}:
            LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
                    if f is best:
                        LOGGER.info(f'\nValidating {f}...')
                        results, _, _ = val.run(
                            data_dict,
                            batch_size=self.batch_size // WORLD_SIZE * 2,
                            imgsz=imgsz,
                            model=attempt_load(f, device).half(),
                            iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                            single_cls=self.single_cls,
                            dataloader=val_loader,
                            save_location=self.save_location,
                            save_json=is_coco,
                            verbose=True,
                            plots=plots,
                            callbacks=callbacks,
                            compute_loss=compute_loss)  # val best model with plots
                        if is_coco:
                            callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

            callbacks.run('on_train_end', last, best, plots, epoch, results)

        torch.cuda.empty_cache()
        return results


    def parse_opt(self, known=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
        parser.add_argument('--img_size', type=list, default=[640,480], help='size of your training images')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data_csv_file', type=str, default="C:/Users/olivi/OneDrive/Documents/Perk/Labeling/bboxes/ALL-CSVS (for yolov5 test)/data.yaml", help='dataset.yaml path')
        parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
        parser.add_argument('--noplots', action='store_true', help='save no plot files')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=5, help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
        parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--seed', type=int, default=0, help='Global training seed')
        parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
        parser.add_argument('--save_location', type=str, default = '', help='where to save weights and training results')
        parser.add_argument('--fold', type=int, default = 0, help='fold number to run')
        # Weights & Biases arguments
        parser.add_argument('--entity', default=None, help='W&B: Entity')
        parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
        parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

        opt = parser.parse_known_args()[0] if known else parser.parse_args()
        return opt


    def main(self, opt, callbacks=Callbacks()):
        # Checks
        if RANK in {-1, 0}:
            print_args(vars(opt))
            check_git_status()
            check_requirements(exclude=['thop'])

        # Resume
        if opt.resume and not opt.evolve: #not check_wandb_resume(opt) # resume an interrupted run
            ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
                opt = argparse.Namespace(**yaml.safe_load(f))  # replace
            opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
            LOGGER.info(f'Resuming training from {ckpt}')
        else:
            self.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
                check_file(self.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            if opt.evolve:
                if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                    opt.project = str(ROOT / 'runs/evolve')
                opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
            if opt.name == 'cfg':
                opt.name = Path(opt.cfg).stem  # use model.yaml as name
            opt.save_location = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

        # DDP mode
        device = select_device(opt.device, batch_size=opt.batch_size)
        if LOCAL_RANK != -1:
            msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
            assert not opt.image_weights, f'--image-weights {msg}'
            assert not opt.evolve, f'--evolve {msg}'
            assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
            assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        # Train
        if not opt.evolve:
            self.train(opt.hyp, opt, device, callbacks)
            if WORLD_SIZE > 1 and RANK == 0:
                LOGGER.info('Destroying process group... ')
                dist.destroy_process_group()

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {
                'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

            with open(opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
                if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                    hyp['anchors'] = 3
            opt.noval, opt.nosave, save_location = True, True, Path(opt.save_location)  # only val/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            evolve_yaml, evolve_csv = save_location / 'hyp_evolve.yaml', save_location / 'evolve.csv'
            if opt.bucket:
                os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

            for _ in range(opt.evolve):  # generations to evolve
                if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                    ng = len(meta)
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

                # Constrain to limits
                for k, v in meta.items():
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)  # significant digits

                # Train mutation
                results = self.train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                print_mutation(results, hyp.copy(), save_location, opt.bucket)

            # Plot results
            plot_evolve(evolve_csv)
            LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                        f"Results saved to {colorstr('bold', save_location)}\n"
                        f'Usage example: $ python train.py --hyp {evolve_yaml}')


    def run(self, **kwargs):
        # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        opt = self.parse_opt(True)
        for k, v in kwargs.items():
            setattr(opt, k, v)
        self.main(opt)
        return opt


if __name__ == "__main__":
    train = TrainYolov5()
    train.run()
