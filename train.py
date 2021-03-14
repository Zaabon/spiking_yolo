import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from plot_grad_flow import plot_grad_flow
#from apex import amp
import SpikingNN.spiking_model as SNN

class Train:
    def __init__(self, cfg, input_channels, epochs, batch_size, data_len, load_file=None):
        # Hyperparameters. NOTE: A lot of these are not used
        hyp = {'giou': 3.54,  # giou loss gain
               'cls': 37.4,  # cls loss gain
               'cls_pw': 1.0,  # cls BCELoss positive_weight
               'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
               'obj_pw': 1.0,  # obj BCELoss positive_weight
               'iou_t': 0.20,  # iou training threshold
               'lr0': 1e-5,  # initial learning rate (SGD=5E-3, Adam=5E-4)
               'lrf': 1e-8,  # doesn't do anything
               'momentum': 0.937,  # SGD momentum
               'weight_decay': 0.0005,  # optimizer weight decay
               'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
               'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
               'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
               'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
               'degrees': 1.98 * 0,  # image rotation (+/- deg)
               'translate': 0.05 * 0,  # image translation (+/- fraction)
               'scale': 0.05 * 0,  # image scale (+/- gain)
               'shear': 0.641 * 0}  # image shear (+/- deg)
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=1)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
        parser.add_argument('--batch-size', type=int, default=1)  # effective bs = batch_size * accumulate = 16 * 4 = 64
        parser.add_argument('--data', type=str, default='../data/coco2017.data', help='*.data path')
        parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
        parser.add_argument('--img-size', nargs='+', type=int, default=[240, 304], help='[min_train, max-train, test]')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--notest', action='store_true', help='only test final epoch')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
        parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
        parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--adam', action='store_true', help='use adam optimizer')
        parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
        parser.add_argument('--freeze-layers', action='store_true', help='Freeze non-output layers')
        self.opt, unknown = parser.parse_known_args()
        self.opt.cfg = cfg
        self.opt.weights = load_file

        print("Using config", self.opt.cfg)

        #self.opt = parser.parse_args()
        self.opt.weights = self.last if self.opt.resume and not self.opt.weights else self.opt.weights
        self.opt.cfg = check_file(self.opt.cfg)  # check file
        self.opt.data = check_file(self.opt.data)  # check file
        self.opt.cache_images = False

        self.opt.img_size.extend(
            [self.opt.img_size[-1]] * (3 - len(self.opt.img_size)))  # extend to 3 sizes (min, max, test)
        self.mixed_precision = False #True
        self.device = torch_utils.select_device(self.opt.device, apex=self.mixed_precision,
                                                batch_size=self.opt.batch_size)
        if self.device.type == 'cpu':
            self.mixed_precision = False

        self.tb_writer = None
        if not self.opt.evolve:  # Train normally
            print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            self.tb_writer = SummaryWriter(comment=self.opt.name)
            # self.train(hyp)  # train normally

        else:  # Evolve hyperparameters (optional)
            self.opt.notest, self.opt.nosave = True, True  # only test/save final epoch
            if self.opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % self.opt.bucket)  # download evolve.txt if exists

            for _ in range(1):  # generations to evolve
                if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                    ng = len(g)
                    if method == 1:
                        v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                    elif method == 2:
                        v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                    elif method == 3:
                        v = np.ones(ng)
                        while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                            # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                            v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = x[i + 7] * v[i]  # mutate

                # Clip to limits
                keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
                limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9),
                          (0, 3)]
                for k, v in zip(keys, limits):
                    hyp[k] = np.clip(hyp[k], v[0], v[1])

                # Train mutation
                results = self.init_train(hyp.copy())
                # Write mutation results
                # print_mutation(hyp, results, self.opt.bucket)

                # Plot results
                # plot_evolution_results(hyp)

        #self.mixed_precision = True
        #try:  # Mixed precision training https://github.com/NVIDIA/apex
        #except:
        #    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
        #    self.mixed_precision = False  # not installed

        self.wdir = 'weights' + os.sep  # weights dir
        self.last = self.wdir + 'last.pt'
        self.best = self.wdir + 'best.pt'
        self.results_file = 'results.txt'

        # Overwrite hyp with hyp*.txt (optional)
        f = glob.glob('hyp*.txt')
        if f:
            print('Using %s' % f[0])
            for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
                hyp[k] = v

        # Print focal loss if gamma > 0
        if hyp['fl_gamma']:
            print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

        # scale hyp['obj'] by img_size (evolved at 320)
        # hyp['obj'] *= opt.img_size[0] / 320.

        self.batch_size = batch_size
        cfg = self.opt.cfg
        weights = self.opt.weights  # initial training weights
        imgsz_min, imgsz_max, imgsz_test = self.opt.img_size  # img sizes (min, max, test)

        gs = 16 #32  # (pixels) grid size
        assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
        self.img_size = imgsz_max  # initialize with max size

        init_seeds()
        nc = 2 # number of classes
        hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

        # Remove previous results
        for f in glob.glob('*_batch*.jpg') + glob.glob(self.results_file):
            os.remove(f)

        self.model = Darknet(cfg, input_channels).to(self.device)

        # Optimizer
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else
        if self.opt.adam:
            # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
            self.optimizer = optim.Adam(pg0, lr=hyp['lr0'])
            print("Using ADAM")
            # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
        else:
            print("Using SGD")
            self.optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Pre-trained
        start_epoch = 0
        best_fitness = 0.0
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            if load_file is not None:
                # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
                ckpt = torch.load(load_file, map_location=self.device) #torch.load(weights, map_location=self.device)

                # load model
                try:
                    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
                    self.model.load_state_dict(ckpt['model'], strict=False)
                except KeyError as e:
                    s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                        "See https://github.com/ultralytics/yolov3/issues/657" % (
                        self.opt.weights, self.opt.cfg, self.opt.weights)
                    raise KeyError(s) #from e
                print("Loaded weights from", weights)

                # Track epochs, probably for learning rate and such

                # load optimizer
                if ckpt['optimizer'] is not None:
                    print("Loading optimizer from checkpoint")
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                    #best_fitness = ckpt['best_fitness']

                # load results
                if ckpt.get('training_results') is not None:
                    with open(self.results_file, 'w') as file:
                        file.write(ckpt['training_results'])  # write results.txt

                # epochs
                """
                start_epoch = ckpt['epoch'] + 1
                if epochs < start_epoch:
                    print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                          (self.opt.weights, ckpt['epoch'], epochs))
                    epochs += ckpt['epoch']  # finetune additional epochs
                """
                del ckpt
        elif len(weights) > 0:  # darknet format
            # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            load_darknet_weights(self.model, weights)

        
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        self.lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.scheduler.last_epoch = start_epoch - 1  # see link below
        
        # Mixed precision training https://github.com/NVIDIA/apex
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

        # Initialize distributed training
        if self.device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            dist.init_process_group(backend='nccl',  # 'distributed backend'
                                    init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                    world_size=1,  # number of nodes for distributed training
                                    rank=0)  # distributed training node rank
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
            self.model.yolo_layers = self.model.module.yolo_layers  # move yolo layer indices to top level
            
        self.model.nc = nc  # attach number of classes to model <- doesn't seem to have any effect
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        
        # dataset_labels is a 2d tensor with length 6 labels
        #self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(self.device)  # attach class weights
        self.ema = torch_utils.ModelEMA(self.model)

        
        self.nb = data_len  # number of batches
        self.n_burn = max(3 * self.nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
        self.maps = np.zeros(nc)  # mAP per class
        # torch.autograd.set_detect_anomaly(True)
        #self.results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
        self.mloss = torch.zeros(4).to(self.device)  # mean losses
        self.epoch = 0

        self.names = ["Airplane", "Motorbike"]  # Class names


    def predict(self, inputs, targets, step, i, snn_grad=None): # ni = number integrated batches (since train start)
        self.model.train() # Set training mode
        
        ni = i + self.nb * self.epoch  # number integrated batches (since train start)

        """
        # Burn-in
        if step and ni <= self.n_burn:
                xi = [0, self.n_burn]  # x interp
                self.model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                    x['weight_decay'] = np.interp(ni, xi, [0.0, self.model.hyp['weight_decay'] if j == 1 else 0.0])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, self.model.hyp['momentum']])
        """
        # - Forward -
        pred = self.model(inputs)
        
        # Loss
        loss, loss_items = compute_loss(pred, targets, self.model)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            return #results

        # - Backward -
        #loss *= self.batch_size / 64  # scale loss
        if self.mixed_precision:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            if snn_grad is not None:
                plot_grad_flow(snn_grad)


        # Optimize
        if step: #ni % accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.ema.update(self.model)

        # Print
        #mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        #mem = '%.3gGB' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        #s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
        #pbar.set_description(s)

        """
        # Plot
        if ni < 1:
            f = 'train_batch%g.jpg' % i  # filename
            res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
            if self.tb_writer:
                self.tb_writer.add_image(f, res, dataformats='HWC', global_step=epoch)
                # tb_writer.add_graph(model, imgs)  # add model to tensorboard
        """

        return loss
    
    def end_epoch(self):
        #self.mloss = torch.zeros(4).to(device)  # mean losses
        self.epoch += 1
        self.scheduler.step()
        self.ema.update_attr(self.model)
        
    def test_init(self):
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        self.iouv = self.iouv[0].view(1)  # comment for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        
        self.seen = 0
        self.model.eval()

        self.s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
        self.p, self.r, self.f1, self.mp, self.mr, self.map, self.mf1, self.t0, self.t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        self.loss = torch.zeros(3, device=self.device)
        self.jdict, self.stats, self.ap, self.ap_class = [], [], [], []
        self.confusion_matrix = torch.zeros((3,3), device=self.device) 
        
    def test(self, imgs, targets, img_id, im0, whwh, conf_thres=0.001, iou_thres=0.5):
        targets = targets.to(self.device)
        nb, _, height, width = im0.shape  # batch size, channels, height, width

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = self.model(imgs)  # inference and training outputs
            self.t0 += torch_utils.time_synchronized() - t
            
            # Compute loss
            if True:  # if model has loss hyperparameters
                self.loss += compute_loss(train_out, targets, self.model)[1][:3]  # GIoU, obj, cls
            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres, iou_thres, multi_label=False, classes=None, agnostic=False)
            self.t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            guessed = False
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            self.seen += 1

            if pred is None:
                if nl:
                    self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            #clip_coords(pred, (176, 240))

            # Append to pycocotools JSON dictionary
            if False:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = img_id
                box = pred[:, :4].clone()  # xyxy

                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    self.jdict.append({'image_id': image_id,
                                  #'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=self.device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices                       

                        guessed = True
                        # Append detections
                        for j in (ious > self.iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            
        # Plot images
        if False:
            im0 = torch.cat((im0, torch.zeros((1,1,im0.shape[2],im0.shape[3]), device=self.device)), axis=1) # Expand to 3 channels
            f = 'test_batch%g_gt.jpg' % img_id  # filename
            plot_images(im0, targets, names=self.names, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % img_id
            plot_images(im0, output_to_target(output, im0.shape[2], im0.shape[3]), names=self.names, fname=f)  # predictions
        
    def test_end(self):
        # Compute statistics
        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(self.stats):
            self.p, self.r, self.ap, self.f1, self.ap_class = ap_per_class(*self.stats)
            if self.niou > 1:
                self.p, self.r, self.ap, self.f1 = self.p[:, 0], sefl.r[:, 0], self.ap.mean(1), self.ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            self.mp, self.mr, self.map, self.mf1 = self.p.mean(), self.r.mean(), self.ap.mean(), self.f1.mean()
            nt = np.bincount(self.stats[3].astype(np.int64), minlength=self.model.nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        print("AP:", self.ap)
        return self.map, self.loss
        
    def save(self, path):
        #with open(self.results_file, 'r') as f:  # create checkpoint
        ckpt = {#'epoch': epoch,
                #'best_fitness': best_fitness,
                #'training_results': f.read(),
                'model': self.ema.ema.module.state_dict() if hasattr(self.model, 'module') else self.ema.ema.state_dict(),
                #'optimizer': None if final_epoch else optimizer.state_dict()}
                'optimizer': self.optimizer.state_dict()}

        torch.save(ckpt, path)
        print("Saved YOLO", path)
        del ckpt
