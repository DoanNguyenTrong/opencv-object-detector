import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import imutils



class ObjectDetector:
    """
    ObjectDetector performs Object detection
    """
    
    def __init__(self, w_fname, cfg_fname, clss_fname, GPU=False):
        self.w_fname    = w_fname
        self.cfg_fname  = cfg_fname
        self.clss_fname = clss_fname
        
        # Read model and classes in
        self.net_     = self.read_model(GPU=GPU)
        self.classes_ = self.read_classes()
        
        # Color for plots
        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        
    def read_classes(self):
        """
        Read in classes
        """
        classes = None
        if self.option_ == 'cf':
            # Read in classes
            classes = { 0: 'background',
                         1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                         5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                         10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                         14: 'motorbike', 15: 'person', 16: 'pottedplant',
                         17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
        elif self.option_ == 'dn':
            with open(self.clss_fname, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        elif self.option_ == 'tf':
            classes = { 0: 'background',
                         1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                         5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                         10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                         14: 'motorbike', 15: 'person', 16: 'pottedplant',
                         17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
        else:
            raise Exception("Could not read you classes file!!!")
        return classes
    
    def read_model(self, GPU = False):
        """
        Read pre-trained model into cv2.dnn object
        """
        net = None
        if 'darknet' in self.w_fname:
            net = cv2.dnn.readNet(self.w_fname, self.cfg_fname)
            self.option_ = 'dn'
        elif 'tensorflow' in self.w_fname:
            net = cv2.dnn.readNetFromTensorflow(self.w_fname, self.cfg_fname)
            self.option_ = 'tf'
        elif 'caffe' in self.w_fname:
            net = cv2.dnn.readNetFromCaffe(self.cfg_fname, self.w_fname)
            self.option_ = 'cf'
        else:
            raise Exception("The model is not implemented!")
        
        
        print('Read model')
        print(self.w_fname)
        print(self.cfg_fname)
        print(self.clss_fname)
        # GPU acceleration
        if GPU:
            print("GPU BACKEND")
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except:
                raise Exception("You need to install opencv CUDA and CUDNN to make it works!!")
            
        else:
            print("CPU BACKEND")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net

    def forward(self, img):
        outs = None
        #  load images in a batch and run them through the network
        if self.option_ == "dn":
            self.net_.setInput(cv2.dnn.blobFromImage(img, 1./255.0, (416,416), (0,0,0), True, crop=False))
            layer_names = self.net_.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in self.net_.getUnconnectedOutLayers()]
            outs = self.net_.forward(output_layers)
            return outs
        
        elif self.option_ == "tf":
            self.net_.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))
            outs = self.net_.forward()
        
        elif self.option_ == 'cf':
            h, w, _ = img.shape
            img_ = cv2.resize(img, (int(h * (400./w)),400))
            # img_ = imutils.resize(img, width=400)
            self.net_.setInput(cv2.dnn.blobFromImage(img_, 0.007843, (300, 300), 127.5))
            outs = self.net_.forward()
        else:
            raise Exception("Your option is not implemented!", self.option_)
        
        
        assert not outs.all() == None, "Error!! forward None!"
        
        return outs

    def post_process(self, outs, Width, Height,  score_thres = 0.1, nms_thres=0.5):
        class_ids_ = []
        confidences_ = []
        boxes_ = []
        
        if self.option_ == 'dn':
            class_ids = []
            confidences = []
            boxes = []
            centers = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > score_thres:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
            
            # Filter proposed regions using Non-maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thres, nms_thres)
            # Output
            class_ids_ = [class_ids[i[0]] for i in indices]
            confidences_ = [confidences[i[0]] for i in indices]
            boxes_ = [boxes[i[0]] for i in indices]

        elif self.option_ == 'cf' or self.option_ == 'tf':
            for i in range(0, outs.shape[2]):
                class_id = int(outs[0, 0, i, 1])
                confidence = outs[0, 0, i, 2]
                if  confidence > score_thres:
                    box = outs[0, 0, i, 3:7]* np.array([Width, Height, Width, Height])
                    startX, startY, endX, endY = box.astype("int")

                    startX = max(0, min(startX, Width - 1))
                    startY = max(0, min(startY, Height - 1))
                    endX = max(0, min(endX, Width - 1))
                    endY = max(0, min(endY, Height - 1))

                    w = endX - startX
                    h = endY - startY
                    class_ids_.append(class_id)
                    confidences_.append(float(confidence))
                    boxes_.append([startX, startY, w, h])
                
        else:
            print("NAH!", self.option_)
        
        # Fail check before return
        assert not (class_ids_ ==None or confidences_ == None or boxes_ == None), "Failed compution, all None!!!"
        
        return class_ids_, confidences_, boxes_
    
    def draw_all(self, img, class_ids, confidences, boxes):
        """
            Draw all boxes, put classes and label of detected object into the image
        """
        for i in range(len(class_ids)):
            box = boxes[i]
            id_ = class_ids[i]
            label = str(self.classes_[id_])
            confi = str(round(confidences[i], 4))
            color = self.COLORS[int(id_) % len(self.COLORS)]

            cv2.rectangle(img, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), color, 2)
            cv2.putText(img, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(img, confi, (round(box[0])-10,round(box[1])-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def draw_(self, img, class_ids, confidences, boxes, human=0):
        
        for i in range(len(class_ids)):
            if class_ids[i] == human:
                box = boxes[i]
                id_ = class_ids[i]
                label = str(self.classes_[id_])
                confi = str(round(confidences[i], 4))
                color = self.COLORS[int(id_) % len(self.COLORS)]

                cv2.rectangle(img, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), color, 2)
                cv2.putText(img, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(img, confi, (round(box[0])-10,round(box[1])-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img