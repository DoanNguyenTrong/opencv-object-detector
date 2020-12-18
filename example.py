import os
import cv2
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from object_detector.object_detector import *
from object_detector.localizer import *



def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--loc", help="Location of data files", type=str, default='/model_zoo/darknet/')
    parser.add_argument( "--m", help="Model name", type=str, default='yolov4')
    parser.add_argument( "--type", help="Model type", type=str, default='yolo')
    parser.add_argument( "--video", help="video file", type=str, default='videos/demo.mp4')
    parser.add_argument( "--gpu", help="GPU enable?", type=bool, default=False)
    parser.add_argument("--out", help="ouput video export folder", type=str, default='videos/')

    return parser.parse_args()

def main(args):
    
    if args.type == 'yolo':
        # file existance?
        if not os.path.isfile(os.getcwd() + args.loc + args.m +'.weights'):
            raise SystemExit('ERROR: file (%s) not found!' % (args.loc + args.m +'.weights'))
        if not os.path.isfile(os.getcwd() + args.loc + args.m +'.cfg'):
            raise SystemExit('ERROR: file (%s) not found!' % (args.loc + args.m +'.cfg'))
        if not os.path.isfile( os.getcwd() + args.loc +'coco.names' ):
            raise SystemExit('ERROR: file (%s) not found!' % (args.loc +'coco.names'))

        # Load Object Detection Model
        HD = ObjectDetector(os.getcwd() +args.loc + args.m +'.weights', 
                            os.getcwd() +args.loc + args.m +'.cfg', 
                            os.getcwd() +args.loc +'coco.names', GPU=args.gpu)
    elif args.type == 'caffe':
        # file existance?
        if not os.path.isfile(os.getcwd() + args.loc + args.m +'.caffemodel'):
            raise SystemExit('ERROR: file (%s) not found!' % (args.loc + args.m +'.caffemodel'))
        if not os.path.isfile(os.getcwd() + args.loc + args.m +'.prototxt'):
            raise SystemExit('ERROR: file (%s) not found!' % (args.loc + args.m +'.prototxt'))
        
        # Load Object Detection Model
        HD = ObjectDetector(os.getcwd() + args.loc + args.m +'.caffemodel', 
                            os.getcwd() + args.loc + args.m +'.prototxt', 
                            None, GPU=args.gpu)
    elif args.type == 'tf':
        # file existance?
        if not os.path.isfile(os.getcwd() + args.loc + args.m +'/frozen_inference_graph.pb'):
            raise SystemExit('ERROR: file (%s) not found!' % (args.loc + args.m +'/frozen_inference_graph.pb'))
        if not os.path.isfile(os.getcwd() + args.loc + args.m +'.pbtxt'):
            raise SystemExit('ERROR: file (%s) not found!' % (args.loc + args.m +'.pbtxt'))
        
        # Load Object Detection Model
        HD = ObjectDetector(os.getcwd() + args.loc + args.m +'/frozen_inference_graph.pb', 
                            os.getcwd() + args.loc + args.m +'.pbtxt', 
                            None, GPU=args.gpu)

    else:
        raise Exception("Not implemented (%s) yet!"% args.type)


    ### Localizer
    # Original
    xo = [334, 165, 931, 1117]
    yo = [180, 591, 171, 571]
    # Transformed
    xt = [0, 0, 600, 600]
    yt = [0, 600, 0, 600]
    LZ = Localizer([xo, yo], [xt, yt])

    vc = cv2.VideoCapture(args.video)

    # Define the codec and create VideoWriter object
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    size = (width, height)

    # NOTE: if the VideoWriter receive improper frame size, 
    # it will not raise any exception other than write a NULL video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.out + args.m +'_output' + '.avi', fourcc, 20.0, size)

    # Check if video/camera opened successfully
    if (vc.isOpened()== False):
        print("Error opening video stream or file")
    else:
        while (vc.isOpened()):
            (grabbed, frame) = vc.read()
            if not grabbed:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            # Object detection
            start = time.time()
            outs_ = HD.forward(frame)
            class_ids_, confidences_, boxes_ = HD.post_process(outs_, width, height)
            
            # Localization
            raw_locs = LZ.raw_location(boxes_)
            class_ids_t, confidences_t, abs_locs = LZ.actual_location(class_ids_, confidences_, raw_locs)
            end = time.time()
            
            
            start_drawing = time.time()

            frame_ = HD.draw_all(frame.copy(), class_ids_, confidences_, boxes_)
            frame_ = LZ.draw_locs(frame_, raw_locs)
            frame_t = cv2.warpPerspective(frame_, LZ.M, (np.max(LZ.xt), np.max(LZ.yt))) 
            frame_t = LZ.draw_locs(frame_t, abs_locs)
            end_drawing = time.time()
            
            # FPS
            fps_label = "FPS: %.2f" % (1 / (end - start))
            cv2.putText(frame_, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            out.write(frame_)

            cv2.imshow("detections", frame_)
            cv2.imshow("localization", frame_t)
            if cv2.waitKey(1) == ord('q'):
                break

    # Release everything if job is finished
    vc.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    args = args_parse()

    main(args)