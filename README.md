### Object Detection and Localization from overhead cameras in fixed area

This micropackage includes:
- `object_detector.py`: A wraper of opencv `dnn` module for different sources of pre-trained models: caffe, tensorflow or darknet ... You can use the same framework, which contains only 2 methods: `forward()` and `post_process()`, for all models from different architectures.
- `localizer.py`: A simple localization algorithm for overhead camera in fixed area


You can:
- load the interested model in and use it directly to infer  from an input. Interestingly, you can enable `CUDA` GPU just by setting a flag. Pretty neat, right?
- Use the localizer to identify true location of object in the selected area.

Repo url: https://github.com/DoanNguyenTrong/opencv-object-detector


*This is a part of an IoT network that I am developing at ICONS Lab. It's a distributed system of multiple cameras and CO2 sensors for occupancy density detection and air/ventilation quality assessment. Our goal is to develop methods and metrics to better assess and maintain healthy indoor environments during a pandemic.*

### Methods

Object detection:
- Creat object: `HD = ObjectDetector(weight_file, config_file, classes_name_file, GPU=True/False)`
- Inference: `out = HD.forward(img)` to obtain ouput of the network after feeding your `img` through
- Post process: `clss_IDs, scores, bbox = HD.post_process(outs, width, height)` to get class IDs of the objects, its confident scores as well as bounding boxes
- Draw:
  - `HD.draw_all(img, IDs, scores, bbox)` to draw all objects in the frame
  - `HD.draw_(img, IDs, scores, bbox, human=0)` to draw human in the frame. You need to specify the id of human that is used by the network (e.g., `0` in the case of YOLO)

Object localization:
- Create object: `LZ = Localizer([x0, y0], [xt, yt])`. Here, [x0 , y0] are x and y lists of 4 original points and [xt, yt] are x and y of 4 corresponding points that will be used for the perspective transformation algorithm `cv2.getPerspectiveTransform()`. This step allows us to get a transformation matrix that can convert from camera's  to top-down view point. Then we can approximate the true location of people in the area.\
  ```py
    # Example
    # Original
    xo = [334, 165, 931, 1117]
    yo = [180, 591, 171, 571]
    # Transformed
    xt = [0, 0, 600, 600]
    yt = [0, 600, 0, 600]
  ```
- Extract raw locations: `raw_locs = LZ.raw_location(bbox, offset=[0,0])`. Here we assume that the location of one object is at the middle of bottom edge of a bounding box. You can tune it by specify `offset` for both x and y axes.
- True locations: `true_locs = LZ.actual_location(IDs, scores, raw_locs, clss= None)`. You need to provide list of IDs and scores so that the method can extract these information corresponding to each location. If you want to filters only interested classed, change `clss` to a list of IDs (e.g., `clss=[0, 14]`).
- Draw: `LZ.draw_locs(img, true_locs, radius, thickness, color)` to draw a circle at `true_locs`

### Some commands to use `example.py`
```bash
python3 example.py

python3 example.py --type caffe --loc /model_zoo/caffe/ --m MobileNetSSD_deploy --video videos/demo2.mp4

python3 example.py --type tf --loc /model_zoo/tensorflow/ --m mask_rcnn_inception_v2_coco_2018_01_28 --video videos/demo2.mp4
```

### Benchmark
I have tested the inference on some scenarios and here is a quick summary. When you enable `GPU` inference, the speed is increased roughly `8X`. Also, I have some tests using TensorRT, and it significant improvement is very interesting.
You can see that I did test `gluoncv` - a handy package that comes with some pre-trained models, but it performance is very poor.

| Algorithm                       | System          | Performance                                      | Accuracy                                                                                                                      |
|---------------------------------|-----------------|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Tiny Yolov3(opencv)             | MacOS (core i7) | ~30 FPS (cpu)                                    | Bad: miss/false detection on both videos                                                                                      |
| Yolov4 (openCV)                 | MacOS (core i7) | ~ 2FPS (cpu)                                     | Good: can detect almost all people in the testing videos.Very little miss/false detection and works well with full HD videos. |
| Yolov3(gluoncv)                 | MacOS (core i7) |  0.4 - 0.5 FPS (cpu)                             | Bad:  miss/false detection on the overhead view video. Good accuracy on the human point of view video                         |
| SSD(gluoncv)                    | MacOS (core i7) | 0.8-0.9 FPS (cpu)                                | Bad:  miss/false detection on the overhead view video. Not so good on the human point of view video                           |
| Faster RCNN(gluoncv)            | MacOS (core i7) | ~0.1 FPS (cpu)                                   | Bad:  miss/false detection on the overhead view video, Good on the human point of view video                                  |
| Center net (gluoncv)            | MacOS (core i7) | 0.3 - 0.4 FPS (cpu)                              | Bad:  miss/false detection on the overhead view video, Good on the human point of view video                                  |
| mask_rcnn_inception_v2 (openCV) | MacOS (core i7) | ~ 1.2 FPS - video 1, 0.7-0.8 FPS (cpu) - video 2 | Quite good:  Some miss/false detection on the overhead view video, good on the human point of view video                      |
| ssd_mobilenet_v1 (openCV)       | MacOS (core i7) | ~3 FPS - video 1, 1.2 FPS (cpu) - video 2        | Bad:  miss/false detection on both videos                                                                                     |
| MobileNetSSD_deploy (openCV)    | MacOS (core i7) | ~ 20 FPS (cpu)                                   | Very bad                                                                                                                      |
| Yolov4 (openCV)                 | Jetson TX2      | ~ 0.25 FPS (cpu), ~ 2 FPS(GPU)                   | Same accuracy as running on my Macbook                                                                                        |
| Yolov4 (TensorRT)               | Jetson TX2      | 6-8 FPS (GPU)                                    | Same accuracy as running on my Macbook                                                                                        |
