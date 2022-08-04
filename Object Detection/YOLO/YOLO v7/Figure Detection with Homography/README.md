# Figure Detection with Homography

Using Homography and YOLOv7 to detect Figure
* I'm still developing this, This is not a complete Project

![alt text](https://github.com/Junst/Computer-Vision/blob/main/Object%20Detection/YOLO/YOLO%20v7/Figure%20Detection%20with%20Homography/Results/outputresults.png)

## Usage

You have to install the -requirements of YOLOv7.<br>
This Link is the best to guide : https://www.youtube.com/watch?v=a9RJV5gI2VA<br>

If you get all of packages or requirements of YOLOv7, You need to get your train pt file. Roboflow sites may be the one way to get pt file. Find some annotation sites and labeling your train data. All done, move your pt file into folder(directory)

Run detect.py

You can click four times to make rectangle and if you push enter key of keyboard you will get the Homography of source image(frame).

Run catch.py

IF you install pydobot packages, then make sure your port available by checking the "port = available_ports[1].device". If you have a many USB, may be you have to change the numeber of available_ports (defaults : 0 to 1, 2, 3... your USB linked). Now you can see the frame when we got in detect.py and click the points and the dobot will move. (But I sayed, I'm still developing and caculating of this process, so you have to check it)

## Pydobot

DOBOT Magician sites : https://en.dobot.cn/products/education/magician.html?utm_source=google&utm_medium=cpc&utm_campaign=ap+all+bd&gclid=Cj0KCQjw0JiXBhCFARIsAOSAKqB4AHP7JYMiunjlsdmEBr_BEaturQZaSoyK45C-0UACiSxrdpe--WMaAvXFEALw_wcB



## Official YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)


### Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

Root
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

Personal
* [https://github.com/FW2022/Realsense_SSD_Model](https://github.com/FW2022/Realsense_SSD_Model)
* [https://github.com/dbloisi/homography-computation](https://github.com/dbloisi/homography-computation)

</details>
