# Copyright (C) 2017 DataArt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# based on https://github.com/devicehive/devicehive-video-analysis
# modified by: 2018 pinguinonice

# import the necessary packages
from __future__ import print_function
import numpy as np
import math
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import time
import logging.config
import tensorflow as tf
from models import yolo
from log_config import LOGGING
import matplotlib.pyplot as plt
from Homography_exercise import get_homography_matrix
from utils import mouse_handler


# 멀티쓰레드용 import
from threading import Thread, Lock
from queue import Queue
import math
import copy

class CameraDetection(object):
    def __init__(self):
        self.input_frame = None
        self.input_image = None
        self.display_image = None
        self.source_height = None
        self.source_width = None
        self.source_channels = None
        self._is_det_res_ready = False
        self.det_result = None
        self.lock = Lock()

        self.input_queue = Queue()
        self.display_queue = Queue()

        self.input_thread = Thread(target=self.input_process)
        self.input_thread.start()

        self.detection_thread = Thread(target=self.detection_process)
        self.detection_thread.start()

        self.display_thread = Thread(target=self.display_process)
        self.display_thread.start()

    def detection_process(self):


        logging.config.dictConfig(LOGGING)
        logger = logging.getLogger('detector')
        FLAGS = tf.flags.FLAGS
        del_all_flags(tf.flags.FLAGS)
        tf.flags.DEFINE_string('video', "0", 'Path to the video file.')
        tf.flags.DEFINE_string('model_name', 'Yolo2Model', 'Model name to use.')


        while True:
            self.lock.acquire()
            source_h = copy.deepcopy(self.source_height)
            source_w = copy.deepcopy(self.source_width)
            channels = copy.deepcopy(self.source_channels)
            frame_rgb = copy.deepcopy(self.display_image)
            self.lock.release()

            if frame_rgb is None:
                time.sleep(0.003)
                continue

            # Start time
            # start = time.time()

            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            out = cv2.VideoWriter('../output/outputYolo.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                  (source_w, source_h))
            model_cls = find_class_by_name(FLAGS.model_name, [yolo])
            model = model_cls(input_shape=(source_h, source_w, channels))
            model.init()

            # detect people in the image
            preds = model.evaluate(frame_rgb)
            print(preds)

            if self._is_det_res_ready is False:
                self.det_result = preds
                self._is_det_res_ready = True

    def input_process(self):
        cam = cv2.VideoCapture(4)  # (2) 0 -> index of camera

        while True :
            # load frame
            s, frame_rgb = cam.read()
            # initializ
            source_h, source_w, channels = frame_rgb.shape

            self.lock.acquire()
            try:
                self.input_frame = frame_rgb
                self.source_height = source_h
                self.source_width = source_w
                self.source_channels = channels
                #self.input_image = image_expanded
                self.display_image = frame_rgb
            except Exception as e :
                print("카메라 인풋 에러")
            self.lock.release()

            time.sleep(0.005)

    def display_process(self):
        begin = time.time()
        cogs = []
        start = time.time()
        while True:
            self.lock.acquire()
            frame_rgb = self.display_image
            source_h = self.source_height
            self.lock.release()
            if frame_rgb is None:
                time.sleep(0.003)
                continue

            if self._is_det_res_ready is True:
                preds = self.det_result
                self._is_det_res_ready = False

                for o in preds:
                    x1 = o['box']['left']
                    x1 = o['box']['left']
                    x2 = o['box']['right']

                    y1 = o['box']['top']
                    y2 = o['box']['bottom']

                    color = o['color']
                    class_name = o['class_name']

                    if class_name == 'person':
                        cogs.append([math.floor((o['box']['left'] + o['box']['right']) / 2),
                                     math.floor(o['box']['bottom']),  # red point : fits quite good
                                     1])
                        # print(cogs)
                        cv2.circle(frame_rgb, (cogs[-1][0], cogs[-1][1]), 5, (0, 0, 255), -1)

                    # Draw box
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    (test_width, text_height), baseline = cv2.getTextSize(
                        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
                    cv2.rectangle(frame_rgb, (x1, y1),
                                  (x1 + test_width, y1 - text_height - baseline),
                                  color, thickness=cv2.FILLED)
                    cv2.putText(frame_rgb, class_name, (x1, y1 - baseline),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

            # Homgraphy matrix (has to be changed if camera changes)
            # https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html
            # SaltCity2:
            rot = np.array([[0.00840278273967597, - 0.000996762883651406, - 0.972939274817017],
                            [-0.000611229109482202, - 0.000363960872335956, - 0.230899966154323],
                            [4.75921782278034e-06, 6.0886306800187e-06, - 0.00150556507364792]])
            # TownCenter:
            rot = np.array([[0.0101240234768616, -0.00847479565930775, -0.974192433448834],
                            [-0.00196859030258317, -0.000929201148491731, -0.225308693846282],
                            [9.56043341206427e-06, 4.71595084773723e-06, -0.0024572747705754]])

            # controls position in topview
            view = np.array([[-1, 0, 100],
                             [0, 1, 0],
                             [0, 0, 1]])

            # define gridpoints in top coordinates
            data = {}
            data['frame_rgb'] = frame_rgb #.copy()
            data['points'] = []
            cv2.imshow("Detection", frame_rgb)
            cv2.setMouseCallback("Detection", mouse_handler, data)
            cv2.waitKey(0)


            data['points'][0].append(1)
            data['points'][1].append(1)
            data['points'][2].append(1)
            data['points'][3].append(1)

            points_org = np.vstack(data['points']).astype(float)

            # points_org = np.array([[340, 373, 1],  # raw points
            #                        [272,393, 1],
            #                        [242, 370, 1],
            #                        [307, 357, 1],
            #                        [340, 373, 1],
            #                        ])
            #if (np.round(start - begin, 2) > 6) and (np.round(start - begin, 2) < 7):
            if len(data) == 4:
                cv2.imwrite("frame.jpg", frame_rgb)
                source_image = cv2.imread("frame.jpg")
                t_source_image = source_image.copy()

                source_points = np.array([[340, 373],  # raw points
                                          [272, 393],
                                          [242, 370],
                                          [307, 357],
                                          ])
                destination_points = np.array([
                    [0, 0],
                    [300, 0],
                    [300, 300],
                    [0, 300],
                ])

                h = get_homography_matrix(source_points, destination_points)
                destination_image = cv2.warpPerspective(t_source_image, h, (300, 300))

                figure = plt.figure(figsize=(12, 6))

                subplot1 = figure.add_subplot(1, 2, 1)
                subplot1.title.set_text("Source Image")
                subplot1.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))

                subplot2 = figure.add_subplot(1, 2, 2)
                subplot2.title.set_text("Destination Image")
                subplot2.imshow(cv2.cvtColor(destination_image, cv2.COLOR_BGR2RGB))

                # plt.show()
                plt.savefig("output.png")

                # Transpose points to image and draw them
                print(rot)
                print(points_org)
                points = np.transpose(np.matmul(rot, np.transpose(points_org)))
                points = np.array([[x / w, y / w] for (x, y, w) in points])  # devide by w (homogene coordinates)
                cv2.polylines(frame_rgb, np.int32([points]), False, (0, 255, 255))

                # Transform  Detection points to topview:
                invrot = np.matmul(view, np.linalg.inv(rot))

                if not cogs:
                    people_cords = np.array([[0, 0, 1]])
                else:
                    people_cords = np.array(cogs)
                # print(people_cords)
                toppeople_cords = np.transpose(np.matmul(invrot, np.transpose(people_cords)))
                toppeople_cords = np.array([[x / w, y / w] for (x, y, w) in toppeople_cords])
                # print(toppeople_cords)

                #########

                # count points in polygon
                A = 0  # set to zero for every frame
                B = 0
                C = 0
                D = 0
                areaA = Polygon([(0, 0), (50, 0), (50, 60), (0, 60)])  # define poligons of area
                areaB = Polygon([(50, 0), (100, 0), (100, 60), (50, 60)])
                areaC = Polygon([(0, 60), (50, 60), (50, 100), (0, 100)])
                areaD = Polygon([(50, 60), (100, 60), (100, 100), (50, 100)])
                for toppeople_cord in toppeople_cords:
                    cv2.circle(frame_rgb, (np.int32(toppeople_cord[0]),
                                           np.int32(toppeople_cord[1])), 2, (0, 0, 255), -1)
                    if areaA.contains(Point(toppeople_cord)):
                        A += 1
                    if areaB.contains(Point(toppeople_cord)):
                        B += 1
                    if areaC.contains(Point(toppeople_cord)):
                        C += 1
                    if areaD.contains(Point(toppeople_cord)):
                        D += 1
                print(("{}  {}  {}  {}  {}".format(A, B, C, D, np.round(start - begin, 2))))
                # plot topvie

                cv2.polylines(frame_rgb, np.int32([points_org[:, :2]]), False, (0, 255, 255))
                cv2.putText(frame_rgb, str(A), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_rgb, str(B), (65, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_rgb, str(C), (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_rgb, str(D), (65, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # End time
                end = time.time()

                # Time elapsed
                seconds = end - start
                cv2.putText(frame_rgb, "FPS:{}".format(np.round(1 / seconds, 1)), (20, source_h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # show some information on the number of bounding boxes
                cv2.putText(frame_rgb, ("Number of People:{}".format(len(cogs))),
                            (20, source_h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2
                            )
                # plt.scatter(topcogs[:, 0], topcogs[:, 1])
                # plt.axis('equal')
                # plt.plot(points_org[:, 0], points_org[:, 1], 'y-')
                # plt.axis([-50, 150, -50, 150])
                # plt.pause(0.05)
                # plt.clf()
                # show the output images
                # cv2.imshow("Before NMS", orig)

            cv2.imshow("Detection", frame_rgb)
            cv2.waitKey(1)
            time.sleep(0.01)

    def stop(self):
        self.input_thread.join(5)
        self.detection_thread.join(5)
        self.display_thread.join(5)

def main():
    cam_process = CameraDetection()
    while True:
        try:
            time.sleep(1)
            # esc to quit
            if cv2.waitKey(1) == 27:
                break
                #
                # # Write the frame into the file 'output.avi'
                # out.write(frame_rgb)
                # cv2.destroyAllWindows()
                # out.release()
                # cam.release()
        except KeyboardInterrupt:
            print("press keyboard interrupt")
            cam_process.stop()


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]

    for keys in keys_list:
        FLAGS.__delattr__(keys)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

if __name__ == '__main__':
    main()