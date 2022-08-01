import cv2
import numpy as np
from utils import get_four_points


video = cv2.videocapture(0)
w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("원본 동영상 너비(가로) : {}, 높이(세로) : {}".format(w, h))

# 동영상 크기 변환
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 가로
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로

# 변환된 동영상 크기 정보
w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("변환된 동영상 너비(가로) : {}, 높이(세로) : {}".format(w, h))

while True:
    # read : 프레임 읽기
    # [return]
    # 1) 읽은 결과(True / False)
    # 2) 읽은 프레임
    retval, frame = video.read()

    # 읽은 프레임이 없는 경우 종료
    if not retval:
        break

    # 프레임 출력
    cv2.imshow("resize_frame", frame)

    # 'q' 를 입력하면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 동영상 파일 또는 카메라를 닫고 메모리를 해제
video.release()

# 모든 창 닫기
cv2.destroyAllWindows()

#img = cv2.imread("pan.jpg")
img_src = cv2.resize(img,(0,0),fx=0.3, fy=0.3,interpolation=cv2.INTER_AREA)
dst_size = (400, 400, 3)
img_dst = np.zeros(dst_size, np.uint8)
cv2.imshow("dst", img_dst)

# 우리가 원본 이미지로부터는 마우스 클릭으로 4개의 점을 가져올 거다.
cv2.imshow("Image", img_src)
points_src = get_four_points(img_src)

# 새로 만들 이미지에서는, 위의 원본 이미지 4개의 점과 매핑할 점을 잡아줘야 한다.
points_dst = np.array([0, 0,
                       dst_size[1], 0,
                       dst_size[1], dst_size[0],
                       0, dst_size[0]])

points_dst = points_dst.reshape(4, 2)
h, status = cv2.findHomography(points_src, points_dst)
img_dst = cv2.warpPerspective(img_src, h, (dst_size[1], dst_size[0]))

####
# provide points from image 1
pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
# corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
pts_dst = np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])

# calculate matrix H
h, status = cv2.findHomography(pts_src, pts_dst)

# provide a point you wish to map from image 1 to image 2
a = np.array([[154, 174]], dtype='float32')
a = np.array([a])

# finally, get the mapping
pointsOut = cv2.perspectiveTransform(a, h)

cv2.imshow("DST", img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()