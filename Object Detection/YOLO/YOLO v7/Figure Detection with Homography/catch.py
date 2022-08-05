import cv2    # OpenCV import
import numpy as np  # 행렬(img)를 만들기 위한 np import

from serial.tools import list_ports
import pydobot


#dobot code
available_ports = list_ports.comports()
print(f'available ports: {[x.device for x in available_ports]}')
port = available_ports[1].device
# sudo python3 catch.py

device = pydobot.Dobot(port=port, verbose=True)

(x, y, z, r, j1, j2, j3, j4) = device.pose()
#print(f'x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}')
#이동시 z값은 45으로 통일, z 맨 아래는 6

# 초기 위치
X = 210
Y = 0
Z = 45
high_z = 45
low_z = 6


def figureMove(pre_x,pre_y,x,y):   #figureMove(초기 x좌표, 초기 y좌표, 이동후 x좌표, 이동후 y좌표)
    if(pre_x**2 + pre_y**2 <=36100):
        print("pre point too small")
        return
    elif (pre_x ** 2 + pre_y ** 2 >= 96100):
        print("pre point too big")
        return
    elif (x ** 2 + y ** 2 >= 96100):
        print("post point too big")
        return
    elif (x ** 2 + y ** 2 >= 96100):
        print("post point too big")
        return
    #else:
    #    print("correct")
    #    return
    device.move_to(X, Y, Z, r, wait=False)  # 도봇 원위치
    device.move_to(pre_x, pre_y, high_z, r,wait=False) #피규어 초기 위치 위로 이동
    device.move_to(pre_x, pre_y, low_z, r, wait=False) #피규어 초기 위치 아래로 이동
    device.move_to(pre_x, pre_y, low_z, r, wait=True)  # 피규어 집기
    device.grip(1)
    device.wait(500)

    device.move_to(pre_x, pre_y, high_z, r, wait=False)  # 피규어 초기 위치 위로 이동
    device.move_to(x,y,high_z,r,wait=False)            # 피규어 이동 위치 위로 이동
    device.move_to(x,y,low_z,r,wait=False)             # 피규어 이동 위치 아래로 이동
    device.move_to(x,y,low_z,r,wait = True)             # 피규어 놓기
    device.grip(2)
    device.wait(500)
    device.grip(0)
    device.move_to(x, y, high_z, r, wait=False)          # 피규어 이동 위치 위로 이동
    device.move_to(X,Y,Z,r,wait=False)             # 도봇 원위치

mouserecog = []

# 마우스 이벤트 콜백함수 정의
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouserecog.append(x)
        print(mouserecog)
        mouserecog.append(y)
        print(mouserecog)
        return

img = cv2.imread('warp_image.png') # 행렬 생성, (가로, 세로, 채널(rgb)),bit)
cv2.namedWindow('image')  #마우스 이벤트 영역 윈도우 생성
cv2.setMouseCallback('image', mouse_callback)


while True :
    cv2.imshow('image', img)

    if len(mouserecog) == 4:
        figureMove(((mouserecog[0])/2)+125, -((mouserecog[1])/2 - 140), ((mouserecog[2]) / 2)+125,
                   -((mouserecog[3])/2 - 140))
        mouserecog = []

    k = cv2.waitKey(1) & 0xFF

    if k == 27:    # ESC 키 눌러졌을 경우 종료
        print("ESC 키 눌러짐")
        break

cv2.destroyAllWindows()


#########################################################################################################################################
