import argparse
import time
import cv2
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# #  呼吸率对应的频率:0.15-0.40Hz，https://www.nature.com/articles/s41598-019-53808-9
RR_Min_HZ = 0.15
RR_Max_HZ = 0.40
# RR_Min_HZ = 0.15
# RR_Max_HZ = 0.70

# 采样频率
FPS = 25


def _x1y1wh_to_xyxy(bbox_x1y1wh):
    x1, y1, w, h = bbox_x1y1wh
    x2 = int(x1+w)
    y2 = int(y1+h)
    return x1, y1, x2, y2

def readvideo_infrared(datapath_infrared):
    img_num = 1500
    vc = cv2.VideoCapture(datapath_infrared)  # 读取视频文件
    c = 0
    count_imgs_num = 0
    videonpy = []
    timeF_infrared = 1
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        if (c % timeF_infrared == 0):  # 每隔timeF帧加到数组内;  # timeF 视频帧计数间隔频率
            if frame is not None:
                videonpy.append(frame)
                count_imgs_num = count_imgs_num + 1
        c = c + 1
        # cv2.waitKey(1)
        # if c >= img_num:
        #     break
    vc.release()
    videonpy = np.array(videonpy)
    return videonpy

# 计算img_ROI转化成灰度图之后的平均像素值
def get_avg_gray_pixel(img_ROI):
    gray_img = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
    avg_pixel = np.mean(gray_img)
    # cv2.imshow('gray_img', gray_img)
    return avg_pixel


def draw_ROI_line(x_data_arr, y_data_arr, image, index_ROI):
    points = []
    img_ROI = []
    xmin = x_data_arr[index_ROI[0]] * image.shape[1]
    xmax = x_data_arr[index_ROI[0]] * image.shape[1]
    ymin = y_data_arr[index_ROI[0]] * image.shape[0]
    ymax = y_data_arr[index_ROI[0]] * image.shape[0]

    for kk in range(len(index_ROI) - 1):
        x1 = int(x_data_arr[index_ROI[kk]] * image.shape[1])
        y1 = int(y_data_arr[index_ROI[kk]] * image.shape[0])
        x2 = int(x_data_arr[index_ROI[kk + 1]] * image.shape[1])
        y2 = int(y_data_arr[index_ROI[kk + 1]] * image.shape[0])
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # 连线

        # 求最大最小值
        if x1 >= xmax:
            xmax = x1
        else:
            xmin = x1

        if y1 >= ymax:
            ymax = y1
        else:
            ymin = y1

    # print('xmin = ', xmin)
    # print('xmax = ', xmax)
    # print('ymin = ', ymin)
    # print('ymax = ', ymax)
    img_ROI = image[int(ymin):int(ymax), int(xmin):int(xmax)]

    # 初始状态的位置信息 [x, y, w, h]
    w = xmax - xmin
    h = ymax - ymin
    init_state = [xmin, ymin, w, h]  
    pos_ROI = [xmin, ymin, xmax, ymax]

    return img_ROI, pos_ROI

def get_face_landmarks(face_landmarks):

    # https://so.muouseo.com/qa/pvw0v2yxx6j1.html
    x_data_arr, y_data_arr, z_data_arr =[],[], []
    for landmark in face_landmarks.landmark:
        x = landmark.x
        y = landmark.y
        z = landmark.z
        x_data_arr.append(x)
        y_data_arr.append(y)
        z_data_arr.append(z)
   
    return x_data_arr,  y_data_arr, z_data_arr



def save_img(img_roi, roi_name, count, sub_file_path):
    if img_roi.shape[0]!= 0 and img_roi.shape[1]!= 0 and  img_roi.shape[2]!= 0:
        path_save = sub_file_path + '/ROIs/' +'/'+'ROI_'+ roi_name +'/'
        # path_save = './ROIs/' + save_img_ID +'/' +'ROI_'+ 'new_'+roi_name +'/'
        print(path_save)
        # 判断路径是否存在
        if os.path.exists(path_save):
            pass
        else:
            os.makedirs(path_save)

        savepath = path_save + str(count) +'.jpg'
        print('savepath = ', savepath)
        # save
        cv2.imwrite(savepath, img_roi)




def position_ROI(pos_ROI):
    xmin, ymin, xmax, ymax = pos_ROI[0], pos_ROI[1], pos_ROI[2], pos_ROI[3]

    return xmin, ymin, xmax, ymax 


def  get_ROI(image, x_data_arr, y_data_arr, pos_ind):
    xmax = 0
    xmin = image.shape[1]
    ymax = 0
    ymin = image.shape[0]
    print('len(x_data_arr) = ', len(x_data_arr))
    for k in range(len(x_data_arr)):
        x = int(x_data_arr[k] * image.shape[1])
        y = int(y_data_arr[k] * image.shape[0])
        for j in range(len(pos_ind)):
            if pos_ind[j] ==  k:
                ## 排除不在图片内的点：x >= 0 and y >= 0
                if x >= xmax and x >= 0:
                    xmax = x
                if x <= xmin and x >= 0:
                    xmin = x
                if y >= ymax and y >= 0:
                    ymax = y
                if y <= ymin and y >= 0:
                    ymin = y
    ROI = image[ymin:ymax, xmin:xmax]
    return ROI


def  get_face(image, x_data_arr, y_data_arr):
    xmax = 0
    xmin = image.shape[1]
    ymax = 0
    ymin = image.shape[0]
    # print('len(x_data_arr) = ', len(x_data_arr))
    for k in range(len(x_data_arr)):
        x = int(x_data_arr[k] * image.shape[1])
        y = int(y_data_arr[k] * image.shape[0])
        # for j in range(len(pos_ind)):
        # if pos_ind[j] ==  k:
        ## 排除不在图片内的点：x >= 0 and y >= 0
        if x >= xmax and x >= 0:
            xmax = x
        if x <= xmin and x >= 0:
            xmin = x
        if y >= ymax and y >= 0:
            ymax = y
        if y <= ymin and y >= 0:
            ymin = y
    ROI = image[ymin:ymax, xmin:xmax]
    print('ROI.shape = ', ROI.shape)
    return ROI

def main():
    path_video = './2021-02-24-视频汇总-anxiety_paper'

    dirs= os.listdir(path_video)
    print(dirs)
    for file in dirs:
        file_video = path_video +'/'+file
        print(file_video)
        print(file[:-4])

        #################### mediapipe人脸检测 #####################
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh

        # For static images:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        # For webcam input:
        face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        #################### mediapipe人脸检测 #####################

       
        # datapath_infrared  = 'AU2.mp4'
        datapath_infrared = file_video
       
        img_arr = readvideo_infrared(datapath_infrared)
        print('len(img_arr):',len(img_arr))
        total_num = len(img_arr)
        ppg_infrared_nose = []
        face_mark = 0
        init_state = []
        count = 0
        for i in range(total_num):
            image = img_arr[i]
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('image', img)

            #################### mediapipe人脸检测 #####################
            if results.multi_face_landmarks:
                # 异常处理
                try:
                    for face_landmarks in results.multi_face_landmarks:
                        # 获取特征点
                        # face_landmarks_str = str(face_landmarks)
                        x_data_arr, y_data_arr, z_data_arr = get_face_landmarks(face_landmarks)

                        for k in range(len(x_data_arr)):
                              # print('len(x_data_arr) = ',len(x_data_arr))
                              ## 画特征点
                              x = int(x_data_arr[k]*image.shape[1])
                              y = int(y_data_arr[k]*image.shape[0])
                              pt_pos = (x, y)
                              # cv2.circle(image, pt_pos, 1, (0, 255, 0), 1) # 画点
                              # 眼睛
                              # eyeRight = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
                              # eyeLeft = [7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33, 7]
                              # eyeRight = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]
                              # eyeLeft = [7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]

                              eyeRight = [463, 414, 286, 258, 257, 259, 260, 467,359, 255, 339, 254, 253, 252, 256, 341, 463 ]# NEW
                              eyeLeft = [130, 247, 30, 29, 27,28, 56, 190, 243, 112,26, 22, 23, 24, 110, 25, 130] # NEW
                              # 眉毛
                              eyebrowRight = [336,296,334,293,300,276,283,282,295,285,336]
                              eyebrowLeft =[70,63,105,66,107,55,65,52,53,46,70]
                              
                              # 嘴唇
                              # lipUpper = [61,76,62,78,191,80,81,82,13,312,311,310,415,308,291,409,270,269,267,0,37,39,40,185,61]
                              # lipLower = [61,76,62,78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]

                              # 嘴巴
                              mouth = [61,76,62,78,191,80,81,82,13,312,311,310,415,308,291,409,270,269,267,0,37,39,40,185,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]

                              # 鼻子
                              # nose = [129, 358, 322, 92, 129]
                              nose = [209, 429, 393, 165, 129]

                              # 额头
                              forehead = [109, 338, 337, 108, 109]

                        # nose_roi, pos_nose_roi = draw_ROI_line(x_data_arr, y_data_arr, image, nose)
                        # mouth_roi, pos_mouth_roi = draw_ROI_line(x_data_arr, y_data_arr, image, mouth)
                        # eyeRight_roi, pos_eyeRight_roi = draw_ROI_line(x_data_arr, y_data_arr, image, eyeRight)
                        # eyeLeft_roi, pos_eyeLeft_roi = draw_ROI_line(x_data_arr, y_data_arr, image, eyeLeft)
                        # xmin, ymin, xmax, ymax = position_ROI(pos_eyeRight_roi)
                        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                        # cv2.imshow('image', image)
                        # cv2.waitKey(0)


                        mouth_roi = get_ROI(image, x_data_arr, y_data_arr, mouth)
                        eyeRight_roi = get_ROI(image, x_data_arr, y_data_arr, eyeRight)
                        eyeLeft_roi = get_ROI(image, x_data_arr, y_data_arr, eyeLeft)
                        face_roi = get_face(image, x_data_arr, y_data_arr)
                        # cv2.imshow('face_roi', face_roi)
                        # cv2.waitKey(0)


                        # SAVE IMG
                        save_img(file[:-4], mouth_roi, 'mouth', count)
                        save_img(file[:-4], eyeRight_roi, 'eyeRight', count)
                        save_img(file[:-4], eyeLeft_roi, 'eyeLeft', count)
                        # save_img(file[:-4], face_roi, 'face', count)


                        count = count + 1
                except:
                    continue

if __name__ == "__main__":
    # args = parse_args()
    main()
