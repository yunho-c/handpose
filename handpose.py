import numpy as np
from PIL import Image
import onnxruntime as ort

import backends.mp_hands_onnx.mediapipe_hands as mp_hands
import calculate as calc
import visualize

import cv2
import matplotlib.pyplot as plt


# TODO a class to hold the models... and an end-to-end wrapper method!
# TODO some way to configure the usage/visualization.


# TODO make a comprehensive map of functions and variables/information passed to make it easier to understand. like I did for ttstokenizer.
# 함수/변수레벨 코드 디펜던시 그래프를 자동으로 그려줬으면 너무, 너무 좋겠다. ㅠㅠ 모듈 디자인하기 힘들어, 시각적 자료 없이는!


def test():
    # palm model
    palm_file = './backends/mp_hands_onnx/model/palm_detection/palm_detection_full_inf_post_192x192.onnx'
    palm_model = ort.InferenceSession(palm_file, providers=['CPUExecutionProvider'])
    palm_size = 192

    # landmark model
    ldmk_file = './backends/mp_hands_onnx/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx'
    ldmk_model = ort.InferenceSession(ldmk_file, providers=['CPUExecutionProvider'])
    ldmk_size = 224

    # test image
    img_path = "./test_images/hand.jpg"
    palm_img = Image.open(img_path).resize((palm_size, palm_size))
    ldmk_img = Image.open(img_path).resize((ldmk_size, ldmk_size))

    # run palm detection model
    x = np.array(palm_img).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)/256
    onnx_preds = palm_model.run(["pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y"], {"input": x})

    # filter bboxes somehow (e.g., score, temporality)
    # NOTE for now, select the first one (i.e., w/ highest confidence) 
    # TODO
    onnx_pred = onnx_preds[0]

    # post-process palm result
    # NOTE output format: [`sqn_rr_size`, `rotation`, `sqn_rr_center_x`, `sqn_rr_center_y`]
    # palms_info = mp_hands.postprocess_palms(np.array(palm_img), onnx_preds)
    palms_info = mp_hands.postprocess_palms(np.array(palm_img), onnx_pred)

    # optionally, visualize palm
    palm_viz = visualize.visualize_palm(palms_info, np.array(palm_img)) # NOTE not sure if module is structured correctly, but fuck it. just think simply & in terms of first principles.
    # NOTE also, how can it visualize palm when img isn't passed to it !?
    cv2.imshow('Palm Visualization', palm_viz)
    cv2.waitKey(0)

    # rotate and crop
    palms_rects = calc.calculate_rects(palms_info, np.array(palm_img))
    palms_imgs = calc.rotate_and_crop_rectangle(image=np.array(palm_img), rects_tmp=palms_rects, operation_when_cropping_out_of_range='padding')

    # run landmark model
    x2 = np.array(ldmk_img).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)/256
    ldmk_onnx_preds = ldmk_model.run(["xyz_x21"], {"input": x2})[0]
    
    # visualize (one-time)
    visualize.visualize_landmarks_0(ldmk_onnx_preds, np.array(ldmk_img))
    plt.show()


def main():
    test()


if __name__ == "__main__":
    main()