import calculate as calc
import numpy as np
import cv2
import backends.mp_hands_onnx.mediapipe_hands as mp_hands
import matplotlib.pyplot as plt

def visualize_palm(palms, img):
    debug_image = img.copy()
    w, h = img.shape[1], img.shape[0] # of output image

    if len(palms) > 0:
        rects = calc.calculate_rects(palms, img)

        # Get the palm images with corrected rotation angles
        cropted_rotated_hands_images = calc.rotate_and_crop_rectangle(image=img, rects_tmp=rects, operation_when_cropping_out_of_range='padding')

        # # DEBUG
        # plt.imshow(cropted_rotated_hands_images[0])
        # plt.show()

        # Debug: visualization of detected palms
        # TODO split into function
        for rect in rects:
            rects_tuple = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
            box = cv2.boxPoints(rects_tuple).astype(np.int0)
            # Draw the rectangle for the detected palm.
            cv2.drawContours(debug_image, [box], 0,(0,0,255), 2, cv2.LINE_AA)

            rcx = int(rect[0])
            rcy = int(rect[1])
            half_w = int(rect[2] // 2)
            half_h = int(rect[3] // 2)
            x1 = rcx - half_w
            y1 = rcy - half_h
            x2 = rcx + half_w
            y2 = rcy + half_h

            # Display dimensions of the bounding box.
            text_x = max(x1, 10)
            text_x = min(text_x, w-120)
            text_y = max(y1-15, 45)
            text_y = min(text_y, h-20)
            # not_rotate_rects.append([rcx, rcy, x1, y1, x2, y2, 0])
            cv2.putText(debug_image, f'{y2-y1}x{x2-x1}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA,)
            cv2.putText(debug_image, f'{y2-y1}x{x2-x1}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59,255,255), 1, cv2.LINE_AA,)
            cv2.rectangle(debug_image, (x1,y1), (x2,y2), (0,128,255), 2, cv2.LINE_AA,)
            cv2.circle(debug_image, (rcx, rcy), 3, (0, 255, 255), -1,)

            base_point = np.asarray(
                [rcx, rcy],
                dtype=np.float32,
            )
            # points = np.asarray(
            #     list(palm_trackid_cxcy.values()),
            #     dtype=np.float32,
            # )

    return debug_image
    # return cropted_rotated_hands_images[0]


# def viz_processed_palm(preds, img): # a helper to visualize // NOTE old! receives preds, not palms
#     palms = mp_hands.postprocess_palms(np.array(img), preds)
#     rects = calc.calculate_rects(palms, np.array(img))
#     return calc.rotate_and_crop_rectangle(image=np.array(img), rects_tmp=rects, operation_when_cropping_out_of_range='padding')[0]

def viz_processed_palm(palms, img): # a helper to visualize
    rects = calc.calculate_rects(palms, np.array(img))
    return calc.rotate_and_crop_rectangle(image=np.array(img), rects_tmp=rects, operation_when_cropping_out_of_range='padding')[0]


# NOTE viz function from original repo - hasn't been adjusted to fit current context
# def visualize_landmarks(hand_landmarks, img):
#     if len(hand_landmarks) > 0:
#         pre_processed_landmarks = []
#         pre_processed_point_histories = []
#         for (trackid, x1y1), landmark, rotated_image_size_leftright, not_rotate_rect in \
#             zip(palm_trackid_box_x1y1s.items(), hand_landmarks, rotated_image_size_leftrights, not_rotate_rects):

#             x1, y1 = x1y1
#             rotated_image_width, _, left_hand_0_or_right_hand_1 = rotated_image_size_leftright
#             thick_coef = rotated_image_width / 400
#             lines = np.asarray(
#                 [
#                     np.array([landmark[point] for point in line]).astype(np.int32) for line in lines_hand
#                 ]
#             )
#             radius = int(1+thick_coef*5)
#             cv2.polylines(
#                 img,
#                 lines,
#                 False,
#                 (255, 0, 0),
#                 int(radius),
#                 cv2.LINE_AA,
#             )
#             _ = [cv2.circle(img, (int(x), int(y)), radius, (0,128,255), -1) for x,y in landmark[:,:2]]
#             left_hand_0_or_right_hand_1 = left_hand_0_or_right_hand_1 # if args.disable_image_flip else (1 - left_hand_0_or_right_hand_1)
#             handedness = 'Left ' if left_hand_0_or_right_hand_1 == 0 else 'Right'
#             _, _, x1, y1, _, _, _ = not_rotate_rect
#             text_x = max(x1, 10)
#             # text_x = min(text_x, cap_width-120)
#             text_y = max(y1-70, 20)
#             # text_y = min(text_y, cap_height-70)
#             cv2.putText(
#                 img,
#                 f'trackid:{trackid} {handedness}',
#                 (text_x, text_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0,0,0),
#                 2,
#                 cv2.LINE_AA,
#             )
#             cv2.putText(
#                 img,
#                 f'trackid:{trackid} {handedness}',
#                 (text_x, text_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (59,255,255),
#                 1,
#                 cv2.LINE_AA,
#             )

#             # pre_processed_landmark = pre_process_landmark(
#             #     landmark,
#             # )
#             # pre_processed_landmarks.append(pre_processed_landmark)

# # cv2_imshow(cv2.cvtColor(visualize_palm([onnx_pred], np.array(test_img)), cv2.COLOR_BGR2RGB))

# # cv2_imshow(cv2.cvtColor(visualize_palm([tvm_pred], np.array(test_img)), cv2.COLOR_BGR2RGB))



def xyz_x21_to_yx(a):
    b = np.reshape(a, (21, 3))
    return b[:, 0:2]

def visualize_landmarks_0(pred, img):
    plt.imshow(img)
    ldmks = xyz_x21_to_yx(pred)
    plt.scatter(ldmks[:, 0], ldmks[:, 1], c='w', s=10)