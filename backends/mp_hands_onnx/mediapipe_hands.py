import numpy as np
import math
import calculate as calc


SCORE_THRESHOLD = 0.50
SQUARE_STANDARD_SIZE = 192
SQUARE_PADDING_HALF_SIZE = 0


# for box in boxes:
#     pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
#     if box_size > 0:
#         kp02_x = kp2_x - kp0_x
#         kp02_y = kp2_y - kp0_y
#         sqn_rr_size = 2.9 * box_size
#         rotation = 0.5 * math.pi - math.atan2(-kp02_y, kp02_x)
#         rotation = normalize_radians(rotation)
#         sqn_rr_center_x = box_x + 0.5*box_size*math.sin(rotation)
#         sqn_rr_center_y = box_y - 0.5*box_size*math.cos(rotation)
#         sqn_rr_center_y = (sqn_rr_center_y * SQUARE_STANDARD_SIZE - SQUARE_PADDING_HALF_SIZE) / image_height
#         hands.append(
#             [
#                 sqn_rr_size,
#                 rotation,
#                 sqn_rr_center_x,
#                 sqn_rr_center_y,
#             ]
#         )


def postprocess_palms(
    image: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """__postprocess

    Parameters
    ----------
    image: np.ndarray
        Entire image.

    boxes: np.ndarray
        float32[N, 8]
        pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y

    Returns
    -------
    hands: np.ndarray
        float32[N, 4]
        sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    hands = []
    keep = boxes[:, 0] > SCORE_THRESHOLD # pd_score > self.score_threshold
    boxes = boxes[keep, :]

    for box in boxes:
        pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
        if box_size > 0:
            kp02_x = kp2_x - kp0_x
            kp02_y = kp2_y - kp0_y
            sqn_rr_size = 2.9 * box_size
            rotation = 0.5 * math.pi - math.atan2(-kp02_y, kp02_x)
            rotation = calc.normalize_radians(rotation)
            sqn_rr_center_x = box_x + 0.5*box_size*math.sin(rotation)
            sqn_rr_center_y = box_y - 0.5*box_size*math.cos(rotation)
            sqn_rr_center_y = (sqn_rr_center_y * SQUARE_STANDARD_SIZE - SQUARE_PADDING_HALF_SIZE) / image_height
            hands.append(
                [
                    sqn_rr_size,
                    rotation,
                    sqn_rr_center_x,
                    sqn_rr_center_y,
                ]
            )

    return np.asarray(hands)


