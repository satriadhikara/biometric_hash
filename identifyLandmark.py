from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

img_path = "reference.jpg"
img = cv2.imread(img_path)

resp = RetinaFace.detect_faces(img_path, threshold=0.5)
print(resp)


def int_tuple(t):
    return tuple(int(x) for x in t)


for key in resp:
    identity = resp[key]

    # ---------------------
    confidence = identity["score"]

    rectangle_color = (255, 255, 255)

    landmarks = identity["landmarks"]
    diameter = 1
    cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), -1)

    facial_area = identity["facial_area"]

    cv2.rectangle(
        img,
        (facial_area[2], facial_area[3]),
        (facial_area[0], facial_area[1]),
        rectangle_color,
        1,
    )
    # facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
    # plt.imshow(facial_img[:, :, ::-1])

plt.imshow(img[:, :, ::-1])
plt.axis("off")
plt.show()
cv2.imwrite("outputs/" + img_path.split("/")[1], img)
