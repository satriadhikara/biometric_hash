from retinaface import RetinaFace
import cv2

img_path = "reference.jpg"
faces = RetinaFace.detect_faces(img_path)

identity = faces['face_1']
 
print(identity)
facial_area = identity["facial_area"]
landmarks = identity["landmarks"]
 
#  extract facial area
img = cv2.imread(img_path)
facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

# #highlight facial area
cv2.rectangle(img, (facial_area[2], facial_area[3])
   , (facial_area[0], facial_area[1]), (255, 255, 255), 1)
 
#highlight the landmarks
cv2.circle(img, tuple(landmarks["left_eye"]), 1, (0, 0, 255), -1)
cv2.circle(img, tuple(landmarks["right_eye"]), 1, (0, 0, 255), -1)
cv2.circle(img, tuple(landmarks["nose"]), 1, (0, 0, 255), -1)
cv2.circle(img, tuple(landmarks["mouth_left"]), 1, (0, 0, 255), -1)
cv2.circle(img, tuple(landmarks["mouth_right"]), 1, (0, 0, 255), -1)