import cv2
import numpy

img = cv2.imread('image/face/apple.jpg')
ratio = img.shape[0]/img.shape[1]

width = 300
height = int(width*ratio)

img_resized = cv2.resize(img, (width, height))
img_resized = cv2.GaussianBlur(img_resized, (3,3), 1)

face_detector = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
face_detections = face_detector.detectMultiScale(img_resized)

mask = cv2.imread('image\mask\sunglasses_test2.png', -1)

ratio_mask = mask.shape[0]/mask.shape[1]

for x, y, w, h in face_detections:
    # cv2.rectangle(img_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img2_resized = cv2.resize(mask, (w, int(w*ratio_mask)))

    alpha_mask = img2_resized[:, :, 3] / 255.0
    alpha_orig = 1.0 - alpha_mask

    y += h//3

    for c in range(0, 3):
        img_resized[y:y+int(w*ratio_mask), x:x+w, c] = (alpha_mask * img2_resized[:, :, c] + alpha_orig * img_resized[y:y+int(w*ratio_mask), x:x+w, c])

cv2.imshow('image', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
