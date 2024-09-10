import cv2
import numpy

xml = 'cascade/haarcascade_frontalface_default.xml'
maskpath = 'image/mask/sunglasses_test2.png'

detector = cv2.CascadeClassifier(xml)
mask_orig = cv2.imread(maskpath, -1)

cap = cv2.VideoCapture(0)

mask = mask_orig.copy()

if cap.isOpened():

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ratio = height / width
    ratio_mask = mask.shape[0] / mask.shape[1]

    width = 800
    height = int(width * ratio)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        ret, frame = cap.read()

        if ret is None:
            print('Camera Error.')
            break

        mask = mask_orig.copy()

        frame_gaussian = cv2.GaussianBlur(frame, (3, 3), 1)
        faces = detector.detectMultiScale(frame_gaussian)

        # faces = detector.detectMultiScale(frame)

        if faces is not None:
            for x, y, w, h in faces:

                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                h2 = int(w * ratio_mask)

                mask = cv2.resize(mask, (w, h2))

                alpha_mask = mask[:, :, 3] / 255.0
                alpha_orig = 1.0 - alpha_mask

                y += h // 3

                for c in range(0, 3):
                    frame[y:y + h2, x:x + w, c] = (
                                alpha_mask * mask[:, :, c] + alpha_orig * frame[y:y + h2, x:x + w, c])

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()   
