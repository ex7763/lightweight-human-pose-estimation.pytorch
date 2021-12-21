#!/usr/bin/env python3
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    count = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('live', frame)
        cv2.imwrite(f'images/{count:06d}.jpg', frame)
        count += 1

        ret = cv2.waitKey(1)
        if ret == ord('q') or ret == '27':
            break

    cap.release()
    cv2.destroyAllWindows()
