import cv2

cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)

if not cap.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cap.read()

    if ret:
        print("Camera working:", frame.shape)
        cv2.imshow("RealSense RGB", frame)
        cv2.waitKey(0)
    else:
        print("Camera opened, but no frame received")

cap.release()
cv2.destroyAllWindows()