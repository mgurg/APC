import numpy as np
from cv2 import cv2

cap = cv2.VideoCapture("./CCTV_2.mp4")
ret, frame = cap.read()

while(True):
    cv2.putText(frame, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    cv2.imshow("APC", frame)
    ret, frame = cap.read()

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
