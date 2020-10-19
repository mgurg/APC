import os,sys
import numpy as np
from cv2 import cv2

video_file = "./CCTV_2.mp4"

def load_video(abs_path):
    if os.path.exists(video_file):
        try:
            cap = cv2.VideoCapture(video_file)
        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}")
        except: #handle other exceptions such as attribute errors
            print("Unexpected error:", sys.exc_info()[0])
    else:
        raise FileNotFoundError("File not found")
    return cap

if __name__ == '__main__':
    cap = load_video(video_file)

    ret, frame = cap.read()
    if frame is None:
        print("Cannot parse file content")
        sys.exit(1)

    width  = int(cap.get(3))
    height = int(cap.get(4))

    print(f'[INFO] Video OK, width: {width}, height: {height}')

    def bbox(img, cols, rows):
        """
        Draw rectangle bounding box around points.
        """
        side_length = 6
        for x, y in zip(cols, rows):
            top_left = (int(x - side_length / 2), int(y - side_length / 2))
            bot_right = (int(x + side_length / 2), int(y + side_length / 2))
            img = cv2.rectangle(
                img, top_left, bot_right,
                color = (0, 0, 255), thickness = 1
                )
        return img

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    prototxt = f"../mobilenet_ssd/MobileNetSSD_deploy.prototxt"
    model = f"../mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)


    while(True):
        # https://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/
        # cv2.putText(frame, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        img_mask = np.zeros((height, width), np.uint8) # mask
        x_center = 180
        y_center = 15
        radius = 5
        cv2.circle(img_mask,(x_center,y_center),radius,(255,255,255),-1) # measuring area
        a,b,g,r = cv2.mean(frame, mask=img_mask)[::-1]

        if np.mean([b,g,r]) > 10 :
            print([b, g, r])
            cv2.circle(frame, (180,15), 5, (255,0,0), -1)
    
        frame = bbox(frame, [x_center,], [y_center,])

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print('fps:', fps)  # float

        cv2.imshow("APC", frame)
        ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
