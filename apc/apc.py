import os,sys
import numpy as np
from cv2 import cv2
import time

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
    W = int(cap.get(3))
    H = int(cap.get(4))

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

    # YOLO
    weightsPath = os.path.sep.join(["../yolo-coco/", "yolov3.weights"])
    configPath = os.path.sep.join(["../yolo-coco/", "yolov3.cfg"])

    labelsPath = os.path.sep.join(["../yolo-coco/", "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    print("Yolo classes: ", len(LABELS))

    print("[INFO] loading YOLO from disk...")
    ynet = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    def mobilenet_ssd(frame):
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

        return frame

    def yolo(image):



        ln = ynet.getLayerNames()
        ln = [ln[i[0] - 1] for i in ynet.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        ynet.setInput(blob)
        start = time.time()
        layerOutputs = ynet.forward(ln)
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                if classIDs[i] == 0: # person
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

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

            # frame = mobilenet_ssd(frame)
            yolo(frame)

        frame = bbox(frame, [x_center,], [y_center,])       

        # fps = cap.get(cv2.CAP_PROP_FPS)
        # print('fps:', fps)  # float

        cv2.imshow("APC", frame)
        ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
