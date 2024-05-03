import cv2
import numpy as np 
import matplotlib.pyplot as plt

from ultralytics import YOLO
from tracker import Tracker


incoming_vehicle = set()
outgoing_vehicle = set()

def draw_tracker_bbox(tracker, frame):
    print("In tracker")
    for track in tracker.tracks:
        bbox = track.bbox
        track_id = track.track_id
        print(bbox)
        bbox = list(map(int, bbox))
        p1x, p1y, p2x, p2y = bbox
        if p2x > 320:
            print("Incoming")
            incoming_vehicle.add(track_id)
            cv2.rectangle(frame, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0, 255, 0), 1)
        else:
            print("Outgoing")
            outgoing_vehicle.add(track_id)
            cv2.rectangle(frame, (int(p1x), int(p1y)), (int(p2x), int(p2y)), (0, 0, 255), 1)

        cv2.putText(frame, f'Incoming: {len(incoming_vehicle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Outgoing: {len(outgoing_vehicle)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)



# Reading the video file
video_path = r"C:\Users\SHUBHAM\Downloads\Road traffic video for object recognition.mp4"

vcap = cv2.VideoCapture(video_path)

if vcap.isOpened():
    print("Able to read the Video file")
else:
    print("Unable to open")

# object detection from stable camera, able to detect only moving objects
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
tracker = Tracker()
model = YOLO('yolov8n.pt')

ret, frame = vcap.read()
print("Which detector do you want to select? 1.Yolov8   2. OpenCV contour")
detector_choice = input()
if detector_choice.isnumeric():
    detector_choice = int(detector_choice)
else:
    detector_choice = 1
    print("Please enter valid input. Detecting items using yolov8")


while ret:
     
    if detector_choice == 1:
        output = model(frame)
        detection_thres = 0.4
        for result in output:
            detections = []
            for r in result.boxes.data.tolist():
                # first 4 index are coordinates then we have confidence and class_id
                p1x, p1y, p2x, p2y, score, class_id = r
                detection_info = list(map(int, r[:-2]))
                detection_info.extend([score, class_id])
                print(detection_info)

                if r[4] > detection_thres:
                    detections.append(detection_info[:-1])

            tracker.update(frame, detections)

            draw_tracker_bbox(tracker, frame)
    elif detector_choice == 2:
        height, width, _ = frame.shape
    
        #incoming region of interest
        roi = frame[190:360, 50:620]
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours_incoming, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours_incoming:
            area = cv2.contourArea(cnt)
            if area > 200 :
                #cv2.drawContours(roi_incoming, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)  
                x2 = int(x + w) + 50
                y2 = int(y + h) + 190
                # giving score of 1 to all the bbox
                detections.append([x + 50, y + 190, x2, y2, 1])
        
        tracker.update(frame, detections)
        draw_tracker_bbox(tracker, frame)
    else:
        print("Please enter a valid input.")
        break

    cv2.imshow("frame", frame)
    key = cv2.waitKey(25)

    if key == 27:
        break

    ret, frame = vcap.read()


vcap.release()
cv2.destroyAllWindows()

print("Outgoing", len(outgoing_vehicle))
print("incoming", len(incoming_vehicle))