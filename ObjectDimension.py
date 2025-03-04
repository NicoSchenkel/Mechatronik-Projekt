import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics.utils import LOGGER
from ultralytics import YOLO
LOGGER.setLevel(50)  # Setzt das Logging-Level auf CRITICAL, sodass nur kritische Fehler ausgegeben werden.

import time
import csv
import os



def getObjectDimension(pipeline, depth_scale, align):
    # Laden des YOLO-Modells
    model = YOLO("yolov8s.pt")
   
    # Criteria for bounding boxes
    minConf = 0.15                 
    maxObjectDistanz = 1  # in m  # maybe blur all behind it?
    # minObjectDistanz?
    ignore = ["person", "dining table", "chair"]    

    # Safe size of object every tfreq seconds
    tmeass = 0.25
    tInterval = 3                                                       #  maybe reduce it to two'''
    tCompare = time.time()
    i = -1  # index for saving values

    # Arrays für die Bounding Box-Größen                                # Stattdessen Mittelpunkt der Box nehmen?
    xValues = [-50] * int(tInterval / tmeass)
    yValues = [-50] * int(tInterval / tmeass)
    tolerance = 20  # tolerance of derivating of object size
    stable_object_detected = False  # Status: Stabil erkannt


    # Data for Gripper
    dataOfInterest = ['label','xsize', 'ysize', 'distance', 'confidence'  ]
    gripCords = [np.nan, np.nan]
    gripDist = 0
    gripData = [] 
    

    try:
        end = None  # Variable für das finale Bild mit der roten Bounding Box  # waum wird das gemacht?

        while True:
            # Get images
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue
            
            # Variables for calculating dimension
            profile = depth_frame.get_profile()
            intrinsics = profile.as_video_stream_profile().get_intrinsics()
            arraySize = 3 # Size for calculating depth of object


            color_img = np.asanyarray(color_frame.get_data())
            #depth_img = np.asanyarray(depth_frame.get_data())
            color_img1 = color_img.copy()
            # Getting YOLO predictions
            results = model.predict(color_img, conf=minConf)

            nearest_box = None
            nearest_distance = float('inf')
            nearest_details = None

            if results[0].boxes:
                for result in results[0].boxes:
                    class_id = int(result.cls.item())  
                    class_name = model.names[class_id] 
                    
                    # Filter for non- interrupting  objects
                    if class_name in ignore:
                        continue

                    box = result.xyxy[0].cpu().numpy().astype(int)
                    xmin, ymin, xmax, ymax = box
                    xCenter = (xmin + xmax) // 2
                    yCenter = (ymin + ymax) // 2
                    
                    zArray = np.asanyarray(depth_frame.get_data())

                    #gripDist = depth_frame.get_distance((xmin + xmax) // 2, (ymin + ymax) // 2)

                    #z1 = zArray[yCenter-arraySize:yCenter+arraySize, xCenter-arraySize: xCenter+arraySize]
                    
                    z1 = zArray[yCenter-arraySize : yCenter+arraySize, xCenter-arraySize : xCenter+arraySize]
                    

                    gripDist = np.median(z1) * depth_scale
                    coordinate_3d1 = rs.rs2_deproject_pixel_to_point(intrinsics, (xmax,ymax), gripDist)
                    coordinate_3d2 = rs.rs2_deproject_pixel_to_point(intrinsics, (xmin,ymin), gripDist)

                    x1,y1,_ = coordinate_3d1
                    x2, y2, z2 = coordinate_3d2

                    xdimension = np.sqrt(abs(x1-x2)**2)*100
                    ydimension = np.sqrt(abs(y1-y2)**2)*100

                    if gripDist > 0:
                        mean_depth = gripDist
                    else:
                        mean_depth = float('inf')           # Hier lieber continue? Ich denke ja
                        #continue

                    # Get closest object
                    if mean_depth < nearest_distance and mean_depth > 0:
                        nearest_distance = mean_depth
                        nearest_box = box
                        nearest_details = {
                            "box": box,
                            "distance": mean_depth,
                            "confidence": result.conf[0].item(),
                            "class_id": int(result.cls.item()),
                        }
                # Filter fo Objects in distance -> check for other methods ( of realsense cam)
                if nearest_details and nearest_details["distance"] <= maxObjectDistanz:
                    box = nearest_details["box"]

                    # Green Box for detected object
                    if not stable_object_detected:
                        color_img1 = color_img.copy()
                        cv2.rectangle(color_img1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        label = f"{model.names.get(nearest_details['class_id'], 'Unknown')} {xdimension:.2f}x{ydimension:.2f} ({nearest_details['distance']:.2f} m)"
                        cv2.putText(color_img1, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                    # Get box size
                    if time.time() - tCompare >= tmeass:
                        tCompare = time.time()
                        
                        i = (i + 1) % len(xValues)
                        yValues[i] = (ymax - ymin)                                    #if poblems with distance get array of distances!
                        xValues[i] = (xmax - xmin)


                        # Check if Object fullfills stability criteria
                        if (max(yValues) - min(yValues)) < tolerance and (max(xValues) - min(xValues)) < tolerance:
                            gripCords = [np.median(xValues), np.median(yValues)]
                            gripDist = nearest_details["distance"]

                          
                            print("*******************************************************************************************************************")
                            print(f"Object: {model.names[nearest_details['class_id']]}")
                            print(f"Bounding Box size: {gripCords}")
                            print(f"Distance: {gripDist:.3f}")
                            print("*******************************************************************************************************************")

                            # Erstelle ein neues Bild nur mit der roten Bounding Box
                            end = color_img.copy()
                            cv2.rectangle(end, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                            cv2.putText(end, f'Object for Gripper detected! Size:  {xdimension:.2f}x{ydimension:.2f}', (box[0], box[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # Programm beenden, da das Objekt stabil erkannt wurde
                            stable_object_detected = True
                            break
            # See what happens by delelting this 
            # Zeige das Bild mit der grünen Bounding Box während der Verarbeitung
            if not stable_object_detected:
                cv2.imshow('Object Detection', color_img1)

            if stable_object_detected:
                break  # Beende die Schleife, wenn das Objekt stabil erkannt wurde

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()

        # Zeige das finale Bild mit der roten Bounding Box
        if end is not None:
            cv2.destroyWindow("Object Detection")
            cv2.imshow('Ready to Grip', end)

            # Get longest and shortes side of object
            if gripCords[0]< gripCords[1]:
                maxSide = gripCords[1]
                minSide = gripCords[0]
            elif gripCords[1]< gripCords[0]:
                    maxSide = gripCords[0]
                    minSide = gripCords[1]
            else:
                maxSide = gripCords[0]      # doesnt really matter 

            
            gripData.append({
                "label"     : int(nearest_details['class_id']),
                "longest side": maxSide,
                "shortesSide" : minSide,
                "distance"  : round((gripDist),4),
                "confidence": round((nearest_details["confidence"]),4),

                })
        
        

            cv2.waitKey(0)
            
        else:
            print('No stable object detected.')
        cv2.destroyAllWindows()
        
    # return  [(obj.values()) for obj in gripData]
    return [list(obj.values()) for obj in gripData]


    #print(f"Box Size: x: {gripCords[0]}, y:{gripCords[1]} \nDistance to center of object: {gripDist:.3f}")
    
# Objekterkennung
# Objektfilterung
# Objektsatbilität
# Worflow



def getObjectData(mode):
    # Laden des YOLO-Modells
    model = YOLO("yolov8s.pt")
    
    # Kriterien für Bounding Boxes
    minConf = 0.15
    maxObjectDistanz = 2  # Maximale Entfernung für Objekte (in Metern)
    ignore = ["person", "dining table", "chair"]
    
    # Messparameter
    tmeass = 0.25
    tInterval = 2  # Zeit für Stabilitätsprüfung
    tCompare = time.time()
    i = -1
    
    # Arrays für Bounding Box-Größen
    xValues = [-50] * int(tInterval / tmeass)
    yValues = [-50] * int(tInterval / tmeass)
    tolerance = 20  # Toleranz für Objektstabilität
    stable_object_detected = False
    
    # Daten für den Greifer
    gripCords = [np.nan, np.nan]
    gripDist = 0
    gripData = []
    
    # RealSense-Pipeline konfigurieren
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    try:
        end = None
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            color_img1 = color_img.copy()
            
            results = model.predict(color_img, conf=minConf)
            nearest_distance = float('inf')
            nearest_details = None
            
            if results[0].boxes:
                for result in results[0].boxes:
                    class_id = int(result.cls.item())
                    class_name = model.names[class_id]
                    
                    if class_name in ignore:
                        continue
                    
                    box = result.xyxy[0].cpu().numpy().astype(int)
                    xmin, ymin, xmax, ymax = box
                    gripDist = depth_frame.get_distance((xmin + xmax) // 2, (ymin + ymax) // 2)
                    
                    if gripDist <= 0 or gripDist > maxObjectDistanz:
                        continue
                    
                    if gripDist < nearest_distance:
                        nearest_distance = gripDist
                        nearest_details = {
                            "box": box,
                            "distance": gripDist,
                            "confidence": result.conf[0].item(),
                            "class_id": class_id,
                        }
            
            if nearest_details and nearest_details["distance"] <= maxObjectDistanz:
                box = nearest_details["box"]
                label = f"{model.names.get(nearest_details['class_id'], 'Unknown')} {box[2] - box[0]}x{box[3] - box[1]} ({nearest_details['distance']:.2f} m)"
                
                if not stable_object_detected:
                    color_img1 = color_img.copy()
                    cv2.rectangle(color_img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(color_img1, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if time.time() - tCompare >= tmeass:
                    tCompare = time.time()
                    i = (i + 1) % len(xValues)
                    yValues[i] = (box[3] - box[1])
                    xValues[i] = (box[2] - box[0])
                    
                    if (max(yValues) - min(yValues)) < tolerance and (max(xValues) - min(xValues)) < tolerance:
                        gripCords = [np.median(xValues), np.median(yValues)]
                        gripDist = nearest_details["distance"]
                        
                        end = color_img.copy()
                        cv2.rectangle(end, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(end, f'Object for Gripper detected! Size: {box[2] - box[0]}x{box[3] - box[1]}',
                                    (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        stable_object_detected = True
                        break
            
            if not stable_object_detected:
                cv2.imshow('Object Detection', color_img1)
            
            if stable_object_detected:
                break
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
    
    finally:
        pipeline.stop()
        if end is not None:
            cv2.destroyWindow("Object Detection")
            cv2.imshow('Ready to Grip', end)
            
            gripData.append({
                "label": int(nearest_details['class_id']),
                "longest side": gripCords[1],
                "shortest side": gripCords[0],
                "distance": round(gripDist, 4),
                "confidence": round(nearest_details["confidence"], 4),
                "mode": mode
            })
            
            file_path = "detected_objects.csv"
            file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
            
            with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                fieldnames = ["label", "longest side", "shortest side", "distance", "confidence", "mode"]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(gripData)
            
            print("Daten erfolgreich in 'detected_objects.csv' gespeichert!")
            cv2.waitKey(0)
        else:
            print('No stable object detected.')
        cv2.destroyAllWindows()
        
    return gripData