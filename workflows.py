import cv2
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
import os, time
from .database import session
from .models import Devices, DwellTime
import logging
from dotenv import load_dotenv
import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS
load_dotenv()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("yolo.log"),
                        logging.StreamHandler()
                    ])


api_token = os.getenv('INFLUXDB_API_KEY')
org = ""
url = "http://localhost:8086"
bucket=""
try:
    client = influxdb_client.InfluxDBClient(url=url, token=api_token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
except Exception as e:
    print ('Error establishing connection with Influxdb',e)




def vehicles_in_or_out(cap, configurations):
    
    start = sv.Point(319, 1555)
    end = sv.Point(1803, 1553)

    track_history = defaultdict(lambda:[])
    crossed_objects = {}

    while True:
        flag, frame = cap.read()
        if flag:
            
            result = configurations['model'](frame, classes=[2,3,5,7])[0]
            detections = sv.Detections.from_ultralytics(result)
            track_detections = configurations['tracker'].update_with_detections(detections)

            

            boxes = track_detections.xyxy.tolist()
            track_ids = track_detections.tracker_id.tolist()
            if len(boxes)!=0 and len(track_ids)!=0:
                for box, track_id in zip(boxes, track_ids):
                    x1,y1,x2,y2 = box
                    cx, cy = round((x1+x2)/2,4), round((y1+y2)/2,4)
                    track = track_history[track_id]
                    track.append((cx, cy))

                    if start.x < cx < end.x and abs(cy - start.y) < 10:
                        if track_id not in crossed_objects:
                            crossed_objects[track_id] = True
                
            else:
                print ('No Detections Found')

        else:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return len(crossed_objects)



def find_dwell_time(is_running_flag,cap, configurations):
# def find_dwell_time(cap, configurations):
    # dwell_times = []
    person_detections = {}
    not_present = False

    while is_running_flag['is_running']:
    # while True:
        flag, frame = cap.read()

        if not flag:
            print ('Error reading frame')
            break

        result = configurations['model'](frame, conf= 0.5)[0]

        detections = sv.Detections.from_ultralytics(result)
        class_ids = detections.class_id.tolist()



        if len(class_ids)!=0:
            classes = detections.data['class_name']
            
            point = Point('Detections1') \
            .tag("object",classes) \
            .tag("timestamp",time.strftime("%y-%m-%d %H:%M:%S",time.localtime())) \
            .field("device id",configurations['device_id'])

            write_api.write(bucket=bucket, org= org, record=point)

            logging.info(classes)
            if 'person' not in classes and 'person' not in person_detections:
                raw_left_time = time.localtime()
                left_time = time.strftime("%y-%m-%d %H:%M:%S",raw_left_time)

                person_detections['person'] = {
                    'raw_left_time':raw_left_time,
                    'left_time':left_time
                    }
                not_present = True

            if not_present:
                if 'person' in classes:
                    raw_return_time = time.localtime()
                    return_time = time.strftime("%y-%m-%d %H:%M:%S", raw_return_time)

                    person_detections['person'].update({'return_time':return_time})
                    
                    t1 = time.mktime(person_detections['person']['raw_left_time'])
                    t2 = time.mktime(raw_return_time)

                    dwell_time = abs(t1 - t2)
                    if dwell_time > 5:
                        
                        dt_event = DwellTime(
                            left_time = person_detections['person']['left_time'],
                            return_time = person_detections['person']['return_time'],
                            dwell_time = f'{dwell_time} sec',
                            device_start_time = configurations['device_start_time'],
                            device_id = configurations['device_id']
                        )
                        session.add(dt_event)
                        session.commit()
                        logging.critical('Event Created')

                        point = Point('Dwell Times') \
                        .tag("left time",person_detections['person']['left_time']) \
                        .tag("return time",person_detections['person']['return_time']) \
                        .field("dwell time",dwell_time)


                        write_api.write(bucket=bucket, org= org, record=point)
                        # dwell_times.append({
                        #     'person':{
                        #         'left_time':person_detections['person']['left_time'],
                        #         'return_time':person_detections['person']['return_time'],
                        #         'dwell_time':dwell_time
                        #     }
                        # })
                    not_present = False
                    person_detections.clear()
        else:
            logging.warn('No Detections')
            if 'person' not in person_detections:
                raw_left_time = time.localtime()
                left_time = time.strftime("%Y-%m-%d %H:%M:%S",raw_left_time)
                person_detections['person'] = {'raw_left_time':raw_left_time,
                                               'left_time':left_time}
                
                not_present = True
        # frame = cv2.flip(frame,1)
        # farme = configurations['box_annotator'].annotate(frame, detections)
        # frame= configurations['label_annotator'].annotate(farme, detections)
        # cv2.imshow('hdh',frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == 13:
        #     
        
    cap.release()
    cv2.destroyAllWindows()
    # return dwell_times


def helmet_detection(is_running_flag,cap, configurations):
    '''
    classes: {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}

    '''

    while is_running_flag['is_running']:

        flag, frame = cap.read()
        if not flag:
            print ('Error reading frame')
            break

        result = configurations['hel_model'](frame, conf = 0.5, classes = [0,1,2,3,5])[0]
        detections = sv.Detections.from_ultralytics(result)
        track_detections = configurations['tracker'].update_with_detections(detections)

        



# cap = cv2.VideoCapture(0)
# model = YOLO('yolo_models/yolov8n.pt')
# tracker = sv.ByteTrack()


# conf_json = {
#     'model':model,
#     'tracker':tracker,
#     'device_id':0
# }

# find_dwell_time(cap, conf_json)
# print (dwells)
# count = vehicles_in_and_out(cap, conf_json)
# print ('count of vehicles',count)

# obj_detect = object_detection(cap, conf_json)
# print (len(obj_detect))