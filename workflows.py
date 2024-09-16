import cv2
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
org = "omniwyse"
url = "http://localhost:8086"
bucket="hongsaw"

try:
    client = influxdb_client.InfluxDBClient(url=url, token=api_token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
except Exception as e:
    print ('Error establishing connection with Influxdb',e)



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
       
    # cap.release()
    # cv2.destroyAllWindows()
    # return dwell_times


def helmet_detection(is_running_flag,cap, configurations):
    '''
    classes: {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}
    '''
    no_helmet_detections = {}
    data = ['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']
    while is_running_flag['is_running']:

        flag, frame = cap.read()
        if not flag:
            print ('Error reading frame')
            break

        result = configurations['hel_model'](frame, conf= 0.65, classes = [0,2,5])[0]
        detections = sv.Detections.from_ultralytics(result)
        track_detections = configurations['tracker'].update_with_detections(detections)

        boxes = track_detections.xyxy.tolist()

        if not boxes:
            print ('No Detections')
            logging.warn('No Detections')
        else:
            classes = track_detections.data['class_name'].tolist()

            logging.info(classes)
            track_ids = track_detections.tracker_id.tolist()

            p_indexes = [i for i in range(len(classes)) if classes[i]=='Person']
            nh_indexes = [i for i in range(len(classes)) if classes[i]=='NO-Hardhat']

            for nh_index in nh_indexes:
                hx1,hy1,hx2,hy2 = boxes[nh_index]

                for p_index in p_indexes:
                    px1,py1,px2,py2 = boxes[p_index]

                    if abs(hy1 - py1) <30 and track_ids[p_index] not in no_helmet_detections:
                        no_helmet_detections[track_ids[p_index]] = True
                        detected_time = time.strftime("%y-%m-%d %H:%M:%S", time.localtime())

                        point = Point('Helmet_Detection') \
                        .tag("helmet_status",'NO-Helmet') \
                        .tag("detected_time",detected_time) \
                        .field("obj_id",data.index('Person'))

                        write_api.write(bucket=bucket, org= org, record=point)

                        helmet_event = models.HelmetDetection(
                            obj_id = data.index('Person'),
                            helmet_status = 'NO-Helmet',
                            device_start_time = configurations['device_start_time'],
                            device_id = configurations['device_id']
                        )
                        session.add(helmet_event)
                        session.commit()

                        logging.critical('Helmet Event Created')



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