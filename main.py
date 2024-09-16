from fastapi import FastAPI, Form, Depends, status, HTTPException, Request, BackgroundTasks
from fastapi.responses import  HTMLResponse, RedirectResponse,JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from ultralytics import YOLO
from typing import Annotated
import supervision as sv
import workflows, cv2
import models
import os, sys, io
import json, time



app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.state.is_running_flag = {"is_running": False}

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# creates all tables defined in models
models.Base.metadata.create_all(bind = engine)
Devices = models.Devices


# load yolo models and tracker to detect and track objects.
model = YOLO('yolo_models/yolov8n.pt')
hel_model = YOLO('yolo_models/yolov8ft(hel).pt')
tracker = sv.ByteTrack()

 

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]


def generate_frames(is_running_flag,cap):
    try:
        while is_running_flag['is_running']:
            success, frame = cap.read()
            if not success:
                break
            else:
                
                frame = cv2.flip(frame, 1)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Return the frame in bytes
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print (f'Error opening cam:{e}')
    finally:
        if cap.isOpened():
            cap.release()
            print ("cam is released")
        cv2.destroyAllWindows()




@app.get("/models", response_class=HTMLResponse)
def list_models(request:Request):
    '''
    Endpoint to list all YOLO models in the 'yolo_models' directory.
    Reads model information and displays it using the 'list_models.html' template.
    '''

    models = os.listdir('yolo_models')
    models_json = {}
    for model in models:
        
        yolo_model = YOLO(f'yolo_models/{model}')
        model_info = yolo_model.info()
        layers, parameters, gradients, gflops = model_info
        
        model_json = {'layers':layers,
                 'parameters':parameters,
                 'gradients':gradients,
                 'gflops':gflops}
        models_json[model] = model_json

    # Render the template with model info
    return templates.TemplateResponse("list_models.html",{"request":request, "models_info":models_json})



@app.get("/devices/{user_id}", response_class=HTMLResponse)
def list_devices(request: Request, user_id: int, db: db_dependency):
    '''
    Endpoint to list devices configured by a specific user.
    Queries the devices table based on the user ID.
    '''
    devices = db.query(models.Devices).filter(models.Devices.user_id == user_id ).all()

    if devices is None:
        raise HTTPException(status_code=404, detail=f'No Devices found with user:{user_id}')
    
    return templates.TemplateResponse("list_devices.html", {"request":request, "devices":devices})



@app.get("/create_device", response_class=HTMLResponse)
def devices(request: Request):
    '''
    Endpoint to render the form for creating a new device.
    '''
    return templates.TemplateResponse("create_device.html", {"request":request})



@app.post('/configure',response_class=HTMLResponse)
def configure_new_device(db: db_dependency,
                         device_ip:str =Form(...),
                         location:str =Form(...),
                         user_id:int =Form(...)):
    '''
    Endpoint to handle form submission for configuring a new device.
    - Takes input from the form (device IP, location, and user ID).
    - Creates a new device entry in the database.
    - Commits the changes and redirects the user to their device list.
    '''
    device = models.Devices(device_ip= device_ip, location= location, user_id= user_id)
    db.add(device)
    db.commit()

    # Redirect to the devices list for the user
    return RedirectResponse(url=f'/devices/{user_id}',status_code=status.HTTP_303_SEE_OTHER)



@app.get("/update_device/{device_id}", response_class=HTMLResponse)
def fetch_device_data(device_id:int, db: db_dependency, request:Request):
    '''
    Endpoint to fetch data for a specific device to be updated.
    - Retrieves the device details from the database using the device ID.
    - Renders the "update_device.html" template with the current device information (device IP, location, user ID).
    '''
    device = db.query(Devices).filter(Devices.id == device_id).first()

    return templates.TemplateResponse("update_device.html",{
        "request":request,
        "device_ip":device.device_ip,
        "location":device.location,
        "user_id":device.user_id
    })



@app.post("/update",response_class=HTMLResponse)
def update_device(db: db_dependency, 
                  request:Request,
                  device_ip:str= Form(...),
                  location:str= Form(...),
                  user_id:int= Form(...)
                  ):
    '''
    Endpoint to update an existing device.
    - Takes the updated device details (IP, location, user ID) from the form.
    - Finds the device in the database using the device IP.
    - Updates the device's information and commits the changes.
    - Redirects the user to their device list.
    '''
    device = db.query(Devices).filter(Devices.device_ip == device_ip).first()
    
    device.device_ip = device_ip
    device.location = location
    device.user_id = user_id

    db.commit()

    return RedirectResponse(url=f'/devices/{user_id}',status_code=status.HTTP_303_SEE_OTHER)



@app.get("/delete_device/{device_id}", response_class=HTMLResponse)
def delete_device(db: db_dependency, device_id:int):
    '''
    Endpoint to delete an existing device.
    - Finds the device in the database using the device ID.
    - Deletes the device if found and commits the changes.
    - Raises a 404 error if the device does not exist.
    - Redirects the user to their device list after deletion.
    '''
    device = db.query(Devices).filter(Devices.id == device_id).first()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    db.delete(device)
    db.commit()

    return RedirectResponse(url=f'/devices/{device.user_id}', status_code=status.HTTP_303_SEE_OTHER)


@app.get("/select_device",response_class=HTMLResponse)
async def run_device(request:Request,db:db_dependency):
    '''
    Fetches a list of devices from the database and renders a template to allow the user 
    to select a device. If no devices are found, it raises an HTTP 404 error.
    '''
    devices = db.query(models.Devices).all()
    if not devices:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail='No Devices')
    
    return templates.TemplateResponse('select_device.html', {"request":request,"devices":devices})
    

@app.post("/start_stream",response_class=HTMLResponse)
async def stream_video(request:Request,db:db_dependency,device_id:int =Form(...)):
    '''
    Starts video streaming from the selected device. It retrieves the device by its ID from the database,
    then attempts to open a video stream from the device's IP address. 
    If it fails, it raises an HTTP 400 error.
    '''
    device = db.query(Devices).filter(Devices.id == device_id).first()
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No Devices Found')
    
    device_ip = device.device_ip
    if device_ip == '0':
        device_ip = 0
    
    global demo_cap
    demo_cap = cv2.VideoCapture(device_ip)
    
    if not demo_cap.isOpened():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail=f'Error in connecting to device:{device_ip}')
    

    return templates.TemplateResponse("video.html",{'request':request})

@app.get("/video_feed")
async def video_feed():
    '''
    Streams the video frames from the currently connected device.
    It continuously sends frames to the client in a streaming format.
    '''
    if not app.state.is_running_flag["is_running"]:
        app.state.is_running_flag["is_running"] = True


    return StreamingResponse(generate_frames(app.state.is_running_flag, demo_cap), media_type='multipart/x-mixed-replace; boundary=frame')



@app.post("/stop_stream")
async def stop_stream():
    '''
    Stops the video stream by setting a flag that controls the streaming process.
    Redirects the user back to the device selection page.
    '''
    app.state.is_running_flag['is_running'] = False
    
    return RedirectResponse(url='/select_device',status_code=status.HTTP_303_SEE_OTHER)


@app.get("/start_workflow/{device_id}",response_class=HTMLResponse)
async def start_workflow(device_id:int, request:Request):
    '''
    Endpoint to render the form for starting workflow.
    '''
    return templates.TemplateResponse('workflow.html', {"request":request, "device_id":device_id})


@app.post('/start_function')
async def start_function(request: Request,
                   db: db_dependency,
                   background_tasks: BackgroundTasks,
                   device_id: int= Form(...),
                   workflow: str= Form(...)):
    """
    Start the specified workflow on the selected device.
    
    - Checks if the function is already running by examining the app state.
    - Retrieves the device information based on `device_id`.
    - Opens the video stream from the device.
    - Initializes the start time of the process.
    - Based on the selected workflow, different tasks (e.g., vehicle_in_out, dwell_time) are run in the background.
    """

    if not app.state.is_running_flag["is_running"]:
        app.state.is_running_flag["is_running"] = True

    device = db.query(Devices).filter(Devices.id == device_id).first()
    if not device:
        raise HTTPException(status_code= status.HTTP_404_NOT_FOUND, detail=f'No Device found with id:{device_id}')
    
    device_ip = device.device_ip

    if device_ip == '0':
        device_ip = 0

    # Open the video capture from the device IP (camera)
    global cap
    cap = cv2.VideoCapture(device_ip)
    
    # Raise an error if the camera couldn't be opened
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail='Error opening camera')
    
    global device_start_time
    device_start_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    
    # Configuration data to be passed to the background task
    configurations = {
        'model':model,
        'hel_model':hel_model,
        'tracker':tracker,
        'device_id':device_id,
        'device_start_time':device_start_time
    }
    
    global selected_workflow
    selected_workflow = workflow
    if workflow == 'helmet_detection':
        background_tasks.add_task(workflows.helmet_detection,
                                  app.state.is_running_flag,
                                  cap,
                                  configurations)
    
    elif workflow == 'dwell_time':
        background_tasks.add_task(workflows.find_dwell_time, 
                                  app.state.is_running_flag,
                                  cap,
                                  configurations)
        
    else:
        # workflows.vehicles_in_or_out()
        pass

    # Render the 'workflow.html' template with the device ID and status
    return templates.TemplateResponse('workflow.html',{"request":request,"device_id":device_id,"status":True})
    


@app.post("/stop_function")
async def stop_function(request:Request, db:db_dependency):
    """
    Stops the running function by setting the flag to False.
    
    - Marks the process as stopped by setting `is_running` to False.
    - Updates the device end time.
    - Finds events that occurred during the device's running time and updates them with the end time.
    """
    app.state.is_running_flag["is_running"] = False
    
    global device_end_time
    device_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if selected_workflow == 'helmet_detection':
        # Query to find all events that match the current device start time
        current_events = db.query(models.HelmetDetection).filter(models.HelmetDetection.device_start_time == device_start_time).all()
        # Update the device end time for each event
        for current_event in current_events:
            current_event.device_end_time = device_end_time
            db.commit()

    elif selected_workflow == 'dwell_time':
        # Query to find all events that match the current device start time
        current_events = db.query(models.DwellTime).filter(models.DwellTime.device_start_time == device_start_time).all()
        # Update the device end time for each event
        for current_event in current_events:
            if current_event:
                current_event.device_end_time = device_end_time
                db.commit()
            
    # Redirect to the current logs page after stopping the function 
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()   
    return RedirectResponse(url='/current_logs', status_code=status.HTTP_303_SEE_OTHER)



@app.get('/current_logs',response_class=HTMLResponse)
async def find_current_logs(request:Request):
    """
    Retrieves and displays logs generated during the device's operation period.
    
    - Reads logs from the `yolo.log` file.
    - Filters logs based on the start and end times of the device's operation.
    """
    with open('yolo.log','r')as l:
        logs = l.readlines()

    # Find log entries within the device's operation period (between start and end time)
    indexes = []
    for index,log in enumerate(logs):
        ts = log.split(',')[0].split(' ')
        ts = f'{ts[0]} {ts[1]}'
        time_stamp = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        
        start_time = time.mktime(time.strptime(device_start_time,  "%Y-%m-%d %H:%M:%S"))
        end_time = time.mktime(time.strptime(device_end_time,  "%Y-%m-%d %H:%M:%S"))

        # Check if the log timestamp falls within the start and end times
        if start_time <= time_stamp and time_stamp <= end_time:
            indexes.append(index)

    current_logs = logs[indexes[0]:indexes[-1]]
    # Render the 'logs.html' template with the filtered logs
    return templates.TemplateResponse('logs.html',{"request":request,"logs":current_logs})



@app.get('/events/{workflow}',response_class=HTMLResponse)
async def list_events(request:Request, db:db_dependency, workflow:str):
    """
    Lists all events for a specified workflow type.
    
    - Queries the database to retrieve events related to the provided workflow.
    - Currently supports only the 'dwelltime' workflow.
    """
    # Query the database for events based on the workflow type
    if workflow == 'dwelltime':
        events = db.query(models.DwellTime).all()

    elif workflow == 'helmet':
        events = db.query(models.HelmetDetection).all()
    # Render the 'events.html' template with the retrieved events
    return templates.TemplateResponse('events.html',{"request":request,"events":events})



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)