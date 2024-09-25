from fastapi import (
    FastAPI,
    Form,
    Depends,
    status,
    HTTPException,
    Request,
    BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from datetime import datetime
from .database import engine, SessionLocal
from sqlalchemy.orm import Session
from ultralytics import YOLO
from typing import Annotated, List
import supervision as sv
import cv2
from .workflows import find_dwell_time, helmet_detection
from .models import Devices, DwellTime, Base, HelmetDetection, Workflow, ModelRuntime
import os, sys, io
import json, time

global model, hel_model, tracker

app = FastAPI()
templates = Jinja2Templates(directory="/app/app/templates")
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
Base.metadata.create_all(bind=engine)
Devices = Devices


# load yolo models and tracker to detect and track objects.
model = YOLO("/app/app/yolo_models/yolov8n.pt")
hel_model = YOLO("/app/app/yolo_models/yolov8ft(hel).pt")
tracker = sv.ByteTrack()


# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]


def generate_frames(is_running_flag, cap):
    try:
        while is_running_flag["is_running"]:
            success, frame = cap.read()
            if not success:
                break
            else:
                frame = cv2.flip(frame, 1)
                ret, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()

                # Return the frame in bytes
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
                )
    except Exception as e:
        print(f"Error opening cam:{e}")
    finally:
        if cap.isOpened():
            cap.release()
            print("cam is released")
        cv2.destroyAllWindows()


@app.get("/models", response_class=JSONResponse)
def list_models(request: Request):
    """
    Endpoint to list all YOLO models in the 'yolo_models' directory.
    Reads model information and displays it using the 'list_models.html' template.
    """

    models = os.listdir("/app/app/yolo_models/")
    models_json = {}
    for model in models:
        yolo_model = YOLO(f"/app/app/yolo_models/{model}")
        model_info = yolo_model.info()
        layers, parameters, gradients, gflops = model_info

        model_json = {
            "layers": layers,
            "parameters": parameters,
            "gradients": gradients,
            "gflops": gflops,
        }
        models_json[model] = model_json

    # Render the template with model info
    return JSONResponse(content={"models_info": models_json})
    # return templates.TemplateResponse("list_models.html",{"request":request, "models_info":models_json})


@app.get("/devices/{user_id}", response_class=HTMLResponse)
def list_devices(request: Request, user_id: int, db: db_dependency):
    """
    Endpoint to list devices configured by a specific user.
    Queries the devices table based on the user ID.
    """
    devices = db.query(Devices).filter(Devices.user_id == user_id).all()

    if devices is None:
        raise HTTPException(
            status_code=404, detail=f"No Devices found with user:{user_id}"
        )

    return templates.TemplateResponse(
        "list_devices.html", {"request": request, "devices": devices}
    )


@app.get("/user_devices/{user_id}", response_class=JSONResponse)
def list_user_devices(request: Request, user_id: int, db: db_dependency):
    devices = db.query(Devices).filter(Devices.user_id == user_id).all()

    devices_json = {}

    for device in devices:
        devices_json[device.id] = {
            "id": device.id,
            "device_ip": device.device_ip,
            "location": device.location,
            "created_on": device.created_on.isoformat(),
            "user_id": device.user_id,
        }

    return JSONResponse(content={"devices": devices_json})


@app.post("/create_device", response_class=JSONResponse)
def create_device(
    db: db_dependency,
    request: Request,
    device_ip: str = Form(...),
    location: str = Form(...),
    user_id: int = Form(...),
):
    """
    Endpoint to create a new device in the database.
    """
    # Create a new device instance
    new_device = Devices(
        device_ip=device_ip,
        location=location,
        created_on=datetime.utcnow(),
        user_id=user_id
    )

    # Add the new device to the database
    db.add(new_device)
    db.commit()
    db.refresh(new_device)

    # Return a success message or the created device data
    return JSONResponse(content={"message": "Device created successfully"}, status_code=200)

@app.post("/configure", response_class=HTMLResponse)
def configure_new_device(
    db: db_dependency,
    device_ip: str = Form(...),
    location: str = Form(...),
    user_id: int = Form(...),
):
    """
    Endpoint to handle form submission for configuring a new device.
    - Takes input from the form (device IP, location, and user ID).
    - Creates a new device entry in the database.
    - Commits the changes and redirects the user to their device list.
    """
    device = Devices(device_ip=device_ip, location=location, user_id=user_id)
    db.add(device)
    db.commit()

    # Redirect to the devices list for the user
    return RedirectResponse(
        url=f"/devices/{user_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.get("/update_device/{device_id}", response_class=HTMLResponse)
def fetch_device_data(device_id: int, db: db_dependency, request: Request):
    """
    Endpoint to fetch data for a specific device to be updated.
    - Retrieves the device details from the database using the device ID.
    - Renders the "update_device.html" template with the current device information (device IP, location, user ID).
    """
    device = db.query(Devices).filter(Devices.id == device_id).first()

    return templates.TemplateResponse(
        "update_device.html",
        {
            "request": request,
            "device_ip": device.device_ip,
            "location": device.location,
            "user_id": device.user_id,
        },
    )


@app.post("/update", response_class=HTMLResponse)
def update_device(
    db: db_dependency,
    request: Request,
    device_ip: str = Form(...),
    location: str = Form(...),
    user_id: int = Form(...),
):
    """
    Endpoint to update an existing device.
    - Takes the updated device details (IP, location, user ID) from the form.
    - Finds the device in the database using the device IP.
    - Updates the device's information and commits the changes.
    - Redirects the user to their device list.
    """
    device = db.query(Devices).filter(Devices.device_ip == device_ip).first()

    device.device_ip = device_ip
    device.location = location
    device.user_id = user_id

    db.commit()

    return RedirectResponse(
        url=f"/devices/{user_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.get("/delete_device/{device_id}", response_class=HTMLResponse)
def delete_device(db: db_dependency, device_id: int):
    """
    Endpoint to delete an existing device.
    - Finds the device in the database using the device ID.
    - Deletes the device if found and commits the changes.
    - Raises a 404 error if the device does not exist.
    - Redirects the user to their device list after deletion.
    """
    device = db.query(Devices).filter(Devices.id == device_id).first()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    db.delete(device)
    db.commit()

    return RedirectResponse(
        url=f"/devices/{device.user_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.get("/select_device", response_class=HTMLResponse)
async def run_device(request: Request, db: db_dependency):
    devices = db.query(Devices).all()
    """
    Fetches a list of devices from the database and renders a template to allow the user 
    to select a device. If no devices are found, it raises an HTTP 404 error.
    """
    devices = db.query(Devices).all()
    if not devices:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No Devices")

    return templates.TemplateResponse(
        "select_device.html", {"request": request, "devices": devices}
    )


@app.post("/start_stream", response_class=JSONResponse)
async def stream_video(request: Request, db: db_dependency, device_id: int = Form(...)):
    """
    Starts video streaming from the selected device. It retrieves the device by its ID from the database,
    then attempts to open a video stream from the device's IP address.
    If it fails, it raises an HTTP 400 error.
    """
    device = db.query(Devices).filter(Devices.id == device_id).first()
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No Devices Found"
        )

    device_ip = device.device_ip
    if device_ip == "0":
        device_ip = 0

    global demo_cap
    demo_cap = cv2.VideoCapture(device_ip)

    if not demo_cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error in connecting to device:{device_ip}",
        )

    return JSONResponse(content = {"message": "Stream started successfully"}, status_code=200)


@app.get("/video_feed")
async def video_feed():
    """
    Streams the video frames from the currently connected device.
    It continuously sends frames to the client in a streaming format.
    """
    if not app.state.is_running_flag["is_running"]:
        app.state.is_running_flag["is_running"] = True

    return StreamingResponse(
        generate_frames(app.state.is_running_flag, demo_cap),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/stop_stream")
async def stop_stream():
    """
    Stops the video stream by setting a flag that controls the streaming process.
    Redirects the user back to the device selection page.
    """
    app.state.is_running_flag["is_running"] = False

    return RedirectResponse(url="/select_device", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/start_workflow/{device_id}", response_class=HTMLResponse)
async def start_workflow(device_id: int, request: Request):
    """
    Endpoint to render the form for starting workflow.
    """
    return templates.TemplateResponse(
        "workflow.html", {"request": request, "device_id": device_id}
    )


@app.post("/start_function")
async def start_function(
    request: Request,
    db: db_dependency,
    background_tasks: BackgroundTasks,
    device_id: int = Form(...),
    workflow: str = Form(...),
):
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No Device found with id:{device_id}",
        )

    device_ip = device.device_ip

    if device_ip == "0":
        device_ip = 0

    # Open the video capture from the device IP (camera)
    global cap
    cap = cv2.VideoCapture(device_ip)

    # Raise an error if the camera couldn't be opened
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Error opening camera")

    global device_start_time
    device_start_time = datetime.now()
    # device_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Configuration data to be passed to the background task
    configurations = {
        "model": model,
        "hel_model": hel_model,
        "tracker": tracker,
        "device_id": device_id,
        "device_start_time": device_start_time,
    }

    global selected_workflow
    selected_workflow = workflow
    if workflow == "helmet_detection":
        background_tasks.add_task(
            helmet_detection, app.state.is_running_flag, cap, configurations
        )

    elif workflow == "dwell_time":
        background_tasks.add_task(
            find_dwell_time, app.state.is_running_flag, cap, configurations
        )

    else:
        # workflows.vehicles_in_or_out()
        pass

    # Render the 'workflow.html' template with the device ID and status
    return templates.TemplateResponse(
        "workflow.html", {"request": request, "device_id": device_id, "status": True}
    )


@app.post("/stop_function")
async def stop_function(request: Request, db: db_dependency):
    """
    Stops the running function by setting the flag to False.

    - Marks the process as stopped by setting `is_running` to False.
    - Updates the device end time.
    - Finds events that occurred during the device's running time and updates them with the end time.
    """
    app.state.is_running_flag["is_running"] = False

    global device_end_time
    device_end_time = datetime.now()
    # device_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if selected_workflow == "helmet_detection":
        # Query to find all events that match the current device start time
        current_events = (
            db.query(HelmetDetection)
            .filter(HelmetDetection.device_start_time == device_start_time)
            .all()
        )
        # Update the device end time for each event
        for current_event in current_events:
            current_event.device_end_time = device_end_time
            db.commit()

    elif selected_workflow == "dwell_time":
        # Query to find all events that match the current device start time
        current_events = (
            db.query(DwellTime)
            .filter(DwellTime.device_start_time == device_start_time)
            .all()
        )
        # Update the device end time for each event
        for current_event in current_events:
            if current_event:
                current_event.device_end_time = device_end_time
                db.commit()

    # Redirect to the current logs page after stopping the function
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    return RedirectResponse(url="/current_logs", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/current_logs", response_class=HTMLResponse)
async def find_current_logs(request: Request):
    """
    Retrieves and displays logs generated during the device's operation period.

    - Reads logs from the `yolo.log` file.
    - Filters logs based on the start and end times of the device's operation.
    """
    with open("yolo.log", "r") as l:
        logs = l.readlines()

    # Find log entries within the device's operation period (between start and end time)
    indexes = []
    for index, log in enumerate(logs):
        ts = log.split(",")[0].split(" ")
        ts = f"{ts[0]} {ts[1]}"
        time_stamp = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))

        start_time = time.mktime(time.strptime(device_start_time, "%Y-%m-%d %H:%M:%S"))
        end_time = time.mktime(time.strptime(device_end_time, "%Y-%m-%d %H:%M:%S"))

        # Check if the log timestamp falls within the start and end times
        if start_time <= time_stamp and time_stamp <= end_time:
            indexes.append(index)

    current_logs = logs[indexes[0] : indexes[-1]]
    # Render the 'logs.html' template with the filtered logs
    return templates.TemplateResponse(
        "logs.html", {"request": request, "logs": current_logs}
    )


@app.get("/events/{workflow}", response_class=HTMLResponse)
async def list_events(request: Request, db: db_dependency, workflow: str):
    """
    Lists all events for a specified workflow type.

    - Queries the database to retrieve events related to the provided workflow.
    - Currently supports only the 'dwelltime' workflow.
    """
    # Query the database for events based on the workflow type
    if workflow == "dwelltime":
        events = db.query(DwellTime).all()

    elif workflow == "helmet":
        events = db.query(HelmetDetection).all()
    # Render the 'events.html' template with the retrieved events
    return templates.TemplateResponse(
        "events.html", {"request": request, "events": events}
    )


# Pydantic model for Workflow response
class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: str
    version: str
    status: bool
    created_on: datetime
    last_updated_on: datetime

    class Config:
        orm_mode = True

@app.post("/create_workflow", response_class=JSONResponse)
def create_workflow(
    db: db_dependency,
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    version: str = Form(...),
    status: bool = Form(...),
):
    """
    Endpoint to create a new workflow in the database.
    """
    new_workflow = Workflow(
        name=name,
        description=description,
        version=version,
        status=status,
        created_on=datetime.utcnow(),
        last_updated_on=datetime.utcnow()
    )

    db.add(new_workflow)
    db.commit()
    db.refresh(new_workflow)

    return JSONResponse(content={"message": "Workflow created successfully", "id": new_workflow.id}, status_code=201)

@app.put("/update_workflow/{workflow_id}", response_class=JSONResponse)
def update_workflow(
    workflow_id: int,
    db: db_dependency,
    request: Request,
    name: str = Form(None),
    description: str = Form(None),
    version: str = Form(None),
    status: bool = Form(None),
):
    """
    Endpoint to update an existing workflow in the database.
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if name is not None:
        workflow.name = name
    if description is not None:
        workflow.description = description
    if version is not None:
        workflow.version = version
    if status is not None:
        workflow.status = status

    workflow.last_updated_on = datetime.utcnow()

    db.commit()
    db.refresh(workflow)

    return JSONResponse(content={"message": "Workflow updated successfully"}, status_code=200)

@app.delete("/delete_workflow/{workflow_id}", response_class=JSONResponse)
def delete_workflow(
    workflow_id: int,
    db: db_dependency,
    request: Request,
):
    """
    Endpoint to delete a workflow from the database.
    """
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    db.delete(workflow)
    db.commit()

    return JSONResponse(content={"message": "Workflow deleted successfully"}, status_code=200)

@app.get("/get_all_workflows", response_model=List[WorkflowResponse])
def get_all_workflows(
    db: db_dependency,
    request: Request,
):
    """
    Endpoint to retrieve all workflows from the database.
    """
    workflows = db.query(Workflow).all()
    if len(workflows) == 0:
        workflow1 = Workflow(
            name="helmet detection",
            description="Detects helmets in CCTV footage from construction sites",
            version="1.0",
            status=True,
            created_on=datetime.utcnow(),
            last_updated_on=datetime.utcnow()
        )
        
        workflow2 = Workflow(
            name="dwell time",
            description="Detects the duration an object is present in a camera feed",
            version="1.0",
            status=True,
            created_on=datetime.utcnow(),
            last_updated_on=datetime.utcnow()
        )
        
        # Add the workflows to the session and commit
        db.add(workflow1)
        db.add(workflow2)
        db.commit()

    workflows = db.query(Workflow).all()
    return workflows

class ModelRuntimeCreate(BaseModel):
    device_id: int
    workflow_id: int
    status: str

class ModelRuntimeResponse(BaseModel):
    id: int
    device_id: int
    workflow_id: int
    start_time: datetime
    last_stop_time: datetime = None
    status: str

    class Config:
        orm_mode = True

@app.post("/create_model_runtime", response_model=ModelRuntimeResponse)
def create_model_runtime(
    db: Session = Depends(get_db),
    device_id: int = Form(...),
    workflow_id: int = Form(...),
    status: str = Form(...)
):
    """
    Endpoint to create a new model runtime in the database.
    """
    new_runtime = ModelRuntime(
        device_id=device_id,
        workflow_id=workflow_id,
        start_time=datetime.utcnow(),
        status=status
    )

    db.add(new_runtime)
    db.commit()
    db.refresh(new_runtime)

    return new_runtime

@app.delete("/delete_model_runtime/{runtime_id}", response_model=dict)
def delete_model_runtime(
    runtime_id: int,
    db: Session = Depends(get_db),
):
    """
    Endpoint to delete a model runtime from the database.
    """
    runtime = db.query(ModelRuntime).filter(ModelRuntime.id == runtime_id).first()
    if not runtime:
        raise HTTPException(status_code=404, detail="Model runtime not found")

    db.delete(runtime)
    db.commit()

    return {"message": "Model runtime deleted successfully"}

@app.get("/get_all_model_runtimes", response_model=List[ModelRuntimeResponse])
def get_all_model_runtimes(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    Endpoint to get all model runtimes from the database.
    """
    runtimes = db.query(ModelRuntime).offset(skip).limit(limit).all()
    return runtimes

@app.post("/start_runtime/{runtime_id}")
async def start_runtime(
    runtime_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start the specified runtime.
    """
    runtime = db.query(ModelRuntime).filter(ModelRuntime.id == runtime_id).first()
    if not runtime:
        raise HTTPException(status_code=404, detail=f"Runtime with id {runtime_id} not found")

    device = db.query(Devices).filter(Devices.id == runtime.device_id).first()
    workflow = db.query(Workflow).filter(Workflow.id == runtime.workflow_id).first()

    if not device or not workflow:
        raise HTTPException(status_code=404, detail="Associated device or workflow not found")

    if runtime.status == "Running":
        raise HTTPException(status_code=400, detail="Runtime is already running")

    try:
        device_ip = device.device_ip if device.device_ip != "0" else 0
        global capture
        capture = cv2.VideoCapture(device_ip)

        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Error opening camera")

        runtime.start_time = datetime.utcnow()
        runtime.status = "Running"

        configurations = {
            "model": model,  # Assuming these are globally defined
            "hel_model": hel_model,
            "tracker": tracker,
            "device_id": device.id,
            "device_start_time": runtime.start_time,
        }

        if workflow.name == "helmet detection":
            background_tasks.add_task(
                helmet_detection, {"is_running": True}, capture, configurations
            )
        elif workflow.name == "dwell time":
            background_tasks.add_task(
                find_dwell_time, {"is_running": True}, capture, configurations
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported workflow")

        db.commit()
        return {"message": "Runtime started successfully", "status": "Running"}

    except Exception as e:
        runtime.status = "Failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Runtime failed to start: {str(e)}")

@app.post("/stop_runtime/{runtime_id}")
async def stop_runtime(
    runtime_id: int,
    db: Session = Depends(get_db)
):
    """
    Stop the specified runtime.
    """
    runtime = db.query(ModelRuntime).filter(ModelRuntime.id == runtime_id).first()
    if not runtime:
        raise HTTPException(status_code=404, detail=f"Runtime with id {runtime_id} not found")

    if runtime.status != "Running":
        raise HTTPException(status_code=400, detail="Runtime is not currently running")

    try:
        runtime.last_stop_time = datetime.utcnow()
        runtime.status = "Stopped"

        workflow = db.query(Workflow).filter(Workflow.id == runtime.workflow_id).first()

        if workflow.name == "helmet detection":
            current_events = (
                db.query(HelmetDetection)
                .filter(HelmetDetection.device_start_time == runtime.start_time)
                .all()
            )
            for event in current_events:
                event.device_end_time = runtime.last_stop_time

        elif workflow.name == "dwell time":
            current_events = (
                db.query(DwellTime)
                .filter(DwellTime.device_start_time == runtime.start_time)
                .all()
            )
            for event in current_events:
                event.device_end_time = runtime.last_stop_time

        db.commit()

        if capture.isOpened():
            capture.release()
        cv2.destroyAllWindows()

        return {"message": "Runtime stopped successfully", "status": "Stopped"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop runtime: {str(e)}")

@app.get("/get_runtime_logs/{runtime_id}")
async def get_runtime_logs(runtime_id: int, db: Session = Depends(get_db)):
    """
    Retrieves logs for a specific runtime.

    - Queries the database for the runtime's start and end times.
    - Reads logs from the 'yolo.log' file.
    - Filters logs based on the start and end times of the runtime.
    - Returns the filtered logs as JSON.
    """
    # Query the database for the runtime
    runtime = db.query(ModelRuntime).filter(ModelRuntime.id == runtime_id).first()
    if not runtime:
        raise HTTPException(status_code=404, detail=f"Runtime with id {runtime_id} not found")

    # Read the log file
    try:
        with open("yolo.log", "r") as log_file:
            logs = log_file.readlines()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found")

    # Convert runtime start and end times to timestamps
    start_time = time.mktime(runtime.start_time.timetuple())
    end_time = time.mktime(runtime.last_stop_time.timetuple()) if runtime.last_stop_time else time.time()

    # Filter logs based on runtime start and end times
    filtered_logs = []
    for log in logs:
        try:
            log_timestamp_str = log.split(",")[0]
            log_timestamp = time.mktime(datetime.strptime(log_timestamp_str, "%Y-%m-%d %H:%M:%S").timetuple())
            if start_time <= log_timestamp <= end_time:
                filtered_logs.append(log.strip())
        except (ValueError, IndexError):
            # Skip malformed log entries
            continue

    return {"runtime_id": runtime_id, "logs": filtered_logs}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
