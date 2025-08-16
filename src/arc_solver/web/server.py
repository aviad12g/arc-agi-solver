"""
This module contains the backend server for the ARC-AGI Solver's interactive web UI.
It uses FastAPI for HTTP endpoints and Socket.IO for real-time WebSocket communication.
"""

import json
import socketio
import os
from pathlib import Path
import asyncio

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from arc_solver.cli.commands import ARCSolver

# --- Initial Setup ---

# Get the directory of the current file to build robust paths
BASE_DIR = Path(__file__).resolve().parent

# --- FastAPI App & Socket.IO Server Setup ---

app = FastAPI(title="ARC-AGI Solver Web UI")

# Mount the 'static' directory to serve CSS, JS, and other static assets
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Set up the Jinja2 templates for rendering the main HTML page
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Set up the Socket.IO asynchronous server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
# Wrap the FastAPI app with the Socket.IO server
socket_app = socketio.ASGIApp(sio, app)

# --- In-memory Storage ---
# A simple in-memory dictionary to store the most recently uploaded/selected task.
# For a multi-user or more robust system, this would be replaced with
# proper session management, a task queue, or a database.
task_storage = {}


# --- FastAPI HTTP Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index.html page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_task(file: UploadFile = File(...)):
    """Handles the upload of an ARC task JSON file from the user's computer."""
    try:
        contents = await file.read()
        task_data = json.loads(contents)
        if 'train' not in task_data or 'test' not in task_data:
            raise HTTPException(status_code=400, detail="Invalid ARC task file. 'train' and 'test' keys are required.")

        task_id = file.filename
        task_storage['latest_task'] = {"id": task_id, "data": task_data}
        return {"task_id": task_id, "task_data": task_data}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Upload is not a valid JSON file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file upload: {str(e)}")

@app.get("/api/tasks")
async def get_task_list():
    """Scans the data directory and returns a list of available pre-loaded tasks."""
    data_dir = Path.cwd() / 'data'
    tasks = {'training': [], 'evaluation': []}
    try:
        training_dir = data_dir / 'training'
        if training_dir.is_dir():
            training_files = os.listdir(training_dir)
            tasks['training'] = sorted([f for f in training_files if f.endswith('.json')])

        evaluation_dir = data_dir / 'evaluation'
        if evaluation_dir.is_dir():
            evaluation_files = os.listdir(evaluation_dir)
            tasks['evaluation'] = sorted([f for f in evaluation_files if f.endswith('.json')])

        return tasks
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data directory not found. Make sure the ARC-AGI-2 dataset is in the 'data' folder.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read tasks: {e}")

@app.get("/api/task/{folder}/{task_name}")
async def get_task_data(folder: str, task_name: str):
    """Reads and returns the content of a specific pre-loaded task JSON file."""
    if folder not in ['training', 'evaluation']:
        raise HTTPException(status_code=400, detail="Invalid folder specified. Must be 'training' or 'evaluation'.")

    task_path = Path.cwd() / 'data' / folder / task_name
    if not task_path.is_file():
        raise HTTPException(status_code=404, detail=f"Task file not found: {task_path}")

    try:
        with open(task_path, 'r') as f:
            task_data = json.load(f)
        return {"task_id": task_name, "task_data": task_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read or parse task file: {e}")


# --- Socket.IO Event Handlers ---

@sio.event
async def connect(sid, environ):
    """Handles a new client connection."""
    print(f"Socket.IO client connected: {sid}")

@sio.event
async def disconnect(sid):
    """Handles a client disconnection."""
    print(f"Socket.IO client disconnected: {sid}")

@sio.on('start_solving')
async def start_solving(sid, data):
    """
    Event handler to start the ARC solving process for the currently loaded task.
    This runs the solver in a background thread to avoid blocking the server's event loop.
    """
    print(f"Received start_solving event from {sid}")
    if 'latest_task' not in task_storage:
        await sio.emit('solver_error', {'error': 'No task loaded. Please upload or select a task first.'}, to=sid)
        return

    task_info = task_storage['latest_task']

    # This callback function is passed down into the solver. It allows the synchronous
    # solver code to send asynchronous WebSocket messages back to the client.
    async def update_callback(event_type, data):
        await sio.emit(event_type, data, to=sid)

    try:
        # Run the synchronous, CPU-bound solver in a separate thread.
        await sio.start_background_task(
            run_solver_and_get_results,
            task_info,
            update_callback
        )
    except Exception as e:
        print(f"Error starting solver background task: {e}")
        await sio.emit('solver_error', {'error': f'Failed to start solver: {e}'}, to=sid)

def run_solver_and_get_results(task_info: dict, callback):
    """
    A synchronous wrapper function to run the solver. This is designed to be
    executed in a background thread by Socket.IO's `start_background_task`.
    """
    try:
        # The solver's `solve_task` method expects a mock object with specific attributes
        # rather than a raw dictionary. We create that mock object here.
        class MockTask:
            def __init__(self, task_data_dict):
                self.train = [{'input': p['input'], 'output': p['output']} for p in task_data_dict['train']]
                self.test = [{'input': p['input']} for p in task_data_dict['test']]
                self.task_id = task_info['id']

        mock_task = MockTask(task_info['data'])

        # Initialize the solver. Configuration is loaded from files by default.
        solver = ARCSolver()

        # Run the solver, passing the callback function for real-time updates.
        solver.solve_task(mock_task, update_callback=callback)

    except Exception as e:
        print(f"An error occurred in the solver thread: {e}")
        # To call the async callback from this sync thread, we need to run it in a new event loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'get_running_loop' fails if no loop is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(callback('solver_error', {'error': str(e)}))
