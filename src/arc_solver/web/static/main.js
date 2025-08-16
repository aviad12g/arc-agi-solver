/**
 * ARC-AGI Solver Interactive UI - Main JavaScript
 *
 * This script handles all frontend logic for the web interface, including:
 * - Loading pre-loaded task data from the backend API.
 * - Handling user interactions (file uploads, task selection, starting the solver).
 * - Managing WebSocket communication for real-time updates from the solver.
 * - Rendering all visualizations (tasks, solver progress, final solution).
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('ARC-AGI Solver UI Initialized');

    // --- DOM Element References ---
    const taskFileInput = document.getElementById('task-file-input');
    const trainingTaskSelect = document.getElementById('training-task-select');
    const evaluationTaskSelect = document.getElementById('evaluation-task-select');
    const startSolverBtn = document.getElementById('start-solver-btn');
    const statusMessage = document.getElementById('status-message');
    const taskGridsContainer = document.getElementById('task-grids-container');
    const solverProgressContainer = document.getElementById('solver-progress-container');
    const finalSolutionContainer = document.getElementById('final-solution-container');

    // --- Application State ---
    let currentTask = null; // Holds the currently loaded task data
    const socket = io(); // Establishes WebSocket connection on page load
    let solutionSteps = []; // Stores steps of the final solution for step-through
    let currentStepIndex = -1; // Index for the step-through visualizer

    // --- Initial Setup ---
    loadTaskList();

    // --- WebSocket Event Handlers ---
    socket.on('connect', () => {
        console.log('Connected to WebSocket server with sid:', socket.id);
        statusMessage.textContent = 'Ready. Load a task.';
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from WebSocket server.');
        statusMessage.textContent = 'Disconnected. Please refresh the page.';
        setControlsEnabled(false);
    });

    socket.on('progress_update', (data) => {
        const stats = data.stats || {};
        statusMessage.textContent = `Solving... (Nodes: ${stats.nodes_expanded || 0}, Beam: ${stats.beam_width || 'N/A'})`;
        renderSolverProgress(data.candidates, stats);
    });

    socket.on('solver_finished', (data) => {
        statusMessage.textContent = `Solver finished! Reason: ${data.termination_reason}`;
        setControlsEnabled(true);
        renderFinalSolution(data);
    });

    socket.on('solver_error', (data) => {
        statusMessage.textContent = `Error: ${data.error}`;
        solverProgressContainer.innerHTML = `<p class="placeholder error">Solver encountered an error.</p>`;
        setControlsEnabled(true);
    });

    // --- DOM Event Listeners ---
    taskFileInput.addEventListener('change', handleFileSelect);
    trainingTaskSelect.addEventListener('change', (e) => handleTaskSelect(e, 'training'));
    evaluationTaskSelect.addEventListener('change', (e) => handleTaskSelect(e, 'evaluation'));
    startSolverBtn.addEventListener('click', startSolver);

    // --- Core Functions ---

    /**
     * Fetches the list of pre-loaded tasks from the backend and populates the dropdowns.
     */
    async function loadTaskList() {
        try {
            const response = await fetch('/api/tasks');
            if (!response.ok) throw new Error('Failed to fetch task list.');
            const tasks = await response.json();

            populateSelect(trainingTaskSelect, tasks.training, "training");
            populateSelect(evaluationTaskSelect, tasks.evaluation, "evaluation");

        } catch (error) {
            console.error('Error loading task list:', error);
            trainingTaskSelect.innerHTML = '<option>Error loading tasks</option>';
            evaluationTaskSelect.innerHTML = '<option>Error loading tasks</option>';
        }
    }

    /**
     * Fills a <select> element with a list of task names.
     * @param {HTMLElement} selectElement The <select> element to populate.
     * @param {string[]} taskList An array of task filenames.
     * @param {string} folder The folder name ('training' or 'evaluation').
     */
    function populateSelect(selectElement, taskList, folder) {
        selectElement.innerHTML = `<option value="">-- Select a ${folder} task --</option>`;
        taskList.forEach(taskName => {
            const option = document.createElement('option');
            option.value = taskName;
            option.textContent = taskName;
            selectElement.appendChild(option);
        });
    }

    /**
     * Handles the selection of a pre-loaded task from a dropdown.
     * @param {Event} event The change event from the <select> element.
     * @param {string} folder The folder the task belongs to ('training' or 'evaluation').
     */
    async function handleTaskSelect(event, folder) {
        const taskName = event.target.value;
        // Reset other inputs to avoid confusion
        if (folder === 'training') evaluationTaskSelect.value = '';
        else trainingTaskSelect.value = '';
        taskFileInput.value = '';

        if (!taskName) {
            resetTaskView();
            return;
        }

        statusMessage.textContent = `Loading task "${taskName}"...`;
        taskGridsContainer.innerHTML = `<p class="placeholder">Loading...</p>`;

        try {
            const response = await fetch(`/api/task/${folder}/${taskName}`);
            if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);
            const result = await response.json();
            if (result.error) throw new Error(result.error);

            loadTaskData(result);
        } catch (error) {
            console.error('Error loading pre-loaded task:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            resetTaskView();
        }
    }

    /**
     * Handles the selection of a local file.
     * @param {Event} event The change event from the <input type="file"> element.
     */
    async function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Reset other inputs
        trainingTaskSelect.value = '';
        evaluationTaskSelect.value = '';

        statusMessage.textContent = `Uploading "${file.name}"...`;
        startSolverBtn.disabled = true;
        taskGridsContainer.innerHTML = `<p class="placeholder">Loading...</p>`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);
            const result = await response.json();
            if (result.error) throw new Error(result.error);

            loadTaskData(result);
        } catch (error) {
            console.error('Error uploading file:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            resetTaskView();
        }
    }

    /**
     * Central function to process loaded task data, whether from file or pre-loaded.
     * @param {object} taskResult The task object from the backend.
     */
    function loadTaskData(taskResult) {
        currentTask = taskResult;
        // The backend needs the task data in a specific format for the solver.
        // We store it here to be sent when the solver starts.
        task_storage['latest_task'] = {id: taskResult.task_id, data: taskResult.task_data};

        statusMessage.textContent = `Task "${currentTask.task_id}" loaded. Ready to solve.`;
        startSolverBtn.disabled = false;
        renderTask(currentTask.task_data);
    }

    /**
     * Resets the main task view to its initial state.
     */
    function resetTaskView() {
        taskGridsContainer.innerHTML = '<p class="placeholder">Load a task to see the training and test grids.</p>';
        currentTask = null;
        startSolverBtn.disabled = true;
    }

    /**
     * Starts the solver process by emitting an event to the WebSocket server.
     */
    function startSolver() {
        if (!currentTask) {
            alert("Please load a task first!");
            return;
        }
        console.log('Starting solver for task:', currentTask.task_id);
        statusMessage.textContent = 'Solver started...';
        setControlsEnabled(false);

        solverProgressContainer.innerHTML = '<p class="placeholder">Solver is running...</p>';
        finalSolutionContainer.innerHTML = '<p class="placeholder">Awaiting final solution...</p>';

        socket.emit('start_solving', { task_id: currentTask.task_id });
    }

    /**
     * Helper to enable or disable all input controls.
     * @param {boolean} isEnabled
     */
    function setControlsEnabled(isEnabled) {
        startSolverBtn.disabled = !isEnabled;
        taskFileInput.disabled = !isEnabled;
        trainingTaskSelect.disabled = !isEnabled;
        evaluationTaskSelect.disabled = !isEnabled;
    }

    // --- Rendering Functions ---
    // ... (All rendering functions below are unchanged) ...
    function renderTask(taskData) {
        taskGridsContainer.innerHTML = '';
        taskData.train.forEach((pair, index) => {
            const pairEl = createGridPairElement(`Train Pair ${index + 1}`, pair.input, pair.output);
            taskGridsContainer.appendChild(pairEl);
        });
        taskData.test.forEach((pair, index) => {
            const pairEl = createGridPairElement(`Test Input ${index + 1}`, pair.input);
            taskGridsContainer.appendChild(pairEl);
        });
    }

    function renderSolverProgress(candidates, stats) {
        solverProgressContainer.innerHTML = '';
        if (!candidates || candidates.length === 0) {
            solverProgressContainer.innerHTML = '<p class="placeholder">Waiting for first candidates...</p>';
            return;
        }

        const statsEl = document.createElement('div');
        statsEl.className = 'progress-stats';
        statsEl.innerHTML = `
            <div>Nodes Expanded: <span>${stats.nodes_expanded}</span></div>
            <div>Queue Size: <span>${stats.queue_size}</span></div>
            <div>Beam Width: <span>${stats.beam_width}</span></div>
        `;
        solverProgressContainer.appendChild(statsEl);

        candidates.forEach(candidate => {
            const candidateEl = document.createElement('div');
            candidateEl.className = 'candidate-solution';

            const gridEl = renderGrid(candidate.grid);
            const infoEl = document.createElement('div');
            infoEl.className = 'candidate-info';
            infoEl.innerHTML = `
                <p><strong>Program:</strong> <code>${candidate.program || '[]'}</code></p>
                <p>f(n): ${candidate.f_score}, g(n): ${candidate.cost}, h(n): ${candidate.heuristic}</p>
            `;

            candidateEl.appendChild(gridEl);
            candidateEl.appendChild(infoEl);
            solverProgressContainer.appendChild(candidateEl);
        });
    }

    function renderFinalSolution(data) {
        finalSolutionContainer.innerHTML = ''; // Clear placeholder

        const title = document.createElement('h3');
        title.textContent = data.success ? '✅ Solution Found' : '❌ No Solution Found';
        finalSolutionContainer.appendChild(title);

        if (!data.success || !data.steps || data.steps.length === 0) {
            const reason = document.createElement('p');
            reason.textContent = `Reason: ${data.termination_reason}`;
            finalSolutionContainer.appendChild(reason);
            return;
        }

        solutionSteps = data.steps;
        currentStepIndex = 0;

        const displayEl = document.createElement('div');
        displayEl.id = 'solution-display';
        displayEl.innerHTML = `
            <div id="solution-program"></div>
            <div id="step-through-visualizer">
                <div id="step-through-grid-container"></div>
                <div id="step-through-controls">
                    <button id="prev-step-btn">&laquo; Prev</button>
                    <span id="step-counter"></span>
                    <button id="next-step-btn">Next &raquo;</button>
                </div>
            </div>
        `;
        finalSolutionContainer.appendChild(displayEl);

        document.getElementById('prev-step-btn').addEventListener('click', showPrevStep);
        document.getElementById('next-step-btn').addEventListener('click', showNextStep);

        updateStepThroughView();
    }

    function updateStepThroughView() {
        if (solutionSteps.length === 0) return;

        const programContainer = document.getElementById('solution-program');
        const gridContainer = document.getElementById('step-through-grid-container');
        const stepCounter = document.getElementById('step-counter');
        const prevBtn = document.getElementById('prev-step-btn');
        const nextBtn = document.getElementById('next-step-btn');

        programContainer.innerHTML = '';
        const ul = document.createElement('ul');
        solutionSteps.forEach((step, index) => {
            const li = document.createElement('li');
            li.textContent = `${index}: ${step.operation}`;
            if (index === currentStepIndex) {
                li.classList.add('active');
            }
            ul.appendChild(li);
        });
        programContainer.appendChild(ul);

        gridContainer.innerHTML = '';
        const currentGridData = solutionSteps[currentStepIndex].grid;
        gridContainer.appendChild(renderGrid(currentGridData));

        stepCounter.textContent = `Step ${currentStepIndex} / ${solutionSteps.length - 1}`;
        prevBtn.disabled = (currentStepIndex === 0);
        nextBtn.disabled = (currentStepIndex === solutionSteps.length - 1);
    }

    function showPrevStep() {
        if (currentStepIndex > 0) {
            currentStepIndex--;
            updateStepThroughView();
        }
    }

    function showNextStep() {
        if (currentStepIndex < solutionSteps.length - 1) {
            currentStepIndex++;
            updateStepThroughView();
        }
    }

    function createGridPairElement(title, inputGridData, outputGridData) {
        const pairEl = document.createElement('div');
        pairEl.className = 'task-pair';
        const titleEl = document.createElement('h3');
        titleEl.textContent = title;
        pairEl.appendChild(titleEl);
        const gridsWrapper = document.createElement('div');
        gridsWrapper.className = 'grids-wrapper';
        const inputGrid = renderGrid(inputGridData);
        gridsWrapper.appendChild(inputGrid);
        if (outputGridData) {
            const arrow = document.createElement('span');
            arrow.className = 'arrow';
            arrow.textContent = '→';
            gridsWrapper.appendChild(arrow);
            const outputGrid = renderGrid(outputGridData);
            gridsWrapper.appendChild(outputGrid);
        }
        pairEl.appendChild(gridsWrapper);
        return pairEl;
    }

    function renderGrid(gridData) {
        const gridEl = document.createElement('div');
        gridEl.className = 'grid';
        if (!gridData || gridData.length === 0) return gridEl;

        gridEl.style.gridTemplateColumns = `repeat(${gridData[0].length}, 1fr)`;

        gridData.forEach(row => {
            row.forEach(cellColor => {
                const cellEl = document.createElement('div');
                cellEl.className = `grid-cell color-${cellColor}`;
                cellEl.style.width = '20px';
                cellEl.style.height = '20px';
                gridEl.appendChild(cellEl);
            });
        });
        return gridEl;
    }
});
