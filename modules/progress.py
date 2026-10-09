import time

import string
import random

current_task = None


def start_task(id_task):
    global current_task

    current_task = id_task


def finish_task(id_task):
    global current_task

    if current_task == id_task:
        current_task = None

def create_task_id(task_type):
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase +
    string.digits, k=N))
    return f"task({task_type}-{res})"


def calculate_progress_and_eta(job_count, job_no, sampling_steps, sampling_step, time_start, *, base_progress=0):
    progress = base_progress

    if job_count > 0:
        progress += job_no / job_count
    if sampling_steps > 0 and job_count > 0:
        progress += 1 / job_count * sampling_step / sampling_steps

    progress = min(progress, 1)

    elapsed_since_start = time.time() - time_start
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    return progress, eta
