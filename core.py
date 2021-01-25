#!/usr/bin/env python3

import multiprocessing as mp
import re
import time
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

from models import LoginResponse, VideoStreamer
from syno_integration import sss_alert, sss_login

'''
Globals
'''
SYNO_ADDRESS = None
SYNO_SESSION = None
JOBS = []

def annotate_image(frame: np.ndarray, results: np.ndarray,
                   labels: dict) -> np.ndarray:
    '''
    Draws boxes on the image and displays it.
    '''
    height, width, _ = frame.shape
    for obj in results:
        if obj['class_id'] != 0:
            continue

        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = max(1, int(xmin * width))
        xmax = min(width, int(xmax * width))
        ymin = max(1, int(ymin * height))
        ymax = min(height, int(ymax * height))

        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                              (50, 255, 0), 2)

        label = labels[obj['class_id']] + ': ' + \
                      str(round(obj['score']*100, 1)) + '%'
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               0.5, 1)
        label_ymin = max(ymin, label_size[1] + 10)
        # Draw white box to put label text in
        frame = cv2.rectangle(frame,
                              (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + baseline - 10),
                              (255, 255, 255), cv2.FILLED)
        # Draw text
        frame = cv2.putText(frame, label, (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


def person_detected(results: np.ndarray) -> bool:
    for obj in results:
        if obj['class_id'] == 0:
            return True


def load_labels(filepath: str):
    ''' Load the labels file w/ or w/o index numbers'''
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()

    return labels


def set_input_tensor(interp: Interpreter, image: Image.Image) -> None:
    '''Sets the input tensor'''
    tensor_index = interp.get_input_details()[0]['index']
    input_tensor = interp.tensor(tensor_index)()[0]
    input_tensor [:, :] = image


def get_output_tensor(interp: Interpreter, index: int) -> np.ndarray:
    '''Returns the output tensor at the given index'''
    output_details = interp.get_output_details()[index]
    tensor = np.squeeze(interp.get_tensor(output_details['index']))
    return tensor


def detect_objects(interp: Interpreter, image: np.ndarray,
                   threshold: float) -> List[dict]:
    '''Returns a list of detection results, each a dict of object info'''
    input_data = np.expand_dims(image, axis=0)
    set_input_tensor(interp, input_data)
    interp.invoke()

    # Get output details
    boxes = get_output_tensor(interp, 0)
    classes = get_output_tensor(interp, 1)
    scores = get_output_tensor(interp, 2)
    count = int(get_output_tensor(interp, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)

    return results


def do_offline_test(config, args: Dict) -> None:
    # TODO FIX THIS WHOLE METHOD
    labels = load_labels(config.labels)
    interp = Interpreter(model_path=model_file)
    interp.allocate_tensors()
    _, input_height, input_width, _ = interp.get_input_details()[0]['shape']

    paths = get_image_paths()
    print('PATHS:', paths)
    if len(paths) == 0:
        print("No images found. Goodbye.")

    print("Found images. Testing...")
    timings = {}
    for path in paths:
        print("Opening %s" % (path.parts[-1]))
        stime = time.time()
        frame = cv2.imread(path.as_posix())
        r_frame = cv2.resize(frame, (input_width, input_height), Image.ANTIALIAS)

        # infer
        results = detect_objects(interp, r_frame, config.threshold)
        print('fs[0:2]', frame.shape[0:2])
        annotated = annotate_image(frame, results, labels)
        if args.save_detections:
            cv2.imwrite("detect_%s" % path.parts[-1], annotated)
        if args.show_detections:
            cv2.imshow('detection', annotated)

        timings[path.parts[-1]] = time.time() - stime

    print("Results:")
    pprint.pprint(timings, width=32, depth=32, compact=False)
    print("Done running. Goodbye.")


def monitor_stream(config: dict, cam_config: dict, run_process: mp.Event,
                   cli_args: Dict) -> None:

    SYNO_ADDRESS = config['synology']['address']
    SYNO_ACCOUNT = config['synology']['account']
    SYNO_PASSWORD = config['synology']['password']

    SHOW_VIDEO = config['general']['show_video']
    SAVE_DETECTIONS = config['general']['save_detections']
    BUFFER_SIZE = config['general']['buffer_size']
    THRESHOLD_ALL = config['general']['threshold_all']
    CAM_THRESHOLD = cam_config['threshold']

    if cli_args.threshold_all:
        threshold = THRESHOLD_ALL
    else:
        threshold = CAM_THRESHOLD

    MODEL_PATH = config['model']['path']
    LABELS_PATH = config['model']['labels']

    print("SYNO_ADDRESS", SYNO_ADDRESS)
    print("SYNO_ACCOUNT", SYNO_ACCOUNT)
    print("SYNO_PASSWORD", ''.join(['*' for _ in range(len(SYNO_PASSWORD))]))

    print("SHOW_VIDEO", SHOW_VIDEO)
    print("SAVE_DETECTIONS", SAVE_DETECTIONS)
    print("BUFFER_SIZE", BUFFER_SIZE)
    print("threshold", threshold)

    CAM_ID = cam_config['syno_cam_id']

    SYNO_SESSION = sss_login(SYNO_ADDRESS, SYNO_ACCOUNT, SYNO_PASSWORD)

    # Load model and labels, allocate memory for tensors, and get input shape
    interp = Interpreter(model_path=MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    interp.allocate_tensors()
    _, in_height, in_width, _ = interp.get_input_details()[0]['shape']

    vstreamer = VideoStreamer(cam_config['rtmp_stream'], CAM_ID)
    vwriter = cv2.VideoWriter()

    stime = time.time()
    alert_time = time.time()
    framect = 0
    missed_frames = 0
    fps = 5.0
    num_detections = 0
    detections_in_a_row = 0
    last_frames = [np.ones((480, 640, 3))
                   for _ in range(BUFFER_SIZE)]
    recording_event = False


    # Video buffer stuff
    detection_buffer = []

    ret, frame = vstreamer.get_frame()

    vstreamer.wait_for_open()
    while run_process:
        # 16200 is every hour assuming 4.5FPS
        # if framect % 16200 is 0:
        if framect % 25 == 0:
            fps = framect / (time.time() - stime)
            if framect % 16200 == 0:
                print("FPS: %f" % fps)
            if framect >= 1000000:
                # prevent overflow
                framect = 0
                stime = time.time()

        if framect % 5 == 0:
            last_frames.pop(0)
            last_frames.append(frame)

        ret, frame = vstreamer.get_frame()

        updated = False
        for lf in last_frames[:10]:
            if not np.array_equal(frame, lf):
                updated = True
                break

        if not ret or not updated:
            missed_frames += 1
            if missed_frames > 50:
                vstreamer.renegotiate_connection()
                missed_frames = 0
            continue

        missed_frames = 0

        if SHOW_VIDEO:
            cv2.imshow("%s : press 'q' to quit" % cam_config['name'], frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        resized = cv2.resize(frame, (in_width, in_height), cv2.INTER_AREA)

        results = detect_objects(interp, resized, threshold)
        detected = person_detected(results)
        if detected and ((time.time() - alert_time) > 10):
            detections_in_a_row += 1
            if detections_in_a_row > 2:
                sss_alert(SYNO_ADDRESS, SYNO_SESSION, CAM_ID)
                ctime = datetime.datetime.today()
                video_name = "detections/videos/detection_%i-%i_%i-%i.avi" % (ctime.month,
                                                                              ctime.day,
                                                                              ctime.hour,
                                                                              ctime.minute)
                video_codec = cv2.VideoWriter_fourcc(*'MJPG')

                if vwriter.open(video_name, video_codec, fps,
                                (int(frame.shape[1]), int(frame.shape[0]))) is not True:
                    # do some error handling here
                    print('Unable to open the VideoWriter for some reason...')
                    continue

                for f in last_frames:
                    vwriter.write(f)

                event_starttime = time.time()
                recording_event = True

                alert_time = time.time()
                detections_in_a_row = 0
                num_detections += 1

                if SHOW_VIDEO:
                    annotated = annotate_image(frame, results, labels)
                    cv2.imshow('detection', annotated)

                if SAVE_DETECTIONS:
                    fname = "detections/frames/detection_%i-%i_%i-%i" % (ctime.month,
                                                                         ctime.day,
                                                                         ctime.hour,
                                                                         ctime.minute)
                    annotated = annotate_image(frame, results, labels)
                    cv2.imwrite(fname+'.jpg', annotated)

                    with open(fname+'.txt', '+w') as f:
                        for obj in results:
                            ymin, xmin, ymax, xmax = obj['bounding_box']
                            f.write(f'{ymin},{xmin},{ymax},{xmax}\n')

        elif recording_event and ((time.time() - alert_time) <= 10):
            vwriter.write(frame)
        elif recording_event and ((time.time() - alert_time) > 10):
            recording_event = False
            vwriter.release()
        else:
            detections_in_a_row = 0

        framect += 1

    vstreamer.destroy(SYNO_SESSION, SYNO_ADDRESS)
    sss_logout()
    if SHOW_VIDEO:
        cv2.destroyAllWindows()
