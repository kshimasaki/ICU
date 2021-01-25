#!/usr/bin/env python3

import threading
import time

import cv2


class LoginResponse:
    '''An object that describes the response to a login request'''
    def __init__(self, sid: str = '', did: str = '', stok: str = '',
                 ipp: bool = False):
        '''Initializes a new LoginResponse object'''
        self.sid = sid
        self.did = did
        self.synotoken = stok
        self.is_portal_port = ipp


class VideoStreamer:
    '''A class to enable capturing the stream(s) in a multithreaded way.'''
    # TODO Add timeout customization for renegotiate_connection
    def __init__(self, stream_addr: str, cam_id: int):
        '''Initializes a new VideoStreamer object running in a new thread'''
        self.cam_id = cam_id
        self.stream_addr = stream_addr

        self._is_running = threading.Event()
        self._is_running.set()
        self._end_thread = threading.Event()
        self._end_thread.clear()

        print('Initializing VideoStreamer for', self.stream_addr)
        self._cap = cv2.VideoCapture(self.stream_addr)
        '''
        while not self._cap.isOpened():
            time.sleep(10)
            self._cap.release()
            self._cap = cv2.VideoCapture(self.stream_addr)
        '''

        self._thread = threading.Thread(target=self._update, args=(), daemon=True)

        # prime self.ret and self.frame
        self._ret, self._frame = self._cap.read()
        self._thread.start()

    def _update(self):
        '''Updates to get the newest frame from the stream.'''
        while not self._end_thread.is_set():
            self._is_running.wait()
            self._ret, self._frame = self._cap.read()
            time.sleep(0.1)

    def get_frame(self):
        '''Returns a tuple of (ret, frame).'''
        return (self._ret, self._frame)

    def get_cap(self):
        '''Returns the cv2.VideoCapture object this object is holding.'''
        return self._cap

    def stop(self):
        '''Clears the internal _is_running flag.'''
        self._is_running.clear()

    def start(self):
        '''Sets the internal _is_running flag'''
        self._is_running.set()

    def stopped(self):
        '''Returns the running status of the object.'''
        return self._is_running.is_set()

    def wait_for_open(self):
        '''Waits until the VideoCapture object is online.'''
        while not self._cap.isOpened():
            time.sleep(0.1)

    def renegotiate_connection(self):
        '''Renegotiates connection with video stream.'''
        print(datetime.datetime.now(), "CONNECTION RENEGOTIATION")
        cv2.destroyAllWindows()
        self.stop()
        print('reconnecting to', self.stream_addr )
        self._cap = cv2.VideoCapture(self.stream_addr)
        while not self._cap.isOpened():
            if self._end_thread.is_set():
                return
            while self._is_running.is_set():
                time.sleep(0.1)
            time.sleep(3)
            self._cap.release()
            self._cap = cv2.VideoCapture(self.stream_addr)
        print('restarting')
        self._ret, self._frame = self._cap.read()
        self.start()

    def destroy(self):
        '''Destroys the VideoStreamer object.'''
        self._end_thread.set()
        self._cap.release()
        self._end_thread.clear()
