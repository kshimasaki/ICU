#!/usr/bin/env python3

import json
import sys
import time
from io import BytesIO
from typing import List, Tuple

import cv2
import requests as rq
import urllib3
from PIL import Image


def sss_login(address: str, account: str, passwd: str,
              otp: str = '') -> rq.sessions.Session:
    '''Takes a LoginInfo object and returns a requests.session'''
    rq_str = [address, "/webapi/auth.cgi?api=SYNO.API.Auth&method=Login",
              "&version=1",
              "&account=%s" % account,
              "&passwd=%s" % passwd,
              "&session=SurveillanceStation"]

    if otp != '':
        rq_str += "&otp_code=%s" % otp

    # TODO Add cert verification
    # TODO Add better session mgmt
    sess = rq.Session()
    while True:
        try:
            response = sess.get(''.join(rq_str), verify=False)
            response_json = response.json()
            print("Login response", response_json)
            return sess
        except urllib3.exceptions.NewConnectionError:
            print("Failed to establish new connection. Retrying.")
            time.sleep(5)
        except json.JSONDecodeError:
            print("Could not login. Dumping response and retrying.")
            print(response.text)
            time.sleep(5)

    return sess


def sss_get_snapshot(sess: rq.Session, address: str, cam_id: int,
                     profile: int = 1) -> Image.Image:
    '''Returns a PIL.Image snapshot of the specified camera.'''
    rq_str = [address, "/webapi/entry.cgi?api=SYNO.SurveillanceStation.Camera"]
    rq_str += "&method=GetSnapshot"
    rq_str += "&version=9"
    rq_str += "&id=%i" % cam_id

    #print('snapshot request url:', ''.join(rq_str))
    resp = sess.get(''.join(rq_str), verify=False)
    try:
        resp_json = resp.json()
        # We must have an error
        # TODO Log properly to stderr
        print("Error grabbing snapshot. Error code %i" %
              resp_json['error']['code'])

    except json.JSONDecodeError:
        # We must not have an error
        i = Image.open(BytesIO(resp.content))
        return i
    return None


def get_last_recording_stream(sess: rq.Session, cam_id: int) -> str:
    '''Gets a streaming link to the last recorded stream.'''
    # TODO Finish this
    request = [SYNO_ADDRESS, "/webman/entry.cgi?", "version=6", "&limit=1",
               "&api=SYNO.SurveillanceStation.Recording", "&method=List",
               "&cameraIds=%s" % cam_id]
    response = sess.get(request)
    recording_id = response.json()['recordings'][0]['id']

    return None


def sss_logout(sess: rq.sessions.Session, address: str):
    '''Logs out and shuts down gracefully.'''

    rqstr = [address, "/webapi/auth.cgi?api=SYNO.API.Auth&method=Logout",
             "&version=1&session=SurveillanceStation"]
    logout_resp = sess.get(''.join(rqstr), verify=False)

    print('checking logout_resp')
    if logout_resp.json()['success']:
        print("Shutting down gracefully")
        sys.exit()
    else:
        print("Could not shutdown gracefully.")
        print("Exiting without grace.")
        sys.exit()


def get_sss_cam_addresses(address: str, sess: rq.sessions.Session,
                          cam_ids: List[int]) -> List[Tuple]:
    url_str = [address, "/webapi/entry.cgi?",
               "api=SYNO.SurveillanceStation.Camera&method=GetLiveViewPath",
               "&version=9"]
    url_str += "&idList=" + ','.join(str(x) for x in cam_ids)

    resp_json = sess.get(''.join(url_str), verify=False).json()
    urls = []
    #print('resp_json', resp_json)
    for obj in resp_json['data']:
        #print('obj', obj)
        tup = (obj['id'], obj['rtspPath'])
        # print(tup)
        urls.append(tup)

    caps = []
    for (cam_id, url) in urls:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            caps.append(cap)
        else:
            print("couldn't open cam.")

    return caps


def sss_alert(address: str, session: rq.sessions.Session,
               cam_id: int) -> None:
    '''
    Triggers the specified camera for person detection
    '''
    rq_str = [address,
              "/webapi/entry.cgi?",
              "api=SYNO.SurveillanceStation.ExternalEvent",
              "&version=1", "&method=Trigger",
              "&eventId=%s" % cam_id]
    print('sending alert to: ', rq_str)
    alert_resp = session.get(''.join(rq_str), verify=False)
    return
