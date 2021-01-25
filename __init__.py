#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import signal
import sys

from core import do_offline_test, monitor_stream
from syno_integration import *

'''
Global variables
'''
SYNO_ADDRESS = None
SYNO_SESSION = None
JOBS = []


def sigint_handler(signum, frame):
    '''Handler that calls shutdown_gracefully'''

    for _, run_process in JOBS:
        run_process.clear()
    for job, _ in JOBS:
        job.join()

    sys.exit()


def load_config_file() -> dict:
    '''
    Loads config.txt from the root dir and parses it to a Config object.
    '''
    # cp = configparser.ConfigParser()
    # cp.read('config.txt')
    # config = SCAConfig(cp['Model']['model_path'], cp['Model']['labels_path'],
    #                    cp['Synology']['address'], cp['Synology']['account'],
    #                    cp['Synology']['passwd'], cp['Camera']['username'],
    #                    cp['Camera']['password'], cp['Camera']['port'],
    #                    cp['Camera']['address'], cp['Camera']['num_cams'],
    #                    float(cp['Model']['Threshold'])
    #                    )
    # return config
    with open('configfile.txt', 'r') as f:
        data = json.load(f)
    return data


def main():
    '''Main method - read images in repo and run and time execution'''
    signal.signal(signal.SIGINT, sigint_handler)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Monitors one or more video streams for people and \
        alerts a Synology Surveillance Station server on detection.\n\n\
        Check config.txt for config details.")

    parser.add_argument('--test', action='store_true',
                        help="Run a timed test inference on images in the \
                        test_images folder and exit.")
    parser.add_argument('--threshold-all', action='store_false',
                        help="Use the threshold_all value as the threshold for all streams.")

    args = parser.parse_args()

    config = load_config_file()

    if args.test:
        do_offline_test(config, args)

    else:
        JOBS = []
        for i in range(len(config['cameras'])):
            e = mp.Event()
            e.set()
            p = mp.Process(target=monitor_stream,
                           args=(config,
                                 config['cameras'][i],
                                 e,
                                 args))
            JOBS.append((p, e))
            p.start()

        for job, _ in JOBS:
            job.join()


if __name__ == '__main__':
    main()
