#!/usr/bin/env python

# internal
import logging
import os

# my lib
from src import image_evaluator

"""
from src import image_editing
from src import data_paths
from src import create_samples
from src import config_handler
from src import opencv_haar_cascade_cmds_v2 as opencv_haar_cascade_cmds
from src import test_detection
"""

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

	image_evaluator.run()

