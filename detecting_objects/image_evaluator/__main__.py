#!/usr/bin/env python

# internal
import logging
import os

# my lib
from src import image_evaluator

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

	image_evaluator.run()

