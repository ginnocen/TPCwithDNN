# pylint: disable=missing-module-docstring, missing-function-docstring

import sys
import resource
import psutil

from tpcwithdnn.logger import get_logger

def log_time(start, end, comment):
    logger = get_logger()
    elapsed_time = end - start
    time_min = int(elapsed_time // 60)
    time_sec = int(elapsed_time - time_min)
    logger.info("Elapsed time %s: %dm %ds", comment, time_min, time_sec)

def get_memory_usage(obj):
    return sys.getsizeof(obj)

def format_memory(size):
    if 1024 <= size < 1024**2:
        return size // 1024, 'k'
    if 1024**2 <= size < 1024**3:
        return size // (1024**2), 'M'
    if 1024**3 <= size:
        return size // (1024**3), 'G'
    return size, ''

def log_memory_usage(objects):
    logger = get_logger()
    for obj, comment in objects:
        size, mult = format_memory(get_memory_usage(obj))
        logger.info("%s memory usage: %d %sB", comment, size, mult)

def log_total_memory_usage(comment=None):
    logger = get_logger()
    if comment is not None:
        logger.info(comment)
    size, mult = format_memory(psutil.virtual_memory().available)
    logger.info("Free RAM: %d %sB", size, mult)
    size, mult = format_memory(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    logger.info("RAM used by application: %d %sB", size, mult)
