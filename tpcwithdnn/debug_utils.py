"""
Utilities for logging elapsed time and memory size.
"""
import sys
import resource
import psutil

from tpcwithdnn.logger import get_logger

def log_time(start, end, comment):
    """
    Write elapsed time to the console.

    :param float stat: start time
    :param float end: end time
    :param str comment: string attached to the console output
    """
    logger = get_logger()
    elapsed_time = end - start
    time_min = int(elapsed_time // 60)
    time_sec = int(elapsed_time % 60)
    logger.info("Elapsed time %s: %dm %ds", comment, time_min, time_sec)

def get_memory_usage(obj):
    """
    Get memory used by the object obj.
    
    :param obj obj: object for inspection
    :return: size of the object
    :rtype: float
    """
    return sys.getsizeof(obj)

def format_memory(size):
    """
    Convert memory size to a pretty string.

    :param float size: memory size
    :return: rounded size and the corresponding value prefix
    :rtype: tuple(int, char)
    """
    if 1024 <= size < 1024**2:
        return size // 1024, 'k'
    if 1024**2 <= size < 1024**3:
        return size // (1024**2), 'M'
    if 1024**3 <= size:
        return size // (1024**3), 'G'
    return size, ''

def log_memory_usage(objects):
    """
    Write memory sizes of the objects to the console.

    :param list objects: list of tuples(obj, str) with objects and logging comments
    """
    logger = get_logger()
    for obj, comment in objects:
        size, mult = format_memory(get_memory_usage(obj))
        logger.info("%s memory usage: %d %sB", comment, size, mult)

def log_total_memory_usage(comment=None):
    """
    Write the memory usage of the program to the console.

    :param str comment: additional comment for logging
    """
    logger = get_logger()
    if comment is not None:
        logger.info(comment)
    size, mult = format_memory(psutil.virtual_memory().available)
    logger.info("Free RAM: %d %sB", size, mult)
    size, mult = format_memory(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    logger.info("RAM used by application: %d %sB", size, mult)
