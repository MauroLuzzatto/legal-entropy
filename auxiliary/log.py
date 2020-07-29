# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:13:34 2020

@author: mauro
"""
import logging

def initalize_logger(language, pathSave, stage):
    """
    Create a log file to record the experiment's logs
    
    Input:
        language (string): corpus type
        pathSave (string): path to the directory
        stage (string): file name
    Output:
        logger (obj): logger that record logs
    """
    # check if the file exist
    log_file = os.path.join(pathSave, "{}_{}.log".format(stage, language))

    # set logging format
    console_logging_format = "%(message)s"
    file_logging_format = '%(asctime)s: %(levelname)s: %(message)s'

    # configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    # Reset the logger.handlers if it already exists.
    if logger.handlers:
        logger.handlers = []
        
    # create a file handler for output file
    handler = logging.FileHandler(filename = log_file, mode='a')
       
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # consolore log
    consoleHandler = logging.StreamHandler()
    # set the logging level for log file
    consoleHandler.setLevel(logging.INFO)
    # set the logging format
    formatter = logging.Formatter(console_logging_format)
    consoleHandler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(consoleHandler)
    return logger
