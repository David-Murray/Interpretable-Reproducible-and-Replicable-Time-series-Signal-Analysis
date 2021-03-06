import logging
import time

log_file_name = './logs/{}.log'.format(time.strftime("%Y-%m-%d-%H:%M:%S").replace(':','-'))

with open(log_file_name, 'w'):
    pass

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
fileHandler = logging.FileHandler("{0}".format(log_file_name))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


def log(string, level='info'):

    if level == 'info':
        rootLogger.info(string)
    elif level == 'debug':
        rootLogger.debug(string)
    elif level == 'warning':
        rootLogger.warning(string)

