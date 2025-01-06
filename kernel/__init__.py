import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', datefmt='%d-%b %H:%M:%S')
