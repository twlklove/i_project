import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter_str='%(asctime)s-[%(levelname)s]-[%(process)d_%(thread)d_%(threadName)s]-[%(filename)s_%(funcName)s_%(lineno)d]-%(message)s'
formatter = logging.Formatter(formatter_str)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("hello")
