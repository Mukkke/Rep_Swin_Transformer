import logging
import os

def create_logger(output_dir, dist_rank=0, name=''):
    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s')

    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    log_file_path = os.path.join(output_dir, f'log_rank{dist_rank}.txt')
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
