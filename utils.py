import logging
import time
import yaml
from functools import wraps

def setup_logging(log_filename):
    logging.basicConfig(filename=log_filename,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Started {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Finished {func.__name__} in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_checkpoint(model, epoch, checkpoint_dir='checkpoints'):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f'Checkpoint saved: {checkpoint_path}')
