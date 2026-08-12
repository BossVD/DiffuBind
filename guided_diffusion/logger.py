"""
Logger stub — minimal replacement for guided-diffusion logger.
Only used by MixedPrecisionTrainer (which we don't use), so all functions are no-ops.
"""
def logkv_mean(key, value):
    pass

def logkv(key, value):
    pass

def log(message):
    pass

def dumpkvs():
    pass

def configure(*args, **kwargs):
    pass

def get_dir():
    return "."
