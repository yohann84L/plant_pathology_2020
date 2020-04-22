from easydict import EasyDict

_C = EasyDict()
cfg = _C

# OPTIMIZER
_C.LEARNING_RATE = 1e-2
# LEARNING RATE SCHEDULER
_C.LR_SCHED_GAMMA = 0.8
_C.LR_SCHED_STEP_SIZE = 5