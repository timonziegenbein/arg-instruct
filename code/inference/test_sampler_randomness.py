import numpy as np
import sys
sys.path.insert(0, '..')
from samplers import InstanceSampler

np.random.seed(42)
instance_sampler = InstanceSampler()

test_instances = []
for instances in instance_sampler.get_batch('test', clf_only=True): # the clf_only flag is only here to compare to the cappr model and should be removed
    for instance in instances:
        print(instance.id)
