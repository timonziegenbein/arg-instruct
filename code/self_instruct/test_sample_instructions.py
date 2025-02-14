from task import Task, Instance

from tasks.qt30 import PropositionalRelationsIdentificationQT30, IllocutionaryRelationsIdentificationQT30
from tasks.argkp import KeyPointMatchingArgKP, KeyPointGenerationArgKP
from tasks.f1000rd import PragmaticTaggingF1000rd
from tasks.webis_sameside_19 import SameSideStanceClassificationWebisSameside19
from tasks.argument_reasoning_comprehension import ArgumentReasoningComprehension

class BalancedSampler():

    _TASK_CHILD_CLASSES = [
            PropositionalRelationsIdentificationQT30,
            IllocutionaryRelationsIdentificationQT30,
            ArgumentReasoningComprehension
    ]

    def __init__(self):
        self.task_child_instances = [cls() for cls in self._TASK_CHILD_CLASSES]

    def get_batch(self, split, num_batches):
        # get a single batch from each task and store the length of the batches in a dictionary
        batch_dict = {}
        for task in self.task_child_instances:
            for batch in task.get_batch(split):
                if batch:
                    batch_dict[task.task_name] = len(batch)
                    break

        # get the maximum batch length
        max_batch_length = max(batch_dict.values())
        # replace values with how many batches are needed to reach approximately the same length at the maximum batch max_batch_length
        for task in self.task_child_instances:
            if task.task_name in batch_dict:
                batch_dict[task.task_name] = max_batch_length // batch_dict[task.task_name]

        # get the batches
        for i in range(num_batches):
            # create generators for each task
            generators = [(task.task_name, task.get_batch(split)) for task in self.task_child_instances if  task.task_name in batch_dict]
            batch = []
            for j, generator in enumerate(generators):
                for _ in range(batch_dict[generator[0]]):
                    mini_batch = next(generator[1], None)
                    if mini_batch:
                        batch.append(mini_batch)
                    else:
                        # reset generator
                        generators[j] = (generator[0], task.get_batch(split))

            # flatten the batch
            flat_batch = [item for sublist in batch for item in sublist]
            yield flat_batch


if __name__ == '__main__':
    sampler = BalancedSampler()
    for batch in sampler.get_batch('train', 5):
        task_names = [instance.task_name for instance in batch]
        print({task_name: task_names.count(task_name) for task_name in task_names})
