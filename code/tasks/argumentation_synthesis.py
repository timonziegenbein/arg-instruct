import numpy as np
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']

class SynthesizeArgumentArgumentationSynthesis(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='synthesize-argument_argumentation-synthesis',
            task_instruction='You are given the following question, stance (Yes vs. No) towards this question and a type of reasoning (logos vs. pathos). Your task is to form a persuasive argument toward the question that supports the given stance based on the following type of reasoning',
            dataset_names=['argumentation-synthesis'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        data_path = base_path + '/data/coling18/strategies-task-sheets'

        tasks = [i for i in range(1, 11)]
        np.random.seed(42)
        np.random.shuffle(tasks)
        train_tasks = tasks[:int(len(tasks)*0.7)]
        dev_tasks = tasks[int(len(tasks)*0.7):int(len(tasks)*0.8)]
        test_tasks = tasks[int(len(tasks)*0.8):]

        # read all csv files in the data_path
        for file in os.listdir(data_path):
            if file.endswith('.csv'):
                with open(os.path.join(data_path, file)) as f:
                    lines = f.readlines()
                    f.close()
                question = lines[3].split('\t')[2]
                stance = lines[4].split('\t')[2]
                reasoning = lines[7].split('\t')[2].replace('"', '')
                output = ' '.join([' '.join(line.split('\t')[1:]).replace('\n', '').strip() for line in lines[44:48]])
                instance = Instance(
                    input=f'Question: {question}\nStance: {stance}\nReasoning: {reasoning}',
                    output=output,
                    split='train' if int(file.split('.')[1][:2]) in train_tasks else 'dev' if int(
                        file.split('.')[1][:2]) in dev_tasks else 'test'
                )
                self.instances.append(instance)


if __name__ == '__main__':
    task = SynthesizeArgumentArgumentationSynthesis()
    # sort instaces by input length
    #task.instances.sort(key=lambda x: len(x.input))
    for i in range(10):
        print(task.instances[i+90].input)
        print(task.instances[i+90].output)
        print(task.instances[i+90].split)
        print()
