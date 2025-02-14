from task import Task, Instance
import numpy as np
import pandas as pd
import csv
import os

base_path = os.environ['ARGPACA_MAJA']

class ClassifyingArgumentStrengthIcleArgumentStrength(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'classifying-argument-strength_icle-argument-strength',
            'Argument strength refers to the strength of the argument an essay makes for its thesis. An essay with a high argument strength score presents a strong argument for its thesis and would convince most readers. Score the argument strength of the given argumentative essay using the following scoring range:\n' +
            '1.0 (essay does not make an argument or it is often unclear what the argument is)\n' +
            '1.5\n' +
            '2.0 (essay makes a weak argument for its thesis or sometimes even argues against it)\n' +
            '2.5\n' +
            '3.0 (essay makes a decent argument for its thesis and could convince some readers)\n' +
            '3.5\n' +
            '4.0 (essay makes a strong argument for its thesis and would convince most readers)',
            ['icle-argument-strength'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        base_path_ = base_path + '/data/'
        icle_path = base_path_ + 'ICLEv3/texts/'
        anno_path = base_path_ + 'icle_argument_strength/ArgumentStrengthScores.txt'
        folds_path = base_path_ + 'icle_argument_strength/ArgumentStrengthFolds.txt'

        with open(anno_path) as f:
            reader = csv.reader(f, delimiter="\t")
            data = list(reader)

            with open(folds_path, 'r') as f:
                folds = f.read()
            folds = folds.split('\n\n')

            for d in data:
                with open(icle_path + d[0]+'.txt', 'r') as f:
                    text = f.readlines()
                text = ''.join(text[1:])  # remove id tag

                split = 'train'
                for i in range(len(folds)):
                    if d[0] in folds[i] and i == 1:
                        split = 'dev'
                    elif d[0] in folds[i] and i == 0:
                        split = 'test'

                instance = Instance(
                    input=text,
                    output=d[1],
                    split=split
                )
                self.instances.append(instance)
