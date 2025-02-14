import numpy as np
import pandas as pd
import os
import sys
from xml.etree import ElementTree as ET
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']
ds_path = base_path + '/data/comarg/'


class StanceDetectionComarg(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='stance-detection_comarg',
            task_instruction='Given the following comment-argument pair, decide which of these statements is true: The comment...\n...explicitly attacks the argument.\n...vaguely/implicitly attacks the argument.\n...makes no use of the argument.\n...vaguely/implicitly supports the argument.\n...explicitly supports the argument.',
            dataset_names=['comarg'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        for ds in ['UGIP.xml', 'GM.xml']:
            with open(os.path.join(ds_path, ds), 'r') as f:
                xml = f.read()
                f.close()
            xml_root = ET.fromstring(xml)
            for child in xml_root:
                for cchild in child:
                    if cchild.tag == 'comment':
                        for ccchild in cchild:
                            if ccchild.tag == 'text':
                                comment_text = ccchild.text
                            elif ccchild.tag == 'stance':
                                comment_stance = ccchild.text
                    elif cchild.tag == 'argument':
                        for ccchild in cchild:
                            if ccchild.tag == 'text':
                                argument_text = ccchild.text
                            elif ccchild.tag == 'stance':
                                argument_stance = ccchild.text
                    elif cchild.tag == 'label':
                        label = cchild.text

                if label and label == '1':
                    output = 'The comment explicitly attacks the argument.'
                elif label == '2':
                    output = 'The comment vaguely/implicitly attacks the argument.'
                elif label == '3':
                    output = 'The comment makes no use of the argument.'
                elif label == '4':
                    output = 'The comment vaguely/implicitly supports the argument.'
                elif label == '5':
                    output = 'The comment explicitly supports the argument.'
                else:
                    print('error')

                split = 'train'
                if ds == 'GM.xml':
                    split = 'test'

                instance = Instance(
                    input=f"Comment: {comment_text}\nArgument: {argument_text}",
                    output=output,
                    split=split,
                )
                self.instances.append(instance)
