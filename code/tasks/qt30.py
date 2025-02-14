from task import Task, Instance
import os
import numpy as np
import json

base_path = os.environ['ARGPACA_MAJA']

class PropositionalRelationsIdentificationQT30(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'propositional-relations-identification_qt30',
            'Detect the argumentative relations between the propositions identified and segmented in an argumentative dialogue. Such relations are: Default Inference (provide a reason to accept another proposition), Default Conflict (provide an incompatible alternative to another proposition), Default Reformulation (rephrase, restate or reformulate another proposition) and No Relation.',
            ['qt30'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/qt30/dataset/'
        files = os.listdir(ds_path)
        nodeset_files = [f for f in files if 'nodeset' in f]

        np.random.shuffle(nodeset_files)
        train_nodeset_files = nodeset_files[:int(len(nodeset_files)*0.7)]
        dev_nodeset_files = nodeset_files[int(len(nodeset_files)*0.7):int(len(nodeset_files)*0.8)]
        test_nodeset_files = nodeset_files[int(len(nodeset_files)*0.8):]

        for nodeset_file in nodeset_files:
            file = open(os.path.join(ds_path, nodeset_file))
            nodeset = json.load(file)
            nodes_dict = {node['nodeID']: node for node in nodeset['nodes']}
            node_combinations = []
            for edge in nodeset['edges']:
                if nodes_dict[edge['fromID']]['type'] == 'I' and nodes_dict[edge['toID']]['type'] in ['RA', 'CA', 'MA']:
                    I_out = nodes_dict[edge['fromID']]
                    S_in = nodes_dict[edge['toID']]
                    for edge2 in nodeset['edges']:
                        if edge2['fromID'] == edge['toID'] and nodes_dict[edge2['toID']]['type'] == 'I':
                            I_in = nodes_dict[edge2['toID']]
                            instance = Instance(
                                'Proposition1: ' + I_out['text'] + '\n' + 'Proposition2: ' + I_in['text'], S_in['text'], 'train' if nodeset_file in train_nodeset_files else 'dev' if nodeset_file in dev_nodeset_files else 'test')
                            node_combinations.append((I_out['nodeID'], I_in['nodeID']))
                            self.instances.append(instance)
                            break
            # create all possible pairs of I nodes that are not connected
            node_combinations = set(node_combinations)
            I_nodes = [node for node in nodeset['nodes'] if node['type'] == 'I']
            for i in range(len(I_nodes)):
                for j in range(i+1, len(I_nodes)):
                    instance = Instance(
                        'Proposition1: ' + I_nodes[i]['text'] + '\n' + 'Proposition2: ' + I_nodes[j]['text'], 'No Relation', 'train' if nodeset_file in train_nodeset_files else 'dev' if nodeset_file in dev_nodeset_files else 'test')
                    already_exists = (I_nodes[i]['nodeID'], I_nodes[j]['nodeID']) in node_combinations
                    if not already_exists:
                        self.instances.append(instance)

            file.close()


class IllocutionaryRelationsIdentificationQT30(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'illocutionary-relations-identification_qt30',
            'Detect illocutonary relations existing between locutions uttered in the dialogue and the argumentative propositions associated with them such as: Agreeing (share the opinion of the interlocutorn), Restating (rephrases a previous claim), Challenging (seeking the grounds for an opinion), Arguing (provides justification to a claim), Assertive Questioning (communicates information and at the same time asks for confirmation/rejection), Asserting (asserts information or communicates an opinion), Rhetorical Questioning (expressing an opinion in the form of an interrogative), Disagreeing (declares not to share the interlocutor’s opinion), Pure Questioning (s seeking information or asking for an opinion), Default Illocuting (captures an answer to a question) and No Relation',
            ['qt30'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/qt30/dataset/'
        files = os.listdir(ds_path)
        nodeset_files = [f for f in files if 'nodeset' in f]

        np.random.shuffle(nodeset_files)
        train_nodeset_files = nodeset_files[:int(len(nodeset_files)*0.7)]
        dev_nodeset_files = nodeset_files[int(len(nodeset_files)*0.7):int(len(nodeset_files)*0.8)]
        test_nodeset_files = nodeset_files[int(len(nodeset_files)*0.8):]

        for nodeset_file in nodeset_files:
            file = open(os.path.join(ds_path, nodeset_file))
            nodeset = json.load(file)
            nodes_dict = {node['nodeID']: node for node in nodeset['nodes']}
            node_combinations = []
            for edge in nodeset['edges']:
                if nodes_dict[edge['fromID']]['type'] == 'L' and nodes_dict[edge['toID']]['type'] == 'YA':
                    L_out = nodes_dict[edge['fromID']]
                    YA_in = nodes_dict[edge['toID']]
                    for edge2 in nodeset['edges']:
                        if edge2['fromID'] == edge['toID'] and nodes_dict[edge2['toID']]['type'] == 'I':
                            I_in = nodes_dict[edge2['toID']]
                            instance = Instance(
                                'Locution: ' + L_out['text'] + '\n' + 'Proposition: ' + I_in['text'],
                                YA_in['text'],
                                'train' if nodeset_file in train_nodeset_files else 'dev' if nodeset_file in dev_nodeset_files else 'test'
                            )
                            node_combinations.append((L_out['nodeID'], I_in['nodeID']))
                            self.instances.append(instance)
                            break

            # create all possible pairs of L and I nodes that are not connected
            node_combinations = set(node_combinations)
            L_nodes = [node for node in nodeset['nodes'] if node['type'] == 'L']
            I_nodes = [node for node in nodeset['nodes'] if node['type'] == 'I']
            for i in range(len(L_nodes)):
                for j in range(len(I_nodes)):
                    instance = Instance(
                        'Locution: ' + L_nodes[i]['text'] + '\n' + 'Proposition: ' + I_nodes[j]['text'],
                        'No Relation',
                        'train' if nodeset_file in train_nodeset_files else 'dev' if nodeset_file in dev_nodeset_files else 'test'
                    )
                    already_exists = (L_nodes[i]['nodeID'], I_nodes[j]['nodeID']) in node_combinations
                    if not already_exists:
                        self.instances.append(instance)

            file.close()


if __name__ == '__main__':
    task = PropositionalRelationsIdentificationQT30()
    print(task.instances[0].apply_template())
    batch = task.get_batch(split='train')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)

    print('+'*50)

    task = IllocutionaryRelationsIdentificationQT30()
    print(task.instances[0].apply_template())
    batch = task.get_batch(split='train')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)
