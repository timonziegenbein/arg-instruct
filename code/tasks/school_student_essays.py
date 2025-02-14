from task import Task, Instance
import numpy as np
import csv
import os
import json

base_path = os.environ['ARGPACA_MAJA']

class DiscourseFunctionTaggingSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='discourse-function-tagging_school-student-essays',
            task_instruction='Extract all corresponding text segments from the given essay for each of the discourse functions Introduction, Body, and Conclusion.',
            dataset_names=['school-student-essays'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            output = ''
            annos = ['"' + e['text'] + '"' for e in elem['macro_l1'] if e['label'] == 'Einleitung']
            if len(annos) > 0:
                output += 'Introduction: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['macro_l1'] if e['label'] == 'Hauptteil']
            if len(annos) > 0:
                output += '\nBody: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['macro_l1'] if e['label'] == 'Konklusion']
            if len(annos) > 0:
                output += '\nConclusion: ' + '; '.join(annos)
            if output == '':
                output = 'No discourse functions found.'

            instance = Instance(
                input=elem['text'],
                output=output.replace('\n\n', '\n'),
                split=cur_split,
            )
            self.instances.append(instance)


class ArgumentTaggingSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='argument-tagging_school-student-essays',
            task_instruction='Extract all corresponding text segments from the given essay for each argument type: Argument, Counterargument.',
            dataset_names=['school-student-essays'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            output = ''
            annos = ['"' + e['text'] + '"' for e in elem['macro_l2'] if e['label'] == 'Argument']
            if len(annos) > 0:
                output += 'Arguments: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['macro_l2'] if e['label'] == 'Gegenargument']
            if len(annos) > 0:
                output += '\nCounterarguments: ' + '; '.join(annos)
            if output == '':
                output = 'No arguments found.'

            instance = Instance(
                input=elem['text'],
                output=output.replace('\n\n', '\n'),
                split=cur_split,
            )
            self.instances.append(instance)


class ComponentTaggingSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='component-tagging_school-student-essays',
            task_instruction='Extract all corresponding text segments from the given essay for each component type: Topic, Thesis, Modified Thesis, Antithesis, Claim and Premise. Thesis is the main standpoint of the whole argumentative text towards the topic, Antithesis is the thesis contrary to the actual thesis and Modified Thesis is any modified version of the actual Thesis (e.g., more detailed or resticted).',
            dataset_names=['school-student-essays'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            output = ''
            annos = ['"' + e['text'] + '"' for e in elem['micro_l1'] if e['label'] == 'Thema']
            if len(annos) > 0:
                output += 'Topic: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l1'] if e['label'] == 'These']
            if len(annos) > 0:
                output += '\nThesis: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l1'] if e['label'] == 'Gegenthese']
            if len(annos) > 0:
                output += '\nAntithesis: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l1'] if e['label'] == 'Modifizierte These']
            if len(annos) > 0:
                output += '\nModified Thesis: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l1'] if e['label'] == 'Claim']
            if len(annos) > 0:
                output += '\nClaim: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l1'] if e['label'] == 'Premise']
            if len(annos) > 0:
                output += '\nPremise: ' + '; '.join(annos)

            if output == '':
                output = 'No components found.'

            instance = Instance(
                input=elem['text'],
                output=output.replace('\n\n', '\n'),
                split=cur_split,
            )
            self.instances.append(instance)


class DiscourseModeTaggingSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='discourse-mode-tagging_school-student-essays',
            task_instruction='Extract all corresponding text segments from the given essay for each discourse mode: Comparing, Conceding, Concluding, Describing, Exemplifying, Instructing, Positioning, Reasoning, Referencing and Qualifying.',
            dataset_names=['school-student-essays'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            output = ''
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Abw\u00e4gen']
            if len(annos) > 0:
                output += 'Comparing: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Konzedieren']
            if len(annos) > 0:
                output += '\nConceding: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Schlussfolgern']
            if len(annos) > 0:
                output += '\nConcluding: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Beschreiben']
            if len(annos) > 0:
                output += '\nDescribing: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Exemplifizieren']
            if len(annos) > 0:
                output += '\nExemplifying: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Auffordern']
            if len(annos) > 0:
                output += '\nInstructing: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Positionieren']
            if len(annos) > 0:
                output += '\nPositioning: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Begr\u00fcnden']
            if len(annos) > 0:
                output += '\nReasoning: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Referieren']
            if len(annos) > 0:
                output += '\nReferencing: ' + '; '.join(annos)
            annos = ['"' + e['text'] + '"' for e in elem['micro_l2'] if e['label'] == 'Einschr\u00e4nken']
            if len(annos) > 0:
                output += '\nQualifying: ' + '; '.join(annos)

            if output == '':
                output = 'No discourse mode found.'

            instance = Instance(
                input=elem['text'],
                output=output.replace('\n\n', '\n'),
                split=cur_split,
            )
            self.instances.append(instance)


class RelevanceScoringSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='relevance-scoring_school-student-essays',
            task_instruction='Score the relevance of the given school student essay on a scale from 1 (unsuccessful), 2 (rather unsuccessful), 3 (rather successful) to 4 (completely successful), with half points in between.',
            dataset_names=['school-student-essays'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            instance = Instance(
                input=elem['text'],
                output=str(elem['textfunktion']),
                split=cur_split,
            )
            self.instances.append(instance)


class ContentScoringSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='content-scoring_school-student-essays',
            task_instruction='Score the content of the given school student essay on a scale from 1 (unsuccessful), 2 (rather unsuccessful), 3 (rather successful) to 4 (completely successful), with half points in between.',
            dataset_names=['school-student-essays'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            instance = Instance(
                input=elem['text'],
                output=str(elem['inhaltliche_ausgestaltung']),
                split=cur_split,
            )
            self.instances.append(instance)


class StructureScoringSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='structure-scoring_school-student-essays',
            task_instruction='Score the structure of the given school student essay on a scale from 1 (unsuccessful), 2 (rather unsuccessful), 3 (rather successful) to 4 (completely successful), with half points in between.',
            dataset_names=['school-student-essays'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            instance = Instance(
                input=elem['text'],
                output=str(elem['textstruktur']),
                split=cur_split,
            )
            self.instances.append(instance)


class StyleScoringSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='style-scoring_school-student-essays',
            task_instruction='Score the style of the given school student essay on a scale from 1 (unsuccessful), 2 (rather unsuccessful), 3 (rather successful) to 4 (completely successful), with half points in between.',
            dataset_names=['school-student-essays'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            instance = Instance(
                input=elem['text'],
                output=str(elem['sprachliche_ausgestaltung']),
                split=cur_split,
            )
            self.instances.append(instance)


class ScoringSchoolStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='scoring_school-student-essays',
            task_instruction='Score the overall quality of the given school student essay on a scale from 1 (unsuccessful), 2 (rather unsuccessful), 3 (rather successful) to 4 (completely successful), with half points in between.',
            dataset_names=['school-student-essays'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/school_student_essays/arg-school-corpus.json'
        with open(ds_path) as f:
            data = json.loads(f.read())

        for elem in data:
            cur_split = 'train'
            if elem['fold'] == 0:
                cur_split = 'test'
            elif elem['fold'] == 1:
                cur_split = 'dev'

            instance = Instance(
                input=elem['text'],
                output=str(elem['gesamteindruck']),
                split=cur_split,
            )
            self.instances.append(instance)
