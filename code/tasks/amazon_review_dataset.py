from task import Task, Instance
import pandas as pd
import numpy as np
import json
import os

base_path = os.environ['ARGPACA_MAJA']
ds_path = base_path + '/data/amazon_review_dataset/'

train_data = []
with open(ds_path + 'training.jsonlist', 'r') as f:
    train_data.extend([json.loads(line) for line in f])
test_data = []
with open(ds_path + 'test.jsonlist', 'r') as f:
    test_data.extend([json.loads(line) for line in f])
products = []
with open(ds_path + 'products.jsonlist', 'r') as f:
    products.extend([json.loads(line) for line in f])


class ReviewHelpfulnessPredictionAmazonReviewDataset(Task):
    def __init__(self, **kwargs):
        super().__init__(
            # Here, each ri,j is assigned a helpfulness score of si,j ∈ {0, . . . , 4} derived from grouping actual helpfulness vote counts into five bins with powers of 2 as boundaries, i.e., [0,1],[2,3],[4,7],[8,15],[16,∞).
            task_name='review-helpfulness-prediction_amazon-review-dataset',
            task_instruction='Score the helpfulness of the following review on a scale from 0 (lowest) to 4 (highest).',
            dataset_names=['amazon-review-dataset'],
            is_reg=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        for review in train_data:
            text = ''
            for proposition in review['propositions']:
                text += ' ' + proposition['text']
            text = text.strip()

            if not 'vote' in review.keys() or int(review['vote']) <= 1:
                label = 0
            elif int(review['vote']) >= 2 and int(review['vote']) <= 3:
                label = 1
            elif int(review['vote']) >= 4 and int(review['vote']) <= 7:
                label = 2
            elif int(review['vote']) >= 8 and int(review['vote']) <= 15:
                label = 3
            elif int(review['vote']) >= 16:
                label = 4

            instance = Instance(
                input=text,
                output=str(label),
                split='train',
            )
            self.instances.append(instance)

        for review in test_data:
            text = ''
            for proposition in review['propositions']:
                text += ' ' + proposition['text']
            text = text.strip()

            if not 'vote' in review.keys() or int(review['vote']) <= 1:
                label = 0
            elif int(review['vote']) >= 2 and int(review['vote']) <= 3:
                label = 1
            elif int(review['vote']) >= 4 and int(review['vote']) <= 7:
                label = 2
            elif int(review['vote']) >= 8 and int(review['vote']) <= 15:
                label = 3
            elif int(review['vote']) >= 16:
                label = 4

            instance = Instance(
                input=text,
                output=str(label),
                split='test',
            )
            self.instances.append(instance)


class UnitSegmentationAmazonReviewDataset(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='unit-segmentation-prediction_amazon-review-dataset',
            task_instruction='Segment the following review into its elementary argumentative units. Elementary argumentative units are the fundamental components of an argument.',
            dataset_names=['amazon-review-dataset'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        for review in train_data:
            text = ''
            for proposition in review['propositions']:
                text += ' ' + proposition['text']
            text = text.strip()
            output = '\n'.join([p['text'] for p in review['propositions']])

            instance = Instance(
                input=text,
                output=output,
                split='train',
            )
            self.instances.append(instance)

        for review in test_data:
            text = ''
            for proposition in review['propositions']:
                text += ' ' + proposition['text']
            text = text.strip()
            output = '\n'.join([p['text'] for p in review['propositions']])

            instance = Instance(
                input=text,
                output=output,
                split='test',
            )
            self.instances.append(instance)


class UnitClassificationAmazonReviewDataset(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='unit-classification-prediction_amazon-review-dataset',
            task_instruction='Label each elementary argumentative unit as REFERENCE or as one of the proposition types FACT, TESTIMONY, POLICY, and VALUE. FACT (Proposition of Non-Experiential Fact) is an objective proposition, meaning it does not leave any room for subjective interpretations or judgements. For example, “and battery life is about 8-10 hours.”. TESTIMONY (Proposition of Experiential Fact) is also an objective proposition. However, it differs from FACT in that it is experiential, i.e., it describes a personal state or experience. For example, “I own Sennheisers, Bose, Ludacris Souls, Beats, etc.”. POLICY (Proposition of Policy) is a subjective proposition that insists on a specific course of action. For example, “They need to take this product off the market until the issue is resolved.”. VALUE (Proposition of Value) is a subjective proposition that is not POLICY. It is a personal opinion or expression of feeling. For example, “They just weren’t appealing to me”. REFERENCE (Reference to a Resource) is the only non-proposition elementary unit that refers to a resource containing objective evidence. In product reviews, REFERENCE is usually a URL to another product page, image or video. Also, REFERENCE cannot be supported by other elementary units. For example, “https://images-na.ssl-images-amazon.com/[...]”.',
            dataset_names=['amazon-review-dataset'], 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        for review in train_data:
            input = '\n'.join([p['text'] for p in review['propositions']])
            output = '\n'.join([p['type'].upper() for p in review['propositions']])

            instance = Instance(
                input=input,
                output=output,
                split='train',
            )
            self.instances.append(instance)

        for review in test_data:
            input = '\n'.join([p['text'] for p in review['propositions']])
            output = '\n'.join([p['type'].upper() for p in review['propositions']])

            instance = Instance(
                input=input,
                output=output,
                split='test',
            )
            self.instances.append(instance)


class RelationDetectionAmazonReviewDataset(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='relation-detection_amazon-review-dataset',
            task_instruction='What kind of support relation, if any, exists from elementary unit X for a proposition Y of the same argument? Differentiate between REASON, EVIDENCE and NO SUPPORT RELATION. Support relations in this scheme are two prevalent ways in which propositions are supported in practical argumentation: REASON and EVIDENCE. The former can support either objective or subjective propositions, whereas the latter can only support objective propositions. That is, you cannot prove that a subjective proposition is true with a piece of evidence. REASON: For an elementary unit X to be a REASON for a proposition Y, it must provide a reason or a justification for Y. For example, “The only issue I have is that the volume starts to degrade a little bit after about six“and I find I have to buy a new pair every year or so.”(Y). EVIDENCE: For an elementary unit X to be EVIDENCE for a proposition Y, it must prove that Y is true. For example, “https://images-na.ssl-images-amazon.com/[...]”(X) and “The product arrived damage[d],”(Y).',
            dataset_names=['amazon-review-dataset'],
            is_clf=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        for review in train_data:
            text = ''
            for proposition in review['propositions']:
                text += ' ' + proposition['text']
            text = text.strip()

            for i in range(len(review['propositions'])):
                for j in range(len(review['propositions'])):
                    if j != i:
                        input = f"Elementary unit X: {review['propositions'][j]['text']}\nProposition Y: {review['propositions'][i]['text']}"
                        if review['propositions'][i]['reasons'] and str(j) in review['propositions'][i]['reasons']:
                            output = 'REASON'
                        elif review['propositions'][i]['evidence'] and str(j) in review['propositions'][i]['evidence']:
                            output = 'EVIDENCE'
                        else:
                            output = 'NO SUPPORT RELATION'

                        instance = Instance(
                            input=f"Argument: {text}\n{input}",
                            output=output,
                            split='train',
                        )
                        self.instances.append(instance)

        for review in test_data:
            text = ''
            for proposition in review['propositions']:
                text += ' ' + proposition['text']
            text = text.strip()

            for i in range(len(review['propositions'])):
                for j in range(len(review['propositions'])):
                    if j != i:
                        input = f"Elementary unit X: {review['propositions'][j]['text']}\nProposition Y: {review['propositions'][i]['text']}"
                        if review['propositions'][i]['reasons'] and str(j) in review['propositions'][i]['reasons']:
                            output = 'REASON'
                        elif review['propositions'][i]['evidence'] and str(j) in review['propositions'][i]['evidence']:
                            output = 'EVIDENCE'
                        else:
                            output = 'NO SUPPORT RELATION'

                        instance = Instance(
                            input=f"Argument: {text}\n{input}",
                            output=output,
                            split='test',
                        )
                        self.instances.append(instance)
