import pandas as pd
import numpy as np
import csv
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']

def get_parsed_data(dim_index):
    ds_path = base_path + '/data/dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-annotated.csv'
    with open(ds_path, newline='', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)

    # split by topic, like Wachsmuth and Werner (2020): Intrinsic Quality Assessment of Arguments
    topics = sorted(list(set([x[19] for x in data[1:]])))
    test_topic = topics[-1]
    val_topic = topics[-2]

    parsed_data = []
    for i in range(1, len(data[1:]), 3):
        argument = data[i][17]

        # scores
        scores = []
        for j in range(3):
            scores.append(data[i+j][dim_index])

        # split
        split = 'train'
        if data[i][19] == test_topic:
            split = 'test'
        elif data[i][19] == val_topic:
            split = 'dev'

        parsed_data.append((argument, scores, split))

    return parsed_data


def majority_vote(scores):
    if len(set(scores)) == 1:
        return scores[0]
    elif len(set(scores)) == 3:
        return '2 (Average)'
    else:
        return max(set(scores), key=scores.count)


# Tasks

class ClassifyingArgumentativenessDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'classifying-argumentativeness_dagstuhl-15512-arg-quality',
            # taken from annotation guidelines
            'A comment should be seen as argumentative if you think that it explicitly or implicitly conveys the stance of the comment’s author on some possibly contoversial issue. Usually, the stance is expressed through an argument (or a set of arguments). An argument can be seen as combination of a conclusion (in terms of a claim) and a set of premises (in terms of supporting reasons or evidence for the claim). However, parts of an argument may be implicit or may simply be missing. Do you think that the comment is argumentative?',
            ['dagstuhl-15512-arg-quality'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(1)
        for x in ad:
            argument = x[0]
            scores = x[1]

            label = 'No'
            if scores.count('y') > 1:
                label = 'Yes'

            instance = Instance(
                input=argument,
                output=label,
                split=x[2]
            )
            self.instances.append(instance)


class RateLocalAcceptabilityDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-local-acceptability_dagstuhl-15512-arg-quality',
            # taken from annotation guidelines
            'A premise of an argument should be seen as acceptable if it is worthy of being believed, i.e., if you rationally think it is true or if you see no reason for not believing that it may be true. If you identify more than one premise in the comment, try to adequately weight the acceptability of each premise when judging about their “aggregate” acceptability - unless there are particular premises that dominate your view of the author’s argumentation. If you identify more than one premise in the comment, try to adequately weight the acceptability of each premise when judging about their “aggregate” acceptability—unless there are particular premises that dominate your view of the author’s argumentation. How would you rate the acceptability of the premises of the author’s argument on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(3)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateLocalRelevanceDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-local-relevance_dagstuhl-15512-arg-quality',
            'A premise of an argument should be seen as relevant if it contributes to the acceptance or rejection of the argument’s conclusion, i.e., if you think it is worthy of being considered as a reason, evidence, or similar regarding the conclusion. If you identify more than one premise in the comment, try to adequately weight the relevance of each premise when judging about their “aggregate” relevance—unless there are particular premises that dominate your view of the author’s argumentation. You should be open to see a premise as relevant even if it does not match your own stance on the issue. How would you rate the relevance of the premises of the author’s argument on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(13)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateLocalSufficiencyDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-local-sufficiency_dagstuhl-15512-arg-quality',
            'The premises of an argument should be seen as sufficient if, together, they provide enough support to make it rational to draw the argument’s conclusion. If you identify more than one conclusion in the comment, try to adequately weight the sufficiency of the premises for each conclusion when judging about their “aggregate” sufficiency—unless there are particular premises or conclusions that dominate your view of the author’s argumentation. Notice that you may see premises as sufficient even though you do not personally accept all of them, i.e., sufficiency does not presuppose acceptability. How would you rate the sufficiency of the premises of the author’s argument on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(16)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateCogencyDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-cogency_dagstuhl-15512-arg-quality',
            'An argument should be seen as cogent if it has individually acceptable premises that are relevant to the argument’s conclusion and that are sufficient to draw the conclusion. Try to adequately weight your judgments about local acceptability, local relevance, and local sufficiency when judging about cogency—unless there is a particular dimension among these that dominates your view of an argument. Accordingly, if you identify more than one argument, try to adequately weight the cogency of each argument when judging about their “aggregate” cogency—unless there is a particular argument that dominates your view of the author’s argumentation. How would you rate the cogency of the author’s argument on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(7)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateCredibilityDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-credibility_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as successful in creating credibility if it conveys arguments and other information in a way that makes the author worthy of credence, e.g., by indicating the honesty of the author or by revealing the author’s knowledge or expertise regarding the discussed issue. It should be seen as not successful if rather the opposite holds. Decide in dubio pro reo, i.e., if you have no doubt about the author’s credibility, then do not judge him or her to be not credible. How would you rate the success of the author’s argumentation in creating credibility on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(14)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateEmotionalAppealDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-emotional-appeal_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as successful in making an emotional appeal if it conveys arguments and other information in a way that creates emotions, which make the target audience more open to the author’s arguments. It should be seen as not successful if rather the opposite holds. Notice that you should not judge about the persuasive effect of the author’s argumentation, but you should decide whether the argumentation makes the target audience willing/unwilling to be persuaded by the author (or to agree/disagree with the author) in principle—or neither. How would you rate the success of the author’s argumentation in making an emotional appeal on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(15)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateClarityDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-clarity_dagstuhl-15512-arg-quality',
            'The style of an argumentation should be seen as clear if it uses gramatically correct and widely unambiguous language as well as if it avoids unnecessary complexity and deviation from the discussed issue. The used language should make it easy for you to understand without doubts what the author argues for and how. How would you rate the clarity of the style of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(6)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateAppropriatenessDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-appropriateness_dagstuhl-15512-arg-quality',
            'The style of an argumentation should be seen as appropriate if the used language supports the creation of credibility and emotions as well as if it is proportional to the discussed issue. The choice of words and the grammatical complexity should, in your view, appear suitable for the discussed issue within the given setting (online debate forum on a given issue), matching with how credibility and emotions are created via the content of the argumentation. How would you rate the appropriateness of the style of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(4)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateArrangementDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-arrangement_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as well-arranged if it presents the given issue, the composed arguments, and its conclusion in the right order. Usually, the general issue and the particularly discussed topics should be clear before arguing and concluding about them. Notice, however, that other orderings may be used on purpose and may still be suitable to achieve persuasion. Besides, notice that, within the given setting (online debate forum on a given issue), some parts may be clear (e.g., the issue) and thus left implicit. How would you rate the arrangement of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(5)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateEffectivenessDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-effectiveness_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as effective if it achieves to persuade you of the author’s stance on the discussed issue or—in case you already agreed with the stance before—if it corroborates your agreement with the stance. Besides the actual arguments, also take into consideration the credibility and the emotional force of the argumentation. Decide in dubio pro reo, i.e., if you have no doubt about the correctness of the author’s arguments, then do not judge him or her to be not effective—unless you explicitly think that the arguments do not support the author’s stance. How would you rate the effectiveness of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(8)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateGlobalAcceptabilityDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-global-acceptability_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as globally acceptable if everyone from the expected target audience would accept both the consideration of the stated arguments within the discussion of the given issue and the way they are stated. Notice that you may see an argumentation as globally acceptable even though the stated arguments do not persuade you of the author’s stance. How would you rate the global acceptability of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(9)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateGlobalRelevanceDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-global-relevance_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as globally relevant if it contributes to the resolution of the given issue, i.e., if it provides arguments and/or other information that help to arrive at an ultimate conclusion regarding the discussed issue. You should be open to see an argumentation as relevant even if it does not your match your stance on the issue. Rather, the question is whether the provided arguments and information are worthy of being considered within the discussion of the issue. How would you rate the global relevance of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(10)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateGlobalSufficiencyDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-global-sufficiency_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as globally sufficient if it adequately rebuts those counter-arguments to its conclusion that can be anticipated. Notice that it is not generally clear which and how many counter-arguments can be anticipated. There may be cases where it is infeasible to rebut all such objections. Please judge about global sufficiency according to whether all main objections of an argumentation that you see are rebutted. How would you rate the global sufficiency of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(11)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateReasonablenessDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-reasonableness_dagstuhl-15512-arg-quality',
            'An argumentation should be seen as reasonable if it contributes to the resolution of the given issue in a sufficient way that is acceptable to everyone from the expected target audience. Try to adequately weight your judgments about global acceptability, global relevance, and global sufficiency when judging about reasonableness—unless there is a particular dimension among these that dominates your view of the author’s argumentation. In doubt, give more credit to global acceptability and global relevance than to global sufficiency due to the limited feasibility of the latter. How would you rate the reasonableness of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(12)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


class RateOverallQualityDagstuhl15512ArgQuality(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'rate-overall-quality_dagstuhl-15512-arg-quality',
            'How would you rate the overall quality of the author’s argumentation on the scale "1" (Low), "2" (Average) or "3" (High)?',
            ['dagstuhl-15512-arg-quality'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ad = get_parsed_data(2)
        for x in ad:
            argument = x[0]
            scores = x[1]
            label = majority_vote(scores)

            if label:
                instance = Instance(
                    input=argument,
                    output=label[0],
                    split=x[2]
                )
                self.instances.append(instance)


if __name__ == '__main__':
    task = RateOverallQualityDagstuhl15512ArgQuality()
    task.load_data()
    task = RateReasonablenessDagstuhl15512ArgQuality()
    task.load_data()
    task = RateGlobalSufficiencyDagstuhl15512ArgQuality()
    task.load_data()
    task = RateGlobalRelevanceDagstuhl15512ArgQuality()
    task.load_data()
    task = RateGlobalAcceptabilityDagstuhl15512ArgQuality()
    task.load_data()
    task = RateEffectivenessDagstuhl15512ArgQuality()
    task.load_data()
    task = RateArrangementDagstuhl15512ArgQuality()
    task.load_data()
    task = RateAppropriatenessDagstuhl15512ArgQuality()
    task.load_data()
    task = RateClarityDagstuhl15512ArgQuality()
    task.load_data()
    task = RateEmotionalAppealDagstuhl15512ArgQuality()
    task.load_data()
    task = RateCredibilityDagstuhl15512ArgQuality()
    task.load_data()
    task = RateCogencyDagstuhl15512ArgQuality()
    task.load_data()
    task = RateLocalSufficiencyDagstuhl15512ArgQuality()
    task.load_data()
    task = RateLocalRelevanceDagstuhl15512ArgQuality()
    task.load_data()
    task = RateLocalAcceptabilityDagstuhl15512ArgQuality()
    task.load_data()
    task = ClassifyingArgumentativenessDagstuhl15512ArgQuality()
    task.load_data()
