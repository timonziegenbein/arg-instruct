import numpy as np
import multiprocess
import os
import json

from task import Task, Instance

from tasks.qt30 import PropositionalRelationsIdentificationQT30, IllocutionaryRelationsIdentificationQT30
from tasks.argkp import KeyPointMatchingArgKP, KeyPointGenerationArgKP
from tasks.f1000rd import PragmaticTaggingF1000rd
from tasks.webis_sameside_19 import SameSideStanceClassificationWebisSameside19
from tasks.argument_reasoning_comprehension import ArgumentReasoningComprehension
from tasks.argsvalidnovel import (
    NoveltyClassificationArgsvalidnovel,
    ValidityClassificationArgsvalidnovel,
    RealtiveNoveltyClassificationArgsvalidnovel,
    RealtiveValidityClassificationArgsvalidnovel,
)
#from tasks.school_student_essays import (
#    DiscourseFunctionTaggingSchoolStudentEssays,
#    ArgumentTaggingSchoolStudentEssays,
#    ComponentTaggingSchoolStudentEssays,
#    DiscourseModeTaggingSchoolStudentEssays,
#    RelevanceScoringSchoolStudentEssays,
#    ContentScoringSchoolStudentEssays,
#    StructureScoringSchoolStudentEssays,
#    StyleScoringSchoolStudentEssays,
#    ScoringSchoolStudentEssays
#)
from tasks.appropriateness_corpus import (
    InappropriatenessDetectionAppropriatenessCorpus,
    ToxicEmotionsDetectionAppropriatenessCorpus,
    MissingCommitmentDetectionAppropriatenessCorpus,
    MissingIntelligibilityDetectionAppropriatenessCorpus,
    OtherInappropriatenessDetectionAppropriatenessCorpus,
    ExcessiveIntensityDetectionAppropriatenessCorpus,
    EmotionalDeceptionDetectionAppropriatenessCorpus,
    MissingSeriousnessDetectionAppropriatenessCorpus,
    MissingOpennessDetectionAppropriatenessCorpus,
    UnclearMeaningDetectionAppropriatenessCorpus,
    MissingRelevanceDetectionAppropriatenessCorpus,
    ConfusingReasoningDetectionAppropriatenessCorpus,
    DetrimentalOrthographyDetectionAppropriatenessCorpus,
    ReasonUnclassifiedDetectionAppropriatenessCorpus
)
from tasks.argument_annotated_essays_2 import (
    IdentifyingArgumentComponentsArgumentAnnotatedEssays2,
    ClassifyingArgumentComponentsArgumentAnnotatedEssays2,
    IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2,
    StanceRecognitionArgumentAnnotatedEssays2
)
from tasks.dagstuhl_15512_argquality import (
    #ClassifyingArgumentativenessDagstuhl15512ArgQuality,
    RateLocalAcceptabilityDagstuhl15512ArgQuality,
    RateLocalRelevanceDagstuhl15512ArgQuality,
    RateLocalSufficiencyDagstuhl15512ArgQuality,
    RateCogencyDagstuhl15512ArgQuality,
    RateCredibilityDagstuhl15512ArgQuality,
    RateEmotionalAppealDagstuhl15512ArgQuality,
    RateClarityDagstuhl15512ArgQuality,
    RateAppropriatenessDagstuhl15512ArgQuality,
    RateArrangementDagstuhl15512ArgQuality,
    RateEffectivenessDagstuhl15512ArgQuality,
    RateGlobalAcceptabilityDagstuhl15512ArgQuality,
    RateGlobalRelevanceDagstuhl15512ArgQuality,
    RateGlobalSufficiencyDagstuhl15512ArgQuality,
    RateReasonablenessDagstuhl15512ArgQuality,
    RateOverallQualityDagstuhl15512ArgQuality
)
from tasks.enthymemes_student_essays import (
    DetectEnthymemesEnthymemesStudentEssays,
    ReconstructEnthymemesEnthymemesStudentEssays
)
from tasks.icle_argument_strength import ClassifyingArgumentStrengthIcleArgumentStrength
from tasks.arg_microtexts_2 import (
    CreateArgumentativeTextMicrotextsV2,
    ExtractCentralClaimMicrotextsV2,
    DetermineArgumentativeRoleMicrotextsV2,
    DetermineFunctionOfSegmentMicrotextsV2,
    IdentifyUnitAttachmentMicrotextsV2
)
from tasks.belief_based_arguments import (
    StancePredictionBeliefBasedArguments,
    BeliefBasedClaimGenerationBeliefBasedArguments
)
from tasks.echr import (
    ArgumentClauseRecognitionEchr,
    ClauseRelationPredictionEchr,
    PremiseRecognitionEchr,
    ConclusionRecognitionEchr
)
from tasks.reason_identification_and_classification import ReasonIdentificationReasonIdentificationAndClassification
from tasks.webis_conclugen_21 import ConclusionGenerationWebisConclugen21
from tasks.ibm_rank_30k import (
    QualityAssessmentIbmRank30k,
    StancePredictionIbmRank30k
)
from tasks.debate_sum import ExtractiveSummarizationDebateSum
from tasks.amazon_review_dataset import (
    ReviewHelpfulnessPredictionAmazonReviewDataset,
    UnitSegmentationAmazonReviewDataset,
    UnitClassificationAmazonReviewDataset,
    RelationDetectionAmazonReviewDataset
)
from tasks.ukp_convarg_2 import (
    ClassifyMoreConvincingArgumentUKPConvArg2,
    ClassifyMoreDetailsArgumentUKPConvArg2,
    ClassifyMoreBalancedArgumentUKPConvArg2,
    ClassifyMoreCredibleArgumentUKPConvArg2,
    #ClassifyMoreTopicSpecificArgumentUKPConvArg2,
    ClassifyMoreClearArgumentUKPConvArg2,
    ClassifyMoreOnTopicArgumentUKPConvArg2,
    ClassifyMoreProvokingArgumentUKPConvArg2,
    ClassifyMoreSmartArgumentUKPConvArg2,
    ClassifyLessAttackingArgumentUKPConvArg2,
    ClassifyLessLanguageIssuesArgumentUKPConvArg2,
    ClassifyLessUnclearArgumentUKPConvArg2,
    ClassifyLessFactsArgumentUKPConvArg2,
    ClassifyLessReasoningArgumentUKPConvArg2,
    ClassifyLessRelevantReasonsArgumentUKPConvArg2,
    ClassifyNotAnArgumentUKPConvArg2,
    ClassifyNonSenseArgumentUKPConvArg2,
    ClassifyOffTopicArgumentUKPConvArg2,
    ClassifyGenerallyWeakArgumentUKPConvArg2
)
from tasks.argument_annotated_user_generated_web_discourse import (
    DetectPersuasiveDocumentsAAUGWD,
    ExtractToulminComponentsAAUGWD
)
from tasks.comarg import StanceDetectionComarg
from tasks.ukp_aspect_corpus import ArgumentSimilarityUKPAspectCorpus
from tasks.upk_sentential_argument_mining import ArgumentIdentificationUKPSententialArgumentMining
from tasks.arguana_counterargs_corpus import (
    SameDebateOpposingCountersArguanaCounterargsCorpus,
    SameDebateCountersArguanaCounterargsCorpus,
    SameDebateOpposingArgumentsArguanaCounterargsCorpus,
    SameDebateArgumentsArguanaCounterargsCorpus
)
from tasks.iac_2 import (
    PredictAgreementIacV2,
    PredictRespectIacV2,
    PredictFactualityIacV2,
    PredictNiceIacV2,
    PredictSarcasmIacV2
)
from tasks.argumentation_synthesis import SynthesizeArgumentArgumentationSynthesis
from tasks.claim_revision_pair_corpus import (
    ClaimRevisionImprovementClaimRevisionPairCorpus,
    SubotimalClaimDetectionClaimRevisionPairCorpus,
    ClaimImprovementSuggestionsClaimRevisionPairCorpus,
    ClaimOptimizationClaimRevisionPairCorpus
)
from tasks.aspect_controlled_argument_generation import AspectControlledArgumentGenerationAspectControlledArgumentGeneration
from tasks.superni import SuperNiTask, superni_path, convert_task_name, ADDITIONAL_CLF_TASKNAMES, CLF_TASKNAMES

_TASK_CHILD_CLASSES = [
    PropositionalRelationsIdentificationQT30,
    IllocutionaryRelationsIdentificationQT30,
    ArgumentReasoningComprehension,
    InappropriatenessDetectionAppropriatenessCorpus,
    ToxicEmotionsDetectionAppropriatenessCorpus,
    MissingCommitmentDetectionAppropriatenessCorpus,
    MissingIntelligibilityDetectionAppropriatenessCorpus,
    OtherInappropriatenessDetectionAppropriatenessCorpus,
    ExcessiveIntensityDetectionAppropriatenessCorpus,
    EmotionalDeceptionDetectionAppropriatenessCorpus,
    MissingSeriousnessDetectionAppropriatenessCorpus,
    MissingOpennessDetectionAppropriatenessCorpus,
    UnclearMeaningDetectionAppropriatenessCorpus,
    MissingRelevanceDetectionAppropriatenessCorpus,
    ConfusingReasoningDetectionAppropriatenessCorpus,
    DetrimentalOrthographyDetectionAppropriatenessCorpus,
    ReasonUnclassifiedDetectionAppropriatenessCorpus,
    KeyPointMatchingArgKP,
    KeyPointGenerationArgKP,
    PragmaticTaggingF1000rd,
    SameSideStanceClassificationWebisSameside19,
    NoveltyClassificationArgsvalidnovel,
    ValidityClassificationArgsvalidnovel,
    RealtiveNoveltyClassificationArgsvalidnovel,
    RealtiveValidityClassificationArgsvalidnovel,
    #DiscourseFunctionTaggingSchoolStudentEssays,
    #ArgumentTaggingSchoolStudentEssays,
    #ComponentTaggingSchoolStudentEssays,
    #DiscourseModeTaggingSchoolStudentEssays,
    #RelevanceScoringSchoolStudentEssays,
    #ContentScoringSchoolStudentEssays,
    #StructureScoringSchoolStudentEssays,
    #StyleScoringSchoolStudentEssays,
    #ScoringSchoolStudentEssays,
    IdentifyingArgumentComponentsArgumentAnnotatedEssays2,
    ClassifyingArgumentComponentsArgumentAnnotatedEssays2,
    IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2,
    StanceRecognitionArgumentAnnotatedEssays2,
    #ClassifyingArgumentativenessDagstuhl15512ArgQuality,
    RateLocalAcceptabilityDagstuhl15512ArgQuality,
    RateLocalRelevanceDagstuhl15512ArgQuality,
    RateLocalSufficiencyDagstuhl15512ArgQuality,
    RateCogencyDagstuhl15512ArgQuality,
    RateCredibilityDagstuhl15512ArgQuality,
    RateEmotionalAppealDagstuhl15512ArgQuality,
    RateClarityDagstuhl15512ArgQuality,
    RateAppropriatenessDagstuhl15512ArgQuality,
    RateArrangementDagstuhl15512ArgQuality,
    RateEffectivenessDagstuhl15512ArgQuality,
    RateGlobalAcceptabilityDagstuhl15512ArgQuality,
    RateGlobalRelevanceDagstuhl15512ArgQuality,
    RateGlobalSufficiencyDagstuhl15512ArgQuality,
    RateReasonablenessDagstuhl15512ArgQuality,
    RateOverallQualityDagstuhl15512ArgQuality,
    DetectEnthymemesEnthymemesStudentEssays,
    ReconstructEnthymemesEnthymemesStudentEssays,
    ClassifyingArgumentStrengthIcleArgumentStrength,
    CreateArgumentativeTextMicrotextsV2,
    ExtractCentralClaimMicrotextsV2,
    DetermineArgumentativeRoleMicrotextsV2,
    DetermineFunctionOfSegmentMicrotextsV2,
    IdentifyUnitAttachmentMicrotextsV2,
    StancePredictionBeliefBasedArguments,
    BeliefBasedClaimGenerationBeliefBasedArguments,
    ArgumentClauseRecognitionEchr,
    ClauseRelationPredictionEchr,
    PremiseRecognitionEchr,
    ConclusionRecognitionEchr,
    ReasonIdentificationReasonIdentificationAndClassification,
    ConclusionGenerationWebisConclugen21,
    QualityAssessmentIbmRank30k,
    StancePredictionIbmRank30k,
    ExtractiveSummarizationDebateSum,
    ReviewHelpfulnessPredictionAmazonReviewDataset,
    UnitSegmentationAmazonReviewDataset,
    UnitClassificationAmazonReviewDataset,
    RelationDetectionAmazonReviewDataset,
    ClassifyMoreConvincingArgumentUKPConvArg2,
    ClassifyMoreDetailsArgumentUKPConvArg2,
    ClassifyMoreBalancedArgumentUKPConvArg2,
    ClassifyMoreCredibleArgumentUKPConvArg2,
    #ClassifyMoreTopicSpecificArgumentUKPConvArg2,
    ClassifyMoreClearArgumentUKPConvArg2,
    ClassifyMoreOnTopicArgumentUKPConvArg2,
    ClassifyMoreProvokingArgumentUKPConvArg2,
    ClassifyMoreSmartArgumentUKPConvArg2,
    ClassifyLessAttackingArgumentUKPConvArg2,
    ClassifyLessLanguageIssuesArgumentUKPConvArg2,
    ClassifyLessUnclearArgumentUKPConvArg2,
    ClassifyLessFactsArgumentUKPConvArg2,
    ClassifyLessReasoningArgumentUKPConvArg2,
    ClassifyLessRelevantReasonsArgumentUKPConvArg2,
    ClassifyNotAnArgumentUKPConvArg2,
    ClassifyNonSenseArgumentUKPConvArg2,
    ClassifyOffTopicArgumentUKPConvArg2,
    ClassifyGenerallyWeakArgumentUKPConvArg2,
    DetectPersuasiveDocumentsAAUGWD,
    ExtractToulminComponentsAAUGWD,
    StanceDetectionComarg,
    ArgumentSimilarityUKPAspectCorpus,
    ArgumentIdentificationUKPSententialArgumentMining,
    SameDebateOpposingCountersArguanaCounterargsCorpus,
    SameDebateCountersArguanaCounterargsCorpus,
    SameDebateOpposingArgumentsArguanaCounterargsCorpus,
    SameDebateArgumentsArguanaCounterargsCorpus,
    PredictAgreementIacV2,
    PredictRespectIacV2,
    PredictFactualityIacV2,
    PredictNiceIacV2,
    PredictSarcasmIacV2,
    SynthesizeArgumentArgumentationSynthesis,
    ClaimRevisionImprovementClaimRevisionPairCorpus,
    SubotimalClaimDetectionClaimRevisionPairCorpus,
    ClaimImprovementSuggestionsClaimRevisionPairCorpus,
    ClaimOptimizationClaimRevisionPairCorpus,
    AspectControlledArgumentGenerationAspectControlledArgumentGeneration
]


TEST_TASKS_NAMES = [
    'argument-annotated-essays-2', 'qt30', 'f1000rd', 'iac-v2', 'ibm-rank-30k', 'arguana-counterargs-corpus', 'aspect-controlled-argument-generation', 'debate-sum', 'webis-conclugen-21'
]


class InstanceSampler():

    def __init__(self):
        try:
            cpus = multiprocess.cpu_count()
        except NotImplementedError:
            cpus = 1   # arbitrary default
        cpus = 1
        cpus = min(len(_TASK_CHILD_CLASSES), cpus)
        pool = multiprocess.Pool(cpus)
        self.task_child_instances = pool.map(self._init_task_child_instances, _TASK_CHILD_CLASSES)

    def _init_task_child_instances(self, cls):
        return cls()

    def get_batch(self, split, batch_size, num_batches, balanced=False, shuffle=False, early_stopping=False):
        for task in self.task_child_instances:
            i = 0
            for instances in task.get_batch(split, batch_size, balanced=balanced, shuffle=shuffle, early_stopping=early_stopping):
                i += 1
                if i > num_batches:
                    break
                else:
                    yield instances


class InstanceGenerationSampler():

    def __init__(self):
        try:
            cpus = multiprocess.cpu_count()
        except NotImplementedError:
            cpus = 1   # arbitrary default
        cpus = min(len(_TASK_CHILD_CLASSES), cpus)
        pool = multiprocess.Pool(cpus)
        self.task_child_instances = pool.map(self._init_task_child_instances, _TASK_CHILD_CLASSES)
        self.task_child_instances = [x for x in self.task_child_instances if x.dataset_names[0] not in TEST_TASKS_NAMES]

    def _init_task_child_instances(self, cls):
        return cls()

    def get_batch(self, split, balanced=True, shuffle=True):
        # get a single batch from each task and store the length of the batches in a dictionary
        batch_dict = {}
        for task in self.task_child_instances:
            if task.is_clf:
                for batch in task.get_batch(split, batch_size=1, balanced=balanced, shuffle=shuffle, early_stopping=True):
                    if batch:
                        batch_dict[task.task_name] = len(batch)
                        break
            else:
                for batch in task.get_batch(split, batch_size=3, balanced=balanced, shuffle=shuffle, early_stopping=True):
                    if batch:
                        batch_dict[task.task_name] = len(batch)
                        break


        # create generators for each task
        generators = [(task, task.get_batch(split, batch_size=3, balanced=balanced, shuffle=shuffle, early_stopping=True))
                      for task in self.task_child_instances if task.task_name in batch_dict]
        # as for loop
        generators = []
        for task in self.task_child_instances:
            if task.is_clf:
                generators.append((task, task.get_batch(split, batch_size=1, balanced=balanced, shuffle=shuffle, early_stopping=True)))
            else:
                generators.append((task, task.get_batch(split, batch_size=3, balanced=balanced, shuffle=shuffle, early_stopping=True)))

        # get the batches
        while True:
            batch = []
            for j, generator in enumerate(generators):
                mini_batch = next(generator[1], None)
                if mini_batch:
                    batch.append(mini_batch)
                else:
                    # reset generator
                    if generator[0].is_clf:
                        generators[j] = (generator[0], generator[0].get_batch(split, batch_size=1, balanced=balanced, shuffle=shuffle, early_stopping=True))
                    else:
                        generators[j] = (generator[0], generator[0].get_batch(split, batch_size=3, balanced=balanced, shuffle=shuffle, early_stopping=True))

            # flatten the batch
            flat_batch = [item for sublist in batch for item in sublist]
            yield flat_batch


class BalancedInstanceSampler():

    def __init__(self):
        try:
            cpus = multiprocess.cpu_count()
        except NotImplementedError:
            cpus = 1   # arbitrary default
        cpus = min(len(_TASK_CHILD_CLASSES), cpus)
        pool = multiprocess.Pool(cpus)
        self.task_child_instances = pool.map(self._init_task_child_instances, _TASK_CHILD_CLASSES)

    def _init_task_child_instances(self, cls):
        return cls()

    def get_batch(self, split, batch_size, num_batches, balanced=True, shuffle=True):
        # get a single batch from each task and store the length of the batches in a dictionary
        batch_dict = {}
        for task in self.task_child_instances:
            for batch in task.get_batch(split, batch_size=1, balanced=balanced, shuffle=shuffle, early_stopping=True):
                if batch:
                    batch_dict[task.task_name] = len(batch)
                    break

        # get the maximum batch length
        max_batch_length = max(batch_dict.values())
        # replace values with how many batches are needed to reach approximately the same length at the maximum batch max_batch_length
        for task in self.task_child_instances:
            if task.task_name in batch_dict:
                batch_dict[task.task_name] = max_batch_length // batch_dict[task.task_name]

        # create generators for each task
        generators = [(task, task.get_batch(split, batch_size=1, balanced=balanced, shuffle=shuffle, early_stopping=True))
                      for task in self.task_child_instances if task.task_name in batch_dict]

        # get the batches
        for i in range(num_batches):
            batch = []
            for j, generator in enumerate(generators):
                for _ in range(batch_dict[generator[0].task_name]):
                    mini_batch = next(generator[1], None)
                    if mini_batch:
                        batch.append(mini_batch)
                    else:
                        # reset generator
                        generators[j] = (generator[0], generator[0].get_batch(split, batch_size=1, balanced=balanced, shuffle=shuffle, early_stopping=True))

            # flatten the batch
            flat_batch = [item for sublist in batch for item in sublist]
            yield flat_batch


class SuperNiInstanceSampler():

    def __init__(self):
        tasks = []
        for task_path in [x for x in os.listdir(superni_path) if x.endswith('.json')]:
            with open(os.path.join(superni_path, task_path)) as f:
                task_data = json.load(f)
            task_name = convert_task_name(task_path)
            TaskClass = type(task_name, (SuperNiTask,),
                             {
                "__init__": SuperNiTask.__init__,
                "load_data": SuperNiTask.load_data
            }
            )

            # English tasks only
            if task_data['Instruction_language'] == ['English'] and task_data['Input_language']  == ['English'] and task_data['Output_language'] == ['English']:
                is_clf = False
                is_reg = False
                #if any(substr in task_path for substr in ['classif', 'binary']) or task_name in ADDITIONAL_CLF_TASKNAMES:
                if task_name in CLF_TASKNAMES: # only complete for test tasks
                    is_clf = True
                elif any(substr in task_path for substr in ['regression', 'count']): # none in test tasks
                    is_reg = True
                tasks.append(TaskClass(
                    task_name=task_name,
                    task_instruction=task_data['Definition'][0],
                    dataset_names=[task_path.split('_')[1]],
                    data_path=task_path,
                    is_clf=is_clf,
                    is_reg=is_reg,
                ))
        self.task_child_instances = tasks

                    
    def get_batch(self, split, batch_size, num_batches, balanced=True, shuffle=True, early_stopping=False):
        for task in self.task_child_instances:
            i = 0
            for instances in task.get_batch(split, batch_size, balanced=balanced, shuffle=shuffle, early_stopping=early_stopping):
                i += 1
                if i > num_batches:
                    break
                else:
                    yield instances

class SuperNiInstructionSampler():

    def __init__(self):
        tasks = []
        for task_path in [x for x in os.listdir(superni_path) if x.endswith('.json')]:
            with open(os.path.join(superni_path, task_path)) as f:
                task_data = json.load(f)
            task_name = convert_task_name(task_path)
            TaskClass = type(task_name, (SuperNiTask,),
                             {
                "__init__": SuperNiTask.__init__,
                "load_data": SuperNiTask.load_data
            }
            )

            # English tasks only
            if task_data['Instruction_language'] == ['English'] and task_data['Input_language']  == ['English'] and task_data['Output_language'] == ['English']:
                is_clf = False
                is_reg = False
                #if any(substr in task_path for substr in ['classif', 'binary']) or task_name in ADDITIONAL_CLF_TASKNAMES:
                if task_name in CLF_TASKNAMES: # only complete for test tasks
                    is_clf = True
                elif any(substr in task_path for substr in ['regression', 'count']): # none in test tasks
                    is_reg = True
                tasks.append(TaskClass(
                    task_name=task_name,
                    task_instruction=task_data['Definition'][0],
                    dataset_names=[task_path.split('_')[1]],
                    data_path=task_path,
                    is_clf=is_clf,
                    is_reg=is_reg,
                    load_data=False
            ))
        self.task_child_instances = tasks

    def get_all(self):
        self.task_child_instances = np.random.permutation(self.task_child_instances)
        return [x.task_instruction for x in self.task_child_instances]


class InstructionSampler():

    def __init__(self):
        '''try:
            cpus = multiprocess.cpu_count()
        except NotImplementedError:
            cpus = 1   # arbitrary default
        cpus = min(len(_TASK_CHILD_CLASSES), cpus)
        pool = multiprocess.Pool(cpus)
        self.task_child_instances = pool.map(self._init_task_child_instances, _TASK_CHILD_CLASSES)'''
        self.task_child_instances = [self._init_task_child_instances(t) for t in _TASK_CHILD_CLASSES]
        print('Loaded {} tasks'.format(len(self.task_child_instances)))
        np.random.seed(24)

    def _init_task_child_instances(self, cls):
        return cls()

    def get_all(self, split=None):
        self.task_child_instances = np.random.permutation(self.task_child_instances)
        if split == 'train':
            return [x.task_instruction for x in self.task_child_instances if x.dataset_names[0] not in TEST_TASKS_NAMES]
        elif split == 'test':
            return [x.task_instruction for x in self.task_child_instances if x.dataset_names[0] in TEST_TASKS_NAMES]
        else:
            return [x.task_instruction for x in self.task_child_instances]

    def get_all_by_type(self, is_clf=False, is_reg=False, is_gen=False, split=None):
        self.task_child_instances = np.random.permutation(self.task_child_instances)
        if is_reg:
            type_tasks = [x for x in self.task_child_instances if x.is_reg]
        elif is_clf:
            type_tasks = [x for x in self.task_child_instances if x.is_clf]
        elif is_gen:
            type_tasks = [x for x in self.task_child_instances if not x.is_clf and not x.is_reg]
                          
        if split == 'train':
            return [x.task_instruction for x in type_tasks if x.dataset_names[0] not in TEST_TASKS_NAMES]
        elif split == 'test':
            return [x.task_instruction for x in type_tasks if x.dataset_names[0] in TEST_TASKS_NAMES]
        else:
            return [x.task_instruction for x in type_tasks]


if __name__ == '__main__':
    sampler = InstanceSampler().get_batch('train', 1, 1, balanced=True, shuffle=True)
    instances = next(sampler)
    # group instances by task
    instances_by_task = {}
    for instance in instances:
        if instance.task_instruction not in instances_by_task:
            instances_by_task[instance.task_instruction] = []
        instances_by_task[instance.task_instruction].append(instance)
    # print task_instructions and all instance outputs
    for task_instruction, task_instances in instances_by_task.items():
        print(task_instances[0].task_instruction)
        for instance in task_instances:
            print(instance.output)
        print()
