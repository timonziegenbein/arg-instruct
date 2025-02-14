import unittest

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

from samplers import BalancedInstanceSampler, InstructionSampler, SuperNiInstanceSampler, SuperNiInstructionSampler


class TaskTest(unittest.TestCase):

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

    def setUp(self):
        self.task_child_instances = [cls(from_cache=False) for cls in self._TASK_CHILD_CLASSES]

    def test_tasks(self):
        for task in self.task_child_instances:
            self.assertEqual(issubclass(type(task), Task), True, f'Task: {task.task_name} is not a subclass of Task')
            self.assertEqual(type(task.instances), list, f'Task: {task.task_name} instances is not a list')
            self.assertEqual(type(task.task_instruction), str,
                             f'Task: {task.task_name} task_instruction is not a string')
            self.assertEqual(type(task.task_name), str, f'Task: {task.task_name} task_name is not a string')
            self.assertEqual(type(task.dataset_names), list, f'Task: {task.task_name} dataset_names is not a list')
            self.assertEqual(type(task.is_clf), bool, f'Task: {task.task_name} is_clf is not a boolean')
            self.assertEqual(type(task.is_reg), bool, f'Task: {task.task_name} is_reg is not a boolean')

            if task.is_clf:
                self.assertEqual(task.is_reg, False, f'Task: {task.task_name} is_reg should be "False" is is_clf')
            if task.is_reg:
                self.assertEqual(task.is_clf, False, f'Task: {task.task_name} is_clf should be "False" is is_clf')

    def test_instances(self):
        for task in self.task_child_instances:
            for instance in task.instances:
                self.assertEqual(type(instance), Instance, f'Task: {task.task_name} instance is not of type Instance')
                self.assertEqual(((type(instance.input) == str) or (instance.input is None)), True,
                                 f'Task: {task.task_name} instance input is not a string or None')
                self.assertEqual(type(instance.output), str, f'Task: {task.task_name} instance output is not a string')
                self.assertEqual(instance.split in ['train', 'dev', 'test'], True,
                                 f'Task: {task.task_name} instance split is not in [train, dev, test]')
                self.assertEqual(task.task_name, instance.task_instruction,
                                 f'Task: {task.task_instruction} instance task_name is not the same as the task_name')
                self.assertEqual(task.task_instruction, instance.task_instruction,
                                 f'Task: {task.task_name} instance task_instruction is not the same as the task_instruction')
                self.assertEqual(type(instance.apply_template()), str,
                                 f'Task: {task.task_name} apply_template is not a string')

    def test_get_batch(self):
        for task in self.task_child_instances:
            previous_batch = None
            for split in ['train', 'dev', 'test']:
                for batch in task.get_batch(split):
                    if batch:
                        if previous_batch is None:
                            previous_batch = batch
                        else:
                            # check if items in the batch are the different from the previous get_batch
                            for i in range(len(batch)):
                                self.assertNotEqual(
                                    batch[i], previous_batch[i], f'Task: {task.task_name}, Split: {split} get_batch is returning the same batch')
                        self.assertEqual(type(batch), list,
                                         f'Task: {task.task_name}, Split: {split} get_batch is not returning a list')
                        self.assertEqual(len(batch), len(
                            previous_batch), f'Task: {task.task_name}, Split: {split} get_batch is returning a batch of different length')
                        # check if split of the instances in the batch are the same
                        for instance in batch:
                            self.assertEqual(
                                instance.split, split, f'Task: {task.task_name}, Split: {split} get_batch is returning instances with different splits')


class InstanceSamplerTest(unittest.TestCase):

    def setUp(self):
        self.instance_sampler = BalancedInstanceSampler()

    def test_sampler(self):
        for split in ['train', 'dev', 'test']:
            for num_batches in [1, 10]:
                for batch in self.instance_sampler.get_batch(split, num_batches):
                    self.assertEqual(type(batch), list, f'Sampler get_batch is not returning a list')
                    for instance in batch:
                        self.assertEqual(type(instance), Instance,
                                         f'Sampler get_batch is not returning instances of type Instance')
                        self.assertEqual(instance.split, split,
                                         f'Sampler get_batch is not returning instances with split train')
                    # check whether the instances are balanced in terms of their task_name
                    task_names = [instance.task_instruction for instance in batch]
                    task_name_counts = {task_name: task_names.count(task_name) for task_name in task_names}
                    # we allow a difference of 3
                    self.assertEqual(max(task_name_counts.values()) - min(task_name_counts.values()) <= 3, True,
                                     f'Sampler get_batch is not returning balanced instances: {max(task_name_counts.values())} - {min(task_name_counts.values())}; split: {split}')


'''class SuperNiInstanceSamplerTest(unittest.TestCase):

    def setUp(self):
        self.superni_instance_sampler = SuperNiInstanceSampler()

    def test_sampler(self):
        tasks = self.superni_instance_sampler.get_all()
        self.assertEqual(type(tasks), list, f'Sampler get_all is not returning a list')
        self.assertNotEqual(tasks, [], f'Sampler get_all is returning an empty list')
        for task in tasks:
            self.assertEqual(issubclass(type(task), Task), True, f'Task: {task.task_name} is not a subclass of Task')
            self.assertEqual(type(task.instances), list, f'Task: {task.task_name} instances is not a list')
            self.assertEqual(type(task.id), str, f'Task: {task.task_name} id is not a string')
            self.assertEqual(type(task.task_instruction), str,
                             f'Task: {task.task_name} task_instruction is not a string')
            self.assertEqual(type(task.task_name), str, f'Task: {task.task_name} task_name is not a string')
            self.assertEqual(type(task.dataset_names), list, f'Task: {task.task_name} dataset_names is not a list')
            self.assertEqual(type(task.is_clf), bool, f'Task: {task.task_name} is_clf is not a boolean')
            self.assertEqual(type(task.is_reg), bool, f'Task: {task.task_name} is_reg is not a boolean')

            if task.is_clf:
                self.assertEqual(task.is_reg, False, f'Task: {task.task_name} is_reg should be "False" is is_clf')
            if task.is_reg:
                self.assertEqual(task.is_clf, False, f'Task: {task.task_name} is_clf should be "False" is is_clf')'''


class InstructionSamplerTest(unittest.TestCase):

    def setUp(self):
        self.instruction_sampler = InstructionSampler()

    def test_sampler(self):
        instructions = self.instruction_sampler.get_all()
        self.assertEqual(type(instructions), list, f'Sampler get_all is not returning a list')
        self.assertNotEqual(instructions, [], f'Sampler get_all is returning an empty list')
        for instruction in instructions:
            self.assertEqual(type(instruction), str, f'Sampler get_all is not returning instances of type string')

            
            
class SuperNiInstructionSamplerTest(unittest.TestCase):

    def setUp(self):
        self.superni_instruction_sampler = SuperNiInstructionSampler()

    def test_sampler(self):
        instructions = self.superni_instruction_sampler.get_all()
        self.assertEqual(type(instructions), list, f'Sampler get_all is not returning a list')
        self.assertNotEqual(instructions, [], f'Sampler get_all is returning an empty list')
        for instruction in instructions:
            self.assertEqual(type(instruction), str, f'Sampler get_all is not returning instances of type string')


if __name__ == '__main__':
    unittest.main(verbosity=2)
