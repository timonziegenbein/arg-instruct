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


import multiprocess

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

if __name__ == '__main__':
    try:
        cpus = multiprocess.cpu_count()
    except NotImplementedError:
        cpus = 1   # arbitrary default
    cpus = min(114, cpus)
    print(f'Using {cpus} CPUs')
    pool = multiprocess.Pool(cpus)

    def _init_task_child_instances(cls):
        return cls(from_cache=False)
    task_child_instances = pool.map(_init_task_child_instances, _TASK_CHILD_CLASSES)

    for task_name in [task.task_name for task in task_child_instances]:
        print(task_name)
