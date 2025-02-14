import numpy as np
import pandas as pd
import os
import sys
from xml.etree import ElementTree as ET
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']


EXAMPLE_XML = '''
    <?xml version="1.0" encoding="UTF-8"?>
    <xmi:XMI xmlns:type3="http:///de/tudarmstadt/ukp/dkpro/core/api/frequency/tfidf/type.ecore" xmlns:cas="http:///uima/cas.ecore" xmlns:type4="http:///de/tudarmstadt/ukp/dkpro/core/api/metadata/type.ecore" xmlns:dependency="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/dependency.ecore" xmlns:type8="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type.ecore" xmlns:type9="http:///de/tudarmstadt/ukp/dkpro/core/type.ecore" xmlns:type5="http:///de/tudarmstadt/ukp/dkpro/core/api/ner/type.ecore" xmlns:type6="http:///de/tudarmstadt/ukp/dkpro/core/api/segmentation/type.ecore" xmlns:tcas="http:///uima/tcas.ecore" xmlns:pathos="http:///de/tudarmstadt/ukp/dkpro/argumentation/types/pathos.ecore" xmlns:tweet="http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/pos/tweet.ecore" xmlns:chunk="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/chunk.ecore" xmlns:type7="http:///de/tudarmstadt/ukp/dkpro/core/api/semantics/type.ecore" xmlns:xmi="http://www.omg.org/XMI" xmlns:type2="http:///de/tudarmstadt/ukp/dkpro/core/api/coref/type.ecore" xmlns:toulmin="http:///de/tudarmstadt/ukp/dkpro/argumentation/types/toulmin.ecore" xmlns:morph="http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/morph.ecore" xmlns:types="http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore" xmlns:constituent="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/constituent.ecore" xmlns:type="http:///de/tudarmstadt/ukp/dkpro/core/api/anomaly/type.ecore" xmlns:pos="http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/pos.ecore" xmi:version="2.0">
        <cas:NULL xmi:id="0"/>
        <cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView" mimeType="text" sofaString="You don't need to go to the US to find reports on how home educated children do. Try looking at the research from Paula Rothermel for example, which can be found via her website. "/>
        <type4:DocumentMetaData xmi:id="13" sofa="1" begin="0" end="179" language="en" documentId="1025" isLastSegment="false"/>
        <types:WebArgumentMetadata xmi:id="24" sofa="1" begin="0" end="0" author="jaxb" date="07 November 2009 1:02pm" docType="artcomment" origUrl="http://discussion.theguardian.com/comment-permalink/6754065" topic="homeschooling" thumbsUp="5" thumbsDown="0" origId="1025" notes="" title="Ridiculous rules for home schools"/>
        <type6:Paragraph xmi:id="39" sofa="1" begin="0" end="179"/>
        <types:PersuasivenessAnnotationMetaData xmi:id="43" sofa="1" begin="0" end="0" annotator="annotator1" isPersuasive="false" labelDetailed="P21" isGold="false" annotationBatchName="batch2-690documents" conflictResolvingAnnotation="false"/>
        <types:PersuasivenessAnnotationMetaData xmi:id="55" sofa="1" begin="0" end="0" annotator="annotator2" isPersuasive="false" labelDetailed="P21" isGold="false" comment="nothing persuasive " annotationBatchName="batch2-690documents" conflictResolvingAnnotation="false"/>
        <types:PersuasivenessAnnotationMetaData xmi:id="67" sofa="1" begin="0" end="0" isPersuasive="false" isGold="true" annotationBatchName="batch2-690documents" conflictResolvingAnnotation="false"/>
        <cas:View sofa="1" members="13 24 39 43 55 67"/>
    </xmi:XMI>
'''


class DetectPersuasiveDocumentsAAUGWD(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='detect-persuasive-documents_aaugwd',
            task_instruction='Distinguish, whether the comment is Persuasive regarding the discussed topic or not (Not Persusasive). The key question to answer is: Does the author intend to convince us clearly about his/her attitude or opinion towards the topic?',
            dataset_names=['aaugwd'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/habernal.gurevych.2015.argumentation.mining.CL.data/data/gold.data.persuasive'
        files = os.listdir(ds_path)
        np.random.seed(42)
        np.random.shuffle(files)
        train_files = files[:int(len(files)*0.7)]
        dev_files = files[int(len(files)*0.7):int(len(files)*0.8)]
        test_files = files[int(len(files)*0.8):]

        # read all .xmi files
        for root, _, files in os.walk(ds_path):
            for file in files:
                if file.endswith('.xmi'):
                    with open(os.path.join(root, file), 'r') as f:
                        xmi = f.read()
                        f.close()
                    xml_root = ET.fromstring(xmi)
                    for child in xml_root:
                        if child.tag == '{http:///uima/cas.ecore}Sofa':
                            input = child.attrib['sofaString']
                        elif child.tag == '{http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore}PersuasivenessAnnotationMetaData' and 'annotator' in child.attrib:
                            output = child.attrib['isPersuasive']
                    instance = Instance(
                        input=input,
                        output='Persuasive' if output == 'true' else 'Not Persuasive',
                        split='train' if file in train_files else 'dev' if file in dev_files else 'test'
                    )
                    self.instances.append(instance)


EXAMPLE_TOULMIN_XMI = '''
    <?xml version="1.0" encoding="UTF-8"?>
    <xmi:XMI xmlns:cas="http:///uima/cas.ecore" xmlns:type2="http:///de/tudarmstadt/ukp/dkpro/core/api/metadata/type.ecore" xmlns:dependency="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/dependency.ecore" xmlns:type6="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type.ecore" xmlns:type3="http:///de/tudarmstadt/ukp/dkpro/core/api/ner/type.ecore" xmlns:type4="http:///de/tudarmstadt/ukp/dkpro/core/api/segmentation/type.ecore" xmlns:tcas="http:///uima/tcas.ecore" xmlns:tweet="http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/pos/tweet.ecore" xmlns:chunk="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/chunk.ecore" xmlns:type5="http:///de/tudarmstadt/ukp/dkpro/core/api/semantics/type.ecore" xmlns:xmi="http://www.omg.org/XMI" xmlns:type="http:///de/tudarmstadt/ukp/dkpro/core/api/coref/type.ecore" xmlns:morph="http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/morph.ecore" xmlns:types="http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore" xmlns:constituent="http:///de/tudarmstadt/ukp/dkpro/core/api/syntax/type/constituent.ecore" xmlns:pos="http:///de/tudarmstadt/ukp/dkpro/core/api/lexmorph/type/pos.ecore" xmi:version="2.0">
        <cas:NULL xmi:id="0"/>
        <cas:Sofa xmi:id="6" sofaNum="1" sofaID="_InitialView" mimeType="text" sofaString="cannibaldave&#10;Not particularly unusual among the people I know. I just had nothing in common with the people I went to school with. Why should I? The coincidence of age and living in the same area were all there was to go on. You could call that in itself a very narrow prospect. The friends I had were of all ages and from a variety of different localities. &#10;A school doesn't need to be 'a nightmarish bully fest' to be a bane. All it needs to do is to fail to provide a totally unstimulating environment. Which mine did. It was boring, tedious, slow and frustrating. I learned nothing that I did not know before other than a handful of French verbs which have so far been of as much use as a chocolate fireguard. &#10;Each to his own, as I said. If you really enjoyed school, good for you. But don't assume that one size fits everybody, because it doesn't. "/>
        <type2:DocumentMetaData xmi:id="13" sofa="6" begin="0" end="854" language="en" documentId="1021" documentUri="1021" isLastSegment="false"/>
        <types:WebArgumentMetadata xmi:id="24" sofa="6" begin="0" end="0" author="kikichan" date="07 November 2009 12:16pm" docType="artcomment" origUrl="http://discussion.theguardian.com/comment-permalink/6753726" topic="homeschooling" thumbsUp="4" thumbsDown="0" origId="1021" notes="" title="Ridiculous rules for home schools"/>
        <types:Premise xmi:id="1615" sofa="6" begin="428" end="505" properties="#java.util.Properties&#10;#Thu Aug 07 15:59:02 CEST 2014&#10;originalView=logos&#10;rephrasedContent=schools provide unstimulating environment&#10;"/>
        <types:Backing xmi:id="1621" sofa="6" begin="522" end="713" properties="#java.util.Properties&#10;#Thu Aug 07 15:59:02 CEST 2014&#10;originalView=logos&#10;rephrasedContent=I was bored at school and learned nothing&#10;"/>
        <types:Claim xmi:id="1627" sofa="6" begin="787" end="853" properties="#java.util.Properties&#10;#Thu Aug 07 15:59:02 CEST 2014&#10;originalView=logos&#10;rephrasedContent=school is not good for everyone&#10;"/>
        <types:ArgumentRelation xmi:id="1634" sofa="6" begin="0" end="0" properties="#java.util.Properties&#10;#Thu Aug 07 15:59:02 CEST 2014&#10;originalView=logos&#10;" source="1615" target="1627"/>
        <types:ArgumentRelation xmi:id="1642" sofa="6" begin="0" end="0" properties="#java.util.Properties&#10;#Thu Aug 07 15:59:02 CEST 2014&#10;originalView=logos&#10;" source="1621" target="1627"/>
        <cas:View sofa="6" members="13 24 39 47 55 63 71 79 87 95 103 111 119 127 135 143 151 159 167 175 183 191 199 207 215 223 231 239 247 255 263 271 279 287 295 303 311 319 327 335 343 351 359 367 375 383 391 399 407 415 423 431 439 447 455 463 471 479 487 495 503 511 519 527 535 543 551 559 567 575 583 591 599 607 615 623 631 639 647 655 663 671 679 687 695 703 711 719 727 735 743 751 759 767 775 783 791 799 807 815 823 831 839 847 855 863 871 879 887 895 903 911 919 927 935 943 951 959 967 975 983 991 999 1007 1015 1023 1031 1039 1047 1055 1063 1071 1079 1087 1095 1103 1111 1119 1127 1135 1143 1151 1159 1167 1175 1183 1191 1199 1207 1215 1223 1231 1239 1247 1255 1263 1271 1279 1287 1295 1303 1311 1319 1327 1335 1343 1351 1359 1367 1375 1383 1391 1399 1407 1415 1423 1431 1439 1447 1455 1463 1471 1479 1487 1495 1503 1511 1519 1527 1535 1543 1547 1551 1555 1559 1563 1567 1571 1575 1579 1583 1587 1591 1595 1599 1603 1607 1611 1615 1621 1627 1634 1642"/>
    </xmi:XMI>
'''


class ExtractToulminComponentsAAUGWD(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='extract-toulmin-components_aaugwd',
            task_instruction='Extract the Toulmin components (Premise, Claim, Backing, Refutation and Rebuttal) from the given argument. The output should be in the format: "Premise: <premise> --> Claim: <claim>" or "Refutation: <refutation> --> Rebuttal: <rebuttal>" or "Rebuttal: <rebuttal> --> Claim: <claim>" or "Backing: <backing>"',
            dataset_names=['aaugwd'],
            is_clf=False, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/habernal.gurevych.2015.argumentation.mining.CL.data/data/gold.data.toulmin'
        files = os.listdir(ds_path)
        np.random.seed(42)
        np.random.shuffle(files)
        train_files = files[:int(len(files)*0.7)]
        dev_files = files[int(len(files)*0.7):int(len(files)*0.8)]
        test_files = files[int(len(files)*0.8):]

        # read all .xmi files
        for root, _, files in os.walk(ds_path):
            for file in files:
                if file.endswith('.xmi'):
                    with open(os.path.join(root, file), 'r') as f:
                        xmi = f.read()
                        f.close()
                    xml_root = ET.fromstring(xmi)
                    components = {'ID': [], 'type': [], 'text': []}
                    connections = {'source_ID': [], 'target_ID': []}
                    for child in xml_root:
                        if child.tag == '{http:///uima/cas.ecore}Sofa':
                            input = child.attrib['sofaString']
                        elif child.tag == '{http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore}Premise':
                            components['ID'].append(child.attrib['{http://www.omg.org/XMI}id'])
                            components['type'].append('Premise')
                            components['text'].append(input[int(child.attrib['begin']):int(child.attrib['end'])])
                        elif child.tag == '{http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore}Backing':
                            components['ID'].append(child.attrib['{http://www.omg.org/XMI}id'])
                            components['type'].append('Backing')
                            components['text'].append(input[int(child.attrib['begin']):int(child.attrib['end'])])
                        elif child.tag == '{http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore}Claim':
                            components['ID'].append(child.attrib['{http://www.omg.org/XMI}id'])
                            components['type'].append('Claim')
                            components['text'].append(input[int(child.attrib['begin']):int(child.attrib['end'])])
                        elif child.tag == '{http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore}Rebuttal':
                            components['ID'].append(child.attrib['{http://www.omg.org/XMI}id'])
                            components['type'].append('Rebuttal')
                            components['text'].append(input[int(child.attrib['begin']):int(child.attrib['end'])])
                        elif child.tag == '{http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore}Refutation':
                            components['ID'].append(child.attrib['{http://www.omg.org/XMI}id'])
                            components['type'].append('Refutation')
                            components['text'].append(input[int(child.attrib['begin']):int(child.attrib['end'])])
                        elif child.tag == '{http:///de/tudarmstadt/ukp/dkpro/argumentation/types.ecore}ArgumentRelation':
                            connections['source_ID'].append(child.attrib['source'])
                            connections['target_ID'].append(child.attrib['target'])
                    # create pandas dataframe
                    components_df = pd.DataFrame(components)
                    connections_df = pd.DataFrame(connections)

                    output = ''

                    # create premise claim pairs
                    for i, row in connections_df.iterrows():
                        if row['source_ID'] in components_df['ID'].values and row['target_ID'] in components_df['ID'].values:
                            if components_df[components_df['ID'] == row['source_ID']]['type'].values[0] == 'Premise' and components_df[components_df['ID'] == row['target_ID']]['type'].values[0] == 'Claim':
                                output += 'Premise: ' + components_df[components_df['ID'] == row['source_ID']]['text'].values[0] + \
                                    ' --> Claim: ' + components_df[components_df['ID']
                                                                   == row['target_ID']]['text'].values[0] + '\n'
                            elif components_df[components_df['ID'] == row['source_ID']]['type'].values[0] == 'Refutation' and components_df[components_df['ID'] == row['target_ID']]['type'].values[0] == 'Rebuttal':
                                output += 'Refutation: ' + components_df[components_df['ID'] == row['source_ID']]['text'].values[0] + \
                                    ' --> Rebuttal: ' + components_df[components_df['ID']
                                                                      == row['target_ID']]['text'].values[0] + '\n'
                            elif components_df[components_df['ID'] == row['source_ID']]['type'].values[0] == 'Rebuttal' and components_df[components_df['ID'] == row['target_ID']]['type'].values[0] == 'Claim':
                                output += 'Rebuttal: ' + components_df[components_df['ID'] == row['source_ID']]['text'].values[0] + \
                                    ' --> Claim: ' + components_df[components_df['ID']
                                                                   == row['target_ID']]['text'].values[0] + '\n'
                    for i, row in components_df.iterrows():
                        if row['type'] == 'Backing':
                            output += 'Backing: ' + row['text'] + '\n'

                    instance = Instance(
                        input=input,
                        output=output,
                        split='train' if file in train_files else 'dev' if file in dev_files else 'test'
                    )
                    self.instances.append(instance)


if __name__ == '__main__':
    task = DetectPersuasiveDocumentsAAUGWD()
    task.load_data()
    task = ExtractToulminComponentsAAUGWD()
    task.load_data()
