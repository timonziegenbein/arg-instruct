import numpy as np
import os
import xml.etree.ElementTree as ET
import csv
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']


EXAMPLE_XML = """
<?xml version='1.0' encoding='UTF-8'?>
<arggraph id="micro_c002" topic_id="hunting_improves_environment" stance="pro">
  <edu id="e1"><![CDATA[In some cases, hunting can remove threats to the environment.]]></edu>
  <edu id="e2"><![CDATA[Texas, in particular, has an overabundance of wild/feral boars that destroy farmers' fields, and root up other ground, too.]]></edu>
  <edu id="e3"><![CDATA[Furthermore, hunting your own meat saves up the natural resources that would otherwise be used in a processing plant.]]></edu>
  <edu id="e4" implicit="true"><![CDATA[Hunting is good for the environment.]]></edu>
  <adu id="a1" type="pro"/>
  <adu id="a2" type="pro"/>
  <adu id="a3" type="pro"/>
  <adu id="a4" type="pro"/>
  <edge id="c4" src="e1" trg="a2" type="seg"/>
  <edge id="c5" src="e2" trg="a3" type="seg"/>
  <edge id="c6" src="e3" trg="a4" type="seg"/>
  <edge id="c7" src="e4" trg="a1" type="seg"/>
  <edge id="c1" src="a2" trg="a1" type="sup"/>
  <edge id="c2" src="a3" trg="a2" type="exa"/>
  <edge id="c3" src="a4" trg="a1" type="sup"/>
</arggraph>
"""

topic_mappping = {
    'hunting_improves_environment': 'Is hunting good or bad for the environment?',
    'older_people_better_parents': 'Do older people make better parents?',
    'removal_of_rhino_horns': 'Should the horns of wild rhinos be removed to prevent them from being poached?',
    'cell_phones_and_social_media_improve_families': 'Have cell phones and social media made families closer?',
    'eco-tourism_protects_nature': 'Can eco-tourism protect wild areas and animals?',
    'dating_before_engagement': 'How long should people date before they become engaged?',
    'fracking': 'Do we need fracking, despite its risks?',
    'romantic_movies_endanger_relationships': 'Are the expectations raised by romantic movies damaging to real relationships?',
    'influence_of_recycling': 'Does recycling really make a difference?',
    'prohibition_of_phones_while_driving': 'Should car drivers be strictly prohibited from using cell phones?',
    'promote_recycling_by_bottle_deposit': 'Should all states adopt a deposit on soft drink bottles and cans in order to promote recycling?',
    'social_media_improves_teenager_lives': 'Have social media improved the lives of teenagers?',
    'charge_for_plastic_bags': 'Should supermarkets charge for plastic bags in order to encourage the use of reusable bags?',
    'violent_video_games_cause_violence': 'Do violent video games cause people to act out violently?',
    'veganism_helps_environment': 'Does being a vegetarian or vegan help the environment?',
    'video_games_as_teaching_tools': 'Should schools use video games as a teaching tool?',
    'LED_lights_reduce_energy': 'Can using LED lights make a difference for our energy consumption?',
    'books_obsolete': 'Will paper and books become obsolete?',
    'composting_helps_environment': 'Can composting help save the environment?',
    'smart_watches_replace_cell_phones': 'Are smart watches going to replace cell phones?',
    'life_in_dirty_city_if_good_job': 'Is it possible to live in a dirty city if you have a good job?',
    'nuclear_energy_safe': 'Is nuclear energy really safe?',
    'responsible_handling_of_nuclear_waste': 'Is there a chance to handle nuclear waste responsibly?',
    'trash_in_landfills': 'Are landfills a good way for handling our trash?',
    'government_regulation_increases_solar_energy': 'Can government regulation speed up the spread of solar energy?',
    'long_distance_relationships': 'Do long distance relationships work?',
    'kids_recovery_from_divorce': 'Is divorce something that kids can recover from?',
    'teenage_marriage': 'Are teenage marriages a good idea?',
    'teenage_parenthood': 'Should teenagers that get pregnant keep their children?',
    'helicopter_parents': 'Are helicopter parents good for their children?',
    'only_child': 'Is it good to be an only child?',
    'sports_as_family_activity': 'Is doing sports together a good thing for families?',
    'video_games_bad_for_families': 'Do video games have a bad impact on family life?',
    'treat_dogs_as_humans': 'Is it OK to treat dogs on a par with family members?',
    'large_families_better_for_children': 'Are large families better for children?',
}

topic_split = {
    'hunting_improves_environment': 'train',
    'older_people_better_parents': 'test',
    'removal_of_rhino_horns': 'train',
    'cell_phones_and_social_media_improve_families': 'train',
    'eco-tourism_protects_nature': 'train',
    'dating_before_engagement': 'test',
    'fracking': 'test',
    'romantic_movies_endanger_relationships': 'train',
    'influence_of_recycling': 'train',
    'prohibition_of_phones_while_driving': 'train',
    'promote_recycling_by_bottle_deposit': 'dev',
    'social_media_improves_teenager_lives': 'train',
    'charge_for_plastic_bags': 'dev',
    'violent_video_games_cause_violence': 'train',
    'veganism_helps_environment': 'train',
    'video_games_as_teaching_tools': 'train',
    'LED_lights_reduce_energy': 'train',
    'books_obsolete': 'test',
    'composting_helps_environment': 'train',
    'smart_watches_replace_cell_phones': 'train',
    'life_in_dirty_city_if_good_job': 'train',
    'nuclear_energy_safe': 'train',
    'responsible_handling_of_nuclear_waste': 'test',
    'trash_in_landfills': 'train',
    'government_regulation_increases_solar_energy': 'train',
    'long_distance_relationships': 'test',
    'kids_recovery_from_divorce': 'dev',
    'teenage_marriage': 'train',
    'teenage_parenthood': 'train',
    'helicopter_parents': 'train',
    'only_child': 'train',
    'sports_as_family_activity': 'train',
    'video_games_bad_for_families': 'dev',
    'treat_dogs_as_humans': 'train',
    'large_families_better_for_children': 'test',
}


class CreateArgumentativeTextMicrotextsV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='argumentative-text-creation_microtexts_v2',
            task_instruction='Produce a short text that argues for or against the following debate topic.',
            dataset_names=['microtexts_v2'],
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/arg-microtexts-part2/corpus'
        texts = {}
        topics = {}
        splits = {}

        # read all xml files in the directory
        for file in sorted(os.listdir(ds_path)):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(ds_path, file))
                root = tree.getroot()
                stance = root.attrib['stance']
                topic_id = root.attrib['topic_id']
                topic = topic_mappping[topic_id]
                topics[file.split('.')[0]] = topic
                splits[file.split('.')[0]] = topic_split[topic_id]
            elif file.endswith('.txt'):
                with open(os.path.join(ds_path, file), 'r') as f:
                    text = f.read()
                    texts[file.split('.')[0]] = text
            else:
                pass

        for key in texts.keys():
            instance = Instance(
                input=topics[key],
                output=texts[key],
                split=splits[key],
            )

            self.instances.append(instance)


class ExtractCentralClaimMicrotextsV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='central-claim-extraction_microtexts_v2',
            task_instruction='Extract the central claim from the following argumentative text.',
            dataset_names=['microtexts_v2'],
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/arg-microtexts-part2/corpus'
        texts = {}
        claims = {}
        splits = {}

        # read all xml files in the directory
        for file in sorted(os.listdir(ds_path)):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(ds_path, file))
                root = tree.getroot()
                # find the adu that is no src in any edge
                adu_ids = [adu.attrib['id'] for adu in root.findall('adu')]
                src_ids = [edge.attrib['src'] for edge in root.findall('edge')]
                central_adu = [adu for adu in adu_ids if adu not in src_ids][0]
                # get edu_id of central_adu where type is seg
                central_edu = [edge.attrib['src'] for edge in root.findall(
                    'edge') if edge.attrib['trg'] == central_adu and edge.attrib['type'] == 'seg'][0]
                # get the text of central_edu
                central_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == central_edu][0]
                # clean the text
                central_text = central_text.replace('![CDATA[', '').replace(']]', '')
                claims[file.split('.')[0]] = central_text
                splits[file.split('.')[0]] = topic_split[root.attrib['topic_id']]
            elif file.endswith('.txt'):
                with open(os.path.join(ds_path, file), 'r') as f:
                    text = f.read()
                    texts[file.split('.')[0]] = text

        for key in texts.keys():
            instance = Instance(
                input=texts[key],
                output=claims[key],
                split=splits[key],
            )

            self.instances.append(instance)


class DetermineArgumentativeRoleMicrotextsV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='argumentative-role-determination_microtexts_v2',
            task_instruction='The argumentation can be thought of as a dialectical exchange between the role of the proponent (who is presenting and defending the central claim) and the role of the opponent (who is critically challenging the proponents claims). Given the following central claim and an argumentative discourse unit (ADU), determine the argumentative role, i.e. Proponent or Opponent of the ADU.',
            dataset_names=['microtexts_v2'],
            is_clf=True, 
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/arg-microtexts-part2/corpus'
        inputs = {}
        outputs = {}
        splits = {}

        # read all xml files in the directory
        for file in sorted(os.listdir(ds_path)):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(ds_path, file))
                root = tree.getroot()
                # get all edu texts
                edu_texts = [edu.text for edu in root.findall('edu')]
                # get all adu types
                adu_types = {adu.attrib['id']: adu.attrib['type'] for adu in root.findall('adu')}
                # get all edge types
                edge_types = {edge.attrib['src']: edge.attrib['type'] for edge in root.findall('edge')}
                # get all adu ids
                adu_ids = [adu.attrib['id'] for adu in root.findall('adu')]
                # get all edu ids
                edu_ids = [edu.attrib['id'] for edu in root.findall('edu')]
                # get all adu ids that are not src in any edge
                central_adu = [adu for adu in adu_ids if adu not in edge_types.keys()][0]
                # get all edu ids that are src in any edge with central_adu as trg
                central_edu = [edge for edge in root.findall(
                    'edge') if edge.attrib['trg'] == central_adu][0].attrib['src']
                # get the text of central_edu
                central_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == central_edu][0]
                # clean the text
                central_text = central_text.replace('![CDATA[', '').replace(']]', '')

                # map edu ids to adu adu ids based on edges of type seg
                edu_adu_mapping = {edge.attrib['src']: edge.attrib['trg']
                                   for edge in root.findall('edge') if edge.attrib['type'] == 'seg'}
                # map edu ids to adu adu_types
                edu_adu_types = {edu_id: adu_types[adu_id] for edu_id, adu_id in edu_adu_mapping.items()}
                for edu_id, edu_text in zip(edu_ids, edu_texts):
                    if edu_id in edu_adu_mapping.keys():
                        inputs[file.split('.')[0] + '_' + edu_id] = 'Central Claim: ' + \
                            central_text + '\n' + 'ADU: ' + edu_text
                        outputs[file.split('.')[0] + '_' + edu_id] = edu_adu_types[edu_id]
                        splits[file.split('.')[0] + '_' + edu_id] = topic_split[root.attrib['topic_id']]

        for key in inputs.keys():
            instance = Instance(
                input=inputs[key],
                output=outputs[key],
                split=splits[key],
            )

            self.instances.append(instance)


class DetermineFunctionOfSegmentMicrotextsV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='function-of-segment-determination_microtexts_v2',
            task_instruction='The edges representing arguments are those that connect argumentative discourse units (ADUs). The scheme distinguishes between supporting and attacking relations. Supporting relations are normal support and support by example. Attacking relations are rebutting attacks (directed against another node, challenging the acceptability of the corresponding claim) and undercutting attacks (directed against another relation, challenging the argumentative inference from the source to the target of the relation). Finally, additional premises of relations with more than one premise are represented by additional source relations. Given the following two argumentative discourse units, determine the function of the segment, i.e. support, support by example, rebutting attack, undercutting attack, or additional premise.',
            dataset_names=['microtexts_v2'],
            is_clf=True, 
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/arg-microtexts-part2/corpus'
        inputs = {}
        outputs = {}
        splits = {}

        # read all xml files in the directory
        for file in sorted(os.listdir(ds_path)):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(ds_path, file))
                root = tree.getroot()
                # get all edu texts
                edu_texts = [edu.text for edu in root.findall('edu')]
                # get all adu types
                adu_types = {adu.attrib['id']: adu.attrib['type'] for adu in root.findall('adu')}
                # get all edge types
                edge_types = {edge.attrib['src']: edge.attrib['type'] for edge in root.findall('edge')}
                # get all adu ids
                adu_ids = [adu.attrib['id'] for adu in root.findall('adu')]
                # get all edu ids
                edu_ids = [edu.attrib['id'] for edu in root.findall('edu')]
                # get all adu ids that are not src in any edge
                central_adu = [adu for adu in adu_ids if adu not in edge_types.keys()][0]
                # get all edu ids that are src in any edge with central_adu as trg
                central_edu = [edge for edge in root.findall(
                    'edge') if edge.attrib['trg'] == central_adu][0].attrib['src']
                # get the text of central_edu
                central_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == central_edu][0]
                # clean the text
                central_text = central_text.replace('![CDATA[', '').replace(']]', '')

                # map edu ids to adu adu ids based on edges of type seg
                adu_edu_mapping = {edge.attrib['trg']: edge.attrib['src']
                                   for edge in root.findall('edge') if edge.attrib['type'] == 'seg'}
                # iterate over all edges of type not seg
                for edge in root.findall('edge'):
                    if edge.attrib['type'] != 'seg':
                        src = adu_edu_mapping[edge.attrib['src']]
                        trg = edge.attrib['trg']
                        # backtrack if the relation is towards another relation c
                        while 'c' in trg:
                            # get edge where trg is id of c
                            tmp_edge = [edge for edge in root.findall('edge') if edge.attrib['id'] == trg][0]
                            trg = tmp_edge.attrib['src']
                        trg = adu_edu_mapping[trg]
                        src_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == src][0]
                        trg_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == trg][0]
                        src_text = src_text.replace('![CDATA[', '').replace(']]', '')
                        trg_text = trg_text.replace('![CDATA[', '').replace(']]', '')
                        type = 'support' if edge.attrib['type'] == 'sup' else 'support by example' if edge.attrib['type'] == 'exa' else 'rebutting attack' if edge.attrib[
                            'type'] == 'reb' else 'undercutting attack' if edge.attrib['type'] == 'und' else 'additional premise'
                        inputs[file.split('.')[0] + '_' + src + '_' + trg] = 'ADU1: ' + \
                            src_text + '\n' + 'ADU2: ' + trg_text
                        outputs[file.split('.')[0] + '_' + src + '_' + trg] = type
                        splits[file.split('.')[0] + '_' + src + '_' + trg] = topic_split[root.attrib['topic_id']]
        for key in inputs.keys():
            instance = Instance(
                input=inputs[key],
                output=outputs[key],
                split=splits[key],
            )

            self.instances.append(instance)


class IdentifyUnitAttachmentMicrotextsV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='unit-attachment-identification_microtexts_v2',
            task_instruction='Given the following two argumentative discourse units (ADUs), determine whether the two ADUs are connected by any argumentative relation (e.g. support or attack).',
            dataset_names=['microtexts_v2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/arg-microtexts-part2/corpus'
        inputs = {}
        outputs = {}
        splits = {}

        # read all xml files in the directory
        for file in sorted(os.listdir(ds_path)):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(ds_path, file))
                root = tree.getroot()
                # get all edu texts
                edu_texts = [edu.text for edu in root.findall('edu')]
                # get all adu types
                adu_types = {adu.attrib['id']: adu.attrib['type'] for adu in root.findall('adu')}
                # get all edge types
                edge_types = {edge.attrib['src']: edge.attrib['type'] for edge in root.findall('edge')}
                # get all adu ids
                adu_ids = [adu.attrib['id'] for adu in root.findall('adu')]
                # get all edu ids
                edu_ids = [edu.attrib['id'] for edu in root.findall('edu')]
                # get all adu ids that are not src in any edge
                central_adu = [adu for adu in adu_ids if adu not in edge_types.keys()][0]
                # get all edu ids that are src in any edge with central_adu as trg
                central_edu = [edge for edge in root.findall(
                    'edge') if edge.attrib['trg'] == central_adu][0].attrib['src']
                # get the text of central_edu
                central_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == central_edu][0]
                # clean the text
                central_text = central_text.replace('![CDATA[', '').replace(']]', '')

                # map edu ids to adu adu ids based on edges of type seg
                edu_adu_mapping = {edge.attrib['src']: edge.attrib['trg']
                                   for edge in root.findall('edge') if edge.attrib['type'] == 'seg'}

                # create all combinations of edu edu_ids
                for i, src_edu in enumerate(edu_ids):
                    for j, trg_edu in enumerate(edu_ids):
                        if i != j and src_edu in edu_adu_mapping.keys() and trg_edu in edu_adu_mapping.keys():
                            src_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == src_edu][0]
                            trg_text = [edu.text for edu in root.findall('edu') if edu.attrib['id'] == trg_edu][0]
                            src_text = src_text.replace('![CDATA[', '').replace(']]', '')
                            trg_text = trg_text.replace('![CDATA[', '').replace(']]', '')
                            src = edu_adu_mapping[src_edu]
                            trg = edu_adu_mapping[trg_edu]
                            # check if src and trg are connected by an edge
                            connected = False
                            for edge in root.findall('edge'):
                                if edge.attrib['src'] == src and edge.attrib['trg'] == trg or edge.attrib['src'] == trg and edge.attrib['trg'] == src:
                                    connected = True
                                    break
                                while 'c' in trg:
                                    tmp_edge = [edge for edge in root.findall('edge') if edge.attrib['id'] == trg][0]
                                    if tmp_edge.attrib['src'] == src and edge.attrib['src'] == trg or tmp_edge.attrib['src'] == trg and edge.attrib['src'] == src:
                                        connected = True
                                        break
                                    else:
                                        trg = tmp_edge.attrib['trg']
                            inputs[file.split('.')[0] + '_' + src + '_' + trg] = 'ADU1: ' + \
                                src_text + '\n' + 'ADU2: ' + trg_text
                            outputs[file.split('.')[0] + '_' + src + '_' + trg] = 'Yes' if connected else 'No'
                            splits[file.split('.')[0] + '_' + src + '_' + trg] = topic_split[root.attrib['topic_id']]

        for key in inputs.keys():
            instance = Instance(
                input=inputs[key],
                output=outputs[key],
                split=splits[key],
            )

            self.instances.append(instance)


if __name__ == '__main__':
    task = CreateArgumentativeTextMicrotextsV2()
    task.load_data()
    task = ExtractCentralClaimMicrotextsV2()
    task.load_data()
    task = DetermineArgumentativeRoleMicrotextsV2()
    task.load_data()
    task = DetermineFunctionOfSegmentMicrotextsV2()
    task.load_data()
    task = IdentifyUnitAttachmentMicrotextsV2()
    task.load_data()
