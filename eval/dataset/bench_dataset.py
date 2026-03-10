import json
import os


class BenchData():
    def __init__(self, data_root, anno_file):
        self.image_root = os.path.join(data_root, 'images')

        with open(anno_file, 'r') as f:
            info = json.load(f)
        self.annos = info
        
    def __len__(self):
        return len(self.annos)

    def get_data(self, idex):
        start_frame = self.annos[idex]['start_frame']
        objects = {'mid_1': [], 'mid_2': [], 'final': []}
        descriptions = {'mid_1': [], 'mid_2': [], 'final': []}
        relations = {'mid_1': [], 'mid_2': [], 'final': []}
        
        scenes = ['mid_1', 'mid_2', 'final']
        fram_info_keys = ['middle_frame_1_info', 'middle_frame_2_info', 'end_frame_info']
        
        for scene, fram_info_key in zip(scenes, fram_info_keys):
            for info in self.annos[idex].get(fram_info_key, []):
                objects[scene].append(info['category'])
                descriptions[scene].append(info['description'])
                relations[scene].append(info['relation'])

        return start_frame, objects, descriptions, relations
    
    
class AbnBenchData():
    def __init__(self,data_root, anno_file):
        self.image_root = os.path.join(data_root, 'images')

        with open(anno_file, 'r') as f:
            info = json.load(f)
        self.annos = info
        
    def __len__(self):
        return len(self.annos)

    def get_data(self, idex):
        start_frame = self.annos[idex]['start_frame']
        objects = []
        descriptions = []
        relations = []
        key_states = []
        
        for info in self.annos[idex].get('end_frame_info', []):
            objects.append(info['category'])
            descriptions.append(info['description'])
            relations.append(info['relation'])
            key_states.append(info.get('key_state', None))

        return start_frame, objects, descriptions, relations, key_states