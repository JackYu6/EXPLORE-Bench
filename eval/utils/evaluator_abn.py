import os
import re
import numpy as np
import json


class BaseEvaluator(object):
    def __init__(self, args):
        self.device = args.device
        self.long_caption = self.load_longcaption(args.description_file)
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_info = {}
        self.reset_record()
        self.save_index = 0

    def reset_record(self):
        self.results = {
            'object coverage': 0.0,     # accumulate (%)
            'attribute score': 0.0,     # accumulate (0~5 depending on scorer, later *20)
            'relation score': 0.0,      # accumulate
            'key_state score': 0.0,     # accumulate
        }
        self.count = 0

    def load_longcaption(self, path):
        with open(path, 'r') as f:
            long_caption = json.load(f)
        return long_caption

    def split_caption(self, text):
        texts = re.split(r'\n|\\n|</s>|[.]', text)
        subcap = []
        for text_prompt in texts:
            text_prompt = text_prompt.strip()
            if len(text_prompt) != 0:
                subcap.append(text_prompt)
        del texts
        return subcap

    def update_object_coverage(self, coverage):
        self.results['object coverage'] += coverage * 100.0
        self.count += 1

    def update_attribute_score(self, score):
        self.results['attribute score'] += score

    def update_relation_score(self, score):
        self.results['relation score'] += score

    def update_key_state_score(self, score):
        self.results['key_state score'] += score

    def start(self):
        pass

    def log_results(self, image_name, results):
        self.log_info[image_name] = results

    def get_results(self):
        result_logs = {}

        # 原始三项 + key_state 平均分
        for k, v in self.results.items():
            result_logs[k] = v / self.count if self.count > 0 else 0.0
            string = f'{k}: {result_logs[k]}'
            if 'coverage' in k:
                string = string + '%'
            print(string)

        # 旧 Unified score（保持原计算方式不变）
        object_coverage = self.results['object coverage'] / self.count if self.count > 0 else 0.0
        attribute_score = (self.results['attribute score'] / self.count * 20.0) if self.count > 0 else 0.0
        relation_score = (self.results['relation score'] / self.count * 20.0) if self.count > 0 else 0.0
        old_unified = 0.25 * object_coverage + 0.35 * attribute_score + 0.4 * relation_score

        # key_state（按同样 *20 缩放）
        key_state_avg = self.results['key_state score'] / self.count if self.count > 0 else 0.0
        key_state_20 = key_state_avg * 20.0

        w = 0.9
        new_unified = (1 - w) * old_unified + w * key_state_20

        # 记录到 result_logs
        result_logs['Old Unified score'] = old_unified
        result_logs['key_state score x20'] = key_state_20
        result_logs['New Unified score'] = new_unified

        print(f'Old Unified score: {old_unified}')
        print(f'key_state score x20: {key_state_20}')
        print(f'New Unified score: {new_unified}')

        with open(os.path.join(self.output_dir, 'result_' + str(self.save_index) + '.json'), 'w') as f:
            log_info = [result_logs] + [self.log_info]
            json.dump(log_info, f, indent=2, ensure_ascii=False)