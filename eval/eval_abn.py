import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import spacy

from utils.evaluator_abn import BaseEvaluator
from dataset.bench_dataset import AbnBenchData


def build_scorer(model, device):
    if 'Qwen' in model or 'qwen' in model:
        from utils.metric.qwen3_abn import Qwen
        scorer = Qwen(model, device)
    else:
        raise ValueError(f"{model} is not avalible")
    return scorer


class Evaluator(BaseEvaluator):
    def __init__(self, args):
        super(Evaluator, self).__init__(args)

        self.scorer = build_scorer(args.llm, args.device)
        self.nlp = spacy.load('en_core_web_lg')
        self.bert = SentenceTransformer(args.bert)

        self.data = AbnBenchData(args.data_root, args.anno)
        self.len_data = len(self.data)
        self.num_data = min(args.data_num, self.len_data) if args.data_num > 0 else self.len_data

        self.soft_coverage = args.soft_coverage

    def get_key_words(self, sub_captions):
        # find all noun words from sub_captions,
        # and the relation between words and sub_captions:
        # word2caption{word_id1: sub_caption_id1, word_id2: sub_caption_id2}
        words = []
        word2caption = {}
        for i, text in enumerate(sub_captions):
            doc = self.nlp(text)
            candidates = []
            for token in doc:
                if token.tag_ and token.tag_[0] == 'N':
                    candidates.append(str(token.lemma_.lower()))

            if len(candidates) == 0:
                continue

            cur_words = candidates

            start_id = len(words)
            words = words + cur_words
            for w in range(start_id, len(words)):
                word2caption[w] = i

        return words, word2caption

    def start(self):
        with torch.no_grad():
            for index in tqdm(range(self.num_data)):
                start_frame, gt_objects, gt_description, gt_relations, gt_key_states = self.data.get_data(index)

                if not gt_objects:
                    continue
                if start_frame not in self.long_caption or not self.long_caption[start_frame]:
                    print('No final scene description for the initial scene image:', start_frame)
                    continue

                sub_captions = self.split_caption(self.long_caption[start_frame])
                words, word2caption = self.get_key_words(sub_captions)

                no_overlap_gt_objects = list(set(gt_objects))
                match_object, object_score = self.get_coverage(no_overlap_gt_objects, words)

                des_score, relation_score, key_state_score, key_state_gt_list = self.get_accuracy(
                    gt_objects=gt_objects,
                    no_overlap_gt_objects=no_overlap_gt_objects,
                    gt_description=gt_description,
                    gt_relations=gt_relations,
                    gt_key_states=gt_key_states,
                    sub_captions=sub_captions,
                    word2caption=word2caption,
                    match_object=match_object,
                )

                result = {'object': {}, 'attribute': {}, 'relation': {}, 'key_state': {}}

                for k, gt_object in enumerate(no_overlap_gt_objects):
                    result['object'][gt_object] = float(object_score[k])

                for k, des in enumerate(gt_description):
                    result['attribute'][des] = float(des_score[k])

                relations = []
                for r in gt_relations:
                    if isinstance(r, list):
                        relations += r
                    elif isinstance(r, str):
                        relations.append(r)
                assert len(relations) == len(relation_score)
                for k, relation in enumerate(relations):
                    result['relation'][relation] = float(relation_score[k])

                assert len(key_state_gt_list) == len(key_state_score)
                for k, ks in enumerate(key_state_gt_list):
                    result['key_state'][ks] = float(key_state_score[k])

                self.log_results(start_frame, result)

    def get_coverage(self, no_overlap_gt_objects, words):
        # calculate object-level coverage
        if len(words) == 0 or len(no_overlap_gt_objects) == 0:
            cur_coverage = 0.0
            self.update_object_coverage(cur_coverage)
            match_object = torch.zeros((len(no_overlap_gt_objects), 0), dtype=torch.bool)
            object_score = torch.zeros((len(no_overlap_gt_objects),), dtype=torch.bool)
            return match_object, object_score

        with torch.no_grad():
            gts_embed = self.bert.encode(no_overlap_gt_objects)
            preds_embed = self.bert.encode(words)
            cosine_scores = self.bert.similarity(gts_embed, preds_embed)

        max_sim = cosine_scores.max(dim=0)[0]
        gt2preds_max_sim = cosine_scores.max(dim=1)[0]

        match_object = (cosine_scores == max_sim.unsqueeze(0)) * (cosine_scores == gt2preds_max_sim.unsqueeze(-1))
        object_score = match_object.int().sum(dim=-1) > 0
        contained_object_idx = torch.where(object_score)[0]

        if self.soft_coverage:
            contained_object_score = [gt2preds_max_sim[i].item() for i in contained_object_idx]
            object_score = object_score * gt2preds_max_sim
            cur_coverage = sum(contained_object_score) / len(no_overlap_gt_objects) if len(no_overlap_gt_objects) else 0.0
        else:
            contained_object = [no_overlap_gt_objects[i] for i in contained_object_idx]
            cur_coverage = len(contained_object) / len(no_overlap_gt_objects) if len(no_overlap_gt_objects) else 0.0

        self.update_object_coverage(cur_coverage)
        return match_object, object_score

    def get_accuracy(
        self,
        gt_objects,
        no_overlap_gt_objects,
        gt_description,
        gt_relations,
        gt_key_states,
        sub_captions,
        word2caption,
        match_object
    ):
        # 将 match_object 扩展到与 gt_objects 一一对应（gt_objects 可能有重复，而 no_overlap_gt_objects 已去重）
        _match_object = []
        for gt in gt_objects:
            object_id = no_overlap_gt_objects.index(gt)
            _match_object.append(match_object[object_id])
        match_object = torch.stack(_match_object, dim=0)  # [len(gt_objects), len(words)]

        preds, gts = [], []
        rels, rels_preds = [], []

        key_state_gts, key_state_preds = [], []

        for k in range(len(gt_description)):
            cur_match = list(match_object[k, :].int().nonzero().view(-1))

            gts.append(gt_description[k])
            object_captions = sorted([word2caption[int(cur_)] for cur_ in cur_match]) if len(cur_match) > 0 else []
            pred_caption = [sub_captions[subcaption_id] for subcaption_id in object_captions]
            preds.append('. '.join(pred_caption))

            # relation：构造包含关系中涉及对象的 caption 片段
            if len(gt_relations[k]) > 0:
                for cur_obj_gt_relation in gt_relations[k]:
                    object_captions_rel = list(object_captions)
                    for k_obj, gt_object in enumerate(gt_objects):
                        if k_obj == k:
                            continue
                        if isinstance(cur_obj_gt_relation, str) and gt_object in cur_obj_gt_relation:
                            cur_match_2 = list(match_object[k_obj, :].int().nonzero().view(-1))
                            object_captions_rel = object_captions_rel + sorted([word2caption[int(cur_)] for cur_ in cur_match_2])

                    relation_pred_caption = [sub_captions[subcaption_id] for subcaption_id in sorted(list(set(object_captions_rel)))]
                    rels_preds.append('. '.join(relation_pred_caption))
                    rels.append(cur_obj_gt_relation)
            else:
                pass

            ks = gt_key_states[k] if k < len(gt_key_states) else None
            if isinstance(ks, str) and len(ks.strip()) > 0:
                key_state_gts.append(ks)
                key_state_preds.append('. '.join(pred_caption))

        des_score, relation_score, key_state_score = self.scorer.evaluate(
            gts=gts,
            preds=preds,
            relations=rels,
            relation_preds=rels_preds,
            key_states=key_state_gts,
            key_state_preds=key_state_preds,
        )

        self.update_attribute_score(sum(des_score) / len(gt_description) if len(gt_description) else 0.0)

        if len(relation_score) > 0:
            self.update_relation_score(sum(relation_score) / len(relation_score))
        else:
            self.update_relation_score(0.0)

        if len(key_state_score) > 0:
            self.update_key_state_score(sum(key_state_score) / len(key_state_score))
        else:
            self.update_key_state_score(0.0)

        return des_score, relation_score, key_state_score, key_state_gts


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SceneDescription-Evaluator", add_help=True)
    parser.add_argument("--data_root", type=str, required=True, help="path to benchmark dataset")
    parser.add_argument("--anno", type=str, required=True, help="path to annotation file")
    parser.add_argument("--description_file", type=str, required=True, help="path to final scene descriptions to be evaluted")
    parser.add_argument("--llm", type=str, help="path to llm scorer")
    parser.add_argument("--bert", type=str, help="path to bert weight")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--soft_coverage", action="store_true", help="use word similarity as coverage")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--k", type=int, default=1, help="eval time")
    parser.add_argument("--data_num", type=int, default=-1, help="the number of data to be used for eval")
    args = parser.parse_args()

    evaluator = Evaluator(args)
    for i in range(args.k):
        print(f"Start evaluating final scene descriptions from {args.description_file}.\nEpoch index: {i}")
        evaluator.reset_record()
        evaluator.save_index = i
        evaluator.start()
        if evaluator.count < evaluator.num_data:
            print('Only ', str(evaluator.count), ' data are used for evaluation (<', str(evaluator.num_data), ').')
        evaluator.get_results()
        