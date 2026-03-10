import os
import re
import json

class BaseEvaluator(object):
    """
    支持：
    - single-scene：只维护一个总的 results
    - multi-scene：维护 per-scene results，并输出加权 final
    """
    def __init__(self, args):
        self.device = args.device
        self.long_caption = self.load_longcaption(args.description_file)
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.eval_mode = getattr(args, "eval_mode", "single-scene")
        self.save_index = 0

        # multi-scene 权重
        self.scene_weights = {"mid_1": 0.2, "mid_2": 0.3, "final": 0.5}
        self.scenes = ["mid_1", "mid_2", "final"]

        self.reset_record()

    def reset_record(self):
        # 单场景汇总
        self.results = {
            'object coverage': 0.,
            'attribute score': 0.,
            'relation score': 0.,
        }
        self.count = 0

        # 多场景汇总
        self.scene_results = {
            s: {'object coverage': 0., 'attribute score': 0., 'relation score': 0.}
            for s in self.scenes
        }
        self.scene_count = {s: 0 for s in self.scenes}

        # 细粒度日志：image -> scene -> details
        self.log_info = {}

    def load_longcaption(self, path):
        with open(path, 'r') as f:
            long_caption = json.load(f)
        return long_caption

    def split_caption(self, text):
        texts = re.split(r'\n|\\n|</s>|[.]', text)
        subcap = []
        for t in texts:
            t = t.strip()
            if len(t) != 0:
                subcap.append(t)
        return subcap

    # ---------- update (single-scene) ----------
    def update_object_coverage(self, coverage):
        self.results['object coverage'] += coverage * 100.
        self.count += 1

    def update_attribute_score(self, score):
        self.results['attribute score'] += score

    def update_relation_score(self, score):
        self.results['relation score'] += score

    # ---------- update (multi-scene) ----------
    def update_object_coverage_scene(self, scene, coverage):
        self.scene_results[scene]['object coverage'] += coverage * 100.
        self.scene_count[scene] += 1

    def update_attribute_score_scene(self, scene, score):
        self.scene_results[scene]['attribute score'] += score

    def update_relation_score_scene(self, scene, score):
        self.scene_results[scene]['relation score'] += score

    def log_results(self, image_name, scene, results):
        """
        results: {'object': {...}, 'attribute': {...}, 'relation': {...}}
        """
        if image_name not in self.log_info:
            self.log_info[image_name] = {}
        self.log_info[image_name][scene] = results

    def _compute_unified(self, object_coverage, attribute_score, relation_score):
        attr20 = attribute_score * 20.0
        rel20 = relation_score * 20.0
        unified = 0.25 * object_coverage + 0.35 * attr20 + 0.4 * rel20
        return attr20, rel20, unified

    def _avg_results(self, results_dict, count):
        if count <= 0:
            return {
                "count": 0,
                "object coverage": 0.0,
                "attribute score": 0.0,
                "relation score": 0.0,
                "attribute score x20": 0.0,
                "relation score x20": 0.0,
                "Unified score": 0.0,
            }

        obj = results_dict['object coverage'] / count
        attr = results_dict['attribute score'] / count
        rel = results_dict['relation score'] / count
        attr20, rel20, unified = self._compute_unified(obj, attr, rel)

        return {
            "count": count,
            "object coverage": obj,
            "attribute score": attr,
            "relation score": rel,
            "attribute score x20": attr20,
            "relation score x20": rel20,
            "Unified score": unified,
        }

    def get_results(self):
        result_logs = {
            "mode": self.eval_mode
        }

        if self.eval_mode == "single-scene":
            avg = self._avg_results(self.results, self.count)
            # 控制台输出
            print(f"object coverage: {avg['object coverage']}%")
            print(f"attribute score: {avg['attribute score']}")
            print(f"relation score: {avg['relation score']}")
            print(f"Unified score: {avg['Unified score']}")

            result_logs.update(avg)

        elif self.eval_mode == "multi-scene":
            scenes_summary = {}
            for s in self.scenes:
                avg = self._avg_results(self.scene_results[s], self.scene_count[s])
                scenes_summary[s] = avg

                # 控制台输出
                print(f"[{s}] object coverage: {avg['object coverage']}%")
                print(f"[{s}] attribute score: {avg['attribute score']}")
                print(f"[{s}] relation score: {avg['relation score']}")
                print(f"[{s}] Unified score: {avg['Unified score']}")

            # 加权最终（对四个核心项都加权）
            w = self.scene_weights
            final_weighted = {
                "weights": w,
                "object coverage": sum(w[s] * scenes_summary[s]["object coverage"] for s in self.scenes),
                "attribute score": sum(w[s] * scenes_summary[s]["attribute score"] for s in self.scenes),
                "relation score": sum(w[s] * scenes_summary[s]["relation score"] for s in self.scenes),
            }
            # 由加权后的(coverage, attr, rel) 再算 unified
            attr20, rel20, unified = self._compute_unified(
                final_weighted["object coverage"],
                final_weighted["attribute score"],
                final_weighted["relation score"]
            )
            final_weighted["attribute score x20"] = attr20
            final_weighted["relation score x20"] = rel20
            final_weighted["Unified score"] = unified

            print(f"[final_weighted] Unified score: {unified}")

            result_logs["scenes"] = scenes_summary
            result_logs["final_weighted"] = final_weighted

        else:
            raise ValueError(f"Unknown eval_mode: {self.eval_mode}")

        with open(os.path.join(self.output_dir, 'result_' + str(self.save_index) + '.json'), 'w') as f:
            log_info = [result_logs] + [self.log_info]
            json.dump(log_info, f, indent=2, ensure_ascii=False)