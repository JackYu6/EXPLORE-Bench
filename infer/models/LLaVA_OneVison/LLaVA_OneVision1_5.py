from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import torch

'''
export HF_HUB_CACHE="~/.cache/huggingface/hub"
'''


class LLaVA_OneVision1_5:
    def __init__(self, model_path):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,   
            device_map={"": 0},           
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        self.llm.eval()

    def process_messages(self, messages):
        prompt = messages["prompt"]
        start_frame_path = messages["start_frame_path"]

        hf_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": start_frame_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(hf_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        return inputs

    @torch.inference_mode()
    def generate_output(self, messages):
        inputs = self.process_messages(messages)

        generated_ids = self.llm.generate(
            **inputs,
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=8192,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]