from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import os

'''
export HF_HUB_CACHE="~/.cache/huggingface/hub"
'''

class Step3_VL:
    def __init__(self, model_path):
        super().__init__()

        self.key_mapping = {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        }

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": 0},              
            torch_dtype=torch.bfloat16,     
            key_mapping=self.key_mapping,
        ).eval()
        
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            if tok.pad_token_id is None:
                if tok.eos_token_id is None:
                    raise ValueError("tokenizer has neither pad_token_id nor eos_token_id.")
                tok.pad_token = tok.eos_token
            self.llm.config.pad_token_id = tok.pad_token_id

    def process_messages(self, messages_dict):
        prompt = messages_dict["prompt"]
        start_frame_path = messages_dict["start_frame_path"]

        hf_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": start_frame_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.llm.device)

        return inputs

    @torch.inference_mode()
    def generate_output(self, messages):
        inputs = self.process_messages(messages)

        generate_ids = self.llm.generate(
            **inputs,
            max_new_tokens=8192,
            repetition_penalty=1.0,
            do_sample=False,
            pad_token_id=self.llm.config.pad_token_id,
        )

        decoded = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return decoded

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]