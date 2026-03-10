from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


class InternVL3_5:
    def __init__(self, model_path):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            if tok.pad_token_id is None:
                if tok.eos_token_id is None:
                    raise ValueError("tokenizer has neither pad_token_id nor eos_token_id.")
                tok.pad_token = tok.eos_token  
            self.model.config.pad_token_id = tok.pad_token_id

    def _build_hf_messages(self, messages_dict):
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
        return hf_messages

    @torch.inference_mode()
    def generate_output(self, messages):
        hf_messages = self._build_hf_messages(messages)

        inputs = self.processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        generate_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=8192, pad_token_id=self.model.config.pad_token_id)

        # only decode newly generated tokens
        decoded = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return decoded

    def generate_outputs(self, messages_list):
        return [self.generate_output(messages) for messages in messages_list]
