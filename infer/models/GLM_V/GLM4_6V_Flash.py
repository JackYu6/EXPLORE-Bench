from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch

'''
pip install transformers==5.1.0
'''

class GLM4_6V_Flash:
    def __init__(self, model_path):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.llm = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,   
            device_map={"": 0},           
        ).eval()

    def process_messages(self, messages):
        prompt = messages["prompt"]
        start_frame_path = messages["start_frame_path"]

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
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.llm.device)

        inputs.pop("token_type_ids", None)
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

        decoded = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,              
            clean_up_tokenization_spaces=False,
        )
        
        return decoded

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]