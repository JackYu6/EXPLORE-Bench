from transformers import AutoModel, AutoProcessor
from keye_vl_utils import process_vision_info
import torch

'''
pip install keye-vl-utils==1.5.2
export HF_HUB_CACHE="~/.cache/huggingface/hub"
'''

class Keye_VL1_5:
    def __init__(
        self, 
        model_path: str = "Kwai-Keye/Keye-VL-1.5-8B",
        enable_thinking: bool = False
    ):
        super().__init__()
        self.enable_thinking = enable_thinking
        
        self.llm = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,            
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map={"": 0},
        ).eval()
        self.llm.to("cuda")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

    def process_messages(self, messages):
        prompt = messages["prompt"]
        start_frame_path = messages["start_frame_path"]
        
        if self.enable_thinking:
            prompt += "/think"      # Thinking mode
        else:
            prompt += "/no_think"   # Non-Thinking Mode

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

        image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(hf_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **mm_processor_kwargs,
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