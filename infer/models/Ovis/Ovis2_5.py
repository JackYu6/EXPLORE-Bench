import torch
from PIL import Image
from transformers import AutoModelForCausalLM

class Ovis2_5:
    def __init__(
        self,
        model_path: str,
        enable_thinking: bool = True,
        enable_thinking_budget: bool = True, # Only effective if enable_thinking is True.
        max_new_tokens: int = 8192, # Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
        thinking_budget: int = 5120,
    ):
        super().__init__()
        self.enable_thinking = enable_thinking
        self.enable_thinking_budget = enable_thinking_budget
        self.max_new_tokens = max_new_tokens
        self.thinking_budget = thinking_budget

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda().eval()

        self.tokenizer = self.llm.text_tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.llm, "generation_config"):
            self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _build_messages(self, messages):
        prompt = messages["prompt"]
        start_frame_path = messages["start_frame_path"]

        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(start_frame_path)},
                {"type": "text", "text": prompt},
            ],
        }]

    @torch.inference_mode()
    def generate_output(self, messages):
        messages = self._build_messages(messages)

        input_ids, pixel_values, grid_thws = self.llm.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        outputs = self.llm.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=self.enable_thinking,
            enable_thinking_budget=self.enable_thinking_budget,
            max_new_tokens=self.max_new_tokens,
            thinking_budget=self.thinking_budget,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]