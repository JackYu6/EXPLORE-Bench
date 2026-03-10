import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class MiniCPM_V4_5:
    def __init__(
        self,
        model_path: str = "openbmb/MiniCPM-V-4_5",
        enable_thinking: bool = True,       # If `enable_thinking=True`, the thinking mode is enabled.
        attn_implementation: str = "sdpa",   # "sdpa" or "flash_attention_2"
        torch_dtype=torch.bfloat16,
        device: str = "cuda",
        seed: int = 100,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.enable_thinking = enable_thinking
        self.device = device

        self.llm = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        ).eval().to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    def _build_msgs(self, messages):
        prompt = messages["prompt"]
        start_frame_path = messages["start_frame_path"]
        image = Image.open(start_frame_path).convert("RGB")
        return [{"role": "user", "content": [image, prompt]}]

    @torch.inference_mode()
    def generate_output(self, messages):
        msgs = self._build_msgs(messages)

        # stream=True -> generator yielding strings
        answer_iter = self.llm.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            enable_thinking=self.enable_thinking,
            stream=True,
        )

        generated_text = ""
        for new_text in answer_iter:
            generated_text += new_text
            # print(new_text, flush=True, end='')
        return generated_text

    def generate_outputs(self, messages_list):
        return [self.generate_output(m) for m in messages_list]