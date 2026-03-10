import base64
from licloud.openai import LiOpenAI

'''
This is an implementation using an internal Python library and is provided for reference only. For more information, please see https://github.com/openai/openai-python.
'''

class GPT:
    def __init__(self, model_name):
        super().__init__()
        self.client = LiOpenAI()
        self.model_name = model_name

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_messages(self, messages):
        prompt = messages["prompt"]
        image_path = messages["start_frame_path"]
        base64_image = self._encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ]
        return messages

    def generate_output(self, messages):
        messages = self.process_messages(messages)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=8192,
        )
        return completion.choices[0].message.content

    def generate_outputs(self, messages_list):
        return [self.generate_output(item) for item in messages_list]