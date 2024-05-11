import anthropic
from anthropic import Message
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Dict

SYSTEM_PROMPT = "You are a helpful assistant designed to generate synthetic realistic data. Answer as a list of json"
USER_PROMPT = "Tell me a description for a business classified in any NAICS code whose first 2 digits are exactly '[NAICS_CODE]' and its corresponding complete (6 digits) NAICS code. Each description must be around 40 words. Repeat it for 20 different business. The output should be a json with the keys naics and description."

MODEL = "claude-3-sonnet-20240229"


class SyntheticDescriptionsCreator:
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT

        self.client = anthropic.Anthropic(
            api_key=self.api_key,
        )

    def _make_api_call(self, naics_code: str, temperature: float = 0.7) -> Message:
        message = self.client.messages.create(
            model=MODEL,
            max_tokens=4000,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_PROMPT.replace("[NAICS_CODE]", naics_code),
                }
            ],
        )
        return message

    def process_api_call(
        self, naics_code: str, temperature: float = 0.7
    ) -> List[Dict[str, str]]:
        response = self._make_api_call(naics_code, temperature)
        json_data = eval(response.content[0])
        return json_data

    def create_dataframe(
        self, naics_code: str, temperature: float = 0.7
    ) -> pd.DataFrame:
        data = self.process_api_call(naics_code, temperature)
        return pd.DataFrame(data)
