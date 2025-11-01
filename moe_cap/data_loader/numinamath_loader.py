import re
from .base_data_loader import DataLoader
from moe_cap.configs.cap_config import CAPConfig
from datasets import load_dataset
from typing import List, Dict, Any

class NuminaMathLoader(DataLoader):
    """Loads and processes the AI-MO/NuminaMath-CoT dataset"""

    def __init__(self, config: CAPConfig) -> None:
        super().__init__(config)
        self.dataset = load_dataset("AI-MO/NuminaMath-CoT", split=config.dataset_split)
        self._process_data()

    def _process_data(self) -> None:
        """
        Extracts the user question and the final boxed answer 
        from the conversational 'messages' column.
        """
        
        def extract_qa(example: Dict[str, Any]) -> Dict[str, str]:
            messages = example.get('messages', [])
            question = ""
            final_answer = ""
            
            # Find the question
            user_msg = next((msg['content'] for msg in messages if msg.get('role') == 'user'), None)
            if user_msg:
                question = user_msg
            
            # Find the answer
            assistant_msg = next((msg['content'] for msg in messages if msg.get('role') == 'assistant'), None)
            if not assistant_msg:
                return {"input_question": question, "processed_answer": ""}
            
            # 1. Find the start index of the last '\boxed{'
            start_marker = r"\boxed{"
            last_box_start_idx = assistant_msg.rfind(start_marker)
            
            if last_box_start_idx == -1:
                # No \boxed{ found
                return {"input_question": question, "processed_answer": ""}
                
            # 2. Get the substring after the marker
            content_start_idx = last_box_start_idx + len(start_marker)
            substring = assistant_msg[content_start_idx:]
            
            # 3. find the matching '}'
            level = 1
            content_end_idx = -1
            for i, char in enumerate(substring):
                if char == '{':
                    level += 1
                elif char == '}':
                    level -= 1
                
                if level == 0:
                    content_end_idx = i
                    break
            
            # 4. Extract the content if a matching brace was found
            if content_end_idx != -1:
                final_answer = substring[:content_end_idx].strip()
                
            return {"input_question": question, "processed_answer": final_answer}

        # Apply the function to the entire dataset
        self.dataset = self.dataset.map(extract_qa)


    def get_input(self) -> List[str]:
        """Returns the list of questions."""
        return self.dataset["input_question"]

    def get_target(self) -> List[str]:
        """Returns the list of processed final answers for EM."""
        return self.dataset["processed_answer"]