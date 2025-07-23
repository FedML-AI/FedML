
from datasets import load_dataset


class DataFormatting:

    def __init__(self):

        self.system_prompt  = """
        
        Respond in the following format:

        <reasoning>
        ...
        </reasoning>

        <answer>

        ...
        </answer>
        
        """



    def extract_answer_from_model_output(self, text):

        """
        Extracts the value from the last <answer> tag in the text.

        Args:
            text (str): The model generated containing XML-style <answer> tags. 
        
        Returs:
            str or None: The content inside the <answer> tags, or None if no valid answer is found 
        
        Explanation: 
            1. Splits the text on the <answer> tag to isolate content after the tag.
            2. Checks if at least one <answer> tag exists in the text. 
            3. For the last <answer> segment:
                - Verifies it contains a closing </answer>
                - Extracts only the content between the tags.
            4. Returns None if the answer is empty (just "...") or if tags are missing
        """


        #split on <answer> and take everything after the last occurane.
        parts = text.split("<answer>")

        if len(parts)<2: # No <answer> tag found

            return None

        last_part = parts[-1]

        #Extract the content up to </answer>

        if "</answer>" not in last_part:
            return None

        answer = last_part.split("</answer>")[0].strip()

        return None if answer =="..." else answer


    def extract_answer_from_dataset(self, text):

        """
        Extracts the answer from the GSM8K dataset examples.

        Args:
            text(str): The dataset example text containing a question and answer
        
        Returns:
            str or None: The extracted answer part after the '####' delimiter, or None
        

        Explanation: 

        1. Checks if the text contains the '####' delimiter that separates questions from answers
        2. If found, splits the text at this delimiter and returns the second part 
        3. The answer is stripped of leading or trailing white spaces. 
        4. Returns None if no delimiter is present. 

        """

        if "####" not in text:
            return None

        return text.split("####")[1].strip()



    def prepare_dataset(self, split="train"):

        """
        Load and prepare GSM8K dataset for training with string prompts.

        Args:
            split(str): The dataset split to load("train" or "test"), Defaults to "train"
        
        Returns:
            list: A list of formatted examples, each containing a prompt string and the role
        
        Explanation:
            1. Loads GSM8K dataset from Hugging Face dataset hub.
            2. For each example in the dataset:
                - Creates a list of messages with system prompt and the question.
                - Converts this list into a single string prompt using build_prompt()
                - Extracts the answer from the dataset example. 
                - Creates a list of formatted examples with prompt and answer. 
            3. Returns the list of formatted examples ready for model training or evaluation. 
        """

        data = load_dataset('openai/gsm8k', 'main')[split]

        formatted_data = []

        for example in data:

            # convert the list of messages to a single string prompt

            prompt_str = self.build_prompt([
                {"role": "system", "content": self.system_prompt},
                {"role":"user", "content": example["question"]}
            ])


            formatted_example = {
                "prompt":prompt_str, # string rather than a list
                "answer": self.extract_answer_from_dataset(example["answer"])
            }
            formatted_data.append(formatted_example)

        return formatted_data



    def build_prompt(self,messages):

        """
        Build a single prompt string from a list of messages.

        Args:
            messages(list): A list of message dictionaries, each with 'role' and 'content'

        Returns:
            str: A concatenated string of all message content.

        Explanation:
            1. Takes a list of message dictionaries in typical chat format. 
            2. Extracts the 'content' field from each message and strips whitespace. 
            3. Joins all content strings with newlines to create a single prompt. 
            4. This preserves the training format while converting from structures messages. 
       """

        return "\n".join(msg["content"].strip() for msg in messages)

