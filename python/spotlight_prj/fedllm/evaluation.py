
import re

import torch
from data_formatting import DataFormatting


class Evaluation:

    def __init__(self):
        self.dat_fmt = DataFormatting()




    def extract_last_number(self, text):

        """
        Extracts the last number appearing in the text

        Args:
            text (str): The text to extract a number from.

        Returns:
            float or None: The last number in the text, or None if no number is found

        
        Explanation:
            1. Removes dollar signs and percentage symbols from text. 
            2. Users regex to find a number that appeares at the end of the text. 
            3. The pattern matches numbers that appear at the end of the string. 
            4 Return the found number as float, or None if no match is found. 
        """

        text = text.replace('$', '').replace('%','')

        pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'

        match = re.search(pattern, text)

        return float(match.group(1)) if match else None




    def extract_single_number(self, text):

        """
        Extracts a single number from text if exactly one number is present.

        Args:
            text (str): The text to extract number from. 
        
        Returns:
            float or None: The single number in the text, or None if zero or multiple numbers. 
        
        Explanation:
            1. Uses regex to find all numbers in the text including the negative numebers.
            2. If exactly one number if found, returns it as float. 
            3. If zero or multiple numbers are found, returns None.
        
        """

        numbers =re.findall(r'-?\d*\.?\d+', text)
        #print("NUMBERS ARE:::", numbers)

        if len(numbers)==0:
            return None
        elif len(numbers)==1:
            return float(numbers[0])

        else:
            return  None



    def evaluate_model(self, model, tokenizer, eval_samples, device):

        """
        Evaluates the  model on a set of examples and prints detailed results. 
        
        Args:
            model: The language model to evaluate. 
            tokenizer: The tokenizer for encoding inputs and decoding outputs. 
            eval_samples (list): List of evaluation examples each containing "prompt" and "answer"
            device: The device (CPU or GPU) to run evaluation on 
        
        Return:
            float: The accuracy percentage (correct predictions / total examples * 100)
        

        Explanation:
            1. Sets the model to evaluation mode. 
            2. For each example in the evaluation set:
                - Encodes the prompt and generates a respnse using the model
                - Extracts the predicted answer from the generated response
                - Compares the predicted answer with the expected answer using multiple methods

                    a. Extract string matching
                    b. Single number extraction and comparion.
                    c. Last number extraction and comparison
                -Prints detailed information about each example
            3. Calculates and returns the overall accuracy. 
            4. Returns the model to training mode. 

        """


        model.eval()

        correct = 0

        total = len(eval_samples)

        print("\n" + "="*50)
        print("EVALUATION ON", total, "EXAMPLES")
        print("="*50)


        for example in eval_samples:

            #get the prompt and expected answer

            full_prompt = example["prompt"]
            expected = example["answer"]

            #Tokenize and generate response

            inputs = tokenizer(full_prompt, return_tensors='pt', padding=False, truncation=False, return_attention_mask=True).to(device)

            with torch.no_grad():

                outputs = model.generate(
                    input_ids = inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    forced_eos_token_id = tokenizer.eos_token_id,
                    early_stopping = False,
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                try:
                    #Extract answers and check correctness
                    predicted = self.dat_fmt.extract_answer_from_model_output(response)

                    #Try different matching method

                    if predicted == expected : # Exact match

                        is_correct = True

                    else:
                        # Try single number matchin
                        pred_num = self.extract_single_number(str(predicted))
                        exp_num = self.extract_single_number(str(expected))

                        if pred_num is not None and exp_num is not None and pred_num==exp_num:

                            is_correct = True
                        else:
                            #Try the last number matchin
                            pre_num = self.extract_last_number(str(predicted))
                            exp_num = self.extract_last_number(str(expected))

                            is_correct = (pred_num is not None and exp_num is not None and pred_num == exp_num)

                    if is_correct:
                        correct+=1


                    # Print evaluation results

                    print("\nPrompt:")
                    print(full_prompt)
                    print("\nExpected Answer:")
                    print(expected)
                    print("\nExtracted Answer:")
                    print(predicted)
                    print("\nFull Generated Response:")
                    print(response)
                    print("\nCorrect:", "✓" if is_correct else "✗")
                    print("--"*50)

                except Exception as e:

                    print("\nFailed to parse the model output from prompt:")
                    print(full_prompt)
                    print('Error:',e)
                    print('-'*50)


        accuracy = (correct / total) * 100

        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})" )

        # return the model to training mode
        model.train()

        return accuracy


