from openai import OpenAI


class FedMLDeployedModel(object):

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.client = self.client()

    def client(self):
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def query_model(self,
              model,
              stream=False,
              extra_body=dict(),
              max_tokens=512,
              temperature=0.5,
              top_p=0.7,              
              messages=[{"role":"user","content":"Test!"}]):
        
        chat_completion = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,            
            messages=messages,
            stream=stream,
            extra_body=extra_body)
        
        return chat_completion
    

if __name__ == "__main__":
    # print the output
    client = FedMLDeployedModel(
        api_key="", # the user's token
        base_url="http://38.101.196.134:2203/inference/943"
    )
    
    input_args = {
        "model": "fedml-dimitris-llama-2-13b",
        "stream": False,
        "messages" : [{
            "role": "user",
            "content": "Talk about San Francisco"
        }]
    }
    
    query_res = client.query_model(**input_args)
    for choice in query_res.choices:
        print(choice.message.content, flush=True)
