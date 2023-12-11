from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import time
from configFolder import config
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers import pipeline


class Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_PATH)

    def generate(self,input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, **config.PARAMS)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class GecModel:
    def __init__(self) -> None:
        print("Loading Model")

        self.corrector = pipeline(
                    'text2text-generation',
                    config.GEC_PATH,
                    )
        print("Model Loaded")
    def get_prediction(self,x:list):
        
        results = self.corrector(x)
        return [i["generated_text"] for i in results]

class GPTneoModel:
    def __init__(self):
        self.model = GPTNeoXForCausalLM.from_pretrained(config.MODEL_PATH)
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(config.MODEL_PATH)
    def generate(self,input_text):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=100,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text

'''

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

prompt = "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
'''