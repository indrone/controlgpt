from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import FAISS,Chroma
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from configFolder import config,LLMConfig

class LLM:
    def __init__(self):
        self.embedding=self._load_embedding()
        self._load_model()
        self.llm=self.load_llm_model()
        self.data=self.data_store()
        self.get_examples(self.data)
        self._prompts()
        self._few_shot_promts()
        self.get_chain()

    def _load_embedding(self):
        model_kwargs = {'device': 'cpu'}
        embedding=HuggingFaceEmbeddings(model_name=config.LLM_EMBED_PATH,model_kwargs=model_kwargs)
        return embedding

    def _load_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(config.LLM_MODEL, device_map="auto",offload_folder=config.LLM_MODEL_OFFLOAD)
        self.pipe=pipeline('text2text-generation',model=self.model,tokenizer=self.tokenizer, **config.PARAMS)
    def load_llm_model(self):
        return HuggingFacePipeline(pipeline=self.pipe)


    def get_examples(self,examples):
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            examples,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            self.embedding,
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=10,
        )

    def _few_shot_promts(self):
        self.similar_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=self.example_selector,
        example_prompt=self.example_prompt,
        prefix="Give the Control Description every input",
        suffix=config.FEW_SHOT_PROMPT_TEMPLATE,
        input_variables=config.FEW_SHOT_PROMPT_INPUT_VAR,
    )
    def _prompts(self):
        self.example_prompt = PromptTemplate(
            input_variables=config.PROMPT_INPUT_VAR,
            template= config.PROMPT_TEMPLATE,
        )

    def get_chain(self):
        self.chain = LLMChain(llm=self.llm, prompt=self.similar_prompt)

    def data_store(self):
        df=pd.read_excel(config.PATH_TO_DB)
        #example_df=df.sample(n=1300)
        #test_df=df.loc[~df.index.isin(example_df.index)]
        #test_df=test_df.reset_index(drop=True)
        examples=[]
        for idx,rows in df.iterrows():
            examples.append({
                "control_desc":rows['Description of controls'],
                "test_performed":rows['Test performed']
            })
        return examples

    def predict(self,text):
        preds=self.chain.run(text)
        return preds
    

class Embedding:
    def __init__(self,emb_type:str="Hugginface") -> None:
        self.emb_type=emb_type
    def load_embedding(self):
        if self.emb_type=="Hugginface":
            return HuggingFaceEmbeddings(**LLMConfig.HUGGINFACE_EMBEDDING)
        

class ModelFactory:
    def __init__(self,model_name:str,kwargs:dict={}) -> None:
        self.model_name=model_name
        self.kwargs=kwargs
    
    def load_model(self):
        if self.model_name=="AzureChatOpenAI":
            return AzureChatOpenAI(
                **LLMConfig.OPENAI_GPT_4,
                **self.kwargs
            )
        
        elif self.model_name=="AzureOpenAI":
            return AzureOpenAI(
                **LLMConfig.OPENAI_GPT_3,
                **self.kwargs
            )
        elif self.model_name=="HugginFaceT5Model":
            tokenizer = T5Tokenizer.from_pretrained(LLMConfig.huggingface_model)
            model = T5ForConditionalGeneration.from_pretrained(LLMConfig.huggingface_model, 
                                                               device_map="auto",
                                                               offload_folder=LLMConfig.huggingface_model_offload)
            pipe=pipeline('text2text-generation',model=model,tokenizer=tokenizer, **LLMConfig.huggingface_model_params)
            HuggingFacePipeline(pipeline=pipe)
            return HuggingFacePipeline(pipeline=pipe)
        

class ModelLLM:
    def __init__(self, embeddings, llm_model) -> None:
        self.embeddings=embeddings
        self.llm_model=llm_model
        self.data=self.data_store()
        self.get_examples(self.data)
        self._prompts()
        self._few_shot_promts()
        self.get_chain()

    def get_examples(self,examples):
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            examples,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            self.embeddings,
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=10,
        )

    def data_store(self):
        df=pd.read_excel(config.PATH_TO_DB)
        #example_df=df.sample(n=1300)
        #test_df=df.loc[~df.index.isin(example_df.index)]
        #test_df=test_df.reset_index(drop=True)
        examples=[]
        for idx,rows in df.iterrows():
            examples.append({
                "control_desc":rows[config.columns_to_consider['InputColumn']],
                "test_performed":rows[config.columns_to_consider['OutputColumn']]
            })
        return examples
    
    def _few_shot_promts(self):
        self.similar_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=self.example_selector,
        example_prompt=self.example_prompt,
        prefix="Give the Control Description every input",
        suffix=config.FEW_SHOT_PROMPT_TEMPLATE,
        input_variables=config.FEW_SHOT_PROMPT_INPUT_VAR,
    )
    def _prompts(self):
        self.example_prompt = PromptTemplate(
            input_variables=config.PROMPT_INPUT_VAR,
            template= config.PROMPT_TEMPLATE,
        )

    def get_chain(self):
        self.chain = LLMChain(llm=self.llm_model, prompt=self.similar_prompt)

    def predict(self,text):
        preds=self.chain.run(text)
        return preds


class ModelLLMPersitantStorage(ModelLLM):
    def get_examples(self, examples):
        vstore=Chroma(embedding_function=self.embeddings,persist_directory="TestV2")
        self.example_selector =SemanticSimilarityExampleSelector(
            vectorstore=vstore,
            k=10
        )