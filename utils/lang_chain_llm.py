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

    # def _load_model(self):
    #     self.tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL)
    #     self.model = T5ForConditionalGeneration.from_pretrained(config.LLM_MODEL, device_map="auto",offload_folder=config.LLM_MODEL_OFFLOAD)
    #     self.pipe=pipeline('text2text-generation',model=self.model,tokenizer=self.tokenizer, **config.PARAMS)
    # def load_llm_model(self):
    #     return HuggingFacePipeline(pipeline=self.pipe)


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
    

# class Embedding:
#     def __init__(self,emb_type:str="Hugginface") -> None:
#         self.emb_type=emb_type
#     def load_embedding(self):
#         if self.emb_type=="Hugginface":
#             return HuggingFaceEmbeddings(**LLMConfig.HUGGINFACE_EMBEDDING)
        

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
        # elif self.model_name=="HugginFaceT5Model":
        #     tokenizer = T5Tokenizer.from_pretrained(LLMConfig.huggingface_model)
        #     model = T5ForConditionalGeneration.from_pretrained(LLMConfig.huggingface_model, 
        #                                                        device_map="auto",
        #                                                        offload_folder=LLMConfig.huggingface_model_offload)
        #     pipe=pipeline('text2text-generation',model=model,tokenizer=tokenizer, **LLMConfig.huggingface_model_params)
        #     HuggingFacePipeline(pipeline=pipe)
        #     return HuggingFacePipeline(pipeline=pipe)
        

# class ModelLLM:
#     def __init__(self, embeddings, llm_model) -> None:
#         self.embeddings=embeddings
#         self.llm_model=llm_model
#         self.data=self.data_store()
#         self.get_examples(self.data)
#         self._prompts()
#         self._few_shot_promts()
#         self.get_chain()

#     def get_examples(self,examples):
#         self.example_selector = SemanticSimilarityExampleSelector.from_examples(
#             # This is the list of examples available to select from.
#             examples,
#             # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
#             self.embeddings,
#             # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
#             FAISS,
#             # This is the number of examples to produce.
#             k=10,
#         )

#     def data_store(self):
#         df=pd.read_excel(config.PATH_TO_DB)
#         #example_df=df.sample(n=1300)
#         #test_df=df.loc[~df.index.isin(example_df.index)]
#         #test_df=test_df.reset_index(drop=True)
#         examples=[]
#         for idx,rows in df.iterrows():
#             examples.append({
#                 "control_desc":rows[config.columns_to_consider['InputColumn']],
#                 "test_performed":rows[config.columns_to_consider['OutputColumn']]
#             })
#         return examples
    
#     def _few_shot_promts(self):
#         self.similar_prompt = FewShotPromptTemplate(
#         # We provide an ExampleSelector instead of examples.
#         example_selector=self.example_selector,
#         example_prompt=self.example_prompt,
#         prefix="Give the Control Description every input",
#         suffix=config.FEW_SHOT_PROMPT_TEMPLATE,
#         input_variables=config.FEW_SHOT_PROMPT_INPUT_VAR,
#     )
#     def _prompts(self):
#         self.example_prompt = PromptTemplate(
#             input_variables=config.PROMPT_INPUT_VAR,
#             template= config.PROMPT_TEMPLATE,
#         )

#     def get_chain(self):
#         self.chain = LLMChain(llm=self.llm_model, prompt=self.similar_prompt)

#     def predict(self,text):
#         preds=self.chain.run(text)
#         return preds


# class ModelLLMPersitantStorage(ModelLLM):
#     def get_examples(self, examples):
#         vstore=Chroma(embedding_function=self.embeddings,persist_directory="TestV2")
#         self.example_selector =SemanticSimilarityExampleSelector(
#             vectorstore=vstore,
#             k=10
#         )


class ModelLLMTechnical:
    def __init__(self,llm_model) -> None:
        #self.embeddings=embeddings
        self.llm_model=llm_model
        self.fewshot_prompting()

    def fewshot_prompting(self):
        examples = [

        {"question" :'''The source files provided by the RevPro team which are available in the sharepoint location should adhere to the format specified.(.xlsx or .csv). Number of columns/rows and field level format is specified in the STTM document(data mapping file)
         "Testing Type": "Data Migration Testing".
        '''
        ,"answer" : '''
        Steps:
        # Count of files: a. Check if we have correct count of files in sharepoint provided by RevPro team i.e., 1 main source file and 5 cross reference files, totalling to 6 files
        # File format: a. The format of file should be .xlsx or.csv.
        # File data validation: a.Check the number of columns are present in the files as per STTM.b.Check the columns which are used as keys must be unique and non-blank.c. Check the data format for every column e.g. date fields, amount fields etc.


        Outcomes:
        # 6 source files are available in the sharepoint
        # Format of the source files is.xlsx or .csv
        # Number of columns in the source files is as per agreed in STTM. The fomrat fo data like date field, amount field etc are all as per STTM specification
        ''',
        "Testing Type": "Data Migration Testing"
        },

        {
        "question":'''Detect and eliminate duplicate records from the Oracle system prior to migration, preventing possible data redundancy and inaccurate reporting in SAP S4 HANA 
        "Testing Type": "Data Migration Testing"''',

        "answer":'''
        Steps:
        # Preparation: a. Log into the Oracle system with the necessary permissions. b. Import the sample dataset with duplicate and unique records into the Oracle system. c. Record the total count of records before any cleansing process.
        # Data Mapping and Cleansing: a. Define the criteria to identify and flag duplicate records (e.g., records with the same name, address, and contact details). b. Implement the data mapping and cleansing process to identify duplicate records within the Oracle system. c. Validate that the duplicate records have been flagged and document the number of duplicate records identified.
        # Deleting Duplicate Records: a. Implement the process to delete duplicate records flagged in step 2. b. Verify that all duplicate records have been removed from the Oracle system.
        # Verify Data Integrity and Accuracy: a. Re-run the data mapping and cleansing process to verify that no duplicate records remain in the Oracle system. b. Check for any missing or incorrectly deleted records and validate the accuracy of unique records in the Oracle system.

        Outcomes:
        # Successfully import the sample dataset into the Oracle system, and note the total count of records.
        # Duplicate records in the Oracle system are identified and flagged correctly based on the defined criteria.
        # All identified duplicate records are successfully eliminated from the Oracle system.
        # No duplicate records are found after the cleansing process, and data integrity is maintained in the Oracle system.

        '''

        },

        {
        "question": '''All the fields in "order condition segment" of Target file format and  the "source to target" mapping of each field for order condition segment should be as defined in the STTM file.All the fields in the order condition segment should be as per Data mapping sheet and follow the transformation rules of all fields from source to target For every order main row of record there can be 1 or many order conditions. The keys field in order main and order condition should be the same so that both can be integrated. Order condition will have “Condition type” field in addition which will make the relation between order main and order condition as 1:1 or 1:N.
        "Testing Type": "Data Migration Testing"''',

        "answer":'''
        Steps:
        # Number of columns: a. Check the number of columns for this segment matches as per Data mapping sheet and STTM
        # File data validation: a. Check the transformation rules for every field as per STTM.b.Check the number of key columns are present in the files as per STTM.c.Check the columns which are used as keys must be unique and non-blank.

        Outcomes:
        # Number of columns present in the segement is as per target file structure specified in STTM
        # The key fields combination is unique and non-blank.The transformation rules for every field is as per STTM'''
        
        }
        ]


        few_shot_promp = PromptTemplate(input_variables=["question", "answer"],
                                            template = "Input: {question}\nOutput: {answer} ")



        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=few_shot_promp,
            #prefix="You are an AI assistant for Manual Testing and to generate the unit test cases and execpeted resutls. ",
            prefix='''You are an tester, your job is to do  create test cases given an requirement and expected results for those test cases.
                    You will be creating all possible test scenarios, even edge cases. You are given the following data. Below are the definitions (‘#definitions’) and a few examples for previous test cases (‘#Examples’).

                    #Definations:
                    Delivery Team: The team for whom you would be writing the test case, 
                    Testing Type: Type of testing , examples: E2E testing, Data Migration Testing

                    #Examples:
                    ''',
            suffix="Question: {question}",
            input_variables=["question"]
        )

        self.chain=LLMChain(llm=self.llm_model ,prompt=prompt)

    def response(self,busines_requirements):
        output=self.chain.run({"question":busines_requirements})
        
        #Method to extract the start and end point for steps and expected results 
        wordlist=output.split()
        for i in range(len(wordlist)):
            if wordlist[i]=="Steps:":
                start=i +1
            if wordlist[i]=="Outcomes:":
                end=i

        #steps
        steps=' '.join(wordlist[start:end])
        outcomes=' '.join(wordlist[end+1:len(wordlist)])
        
        return steps,outcomes
    
        