from configFolder import LLMConfig
from utils import lang_chain_llm

embeddings=lang_chain_llm.Embedding().load_embedding()
llm_gpt_4=lang_chain_llm.ModelFactory("AzureChatOpenAI",{"temperature":0}).load_model()

model=lang_chain_llm.ModelLLMPersitantStorage(embeddings,llm_gpt_4)
#print(model.predict("every system should have antivirus installed"))

