
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

def initialize_langchain(pipeline):
   template = """
                 classify the input to emotions and evaluate the overall sentiment. Answer with just \"Happy\", \"Sad\", \"Fear\", \"Anger\", \"Excitement\", \"Neutral\".
                  ```{text}```
              """

   prompt = PromptTemplate(template=template, input_variables=["text"])

   llm_chain = LLMChain(prompt=prompt, llm=pipeline)

   return llm_chain
