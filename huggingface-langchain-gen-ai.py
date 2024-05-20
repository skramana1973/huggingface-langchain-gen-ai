from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.1,
        "do_sample": True,
        "max_new_tokens": 100,
        "top_k": 50,
    },
)

resp = llm.invoke("Hugging face is")
print("********** Response resp ********")
print(resp)
print("********** Response resp END ********")

## Langchain with HuggingFaceEnd Point
load_dotenv()

key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("HUGGINGFACEHUB_API_TOKEN   :   ", key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = key

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
resp_1 = llm.invoke("Hugging Face is")

print("********** Response resp_1 ********")
print(resp_1)
print("********** Response resp_1 END ********")
