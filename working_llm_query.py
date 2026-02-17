from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Falcon-H1R-7B-UD-Q5_K_XL.gguf",   # can be anything; llama-server usually ignores/accepts
    base_url="http://172.31.222.6:8080/v1",     # IMPORTANT: include /v1
    api_key="sk-no-key-required",            # llama.cpp doesn't need it, but client requires a value
    temperature=0.2,
)

resp = llm.invoke("Say hi in one sentence.")
print(resp.content)
