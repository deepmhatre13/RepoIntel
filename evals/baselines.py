import os
from packages.agentic.llm_client import LLMClient

class BaselineNaiveLLM:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def run(self, repo_path: str) -> str:
        # Load up to 5 files into simple context
        contents = []
        for root, _, files in os.walk(repo_path):
            for f in files[:5]:
                fp = os.path.join(root, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as file:
                        contents.append(f"--- File: {f} ---\n" + file.read()[:1000])
                except Exception:
                    pass

        prompt = "Generate a README for this repository based on the provided code contents:\n\n" + "\n\n".join(contents)
        return self.llm_client.complete(prompt)

class BaselineSimpleRAG:
    def __init__(self, llm_client: LLMClient, vector_store):
        self.llm_client = llm_client
        self.vector_store = vector_store

    def run(self, repo_id: str, query: str = "repository overview installation usage") -> str:
        results = self.vector_store.vector_search(repo_id, query, top_k=5)
        context = "\n\n".join([r.content for r in results])
        prompt = f"Generate documentation using the retrieved context:\n\n{context}"
        return self.llm_client.complete(prompt)
