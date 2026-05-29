import os
import time
from pathlib import Path
import hashlib
from typing import Tuple, Optional
import httpx
from openai import OpenAI
import numpy as np


class QueryAPI:
    def __init__(
        self, model: str, temp_folder: str, max_retries: int = int(1e3), cache=True
    ) -> None:
        self.temp_folder = f"{temp_folder}/query"
        Path(self.temp_folder).mkdir(parents=True, exist_ok=True)
        self.api_model = model
        self.max_retries = max_retries
        self.cache = cache
        self.queries = 0
        self.cached = 0
        self.setAPIvars()
        open_ai_key = os.environ.get("OPENAI_API_KEY")
        assert open_ai_key, "source openai_key.sh"

        self.client = OpenAI(
            base_url="#FIXME",
            api_key=False,  # type: ignore[arg-type]
            default_headers={
                "Ocp-Apim-Subscription-Key": open_ai_key,
            },
            http_client=httpx.Client(event_hooks={"request": [self.update_base_url]}),
        )

    def update_base_url(self, request: httpx.Request) -> None:
        if request.url.path == "/chat/completions" or request.url.path == "/embeddings":
            request.url = request.url.copy_with(path=self.openai_endpoint_url)

    def setAPIvars(self) -> None:
        if self.api_model == "gpt4-turbo":
            self.openai_endpoint_url = "/v1/openai/gpt4-turbo/chat/completions"
        if self.api_model == "gpt4o":
            self.openai_endpoint_url = "/v1/openai/gpt4o/chat/completions"
        if self.api_model == "gpt3.5-turbo":
            self.openai_endpoint_url = "/v1/chat"
        if self.api_model == "gpt4-8k":
            self.openai_endpoint_url = "/v1/chat/gpt4-8k"
        if self.api_model == "gpt3.5-turbo-1106":
            self.openai_endpoint_url = "/v1/chat/gpt-35-turbo-1106"
        if self.api_model == "text-embedding-3-large":
            self.openai_endpoint_url = (
                "/v1/openai/text-embedding-3-large/embeddings"
            )

    def get_hash(self, string: str) -> str:
        return hashlib.md5(string.encode("utf-8")).hexdigest()

    def query(
        self, system_prompt: str, query_prompt: str, verbose: bool = False
    ) -> str:
        sph = self.get_hash(system_prompt)
        qph = self.get_hash(query_prompt)
        mh = self.get_hash(self.api_model)
        eph = self.get_hash(self.openai_endpoint_url)
        filename = sph + "-" + qph + "-" + mh + "-" + eph
        filename = f"{self.temp_folder}/{filename}"
        exists = os.path.exists(filename)
        if exists and self.cache:
            if verbose:
                print("Load cached")
            return self.load_query(filename)
        else:
            if verbose:
                print("Perform query")
            return self.perform_query(system_prompt, query_prompt, filename)

    def load_query(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            response = f.read()
        self.cached += 1
        return response

    def perform_query(
        self, system_prompt: str, query_prompt: str, filename: str
    ) -> str:
        i = 0
        response = None
        while i < self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query_prompt},
                    ],
                )
                if completion is not None:
                    response = completion.choices[0].message.content
                    break
            except Exception as e:
                print(e)
            i += 1
            print(i)
            time.sleep(1)

        assert isinstance(response, str)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response)

        self.queries += 1
        return response

    def embed(self, query: str) -> Optional[np.ndarray]:
        q = self.get_hash(query)
        mh = self.get_hash(self.api_model)
        eph = self.get_hash(self.openai_endpoint_url)
        filename = q + "-" + mh + "-" + eph
        filename = self.temp_folder + "/" + filename + ".npy"
        exists = os.path.exists(filename)
        if exists and self.cache:
            return self.load_embed(filename)
        else:
            return self.preform_embed(query, filename)

    def load_embed(self, filename: str) -> np.ndarray:
        np_embedding = np.load(filename, allow_pickle=True)
        return np_embedding

    def preform_embed(self, query: str, filename: str) -> Optional[np.ndarray]:
        i = 0
        while i < self.max_retries:
            try:
                response = self.client.embeddings.create(
                    input=query, model=self.api_model
                )
                embedding = response.data[0].embedding
                np_embedding = np.array(embedding)
                np.save(filename, np_embedding, allow_pickle=True)

                return np_embedding
            except Exception as e:
                print(e)
            i += 1
            print(i)
            time.sleep(1)
        return None

    def get_num_queries(self) -> Tuple[int, int]:
        return self.queries, self.cached
