"""
Neuronpedia API SAE Implementation

Uses the Neuronpedia API to compute SAE feature activations for text inputs.

API Endpoints:
- POST /api/activation/new - Get feature activations for custom text
- POST /api/search-all - Find top features for text input
- GET /api/feature/{modelId}/{source}/{index} - Get feature details

Authentication: x-api-key header with NEURONPEDIA_API_KEY
"""

import asyncio
import os
import aiohttp
import numpy as np
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer
from typing import List, Dict, Optional

from .base_sae import BaseSAE, SAEType
from .api_sae import ApiSAE
from .utils import ensure_loaded, try_to_load_feature_labels
from ..utils.helpers import run_async_in_any_context, log_tqdm_message

load_dotenv()

# Known model configurations
NEURONPEDIA_MODELS = {
    "gpt2-small": {
        "hf_model": "gpt2",
        "sources": ["0-res_scefr-ajt", "6-res_scefr-ajt", "11-res_scefr-ajt"],
        "num_features": 49152,  # 768 * 64
    },
    "llama3.1-8b": {
        "hf_model": "meta-llama/Llama-3.1-8B",
        "sources": [f"{i}-llamascope-res-32k" for i in range(32)],
        "num_features": 32768,
    },
    "llama3.1-8b-it": {
        "hf_model": "meta-llama/Llama-3.1-8B-Instruct",
        "sources": [f"{i}-resid-post-aa" for i in [3, 7, 11, 15, 19, 23, 27]],
        "num_features": 131072,
    },
    "llama3.3-70b-it": {
        "hf_model": "meta-llama/Llama-3.3-70B-Instruct",
        "sources": ["50-resid-post-gf"],
        "num_features": 131072,
    }
}


class NeuronpediaApiSAE(ApiSAE):
    """
    SAE implementation using the Neuronpedia API for feature activation computation.

    Example:
        sae = NeuronpediaApiSAE(
            model_id="gpt2-small",
            source_id="6-res_scefr-ajt",
        )
        sae.load()
        activations = sae.encode(["Hello, world!"])
    """

    BASE_URL = "https://www.neuronpedia.org"

    def __init__(
        self,
        model_id: str,
        source_id: str,
        num_features: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.api_key = os.getenv("NEURONPEDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NEURONPEDIA_API_KEY environment variable is not set. "
                "Get your API key from https://neuronpedia.org/account"
            )

        self.model_id = model_id
        self.source_id = source_id

        self.model_config = NEURONPEDIA_MODELS.get(model_id, {})
        self.hf_model = self.model_config.get("hf_model", model_id)
        self.num_features = num_features or self.model_config.get("num_features", 32768)

    def metadata(self) -> Dict:
        parent_metadata = super().metadata()
        parent_metadata.update({
            "model_id": self.model_id,
            "source_id": self.source_id,
            "num_features": self.num_features,
            "sae_type": SAEType.NEURONPEDIA_API,
        })
        return parent_metadata

    # TODO: link to actual neuronpedia feature labels
    def load_feature_labels(self):
        label_path = f"neuronpedia/{self.model_id}/{self.source_id}.json"
        self._feature_labels = try_to_load_feature_labels(label_path)
        if self._feature_labels:
            self._feature_labels = {
                int(key): value for key, value in self._feature_labels.items()
            }

    def load_models(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
            log_tqdm_message(f"Loaded tokenizer for {self.hf_model}", level="INFO")
        except Exception as e:
            raise Exception(f"Failed to load tokenizer for {self.hf_model}: {e}\n")

    async def _get_activation(self, session: aiohttp.ClientSession, text: str) -> Dict:
        """
        Get all feature activations for a text using /api/activation/source.

        Returns the first result from the response, which contains:
        - tokens: list of token strings
        - activeFeatures: dict mapping feature_index -> list of [token_index, activation_value]
        """
        url = f"{self.BASE_URL}/api/activation/source"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
        payload = {
            "modelId": self.model_id,
            "source": self.source_id,
            "customText": text
        }

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Neuronpedia API error ({response.status}): {error_text}"
                )
            data = await response.json()
            results = data.get("results", [])
            if not results:
                raise Exception("Neuronpedia API returned empty results")
            return results[0]

    def _response_to_csr_matrix(self, response_data: Dict) -> csr_matrix:
        """
        Convert Neuronpedia /api/activation/source response to a scipy CSR sparse matrix.

        Response format:
        - tokens: list of token strings
        - activeFeatures: dict mapping feature_index (str) -> list of [token_index, activation_value]

        Returns:
            csr_matrix of shape (num_tokens, num_features)
        """
        api_tokens = response_data.get("tokens", [])
        active_features = response_data.get("activeFeatures", {})

        num_api_tokens = len(api_tokens)

        if num_api_tokens == 0:
            return csr_matrix((0, self.num_features))

        num_tokens = num_api_tokens
        rows, cols, values = [], [], []

        for feature_idx_str, token_activations in active_features.items():
            feature_idx = int(feature_idx_str)
            for pair in token_activations:
                token_idx, act_value = pair[0], pair[1]
                if act_value > 0 and token_idx >= 0:
                    rows.append(token_idx)
                    cols.append(feature_idx)
                    values.append(float(act_value))

        if not rows:
            return csr_matrix((num_tokens, self.num_features))

        # Auto-expand num_features if the API returns higher feature indices
        max_feature = max(cols)
        effective_num_features = max(self.num_features, max_feature + 1)
        if effective_num_features > self.num_features:
            self.num_features = effective_num_features

        return csr_matrix(
            (values, (rows, cols)),
            shape=(num_tokens, self.num_features)
        )

    @ensure_loaded
    def encode(self, texts):
        chat_conversations = [
            [
                {
                    "role": "assistant" if self.use_assistant_role else "user",
                    "content": text
                }
            ]
            for text in texts
        ]
        return run_async_in_any_context(self.async_encode_chat(chat_conversations))

    @ensure_loaded
    def encode_chat(self, chat_conversations):
        return run_async_in_any_context(self.async_encode_chat(chat_conversations))

    async def async_encode_chat(self, chat_conversations: List[List[Dict[str, str]]]):
        texts = []
        for conv in chat_conversations:
            if self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(conv, tokenize=False)
            else:
                text = conv[0]["content"]
            texts.append(text)

        async def get_activations(idx):
            async with aiohttp.ClientSession() as session:
                response_data = await self._get_activation(session, texts[idx])
                return self._response_to_csr_matrix(response_data)

        coroutine_funcs = [
            lambda i=i: get_activations(i)
            for i in range(len(chat_conversations))
        ]
        return await self.retry_api_with_backoff(coroutine_funcs)

    def destroy_models(self):
        pass
