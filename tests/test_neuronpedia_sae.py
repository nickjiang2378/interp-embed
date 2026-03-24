"""
Tests for NeuronpediaApiSAE.

Unit tests use mock data. Integration tests hit the live Neuronpedia API.
Requires NEURONPEDIA_API_KEY environment variable for integration tests.
"""
import pytest
import numpy as np
from scipy.sparse import issparse
from dotenv import load_dotenv

load_dotenv()

from interp_embed.sae.neuronpedia_sae import NeuronpediaApiSAE, NEURONPEDIA_MODELS


class TestNeuronpediaSAEInit:
    """Test initialization and configuration."""

    def test_init_known_model(self):
        sae = NeuronpediaApiSAE(model_id="gpt2-small", source_id="6-res_scefr-ajt")
        assert sae.model_id == "gpt2-small"
        assert sae.source_id == "6-res_scefr-ajt"
        assert sae.hf_model == "gpt2"
        assert sae.num_features == 49152

    def test_init_llama_model(self):
        sae = NeuronpediaApiSAE(model_id="llama3.1-8b", source_id="19-llamascope-res-32k")
        assert sae.hf_model == "meta-llama/Llama-3.1-8B"
        assert sae.num_features == 32768

    def test_init_llama_instruct_model(self):
        sae = NeuronpediaApiSAE(model_id="llama3.1-8b-it", source_id="15-resid-post-aa")
        assert sae.hf_model == "meta-llama/Llama-3.1-8B-Instruct"
        assert sae.num_features == 131072

    def test_init_custom_num_features(self):
        sae = NeuronpediaApiSAE(model_id="gpt2-small", source_id="6-res_scefr-ajt", num_features=1024)
        assert sae.num_features == 1024

    def test_metadata(self):
        sae = NeuronpediaApiSAE(model_id="gpt2-small", source_id="6-res_scefr-ajt")
        meta = sae.metadata()
        assert meta["model_id"] == "gpt2-small"
        assert meta["source_id"] == "6-res_scefr-ajt"
        assert meta["num_features"] == 49152


class TestNeuronpediaSAELoad:
    """Test loading the SAE (tokenizer)."""

    def test_load(self):
        sae = NeuronpediaApiSAE(model_id="gpt2-small", source_id="6-res_scefr-ajt")
        sae.load()
        assert sae.is_loaded()
        assert sae.tokenizer is not None


class TestResponseToCsrMatrix:
    """Test _response_to_csr_matrix parsing."""

    def setup_method(self):
        self.sae = NeuronpediaApiSAE(model_id="gpt2-small", source_id="6-res_scefr-ajt")
        self.sae.load()

    def test_empty_tokens(self):
        response = {"tokens": [], "activeFeatures": {}}
        result = self.sae._response_to_csr_matrix(response)
        assert issparse(result)
        assert result.shape == (0, 49152)

    def test_no_active_features(self):
        response = {"tokens": ["hello", " world"], "activeFeatures": {}}
        result = self.sae._response_to_csr_matrix(response)
        assert issparse(result)
        assert result.shape == (2, 49152)

    def test_with_active_features(self):
        response = {
            "tokens": ["hello", " world", "!"],
            "activeFeatures": {
                "5": [[0, 1.5], [2, 0.3]],
                "10": [[1, 2.0]],
            }
        }
        result = self.sae._response_to_csr_matrix(response)
        assert issparse(result)
        assert result.shape == (3, 49152)
        dense = result.toarray()
        assert dense[0, 5] == pytest.approx(1.5)
        assert dense[2, 5] == pytest.approx(0.3)
        assert dense[1, 10] == pytest.approx(2.0)
        assert dense[0, 10] == 0.0
        assert dense[1, 5] == 0.0

    def test_zero_activations_filtered(self):
        response = {
            "tokens": ["hello"],
            "activeFeatures": {
                "5": [[0, 0.0]],  # zero activation should be filtered
                "10": [[0, 1.0]],
            }
        }
        result = self.sae._response_to_csr_matrix(response)
        dense = result.toarray()
        assert dense[0, 5] == 0.0
        assert dense[0, 10] == pytest.approx(1.0)

    def test_auto_expand_features(self):
        """If API returns feature indices beyond num_features, the matrix expands."""
        sae = NeuronpediaApiSAE(model_id="gpt2-small", source_id="6-res_scefr-ajt", num_features=10)
        sae.load()
        response = {
            "tokens": ["hello"],
            "activeFeatures": {
                "15": [[0, 1.0]],  # beyond num_features=10
            }
        }
        result = sae._response_to_csr_matrix(response)
        assert result.shape[1] >= 16
        assert sae.num_features >= 16


class TestNeuronpediaSAEEncode:
    """Integration tests hitting the live Neuronpedia API with gpt2-small."""

    @pytest.fixture(autouse=True)
    def setup_sae(self):
        self.sae = NeuronpediaApiSAE(model_id="gpt2-small", source_id="6-res_scefr-ajt")
        self.sae.load()

    def test_encode_single_text(self):
        results = self.sae.encode(["Hello, world!"])
        assert len(results) == 1
        matrix = results[0]
        assert issparse(matrix)
        assert matrix.shape[1] == self.sae.num_features
        assert matrix.shape[0] > 0
        assert matrix.nnz > 0

    def test_encode_multiple_texts(self):
        texts = ["The cat sat on the mat.", "Machine learning is fascinating."]
        results = self.sae.encode(texts)
        assert len(results) == 2
        for matrix in results:
            assert issparse(matrix)
            assert matrix.shape[1] == self.sae.num_features
            assert matrix.nnz > 0

    def test_encode_short_text(self):
        results = self.sae.encode(["Hi"])
        assert len(results) == 1
        matrix = results[0]
        assert issparse(matrix)
        assert matrix.shape[0] > 0


class TestNeuronpediaSAEEncodeLlama:
    """Integration tests hitting the live Neuronpedia API with llama3.1-8b-it."""

    @pytest.fixture(autouse=True)
    def setup_sae(self):
        self.sae = NeuronpediaApiSAE(model_id="llama3.1-8b-it", source_id="15-resid-post-aa")
        self.sae.load()

    def test_encode_single_text(self):
        results = self.sae.encode(["Hello, world!"])
        assert len(results) == 1
        matrix = results[0]
        assert issparse(matrix)
        assert matrix.shape[1] == self.sae.num_features
        assert matrix.shape[0] > 0
        assert matrix.nnz > 0

    def test_encode_multiple_texts(self):
        texts = ["The cat sat on the mat.", "Machine learning is fascinating."]
        results = self.sae.encode(texts)
        assert len(results) == 2
        for matrix in results:
            assert issparse(matrix)
            assert matrix.shape[1] == self.sae.num_features
            assert matrix.nnz > 0
