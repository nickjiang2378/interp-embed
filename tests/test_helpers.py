import pytest
import numpy as np
import tempfile
import os
from scipy.sparse import csr_matrix

from interp_embed.utils.helpers import (
    convert_text_to_dict,
    truncate_chat_template_activations,
    truncate_chat_template_tokens,
    activation_dict_to_string,
    sets_are_equal,
    highlight_activations_as_string,
    token_count_as_string,
    safe_save_pkl,
    safe_load_pkl,
    dict_astype,
    CHAT_TEMPLATE_END_POSITION_TOKENS,
    CHAT_TEMPLATE_END_POSITION_ACTIVATIONS,
)


class TestConvertTextToDict:
    def test_basic_conversion(self):
        texts = ["hello", "world"]
        result = convert_text_to_dict(texts)
        assert result == [{"text": "hello"}, {"text": "world"}]

    def test_custom_field(self):
        texts = ["hello", "world"]
        result = convert_text_to_dict(texts, text_field="content")
        assert result == [{"content": "hello"}, {"content": "world"}]

    def test_empty_list(self):
        result = convert_text_to_dict([])
        assert result == []


class TestTruncateChatTemplate:
    def test_truncate_activations_keep_eot(self):
        activations = np.arange(50).reshape(50, 1)
        result = truncate_chat_template_activations(activations, remove_eot_token=False)
        expected_start = CHAT_TEMPLATE_END_POSITION_ACTIVATIONS + 1
        assert result.shape[0] == 50 - expected_start
        assert result[0, 0] == expected_start

    def test_truncate_activations_remove_eot(self):
        activations = np.arange(50).reshape(50, 1)
        result = truncate_chat_template_activations(activations, remove_eot_token=True)
        expected_start = CHAT_TEMPLATE_END_POSITION_ACTIVATIONS + 1
        # Should remove both start tokens and last token
        assert result.shape[0] == 50 - expected_start - 1

    def test_truncate_tokens(self):
        tokens = [f"tok_{i}" for i in range(50)]
        result = truncate_chat_template_tokens(tokens)
        expected_start = CHAT_TEMPLATE_END_POSITION_TOKENS + 1
        assert len(result) == 50 - expected_start
        assert result[0] == f"tok_{expected_start}"


class TestActivationDictToString:
    def test_with_activations(self):
        activation_dict = [
            {"token": "Hello", "activation": 0.5},
            {"token": " ", "activation": 0.0},
            {"token": "world", "activation": 0.8},
        ]
        result = activation_dict_to_string(activation_dict)
        assert "[token: Hello, activation: 0.5]" in result
        assert " " in result  # space token with 0 activation
        assert "[token: world, activation: 0.8]" in result

    def test_no_activations(self):
        activation_dict = [
            {"token": "Hello", "activation": 0.0},
            {"token": " ", "activation": 0.0},
        ]
        result = activation_dict_to_string(activation_dict)
        assert result == "Hello "

    def test_empty_list(self):
        result = activation_dict_to_string([])
        assert result == ""


class TestSetsAreEqual:
    def test_equal_sets(self):
        assert sets_are_equal({1, 2, 3}, {3, 2, 1})

    def test_unequal_sets(self):
        assert not sets_are_equal({1, 2}, {1, 2, 3})

    def test_empty_sets(self):
        assert sets_are_equal(set(), set())


class TestHighlightActivationsAsString:
    def test_basic_highlighting(self):
        tokens = ["Hello", " ", "world", "!"]
        activations = np.array([0.5, 0.0, 0.8, 0.0])
        result = highlight_activations_as_string(tokens, activations, "<<", ">>")
        assert result == "<<Hello>> <<world>>!"

    def test_consecutive_activations(self):
        tokens = ["Hello", " ", "world"]
        activations = np.array([0.5, 0.3, 0.8])
        result = highlight_activations_as_string(tokens, activations, "<<", ">>")
        assert result == "<<Hello world>>"

    def test_no_activations(self):
        tokens = ["Hello", " ", "world"]
        activations = np.array([0.0, 0.0, 0.0])
        result = highlight_activations_as_string(tokens, activations, "<<", ">>")
        assert result == "Hello world"

    def test_all_activations(self):
        tokens = ["Hello", " ", "world"]
        activations = np.array([0.5, 0.3, 0.8])
        result = highlight_activations_as_string(tokens, activations, "<<", ">>")
        assert result == "<<Hello world>>"

    def test_activation_at_end(self):
        tokens = ["Hello", " ", "world"]
        activations = np.array([0.0, 0.0, 0.8])
        result = highlight_activations_as_string(tokens, activations, "<<", ">>")
        assert result == "Hello <<world>>"


class TestTokenCountAsString:
    def test_small_number(self):
        assert token_count_as_string(500) == "500"

    def test_thousands(self):
        assert token_count_as_string(1500) == "1.5k"
        assert token_count_as_string(12300) == "12.3k"

    def test_millions(self):
        assert token_count_as_string(1_500_000) == "1m"
        assert token_count_as_string(5_000_000) == "5m"

    def test_edge_cases(self):
        assert token_count_as_string(999) == "999"
        assert token_count_as_string(1000) == "1.0k"
        assert token_count_as_string(999_999) == "999.9k"


class TestSafePickle:
    def test_save_and_load(self):
        data = {"key": "value", "numbers": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pkl")
            safe_save_pkl(data, path)
            loaded = safe_load_pkl(path)
            assert loaded == data

    def test_save_creates_directory(self):
        data = {"test": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "test.pkl")
            safe_save_pkl(data, path)
            assert os.path.exists(path)
            loaded = safe_load_pkl(path)
            assert loaded == data

    def test_save_numpy_array(self):
        data = {"array": np.array([1, 2, 3])}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pkl")
            safe_save_pkl(data, path)
            loaded = safe_load_pkl(path)
            np.testing.assert_array_equal(loaded["array"], data["array"])


class TestDictAstype:
    def test_convert_float64_to_float32(self):
        data = {
            "a": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "b": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        }
        result = dict_astype(data, np.float32)
        assert result["a"].dtype == np.float32
        assert result["b"].dtype == np.float32

    def test_convert_sparse_matrix(self):
        data = {
            "sparse": csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        }
        result = dict_astype(data, np.float32)
        assert result["sparse"].dtype == np.float32
