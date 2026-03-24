import pytest
import numpy as np
from scipy.sparse import csr_matrix

from interp_embed.dataset_analysis import DatasetRow


class TestDatasetRowInit:
    def test_basic_init(self):
        activations = csr_matrix(np.random.rand(5, 100))
        tokens = ["Hello", " ", "world", "!", "."]
        row = DatasetRow(
            row={"text": "Hello world!."},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )
        assert row.data == "Hello world!."
        assert row.n_tokens == 5
        assert row.field == "text"

    def test_requires_dict_row(self):
        activations = csr_matrix(np.random.rand(5, 100))
        tokens = ["Hello", " ", "world", "!", "."]
        with pytest.raises(AssertionError, match="must be a dictionary"):
            DatasetRow(
                row="not a dict",
                tokenized_document=tokens,
                activations=activations,
                field="text",
            )

    def test_requires_sparse_activations(self):
        activations = np.random.rand(5, 100)  # dense, not sparse
        tokens = ["Hello", " ", "world", "!", "."]
        with pytest.raises(AssertionError, match="must be a scipy.sparse.csr_matrix"):
            DatasetRow(
                row={"text": "Hello world!."},
                tokenized_document=tokens,
                activations=activations,
                field="text",
            )

    def test_requires_matching_token_count(self):
        activations = csr_matrix(np.random.rand(10, 100))  # 10 tokens
        tokens = ["Hello", " ", "world"]  # only 3 tokens
        with pytest.raises(AssertionError, match="must match"):
            DatasetRow(
                row={"text": "Hello world"},
                tokenized_document=tokens,
                activations=activations,
                field="text",
            )

    def test_requires_field_in_row(self):
        activations = csr_matrix(np.random.rand(3, 100))
        tokens = ["Hello", " ", "world"]
        with pytest.raises(AssertionError, match="not found in row"):
            DatasetRow(
                row={"content": "Hello world"},  # wrong field name
                tokenized_document=tokens,
                activations=activations,
                field="text",
            )

    def test_requires_nonempty_document(self):
        activations = csr_matrix(np.random.rand(0, 100))  # 0 tokens
        tokens = []
        with pytest.raises(AssertionError, match="Empty documents not allowed"):
            DatasetRow(
                row={"text": ""},
                tokenized_document=tokens,
                activations=activations,
                field="text",
            )


class TestDatasetRowLatents:
    @pytest.fixture
    def sample_row(self):
        # Create activations where we know the values
        # 5 tokens, 10 features
        data = np.array([
            [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        activations = csr_matrix(data)
        tokens = ["a", "b", "c", "d", "e"]
        return DatasetRow(
            row={"text": "abcde"},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )

    def test_latents_all(self, sample_row):
        result = sample_row.latents("all", compress=False)
        assert result.shape == (5, 10)
        assert result[0, 0] == 1.0
        assert result[1, 2] == 3.0

    def test_latents_all_compressed(self, sample_row):
        result = sample_row.latents("all", compress=True)
        assert isinstance(result, csr_matrix)
        assert result.shape == (5, 10)

    def test_latents_max(self, sample_row):
        result = sample_row.latents("max", compress=False)
        assert result.shape == (1, 10)
        assert result[0, 0] == 2.0  # max of [1.0, 0.5, 0.0, 2.0, 0.0]
        assert result[0, 2] == 3.0  # max of [2.0, 3.0, 1.0, 0.0, 0.5]

    def test_latents_sum(self, sample_row):
        result = sample_row.latents("sum", compress=False)
        assert result.shape == (1, 10)
        assert result[0, 0] == 3.5  # sum of [1.0, 0.5, 0.0, 2.0, 0.0]
        assert result[0, 2] == 6.5  # sum of [2.0, 3.0, 1.0, 0.0, 0.5]

    def test_latents_mean(self, sample_row):
        result = sample_row.latents("mean", compress=False)
        assert result.shape == (1, 10)
        np.testing.assert_almost_equal(result[0, 0], 3.5 / 5)  # sum/n_tokens
        np.testing.assert_almost_equal(result[0, 2], 6.5 / 5)

    def test_latents_binarize(self, sample_row):
        result = sample_row.latents("binarize", compress=False)
        assert result.shape == (1, 10)
        assert result[0, 0] == 1.0  # has activations
        assert result[0, 1] == 0.0  # no activations
        assert result[0, 2] == 1.0  # has activations

    def test_latents_binarize_with_threshold(self, sample_row):
        result = sample_row.latents("binarize", compress=False, activated_threshold=1.5)
        assert result.shape == (1, 10)
        assert result[0, 0] == 1.0  # max is 2.0 > 1.5
        assert result[0, 2] == 1.0  # max is 3.0 > 1.5

    def test_latents_count(self, sample_row):
        result = sample_row.latents("count", compress=False)
        assert result.shape == (1, 10)
        assert result[0, 0] == 3  # 3 tokens have activation > 0 for feature 0
        assert result[0, 2] == 4  # 4 tokens have activation > 0 for feature 2

    def test_latents_invalid_type(self, sample_row):
        with pytest.raises(ValueError, match="Unsupported"):
            sample_row.latents("invalid")


class TestDatasetRowTokenActivations:
    def test_token_activations_as_string(self):
        data = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
        ])
        activations = csr_matrix(data)
        tokens = ["Hello", " ", "world"]
        row = DatasetRow(
            row={"text": "Hello world"},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )
        result = row.token_activations(feature=0, as_string=True)
        assert "<<Hello>>" in result
        assert "<<world>>" in result

    def test_token_activations_as_list(self):
        data = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
        ])
        activations = csr_matrix(data)
        tokens = ["Hello", " ", "world"]
        row = DatasetRow(
            row={"text": "Hello world"},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )
        result = row.token_activations(feature=0, as_string=False)
        assert len(result) == 3
        assert result[0] == {"token": "Hello", "activation": 1.0}
        assert result[1] == {"token": " ", "activation": 0.0}
        assert result[2] == {"token": "world", "activation": 0.5}

    def test_token_activations_custom_markers(self):
        # Test custom markers where activation ends mid-string (not at end)
        data = np.array([[1.0], [0.0], [0.0]])
        activations = csr_matrix(data)
        tokens = ["Hello", " ", "world"]
        row = DatasetRow(
            row={"text": "Hello world"},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )
        result = row.token_activations(feature=0, left_marker="[", right_marker="]")
        assert "[Hello]" in result
        assert result == "[Hello] world"


class TestDatasetRowMethods:
    @pytest.fixture
    def sample_row(self):
        activations = csr_matrix(np.random.rand(3, 10))
        tokens = ["Hello", " ", "world"]
        return DatasetRow(
            row={"text": "Hello world", "label": "greeting"},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )

    def test_row_record(self, sample_row):
        record = sample_row.row_record()
        assert record == {"text": "Hello world", "label": "greeting"}

    def test_document(self, sample_row):
        assert sample_row.document() == "Hello world"

    def test_repr_short_text(self, sample_row):
        repr_str = repr(sample_row)
        assert "DatasetRow" in repr_str
        assert "Hello world" in repr_str

    def test_repr_long_text(self):
        long_text = "x" * 200
        activations = csr_matrix(np.random.rand(3, 10))
        tokens = ["x", "x", "x"]
        row = DatasetRow(
            row={"text": long_text},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )
        repr_str = repr(row)
        assert "..." in repr_str  # truncated

    def test_repr_with_newlines(self):
        text_with_newlines = "Hello\nworld"
        activations = csr_matrix(np.random.rand(3, 10))
        tokens = ["Hello", "\n", "world"]
        row = DatasetRow(
            row={"text": text_with_newlines},
            tokenized_document=tokens,
            activations=activations,
            field="text",
        )
        repr_str = repr(row)
        assert "\\n" in repr_str  # newline escaped
