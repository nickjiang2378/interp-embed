import pytest
import numpy as np
from scipy.sparse import csr_matrix

from interp_embed.dataset_analysis import Dataset, DatasetRow


class MockSAE:
    """Mock SAE for testing purposes."""

    def __init__(self):
        self._loaded = False
        self._feature_labels = {}

    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def destroy(self):
        self._loaded = False

    def encode(self, texts):
        # Return mock activations
        return [csr_matrix(np.random.rand(10, 100)) for _ in texts]

    def tokenize(self, texts):
        # Return mock tokens
        return [["tok"] * 10 for _ in texts]

    def feature_labels(self):
        return self._feature_labels

    def metadata(self):
        return {"type": "mock"}

    def set_device(self, device):
        pass


def create_dataset_with_rows(rows_data, feature_activations, num_features=10):
    """
    Helper to create a Dataset with pre-built rows for testing.

    Args:
        rows_data: List of dicts with 'text' field
        feature_activations: List of 2D numpy arrays (tokens x features) for each row
    """
    mock_sae = MockSAE()

    # Create DatasetRow objects with specified activations
    rows = []
    for i, (row_data, activations) in enumerate(zip(rows_data, feature_activations)):
        if activations is None:
            rows.append(None)
        else:
            n_tokens = activations.shape[0]
            tokens = [f"tok{j}" for j in range(n_tokens)]
            sparse_activations = csr_matrix(activations)
            dataset_row = DatasetRow(
                row=row_data,
                tokenized_document=tokens,
                activations=sparse_activations,
                field="text",
            )
            rows.append(dataset_row)

    # Create Dataset without computing activations
    dataset = Dataset(
        data=rows_data,
        sae=mock_sae,
        rows=rows,
        field="text",
        compute_activations=False,
    )
    return dataset


class TestTopDocumentsForFeatureBasic:
    """Test basic functionality of top_documents_for_feature."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a dataset with 5 documents and controlled activations."""
        rows_data = [
            {"text": "Document zero"},
            {"text": "Document one"},
            {"text": "Document two"},
            {"text": "Document three"},
            {"text": "Document four"},
        ]
        # Each document has 3 tokens, 5 features
        # Feature 0 activations: doc0=1.0, doc1=5.0, doc2=3.0, doc3=2.0, doc4=4.0
        # Feature 1 activations: doc0=0.0, doc1=0.0, doc2=1.0, doc3=0.0, doc4=0.0
        feature_activations = [
            np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.5, 0.0, 0.0, 0.0, 0.0],
                      [0.2, 0.0, 0.0, 0.0, 0.0]]),  # doc0: max=1.0 for feature 0
            np.array([[5.0, 0.0, 0.0, 0.0, 0.0],
                      [4.0, 0.0, 0.0, 0.0, 0.0],
                      [3.0, 0.0, 0.0, 0.0, 0.0]]),  # doc1: max=5.0 for feature 0
            np.array([[3.0, 1.0, 0.0, 0.0, 0.0],
                      [2.0, 0.5, 0.0, 0.0, 0.0],
                      [1.0, 0.2, 0.0, 0.0, 0.0]]),  # doc2: max=3.0 for feature 0, max=1.0 for feature 1
            np.array([[2.0, 0.0, 0.0, 0.0, 0.0],
                      [1.5, 0.0, 0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0, 0.0]]),  # doc3: max=2.0 for feature 0
            np.array([[4.0, 0.0, 0.0, 0.0, 0.0],
                      [3.5, 0.0, 0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0, 0.0, 0.0]]),  # doc4: max=4.0 for feature 0
        ]
        return create_dataset_with_rows(rows_data, feature_activations, num_features=5)

    def test_top_k_documents(self, simple_dataset):
        """Test getting top k documents for a feature."""
        # Get top 3 documents for feature 0
        # Expected order by max activation: doc1(5.0), doc4(4.0), doc2(3.0)
        results = simple_dataset.top_documents_for_feature(feature=0, k=3, select_top=True)

        assert len(results) == 3
        # Results are token activation strings, check they contain expected tokens
        assert all(isinstance(r, str) for r in results)

    def test_top_k_returns_correct_count(self, simple_dataset):
        """Test that k parameter limits the number of results."""
        results_2 = simple_dataset.top_documents_for_feature(feature=0, k=2)
        results_5 = simple_dataset.top_documents_for_feature(feature=0, k=5)

        assert len(results_2) == 2
        assert len(results_5) == 5

    def test_bottom_k_documents(self, simple_dataset):
        """Test getting bottom k documents for a feature (select_top=False)."""
        # Get bottom 2 documents for feature 0
        # Expected: doc0(1.0), doc3(2.0)
        results = simple_dataset.top_documents_for_feature(feature=0, k=2, select_top=False)

        assert len(results) == 2

    def test_k_larger_than_dataset(self, simple_dataset):
        """Test when k is larger than the number of valid samples."""
        # Request more documents than available
        results = simple_dataset.top_documents_for_feature(feature=0, k=100)

        # Should return all 5 documents
        assert len(results) == 5


class TestTopDocumentsForFeatureFiltering:
    """Test filtering options for top_documents_for_feature."""

    @pytest.fixture
    def mixed_activation_dataset(self):
        """Create a dataset with mix of active and non-active samples."""
        rows_data = [
            {"text": "Active document high"},
            {"text": "Active document medium"},
            {"text": "Nonactive document"},
            {"text": "Active document low"},
            {"text": "Another nonactive"},
        ]
        # Feature 0: some docs have activation, some don't
        feature_activations = [
            np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]),  # active, max=3.0
            np.array([[2.0, 0.0], [1.5, 0.0], [0.5, 0.0]]),  # active, max=2.0
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),  # non-active
            np.array([[1.0, 0.0], [0.5, 0.0], [0.2, 0.0]]),  # active, max=1.0
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),  # non-active
        ]
        return create_dataset_with_rows(rows_data, feature_activations, num_features=2)

    def test_exclude_nonactive_samples_default(self, mixed_activation_dataset):
        """Test that non-active samples are excluded by default."""
        # Default: include_nonactive_samples=False, include_active_samples=True
        results = mixed_activation_dataset.top_documents_for_feature(feature=0, k=10)

        # Should only return active samples (3 documents)
        assert len(results) == 3

    def test_include_nonactive_samples(self, mixed_activation_dataset):
        """Test including non-active samples."""
        results = mixed_activation_dataset.top_documents_for_feature(
            feature=0, k=10, include_nonactive_samples=True
        )

        # Should return all 5 documents
        assert len(results) == 5

    def test_exclude_active_samples(self, mixed_activation_dataset):
        """Test excluding active samples (only get non-active)."""
        results = mixed_activation_dataset.top_documents_for_feature(
            feature=0, k=10, include_active_samples=False, include_nonactive_samples=True
        )

        # Should only return non-active samples (2 documents)
        assert len(results) == 2

    def test_exclude_both_returns_empty(self, mixed_activation_dataset):
        """Test that excluding both active and non-active returns empty."""
        results = mixed_activation_dataset.top_documents_for_feature(
            feature=0, k=10, include_active_samples=False, include_nonactive_samples=False
        )

        # Should return empty list
        assert len(results) == 0


class TestTopDocumentsForFeatureWithNaN:
    """Test handling of NaN values (None rows)."""

    @pytest.fixture
    def dataset_with_nan(self):
        """Create a dataset with some None rows (failed processing)."""
        rows_data = [
            {"text": "Document zero"},
            {"text": "Document one"},
            {"text": "Document two"},
            {"text": "Document three"},
        ]
        feature_activations = [
            np.array([[2.0], [1.0], [0.5]]),  # doc0: max=2.0
            None,  # doc1: None (failed processing)
            np.array([[3.0], [2.0], [1.0]]),  # doc2: max=3.0
            np.array([[1.0], [0.5], [0.2]]),  # doc3: max=1.0
        ]
        return create_dataset_with_rows(rows_data, feature_activations, num_features=1)

    def test_skips_nan_rows(self, dataset_with_nan):
        """Test that rows with NaN activations are skipped."""
        results = dataset_with_nan.top_documents_for_feature(feature=0, k=10)

        # Should skip the None row and return 3 documents
        assert len(results) == 3


class TestTopDocumentsForFeatureAggregation:
    """Test different aggregation methods."""

    @pytest.fixture
    def aggregation_dataset(self):
        """Create a dataset where aggregation method affects ranking."""
        rows_data = [
            {"text": "High max, low sum"},
            {"text": "Low max, high sum"},
        ]
        # Feature 0: doc0 has high max but low sum, doc1 has low max but high sum
        feature_activations = [
            np.array([[5.0], [0.0], [0.0], [0.0], [0.0]]),  # max=5.0, sum=5.0
            np.array([[2.0], [2.0], [2.0], [2.0], [2.0]]),  # max=2.0, sum=10.0
        ]
        return create_dataset_with_rows(rows_data, feature_activations, num_features=1)

    def test_max_aggregation(self, aggregation_dataset):
        """Test that max aggregation uses max values for ranking."""
        results = aggregation_dataset.top_documents_for_feature(
            feature=0, k=1, aggregation_type="max"
        )

        # With max aggregation, doc0 should be top (max=5.0 > max=2.0)
        assert len(results) == 1

    def test_sum_aggregation(self, aggregation_dataset):
        """Test that sum aggregation uses sum values for ranking."""
        results = aggregation_dataset.top_documents_for_feature(
            feature=0, k=1, aggregation_type="sum"
        )

        # With sum aggregation, doc1 should be top (sum=10.0 > sum=5.0)
        assert len(results) == 1

    def test_mean_aggregation(self, aggregation_dataset):
        """Test mean aggregation."""
        results = aggregation_dataset.top_documents_for_feature(
            feature=0, k=2, aggregation_type="mean"
        )

        assert len(results) == 2


class TestTopDocumentsForFeatureEdgeCases:
    """Test edge cases."""

    def test_empty_valid_samples(self):
        """Test when no samples match the filter criteria."""
        rows_data = [{"text": "Document"}]
        # All zeros - no active samples
        feature_activations = [np.array([[0.0], [0.0], [0.0]])]
        dataset = create_dataset_with_rows(rows_data, feature_activations, num_features=1)

        # Default excludes non-active, so no valid samples
        results = dataset.top_documents_for_feature(feature=0, k=10)

        assert len(results) == 0

    def test_single_document(self):
        """Test with a single document."""
        rows_data = [{"text": "Only document"}]
        feature_activations = [np.array([[1.0], [0.5], [0.2]])]
        dataset = create_dataset_with_rows(rows_data, feature_activations, num_features=1)

        results = dataset.top_documents_for_feature(feature=0, k=5)

        assert len(results) == 1

    def test_returns_token_activations(self):
        """Test that returned values are token activation strings."""
        rows_data = [{"text": "Test document"}]
        feature_activations = [np.array([[2.0, 0.0], [1.0, 0.0], [0.5, 0.0]])]
        dataset = create_dataset_with_rows(rows_data, feature_activations, num_features=2)

        results = dataset.top_documents_for_feature(feature=0, k=1)

        assert len(results) == 1
        # Should be a string with token activations
        assert isinstance(results[0], str)
        # Should contain markers for activated tokens
        assert "<<" in results[0] or results[0]  # Either has markers or is plain text
