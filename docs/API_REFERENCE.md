# interp_embed API Reference

This document provides a comprehensive API reference for the `interp_embed` package, which is designed for analyzing datasets using Sparse Autoencoders (SAEs). The package enables computing feature activations over text data, labeling features using LLMs, and performing various dataset operations.

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Dataset Class](#dataset-class)
3. [DatasetRow Class](#datasetrow-class)
4. [SAE Classes](#sae-classes)
   - [LocalSAE](#localsae)
   - [GoodfireSAE](#goodfiresae)

---

## Package Overview

The `interp_embed` package provides tools for:

- Computing sparse autoencoder feature activations over text datasets
- Aggregating and analyzing feature activations at both document and token levels
- Labeling and scoring SAE features using LLM-based evaluation
- Saving and loading datasets with computed activations
- Supporting multiple SAE backends (local models, Goodfire API, etc.)

### Installation and Import

```python
from interp_embed import Dataset
from interp_embed.sae import BaseSAE, LocalSAE, GoodfireSAE, ApiSAE, GoodfireApiSAE
```

---

## Dataset Class

The `Dataset` class is the primary interface for working with text data and SAE feature activations. It wraps a collection of text documents, computes feature activations using a specified SAE, and provides methods for analysis, sorting, filtering, and feature labeling.

### Constructor

```python
Dataset(
    data,
    sae,
    dataset_description="",
    rows=None,
    field="text",
    compute_activations=True,
    feature_labels=None,
    save_path=None,
    save_every_batch=5,
    batch_size=8
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` or `list[dict]` | Required | The input data containing text documents. If a DataFrame, it will be converted to a list of dictionaries. Each dictionary must contain the field specified by `field`. |
| `sae` | `BaseSAE` subclass | Required | An SAE instance used to compute feature activations, tokenize data, and provide feature labels. Must be a subclass of `BaseSAE`. |
| `dataset_description` | `str` | `""` | Optional human-readable description of the dataset. |
| `rows` | `list[DatasetRow]` or `None` | `None` | Optional pre-computed `DatasetRow` objects. If `None`, rows will be initialized as empty and computed if `compute_activations=True`. |
| `field` | `str` | `"text"` | The key/column name in the data that contains the text to analyze. |
| `compute_activations` | `bool` | `True` | Whether to compute feature activations upon initialization. Set to `False` if loading pre-computed activations. |
| `feature_labels` | `dict[int, str]` or `None` | `None` | Optional dictionary mapping feature indices to human-readable labels. If `None`, labels will be loaded from the SAE if available. |
| `save_path` | `str` or `None` | `None` | Optional file path for saving intermediate results during activation computation. Enables recovery if the computation fails partway through. |
| `save_every_batch` | `int` | `5` | Number of batches between automatic saves when `save_path` is provided. |
| `batch_size` | `int` | `8` | Number of documents to process in each batch when computing activations. |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | A unique 6-character identifier for this dataset instance. |
| `dataset` | `pd.DataFrame` | The underlying pandas DataFrame containing the original data. |
| `num_documents` | `int` | The total number of documents in the dataset. |
| `field` | `str` | The field name containing the text data. |
| `sae` | `BaseSAE` | The SAE instance used for encoding. |
| `rows` | `list[DatasetRow or None]` | List of `DatasetRow` objects containing per-document activations. |
| `token_count` | `int` | Total number of tokens processed across all documents. |
| `columns` | `list[str]` | List of column names from the underlying DataFrame (property). |

---

### Instance Methods

#### `save_to_file(file_path=None, dtype=np.float32)`

Saves the dataset (including computed activations) to a pickle file.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` or `None` | `None` | Path to save the file. If `None`, defaults to `dataset_{id}.pkl`. |
| `dtype` | `np.dtype` | `np.float32` | Data type for storing activation values. Use `np.float16` to reduce file size. |

**Returns:** `None`

**Example:**
```python
dataset.save_to_file("my_dataset.pkl")
```

---

#### `feature_labels()`

Returns the dictionary of feature labels.

**Returns:** `dict[int, str]` - A dictionary mapping feature indices (integers) to their human-readable labels (strings).

**Example:**
```python
labels = dataset.feature_labels()
print(labels.get(42, "Unknown feature"))
```

---

#### `latents(aggregation_method="max", compress=False, activated_threshold=0)`

Retrieves the feature activations for all samples in the dataset, aggregated according to the specified method.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregation_method` | `str` | `"max"` | Method for aggregating per-token activations into per-document activations. Options: `"max"`, `"mean"`, `"sum"`, `"binarize"`, `"count"`, `"all"`. |
| `compress` | `bool` | `False` | If `True`, return sparse matrices (`csr_matrix`). If `False`, return dense numpy arrays. |
| `activated_threshold` | `float` | `0` | Threshold for determining if a feature is "activated" (used with `"binarize"` and `"count"` methods). |

**Aggregation Methods:**

| Method | Description |
|--------|-------------|
| `"max"` | Maximum activation value across all tokens in the document. |
| `"mean"` | Mean activation value across all tokens. |
| `"sum"` | Sum of activation values across all tokens. |
| `"binarize"` | Binary indicator (1 if any token exceeds `activated_threshold`, else 0). |
| `"count"` | Number of tokens with activation above `activated_threshold`. |
| `"all"` | Returns all per-token activations without aggregation (list of matrices). |

**Returns:**
- If `aggregation_method != "all"`: `np.ndarray` (shape: `[num_documents, d_sae]`) or `csr_matrix` if `compress=True`.
- If `aggregation_method == "all"`: `np.ndarray` of objects or list of `csr_matrix`.

**Example:**
```python
# Get max activations as dense array
max_activations = dataset.latents("max")

# Get binary indicators as sparse matrix
binary_sparse = dataset.latents("binarize", compress=True, activated_threshold=0.5)
```

---

#### `top_documents_for_feature(feature, aggregation_type="max", document_only=True, k=10, select_top=True, include_nonactive_samples=False)`

Retrieves the top (or bottom) k documents for a specific feature.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature` | `int` | Required | The feature index to analyze. |
| `aggregation_type` | `str` | `"max"` | Aggregation method for comparing documents. |
| `document_only` | `bool` | `True` | If `True`, return token activation strings. If `False`, return full `DatasetRow` objects. |
| `k` | `int` | `10` | Number of documents to return. |
| `select_top` | `bool` | `True` | If `True`, return highest-activating documents. If `False`, return lowest. |
| `include_nonactive_samples` | `bool` | `False` | Whether to include samples where the feature did not activate. |

**Returns:**
- If `document_only=True`: `list[str]` - Token activation strings with highlighted tokens.
- If `document_only=False`: `list[DatasetRow]` - Full DatasetRow objects.

**Example:**
```python
# Get top 5 documents for feature 123
top_docs = dataset.top_documents_for_feature(123, k=5)
for doc in top_docs:
    print(doc)  # Prints text with <<highlighted>> activated tokens
```

---

#### `token_activations(feature)`

Returns token-level activation strings for all documents in the dataset for a specific feature.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `feature` | `int` | The feature index to analyze. |

**Returns:** `list[str]` - List of strings where activated tokens are surrounded by `<<` and `>>` markers.

**Example:**
```python
activations = dataset.token_activations(42)
# Returns: ["The <<cat>> sat on the mat", "Dogs and <<cats>> are pets", ...]
```

---

#### `sort_by_columns(columns, descending=True)`

Sorts the dataset by the specified DataFrame columns.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | `list[str]` | Required | Column names to sort by. |
| `descending` | `bool` | `True` | Sort order. |

**Returns:** `Dataset` - A new Dataset instance with sorted rows.

**Example:**
```python
sorted_ds = dataset.sort_by_columns(["score", "date"], descending=True)
```

---

#### `sort_by_features(features, aggregation_type="max", descending=True, include_top_feature=True, include_nonactive_samples=True)`

Sorts the dataset by activation values for the specified features.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features` | `list[int]` | Required | List of feature indices to sort by. |
| `aggregation_type` | `str` | `"max"` | Aggregation method for feature activations. |
| `descending` | `bool` | `True` | Sort order (highest activations first if `True`). |
| `include_top_feature` | `bool` | `True` | If `True`, adds columns `feature_activation`, `top_feature_label`, and `top_feature_index` to each row. |
| `include_nonactive_samples` | `bool` | `True` | Whether to include samples where none of the features activated. |

**Returns:** `Dataset` - A new Dataset instance sorted by feature activations.

**Example:**
```python
# Sort by features 10, 20, 30 in descending order
sorted_ds = dataset.sort_by_features([10, 20, 30])
```

---

#### `filter_na_rows()`

Returns a new Dataset with rows that have `None` activations removed.

**Returns:** `Dataset` - A new Dataset with only successfully processed rows.

**Example:**
```python
clean_dataset = dataset.filter_na_rows()
```

---

#### `dataset_rows()`

Returns the list of `DatasetRow` objects.

**Returns:** `list[DatasetRow or None]`

---

#### `pandas()`

Returns the underlying pandas DataFrame.

**Returns:** `pd.DataFrame`

---

#### `list()`

Returns the data as a list of dictionaries.

**Returns:** `list[dict]`

---

#### `documents()`

Returns a list of the text content from the specified field.

**Returns:** `list[str]`

---

### Async Methods

#### `async label_feature(feature, model="google/gemini-2.5-flash", label_and_score=None, positive_dataset=None, negative_dataset=None, k=20)`

Uses an LLM to generate a human-readable label for a feature based on positive and negative samples.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature` | `int` | Required | The feature index to label. |
| `model` | `str` | `"google/gemini-2.5-flash"` | LLM model identifier. Use `"openai/..."` prefix for OpenAI models. |
| `label_and_score` | `dict` or `None` | `None` | Optional previous label and score for refinement. |
| `positive_dataset` | `Dataset` or `None` | `None` | Dataset to draw positive samples from. Defaults to `self`. |
| `negative_dataset` | `Dataset` or `None` | `None` | Dataset to draw negative samples from. Defaults to `self`. |
| `k` | `int` | `20` | Number of positive and negative samples to use. |

**Returns:** `FeatureLabelResponse` - Pydantic model containing:
- `label`: Concise phrase describing the feature
- `brief_description`: One-sentence expansion of the label
- `detailed_explanation`: Extended explanation (optional)

**Example:**
```python
import asyncio

async def label_features():
    label = await dataset.label_feature(42, model="google/gemini-2.5-flash")
    print(f"Feature 42: {label.label}")
    print(f"Description: {label.brief_description}")

asyncio.run(label_features())
```

---

#### `async score_feature(feature, label, model="google/gemini-2.5-flash", positive_dataset=None, negative_dataset=None, k=10)`

Scores how well a label describes a feature by evaluating it against positive and negative samples.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature` | `int` | Required | The feature index to score. |
| `label` | `str` | Required | The feature label/description to evaluate. |
| `model` | `str` | `"google/gemini-2.5-flash"` | LLM model identifier. |
| `positive_dataset` | `Dataset` or `None` | `None` | Dataset to draw positive samples from. |
| `negative_dataset` | `Dataset` or `None` | `None` | Dataset to draw negative samples from. |
| `k` | `int` | `10` | Number of samples to evaluate. |

**Returns:** `dict` containing:
- `score`: Float between 0 and 1 indicating label accuracy
- `total_count`: Number of samples evaluated
- `responses`: List of individual `SingleSampleScoringResponse` objects

**Example:**
```python
result = await dataset.score_feature(42, "References to cats or felines")
print(f"Score: {result['score']:.2%}")
```

---

### Class Methods

#### `Dataset.load_from_file(file_path, resume=False, batch_size=8, device="cuda:0")`

Loads a Dataset from a previously saved pickle file.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | Required | Path to the saved dataset file. |
| `resume` | `bool` | `False` | If `True`, continues computing activations for any unprocessed rows. |
| `batch_size` | `int` | `8` | Batch size for resumed computation. |
| `device` | `str` | `"cuda:0"` | Device to use for the SAE model. |

**Returns:** `Dataset` - The loaded Dataset instance.

**Example:**
```python
# Load a saved dataset
dataset = Dataset.load_from_file("my_dataset.pkl")

# Load and resume processing incomplete rows
dataset = Dataset.load_from_file("my_dataset.pkl", resume=True)
```

---

### Indexing and Iteration

The `Dataset` class supports various forms of indexing and iteration:

```python
# Integer indexing - returns a DatasetRow
row = dataset[0]

# Slice indexing - returns a new Dataset
subset = dataset[10:20]

# Boolean array indexing
mask = np.array([True, False, True, ...])
filtered = dataset[mask]

# Integer array indexing
indices = np.array([0, 5, 10, 15])
selected = dataset[indices]

# Iteration
for row in dataset:
    print(row.document())

# Length
num_samples = len(dataset)
```

---

## DatasetRow Class

The `DatasetRow` class represents a single document with its computed feature activations. It is typically created internally by the `Dataset` class.

### Constructor

```python
DatasetRow(
    row,
    tokenized_document,
    activations,
    truncate_chat_template=False,
    aggregate_activations=None,
    field="text",
    low_memory=False
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `row` | `dict` | Required | Dictionary containing the document data. |
| `tokenized_document` | `list[str]` | Required | List of token strings for the document. |
| `activations` | `csr_matrix` | Required | Sparse matrix of shape `[num_tokens, d_sae]` containing per-token feature activations. |
| `truncate_chat_template` | `bool` | `False` | Whether to remove chat template tokens from output. |
| `aggregate_activations` | `dict` or `None` | `None` | Pre-computed aggregate activations. |
| `field` | `str` | `"text"` | Field name containing the text in `row`. |
| `low_memory` | `bool` | `False` | If `True`, skip pre-computing aggregate activations to save memory. |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `str` | The text content of the document. |
| `field` | `str` | The field name containing the text. |
| `row` | `dict` | The full row dictionary. |
| `tokenized_document` | `list[str]` | List of tokens. |
| `n_tokens` | `int` | Number of tokens in the document. |
| `activations` | `csr_matrix` | Raw per-token activations (sparse matrix). |
| `aggregate_activations` | `dict` | Dictionary of pre-computed aggregations (`"max"`, `"sum"`). |

---

### Methods

#### `row_record()`

Returns the original row dictionary.

**Returns:** `dict`

---

#### `latents(activation_type="max", compress=False, activated_threshold=0)`

Retrieves feature activations for this document.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `activation_type` | `str` | `"max"` | Aggregation method: `"max"`, `"mean"`, `"sum"`, `"binarize"`, `"count"`, or `"all"`. |
| `compress` | `bool` | `False` | If `True`, return sparse matrix. If `False`, return dense array. |
| `activated_threshold` | `float` | `0` | Threshold for `"binarize"` and `"count"` methods. |

**Returns:** `np.ndarray` or `csr_matrix` - Feature activation vector or matrix.

---

#### `token_activations(feature, as_string=True, left_marker="<<", right_marker=">>")`

Gets token-level activations for a specific feature.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature` | `int` | Required | The feature index. |
| `as_string` | `bool` | `True` | If `True`, return a string with markers. If `False`, return list of dicts. |
| `left_marker` | `str` | `"<<"` | Left marker for highlighting activated tokens. |
| `right_marker` | `str` | `">>"` | Right marker for highlighting activated tokens. |

**Returns:**
- If `as_string=True`: `str` - Text with activated tokens surrounded by markers.
- If `as_string=False`: `list[dict]` - List of `{"token": str, "activation": float}` dicts.

**Example:**
```python
# As string
text = row.token_activations(42)
# Returns: "The <<cat>> sat on the <<mat>>"

# As structured data
data = row.token_activations(42, as_string=False)
# Returns: [{"token": "The", "activation": 0.0}, {"token": "cat", "activation": 2.5}, ...]
```

---

#### `document()`

Returns the text content of the document.

**Returns:** `str`

---

## SAE Classes

The package provides SAE classes for different backends and use cases.

### LocalSAE

A local SAE implementation using SAE-Lens and TransformerLens.

```python
from interp_embed.sae import LocalSAE
```

#### Constructor

```python
LocalSAE(
    sae_id="blocks.8.hook_resid_pre",
    release="gpt2-small-res-jb",
    **kwargs  # Inherits BaseSAE parameters
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sae_id` | `str` | `"blocks.8.hook_resid_pre"` | The SAE hook point identifier from SAE-Lens. |
| `release` | `str` | `"gpt2-small-res-jb"` | The SAE release name from SAE-Lens. |

**Example:**
```python
sae = LocalSAE(
    sae_id="blocks.8.hook_resid_pre",
    release="gpt2-small-res-jb",
    device="cuda:0"
)
sae.load()
activations = sae.encode(["Hello world!"])
```

---

### GoodfireSAE

A local SAE implementation using Goodfire's SAE weights with a local language model.

```python
from interp_embed.sae import GoodfireSAE
```

#### Constructor

```python
GoodfireSAE(
    variant_name="Llama-3.1-8B-Instruct-SAE-l19",
    quantize=False,
    **kwargs  # Inherits BaseSAE parameters
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variant_name` | `str` | `"Llama-3.1-8B-Instruct-SAE-l19"` | The Goodfire SAE variant. Options: `"Llama-3.1-8B-Instruct-SAE-l19"`, `"Llama-3.3-70B-Instruct-SAE-l50"`. |
| `quantize` | `bool` | `False` | Whether to use 8-bit quantization for the language model (saves memory but may reduce accuracy). |

**Example:**
```python
sae = GoodfireSAE(
    variant_name="Llama-3.1-8B-Instruct-SAE-l19",
    device={"model": "auto", "sae": "cuda:0"},
    quantize=True  # For lower memory usage
)
sae.load()
```

---

## Complete Example

Here is a complete example demonstrating the main features of the package:

```python
import pandas as pd
import asyncio
from interp_embed import Dataset
from interp_embed.sae import GoodfireSAE

# Prepare your data
data = pd.DataFrame({
    "text": [
        "The cat sat on the mat.",
        "Dogs are loyal pets.",
        "Python is a programming language.",
        "Machine learning is fascinating.",
    ],
    "category": ["animals", "animals", "tech", "tech"]
})

# Initialize an SAE
sae = GoodfireSAE(
    variant_name="Llama-3.1-8B-Instruct-SAE-l19",
    device="cuda:0"
)

# Create a dataset (automatically computes activations)
dataset = Dataset(
    data=data,
    sae=sae,
    dataset_description="Example dataset with mixed topics",
    field="text",
    save_path="example_dataset.pkl"  # Auto-saves during computation
)

# Get feature activations
max_activations = dataset.latents("max")
print(f"Activation shape: {max_activations.shape}")

# Find top documents for a specific feature
top_docs = dataset.top_documents_for_feature(feature=42, k=3)
for doc in top_docs:
    print(doc)

# Sort by feature activation
sorted_dataset = dataset.sort_by_features([42, 100, 200])

# Label a feature using an LLM
async def label_example():
    label = await dataset.label_feature(
        feature=42,
        model="google/gemini-2.5-flash"
    )
    print(f"Feature 42: {label.label}")
    print(f"Description: {label.brief_description}")

asyncio.run(label_example())

# Save and load
dataset.save_to_file("my_dataset.pkl")
loaded_dataset = Dataset.load_from_file("my_dataset.pkl")
```