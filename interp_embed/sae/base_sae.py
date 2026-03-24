import torch
from abc import ABC, abstractmethod
from enum import Enum
from .utils import ensure_loaded, process_device_config

class BaseSAE(ABC):
  def __init__(self, truncate = True, use_assistant_role: bool = True, device: str = "cpu"):
    self.loaded = False
    self.tokenizer = None
    self.truncate = truncate
    self._feature_labels = dict()
    self.use_assistant_role = use_assistant_role
    self.model_device, self.sae_device = process_device_config(device)

  @classmethod
  def from_metadata(cls, metadata):
    return cls(**metadata)

  def set_device(self, device: str):
    self.model_device, self.sae_device = process_device_config(device)

  def metadata(self):
    return {
      "truncate": self.truncate,
      "use_assistant_role": self.use_assistant_role
    }

  def load(self):
    self.load_models()
    self.load_feature_labels()
    self.loaded = True

  def load_feature_labels(self):
    pass

  def feature_labels(self):
    return self._feature_labels

  def is_loaded(self):
    return self.loaded

  @ensure_loaded
  def encode_chat(self, chat_conversations):
    if self.tokenizer == None:
      raise ValueError("Tokenizer not defined. Chat template doesn't exist.")
    texts = [self.tokenizer.apply_chat_template(chat_conversation, tokenize=False) for chat_conversation in chat_conversations]
    return self.encode(texts)

  @ensure_loaded
  def destroy(self):
    self.destroy_models()
    torch.cuda.empty_cache()
    self.loaded = False

  @ensure_loaded
  def chat_template_exists(self):
    return self.tokenizer is not None and self.tokenizer.chat_template is not None

  @abstractmethod
  def load_models(self):
    pass

  @abstractmethod
  def encode(self, texts):
    pass

  @abstractmethod
  def destroy_models(self):
    pass

class SAEType(Enum):
    LOCAL = "local"
    GOODFIRE = "goodfire"
    NEURONPEDIA_API = "neuronpedia_api"
