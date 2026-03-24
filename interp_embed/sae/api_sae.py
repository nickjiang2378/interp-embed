import asyncio
from typing import List, Callable

from .base_sae import BaseSAE
from ..utils.helpers import log_tqdm_message

class ApiSAE(BaseSAE):
  def __init__(self, max_concurrency: int = 8, max_retries: int = 3, base_delay: int = 2.0, **kwargs):
    super().__init__(**kwargs)
    self.max_retries = max_retries
    self.max_concurrency = max_concurrency
    self.base_delay = base_delay
    self.sem = asyncio.Semaphore(max_concurrency)

  def metadata(self):
    parent_metadata = super().metadata()
    parent_metadata.update({
      "max_retries": self.max_retries,
      "base_delay": self.base_delay,
      "max_concurrency": self.max_concurrency
    })
    return parent_metadata

  async def retry_api_with_backoff(self, coroutine_funcs: List[Callable]):
    """
    Retries a list of asynchronous API call coroutines with exponential backoff.

    Each coroutine in `coros` will be executed with up to `self.max_retries` attempts.
    If a coroutine fails, it will be retried after an exponentially increasing delay, starting from `self.base_delay` seconds.
    All coroutines are run concurrently, but concurrency is limited by the asyncio semaphore (`self.sem`).
    Note: Creating too many coroutines at once can consume significant memory; consider batching calls to this method if needed.

    Args:
        coros (List[Coroutine]): A list of coroutine objects representing API calls.

    Returns:
        List[Any]: A list of results from the successfully completed coroutines, in the same order as `coros`.

    Raises:
        Exception: If a coroutine fails after the maximum number of retries, the exception is raised.
    """
    async def worker(coroutine_func):
      for i in range(self.max_retries):
        try:
          async with self.sem:
            matrix = await coroutine_func()
            return matrix
        except Exception as e:
          if i == self.max_retries - 1:
            raise e
          else:
            print(f"Error calling API: {e}")
            await asyncio.sleep(self.base_delay * (2 ** i))
        log_tqdm_message("Testing 123", level="INFO")
    return await asyncio.gather(*[worker(coroutine_func) for coroutine_func in coroutine_funcs])

