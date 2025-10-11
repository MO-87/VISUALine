"""
    >>>> CRITICAL CORE COMPONENT <<<<

A singleton manager for lazy-loading and caching AI models.

This module implements a Least Recently Used (LRU) cache for AI models to
minimize redundant loading and manage GPU memory efficiently. It follows the
Singleton design pattern to ensure a single instance manages all resources.
"""

import logging
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages the lifecycle of heavy resources like AI models using an LRU cache.

    This is a Singleton class.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cache_size: int = 3):
        """
        Initializes the resource manager. The __init__ will only run the first time.
        
        Args:
            cache_size (int): The maximum number of models to keep in the cache.
        """
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not hasattr(self, '_initialized'):
                    self.cache_size = cache_size
                    self._model_cache = OrderedDict()  ## OrderedDict used for LRU logic
                    self._initialized = True
                    logger.info(f"ResourceManager initialized with cache size {self.cache_size}.")

    def get_model(
        self,
        model_name: str,
        model_loader: Callable[[], Any],
        device: str = "cuda"
    ) -> Any:
        """
        Retrieves a model from the cache or loads it if not present.

        Args:
            model_name (str): A unique identifier for the model.
            model_loader (Callable[[], Any]): A function to load the model.
            device (str): The target device ("cuda" or "cpu"). Defaults to "cuda".

        Returns:
            Any: The loaded model object, moved to the appropriate device.
        """
        if model_name in self._model_cache:
            self._model_cache.move_to_end(model_name)
            logger.debug(f"Cache HIT for model: '{model_name}'")
            return self._model_cache[model_name]

        logger.info(f"Cache MISS for model: '{model_name}'. Loading...")

        ## evicting least recently used model if cache is full
        if len(self._model_cache) >= self.cache_size:
            self._evict_lru_model()

        ## loading the new model using the provided loader
        model = model_loader()
        
        ## put model onto appropriate device (cuda if available)
        target_device = device if torch.cuda.is_available() else "cpu"
        if target_device == "cuda":
            try:
                model = model.to(target_device)
                logger.info(f"Moved model '{model_name}' to {target_device}.")
            except Exception as e:
                logger.warning(f"Failed to move model '{model_name}' to {target_device}, keeping on CPU. Error: {e}")
        
        self._model_cache[model_name] = model
        logger.info(f"Model '{model_name}' loaded and cached.")
        self._log_cache_status()

        return model

    def _evict_lru_model(self):
        """Evicts the least recently used model from the cache."""
        if not self._model_cache:
            return
            
        evicted_model_name, evicted_model = self._model_cache.popitem(last=False)
        logger.info(f"Cache is full. Evicting LRU model: '{evicted_model_name}'")

        ## adding a graceful eviction callback for custom cleanup
        if hasattr(evicted_model, "cleanup") and callable(evicted_model.cleanup):
            try:
                logger.debug(f"Calling custom cleanup method for '{evicted_model_name}'...")
                evicted_model.cleanup()
            except Exception as e:
                logger.warning(f"Custom cleanup failed for model '{evicted_model_name}': {e}")
        
        ## lazy GPU memory release to reduce fragmentation
        del evicted_model
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()  ## More efficient cleanup before empty_cache()
            torch.cuda.empty_cache()
            
        self._log_cache_status()

    def clear_cache(self):
        """Clears all models from the cache, calling cleanup on each."""
        model_names = list(self._model_cache.keys())
        for model_name in model_names:
            self._evict_lru_model()  ## evict one by one to trigger cleanup
        logger.info("Resource cache cleared.")

    def _log_cache_status(self):
        """Logs the current state of the model cache."""
        cached_models = list(self._model_cache.keys())
        logger.debug(f"Current cache state (LRU -> MRU): {cached_models}")
