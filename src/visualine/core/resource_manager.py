import logging
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


class ResourceManager:
    _instance = None
    _singleton_lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._singleton_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, vram_limit_ratio: float = 0.85):
        if not hasattr(self, '_initialized'):
            with self._singleton_lock:
                if not hasattr(self, '_initialized'):
                    self.vram_limit_ratio = vram_limit_ratio
                    self._model_cache = OrderedDict()
                    self._is_locked = False
                    self._initialized = True
                    
                    if torch.cuda.is_available():
                        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        self.vram_limit_gb = total_vram * self.vram_limit_ratio
                        logger.info(f"ResourceManager initialized. VRAM Soft Limit: {self.vram_limit_gb:.2f} GB "
                                    f"({self.vram_limit_ratio*100}% of {total_vram:.2f} GB total).")
                    else:
                        self.vram_limit_gb = float('inf')
                        logger.info("ResourceManager initialized on CPU. VRAM limits bypassed.")

    def lock_for_execution(self):
        self._is_locked = True
        logger.info("ResourceManager LOCKED. Dynamic loading/eviction disabled for execution.")

    def unlock(self):
        self._is_locked = False
        logger.info("ResourceManager UNLOCKED.")

    def _get_current_vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024**3)

    def get_model(self, model_name: str, model_loader: Callable[[], Any], device: str = "cuda") -> Any:
        with self._singleton_lock:
            if model_name in self._model_cache:
                self._model_cache.move_to_end(model_name)
                logger.debug(f"Cache HIT for model: '{model_name}'")
                return self._model_cache[model_name]

            if self._is_locked:
                logger.critical(f"Attempted to load '{model_name}' mid-execution!")
                raise RuntimeError(
                    f"Pipeline is locked! Cannot load '{model_name}' mid-stream. "
                    f"Ensure all required models are loaded during the Node setup() phase."
                )

            logger.info(f"Cache MISS for model: '{model_name}'. Preparing to load...")

            while self._get_current_vram_gb() > self.vram_limit_gb and self._model_cache:
                logger.info(f"VRAM near limit ({self._get_current_vram_gb():.2f} GB). Triggering preventative eviction.")
                self._evict_lru_model()

            model = model_loader()
            target_device = device if torch.cuda.is_available() else "cpu"
            
            if target_device == "cuda":
                try:
                    model = model.to(target_device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM while loading '{model_name}'. Executing cache flush...")
                        while self._model_cache:
                            self._evict_lru_model()
                        model = model.to(target_device)
                    else:
                        raise e

            self._model_cache[model_name] = model
            logger.info(f"Model '{model_name}' successfully cached. Current VRAM: {self._get_current_vram_gb():.2f} GB.")
            return model

    def _evict_lru_model(self):
        if not self._model_cache:
            return
            
        evicted_model_name, evicted_model = self._model_cache.popitem(last=False)
        logger.info(f"Evicting LRU model: '{evicted_model_name}'")

        if hasattr(evicted_model, "cleanup") and callable(evicted_model.cleanup):
            try:
                evicted_model.cleanup()
            except Exception as e:
                logger.warning(f"Custom cleanup failed for model '{evicted_model_name}': {e}")
        
        del evicted_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self):
        with self._singleton_lock:
            while self._model_cache:
                self._evict_lru_model()
            logger.info("Resource cache completely cleared.")