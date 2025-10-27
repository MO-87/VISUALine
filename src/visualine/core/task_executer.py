import logging
from typing import Any, List

import numpy as np
import torch
from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)


class TaskExecuter:
    """
    Executes a node (or series of nodes) efficiently on a batch of data.
    Optimized to minimize format conversions and memory allocations.
    """
    
    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            logger.info(f"TaskExecuter initialized with GPU: {torch.cuda.get_device_name(0)}")
            ## enable cudnn benchmarking for faster convolutions (5-10% speedup)
            torch.backends.cudnn.benchmark = True
            ## enable TF32 on Ampere+ GPUs for better performance without accuracy loss
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision('high')
                logger.debug("Enabled TF32 (high precision) for faster computation on Ampere+ GPU.")
        else:
            logger.info("TaskExecuter initialized on CPU.")

    def execute_batch(self, nodes: List[NodeBase], data_batch: torch.Tensor) -> torch.Tensor:
        """
        Executes a batch of data sequentially through all given nodes with optimized format handling.
        
        Minimizes format conversions by batching consecutive nodes of the same type together,
        reducing the number of torch<->numpy conversions which are expensive operations.
        
        Args:
            nodes (List[NodeBase]): The nodes to execute in order.
            data_batch (torch.Tensor): Batch of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: The processed batch (on same device as input).
        """
        try:
            is_cuda = data_batch.is_cuda
            current_format = 'torch'  ## track current data format
            
            ## pre-analyze node format requirements to optimize conversion strategy
            node_formats = [getattr(node, "use_torch", True) for node in nodes]
            
            ## check if all nodes use the same format (common case optimization)
            if all(node_formats):
                ## all nodes use torch - no conversions needed!
                for node in nodes:
                    data_batch = node.process(data_batch)
                
                ## ensure output matches input device
                if is_cuda and not data_batch.is_cuda:
                    data_batch = data_batch.cuda(non_blocking=True)
                elif not is_cuda and data_batch.is_cuda:
                    data_batch = data_batch.cpu()
                
                return data_batch
            
            elif not any(node_formats):
                ## all nodes use numpy - convert once at start, once at end
                data_batch = self._torch_to_numpy_batch(data_batch)
                current_format = 'numpy'
                
                for node in nodes:
                    data_batch = node.process(data_batch)
                
                ## convert back to torch
                data_batch = self._numpy_to_torch_batch(data_batch, target_cuda=is_cuda)
                return data_batch
            
            ## mixed format pipeline - optimize conversions
            for node, uses_torch in zip(nodes, node_formats):
                ## convert torch -> numpy if needed
                if not uses_torch and current_format == 'torch':
                    data_batch = self._torch_to_numpy_batch(data_batch)
                    current_format = 'numpy'
                
                ## convert numpy -> torch if needed
                elif uses_torch and current_format == 'numpy':
                    data_batch = self._numpy_to_torch_batch(data_batch, target_cuda=is_cuda)
                    current_format = 'torch'
                
                ## process through node
                data_batch = node.process(data_batch)
            
            ## ensure final output is torch tensor on correct device
            if current_format == 'numpy':
                data_batch = self._numpy_to_torch_batch(data_batch, target_cuda=is_cuda)
            
            ## final device check
            if is_cuda and not data_batch.is_cuda:
                data_batch = data_batch.cuda(non_blocking=True)
            elif not is_cuda and data_batch.is_cuda:
                data_batch = data_batch.cpu()
            
            return data_batch
            
        except Exception as e:
            logger.error(f"Error executing batch: {e}", exc_info=True)
            ## attempt recovery by returning tensor on original device
            if isinstance(data_batch, np.ndarray):
                data_batch = self._numpy_to_torch_batch(data_batch, target_cuda=is_cuda)
            return data_batch

    def _torch_to_numpy_batch(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Efficiently converts a torch tensor batch to numpy array.
        
        Optimized with explicit synchronization and memory cleanup to reduce
        fragmentation and ensure GPU operations complete before CPU access.
        
        Args:
            tensor (torch.Tensor): Input tensor (B, C, H, W).
        
        Returns:
            np.ndarray: Output array (B, H, W, C) in uint8 format.
        """
        ## synchronize if on CUDA to ensure all GPU ops complete
        if tensor.is_cuda and self._use_cuda:
            torch.cuda.synchronize()
        
        ## convert: (B, C, H, W) -> (B, H, W, C)
        arr = tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        ## cleanup tensor reference
        del tensor
        
        return arr

    def _numpy_to_torch_batch(self, arr: np.ndarray, target_cuda: bool = False) -> torch.Tensor:
        """
        Efficiently converts a numpy array batch to torch tensor.
        
        Uses pinned memory for faster GPU transfers when CUDA is available.
        
        Args:
            arr (np.ndarray): Input array (B, H, W, C) in uint8 format.
            target_cuda (bool): Whether to place tensor on CUDA device.
        
        Returns:
            torch.Tensor: Output tensor (B, C, H, W) in float32.
        """
        ## convert: (B, H, W, C) -> (B, C, H, W)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
        
        ## transfer to GPU with pinned memory if needed
        if self._use_cuda and target_cuda:
            tensor = tensor.pin_memory().cuda(non_blocking=True)
        
        ## cleanup array reference
        del arr
        
        return tensor

    def execute_batch_optimized(self, nodes: List[NodeBase], data_batch: np.ndarray) -> np.ndarray:
        """
        Alternative entry point that accepts numpy input and returns numpy output.
        
        Useful for video processing pipelines where frames are read/written as numpy arrays.
        Minimizes conversions by keeping data in numpy format when possible.
        
        Args:
            nodes (List[NodeBase]): The nodes to execute in order.
            data_batch (np.ndarray): Batch of images (B, H, W, 3) in numpy format.
        
        Returns:
            np.ndarray: Processed batch (B, H, W, 3) in numpy format.
        """
        try:
            ## detect pipeline format requirements upfront
            node_formats = [getattr(n, "use_torch", True) for n in nodes]
            
            ## if all nodes use same format, optimize conversion strategy
            if all(node_formats):
                ## all torch - convert once at start, once at end
                data_batch = self._numpy_to_torch_batch(data_batch, target_cuda=self._use_cuda)
                for node in nodes:
                    data_batch = node.process(data_batch)
                return self._torch_to_numpy_batch(data_batch)
            
            elif not any(node_formats):
                ## all numpy - no conversions needed!
                for node in nodes:
                    data_batch = node.process(data_batch)
                return data_batch
            
            ## mixed format: minimize conversions
            current_format = 'numpy'
            for node, uses_torch in zip(nodes, node_formats):
                if uses_torch and current_format == 'numpy':
                    data_batch = self._numpy_to_torch_batch(data_batch, target_cuda=self._use_cuda)
                    current_format = 'torch'
                elif not uses_torch and current_format == 'torch':
                    data_batch = self._torch_to_numpy_batch(data_batch)
                    current_format = 'numpy'
                
                data_batch = node.process(data_batch)
            
            ## ensure numpy output
            if current_format == 'torch':
                data_batch = self._torch_to_numpy_batch(data_batch)
            
            return data_batch
            
        except Exception as e:
            logger.error(f"Error executing optimized batch: {e}", exc_info=True)
            ## attempt recovery
            if isinstance(data_batch, torch.Tensor):
                data_batch = self._torch_to_numpy_batch(data_batch)
            return data_batch