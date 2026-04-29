import logging
from typing import Any, Callable, List, Tuple

import torch
from visualine.core.node_base import NodeBase

logger = logging.getLogger(__name__)

class TaskExecuter:
    """
    Lightweight sequential execution engine for VISUALine pipelines.

    Supports nodes that pass either:
    - torch.Tensor
    - dict payloads, e.g. {"tensor": tensor, "boxes_xyxy": boxes}

    The executor intentionally stays simple:
    - no graph optimization
    - no automatic autocast
    - no device movement

    Nodes are responsible for their own inference behavior.
    PipelineManager is responsible for input/output conversion.
    """

    def __init__(self):
        self._use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self._use_cuda else "cpu")

        self._execution_plan: List[Tuple[str, Callable[[Any], Any]]] = []

        if self._use_cuda:
            logger.info(
                f"TaskExecuter initialized with GPU: {torch.cuda.get_device_name(0)}"
            )

            torch.backends.cudnn.benchmark = True

            major, _minor = torch.cuda.get_device_capability()

            if major >= 8:
                torch.set_float32_matmul_precision("high")
        else:
            logger.info("TaskExecuter initialized on CPU.")

    def compile(self, nodes: List[NodeBase]) -> None:
        """
        Compile a simple sequential execution plan from pipeline nodes.
        """
        logger.info(f"Compiling execution graph for {len(nodes)} nodes...")

        execution_plan: List[Tuple[str, Callable[[Any], Any]]] = []

        for node in nodes:
            if not hasattr(node, "process") or not callable(node.process):
                raise TypeError(
                    f"Node {node.__class__.__name__} does not expose a callable process() method."
                )

            node_name = getattr(node, "node_name", node.__class__.__name__)
            execution_plan.append((node_name, node.process))

        self._execution_plan = execution_plan

        logger.debug(
            "Compiled execution plan: "
            + " -> ".join(node_name for node_name, _ in self._execution_plan)
        )

    @torch.inference_mode()
    def __call__(self, data_batch: Any) -> Any:
        """
        Execute the compiled pipeline.

        Args:
            data_batch:
                torch.Tensor or dict payload.

        Returns:
            Output from the final node.
        """
        if not self._execution_plan:
            raise RuntimeError(
                "TaskExecuter was called before compile(), or the compiled plan is empty."
            )

        current = data_batch

        for node_name, step in self._execution_plan:
            try:
                current = step(current)

            except Exception as e:
                logger.error(
                    f"Error executing node '{node_name}': {e}",
                    exc_info=True,
                )
                raise

        return current

    def clear(self) -> None:
        """
        Clear the compiled execution plan.
        """
        self._execution_plan.clear()