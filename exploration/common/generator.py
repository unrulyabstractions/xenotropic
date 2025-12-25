"""
Abstract generator base class.

Defines interface for generation strategies with optional subprocess isolation.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from xenotechnics.common import String
from xenotechnics.trees.tree import TreeNode

from .model import ModelWrapper
from .runner import Runner

# Use spawn for complete memory isolation
_mp_ctx = mp.get_context("spawn")


@dataclass
class GeneratorResult:
    """Result from generator execution."""

    success: bool
    tree_data: dict | None  # Serialized tree data
    load_time: float
    inference_time: float
    total_time: float
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "load_time": round(self.load_time, 2),
            "inference_time": round(self.inference_time, 2),
            "total_time": round(self.total_time, 2),
            "error": self.error,
            "metadata": self.metadata,
        }


class AbstractGenerator(ABC):
    """
    Abstract base class for generation strategies.

    Provides common infrastructure for:
    - Model and runner management
    - State management during generation
    - Tree building with proper probabilities
    - Subprocess isolation for clean memory between runs

    Subclasses implement specific generation strategies via step_impl().
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_chat_template: bool = True,
        lazy_load: bool = False,
        model: Optional[ModelWrapper] = None,
        debug: bool = False,
    ):
        """
        Initialize generator.

        Args:
            model_name: HuggingFace model name
            device: Device to use (auto-detected if None)
            dtype: Data type (auto-detected if None)
            use_chat_template: Whether to use chat template for prompts
            lazy_load: If True, don't load model until needed (for subprocess mode)
            model: Optional existing ModelWrapper to reuse (avoids reloading)
            debug: If True, print debug info about token selection
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_chat_template = use_chat_template
        self.debug = debug

        # Use existing model if provided
        if model is not None:
            self.model = model
            self.runner = Runner(self.model, debug=self.debug)
        # Lazy load for subprocess mode
        elif not lazy_load:
            self._load_model()
        else:
            self.model = None
            self.runner = None

        # State for current generation
        self.distributions = []
        self.current_node = None
        self.root_node = None
        self.step_count = 0
        self.prompt_token_count = 0

    def _load_model(self) -> None:
        """Load model and create runner."""
        self.model = ModelWrapper(
            model_name=self.model_name, device=self.device, dtype=self.dtype
        )
        self.runner = Runner(self.model, debug=self.debug)

    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded (for lazy loading)."""
        if self.model is None:
            self._load_model()

    def run(
        self,
        prompt: Optional[String] = None,
        max_new_tokens: int = 100,
        verbose: bool = True,
        existing_tree: Optional[TreeNode] = None,
        isolate: bool = False,
        **kwargs,
    ) -> TreeNode:
        """
        Run generation and return TreeNode.

        Args:
            prompt: Prompt string (None for empty prompt)
            max_new_tokens: Maximum tokens to generate
            verbose: Whether to print progress
            existing_tree: Optional existing tree to reuse
            isolate: If True, run in subprocess for clean memory
            **kwargs: Strategy-specific parameters

        Returns:
            TreeNode with trajectory (root of tree)
        """
        if isolate:
            return self._run_isolated(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
                **kwargs,
            )
        return self.process_run(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            existing_tree=existing_tree,
            **kwargs,
        )

    def process_run(
        self,
        prompt: Optional[String] = None,
        max_new_tokens: int = 100,
        verbose: bool = True,
        existing_tree: Optional[TreeNode] = None,
        **kwargs,
    ) -> TreeNode:
        """
        Core generation logic. Called directly or in subprocess.

        Args:
            prompt: Prompt string (None for empty prompt)
            max_new_tokens: Maximum tokens to generate
            verbose: Whether to print progress
            existing_tree: Optional existing tree to reuse
            **kwargs: Strategy-specific parameters

        Returns:
            TreeNode with trajectory (root of tree)
        """
        self._ensure_model_loaded()

        # Convert prompt to text
        if prompt is None:
            prompt_text = ""
        else:
            prompt_text = prompt.to_text()

        if verbose:
            print("=" * 60)
            print(f"Prompt: {prompt_text}")
            print("=" * 60)
            print()

        # Tokenize prompt
        input_ids = self.model.tokenize_prompt(
            prompt_text, use_chat_template=self.use_chat_template
        )

        # Initialize state (reuse existing tree if provided)
        # Pass prompt_text for subclasses that need it (e.g., ScorerGenerator)
        self._init_generation_state(
            input_ids, max_new_tokens, existing_tree, prompt_text=prompt_text, **kwargs
        )

        # Define step function
        def step_fn(
            logits: torch.Tensor, model: ModelWrapper, generated_ids: torch.Tensor
        ) -> Optional[torch.Tensor]:
            return self.step_impl(logits, model, generated_ids, verbose, **kwargs)

        # Run generation
        generated_ids = self.runner.run_generation(
            input_ids=input_ids, run_step=step_fn
        )

        # Mark the final node as a complete trajectory
        self.current_node.mark_as_trajectory()

        if verbose:
            self._print_results(generated_ids)

        return self.root_node

    def _run_isolated(
        self,
        prompt: Optional[String] = None,
        max_new_tokens: int = 100,
        verbose: bool = True,
        timeout: float = 300.0,
        **kwargs,
    ) -> TreeNode:
        """
        Run generation in isolated subprocess.

        Args:
            prompt: Prompt string
            max_new_tokens: Maximum tokens to generate
            verbose: Whether to print progress
            timeout: Timeout in seconds
            **kwargs: Strategy-specific parameters

        Returns:
            TreeNode with trajectory
        """
        result_queue = _mp_ctx.Queue()

        proc = _mp_ctx.Process(
            target=_isolated_worker,
            args=(
                self.__class__,
                self.model_name,
                self.device,
                self.dtype,
                self.use_chat_template,
                prompt.to_text() if prompt else None,
                max_new_tokens,
                verbose,
                kwargs,
                result_queue,
            ),
        )

        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            raise TimeoutError(f"Generation timed out after {timeout}s")

        if result_queue.empty():
            raise RuntimeError("Process died without result")

        result: GeneratorResult = result_queue.get()

        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")

        # Reconstruct tree from serialized data
        return _deserialize_tree(result.tree_data)

    @abstractmethod
    def step_impl(
        self,
        logits: torch.Tensor,
        model: ModelWrapper,
        generated_ids: torch.Tensor,
        verbose: bool,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Implement step function for this exploration strategy.

        Args:
            logits: Next token logits
            model: Model wrapper
            generated_ids: Current generated token IDs (including prompt)
            verbose: Whether to print progress
            **kwargs: Strategy-specific parameters

        Returns:
            Next token ID or None to stop
        """
        pass

    def _init_generation_state(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        existing_tree: Optional[TreeNode] = None,
        prompt_text: str = "",
        **kwargs,
    ) -> None:
        """
        Initialize generation state and compute prompt probabilities.

        Args:
            input_ids: Prompt token IDs
            max_new_tokens: Maximum tokens to generate
            existing_tree: Optional existing tree to reuse
            prompt_text: Original prompt text (for subclasses that need it)
            **kwargs: Strategy-specific initialization parameters
        """
        self._prompt_text = prompt_text  # Store for _init_strategy_state
        self.distributions = []
        self.step_count = 0
        self.max_steps = max_new_tokens

        prompt_token_ids = input_ids[0].tolist()

        # If existing tree provided, find the prompt node and reuse
        if existing_tree is not None:
            prompt_node = self._find_prompt_node(existing_tree, prompt_token_ids)
            if prompt_node is not None:
                self.root_node = existing_tree
                self.current_node = prompt_node
                self.prompt_token_count = len(prompt_token_ids)
                self._init_strategy_state(prompt_text=self._prompt_text, **kwargs)
                return

        # Create new root node with empty string
        self.root_node = TreeNode(string=String.empty())
        self.current_node = self.root_node

        # Build prompt chain with proper probabilities
        for i, token_id in enumerate(prompt_token_ids):
            token_str = self.model.tokenizer.decode([token_id])

            # For first token, we can't compute probability (no prior context)
            if i == 0:
                child = self.current_node.add_child(
                    token=token_str, logprob=0.0, token_id=token_id
                )
                self.current_node = child
                continue

            # Get prefix up to this point
            prefix_ids = input_ids[:, :i]

            # Get logits for next token given prefix
            logits, _ = self.model.get_next_token_logits(
                input_ids=prefix_ids, past_key_values=None
            )

            # Compute distribution
            probs = self.model.compute_distribution(logits[0])
            dist = probs.cpu().numpy().astype(np.float32)

            # Set distribution on current node
            self.current_node.set_distribution(probs=dist)

            # Set child_logprobs from distribution
            self.current_node.set_child_logprobs_from_distribution(
                distribution=dist, tokenizer=self.model.tokenizer
            )

            # Get token string and its log probability
            token_logprob = float(np.log(probs[token_id].item() + 1e-10))

            # Add child node
            child = self.current_node.add_child(
                token=token_str, logprob=token_logprob, token_id=token_id
            )
            self.current_node = child

        # Store prompt token count
        self.prompt_token_count = len(prompt_token_ids)

        # Allow subclasses to add custom initialization
        self._init_strategy_state(prompt_text=self._prompt_text, **kwargs)

    def _find_prompt_node(
        self, tree: TreeNode, prompt_token_ids: list
    ) -> Optional[TreeNode]:
        """Find the node corresponding to the end of prompt in existing tree."""
        current = tree

        for token_id in prompt_token_ids:
            found = False
            for child in current.children.values():
                if child.token_id == token_id:
                    current = child
                    found = True
                    break

            if not found:
                return None

        return current

    def _init_strategy_state(self, **kwargs) -> None:
        """Initialize strategy-specific state. Subclasses can override."""
        pass

    def _print_results(self, generated_ids: torch.Tensor) -> None:
        """Print generation results."""
        print()
        print("=" * 60)
        print("Full Response:")
        print()
        generated_text = self.model.decode_tokens(
            generated_ids[0], skip_special_tokens=False
        )
        print(generated_text)
        print("=" * 60)

    def _store_step_data(
        self,
        logits: torch.Tensor,
        next_token_id: torch.Tensor,
        token_str: str,
        token_logprob: float,
    ) -> None:
        """Store data for current step and build tree."""
        # Compute distribution
        probs = self.model.compute_distribution(logits[0])
        dist = probs.cpu().numpy().astype(np.float32)

        # Debug output
        if self.debug:
            token_id = next_token_id[0].item()
            token_prob = np.exp(token_logprob)
            print(
                f"  [DEBUG] step={self.step_count:3d} "
                f"token_id={token_id:6d} "
                f"p={token_prob:.4e} "
                f"logp={token_logprob:8.4f} "
                f"'{token_str}'"
            )

        # Store distribution
        self.distributions.append(dist)

        # Set distribution on current node
        self.current_node.set_distribution(probs=dist)

        # Set child_logprobs from distribution
        self.current_node.set_child_logprobs_from_distribution(
            distribution=dist, tokenizer=self.model.tokenizer
        )

        # Add child node
        child = self.current_node.add_child(
            token=token_str, logprob=token_logprob, token_id=next_token_id[0].item()
        )

        # Move to child
        self.current_node = child

        # Increment step
        self.step_count += 1


# -----------------------------------------------------------------------------
# Subprocess isolation helpers
# -----------------------------------------------------------------------------


def _isolated_worker(
    generator_class: type,
    model_name: str,
    device: Optional[str],
    dtype: Optional[torch.dtype],
    use_chat_template: bool,
    prompt_text: Optional[str],
    max_new_tokens: int,
    verbose: bool,
    kwargs: dict,
    result_queue: _mp_ctx.Queue,
) -> None:
    """Worker function that runs in subprocess."""
    t_start = time.time()

    try:
        # Create generator (will load model fresh)
        generator = generator_class(
            model_name=model_name,
            device=device,
            dtype=dtype,
            use_chat_template=use_chat_template,
            lazy_load=False,
        )

        t_load = time.time()
        load_time = t_load - t_start

        if verbose:
            print(f"  Model loaded in {load_time:.2f}s", flush=True)

        # Convert prompt text back to String
        prompt = String.from_text(prompt_text) if prompt_text else None

        # Run generation
        tree = generator.process_run(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
            **kwargs,
        )

        t_end = time.time()
        inference_time = t_end - t_load

        # Serialize tree for transfer
        tree_data = _serialize_tree(tree)

        result_queue.put(
            GeneratorResult(
                success=True,
                tree_data=tree_data,
                load_time=load_time,
                inference_time=inference_time,
                total_time=t_end - t_start,
            )
        )

    except Exception as e:
        import traceback

        result_queue.put(
            GeneratorResult(
                success=False,
                tree_data=None,
                load_time=0.0,
                inference_time=0.0,
                total_time=time.time() - t_start,
                error=f"{e}\n{traceback.format_exc()}",
            )
        )


def _serialize_tree(tree: TreeNode) -> dict:
    """Serialize TreeNode to dict for subprocess transfer."""

    def serialize_node(node: TreeNode) -> dict:
        return {
            "string": node.string.tokens if node.string else (),
            "token_id": node.token_id,
            "logprob": node.child_logprobs.get(
                node.string.tokens[-1] if node.string and node.string.tokens else "",
                0.0,
            ),
            "is_trajectory": node._is_trajectory,
            "children": {
                tok: serialize_node(child) for tok, child in node.children.items()
            },
            "child_logprobs": dict(node.child_logprobs),
        }

    return serialize_node(tree)


def _deserialize_tree(data: dict, parent: Optional[TreeNode] = None) -> TreeNode:
    """Deserialize dict back to TreeNode."""
    node = TreeNode(
        string=String(tokens=tuple(data["string"]))
        if data["string"]
        else String.empty(),
        parent=parent,
    )
    node._is_trajectory = data.get("is_trajectory", False)
    node.token_id = data.get("token_id")
    node.child_logprobs = dict(data.get("child_logprobs", {}))

    for tok, child_data in data.get("children", {}).items():
        child = _deserialize_tree(child_data, parent=node)
        node.children[tok] = child

    return node
