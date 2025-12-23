import logging
from typing import Dict, Any, Generator, Tuple, Optional
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class PanguInferenceEngine:
    """
    Core Inference Engine for Pangu-Weather.

    Responsibilities:
    1. Manage ONNX Runtime sessions.
    2. Execute valid single-step predictions.
    3. Orchestrate multi-step iterative forecasts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the engine with configuration dictionary.

        Args:
            config: Inference configuration containing 'model_paths' and 'onnx_options'.
        """
        self.config = config
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Initialize ONNX sessions based on configuration."""
        model_paths = self.config.get("model_paths", {})
        providers = self._get_providers()
        sess_opts = self._get_session_options()

        for lead_time, path in model_paths.items():
            logger.info(f"Loading model {lead_time} from {path}")
            try:
                self.sessions[lead_time] = ort.InferenceSession(
                    path, sess_options=sess_opts, providers=providers
                )
            except Exception as e:
                logger.error(f"Failed to load model {path}: {e}")
                raise

    def _get_providers(self) -> list:
        provider = self.config.get("execution_provider", "CPUExecutionProvider")
        if provider == "CUDAExecutionProvider":
            return [(provider, self.config.get("cuda_provider_options", {}))]
        return [provider]

    def _get_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        cfg = self.config.get("ort_options", {})
        if cfg.get("enable_cpu_mem_arena"):
            opts.enable_cpu_mem_arena = True
        return opts

    def predict(
        self, input_upper: np.ndarray, input_surface: np.ndarray, lead_time: str = "24h"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single inference step.
        """
        if lead_time not in self.sessions:
            raise ValueError(f"Model for {lead_time} not loaded.")

        session = self.sessions[lead_time]
        inputs = {"input": input_upper, "input_surface": input_surface}
        return session.run(None, inputs)

    def run_sequence(
        self, input_upper: np.ndarray, input_surface: np.ndarray, total_hours: int
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Execute an iterative forecast sequence.

        Yields:
            Tuple(current_hour, output_upper, output_surface)
        """
        current_upper, current_surface = input_upper, input_surface
        # Buffer for 24h steps
        buffer_24h_upper, buffer_24h_surface = input_upper, input_surface

        steps_6h = total_hours // 6

        for step in range(1, steps_6h + 1):
            hour = step * 6

            # Strategy: Use 24h model every 4th step (24h)
            if step % 4 == 0 and "24h" in self.sessions:
                logger.debug(f"Step {step} ({hour}h): Using 24h model")
                out_upper, out_surface = self.predict(
                    buffer_24h_upper, buffer_24h_surface, "24h"
                )
                # Update 24h buffer for next 24h jump
                buffer_24h_upper, buffer_24h_surface = out_upper, out_surface
            else:
                logger.debug(f"Step {step} ({hour}h): Using 6h model")
                out_upper, out_surface = self.predict(
                    current_upper, current_surface, "6h"
                )

            # Update current state for next 6h step
            current_upper, current_surface = out_upper, out_surface

            yield hour, out_upper, out_surface
