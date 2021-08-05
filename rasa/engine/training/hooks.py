import logging
from typing import Any, Dict, Text

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import GraphNodeHook
from rasa.engine.storage.storage import ModelStorage
import rasa.shared.utils.io
from rasa.engine.training import fingerprinting

logger = logging.getLogger(__name__)


class TrainingHook(GraphNodeHook):
    """Caches fingerprints and outputs of nodes during model training."""

    def __init__(self, cache: TrainingCache, model_storage: ModelStorage):
        """Initializes a `TrainingHook`.

        Args:
            cache: Cache used to store fingerprints and outputs.
            model_storage: Used to cache `Resource`s.
        """
        self._cache = cache
        self._model_storage = model_storage

    def on_before_node(
        self,
        node_name: Text,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Calculates the run fingerprint for use in `on_after_node`."""
        fingerprint_key = fingerprinting.calculate_fingerprint_key(
            node_name=node_name, config=config, inputs=received_inputs,
        )

        return {"fingerprint_key": fingerprint_key}

    # TODO: don't cache from CachedComponent
    def on_after_node(
        self,
        node_name: Text,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Stores the fingerprints and caches the output of the node."""
        output_fingerprint = rasa.shared.utils.io.deep_container_fingerprint(output)
        fingerprint_key = input_hook_data["fingerprint_key"]

        logger.debug(
            f"Caching {output} for fingerprint_key: {fingerprint_key} "
            f"and output_fingerprint: {output_fingerprint} calculated with data "
            f"{output}."
        )

        self._cache.cache_output(
            fingerprint_key=fingerprint_key,
            output=output,
            output_fingerprint=output_fingerprint,
            model_storage=self._model_storage,
        )
