from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RegEvoPopulationConfig:
    """Configuration of a ReEvo population.
    Attributes:
        cluster_sampling_temperature_init: Initial temperature for softmax sampling of clusters.
        cluster_sampling_temperature_period: Period of linear decay of the cluster sampling temperature.
    """
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000
