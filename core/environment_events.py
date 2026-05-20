from config.sim_config import EnvironmentEventConfig


class EnvironmentEvent:
    """
    Base class for temporary environmental events.

    Event does not store or mutate Environment directly.
    It receives current environment parameters and returns modified parameters.
    """

    def __init__(self, event_config: EnvironmentEventConfig):
        self.start_step = event_config.start_step
        self.duration = event_config.duration
        self.name = "EnvironmentEvent"

        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")

        if self.duration <= 0:
            raise ValueError("duration must be positive")

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration

    def is_active(self, step: int) -> bool:
        return self.start_step <= step < self.end_step

    def apply(self, params: dict[str, float]) -> dict[str, float]:
        return params.copy()


class HeatWaveEvent(EnvironmentEvent):
    """
    Temporarily increases environment temperature.
    """

    def __init__(self, event_config: EnvironmentEventConfig):
        super().__init__(event_config)

        self.name = "HeatWave"
        self.temperature_delta = event_config.temperature_delta

        if self.temperature_delta < 0:
            raise ValueError("temperature_delta must be non-negative")

    def apply(self, params: dict[str, float]) -> dict[str, float]:
        modified = params.copy()
        modified["temperature"] += self.temperature_delta
        return modified


class ColdSnapEvent(EnvironmentEvent):
    """
    Temporarily decreases environment temperature.
    """

    def __init__(self, event_config: EnvironmentEventConfig):
        super().__init__(event_config)

        self.name = "ColdSnap"
        self.temperature_delta = event_config.temperature_delta

        if self.temperature_delta < 0:
            raise ValueError("temperature_delta must be non-negative")

    def apply(self, params: dict[str, float]) -> dict[str, float]:
        modified = params.copy()
        modified["temperature"] -= self.temperature_delta
        return modified


class EpidemicEvent(EnvironmentEvent):
    """
    Temporarily increases hazard level.
    """

    def __init__(self, event_config: EnvironmentEventConfig):
        super().__init__(event_config)

        self.name = "Epidemic"
        self.hazard_delta = event_config.hazard_delta

        if self.hazard_delta < 0:
            raise ValueError("hazard_delta must be non-negative")

    def apply(self, params: dict[str, float]) -> dict[str, float]:
        modified = params.copy()
        modified["hazard_level"] += self.hazard_delta
        return modified


class EventScheduler:
    """
    Stores environment events and applies active events to environment parameters.
    """

    def __init__(self, events: list[EnvironmentEvent] | None = None):
        self.events = events or []

    def active_events(self, step: int) -> list[EnvironmentEvent]:
        return [event for event in self.events if event.is_active(step)]

    def active_event_names(self, step: int) -> str:
        names = [event.name for event in self.active_events(step)]
        return ",".join(names)

    def event_count(self, step: int) -> int:
        return len(self.active_events(step))

    def apply_events(self, params: dict[str, float], step: int) -> dict[str, float]:
        modified = params.copy()

        for event in self.active_events(step):
            modified = event.apply(modified)

        return self._sanitize_params(modified)

    def _sanitize_params(self, params: dict[str, float]) -> dict[str, float]:
        sanitized = params.copy()

        if "hazard_level" in sanitized:
            sanitized["hazard_level"] = min(max(sanitized["hazard_level"], 0.0), 1.0)

        if "food_availability" in sanitized:
            sanitized["food_availability"] = max(sanitized["food_availability"], 0.0)

        return sanitized


def build_environment_event(event_config: EnvironmentEventConfig) -> EnvironmentEvent:
    event_type = event_config.event_type.lower()

    if event_type == "heat_wave":
        return HeatWaveEvent(event_config)

    if event_type == "cold_snap":
        return ColdSnapEvent(event_config)

    if event_type == "epidemic":
        return EpidemicEvent(event_config)

    raise ValueError(f"Unknown environment event type: {event_config.event_type}")


def build_event_scheduler(event_configs: list[EnvironmentEventConfig]) -> EventScheduler:
    events = []

    for event_config in event_configs:
        events.append(build_environment_event(event_config))

    return EventScheduler(events)