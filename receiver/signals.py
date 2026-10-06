"""Galileo OS SIS ICD 2.2, §§2–4: explicit RF and spreading-code contracts."""

from dataclasses import dataclass
import importlib
import math
from types import SimpleNamespace
import numpy as np


@dataclass(frozen=True)
class Signal:
    name: str
    frequency_hz: float
    chip_rate: float
    code_length: int
    secondary_length: int
    data_module: str
    pilot_module: str
    symbol_epochs: int
    navigation: str
    sample_rate: float

    @property
    def period(self):
        return self.code_length / self.chip_rate

    def codes(self, root=None):
        from .tracking import load_codes

        load_codes(SimpleNamespace(gnss_tools=root))
        return tuple(
            importlib.import_module("gnsstools.galileo." + m)
            for m in (self.data_module, self.pilot_module)
        )

    def pilot_secondary(self, module, prn):
        value = module.secondary_code
        return np.asarray(value[prn] if isinstance(value, dict) else value, dtype=float)

    def validate_rate(self, rate):
        minimum = 12_500_000 if self.name == "e1" else 25_000_000
        if (
            not math.isfinite(rate)
            or rate < minimum
            or not math.isclose(
                rate * self.period, round(rate * self.period), abs_tol=1e-6
            )
        ):
            raise ValueError(
                f"{self.name}: sample rate must be >= {minimum:g} and give integer nominal code epochs"
            )


SIGNALS = {
    "e1": Signal("e1", 1575.42e6, 1.023e6, 4092, 25, "e1b", "e1c", 1, "I/NAV", 12.5e6),
    "e5a": Signal(
        "e5a", 1176.45e6, 10.23e6, 10230, 100, "e5ai", "e5aq", 20, "F/NAV", 25e6
    ),
    "e5b": Signal(
        "e5b", 1207.14e6, 10.23e6, 10230, 100, "e5bi", "e5bq", 4, "I/NAV", 25e6
    ),
}
