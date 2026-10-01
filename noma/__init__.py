"""Noma: an open, calibrated decision model (Jev class) by Blackdrome AI Labs."""

__version__ = "1.0.1"


def __getattr__(name):  # lazy: `from noma import Noma` without importing torch at package import
    if name == "Noma":
        from .model.noma import Noma
        return Noma
    raise AttributeError(name)
