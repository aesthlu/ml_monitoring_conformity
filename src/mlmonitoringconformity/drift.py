"""Mesures de dérive (drift) entre une distribution de référence et une distribution courante."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy import stats  # type: ignore[import-untyped]

FloatArray = npt.NDArray[np.float64]

_EPSILON = 1e-6
PSI_WARNING_THRESHOLD = 0.1
PSI_DRIFT_THRESHOLD = 0.25


class DriftLevel(str, Enum):
    """Interprétation usuelle d'une valeur de PSI."""

    STABLE = "stable"
    WARNING = "à surveiller"
    DRIFT = "dérive"


@dataclass(frozen=True)
class DriftResult:
    """Résultat d'un test de dérive statistique."""

    statistic: float
    p_value: float
    drift: bool


def _to_array(values: npt.ArrayLike, name: str) -> FloatArray:
    """Convertit une entrée en tableau 1D de flottants, non vide et sans NaN."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} doit être un tableau à une dimension.")
    if array.size == 0:
        raise ValueError(f"{name} ne peut pas être vide.")
    if np.isnan(array).any():
        raise ValueError(f"{name} contient des valeurs manquantes (NaN).")
    return array


def psi(reference: npt.ArrayLike, current: npt.ArrayLike, bins: int = 10) -> float:
    """Calcule le Population Stability Index entre deux distributions.

    Les intervalles sont construits sur les quantiles de la distribution de
    référence. Un epsilon évite les divisions par zéro et les log(0) quand un
    intervalle est vide.

    Args:
        reference: Valeurs de la période de référence (ex. données d'entraînement).
        current: Valeurs de la période courante (ex. données de production).
        bins: Nombre d'intervalles, au moins 2.

    Returns:
        La valeur du PSI, positive ou nulle.
    """
    if bins < 2:
        raise ValueError("bins doit être supérieur ou égal à 2.")
    ref = _to_array(reference, "reference")
    cur = _to_array(current, "current")

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    inner_edges = np.unique(np.quantile(ref, quantiles))

    ref_counts = np.bincount(
        np.searchsorted(inner_edges, ref, side="right"), minlength=inner_edges.size + 1
    )
    cur_counts = np.bincount(
        np.searchsorted(inner_edges, cur, side="right"), minlength=inner_edges.size + 1
    )

    ref_share = np.clip(ref_counts / ref.size, _EPSILON, None)
    cur_share = np.clip(cur_counts / cur.size, _EPSILON, None)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def interpret_psi(value: float) -> DriftLevel:
    """Traduit une valeur de PSI en niveau de dérive selon les seuils usuels."""
    if value < 0:
        raise ValueError("Un PSI ne peut pas être négatif.")
    if value < PSI_WARNING_THRESHOLD:
        return DriftLevel.STABLE
    if value < PSI_DRIFT_THRESHOLD:
        return DriftLevel.WARNING
    return DriftLevel.DRIFT


def ks_drift(
    reference: npt.ArrayLike, current: npt.ArrayLike, alpha: float = 0.05
) -> DriftResult:
    """Détecte une dérive avec le test de Kolmogorov-Smirnov à deux échantillons.

    Args:
        reference: Valeurs de la période de référence.
        current: Valeurs de la période courante.
        alpha: Seuil de significativité, strictement entre 0 et 1.

    Returns:
        La statistique KS, la p-value et un booléen indiquant une dérive.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha doit être strictement compris entre 0 et 1.")
    ref = _to_array(reference, "reference")
    cur = _to_array(current, "current")

    result = stats.ks_2samp(ref, cur)
    p_value = float(result.pvalue)
    return DriftResult(
        statistic=float(result.statistic), p_value=p_value, drift=p_value < alpha
    )