"""Indicateurs d'équité pour des modèles de classification binaire."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _to_binary(values: npt.ArrayLike, name: str) -> npt.NDArray[np.int_]:
    """Convertit une entrée en tableau 1D ne contenant que des 0 et des 1."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} doit être un tableau 1D non vide.")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} ne doit contenir que des 0 et des 1.")
    return array.astype(np.int_)


def _to_groups(sensitive: npt.ArrayLike, expected_size: int) -> npt.NDArray[np.generic]:
    """Valide l'attribut sensible : même taille que les prédictions, au moins 2 groupes."""
    groups = np.asarray(sensitive)
    if groups.ndim != 1 or groups.size != expected_size:
        raise ValueError("sensitive doit avoir la même taille que les prédictions.")
    if np.unique(groups).size < 2:
        raise ValueError("Il faut au moins deux groupes pour mesurer un écart.")
    return groups


def demographic_parity_difference(
    y_pred: npt.ArrayLike, sensitive: npt.ArrayLike
) -> float:
    """Écart maximal de taux de prédictions positives entre groupes.

    Vaut 0 quand tous les groupes reçoivent des prédictions positives au même taux.
    """
    pred = _to_binary(y_pred, "y_pred")
    groups = _to_groups(sensitive, pred.size)

    rates = [pred[groups == group].mean() for group in np.unique(groups)]
    return float(max(rates) - min(rates))


def equal_opportunity_difference(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, sensitive: npt.ArrayLike
) -> float:
    """Écart maximal de taux de vrais positifs (rappel) entre groupes.

    Vaut 0 quand le modèle détecte les cas positifs aussi bien dans chaque groupe.
    """
    true = _to_binary(y_true, "y_true")
    pred = _to_binary(y_pred, "y_pred")
    if true.size != pred.size:
        raise ValueError("y_true et y_pred doivent avoir la même taille.")
    groups = _to_groups(sensitive, pred.size)

    true_positive_rates = []
    for group in np.unique(groups):
        positives = (groups == group) & (true == 1)
        if not positives.any():
            raise ValueError(f"Le groupe {group!r} ne contient aucun cas positif réel.")
        true_positive_rates.append(pred[positives].mean())
    return float(max(true_positive_rates) - min(true_positive_rates))