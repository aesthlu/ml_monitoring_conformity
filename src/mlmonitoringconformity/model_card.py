"""Génération de model cards au format Markdown, alignées sur les exigences de l'AI Act."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_EMPTY = "_Non renseigné_"


class RiskLevel(str, Enum):
    """Niveaux de risque définis par l'AI Act."""

    MINIMAL = "minimal"
    LIMITED = "limité"
    HIGH = "élevé"
    UNACCEPTABLE = "inacceptable"


@dataclass(frozen=True)
class Metric:
    """Une métrique de performance mesurée sur un jeu de données."""

    name: str
    value: float
    dataset: str = "test"


@dataclass(frozen=True)
class ModelInfo:
    """Informations décrivant un modèle, utilisées pour générer sa model card."""

    name: str
    version: str
    owner: str
    description: str
    intended_use: str
    risk_level: RiskLevel
    training_data: str
    metrics: list[Metric] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)
    fairness_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Le nom du modèle ne peut pas être vide.")
        if not _SEMVER_PATTERN.match(self.version):
            raise ValueError(
                f"Version invalide : {self.version!r}. Format attendu : X.Y.Z"
            )


def _render_list(items: list[str]) -> str:
    """Rend une liste Markdown à puces, ou un texte par défaut si elle est vide."""
    if not items:
        return _EMPTY
    return "\n".join(f"- {item}" for item in items)


def _render_metrics(metrics: list[Metric]) -> str:
    """Rend le tableau des métriques de performance."""
    if not metrics:
        return _EMPTY
    rows = [
        "| Métrique | Valeur | Jeu de données |",
        "|---|---|---|",
        *(f"| {m.name} | {m.value:.3f} | {m.dataset} |" for m in metrics),
    ]
    return "\n".join(rows)


def _render_fairness(fairness_metrics: dict[str, float]) -> str:
    """Rend le tableau des indicateurs d'équité."""
    if not fairness_metrics:
        return _EMPTY
    rows = [
        "| Indicateur | Valeur |",
        "|---|---|",
        *(f"| {name} | {value:.3f} |" for name, value in fairness_metrics.items()),
    ]
    return "\n".join(rows)


def _render_risk(risk_level: RiskLevel) -> str:
    """Rend la section sur le niveau de risque AI Act."""
    section = f"## Niveau de risque AI Act : {risk_level.value}"
    if risk_level is RiskLevel.HIGH:
        section += (
            "\n\n> ⚠️ Système à haut risque : supervision humaine, journalisation et\n"
            "> documentation technique obligatoires."
        )
    return section


def generate_model_card(info: ModelInfo, generated_on: date | None = None) -> str:
    """Génère la model card d'un modèle au format Markdown.

    Args:
        info: Les informations décrivant le modèle.
        generated_on: Date de génération. Par défaut, la date du jour.
            Injectable pour rendre la sortie déterministe dans les tests.

    Returns:
        La model card au format Markdown.

    Raises:
        ValueError: Si le niveau de risque est inacceptable, car un tel système
            est interdit par l'AI Act.
    """
    if info.risk_level is RiskLevel.UNACCEPTABLE:
        raise ValueError(
            "Un système à risque inacceptable est interdit par l'AI Act : "
            "aucune model card ne peut être générée."
        )

    day = generated_on or date.today()

    sections = [
        f"# Model Card — {info.name} v{info.version}",
        f"*Générée le {day.isoformat()} · Responsable : {info.owner}*",
        f"## Description\n{info.description}",
        f"## Usage prévu\n{info.intended_use}",
        f"### Usages hors périmètre\n{_render_list(info.out_of_scope_uses)}",
        _render_risk(info.risk_level),
        f"## Données d'entraînement\n{info.training_data}",
        f"## Performances\n{_render_metrics(info.metrics)}",
        f"## Équité\n{_render_fairness(info.fairness_metrics)}",
        f"## Limites\n{_render_list(info.limitations)}",
    ]
    return "\n\n".join(sections) + "\n"