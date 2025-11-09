"""Utility functions for biomedical engineering and bioinformatics workflows.

This module centralises a handful of frequently used calculations that appear in
biomedical engineering and bioinformatics projects.  The goal is to provide a
lightweight, dependency-free toolbox that can be easily imported into scripts
or notebooks.

Each public function performs input validation and documents its assumptions so
that the behaviour is predictable and transparent.
"""

from __future__ import annotations

from typing import Iterable, Sequence


__all__ = [
    "calculate_body_mass_index",
    "calculate_body_surface_area",
    "mean_arterial_pressure",
    "cardiac_output",
    "estimate_gfr_ckd_epi",
    "transcribe_dna_to_rna",
    "reverse_complement_dna",
    "gc_content",
]


def _require_positive(value: float, name: str) -> None:
    """Raise a ``ValueError`` if *value* is not positive."""

    if value <= 0:
        raise ValueError(f"{name} must be positive, received {value!r}.")


def calculate_body_mass_index(weight_kg: float, height_m: float) -> float:
    """Return the body mass index (BMI).

    Parameters
    ----------
    weight_kg:
        Body mass in kilograms.
    height_m:
        Height in metres.
    """

    _require_positive(weight_kg, "weight_kg")
    _require_positive(height_m, "height_m")
    return weight_kg / (height_m**2)


def calculate_body_surface_area(height_cm: float, weight_kg: float) -> float:
    """Calculate the body surface area (Mosteller formula) in square metres."""

    _require_positive(height_cm, "height_cm")
    _require_positive(weight_kg, "weight_kg")
    return (height_cm * weight_kg / 3600) ** 0.5


def mean_arterial_pressure(systolic_mm_hg: float, diastolic_mm_hg: float) -> float:
    """Calculate mean arterial pressure (MAP) in mmHg."""

    _require_positive(systolic_mm_hg, "systolic_mm_hg")
    _require_positive(diastolic_mm_hg, "diastolic_mm_hg")
    if systolic_mm_hg < diastolic_mm_hg:
        raise ValueError("Systolic pressure must be greater than diastolic pressure.")
    return diastolic_mm_hg + (systolic_mm_hg - diastolic_mm_hg) / 3


def cardiac_output(heart_rate_bpm: float, stroke_volume_ml: float) -> float:
    """Return the cardiac output in L/min."""

    _require_positive(heart_rate_bpm, "heart_rate_bpm")
    _require_positive(stroke_volume_ml, "stroke_volume_ml")
    return heart_rate_bpm * stroke_volume_ml / 1000


def estimate_gfr_ckd_epi(
    creatinine_mg_dl: float,
    age_years: float,
    sex: str,
    *,
    is_black: bool | None = None,
) -> float:
    """Estimate glomerular filtration rate (GFR) using the CKD-EPI 2009 equation.

    Parameters
    ----------
    creatinine_mg_dl:
        Serum creatinine concentration in mg/dL.
    age_years:
        Age of the patient in years.
    sex:
        Biological sex (``"male"`` or ``"female"``).
    is_black:
        ``True`` if the patient is African American/Black as defined in the
        original CKD-EPI equation, ``False`` otherwise.  When ``None`` the race
        factor is omitted.
    """

    _require_positive(creatinine_mg_dl, "creatinine_mg_dl")
    _require_positive(age_years, "age_years")
    try:
        sex_key = sex.strip().lower()
    except AttributeError as exc:  # pragma: no cover - defensive coding
        raise TypeError("sex must be a string") from exc

    if sex_key not in {"male", "female"}:
        raise ValueError("sex must be 'male' or 'female'.")

    if sex_key == "female":
        k = 0.7
        alpha = -0.329
        coeff = 144
    else:
        k = 0.9
        alpha = -0.411
        coeff = 141

    scr_k = creatinine_mg_dl / k
    scr_component = (scr_k) ** alpha if scr_k <= 1 else (scr_k) ** -1.209
    age_component = 0.993 ** age_years

    race_component = 1.159 if is_black else 1.0

    return coeff * scr_component * age_component * race_component


DNA_COMPLEMENTS = str.maketrans("ACGTacgt", "TGCAtgca")


def transcribe_dna_to_rna(sequence: str) -> str:
    """Return the RNA transcript for a DNA sequence (replacing T with U)."""

    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("sequence must not be empty.")
    if any(base not in {"A", "C", "G", "T"} for base in seq):
        raise ValueError("sequence must contain only A, C, G, T characters.")
    return seq.replace("T", "U")


def reverse_complement_dna(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    seq = sequence.strip()
    if not seq:
        raise ValueError("sequence must not be empty.")
    if any(base not in "ACGTacgt" for base in seq):
        raise ValueError("sequence must contain only A, C, G, T characters.")
    return seq.translate(DNA_COMPLEMENTS)[::-1]


def gc_content(sequence: Sequence[str] | str) -> float:
    """Calculate the GC content fraction for a DNA/RNA sequence.

    Accepts strings and other iterables of nucleotide characters.  Returns a
    float between 0 and 1 inclusive.
    """

    if isinstance(sequence, str):
        iterable: Iterable[str] = sequence.strip().upper()
    else:
        iterable = (str(base).upper() for base in sequence)

    total = 0
    gc_count = 0
    for base in iterable:
        if base not in {"A", "C", "G", "T", "U"}:
            raise ValueError("sequence must contain only nucleotide characters.")
        total += 1
        if base in {"G", "C"}:
            gc_count += 1

    if total == 0:
        raise ValueError("sequence must contain at least one nucleotide.")

    return gc_count / total


if __name__ == "__main__":  # pragma: no cover - example usage
    example_dna = "ATGCGT"
    print("BMI (70 kg, 1.75 m):", calculate_body_mass_index(70, 1.75))
    print("BSA (175 cm, 70 kg):", calculate_body_surface_area(175, 70))
    print("MAP (120/80 mmHg):", mean_arterial_pressure(120, 80))
    print("CO (70 bpm, 70 mL):", cardiac_output(70, 70))
    print(
        "GFR (Scr=1.0 mg/dL, age=40, male):",
        estimate_gfr_ckd_epi(1.0, 40, "male"),
    )
    print("RNA transcript:", transcribe_dna_to_rna(example_dna))
    print("Reverse complement:", reverse_complement_dna(example_dna))
    print("GC content:", gc_content(example_dna))
