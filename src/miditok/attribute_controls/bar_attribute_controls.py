"""Bar-level attribute controls modules."""

from __future__ import annotations

import numpy as np

from miditok import Event

from .classes import BarAttributeControl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class BarOnsetPolyphony(BarAttributeControl):
    """
    Onset polyphony attribute control at the bar level.

    It specifies the minimum and maximum number of notes played simultaneously at a
    given time onset.
    It can be enabled with the ``ac_polyphony_bar`` argument of
    :class:`miditok.TokenizerConfig`.

    :param polyphony_min: minimum number of simultaneous notes to consider.
    :param polyphony_max: maximum number of simultaneous notes to consider.
    """

    def __init__(
        self,
        polyphony_min: int,
        polyphony_max: int,
    ) -> None:
        self.min_polyphony = polyphony_min
        self.max_polyphony = polyphony_max
        super().__init__(
            tokens=[
                f"{tok_type}_{val}"
                for tok_type in ("ACBarOnsetPolyphonyMin", "ACBarOnsetPolyphonyMax")
                for val in range(polyphony_min, polyphony_max + 1)
            ],
        )

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa, time_division
        _, counts_onsets = np.unique(notes_soa["time"], return_counts=True)

        onset_poly_max, onset_poly_min = 0, 0
        if len(counts_onsets) > 0:
            onset_poly_min, onset_poly_max = np.min(counts_onsets), np.max(counts_onsets)

        min_poly = min(max(onset_poly_min, self.min_polyphony), self.max_polyphony)
        max_poly = min(max(onset_poly_max, self.min_polyphony), self.max_polyphony)
        return [
            Event("ACBarOnsetPolyphonyMin", min_poly),
            Event("ACBarOnsetPolyphonyMax", max_poly),
        ]


class BarPitchClass(BarAttributeControl):
    """
    Bar-level pitch classes attribute control.

    This attribute control specifies which pitch classes are present within a bar.
    """

    def __init__(self) -> None:
        super().__init__(tokens=[f"ACBarPitchClass_{i}" for i in range(12)])

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa, time_division
        pitch_values = notes_soa["pitch"] % 12
        pitch_values = np.unique(pitch_values)
        return [Event("ACBarPitchClass", pitch) for pitch in pitch_values]


class BarNoteDensity(BarAttributeControl):
    """
    Bar-level note density attribute control.

    It specifies the number of notes per bar. If a bar contains more that the maximum
    density (``density_max``), a ``density_max+`` token will be returned.

    :param density_max: maximum note density per bar to consider.
    """

    def __init__(self, density_max: int) -> None:
        self.density_max = density_max
        super().__init__(
            tokens=[
                *(f"ACBarNoteDensity_{i}" for i in range(self.density_max)),
                f"ACBarNoteDensity_{self.density_max}+",
            ],
        )

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa, time_division
        n_notes = len(notes_soa["time"])
        if n_notes >= self.density_max:
            return [Event("ACBarNoteDensity", f"{self.density_max}+")]
        return [Event("ACBarNoteDensity", n_notes)]


class BarNoteDuration(BarAttributeControl):
    """
    Note duration attribute control.

    This attribute controls specifies the note durations (whole, half, quarter, eight,
    sixteenth and thirty-second) present in a bar.
    """

    def __init__(self) -> None:
        self._note_durations = (
            "Whole",
            "Half",
            "Quarter",
            "Eight",
            "Sixteenth",
            "ThirtySecond",
        )
        super().__init__(
            tokens=[
                f"ACBarNoteDuration{duration}_{val}"
                for duration in self._note_durations
                for val in (0, 1)
            ],
        )
        # Factors multiplying ticks/quarter time division
        self.factors = (4, 2, 1, 0.5, 0.25)

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        del controls_soa, pitch_bends_soa
        durations = np.unique(notes_soa["duration"])
        controls = []
        for fi, factor in enumerate(self.factors):
            controls.append(
                Event(
                    f"ACBarNoteDuration{self._note_durations[fi]}",
                    1 if time_division * factor in durations else 0,
                )
            )
        return controls
    
class BarTension(BarAttributeControl):
    """
    Bar-level tension attribute control with standalone implementation.

    This control computes musical tension from note-level features without
    requiring external tension computation libraries. Tension is computed
    from multiple musical dimensions and binned into discrete values.

    Features considered:
    - Note density (onset frequency)
    - Pitch range and variance
    - Velocity dynamics
    - Rhythmic irregularity
    - Harmonic dissonance (interval-based)
    - Register (pitch height)

    :param tension_min: minimum tension value (default: 0.0)
    :param tension_max: maximum tension value (default: 1.0)
    :param num_bins: number of discrete tension bins (default: 10)
    :param weights: dict of feature weights with keys:
        - 'density': weight for note density (default: 1.5)
        - 'pitch_range': weight for pitch range (default: 1.0)
        - 'velocity': weight for velocity variance (default: 1.2)
        - 'rhythm': weight for rhythmic irregularity (default: 1.0)
        - 'dissonance': weight for harmonic dissonance (default: 1.3)
        - 'register': weight for pitch register (default: 0.8)
    """

    def __init__(
        self,
        tension_min: float = 0.0,
        tension_max: float = 1.0,
        num_bins: int = 10,
        weights: dict[str, float] | None = None,
        exclude_programs: Sequence[int] = [],
        token_suffix: str = ""
    ) -> None:
        self.tension_min = tension_min
        self.tension_max = tension_max
        self.num_bins = num_bins
        
        # Default feature weights
        default_weights = {
            'density': 1.5,
            'pitch_range': 1.0,
            'velocity': 1.2,
            'rhythm': 1.0,
            'dissonance': 1.3,
            'register': 0.8,
        }
        self.weights = weights if weights is not None else default_weights

        # Normalization factors (empirically determined)
        self.norm_factors = {
            'density': 20.0,      # notes per bar
            'pitch_range': 36.0,  # semitones (3 octaves)
            'velocity': 64.0,     # velocity range
            'rhythm': 1.0,        # already normalized
            'dissonance': 1.0,    # already normalized
            'register': 60.0,     # MIDI pitch (middle C = 60)
        }
        
        # Create bin edges
        self.bin_edges = np.linspace(tension_min, tension_max, num_bins + 1)

        self.token_type = f"ACBarTension{token_suffix}"
        
        # Create tokens for each bin
        super().__init__(
            tokens=[f"{self.token_type}_{i}" for i in range(num_bins)],
            exclude_programs=exclude_programs
        )

    def _compute_note_density(
        self,
        notes_soa: dict[str, np.ndarray],
    ) -> float:
        """
        Compute note density (onset frequency).
        
        Higher density = more tension.
        """
        n_notes = len(notes_soa["time"])
        return float(n_notes)

    def _compute_pitch_range(
        self,
        notes_soa: dict[str, np.ndarray],
    ) -> float:
        """
        Compute pitch range (span of pitches used).
        
        Wider range = more tension.
        """
        if len(notes_soa["pitch"]) == 0:
            return 0.0
        
        pitches = notes_soa["pitch"]
        pitch_range = float(np.max(pitches) - np.min(pitches))
        return pitch_range

    def _compute_pitch_variance(
        self,
        notes_soa: dict[str, np.ndarray],
    ) -> float:
        """
        Compute pitch variance (melodic motion).
        
        Higher variance = more tension.
        """
        if len(notes_soa["pitch"]) < 2:
            return 0.0
        
        pitches = notes_soa["pitch"]
        return float(np.std(pitches))

    def _compute_velocity_dynamics(
        self,
        notes_soa: dict[str, np.ndarray],
    ) -> float:
        """
        Compute velocity variance (dynamic contrast).
        
        Higher variance = more tension.
        """
        if len(notes_soa["velocity"]) < 2:
            return 0.0
        
        velocities = notes_soa["velocity"]
        
        # Consider both variance and mean velocity
        vel_std = float(np.std(velocities))
        vel_mean = float(np.mean(velocities))
        
        # Combine: high velocity + high variance = most tension
        return vel_std + (vel_mean / 127.0) * 20.0

    def _compute_rhythmic_irregularity(
        self,
        notes_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> float:
        """
        Compute rhythmic irregularity (onset timing variance).
        
        More irregular rhythms = more tension.
        """
        if len(notes_soa["time"]) < 2:
            return 0.0
        
        times = np.sort(notes_soa["time"])
        inter_onset_intervals = np.diff(times)
        
        if len(inter_onset_intervals) == 0:
            return 0.0
        
        # Coefficient of variation (normalized standard deviation)
        mean_ioi = np.mean(inter_onset_intervals)
        if mean_ioi == 0:
            return 0.0
        
        cv = np.std(inter_onset_intervals) / mean_ioi
        return float(cv)

    def _compute_harmonic_dissonance(
        self,
        notes_soa: dict[str, np.ndarray],
    ) -> float:
        """
        Compute harmonic dissonance based on interval content.
        
        More dissonant intervals = more tension.
        Uses a simplified dissonance model based on interval classes.
        """
        if len(notes_soa["pitch"]) < 2:
            return 0.0
        
        # Get unique pitch classes present at each time point
        times = notes_soa["time"]
        pitches = notes_soa["pitch"]
        
        # Group notes by time (simultaneous notes)
        unique_times = np.unique(times)
        
        total_dissonance = 0.0
        n_simultaneities = 0
        
        for t in unique_times:
            simultaneous_pitches = pitches[times == t]
            
            if len(simultaneous_pitches) < 2:
                continue
            
            # Compute pairwise intervals
            for i in range(len(simultaneous_pitches)):
                for j in range(i + 1, len(simultaneous_pitches)):
                    interval = abs(simultaneous_pitches[i] - simultaneous_pitches[j])
                    interval_class = interval % 12
                    
                    # Dissonance weights for each interval class
                    # Based on traditional consonance/dissonance hierarchy
                    dissonance_map = {
                        0: 0.0,   # Unison - consonant
                        1: 1.0,   # Minor 2nd - very dissonant
                        2: 0.8,   # Major 2nd - dissonant
                        3: 0.4,   # Minor 3rd - consonant
                        4: 0.3,   # Major 3rd - consonant
                        5: 0.5,   # Perfect 4th - moderately consonant
                        6: 1.0,   # Tritone - very dissonant
                        7: 0.2,   # Perfect 5th - very consonant
                        8: 0.4,   # Minor 6th - consonant
                        9: 0.3,   # Major 6th - consonant
                        10: 0.7,  # Minor 7th - dissonant
                        11: 0.9,  # Major 7th - very dissonant
                    }
                    
                    total_dissonance += dissonance_map.get(interval_class, 0.5)
                    n_simultaneities += 1
        
        if n_simultaneities == 0:
            return 0.0
        
        # Average dissonance across all intervals
        return total_dissonance / n_simultaneities

    def _compute_register(
        self,
        notes_soa: dict[str, np.ndarray],
    ) -> float:
        """
        Compute average register (pitch height).
        
        Extreme registers (very high or very low) = more tension.
        """
        if len(notes_soa["pitch"]) == 0:
            return 0.0
        
        pitches = notes_soa["pitch"]
        mean_pitch = float(np.mean(pitches))
        
        # Distance from middle register (around 60 = middle C)
        # Higher distances = more tension
        middle_register = 60.0
        register_deviation = abs(mean_pitch - middle_register)
        
        return register_deviation

    def _normalize_feature(
        self,
        value: float,
        feature_name: str,
    ) -> float:
        """
        Normalize a feature value to [0, 1] range.
        """
        norm_factor = self.norm_factors.get(feature_name, 1.0)
        normalized = value / norm_factor
        # Clip to [0, 1]
        return float(np.clip(normalized, 0.0, 1.0))

    def _compute_tension_value(
        self,
        notes_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> float:
        """
        Compute overall tension value from all features.
        
        Only computes features with non-zero weights for efficiency.
        
        :return: tension value in range [tension_min, tension_max]
        """
        if len(notes_soa["time"]) == 0:
            return self.tension_min
        
        # Compute only features with non-zero weights
        features = {}
        
        if self.weights.get('density', 0) > 0:
            density = self._compute_note_density(notes_soa)
            features['density'] = self._normalize_feature(density, 'density')
        
        if self.weights.get('pitch_range', 0) > 0:
            pitch_range = self._compute_pitch_range(notes_soa)
            features['pitch_range'] = self._normalize_feature(pitch_range, 'pitch_range')
        
        if self.weights.get('velocity', 0) > 0:
            velocity = self._compute_velocity_dynamics(notes_soa)
            features['velocity'] = self._normalize_feature(velocity, 'velocity')
        
        if self.weights.get('rhythm', 0) > 0:
            rhythm = self._compute_rhythmic_irregularity(notes_soa, time_division)
            features['rhythm'] = rhythm  # Already normalized
        
        if self.weights.get('dissonance', 0) > 0:
            dissonance = self._compute_harmonic_dissonance(notes_soa)
            features['dissonance'] = dissonance  # Already normalized
        
        if self.weights.get('register', 0) > 0:
            register = self._compute_register(notes_soa)
            features['register'] = self._normalize_feature(register, 'register')
        
        # Weighted combination
        # Only sum weights that are actually used
        active_weights = {k: v for k, v in self.weights.items() if k in features}
        total_weight = sum(active_weights.values())
        
        if total_weight == 0:
            # No features enabled, return minimum tension
            return self.tension_min
        
        weighted_tension = sum(
            features[name] * active_weights[name]
            for name in features.keys()
        )
        
        # Normalize by total weight
        tension = weighted_tension / total_weight
        
        # Scale to [tension_min, tension_max]
        scaled_tension = (
            self.tension_min + tension * (self.tension_max - self.tension_min)
        )
        
        return float(scaled_tension)

    def _tension_to_bin(
        self,
        tension: float,
    ) -> int:
        """
        Convert tension value to bin index.
        
        :param tension: tension value
        :return: bin index (0 to num_bins-1)
        """
        # Clip to valid range
        tension = np.clip(tension, self.tension_min, self.tension_max)
        
        # Find bin
        bin_idx = np.searchsorted(self.bin_edges[1:], tension)
        
        # Ensure within valid range
        bin_idx = min(bin_idx, self.num_bins - 1)
        
        return int(bin_idx)

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        """
        Compute tension control token for a bar.
        
        :param notes_soa: notes structure of arrays for the bar
        :param controls_soa: controls structure of arrays for the bar (unused)
        :param pitch_bends_soa: pitch bends structure of arrays for the bar (unused)
        :param time_division: ticks per quarter note
        :return: list containing single tension event
        """
        del controls_soa, pitch_bends_soa
        
        # Compute tension value
        tension = self._compute_tension_value(notes_soa, time_division)
        
        # Convert to bin
        tension_bin = self._tension_to_bin(tension)
        
        return [Event(self.token_type, tension_bin)]


class BarTensionMulti(BarAttributeControl):
    """
    Multi-component bar-level tension control.
    
    Provides separate control tokens for different tension aspects:
    - Density: based on note count and rhythm
    - Harmonic: based on dissonance
    - Melodic: based on pitch range and variance
    - Dynamic: based on velocity
    
    :param num_bins: number of bins for each component (default: 5)
    :param weights: dict of weights for each component (default: all 1.0)
    """

    def __init__(
        self,
        num_bins: int = 5,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.num_bins = num_bins
        
        # Default weights for sub-features
        default_weights = {
            'density': {'density': 1.0, 'rhythm': 1.0},
            'harmonic': {'dissonance': 1.0},
            'melodic': {'pitch_range': 1.0, 'register': 0.5},
            'dynamic': {'velocity': 1.0},
        }
        self.weights = weights if weights is not None else default_weights
        
        # Create individual tension computers for each component
        self._density_tension = BarTension(
            num_bins=num_bins,
            weights={'density': 1.5, 'rhythm': 1.0, 'pitch_range': 0.0, 
                    'velocity': 0.0, 'dissonance': 0.0, 'register': 0.0}
        )
        
        self._harmonic_tension = BarTension(
            num_bins=num_bins,
            weights={'density': 0.0, 'rhythm': 0.0, 'pitch_range': 0.0,
                    'velocity': 0.0, 'dissonance': 2.0, 'register': 0.0}
        )
        
        self._melodic_tension = BarTension(
            num_bins=num_bins,
            weights={'density': 0.0, 'rhythm': 0.0, 'pitch_range': 1.5,
                    'velocity': 0.0, 'dissonance': 0.0, 'register': 1.0}
        )
        
        self._dynamic_tension = BarTension(
            num_bins=num_bins,
            weights={'density': 0.0, 'rhythm': 0.0, 'pitch_range': 0.0,
                    'velocity': 2.0, 'dissonance': 0.0, 'register': 0.0}
        )
        
        # Create tokens for each component
        tokens = []
        for component in ["Density", "Harmonic", "Melodic", "Dynamic"]:
            tokens.extend([
                f"ACBarTension{component}_{i}" for i in range(num_bins)
            ])
        
        super().__init__(tokens=tokens)

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        """
        Compute multi-component tension control tokens for a bar.
        
        :return: list of 4 events (one per component)
        """
        # Compute each component using the specialized tension computers
        density_tension = self._density_tension._compute_tension_value(
            notes_soa, time_division
        )
        density_bin = self._density_tension._tension_to_bin(density_tension)
        
        harmonic_tension = self._harmonic_tension._compute_tension_value(
            notes_soa, time_division
        )
        harmonic_bin = self._harmonic_tension._tension_to_bin(harmonic_tension)
        
        melodic_tension = self._melodic_tension._compute_tension_value(
            notes_soa, time_division
        )
        melodic_bin = self._melodic_tension._tension_to_bin(melodic_tension)
        
        dynamic_tension = self._dynamic_tension._compute_tension_value(
            notes_soa, time_division
        )
        dynamic_bin = self._dynamic_tension._tension_to_bin(dynamic_tension)
        
        return [
            Event("ACBarTensionDensity", density_bin),
            Event("ACBarTensionHarmonic", harmonic_bin),
            Event("ACBarTensionMelodic", melodic_bin),
            Event("ACBarTensionDynamic", dynamic_bin),
        ]


class BarTensionSimple(BarAttributeControl):
    """
    Simplified bar-level tension control using only basic features.
    
    A lightweight alternative that uses only note density and dissonance.
    Useful when computational efficiency is important or for simpler models.
    
    :param num_bins: number of discrete tension bins (default: 5)
    :param density_weight: weight for note density (default: 1.0)
    :param dissonance_weight: weight for dissonance (default: 1.0)
    """

    def __init__(
        self,
        num_bins: int = 5,
        density_weight: float = 1.0,
        dissonance_weight: float = 1.0,
    ) -> None:
        self.num_bins = num_bins
        self.density_weight = density_weight
        self.dissonance_weight = dissonance_weight
        
        super().__init__(
            tokens=[f"ACBarTension_{i}" for i in range(num_bins)],
        )

    def _compute_on_bar(
        self,
        notes_soa: dict[str, np.ndarray],
        controls_soa: dict[str, np.ndarray],
        pitch_bends_soa: dict[str, np.ndarray],
        time_division: int,
    ) -> list[Event]:
        """Compute simplified tension using only density and dissonance."""
        del controls_soa, pitch_bends_soa, time_division
        
        if len(notes_soa["time"]) == 0:
            return [Event("ACBarTension", 0)]
        
        # Note density (normalized to typical range 0-20 notes)
        n_notes = len(notes_soa["time"])
        density = min(n_notes / 20.0, 1.0)
        
        # Simple dissonance: count minor 2nds and tritones
        pitches = notes_soa["pitch"]
        times = notes_soa["time"]
        unique_times = np.unique(times)
        
        dissonance_count = 0
        total_pairs = 0
        
        for t in unique_times:
            sim_pitches = pitches[times == t]
            if len(sim_pitches) < 2:
                continue
                
            for i in range(len(sim_pitches)):
                for j in range(i + 1, len(sim_pitches)):
                    interval = abs(sim_pitches[i] - sim_pitches[j]) % 12
                    if interval in [1, 6, 11]:  # m2, tritone, M7
                        dissonance_count += 1
                    total_pairs += 1
        
        dissonance = dissonance_count / total_pairs if total_pairs > 0 else 0.0
        
        # Weighted combination
        total_weight = self.density_weight + self.dissonance_weight
        tension = (
            density * self.density_weight + dissonance * self.dissonance_weight
        ) / total_weight
        
        # Convert to bin
        bin_idx = int(tension * self.num_bins)
        bin_idx = min(bin_idx, self.num_bins - 1)
        
        return [Event("ACBarTension", bin_idx)]


class BarTensionInstrument(BarTension):
    """
    Bar-level tension control optimized for pitched instruments.
    
    Uses feature weights tuned for melodic instruments:
    - Density: 2.0
    - Pitch range: 3.0 (melodic contour - emphasize pitch-based features)
    - Velocity: 3.0 (dynamics are important for instruments)
    - Rhythm: 2.0
    - Dissonance: 1.0 (harmonic tension)
    - Register: 1.0
    
    This configuration emphasizes melodic and dynamic aspects which are
    most relevant for pitched instruments.
    
    :param num_bins: number of discrete tension bins (default: 10)
    """

    def __init__(self, num_bins: int = 10) -> None:
        # Feature weights for pitched instruments
        # [density, pitch_range, velocity, rhythm, dissonance, register]
        # corresponds to [2, 3, 3, 2, 1, 1] from the reference model
        instrument_weights = {
            'density': 2.0,
            'pitch_range': 3.0,  # Melodic contour
            'velocity': 3.0,
            'rhythm': 2.0,
            'dissonance': 1.0,   # Harmonic tension
            'register': 1.0,
        }
        
        super().__init__(
            tension_min=0.0,
            tension_max=1.0,
            num_bins=num_bins,
            weights=instrument_weights,
            exclude_programs=[-1],
            token_suffix="Inst"
        )


class BarTensionDrum(BarTension):
    """
    Bar-level tension control optimized for drum tracks.
    
    Uses feature weights tuned for percussion/drums:
    - Density: 2.0 (onset frequency is key for drums)
    - Pitch range: 0.0 (not relevant for drums)
    - Velocity: 3.0 (dynamics are very important for drums)
    - Rhythm: 2.0 (rhythmic patterns are essential)
    - Dissonance: 0.0 (harmonic concepts don't apply)
    - Register: 0.0 (pitch height not meaningful)
    
    This configuration focuses on rhythmic and dynamic aspects which are
    most relevant for drum/percussion tracks, while excluding pitch-based
    features that aren't meaningful for unpitched instruments.
    
    :param num_bins: number of discrete tension bins (default: 10)
    """

    def __init__(self, num_bins: int = 10) -> None:
        # Feature weights for drums
        # [density, pitch_range, velocity, rhythm, dissonance, register]
        # corresponds to [2, 0, 3, 2, 0, 0] from the reference model
        drum_weights = {
            'density': 2.0,
            'pitch_range': 0.0,  # Not relevant for drums
            'velocity': 3.0,
            'rhythm': 2.0,
            'dissonance': 0.0,   # Not relevant for drums
            'register': 0.0,     # Not relevant for drums
        }
        
        super().__init__(
            tension_min=0.0,
            tension_max=1.0,
            num_bins=num_bins,
            weights=drum_weights,
            exclude_programs=list(range(128)),
            token_suffix="Drum"
        )