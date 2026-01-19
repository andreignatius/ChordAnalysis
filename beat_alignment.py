"""
Beat Alignment and Drift Correction
Author: Andre Lim

Uses musical heuristics to correct timing drift in transcription:
1. Onsets tend to be on beat or half-beat positions
2. There should be onsets at/near bar boundaries (STRONG heuristic)
3. Most notes don't exceed 1.5 bars in duration
4. BPM should be in a "musical" range (prefer 60-140 for most music)

Algorithm:
1. Detect candidate beat grid from onsets
2. Try BPM and BPM/2 candidates (detect double-time issues)
3. Score grids heavily on bar boundary alignment
4. Snap onsets to nearest valid beat positions
5. Validate and correct note durations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from pathlib import Path
import json
import math


# =============================================================================
# Musical Constants
# =============================================================================

# Preferred BPM range for most music (ballads, pop, etc.)
PREFERRED_BPM_MIN = 60
PREFERRED_BPM_MAX = 140

# Extended range (for fast music)
ABSOLUTE_BPM_MIN = 40
ABSOLUTE_BPM_MAX = 200

# Valid note durations in beats
VALID_DURATIONS = [
    0.25,   # Sixteenth
    0.5,    # Eighth
    0.75,   # Dotted eighth
    1.0,    # Quarter
    1.5,    # Dotted quarter
    2.0,    # Half
    3.0,    # Dotted half
    4.0,    # Whole
    6.0,    # Dotted whole (1.5 bars in 4/4)
]


# =============================================================================
# Beat Grid
# =============================================================================

@dataclass
class BeatGrid:
    """A candidate beat grid with BPM and phase."""
    bpm: float
    first_beat_time: float  # Phase offset
    time_signature_numerator: int = 4
    
    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm
    
    @property
    def seconds_per_bar(self) -> float:
        return self.seconds_per_beat * self.time_signature_numerator
    
    def time_to_beat(self, time: float) -> float:
        """Convert time to beat position (can be fractional)."""
        return (time - self.first_beat_time) / self.seconds_per_beat
    
    def beat_to_time(self, beat: float) -> float:
        """Convert beat position to time."""
        return self.first_beat_time + beat * self.seconds_per_beat
    
    def time_to_bar_beat(self, time: float) -> tuple[int, float]:
        """Convert time to (bar, beat_in_bar). Both 1-indexed."""
        total_beats = self.time_to_beat(time)
        if total_beats < 0:
            return (0, 1.0)
        bar = int(total_beats // self.time_signature_numerator) + 1
        beat_in_bar = (total_beats % self.time_signature_numerator) + 1
        return (bar, beat_in_bar)
    
    def get_bar_start_time(self, bar: int) -> float:
        """Get the start time of a bar (1-indexed)."""
        return self.first_beat_time + (bar - 1) * self.seconds_per_bar
    
    def snap_to_grid(self, time: float, resolution: float = 0.5) -> float:
        """Snap time to nearest beat grid position."""
        beat = self.time_to_beat(time)
        snapped_beat = round(beat / resolution) * resolution
        return self.beat_to_time(snapped_beat)
    
    def distance_to_grid(self, time: float, resolution: float = 0.5) -> float:
        """Calculate distance from time to nearest grid position in beats."""
        beat = self.time_to_beat(time)
        snapped_beat = round(beat / resolution) * resolution
        return abs(beat - snapped_beat)
    
    def distance_to_bar_start(self, time: float) -> float:
        """Calculate distance from time to nearest bar start in beats."""
        beat = self.time_to_beat(time)
        bar_beat = round(beat / self.time_signature_numerator) * self.time_signature_numerator
        return abs(beat - bar_beat)


# =============================================================================
# Improved BPM Detection
# =============================================================================

def compute_onset_strength(
    audio: np.ndarray,
    sample_rate: int,
    window_size: int = 2048,
    hop_size: int = 512
) -> tuple[np.ndarray, float]:
    """
    Compute onset strength function.
    
    Returns (onset_strength, frames_per_second)
    """
    num_frames = 1 + (len(audio) - window_size) // hop_size
    onset_strength = np.zeros(num_frames)
    prev_spectrum = None
    
    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + window_size] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(frame))
        
        if prev_spectrum is not None:
            diff = np.maximum(spectrum - prev_spectrum, 0)
            onset_strength[i] = np.sum(diff)
        prev_spectrum = spectrum
    
    if onset_strength.max() > 0:
        onset_strength = onset_strength / onset_strength.max()
    
    frames_per_second = sample_rate / hop_size
    return onset_strength, frames_per_second


def get_tempo_octave_candidates(
    detected_bpm: float,
    prefer_range: tuple[float, float] = (60, 140)
) -> list[float]:
    """
    Generate tempo octave candidates from a detected BPM.
    
    Returns all valid candidates within the preferred range,
    so they can be tested for bar alignment.
    
    Common tempo octave relationships:
    - x/2: Detected eighth notes as quarter notes
    - x*2: Detected half notes as quarter notes  
    - x/1.5: Dotted rhythm correction (e.g., 133 -> 88.7)
    - x*1.5: Dotted rhythm correction (e.g., 67 -> 100.5)
    """
    min_pref, max_pref = prefer_range
    
    candidates = [
        detected_bpm,
        detected_bpm / 2,
        detected_bpm * 2,
        detected_bpm / 1.5,
        detected_bpm * 1.5,
        detected_bpm * 2 / 3,
        detected_bpm * 3 / 2,
    ]
    
    # Filter to valid range and deduplicate
    valid = list(set(round(c) for c in candidates if min_pref <= c <= max_pref))
    
    return sorted(valid)


def detect_bpm_tempogram(
    audio: np.ndarray,
    sample_rate: int,
    min_bpm: float = 60,
    max_bpm: float = 160,
    window_seconds: float = 8.0,
    hop_seconds: float = 4.0
) -> tuple[float, float, list[tuple[float, float, float]]]:
    """
    Detect BPM using tempogram analysis (tempo over time).
    
    More robust than single autocorrelation - analyzes tempo in windows
    and takes the median, which handles tempo octave ambiguity better.
    
    Returns (bpm, confidence, tempo_curve) where tempo_curve is list of
    (time, bpm, confidence) tuples.
    """
    from scipy.signal import find_peaks
    
    # Compute onset envelope
    window_size = 2048
    hop_size = 512
    num_frames = 1 + (len(audio) - window_size) // hop_size
    
    onset_env = np.zeros(num_frames)
    prev_spectrum = None
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size
        if end > len(audio):
            break
        frame = audio[start:end] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(frame))
        if prev_spectrum is not None:
            onset_env[i] = np.sum(np.maximum(spectrum - prev_spectrum, 0))
        prev_spectrum = spectrum
    
    if onset_env.max() > 0:
        onset_env = onset_env / onset_env.max()
    
    fps = sample_rate / hop_size
    window_frames = int(window_seconds * fps)
    hop_frames = int(hop_seconds * fps)
    
    bpm_range = np.arange(int(min_bpm), int(max_bpm) + 1, 1)
    tempo_estimates = []
    
    for start_frame in range(0, len(onset_env) - window_frames, hop_frames):
        segment = onset_env[start_frame:start_frame + window_frames]
        
        # Autocorrelation
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(segment)-1:]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        # Find best BPM
        best_bpm = 120
        best_strength = 0
        
        for bpm in bpm_range:
            lag = int(fps * 60 / bpm)
            if 0 < lag < len(autocorr):
                strength = autocorr[lag]
                if strength > best_strength:
                    best_strength = strength
                    best_bpm = bpm
        
        time_sec = start_frame / fps
        tempo_estimates.append((time_sec, float(best_bpm), float(best_strength)))
    
    # Get confident estimates only
    confident_bpms = [t[1] for t in tempo_estimates if t[2] > 0.2]
    
    if not confident_bpms:
        return 120.0, 0.0, tempo_estimates
    
    # Use median (robust to outliers and tempo octave errors)
    median_bpm = float(np.median(confident_bpms))
    
    # Confidence based on consistency
    bpm_std = np.std(confident_bpms)
    confidence = max(0, 1.0 - bpm_std / 30)  # Lower std = higher confidence
    
    return median_bpm, confidence, tempo_estimates


def detect_bpm_autocorrelation(
    audio: np.ndarray,
    sample_rate: int,
    min_bpm: float = 60,
    max_bpm: float = 150
) -> list[tuple[float, float]]:
    """
    Detect BPM candidates using autocorrelation of onset strength.
    
    Returns list of (bpm, confidence) tuples sorted by confidence.
    """
    from scipy.signal import find_peaks
    
    onset_strength, fps = compute_onset_strength(audio, sample_rate)
    
    # Lag range for BPM search
    min_lag = int(fps * 60 / max_bpm)
    max_lag = int(fps * 60 / min_bpm)
    
    # Compute autocorrelation
    n = len(onset_strength)
    autocorr = np.correlate(onset_strength, onset_strength, mode='full')
    autocorr = autocorr[n-1:]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    
    # Find peaks
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    
    valid_autocorr = autocorr[min_lag:max_lag]
    peaks, props = find_peaks(valid_autocorr, height=0.1, distance=5)
    
    candidates = []
    
    if len(peaks) > 0:
        for i, peak in enumerate(peaks):
            lag = peak + min_lag
            bpm = 60 * fps / lag
            confidence = props['peak_heights'][i]
            candidates.append((bpm, confidence))
    
    # Sort by confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    return candidates[:10]


def estimate_bpm_from_onsets(
    onset_times: list[float],
    min_bpm: float = ABSOLUTE_BPM_MIN,
    max_bpm: float = ABSOLUTE_BPM_MAX
) -> list[float]:
    """
    Estimate candidate BPMs from onset intervals.
    
    Returns list of candidate BPMs sorted by likelihood.
    """
    if len(onset_times) < 3:
        return [120.0]
    
    # Calculate all inter-onset intervals
    intervals = np.diff(sorted(onset_times))
    
    # Filter to reasonable beat intervals
    min_interval = 60.0 / max_bpm
    max_interval = 60.0 / min_bpm
    
    valid_intervals = intervals[(intervals >= min_interval) & (intervals <= max_interval)]
    
    if len(valid_intervals) < 3:
        return [120.0, 90.0, 100.0]
    
    # Build histogram of intervals
    hist, bin_edges = np.histogram(valid_intervals, bins=100)
    
    # Find peaks in histogram
    candidates = []
    
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 2:
            interval = (bin_edges[i] + bin_edges[i+1]) / 2
            bpm = 60.0 / interval
            count = hist[i]
            candidates.append((bpm, count))
    
    if not candidates:
        # Use median interval
        median_interval = np.median(valid_intervals)
        candidates = [(60.0 / median_interval, 1)]
    
    # Sort by count (most common first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Extract BPMs and also add half/double variants
    bpms = []
    for bpm, _ in candidates[:5]:
        bpms.append(bpm)
        if bpm / 2 >= min_bpm:
            bpms.append(bpm / 2)
        if bpm * 2 <= max_bpm:
            bpms.append(bpm * 2)
    
    return list(set(bpms))


def score_beat_grid(
    grid: BeatGrid,
    onset_times: list[float],
    duration: float,
    resolution: float = 0.5,
    prefer_bpm_range: tuple[float, float] = (70, 130)
) -> dict:
    """
    Score how well a beat grid fits the onsets.
    
    HEAVILY weights bar boundary alignment and BPM preference.
    """
    if not onset_times:
        return {'total': 0, 'bar_alignment': 0, 'beat_alignment': 0, 'bpm_preference': 0}
    
    # 1. BAR ALIGNMENT: Do onsets fall on bar boundaries?
    bar_alignment_scores = []
    
    for onset in onset_times:
        dist_to_bar = grid.distance_to_bar_start(onset)
        if dist_to_bar < 0.3:
            bar_alignment_scores.append(1.0)
        elif dist_to_bar < 0.6:
            bar_alignment_scores.append(0.7)
        elif abs(dist_to_bar - grid.time_signature_numerator/2) < 0.3:
            bar_alignment_scores.append(0.5)
        else:
            bar_alignment_scores.append(0.0)
    
    bar_alignment = np.mean(bar_alignment_scores) if bar_alignment_scores else 0
    
    # 2. BEAT ALIGNMENT: Do onsets fall on beat/half-beat?
    beat_distances = [grid.distance_to_grid(t, resolution) for t in onset_times]
    avg_beat_distance = np.mean(beat_distances)
    beat_alignment = max(0, 1.0 - avg_beat_distance * 2)
    
    # 3. BPM PREFERENCE: STRONG penalty for being outside preferred range
    min_pref, max_pref = prefer_bpm_range
    if min_pref <= grid.bpm <= max_pref:
        bpm_preference = 1.0
    else:
        # Distance from preferred range
        if grid.bpm < min_pref:
            distance = (min_pref - grid.bpm) / min_pref
        else:
            distance = (grid.bpm - max_pref) / max_pref
        # Strong penalty - score drops quickly outside range
        bpm_preference = max(0, 1.0 - distance * 3)
    
    # 4. BAR COVERAGE: What percentage of bars have an onset near their start?
    num_bars = max(1, int(duration / grid.seconds_per_bar))
    bars_with_onset = 0
    
    for bar in range(1, num_bars + 1):
        bar_start = grid.get_bar_start_time(bar)
        for onset in onset_times:
            if abs(grid.time_to_beat(onset) - grid.time_to_beat(bar_start)) < 0.6:
                bars_with_onset += 1
                break
    
    bar_coverage = bars_with_onset / num_bars
    
    # Weighted total - BPM PREFERENCE is now more important
    total = (
        0.30 * bar_alignment +
        0.20 * bar_coverage +
        0.15 * beat_alignment +
        0.35 * bpm_preference  # Increased weight
    )
    
    return {
        'total': total,
        'bar_alignment': bar_alignment,
        'bar_coverage': bar_coverage,
        'beat_alignment': beat_alignment,
        'bpm_preference': bpm_preference,
        'avg_beat_distance': avg_beat_distance
    }


def find_best_beat_grid(
    onset_times: list[float],
    duration: float,
    initial_bpm: float,
    time_signature_numerator: int = 4,
    prefer_bpm_range: tuple[float, float] = (70, 130),
    audio: np.ndarray = None,
    sample_rate: int = 44100
) -> tuple[BeatGrid, dict]:
    """
    Find the best beat grid by searching BPM and phase space.
    
    Uses multiple strategies:
    1. Tempogram analysis (most reliable for real BPM)
    2. Autocorrelation of onset strength
    3. BPM halving/doubling for tempo octave errors
    """
    if not onset_times:
        return BeatGrid(initial_bpm, 0.0, time_signature_numerator), {'total': 0}
    
    print(f"    Searching for best grid (initial BPM: {initial_bpm:.1f})...")
    
    # Generate BPM candidates
    bpm_candidates = set()
    
    # 1. From tempogram (most reliable - uses median across song)
    if audio is not None:
        tempogram_bpm, tempogram_conf, tempo_curve = detect_bpm_tempogram(
            audio, sample_rate,
            min_bpm=ABSOLUTE_BPM_MIN,
            max_bpm=ABSOLUTE_BPM_MAX
        )
        
        # Get all tempo octave candidates (don't pick one yet - test them all)
        tempo_candidates = get_tempo_octave_candidates(tempogram_bpm, prefer_bpm_range)
        
        print(f"    Tempogram BPM: {tempogram_bpm:.1f} -> candidates: {tempo_candidates}")
        
        # Add all candidates
        for bpm in tempo_candidates:
            bpm_candidates.add(bpm)
        
        # Also get autocorrelation candidates
        autocorr_candidates = detect_bpm_autocorrelation(
            audio, sample_rate, 
            min_bpm=ABSOLUTE_BPM_MIN, 
            max_bpm=ABSOLUTE_BPM_MAX
        )
        for bpm, conf in autocorr_candidates[:3]:
            # Add this BPM and its octave variants
            for variant in get_tempo_octave_candidates(bpm, prefer_bpm_range):
                bpm_candidates.add(variant)
    
    # 2. From initial estimate
    bpm_candidates.add(round(initial_bpm))
    bpm_candidates.add(round(initial_bpm / 2))
    bpm_candidates.add(round(initial_bpm * 2))
    
    # 3. Add preferred range candidates
    for bpm in range(prefer_bpm_range[0], prefer_bpm_range[1] + 1, 5):
        bpm_candidates.add(float(bpm))
    
    # Filter to valid range
    bpm_candidates = [b for b in bpm_candidates 
                      if ABSOLUTE_BPM_MIN <= b <= ABSOLUTE_BPM_MAX]
    
    # Add fine variations around tempogram result (most likely correct)
    if audio is not None:
        for delta in range(-5, 6):
            candidate = round(tempogram_bpm) + delta
            if ABSOLUTE_BPM_MIN <= candidate <= ABSOLUTE_BPM_MAX:
                bpm_candidates.append(candidate)
    
    bpm_candidates = sorted(set(bpm_candidates))
    print(f"    Testing {len(bpm_candidates)} BPM candidates...")
    
    # Filter to strong onsets for scoring
    if len(onset_times) > 50:
        onset_times_for_scoring = sorted(onset_times)[::3]
    else:
        onset_times_for_scoring = onset_times
    
    best_grid = None
    best_score = {'total': -1}
    
    for bpm in bpm_candidates:
        seconds_per_beat = 60.0 / bpm
        seconds_per_bar = seconds_per_beat * time_signature_numerator
        
        phase_candidates = set()
        
        for onset in onset_times[:30]:
            for beat_offset in range(time_signature_numerator):
                phase = onset - beat_offset * seconds_per_beat
                if 0 <= phase <= seconds_per_bar:
                    phase_candidates.add(round(phase, 3))
        
        for i in range(20):
            phase_candidates.add(round(i * seconds_per_beat / 4, 3))
        
        for phase in phase_candidates:
            if phase < 0:
                continue
            grid = BeatGrid(bpm, phase, time_signature_numerator)
            score = score_beat_grid(grid, onset_times_for_scoring, duration, 
                                    prefer_bpm_range=prefer_bpm_range)
            
            if score['total'] > best_score['total']:
                best_score = score
                best_grid = grid
    
    print(f"    Best: BPM={best_grid.bpm:.1f}, phase={best_grid.first_beat_time:.3f}s")
    print(f"    Score: {best_score['total']:.3f} "
          f"(bar_align={best_score['bar_alignment']:.2f}, "
          f"bar_cov={best_score['bar_coverage']:.2f}, "
          f"beat={best_score['beat_alignment']:.2f})")
    
    return best_grid, best_score


def find_best_phase_for_bpm(
    bpm: float,
    onset_times: list[float],
    duration: float,
    time_signature_numerator: int = 4
) -> BeatGrid:
    """
    Given a fixed BPM, find the best phase (first beat time).
    Constrains phase to be non-negative (music starts at or after time 0).
    """
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * time_signature_numerator
    
    best_grid = None
    best_score = {'total': -1}
    
    # Try phases from 0 to 2 bars
    # This covers cases where first downbeat is at the start or after an intro
    phase_candidates = set()
    
    # Regular grid search
    for phase in np.linspace(0, seconds_per_bar * 2, 100):
        phase_candidates.add(round(phase, 3))
    
    # Also try phases that put onsets exactly on bar starts
    for onset in onset_times[:30]:
        if onset >= 0:
            # This onset could be bar 1
            phase_candidates.add(round(onset, 3))
            # Or bar 2, 3...
            for bar_offset in range(1, 4):
                phase = onset - bar_offset * seconds_per_bar
                if 0 <= phase <= seconds_per_bar * 2:
                    phase_candidates.add(round(phase, 3))
    
    for phase in phase_candidates:
        if phase < 0:
            continue  # Skip negative phases
            
        grid = BeatGrid(bpm, phase, time_signature_numerator)
        score = score_beat_grid(grid, onset_times, duration)
        if score['total'] > best_score['total']:
            best_score = score
            best_grid = grid
    
    if best_grid is None:
        # Fallback
        best_grid = BeatGrid(bpm, 0.0, time_signature_numerator)
    
    return best_grid


def detect_bpm_with_constraints(
    onset_times: list[float],
    duration: float,
    time_signature_numerator: int = 4,
    prefer_range: tuple[float, float] = (70, 130),
    require_bar_alignment: float = 0.3,
    audio: np.ndarray = None,
    sample_rate: int = 44100
) -> BeatGrid:
    """
    Detect BPM with strong constraints on bar alignment.
    
    Args:
        onset_times: Detected onset times
        duration: Audio duration
        time_signature_numerator: Beats per bar
        prefer_range: Preferred BPM range
        require_bar_alignment: Minimum bar alignment score
        audio: Raw audio for autocorrelation (optional but recommended)
        sample_rate: Audio sample rate
    """
    # First, get initial estimate from onset intervals
    onset_bpms = estimate_bpm_from_onsets(onset_times)
    
    if onset_bpms:
        initial_bpm = onset_bpms[0]
    else:
        initial_bpm = 120.0
    
    # If initial BPM is outside preferred range, try halving/doubling
    if initial_bpm > prefer_range[1]:
        if initial_bpm / 2 >= prefer_range[0]:
            initial_bpm = initial_bpm / 2
    elif initial_bpm < prefer_range[0]:
        if initial_bpm * 2 <= prefer_range[1]:
            initial_bpm = initial_bpm * 2
    
    # Find best grid (now with audio for autocorrelation)
    best_grid, best_score = find_best_beat_grid(
        onset_times, duration, initial_bpm, 
        time_signature_numerator, prefer_range,
        audio=audio, sample_rate=sample_rate
    )
    
    # Validate bar alignment
    if best_score['bar_alignment'] < require_bar_alignment:
        print(f"    Warning: Low bar alignment ({best_score['bar_alignment']:.2f})")
        print(f"    Trying alternative BPMs...")
        
        # Try BPM/2 explicitly
        half_grid, half_score = find_best_beat_grid(
            onset_times, duration, best_grid.bpm / 2,
            time_signature_numerator, prefer_range,
            audio=audio, sample_rate=sample_rate
        )
        
        if half_score['bar_alignment'] > best_score['bar_alignment']:
            print(f"    Half-time BPM works better!")
            return half_grid
    
    return best_grid


# =============================================================================
# Onset Snapping
# =============================================================================

def snap_onsets_to_grid(
    onset_times: list[float],
    grid: BeatGrid,
    resolution: float = 0.5,
    max_snap_distance: float = 0.3
) -> list[dict]:
    """Snap onset times to nearest grid positions."""
    snapped = []
    
    for onset in onset_times:
        distance = grid.distance_to_grid(onset, resolution)
        
        if distance <= max_snap_distance:
            snapped_time = grid.snap_to_grid(onset, resolution)
            bar, beat = grid.time_to_bar_beat(snapped_time)
            
            snapped.append({
                'original_time': onset,
                'snapped_time': snapped_time,
                'bar': bar,
                'beat': beat,
                'snap_distance': distance,
                'on_grid': True
            })
        else:
            bar, beat = grid.time_to_bar_beat(onset)
            snapped.append({
                'original_time': onset,
                'snapped_time': onset,
                'bar': bar,
                'beat': beat,
                'snap_distance': distance,
                'on_grid': False
            })
    
    return snapped


# =============================================================================
# Duration Validation
# =============================================================================

def validate_duration(duration_beats: float, beats_per_bar: int = 4) -> float:
    """Validate and correct note duration using musical heuristics."""
    max_duration = beats_per_bar * 1.5  # 1.5 bars
    
    if duration_beats > max_duration:
        valid = [d for d in VALID_DURATIONS if d <= max_duration]
        if valid:
            return max(valid)
        return max_duration
    
    if duration_beats < 0.25:
        return 0.25
    
    nearest = min(VALID_DURATIONS, key=lambda d: abs(d - duration_beats))
    
    if abs(nearest - duration_beats) / max(duration_beats, 0.1) < 0.3:
        return nearest
    
    return duration_beats


# =============================================================================
# Full Alignment Pipeline
# =============================================================================

def align_transcription(
    fused_transcription,
    onset_times: list[float],
    snap_resolution: float = 0.5,
    max_snap_distance: float = 0.3,
    max_note_duration_bars: float = 1.5,
    refine_grid: bool = True,
    prefer_bpm_range: tuple[float, float] = (70, 130),
    audio: np.ndarray = None,
    sample_rate: int = 44100,
    manual_bpm: float = None  # User can specify BPM directly
) -> tuple:
    """
    Full alignment pipeline with improved BPM detection.
    
    Args:
        audio: Raw audio array for autocorrelation-based BPM detection
        sample_rate: Audio sample rate
        manual_bpm: If specified, use this BPM instead of auto-detection
    """
    from note_event_fusion import FusedTranscription, NoteEvent
    
    print("\n" + "=" * 70)
    print("BEAT ALIGNMENT AND DRIFT CORRECTION")
    print("=" * 70)
    
    ts_parts = fused_transcription.time_signature.split('/')
    ts_num = int(ts_parts[0])
    
    initial_grid = BeatGrid(
        bpm=fused_transcription.bpm,
        first_beat_time=fused_transcription.first_beat_time,
        time_signature_numerator=ts_num
    )
    
    print(f"Initial: BPM={initial_grid.bpm:.1f}, first_beat={initial_grid.first_beat_time:.3f}s")
    
    if manual_bpm is not None:
        # User specified BPM - just find best phase
        print(f"\nUsing manual BPM: {manual_bpm}")
        grid = find_best_phase_for_bpm(
            manual_bpm, onset_times, 
            fused_transcription.duration_seconds,
            ts_num
        )
        grid_score = score_beat_grid(grid, onset_times, fused_transcription.duration_seconds)
        print(f"  Best phase: {grid.first_beat_time:.3f}s")
        print(f"  Score: {grid_score['total']:.3f} (bar_align={grid_score['bar_alignment']:.2f})")
        
    elif refine_grid and onset_times:
        print(f"\nRefining beat grid (prefer {prefer_bpm_range[0]}-{prefer_bpm_range[1]} BPM)...")
        
        grid = detect_bpm_with_constraints(
            onset_times,
            fused_transcription.duration_seconds,
            ts_num,
            prefer_range=prefer_bpm_range,
            audio=audio,
            sample_rate=sample_rate
        )
        
        grid_score = score_beat_grid(grid, onset_times, fused_transcription.duration_seconds)
    else:
        grid = initial_grid
        grid_score = score_beat_grid(grid, onset_times, fused_transcription.duration_seconds)
    
    print(f"\nFinal grid: BPM={grid.bpm:.1f}, first_beat={grid.first_beat_time:.3f}s")
    
    # Snap notes
    print("\nSnapping note start times to grid...")
    
    aligned_events = []
    snap_stats = {'on_grid': 0, 'off_grid': 0, 'total_snap': 0}
    
    for event in fused_transcription.note_events:
        distance = grid.distance_to_grid(event.start_time, snap_resolution)
        
        if distance <= max_snap_distance:
            snapped_time = grid.snap_to_grid(event.start_time, snap_resolution)
            snap_stats['on_grid'] += 1
        else:
            snapped_time = event.start_time
            snap_stats['off_grid'] += 1
        
        snap_stats['total_snap'] += distance
        
        bar, beat = grid.time_to_bar_beat(snapped_time)
        
        corrected_duration = validate_duration(event.duration_beats, ts_num)
        corrected_duration = min(corrected_duration, max_note_duration_bars * ts_num)
        
        aligned_event = NoteEvent(
            midi_note=event.midi_note,
            note_name=event.note_name,
            start_time=snapped_time,
            duration=corrected_duration * grid.seconds_per_beat,
            bar=bar,
            beat=beat,
            duration_beats=corrected_duration,
            velocity=event.velocity,
            role=event.role,
            is_tied=event.is_tied
        )
        
        aligned_events.append(aligned_event)
    
    aligned_events.sort(key=lambda e: (e.start_time, -e.midi_note))
    
    aligned = FusedTranscription(
        duration_seconds=fused_transcription.duration_seconds,
        source_file=fused_transcription.source_file,
        bpm=grid.bpm,
        time_signature=fused_transcription.time_signature,
        first_beat_time=grid.first_beat_time,
        note_events=aligned_events
    )
    
    total_events = len(fused_transcription.note_events)
    print(f"  On-grid: {snap_stats['on_grid']}/{total_events} "
          f"({100*snap_stats['on_grid']/max(1,total_events):.1f}%)")
    print(f"  Avg snap distance: {snap_stats['total_snap']/max(1,total_events):.3f} beats")
    
    # Show bar alignment stats
    bar_aligned = sum(1 for e in aligned_events if abs(e.beat - 1.0) < 0.1)
    print(f"  Notes on beat 1: {bar_aligned}/{total_events} ({100*bar_aligned/max(1,total_events):.1f}%)")
    
    print("=" * 70)
    
    stats = {
        'grid_bpm': grid.bpm,
        'grid_first_beat': grid.first_beat_time,
        'grid_score': grid_score,
        'on_grid_count': snap_stats['on_grid'],
        'off_grid_count': snap_stats['off_grid']
    }
    
    return aligned, stats


# =============================================================================
# Convenience Functions
# =============================================================================

def align_fused_json(
    input_path: str | Path,
    output_path: str | Path = None,
    onset_times: list[float] = None,
    refine_grid: bool = True,
    prefer_bpm_range: tuple[float, float] = (70, 130)
) -> 'FusedTranscription':
    """Load, align, and save a fused transcription."""
    from note_event_fusion import FusedTranscription, export_fused_txt
    
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / 'transcription_aligned.json'
    else:
        output_path = Path(output_path)
    
    fused = FusedTranscription.from_json_file(input_path)
    
    if onset_times is None:
        onset_times = sorted(set(e.start_time for e in fused.note_events))
    
    aligned, stats = align_transcription(
        fused, onset_times,
        refine_grid=refine_grid,
        prefer_bpm_range=prefer_bpm_range
    )
    
    aligned.save(output_path)
    
    txt_path = output_path.with_suffix('.txt')
    export_fused_txt(aligned, txt_path)
    
    return aligned

    
    print("=" * 90)