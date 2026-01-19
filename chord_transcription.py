"""
Two-Pass Transcription with BPM Detection and Bar/Beat Assignment
Author: Andre Lim

Extended features:
- Automatic BPM detection from onset patterns
- Time signature assignment (defaults to 4/4)
- Bar and beat position for each slice
- Enhanced transcription.txt with bar/beat columns

Outputs:
- transcription.json (full data with timing info)
- transcription.txt (human readable with bar/beat/chord)
- reconstructed WAV files
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, get_window
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict
import json
from datetime import datetime
import subprocess
import math


# =============================================================================
# Audio Loading
# =============================================================================

def mp3_to_wav(input_file: str, output_file: str) -> None:
    subprocess.run([
        'ffmpeg', '-y', '-i', input_file,
        '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
        output_file
    ], capture_output=True)
    print(f"Converted {input_file} -> {output_file}")


def load_wav(filepath: str) -> tuple[np.ndarray, int]:
    sample_rate, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    return data, sample_rate


# =============================================================================
# Utilities
# =============================================================================

def freq_to_midi(freq: float) -> int:
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_freq(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def midi_to_note_name(midi: int) -> str:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HarmonicInfo:
    frequency: float
    amplitude: float
    harmonic_number: int
    
    def to_dict(self) -> dict:
        return {'freq': round(self.frequency, 2), 'amp': round(self.amplitude, 2), 'num': self.harmonic_number}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HarmonicInfo':
        return cls(frequency=d['freq'], amplitude=d['amp'], harmonic_number=d['num'])


@dataclass
class DetectedNote:
    midi_note: int
    note_name: str
    fundamental_freq: float
    fundamental_amp: float
    total_energy: float
    harmonics: list[HarmonicInfo] = field(default_factory=list)
    confidence: float = 1.0
    role: str = 'unknown'
    
    def to_dict(self) -> dict:
        return {
            'midi': self.midi_note,
            'name': self.note_name,
            'freq': round(self.fundamental_freq, 2),
            'amp': round(self.fundamental_amp, 2),
            'energy': round(self.total_energy, 2),
            'confidence': round(self.confidence, 3),
            'role': self.role,
            'harmonics': [h.to_dict() for h in self.harmonics]
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DetectedNote':
        harmonics = [HarmonicInfo.from_dict(h) for h in d.get('harmonics', [])]
        return cls(
            midi_note=d['midi'],
            note_name=d['name'],
            fundamental_freq=d['freq'],
            fundamental_amp=d['amp'],
            total_energy=d['energy'],
            confidence=d.get('confidence', 1.0),
            role=d.get('role', 'unknown'),
            harmonics=harmonics
        )


@dataclass
class TimeSlice:
    """Extended TimeSlice with bar/beat information."""
    time: float
    bar: int = 0           # 1-indexed bar number
    beat: int = 0          # 1-indexed beat within bar
    beat_fraction: float = 0.0  # Fraction within beat (0.0-1.0)
    melody_notes: list[DetectedNote] = field(default_factory=list)
    chord_notes: list[DetectedNote] = field(default_factory=list)
    bass_notes: list[DetectedNote] = field(default_factory=list)
    
    @property
    def all_notes(self) -> list[DetectedNote]:
        return self.bass_notes + self.chord_notes + self.melody_notes
    
    @property
    def bar_beat_str(self) -> str:
        """Format as 'Bar.Beat' string, e.g., '1.1', '2.3'"""
        if self.beat_fraction > 0.01:
            # Show subdivision: 1.2.50 means bar 1, beat 2, halfway through
            return f"{self.bar}.{self.beat}.{int(self.beat_fraction * 100):02d}"
        return f"{self.bar}.{self.beat}"
    
    def to_dict(self) -> dict:
        return {
            'time': round(self.time, 4),
            'bar': self.bar,
            'beat': self.beat,
            'beat_fraction': round(self.beat_fraction, 3),
            'melody': [n.to_dict() for n in self.melody_notes],
            'chord': [n.to_dict() for n in self.chord_notes],
            'bass': [n.to_dict() for n in self.bass_notes]
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TimeSlice':
        melody = [DetectedNote.from_dict(n) for n in d.get('melody', [])]
        chord = [DetectedNote.from_dict(n) for n in d.get('chord', [])]
        bass = [DetectedNote.from_dict(n) for n in d.get('bass', [])]
        return cls(
            time=d['time'],
            bar=d.get('bar', 0),
            beat=d.get('beat', 0),
            beat_fraction=d.get('beat_fraction', 0.0),
            melody_notes=melody,
            chord_notes=chord,
            bass_notes=bass
        )


@dataclass
class TimingInfo:
    """Timing information for the transcription."""
    bpm: float = 120.0
    time_signature_numerator: int = 4
    time_signature_denominator: int = 4
    first_beat_time: float = 0.0  # Time of first detected downbeat
    confidence: float = 0.0  # Confidence in BPM detection
    
    @property
    def time_signature(self) -> str:
        return f"{self.time_signature_numerator}/{self.time_signature_denominator}"
    
    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm
    
    @property
    def seconds_per_bar(self) -> float:
        return self.seconds_per_beat * self.time_signature_numerator
    
    def to_dict(self) -> dict:
        return {
            'bpm': round(self.bpm, 2),
            'time_signature': self.time_signature,
            'time_signature_numerator': self.time_signature_numerator,
            'time_signature_denominator': self.time_signature_denominator,
            'first_beat_time': round(self.first_beat_time, 4),
            'confidence': round(self.confidence, 3)
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TimingInfo':
        return cls(
            bpm=d.get('bpm', 120.0),
            time_signature_numerator=d.get('time_signature_numerator', 4),
            time_signature_denominator=d.get('time_signature_denominator', 4),
            first_beat_time=d.get('first_beat_time', 0.0),
            confidence=d.get('confidence', 0.0)
        )


@dataclass
class TwoPassTranscription:
    """Extended transcription with timing information."""
    duration_seconds: float
    source_file: str
    slice_interval: float
    timing: TimingInfo = field(default_factory=TimingInfo)
    sample_rate: int = 44100
    hop_size: int = 4410
    window_size: int = 8820
    slices: list[TimeSlice] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def num_bars(self) -> int:
        if not self.slices:
            return 0
        return max(sl.bar for sl in self.slices)
    
    def to_dict(self) -> dict:
        return {
            'metadata': {
                'duration_seconds': round(self.duration_seconds, 2),
                'source_file': self.source_file,
                'slice_interval': self.slice_interval,
                'sample_rate': self.sample_rate,
                'hop_size': self.hop_size,
                'window_size': self.window_size,
                'num_slices': len(self.slices),
                'num_bars': self.num_bars,
                'created_at': self.created_at
            },
            'timing': self.timing.to_dict(),
            'slices': [s.to_dict() for s in self.slices]
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        print(f"Saved: {filepath}")
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TwoPassTranscription':
        meta = d['metadata']
        timing = TimingInfo.from_dict(d.get('timing', {}))
        slices = [TimeSlice.from_dict(s) for s in d.get('slices', [])]
        return cls(
            duration_seconds=meta['duration_seconds'],
            source_file=meta['source_file'],
            slice_interval=meta['slice_interval'],
            timing=timing,
            sample_rate=meta.get('sample_rate', 44100),
            hop_size=meta.get('hop_size', 4410),
            window_size=meta.get('window_size', 8820),
            slices=slices,
            created_at=meta.get('created_at', '')
        )
    
    @classmethod
    def from_json_file(cls, filepath: Path) -> 'TwoPassTranscription':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# BPM Detection
# =============================================================================

def compute_onset_envelope(
    audio: np.ndarray,
    sample_rate: int,
    window_size: int = 2048,
    hop_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Compute onset detection envelope combining spectral flux and energy."""
    num_frames = 1 + (len(audio) - window_size) // hop_size
    
    flux = np.zeros(num_frames)
    energy = np.zeros(num_frames)
    prev_spectrum = None
    
    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + window_size] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(frame))
        
        if prev_spectrum is not None:
            diff = np.maximum(spectrum - prev_spectrum, 0)
            flux[i] = np.sum(diff)
        
        energy[i] = np.sum(frame ** 2)
        prev_spectrum = spectrum
    
    energy_diff = np.maximum(np.diff(energy, prepend=energy[0]), 0)
    
    if flux.max() > 0:
        flux = flux / flux.max()
    if energy_diff.max() > 0:
        energy_diff = energy_diff / energy_diff.max()
    
    combined = 0.6 * flux + 0.4 * energy_diff
    times = np.arange(num_frames) * hop_size / sample_rate
    
    return combined, times


def detect_bpm_autocorrelation(
    onset_envelope: np.ndarray,
    sample_rate: int,
    hop_size: int = 512,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0
) -> tuple[float, float]:
    """
    Detect BPM using autocorrelation of onset envelope.
    Returns (bpm, confidence).
    """
    # Calculate lag range in frames
    min_lag = int(60.0 / max_bpm * sample_rate / hop_size)
    max_lag = int(60.0 / min_bpm * sample_rate / hop_size)
    
    # Compute autocorrelation
    n = len(onset_envelope)
    autocorr = np.correlate(onset_envelope, onset_envelope, mode='full')
    autocorr = autocorr[n-1:]  # Positive lags only
    
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    
    # Find peaks in valid range
    if max_lag > len(autocorr):
        max_lag = len(autocorr)
    
    valid_autocorr = autocorr[min_lag:max_lag]
    
    if len(valid_autocorr) == 0:
        return 120.0, 0.0
    
    peaks, properties = find_peaks(valid_autocorr, height=0.1, distance=5)
    
    if len(peaks) == 0:
        peak_idx = np.argmax(valid_autocorr)
        peak_value = valid_autocorr[peak_idx]
    else:
        best_peak = peaks[np.argmax(properties['peak_heights'])]
        peak_idx = best_peak
        peak_value = valid_autocorr[peak_idx]
    
    lag_frames = peak_idx + min_lag
    lag_seconds = lag_frames * hop_size / sample_rate
    
    if lag_seconds > 0:
        bpm = 60.0 / lag_seconds
    else:
        bpm = 120.0
    
    return bpm, float(peak_value)


def detect_bpm_onset_intervals(
    audio: np.ndarray,
    sample_rate: int,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0
) -> tuple[float, float, list[float]]:
    """
    Detect BPM by analyzing intervals between onsets.
    Returns (bpm, confidence, onset_times).
    """
    onset_env, times = compute_onset_envelope(audio, sample_rate)
    
    # Detect onsets
    threshold = onset_env.mean() + 0.5 * onset_env.std()
    min_gap_frames = int(0.05 * sample_rate / 512)
    
    peaks, _ = find_peaks(onset_env, height=threshold, distance=min_gap_frames)
    onset_times = times[peaks]
    
    if len(onset_times) < 4:
        bpm, conf = detect_bpm_autocorrelation(onset_env, sample_rate)
        return bpm, conf, list(onset_times)
    
    # Calculate inter-onset intervals
    intervals = np.diff(onset_times)
    
    min_interval = 60.0 / max_bpm
    max_interval = 60.0 / min_bpm
    
    valid_intervals = intervals[(intervals >= min_interval) & (intervals <= max_interval)]
    
    if len(valid_intervals) < 3:
        bpm, conf = detect_bpm_autocorrelation(onset_env, sample_rate)
        return bpm, conf, list(onset_times)
    
    # Histogram-based approach
    bins = np.linspace(min_interval, max_interval, 100)
    hist, bin_edges = np.histogram(valid_intervals, bins=bins)
    
    peak_bin = np.argmax(hist)
    beat_interval = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
    
    # Consider half/double time
    candidate_intervals = [beat_interval, beat_interval / 2, beat_interval * 2]
    
    best_bpm = 120.0
    best_score = 0
    
    for interval in candidate_intervals:
        bpm_candidate = 60.0 / interval
        
        if bpm_candidate < min_bpm or bpm_candidate > max_bpm:
            continue
        
        matches = 0
        for iv in valid_intervals:
            ratio = iv / interval
            nearest = round(ratio)
            if nearest > 0 and abs(ratio - nearest) / nearest < 0.1:
                matches += 1
        
        score = matches / len(valid_intervals)
        
        if score > best_score:
            best_score = score
            best_bpm = bpm_candidate
    
    # Cross-check with autocorrelation
    autocorr_bpm, autocorr_conf = detect_bpm_autocorrelation(onset_env, sample_rate)
    
    if abs(best_bpm - autocorr_bpm) / best_bpm < 0.1:
        best_score = min(1.0, best_score + 0.2)
    
    return best_bpm, best_score, list(onset_times)


def detect_first_downbeat(
    onset_times: list[float],
    bpm: float,
    search_window: float = 2.0
) -> float:
    """Detect the time of the first downbeat."""
    if not onset_times:
        return 0.0
    
    beat_interval = 60.0 / bpm
    candidates = [t for t in onset_times if t < search_window]
    
    if not candidates:
        return onset_times[0] if onset_times else 0.0
    
    best_time = candidates[0]
    best_score = 0
    
    for candidate in candidates:
        score = 0
        for onset in onset_times:
            if onset < candidate:
                continue
            
            offset = onset - candidate
            beat_num = offset / beat_interval
            nearest_beat = round(beat_num)
            
            if nearest_beat >= 0:
                error = abs(beat_num - nearest_beat)
                if error < 0.15:
                    score += 1 - error
        
        if score > best_score:
            best_score = score
            best_time = candidate
    
    return best_time


def detect_timing(
    audio: np.ndarray,
    sample_rate: int,
    time_signature_numerator: int = 4,
    time_signature_denominator: int = 4,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0
) -> TimingInfo:
    """Detect timing information from audio."""
    print("  Detecting BPM...")
    
    bpm, confidence, onset_times = detect_bpm_onset_intervals(
        audio, sample_rate, min_bpm, max_bpm
    )
    
    bpm_rounded = round(bpm)
    
    print(f"    Detected BPM: {bpm:.1f} (rounded to {bpm_rounded})")
    print(f"    Confidence: {confidence:.2f}")
    
    first_beat = detect_first_downbeat(onset_times, bpm_rounded)
    print(f"    First downbeat at: {first_beat:.3f}s")
    
    return TimingInfo(
        bpm=float(bpm_rounded),
        time_signature_numerator=time_signature_numerator,
        time_signature_denominator=time_signature_denominator,
        first_beat_time=first_beat,
        confidence=confidence
    )


# =============================================================================
# Bar/Beat Assignment
# =============================================================================

def assign_bar_beat_positions(
    slices: list[TimeSlice],
    timing: TimingInfo
) -> list[TimeSlice]:
    """Assign bar and beat positions to each slice."""
    seconds_per_beat = timing.seconds_per_beat
    beats_per_bar = timing.time_signature_numerator
    
    for sl in slices:
        relative_time = sl.time - timing.first_beat_time
        
        if relative_time < 0:
            # Pickup bar (bar 0)
            sl.bar = 0
            sl.beat = 1
            sl.beat_fraction = 0.0
        else:
            total_beats = relative_time / seconds_per_beat
            sl.bar = int(total_beats // beats_per_bar) + 1
            beat_in_bar = total_beats % beats_per_bar
            sl.beat = int(beat_in_bar) + 1
            sl.beat_fraction = beat_in_bar - int(beat_in_bar)
    
    return slices


# =============================================================================
# Chord Recognition
# =============================================================================

CHORD_TEMPLATES = {
    'maj': [0, 4, 7], 'min': [0, 3, 7], 'dim': [0, 3, 6], 'aug': [0, 4, 8],
    'sus2': [0, 2, 7], 'sus4': [0, 5, 7],
    'maj7': [0, 4, 7, 11], 'min7': [0, 3, 7, 10], '7': [0, 4, 7, 10],
    'dim7': [0, 3, 6, 9], 'min7b5': [0, 3, 6, 10],
    '6': [0, 4, 7, 9], 'min6': [0, 3, 7, 9],
    'add9': [0, 4, 7, 14], '9': [0, 4, 7, 10, 14],
    '5': [0, 7],
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_chord(midi_notes: list[int], min_score: float = 0.3) -> Optional[dict]:
    if not midi_notes or len(midi_notes) < 2:
        return None
    
    unique_notes = sorted(set(midi_notes))
    best_match = None
    best_score = min_score
    
    for root_midi in unique_notes:
        intervals = sorted(set((n - root_midi) % 12 for n in unique_notes))
        
        for template_name, template_intervals in CHORD_TEMPLATES.items():
            template_pcs = set(i % 12 for i in template_intervals)
            input_pcs = set(intervals)
            
            if 0 not in input_pcs:
                continue
            
            matched = template_pcs & input_pcs
            if len(matched) < 2:
                continue
            
            score = len(matched) / len(template_pcs)
            
            if root_midi == unique_notes[0]:
                score += 0.15
            
            simplicity = {'maj': 0.1, 'min': 0.1, '5': 0.05, '7': 0.05, 'maj7': 0.05, 'min7': 0.05}.get(template_name, 0)
            score += simplicity
            
            if score > best_score:
                best_score = score
                root_name = NOTE_NAMES[root_midi % 12]
                
                if template_name == 'maj':
                    chord_name = root_name
                elif template_name == 'min':
                    chord_name = f"{root_name}m"
                else:
                    chord_name = f"{root_name}{template_name}"
                
                best_match = {'name': chord_name, 'root': root_name, 'score': score}
    
    return best_match


def analyze_slice_chord(bass_notes: list, chord_notes: list) -> Optional[dict]:
    midi_notes = [n.midi_note for n in bass_notes] + [n.midi_note for n in chord_notes]
    return detect_chord(midi_notes)


def format_chord(chord: Optional[dict], width: int = 10) -> str:
    if chord is None:
        return "-".ljust(width)
    conf = "" if chord['score'] >= 0.6 else "?"
    return f"{chord['name']}{conf}".ljust(width)


# =============================================================================
# Chord Segmentation
# =============================================================================

@dataclass
class ChordSegment:
    start_time: float
    end_time: float
    start_bar: int
    start_beat: int
    chord_name: str
    bass_note: str
    confidence: float


def segment_chords_by_onsets(
    transcription: TwoPassTranscription,
    onsets: list[dict]
) -> list[ChordSegment]:
    """Cluster slices between onsets to determine stable chords."""
    onset_times = sorted([o['time'] for o in onsets])
    if not onset_times or onset_times[0] > 0.1:
        onset_times.insert(0, 0.0)
    if onset_times[-1] < transcription.duration_seconds:
        onset_times.append(transcription.duration_seconds)
    
    segments = []
    
    for i in range(len(onset_times) - 1):
        start = onset_times[i]
        end = onset_times[i+1]
        
        if end - start < 0.15:
            continue
        
        segment_chroma = defaultdict(float)
        segment_bass = defaultdict(float)
        
        relevant_slices = [s for s in transcription.slices if start <= s.time < end]
        
        if not relevant_slices:
            continue
        
        # Get bar/beat from first slice in segment
        start_bar = relevant_slices[0].bar
        start_beat = relevant_slices[0].beat
        
        for sl in relevant_slices:
            for note in sl.chord_notes:
                segment_chroma[note.midi_note % 12] += (note.fundamental_amp * note.confidence)
            for note in sl.bass_notes:
                segment_bass[note.midi_note] += (note.fundamental_amp * note.confidence)
        
        if not segment_bass:
            best_bass_midi = None
            bass_name = "-"
        else:
            best_bass_midi = max(segment_bass, key=segment_bass.get)
            bass_name = midi_to_note_name(best_bass_midi)
        
        sorted_chroma = sorted(segment_chroma.items(), key=lambda x: x[1], reverse=True)
        top_pcs = [pc for pc, amp in sorted_chroma[:4] if amp > 0.05]
        
        if top_pcs:
            synthetic_midis = []
            if best_bass_midi:
                synthetic_midis.append(best_bass_midi)
            for pc in top_pcs:
                synthetic_midis.append(60 + pc)
            
            detected = detect_chord(synthetic_midis)
            chord_name = detected['name'] if detected else "N.C."
            score = detected['score'] if detected else 0.0
        else:
            chord_name = "N.C."
            score = 0.0
        
        segments.append(ChordSegment(start, end, start_bar, start_beat, chord_name, bass_name, score))
    
    return segments


# =============================================================================
# Melody Cleaning
# =============================================================================

def analyze_melody_note(
    melody_note: DetectedNote,
    bass_notes: list[DetectedNote],
    chord_notes: list[DetectedNote],
    cent_tolerance: float = 50.0,
    base_amp_ratio: float = 1.5
) -> dict:
    analysis = {
        'note': melody_note.note_name,
        'is_real': True,
        'dominated_by': None,
        'issues': []
    }
    
    m_freq = melody_note.fundamental_freq
    m_amp = melody_note.fundamental_amp
    if m_freq <= 0:
        return analysis
    
    other_notes = bass_notes + chord_notes
    
    for other in other_notes:
        for harm in other.harmonics:
            if harm.frequency <= 0:
                continue
            diff_cents = abs(1200 * math.log2(m_freq / harm.frequency))
            
            if diff_cents < cent_tolerance:
                n = harm.harmonic_number
                dynamic_ratio = base_amp_ratio * (1 + (n * 0.15))
                
                if harm.amplitude > m_amp * dynamic_ratio:
                    analysis['is_real'] = False
                    analysis['dominated_by'] = f"H{n} of {other.note_name}"
                    return analysis
    
    return analysis


def clean_melody_slice(
    melody_notes: list[DetectedNote],
    bass_notes: list[DetectedNote],
    chord_notes: list[DetectedNote],
    min_confidence: float = 0.3,
    apply_skyline: bool = True,
    skyline_max: int = 2,
    verbose: bool = False
) -> list[DetectedNote]:
    if not melody_notes:
        return []
    
    cleaned = []
    
    for melody in melody_notes:
        analysis = analyze_melody_note(melody, bass_notes, chord_notes)
        if analysis['is_real']:
            cleaned.append(melody)
    
    # Deduplicate frequencies
    freq_groups = defaultdict(list)
    for note in cleaned:
        freq_key = round(note.fundamental_freq / 5) * 5
        freq_groups[freq_key].append(note)
    
    deduplicated = []
    for freq_key, notes in freq_groups.items():
        best = max(notes, key=lambda n: n.confidence)
        deduplicated.append(best)
    
    cleaned = [n for n in deduplicated if n.confidence >= min_confidence]
    
    if apply_skyline and len(cleaned) > skyline_max:
        cleaned = sorted(cleaned, key=lambda n: n.midi_note, reverse=True)[:skyline_max]
    
    return cleaned


def clean_melody_transcription(
    transcription: TwoPassTranscription,
    min_confidence: float = 0.3,
    apply_skyline: bool = True,
    skyline_max: int = 2,
    remove_isolated: bool = True,
    min_consecutive: int = 2,
    verbose: bool = False
) -> TwoPassTranscription:
    print("\n  Cleaning melody (harmonic artifact removal)...")
    
    total_before = sum(len(sl.melody_notes) for sl in transcription.slices)
    
    for sl in transcription.slices:
        sl.melody_notes = clean_melody_slice(
            sl.melody_notes, sl.bass_notes, sl.chord_notes,
            min_confidence, apply_skyline, skyline_max, verbose
        )
    
    if remove_isolated:
        note_presence = defaultdict(list)
        for i, sl in enumerate(transcription.slices):
            for note in sl.melody_notes:
                note_presence[note.midi_note].append(i)
        
        valid_notes_at_slice = defaultdict(set)
        
        for midi, indices in note_presence.items():
            indices = sorted(indices)
            i = 0
            while i < len(indices):
                run_start = i
                run_end = i
                while run_end + 1 < len(indices) and indices[run_end + 1] - indices[run_end] <= 2:
                    run_end += 1
                if run_end - run_start + 1 >= min_consecutive:
                    for j in range(run_start, run_end + 1):
                        valid_notes_at_slice[indices[j]].add(midi)
                i = run_end + 1
        
        for i, sl in enumerate(transcription.slices):
            valid_midis = valid_notes_at_slice.get(i, set())
            sl.melody_notes = [n for n in sl.melody_notes if n.midi_note in valid_midis]
    
    total_after = sum(len(sl.melody_notes) for sl in transcription.slices)
    print(f"    Melody notes: {total_before} -> {total_after}")
    
    return transcription


# =============================================================================
# Onset Detection
# =============================================================================

def compute_spectral_flux(audio: np.ndarray, sample_rate: int,
                          window_size: int = 2048, hop_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    num_frames = 1 + (len(audio) - window_size) // hop_size
    onset_function = np.zeros(num_frames)
    prev_spectrum = None
    
    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + window_size] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(frame))
        
        if prev_spectrum is not None:
            diff = np.maximum(spectrum - prev_spectrum, 0)
            onset_function[i] = np.sum(diff)
        prev_spectrum = spectrum
    
    times = np.arange(num_frames) * hop_size / sample_rate
    return onset_function, times


def compute_energy_onset(audio: np.ndarray, sample_rate: int,
                         window_size: int = 2048, hop_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    num_frames = 1 + (len(audio) - window_size) // hop_size
    energy = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + window_size]
        energy[i] = np.sum(frame ** 2)
    
    onset_function = np.maximum(np.diff(energy, prepend=energy[0]), 0)
    times = np.arange(num_frames) * hop_size / sample_rate
    return onset_function, times


def compute_hfc(audio: np.ndarray, sample_rate: int,
                window_size: int = 2048, hop_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    num_frames = 1 + (len(audio) - window_size) // hop_size
    hfc = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start:start + window_size] * np.hanning(window_size)
        spectrum = np.abs(np.fft.rfft(frame))
        weights = np.arange(len(spectrum)) + 1
        hfc[i] = np.sum(spectrum * weights)
    
    onset_function = np.maximum(np.diff(hfc, prepend=hfc[0]), 0)
    times = np.arange(num_frames) * hop_size / sample_rate
    return onset_function, times


def detect_onsets(audio: np.ndarray, sample_rate: int,
                  threshold_rel: float = 0.1, min_gap_ms: float = 30) -> list[dict]:
    flux, times = compute_spectral_flux(audio, sample_rate)
    energy, _ = compute_energy_onset(audio, sample_rate)
    hfc, _ = compute_hfc(audio, sample_rate)
    
    if flux.max() > 0: flux = flux / flux.max()
    if energy.max() > 0: energy = energy / energy.max()
    if hfc.max() > 0: hfc = hfc / hfc.max()
    
    combined = 0.4 * flux + 0.3 * energy + 0.3 * hfc
    threshold = combined.mean() + threshold_rel * combined.std()
    
    min_gap_frames = max(1, int(min_gap_ms / 1000 * sample_rate / 512))
    peaks, _ = find_peaks(combined, height=threshold, distance=min_gap_frames, prominence=0.01)
    
    max_val = combined.max() if combined.max() > 0 else 1.0
    return [{'time': times[p], 'strength': combined[p] / max_val} for p in peaks]


# =============================================================================
# Spectrum Analysis & Note Detection
# =============================================================================

def get_spectrum_at_time(audio: np.ndarray, sample_rate: int, time: float,
                         window_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    center = int(time * sample_rate)
    start = max(0, center - window_size // 2)
    end = min(len(audio), start + window_size)
    
    frame = np.zeros(window_size)
    frame[:end - start] = audio[start:end]
    frame *= get_window('hann', window_size)
    
    fft_size = window_size * 2
    padded = np.zeros(fft_size)
    padded[:window_size] = frame
    
    spectrum = np.abs(np.fft.rfft(padded))
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    
    return spectrum, frequencies


def find_spectral_peaks(spectrum: np.ndarray, frequencies: np.ndarray,
                        min_freq: float, max_freq: float,
                        amplitude_threshold: float = 0.02) -> list[tuple[float, float]]:
    min_idx = np.searchsorted(frequencies, min_freq)
    max_idx = np.searchsorted(frequencies, max_freq)
    
    region = spectrum[min_idx:max_idx]
    if len(region) == 0 or region.max() == 0:
        return []
    
    region_norm = region / region.max()
    peak_indices, _ = find_peaks(region_norm, height=amplitude_threshold, distance=3, prominence=0.01)
    peak_indices += min_idx
    
    peaks = [(frequencies[i], spectrum[i]) for i in peak_indices]
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    return peaks[:20]


def compute_harmonic_score(candidate_freq: float, all_peaks: list[tuple[float, float]],
                           tolerance: float = 0.03) -> tuple[float, list, float]:
    matched = []
    total_energy = 0.0
    weighted_score = 0.0
    
    for peak_freq, peak_amp in all_peaks:
        ratio = peak_freq / candidate_freq
        nearest = round(ratio)
        
        if nearest < 1 or nearest > 8:
            continue
        
        if abs(ratio - nearest) / nearest < tolerance:
            matched.append((peak_freq, peak_amp, int(nearest)))
            total_energy += peak_amp
            weighted_score += peak_amp / nearest
    
    harmonic_nums = sorted([m[2] for m in matched])
    bonus = sum(0.1 for i in range(len(harmonic_nums) - 1) if harmonic_nums[i + 1] == harmonic_nums[i] + 1)
    
    if any(m[2] == 1 for m in matched):
        weighted_score *= 1.2
    
    return weighted_score * (1 + bonus), matched, total_energy


def detect_notes_in_range(audio: np.ndarray, sample_rate: int, time: float,
                          min_freq: float, max_freq: float,
                          amplitude_threshold: float = 0.02,
                          max_notes: int = 6,
                          min_confidence: float = 0.2) -> list[DetectedNote]:
    spectrum, frequencies = get_spectrum_at_time(audio, sample_rate, time)
    peaks = find_spectral_peaks(spectrum, frequencies, min_freq, max_freq, amplitude_threshold)
    
    if not peaks:
        return []
    
    candidates = set()
    for freq, amp in peaks:
        candidates.add(freq)
        for div in [2, 3, 4, 5]:
            sub = freq / div
            if sub >= min_freq:
                candidates.add(sub)
    
    scored = []
    for cand in candidates:
        score, harmonics, energy = compute_harmonic_score(cand, peaks)
        if score > 0 and harmonics:
            scored.append({'freq': cand, 'score': score, 'harmonics': harmonics, 'energy': energy})
    
    if not scored:
        return []
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    max_score = scored[0]['score']
    
    notes = []
    used_peaks = set()
    
    for cand in scored:
        cand_peaks = set(h[0] for h in cand['harmonics'])
        if len(cand_peaks & used_peaks) > len(cand_peaks) * 0.5:
            continue
        
        confidence = cand['score'] / max_score if max_score > 0 else 0
        if confidence < min_confidence:
            continue
        
        fund_amp = 0
        fund_freq = cand['freq']
        for hf, ha, hn in cand['harmonics']:
            if hn == 1:
                fund_amp = ha
                fund_freq = hf
                break
        if fund_amp == 0:
            fund_amp = cand['energy'] / 3
        
        midi = freq_to_midi(cand['freq'])
        harmonics = [HarmonicInfo(h[0], h[1], h[2]) for h in cand['harmonics'] if h[2] > 1]
        
        notes.append(DetectedNote(
            midi_note=midi,
            note_name=midi_to_note_name(midi),
            fundamental_freq=fund_freq,
            fundamental_amp=fund_amp,
            total_energy=cand['energy'],
            harmonics=harmonics,
            confidence=confidence
        ))
        
        used_peaks.update(cand_peaks)
        if len(notes) >= max_notes:
            break
    
    notes.sort(key=lambda n: n.midi_note)
    return notes


# =============================================================================
# Two-Pass Detection
# =============================================================================

def detect_melody_pass(audio: np.ndarray, sample_rate: int, slice_times: np.ndarray,
                       onset_threshold_rel: float = 0.1,
                       melody_min_freq: float = 300.0,
                       melody_max_freq: float = 2000.0,
                       min_gap_ms: float = 30) -> tuple[dict, list]:
    print("  Pass 1: Melody detection (onset-based)...")
    
    onsets = detect_onsets(audio, sample_rate, threshold_rel=onset_threshold_rel, min_gap_ms=min_gap_ms)
    print(f"    Found {len(onsets)} onsets")
    
    melody_by_time = {}
    
    for onset in onsets:
        notes = detect_notes_in_range(
            audio, sample_rate, onset['time'],
            min_freq=melody_min_freq, max_freq=melody_max_freq,
            amplitude_threshold=0.03, max_notes=3, min_confidence=0.25
        )
        for n in notes:
            n.role = 'melody'
        if notes:
            melody_by_time[onset['time']] = notes
    
    print(f"    Melody notes at {len(melody_by_time)} onsets")
    return melody_by_time, onsets


def detect_chords_pass(audio: np.ndarray, sample_rate: int, slice_times: np.ndarray,
                       bass_min_freq: float = 50.0, bass_max_freq: float = 200.0,
                       chord_min_freq: float = 130.0, chord_max_freq: float = 500.0,
                       smoothing_window: int = 3) -> tuple[dict, dict]:
    print("  Pass 2: Chord/Bass detection (slice-based)...")
    
    raw_bass = {}
    raw_chords = {}
    
    for t in slice_times:
        bass_notes = detect_notes_in_range(
            audio, sample_rate, t,
            min_freq=bass_min_freq, max_freq=bass_max_freq,
            amplitude_threshold=0.02, max_notes=2, min_confidence=0.2
        )
        for n in bass_notes:
            n.role = 'bass'
        raw_bass[t] = bass_notes
        
        chord_notes = detect_notes_in_range(
            audio, sample_rate, t,
            min_freq=chord_min_freq, max_freq=chord_max_freq,
            amplitude_threshold=0.02, max_notes=6, min_confidence=0.2
        )
        for n in chord_notes:
            n.role = 'chord'
        raw_chords[t] = chord_notes
    
    print(f"    Applying temporal smoothing (window={smoothing_window})...")
    smoothed_bass = smooth_note_detections(raw_bass, slice_times, smoothing_window)
    smoothed_chords = smooth_note_detections(raw_chords, slice_times, smoothing_window)
    
    return smoothed_bass, smoothed_chords


def smooth_note_detections(notes_by_time: dict, slice_times: np.ndarray, window_size: int = 3) -> dict:
    if window_size < 2:
        return notes_by_time
    
    smoothed = {}
    half_window = window_size // 2
    times_list = list(slice_times)
    
    for i, t in enumerate(times_list):
        start_idx = max(0, i - half_window)
        end_idx = min(len(times_list), i + half_window + 1)
        
        note_counts = defaultdict(int)
        note_data = {}
        
        for j in range(start_idx, end_idx):
            window_time = times_list[j]
            for note in notes_by_time.get(window_time, []):
                note_counts[note.midi_note] += 1
                if note.midi_note not in note_data or note.total_energy > note_data[note.midi_note].total_energy:
                    note_data[note.midi_note] = note
        
        window_len = end_idx - start_idx
        threshold = window_len / 2
        
        filtered = [note_data[midi] for midi, count in note_counts.items()
                    if count >= threshold and midi in note_data]
        filtered.sort(key=lambda n: n.total_energy, reverse=True)
        smoothed[t] = filtered
    
    return smoothed


def merge_passes(slice_times: np.ndarray, melody_by_time: dict, onsets: list,
                 bass_by_time: dict, chords_by_time: dict,
                 melody_sustain_slices: int = 3,
                 slice_interval: float = 0.1) -> list[TimeSlice]:
    print("  Merging passes...")
    
    active_melody = {}
    
    for onset_time, notes in melody_by_time.items():
        for i, t in enumerate(slice_times):
            if t >= onset_time and t < onset_time + melody_sustain_slices * slice_interval:
                if t not in active_melody:
                    active_melody[t] = []
                active_melody[t].extend(notes)
    
    slices = []
    
    for t in slice_times:
        melody = active_melody.get(t, [])
        bass = bass_by_time.get(t, [])
        chords = chords_by_time.get(t, [])
        
        melody_midis = set(n.midi_note for n in melody)
        chords = [n for n in chords if n.midi_note not in melody_midis]
        
        bass_midis = set(n.midi_note for n in bass)
        chords = [n for n in chords if n.midi_note not in bass_midis]
        
        slices.append(TimeSlice(time=t, melody_notes=melody, chord_notes=chords, bass_notes=bass))
    
    return slices


# =============================================================================
# Reconstruction
# =============================================================================

def render_notes_to_signal(notes: list[DetectedNote], window_size: int,
                           sample_rate: int, max_notes: int = 8) -> np.ndarray:
    t = np.arange(window_size) / sample_rate
    signal = np.zeros(window_size)
    
    for note in notes[:max_notes]:
        amp = note.fundamental_amp * note.confidence
        freq = note.fundamental_freq
        signal += amp * np.sin(2 * np.pi * freq * t)
        
        for harm in note.harmonics:
            if harm.frequency < sample_rate / 2:
                harm_amp = harm.amplitude * note.confidence
                signal += harm_amp * np.sin(2 * np.pi * harm.frequency * t)
    
    return signal


def reconstruct_separated(transcription: TwoPassTranscription,
                          sample_rate: int = None) -> dict[str, np.ndarray]:
    sample_rate = sample_rate or transcription.sample_rate
    hop_size = transcription.hop_size
    window_size = transcription.window_size
    
    num_slices = len(transcription.slices)
    output_length = num_slices * hop_size + window_size
    
    bass_audio = np.zeros(output_length)
    chord_audio = np.zeros(output_length)
    melody_audio = np.zeros(output_length)
    
    synth_window = get_window('hann', window_size)
    
    for i, sl in enumerate(transcription.slices):
        start = i * hop_size
        
        bass_signal = render_notes_to_signal(sl.bass_notes, window_size, sample_rate, max_notes=4)
        chord_signal = render_notes_to_signal(sl.chord_notes, window_size, sample_rate, max_notes=6)
        melody_signal = render_notes_to_signal(sl.melody_notes, window_size, sample_rate, max_notes=3)
        
        bass_audio[start:start + window_size] += bass_signal * synth_window
        chord_audio[start:start + window_size] += chord_signal * synth_window
        melody_audio[start:start + window_size] += melody_signal * synth_window
    
    result = {}
    
    for name, audio in [('bass', bass_audio), ('chords', chord_audio), ('melody', melody_audio)]:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9 * 32767
        result[name] = audio.astype(np.int16)
    
    combined = bass_audio * 0.9 + chord_audio * 0.7 + melody_audio * 1.0
    max_val = np.abs(combined).max()
    if max_val > 0:
        combined = combined / max_val * 0.9 * 32767
    result['combined'] = combined.astype(np.int16)
    
    return result


# =============================================================================
# Export Functions
# =============================================================================

def export_transcription_txt(transcription: TwoPassTranscription, output_path: Path) -> None:
    """Export transcription with bar/beat information."""
    with open(output_path, 'w') as f:
        # Header with timing info
        f.write(f"# Transcription: {transcription.source_file}\n")
        f.write(f"# Duration: {transcription.duration_seconds:.2f}s\n")
        f.write(f"# BPM: {transcription.timing.bpm:.0f}\n")
        f.write(f"# Time Signature: {transcription.timing.time_signature}\n")
        f.write(f"# Total Bars: {transcription.num_bars}\n")
        f.write(f"# First Downbeat: {transcription.timing.first_beat_time:.3f}s\n")
        f.write(f"# BPM Confidence: {transcription.timing.confidence:.2f}\n\n")
        
        f.write(f"{'Bar.Beat':<10} {'Time':<8} {'Chord':<10} {'Bass':<12} {'Chords':<24} {'Melody'}\n")
        f.write("-" * 100 + "\n")
        
        for sl in transcription.slices:
            bar_beat = sl.bar_beat_str
            bass = " ".join(n.note_name for n in sl.bass_notes) or "-"
            chords = " ".join(n.note_name for n in sl.chord_notes) or "-"
            melody = " ".join(n.note_name for n in sl.melody_notes) or "-"
            
            chord = analyze_slice_chord(sl.bass_notes, sl.chord_notes)
            chord_str = format_chord(chord)
            
            f.write(f"{bar_beat:<10} {sl.time:<8.2f} {chord_str} {bass:<12} {chords:<24} {melody}\n")
    
    print(f"Saved: {output_path}")


def export_segmented_txt(transcription: TwoPassTranscription,
                         segments: list[ChordSegment],
                         output_path: Path) -> None:
    """Export chord segments with bar/beat information."""
    with open(output_path, 'w') as f:
        f.write(f"# Segmented Transcription: {transcription.source_file}\n")
        f.write(f"# BPM: {transcription.timing.bpm:.0f} | Time Signature: {transcription.timing.time_signature}\n")
        f.write(f"# Total Bars: {transcription.num_bars}\n\n")
        
        f.write(f"{'Bar.Beat':<10} {'Time':<8} {'Chord':<10} {'Bass':<8} {'Melody Notes'}\n")
        f.write("-" * 80 + "\n")
        
        for seg in segments:
            melody_in_seg = []
            for sl in transcription.slices:
                if seg.start_time <= sl.time < seg.end_time:
                    for n in sl.melody_notes:
                        melody_in_seg.append(n.note_name)
            
            seen = set()
            clean_melody = []
            for m in melody_in_seg:
                if m not in seen:
                    clean_melody.append(m)
                    seen.add(m)
            
            bar_beat = f"{seg.start_bar}.{seg.start_beat}"
            melody_str = " ".join(clean_melody) or "-"
            
            f.write(f"{bar_beat:<10} {seg.start_time:<8.2f} {seg.chord_name:<10} {seg.bass_note:<8} {melody_str}\n")
    
    print(f"Saved: {output_path}")


def print_transcription_summary(transcription: TwoPassTranscription, max_slices: int = 25) -> None:
    print("\n" + "=" * 105)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 105)
    
    print(f"Duration: {transcription.duration_seconds:.2f}s | "
          f"BPM: {transcription.timing.bpm:.0f} | "
          f"Time Sig: {transcription.timing.time_signature} | "
          f"Bars: {transcription.num_bars}")
    
    total_melody = sum(len(s.melody_notes) for s in transcription.slices)
    total_chord = sum(len(s.chord_notes) for s in transcription.slices)
    total_bass = sum(len(s.bass_notes) for s in transcription.slices)
    
    print(f"Notes - Melody: {total_melody} | Chords: {total_chord} | Bass: {total_bass}")
    print("-" * 105)
    
    print(f"{'Bar.Beat':<10} {'Time':<8} {'Chord':<10} {'Bass':<12} {'Chords':<24} {'Melody'}")
    print("-" * 105)
    
    for sl in transcription.slices[:max_slices]:
        bar_beat = sl.bar_beat_str
        chord = analyze_slice_chord(sl.bass_notes, sl.chord_notes)
        chord_str = format_chord(chord)
        bass = " ".join(n.note_name for n in sl.bass_notes)[:11] or "-"
        chords = " ".join(n.note_name for n in sl.chord_notes)[:23] or "-"
        melody = " ".join(n.note_name for n in sl.melody_notes) or "-"
        
        print(f"{bar_beat:<10} {sl.time:<8.2f} {chord_str} {bass:<12} {chords:<24} {melody}")
    
    if len(transcription.slices) > max_slices:
        print(f"... and {len(transcription.slices) - max_slices} more slices")
    print("=" * 105)


# =============================================================================
# Main Pipeline
# =============================================================================

def transcribe_two_pass(
    filepath: str | Path,
    output_dir: str | Path,
    slice_interval: float = 0.1,
    # Timing params
    detect_bpm: bool = True,
    manual_bpm: float = None,
    time_signature_numerator: int = 4,
    time_signature_denominator: int = 4,
    # Melody params
    onset_threshold_rel: float = 0.1,
    melody_min_freq: float = 300.0,
    melody_max_freq: float = 2000.0,
    melody_sustain_slices: int = 3,
    # Chord params
    chord_min_freq: float = 130.0,
    chord_max_freq: float = 500.0,
    chord_smoothing_window: int = 3,
    # Bass params
    bass_min_freq: float = 50.0,
    bass_max_freq: float = 200.0,
    # Melody cleaning params
    clean_melody: bool = True,
    melody_min_confidence: float = 0.3,
    melody_skyline_max: int = 2,
    melody_remove_isolated: bool = True,
    verbose_cleaning: bool = False
) -> TwoPassTranscription:
    """
    Full two-pass transcription with BPM detection and bar/beat assignment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.mp3':
        wav_file = output_dir / 'temp.wav'
        mp3_to_wav(str(filepath), str(wav_file))
    else:
        wav_file = filepath
    
    print(f"\nLoading: {wav_file}")
    audio, sample_rate = load_wav(str(wav_file))
    duration = len(audio) / sample_rate
    print(f"  Duration: {duration:.2f}s")
    
    # Detect timing
    if detect_bpm and manual_bpm is None:
        timing = detect_timing(
            audio, sample_rate,
            time_signature_numerator=time_signature_numerator,
            time_signature_denominator=time_signature_denominator
        )
    else:
        bpm = manual_bpm if manual_bpm else 120.0
        timing = TimingInfo(
            bpm=bpm,
            time_signature_numerator=time_signature_numerator,
            time_signature_denominator=time_signature_denominator,
            first_beat_time=0.0,
            confidence=1.0 if manual_bpm else 0.0
        )
        print(f"  Using {'manual' if manual_bpm else 'default'} BPM: {bpm}")
    
    hop_size = int(slice_interval * sample_rate)
    window_size = hop_size * 2
    
    slice_times = np.arange(0, duration, slice_interval)
    print(f"  Slices: {len(slice_times)} (interval={slice_interval}s)")
    
    # Pass 1: Melody
    melody_by_time, onsets = detect_melody_pass(
        audio, sample_rate, slice_times,
        onset_threshold_rel=onset_threshold_rel,
        melody_min_freq=melody_min_freq,
        melody_max_freq=melody_max_freq
    )
    
    # Pass 2: Chords/Bass
    bass_by_time, chords_by_time = detect_chords_pass(
        audio, sample_rate, slice_times,
        bass_min_freq=bass_min_freq,
        bass_max_freq=bass_max_freq,
        chord_min_freq=chord_min_freq,
        chord_max_freq=chord_max_freq,
        smoothing_window=chord_smoothing_window
    )
    
    # Merge
    slices = merge_passes(
        slice_times, melody_by_time, onsets,
        bass_by_time, chords_by_time,
        melody_sustain_slices=melody_sustain_slices,
        slice_interval=slice_interval
    )
    
    # Assign bar/beat positions
    print("  Assigning bar/beat positions...")
    slices = assign_bar_beat_positions(slices, timing)
    
    # Create transcription
    result = TwoPassTranscription(
        duration_seconds=duration,
        source_file=str(filepath),
        slice_interval=slice_interval,
        timing=timing,
        sample_rate=sample_rate,
        hop_size=hop_size,
        window_size=window_size,
        slices=slices
    )
    
    # Clean melody
    if clean_melody:
        result = clean_melody_transcription(
            result,
            min_confidence=melody_min_confidence,
            apply_skyline=True,
            skyline_max=melody_skyline_max,
            remove_isolated=melody_remove_isolated,
            verbose=verbose_cleaning
        )
    
    # Chord clustering
    print("\n  Performing onset-based chord clustering...")
    segments = segment_chords_by_onsets(result, onsets)
    
    # Fuse with onset timing for proper note durations
    print("\n  Fusing notes with onset timing...")
    try:
        from note_event_fusion import (
            fuse_from_transcription_result, 
            export_fused_txt, 
            print_fused_summary
        )
        
        fused = fuse_from_transcription_result(
            result, onsets,
            quantize_durations=True,
            quantize_starts=True
        )
        
        # Apply beat alignment to fix drift
        try:
            from beat_alignment import align_transcription
            
            onset_times = [o['time'] for o in onsets]
            aligned, align_stats = align_transcription(
                fused, onset_times,
                snap_resolution=0.5,  # Snap to half-beats
                max_snap_distance=0.3,
                max_note_duration_bars=1.5,
                refine_grid=True,
                prefer_bpm_range=(60, 140),  # Wide range to cover ballads and upbeat songs
                audio=audio,  # Pass audio for autocorrelation
                sample_rate=sample_rate,
                manual_bpm=manual_bpm  # Allow user to override BPM
            )
            
            # Save aligned version
            aligned.save(output_dir / 'transcription_aligned.json')
            export_fused_txt(aligned, output_dir / 'transcription_aligned.txt')
            
            # Also save the raw fused version for comparison
            fused.save(output_dir / 'transcription_fused.json')
            export_fused_txt(fused, output_dir / 'transcription_fused.txt')
            
            print_fused_summary(aligned, max_events=20)
            
        except ImportError:
            print("    Note: beat_alignment.py not found, skipping alignment")
            fused.save(output_dir / 'transcription_fused.json')
            export_fused_txt(fused, output_dir / 'transcription_fused.txt')
            print_fused_summary(fused, max_events=20)
        
    except ImportError:
        print("    Note: note_event_fusion.py not found, skipping fusion")
        fused = None
    
    # Print summary
    print_transcription_summary(result, max_slices=25)
    
    # Save outputs
    result.save(output_dir / 'transcription.json')
    export_transcription_txt(result, output_dir / 'transcription.txt')
    export_segmented_txt(result, segments, output_dir / 'transcription_chords.txt')
    
    # Reconstruct audio
    print("\nReconstructing audio...")
    separated = reconstruct_separated(result, sample_rate)
    
    wavfile.write(output_dir / 'reconstructed_combined.wav', sample_rate, separated['combined'])
    wavfile.write(output_dir / 'reconstructed_bass.wav', sample_rate, separated['bass'])
    wavfile.write(output_dir / 'reconstructed_chords.wav', sample_rate, separated['chords'])
    wavfile.write(output_dir / 'reconstructed_melody.wav', sample_rate, separated['melody'])
    
    # Save original
    original_int16 = (audio * 0.9 * 32767).astype(np.int16)
    wavfile.write(output_dir / 'original.wav', sample_rate, original_int16)
    
    print(f"\n✓ Done! Outputs in: {output_dir}/")
    print("  - transcription.json (with timing info)")
    print("  - transcription.txt (with bar/beat positions)")
    print("  - transcription_chords.txt (chord segments)")
    print("  - reconstructed_*.wav")
    
    return result


def reconstruct_from_json(json_path: str | Path,
                          output_dir: str | Path = None) -> TwoPassTranscription:
    """Load and reconstruct from JSON."""
    json_path = Path(json_path)
    output_dir = Path(output_dir) if output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {json_path}")
    transcription = TwoPassTranscription.from_json_file(json_path)
    
    print(f"  Duration: {transcription.duration_seconds:.2f}s")
    print(f"  BPM: {transcription.timing.bpm:.0f}")
    print(f"  Time Sig: {transcription.timing.time_signature}")
    print(f"  Bars: {transcription.num_bars}")
    
    sample_rate = transcription.sample_rate
    separated = reconstruct_separated(transcription, sample_rate)
    
    wavfile.write(output_dir / 'reconstructed_from_json_combined.wav', sample_rate, separated['combined'])
    wavfile.write(output_dir / 'reconstructed_from_json_bass.wav', sample_rate, separated['bass'])
    wavfile.write(output_dir / 'reconstructed_from_json_chords.wav', sample_rate, separated['chords'])
    wavfile.write(output_dir / 'reconstructed_from_json_melody.wav', sample_rate, separated['melody'])
    
    print(f"\n✓ Reconstructed from JSON!")
    
    return transcription


# =============================================================================
# Main
# =============================================================================

def main():
    result = transcribe_two_pass(
        filepath="best_part.mp3",
        output_dir=Path("transcription_output_bars_beats"),
        slice_interval=0.1,
        # Timing
        detect_bpm=True,
        manual_bpm=67,  # Set to override auto-detection (e.g., 120.0)
        time_signature_numerator=4,
        time_signature_denominator=4,
        # Melody
        onset_threshold_rel=0.1,
        melody_min_freq=300.0,
        melody_max_freq=2000.0,
        melody_sustain_slices=3,
        # Chords
        chord_min_freq=130.0,
        chord_max_freq=500.0,
        chord_smoothing_window=3,
        # Bass
        bass_min_freq=50.0,
        bass_max_freq=200.0,
        # Melody cleaning
        clean_melody=True,
        melody_min_confidence=0.3,
        melody_skyline_max=2,
        melody_remove_isolated=True,
    )


if __name__ == '__main__':
    main()