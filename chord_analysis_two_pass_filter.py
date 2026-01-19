"""
Two-Pass Transcription with JSON Reconstruction
Author: Andre Lim

Adds:
1. Load and reconstruct from JSON
2. Improved overlap-add synthesis (like the original reconstruct_audio_with_harmonics)
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

from chord_recognition import (
    export_transcription_with_chords,
    print_transcription_with_chords,
    analyze_slice_chord,
    format_chord_for_display
)

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
    time: float
    melody_notes: list[DetectedNote] = field(default_factory=list)
    chord_notes: list[DetectedNote] = field(default_factory=list)
    bass_notes: list[DetectedNote] = field(default_factory=list)
    
    @property
    def all_notes(self) -> list[DetectedNote]:
        return self.bass_notes + self.chord_notes + self.melody_notes
    
    def to_dict(self) -> dict:
        return {
            'time': round(self.time, 4),
            'melody': [n.to_dict() for n in self.melody_notes],
            'chord': [n.to_dict() for n in self.chord_notes],
            'bass': [n.to_dict() for n in self.bass_notes]
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TimeSlice':
        melody = [DetectedNote.from_dict(n) for n in d.get('melody', [])]
        chord = [DetectedNote.from_dict(n) for n in d.get('chord', [])]
        bass = [DetectedNote.from_dict(n) for n in d.get('bass', [])]
        return cls(time=d['time'], melody_notes=melody, chord_notes=chord, bass_notes=bass)


@dataclass
class TwoPassTranscription:
    duration_seconds: float
    source_file: str
    slice_interval: float
    sample_rate: int = 44100
    hop_size: int = 4410  # Default for 0.1s at 44100Hz
    window_size: int = 8820  # 2x hop for overlap-add
    slices: list[TimeSlice] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
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
                'created_at': self.created_at
            },
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
        slices = [TimeSlice.from_dict(s) for s in d.get('slices', [])]
        return cls(
            duration_seconds=meta['duration_seconds'],
            source_file=meta['source_file'],
            slice_interval=meta['slice_interval'],
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
    if region.max() == 0:
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
# Pass 1: Melody Detection
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
            min_freq=melody_min_freq,
            max_freq=melody_max_freq,
            amplitude_threshold=0.03,
            max_notes=3,
            min_confidence=0.25
        )
        
        for n in notes:
            n.role = 'melody'
        
        if notes:
            melody_by_time[onset['time']] = notes
    
    print(f"    Melody notes at {len(melody_by_time)} onsets")
    return melody_by_time, onsets


# =============================================================================
# Pass 2: Chord/Bass Detection
# =============================================================================

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


# =============================================================================
# Merge Passes
# =============================================================================

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
# Reconstruction - Overlap-Add Method (Like Original)
# =============================================================================

def reconstruct_overlap_add(
    transcription: TwoPassTranscription,
    sample_rate: int = None,
    hop_size: int = None,
    window_size: int = None
) -> np.ndarray:
    """
    Reconstruct audio using overlap-add synthesis.
    
    This matches the quality of the original reconstruct_audio_with_harmonics.
    """
    # Use transcription parameters if not specified
    sample_rate = sample_rate or transcription.sample_rate
    hop_size = hop_size or transcription.hop_size
    window_size = window_size or transcription.window_size
    
    num_slices = len(transcription.slices)
    output_length = num_slices * hop_size + window_size
    reconstructed = np.zeros(output_length)
    
    synth_window = get_window('hann', window_size)
    
    # Role volume weights
    role_volumes = {'bass': 0.9, 'chord': 0.7, 'melody': 1.0, 'unknown': 0.8}
    
    for i, sl in enumerate(transcription.slices):
        start = i * hop_size
        t = np.arange(window_size) / sample_rate
        
        frame_signal = np.zeros(window_size)
        
        # Process all notes in this slice
        all_notes = sl.all_notes
        
        for note in all_notes[:8]:  # Limit total notes per frame
            role_vol = role_volumes.get(note.role, 0.8)
            
            # Add fundamental
            amp = note.fundamental_amp * note.confidence * role_vol
            freq = note.fundamental_freq
            frame_signal += amp * np.sin(2 * np.pi * freq * t)
            
            # Add harmonics with their actual detected amplitudes
            for harm in note.harmonics:
                if harm.frequency < sample_rate / 2:
                    harm_amp = harm.amplitude * note.confidence * role_vol
                    frame_signal += harm_amp * np.sin(2 * np.pi * harm.frequency * t)
        
        # Apply window and overlap-add
        reconstructed[start:start + window_size] += frame_signal * synth_window
    
    # Normalize
    max_val = np.abs(reconstructed).max()
    if max_val > 0:
        reconstructed = reconstructed / max_val * 0.9 * 32767
    
    return reconstructed.astype(np.int16)


def reconstruct_overlap_add_separated(
    transcription: TwoPassTranscription,
    sample_rate: int = None,
    hop_size: int = None,
    window_size: int = None
) -> dict[str, np.ndarray]:
    """
    Reconstruct separate audio for bass, chords, and melody.
    
    Returns dict with 'bass', 'chords', 'melody', 'combined' arrays.
    """
    sample_rate = sample_rate or transcription.sample_rate
    hop_size = hop_size or transcription.hop_size
    window_size = window_size or transcription.window_size
    
    num_slices = len(transcription.slices)
    output_length = num_slices * hop_size + window_size
    
    bass_audio = np.zeros(output_length)
    chord_audio = np.zeros(output_length)
    melody_audio = np.zeros(output_length)
    
    synth_window = get_window('hann', window_size)
    
    for i, sl in enumerate(transcription.slices):
        start = i * hop_size
        t = np.arange(window_size) / sample_rate
        
        def render_notes(notes: list[DetectedNote]) -> np.ndarray:
            signal = np.zeros(window_size)
            for note in notes[:6]:
                amp = note.fundamental_amp * note.confidence
                freq = note.fundamental_freq
                signal += amp * np.sin(2 * np.pi * freq * t)
                
                for harm in note.harmonics:
                    if harm.frequency < sample_rate / 2:
                        signal += harm.amplitude * note.confidence * np.sin(2 * np.pi * harm.frequency * t)
            return signal
        
        bass_audio[start:start + window_size] += render_notes(sl.bass_notes) * synth_window
        chord_audio[start:start + window_size] += render_notes(sl.chord_notes) * synth_window
        melody_audio[start:start + window_size] += render_notes(sl.melody_notes) * synth_window
    
    # Normalize each
    result = {}
    for name, audio in [('bass', bass_audio), ('chords', chord_audio), ('melody', melody_audio)]:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9 * 32767
        result[name] = audio.astype(np.int16)
    
    # Combined
    combined = bass_audio * 0.9 + chord_audio * 0.7 + melody_audio * 1.0
    max_val = np.abs(combined).max()
    if max_val > 0:
        combined = combined / max_val * 0.9 * 32767
    result['combined'] = combined.astype(np.int16)
    
    return result


# =============================================================================
# Load and Reconstruct from JSON
# =============================================================================

def reconstruct_from_json(
    json_path: str | Path,
    output_dir: str | Path = None,
    sample_rate: int = None
) -> tuple[TwoPassTranscription, np.ndarray]:
    """
    Load transcription from JSON and reconstruct audio.
    
    Args:
        json_path: Path to transcription.json
        output_dir: Output directory (if None, uses same directory as JSON)
        sample_rate: Override sample rate (if None, uses value from JSON)
    
    Returns:
        (transcription, combined_audio)
    """
    json_path = Path(json_path)
    
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    print(f"Loading transcription from: {json_path}")
    transcription = TwoPassTranscription.from_json_file(json_path)
    
    print(f"  Duration: {transcription.duration_seconds:.2f}s")
    print(f"  Slices: {len(transcription.slices)}")
    print(f"  Slice interval: {transcription.slice_interval}s")
    print(f"  Sample rate: {transcription.sample_rate}")
    print(f"  Hop size: {transcription.hop_size}")
    print(f"  Window size: {transcription.window_size}")
    
    # Count notes
    total_melody = sum(len(s.melody_notes) for s in transcription.slices)
    total_chord = sum(len(s.chord_notes) for s in transcription.slices)
    total_bass = sum(len(s.bass_notes) for s in transcription.slices)
    print(f"  Melody notes: {total_melody}")
    print(f"  Chord notes: {total_chord}")
    print(f"  Bass notes: {total_bass}")
    
    # Use specified sample rate or from transcription
    sr = sample_rate or transcription.sample_rate
    
    # Reconstruct
    print("\nReconstructing audio (overlap-add method)...")
    
    # Combined
    combined = reconstruct_overlap_add(transcription, sample_rate=sr)
    wavfile.write(output_dir / 'reconstructed_from_json.wav', sr, combined)
    print(f"  Saved: {output_dir / 'reconstructed_from_json.wav'}")
    
    # Separated
    print("Reconstructing separated tracks...")
    separated = reconstruct_overlap_add_separated(transcription, sample_rate=sr)
    
    for name, audio in separated.items():
        if name != 'combined':  # Already saved combined
            wavfile.write(output_dir / f'reconstructed_from_json_{name}.wav', sr, audio)
            print(f"  Saved: {output_dir / f'reconstructed_from_json_{name}.wav'}")
    
    print(f"\n✓ Reconstruction complete!")
    
    return transcription, combined


# =============================================================================
# Main Transcription Pipeline
# =============================================================================

def transcribe_two_pass(
    filepath: str | Path,
    output_dir: str | Path,
    slice_interval: float = 0.1,
    onset_threshold_rel: float = 0.1,
    melody_min_freq: float = 300.0,
    melody_max_freq: float = 2000.0,
    melody_sustain_slices: int = 3,
    chord_min_freq: float = 130.0,
    chord_max_freq: float = 500.0,
    chord_smoothing_window: int = 3,
    bass_min_freq: float = 50.0,
    bass_max_freq: float = 200.0
) -> TwoPassTranscription:
    """
    Two-pass transcription pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.mp3':
        wav_file = output_dir / 'temp.wav'
        mp3_to_wav(str(filepath), str(wav_file))
    else:
        wav_file = filepath
    
    print(f"Loading: {wav_file}")
    audio, sample_rate = load_wav(str(wav_file))
    duration = len(audio) / sample_rate
    print(f"  Duration: {duration:.2f}s")
    
    # Calculate synthesis parameters based on slice interval
    hop_size = int(slice_interval * sample_rate)
    window_size = hop_size * 2  # 2x for good overlap
    
    slice_times = np.arange(0, duration, slice_interval)
    print(f"  Slice interval: {slice_interval}s ({len(slice_times)} slices)")
    print(f"  Hop size: {hop_size}, Window size: {window_size}")
    
    # Pass 1: Melody
    melody_by_time, onsets = detect_melody_pass(
        audio, sample_rate, slice_times,
        onset_threshold_rel=onset_threshold_rel,
        melody_min_freq=melody_min_freq,
        melody_max_freq=melody_max_freq
    )
    
    # Pass 2: Chords and Bass
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
    
    # Create result with synthesis parameters
    result = TwoPassTranscription(
        duration_seconds=duration,
        source_file=str(filepath),
        slice_interval=slice_interval,
        sample_rate=sample_rate,
        hop_size=hop_size,
        window_size=window_size,
        slices=slices
    )
    
    print_summary(result)
    
    # Save JSON
    result.save(output_dir / 'transcription.json')
    export_txt_with_chords(result, output_dir / 'transcription.txt')
    
    # Reconstruct using overlap-add
    print("\nReconstructing audio (overlap-add method)...")
    combined = reconstruct_overlap_add(result)
    wavfile.write(output_dir / 'reconstructed_combined.wav', sample_rate, combined)
    
    # Separated tracks
    print("Reconstructing separated tracks...")
    separated = reconstruct_overlap_add_separated(result)
    for name, audio_data in separated.items():
        if name != 'combined':
            wavfile.write(output_dir / f'reconstructed_{name}.wav', sample_rate, audio_data)
    
    # Original for comparison
    original_int16 = (audio * 0.9 * 32767).astype(np.int16)
    wavfile.write(output_dir / 'original.wav', sample_rate, original_int16)
    
    print(f"\n✓ Done! Outputs in: {output_dir}")
    
    return result


def print_summary(result: TwoPassTranscription, max_slices: int = 20) -> None:
    print("\n" + "=" * 90)
    print("TWO-PASS TRANSCRIPTION SUMMARY")
    print("=" * 90)
    
    total_melody = sum(len(s.melody_notes) for s in result.slices)
    total_chord = sum(len(s.chord_notes) for s in result.slices)
    total_bass = sum(len(s.bass_notes) for s in result.slices)
    
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Sample rate: {result.sample_rate}, Hop: {result.hop_size}, Window: {result.window_size}")
    print(f"Melody: {total_melody}, Chords: {total_chord}, Bass: {total_bass}")
    
    print("\n" + "-" * 90)
    print(f"{'Time':<8} {'Bass':<15} {'Chords':<30} {'Melody'}")
    print("-" * 90)
    
    for sl in result.slices[:max_slices]:
        bass = " ".join(n.note_name for n in sl.bass_notes) or "-"
        chords = " ".join(n.note_name for n in sl.chord_notes) or "-"
        melody = " ".join(n.note_name for n in sl.melody_notes) or "-"
        print(f"{sl.time:<8.2f} {bass:<15} {chords:<30} {melody}")
    
    if len(result.slices) > max_slices:
        print(f"... and {len(result.slices) - max_slices} more slices")
    print("=" * 90)


def export_txt(result: TwoPassTranscription, output_path: Path) -> None:
    with open(output_path, 'w') as f:
        f.write(f"# Two-Pass Transcription: {result.source_file}\n")
        f.write(f"# Duration: {result.duration_seconds:.2f}s\n")
        f.write(f"# Sample rate: {result.sample_rate}, Hop: {result.hop_size}, Window: {result.window_size}\n\n")
        
        f.write(f"{'Time':<10} {'Bass':<15} {'Chords':<35} {'Melody'}\n")
        f.write("-" * 80 + "\n")
        
        for sl in result.slices:
            bass = " ".join(n.note_name for n in sl.bass_notes) or "-"
            chords = " ".join(n.note_name for n in sl.chord_notes) or "-"
            melody = " ".join(n.note_name for n in sl.melody_notes) or "-"
            f.write(f"{sl.time:<10.3f} {bass:<15} {chords:<35} {melody}\n")
    
    print(f"Saved: {output_path}")

def transcribe_two_pass_with_chords(
    filepath: str | Path,
    output_dir: str | Path,
    # ... all existing parameters ...
    **kwargs
) -> TwoPassTranscription:
    """
    Two-pass transcription with chord analysis.
    """
    # Run normal transcription
    result = transcribe_two_pass(filepath, output_dir, **kwargs)
    
    # Export with chord analysis
    output_dir = Path(output_dir)
    export_transcription_with_chords(
        result, 
        output_dir / 'transcription_with_chords.txt',
        include_melody_in_chord=False
    )
    
    # Print summary with chords
    print_transcription_with_chords(result, max_slices=30)
    
    return result


# Updated export function
def export_txt_with_chords(result: TwoPassTranscription, output_path: Path) -> None:
    """Export transcription with chord column."""
    with open(output_path, 'w') as f:
        f.write(f"# Two-Pass Transcription: {result.source_file}\n")
        f.write(f"# Duration: {result.duration_seconds:.2f}s\n\n")
        
        f.write(f"{'Time':<8} {'Chord':<12} {'Bass':<12} {'Chords':<28} {'Melody'}\n")
        f.write("-" * 90 + "\n")
        
        for sl in result.slices:
            chord = analyze_slice_chord(sl.bass_notes, sl.chord_notes)
            
            chord_str = format_chord_for_display(chord)
            bass = " ".join(n.note_name for n in sl.bass_notes) or "-"
            chords = " ".join(n.note_name for n in sl.chord_notes) or "-"
            melody = " ".join(n.note_name for n in sl.melody_notes) or "-"
            
            f.write(f"{sl.time:<8.2f} {chord_str} {bass:<12} {chords:<28} {melody}\n")
    
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # Step 1: Create transcription
    print("=" * 60)
    print("STEP 1: Transcribe audio")
    print("=" * 60)
    
    result = transcribe_two_pass_with_chords(
        filepath="best_part.mp3",
        output_dir=Path("transcription_two_pass"),
        slice_interval=0.1,
        onset_threshold_rel=0.1,
        melody_min_freq=300.0,
        melody_max_freq=2000.0,
        melody_sustain_slices=3,
        chord_min_freq=130.0,
        chord_max_freq=500.0,
        chord_smoothing_window=3,
        bass_min_freq=50.0,
        bass_max_freq=200.0
    )
    
    # Step 2: Test JSON reconstruction
    print("\n" + "=" * 60)
    print("STEP 2: Reconstruct from JSON (verify serialization)")
    print("=" * 60)
    
    transcription, audio = reconstruct_from_json(
        json_path="transcription_two_pass/transcription.json",
        output_dir="transcription_two_pass"
    )
    
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print("  transcription.json              - Full structured data")
    print("  transcription.txt               - Human readable")
    print("  original.wav                    - Original audio")
    print("  reconstructed_combined.wav      - Direct reconstruction")
    print("  reconstructed_bass.wav          - Bass only")
    print("  reconstructed_chords.wav        - Chords only")
    print("  reconstructed_melody.wav        - Melody only")
    print("  reconstructed_from_json.wav     - From JSON (verify)")
    print("  reconstructed_from_json_bass.wav")
    print("  reconstructed_from_json_chords.wav")
    print("  reconstructed_from_json_melody.wav")
    print("\nCompare reconstructed_combined.wav vs reconstructed_from_json.wav")
    print("They should sound identical if JSON serialization is working!")


if __name__ == '__main__':
    main()