import numpy as np
from scipy.io import wavfile
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

# =============================================================================
# Configuration
# =============================================================================
SAMPLE_RATE = 44100
SLICE_DURATION = 0.1  # We assume 0.1s based on your file structure
TRANSITION_SMOOTHING = 0.01  # 10ms crossfade to prevent clicking

@dataclass
class TextSlice:
    time: float
    bass_notes: List[str]
    chord_notes: List[str]
    melody_notes: List[str]

# =============================================================================
# Note Parsing Utilities
# =============================================================================

NOTE_OFFSETS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

def note_to_freq(note_name: str) -> float:
    """Converts 'C#4' -> 277.18"""
    if not note_name or note_name == '-':
        return 0.0
    
    # Regex to separate Note (C#) from Octave (4)
    match = re.match(r"([A-G][#b]?)(-?\d+)", note_name)
    if not match:
        return 0.0
    
    name, octave = match.groups()
    semitone = NOTE_OFFSETS.get(name, 0)
    octave_num = int(octave)
    
    midi_num = (octave_num + 1) * 12 + semitone
    freq = 440.0 * (2 ** ((midi_num - 69) / 12))
    return freq

def parse_txt_line(line: str) -> TextSlice:
    """
    Parses a fixed-width line from transcription.txt.
    Based on your header: Time(8) Chord(10) Bass(12) Chords(28) Melody
    """
    # Fixed width slicing based on visual inspection of your file
    # Time: 0-8, Chord: 9-19, Bass: 20-32, Chords: 33-61, Melody: 62+
    time_str = line[0:8].strip()
    # chord_name_str = line[9:19].strip() # We don't synthesize the label
    bass_str = line[20:32].strip()
    inner_chords_str = line[33:61].strip()
    melody_str = line[61:].strip()

    try:
        t = float(time_str)
    except ValueError:
        return None

    def split_notes(s):
        return [n for n in s.split(' ') if n and n != '-']

    return TextSlice(
        time=t,
        bass_notes=split_notes(bass_str),
        chord_notes=split_notes(inner_chords_str),
        melody_notes=split_notes(melody_str)
    )

def parse_transcription_txt(filepath: str) -> List[TextSlice]:
    slices = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    start_parsing = False
    for line in lines:
        if line.startswith('----'):
            start_parsing = True
            continue
        if not start_parsing:
            continue
        if not line.strip():
            continue
            
        sl = parse_txt_line(line)
        if sl:
            slices.append(sl)
    return slices

# =============================================================================
# Audio Synthesis
# =============================================================================

def generate_sine(freq: float, duration: float, volume: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Simple Envelope to prevent clicking
    envelope = np.ones_like(t)
    attack_rel = int(TRANSITION_SMOOTHING * SAMPLE_RATE)
    
    # Check if duration is long enough for full envelope
    if len(t) > 2 * attack_rel:
        envelope[:attack_rel] = np.linspace(0, 1, attack_rel)
        envelope[-attack_rel:] = np.linspace(1, 0, attack_rel)
    
    return volume * np.sin(2 * np.pi * freq * t) * envelope

def synthesize_track(notes_sequence: List[List[float]], total_len_samples: int) -> np.ndarray:
    """
    Synthesizes a track. 
    notes_sequence: List of [freq1, freq2] for each time slice.
    """
    track_audio = np.zeros(total_len_samples)
    samples_per_slice = int(SLICE_DURATION * SAMPLE_RATE)
    
    # We use a naive overlap-add approach for slices
    for i, freqs in enumerate(notes_sequence):
        start_sample = i * samples_per_slice
        
        # Determine volume per note (normalize so chords aren't super loud)
        count = len(freqs)
        vol = 0.5 / count if count > 0 else 0
        
        for f in freqs:
            if f > 0:
                # Generate audio for this slice
                audio_slice = generate_sine(f, SLICE_DURATION, vol)
                
                # Add to buffer (safe bounds check)
                end_sample = min(start_sample + len(audio_slice), total_len_samples)
                length = end_sample - start_sample
                track_audio[start_sample:end_sample] += audio_slice[:length]
                
    return track_audio

# =============================================================================
# Main
# =============================================================================

def txt_to_wav(txt_path: str, wav_path: str):
    print(f"Parsing {txt_path}...")
    slices = parse_transcription_txt(txt_path)
    
    if not slices:
        print("No data found or parsing failed.")
        return

    duration = slices[-1].time + SLICE_DURATION
    total_samples = int(duration * SAMPLE_RATE) + 44100 # buffer
    
    print(f"Synthesizing {len(slices)} slices ({duration:.2f}s)...")

    # 1. Prepare frequency lists
    bass_freqs = [[note_to_freq(n) for n in s.bass_notes] for s in slices]
    chord_freqs = [[note_to_freq(n) for n in s.chord_notes] for s in slices]
    melody_freqs = [[note_to_freq(n) for n in s.melody_notes] for s in slices]

    # 2. Synthesize Parts (Mixing levels)
    # Bass: Louder, Lower
    audio_bass = synthesize_track(bass_freqs, total_samples) * 0.6
    
    # Chords: Softer, Background
    audio_chords = synthesize_track(chord_freqs, total_samples) * 0.45
    
    # Melody: Prominent
    audio_melody = synthesize_track(melody_freqs, total_samples) * 0.1

    # 3. Mix
    mix = audio_bass + audio_chords + audio_melody
    
    # 4. Normalize
    max_val = np.abs(mix).max()
    if max_val > 0:
        mix = mix / max_val * 0.9  # Headroom
    
    # 5. Export
    wavfile.write(wav_path, SAMPLE_RATE, (mix * 32767).astype(np.int16))
    print(f"Saved litmus test audio: {wav_path}")

# Run
if __name__ == "__main__":
    txt_to_wav("transcription_two_pass/transcription.txt", "transcription_two_pass/reconstruct_from_transcription.wav")