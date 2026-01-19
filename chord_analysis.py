"""
Chord Analysis Module - Improved Bass Recovery
Author: Andre Lim
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, find_peaks
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import ffmpeg


@dataclass
class AudioConfig:
    """Configuration for audio analysis."""
    sample_rate: int = 44100
    window_size: int = 8192      # Increased for better bass resolution (~5.4 Hz bins)
    hop_size: int = 2048         # 75% overlap
    window_type: str = 'hann'    # Hann often better for music
    amplitude_threshold: float = 0.01  # Relative threshold (0-1)
    
    # Frequency range of interest
    min_freq: float = 50.0       # Below E1, not much musical content
    max_freq: float = 4000.0     # Above this, mostly harmonics/noise
    
    # Peak detection
    num_peaks: int = 12          # More peaks to capture full chord
    harmonic_tolerance: float = 0.03  # 3% tolerance for harmonic detection


@dataclass 
class AnalysisResult:
    """Container for analysis results."""
    spectrogram: np.ndarray
    frequencies: np.ndarray
    times: np.ndarray
    sample_rate: int


@dataclass
class NoteDetection:
    """A detected note with its harmonics."""
    fundamental_freq: float
    fundamental_amp: float
    harmonics: list  # List of (freq, amp) tuples
    midi_note: int
    note_name: str


# =============================================================================
# Utility Functions
# =============================================================================

def freq_to_midi(freq: float) -> int:
    """Convert frequency to nearest MIDI note number."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_freq(midi: int) -> float:
    """Convert MIDI note to frequency."""
    return 440.0 * (2 ** ((midi - 69) / 12))


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"


def mp3_to_wav_ffmpeg(input_file: str, output_file: str) -> None:
    """Convert MP3 to WAV using ffmpeg."""
    (
        ffmpeg
        .input(input_file)
        .output(output_file, acodec='pcm_s16le', ar=44100, ac=1)  # Force mono, 16-bit
        .run(overwrite_output=True, quiet=True)
    )
    print(f"Converted {input_file} to {output_file}")


def load_audio(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Load audio file and normalize to [-1, 1]."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    sample_rate, data = wavfile.read(filepath)
    
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    elif data.dtype != np.float32 and data.dtype != np.float64:
        data = data.astype(np.float64)
    
    return data, sample_rate


# =============================================================================
# Improved Spectrogram with Zero-Padding
# =============================================================================

def compute_spectrogram(
    audio_data: np.ndarray,
    config: AudioConfig,
    zero_pad_factor: int = 2
) -> AnalysisResult:
    """
    Compute spectrogram with optional zero-padding for better frequency resolution.
    
    Zero-padding doesn't add information, but it interpolates between FFT bins,
    giving us more precise peak locations (useful for low frequencies).
    """
    window = get_window(config.window_type, config.window_size)
    
    # Zero-padding increases FFT size without changing window
    fft_size = config.window_size * zero_pad_factor
    
    num_frames = 1 + (len(audio_data) - config.window_size) // config.hop_size
    num_bins = fft_size // 2 + 1
    spectrogram = np.zeros((num_frames, num_bins))
    
    for i in range(num_frames):
        start = i * config.hop_size
        end = start + config.window_size
        
        frame = audio_data[start:end] * window
        
        # Zero-pad the frame
        padded_frame = np.zeros(fft_size)
        padded_frame[:config.window_size] = frame
        
        spectrum = rfft(padded_frame)
        spectrogram[i] = np.abs(spectrum)
    
    # Frequency axis now has finer resolution due to zero-padding
    frequencies = rfftfreq(fft_size, 1.0 / config.sample_rate)
    times = np.arange(num_frames) * config.hop_size / config.sample_rate
    
    return AnalysisResult(
        spectrogram=spectrogram,
        frequencies=frequencies,
        times=times,
        sample_rate=config.sample_rate
    )


# =============================================================================
# Harmonic-Aware Peak Detection
# =============================================================================

def is_harmonic_of(freq: float, fundamental: float, tolerance: float = 0.03) -> Optional[int]:
    """
    Check if freq is a harmonic of fundamental.
    
    Returns harmonic number (2, 3, 4...) if it is, None otherwise.
    Tolerance is relative (e.g., 0.03 = 3%).
    """
    if fundamental <= 0 or freq <= fundamental:
        return None
    
    ratio = freq / fundamental
    nearest_harmonic = round(ratio)
    
    if nearest_harmonic < 2:
        return None
    
    # Check if ratio is close to an integer
    if abs(ratio - nearest_harmonic) / nearest_harmonic < tolerance:
        return int(nearest_harmonic)
    
    return None


def extract_notes_with_harmonics(
    result: AnalysisResult,
    config: AudioConfig
) -> list[list[NoteDetection]]:
    """
    Extract notes by identifying fundamentals and grouping their harmonics.
    
    Strategy:
    1. Find all significant peaks in the spectrum
    2. For each peak (starting from lowest), check if higher peaks are harmonics
    3. If a peak is a harmonic of an existing fundamental, group it
    4. Otherwise, consider it a new fundamental
    
    Returns:
        List of frames, each containing list of NoteDetection objects
    """
    spectrogram = result.spectrogram
    frequencies = result.frequencies
    num_frames = spectrogram.shape[0]
    
    # Find valid frequency range indices
    min_idx = np.searchsorted(frequencies, config.min_freq)
    max_idx = np.searchsorted(frequencies, config.max_freq)
    
    all_frame_notes = []
    
    for frame_idx in range(num_frames):
        frame = spectrogram[frame_idx]
        frame_max = frame.max()
        
        if frame_max < 1e-10:
            all_frame_notes.append([])
            continue
        
        # Normalize frame
        frame_norm = frame / frame_max
        
        # Find peaks using scipy (more robust than argpartition)
        peak_indices, peak_props = find_peaks(
            frame_norm[min_idx:max_idx],
            height=config.amplitude_threshold,
            distance=3,  # Minimum distance between peaks in bins
            prominence=0.01
        )
        
        # Adjust indices back to full spectrum
        peak_indices = peak_indices + min_idx
        
        if len(peak_indices) == 0:
            all_frame_notes.append([])
            continue
        
        # Sort peaks by frequency (low to high) for harmonic analysis
        peak_freqs = frequencies[peak_indices]
        peak_amps = frame[peak_indices]
        sorted_order = np.argsort(peak_freqs)
        peak_freqs = peak_freqs[sorted_order]
        peak_amps = peak_amps[sorted_order]
        
        # Group into fundamentals and harmonics
        notes = []
        used_peaks = set()
        
        for i, (freq, amp) in enumerate(zip(peak_freqs, peak_amps)):
            if i in used_peaks:
                continue
            
            # Check if this peak is a harmonic of an existing note
            is_harmonic = False
            for note in notes:
                harmonic_num = is_harmonic_of(freq, note.fundamental_freq, config.harmonic_tolerance)
                if harmonic_num:
                    note.harmonics.append((freq, amp, harmonic_num))
                    used_peaks.add(i)
                    is_harmonic = True
                    break
            
            if is_harmonic:
                continue
            
            # This is a new fundamental
            midi = freq_to_midi(freq)
            note = NoteDetection(
                fundamental_freq=freq,
                fundamental_amp=amp,
                harmonics=[],
                midi_note=midi,
                note_name=midi_to_note_name(midi)
            )
            
            # Look for harmonics of this new fundamental
            for j in range(i + 1, len(peak_freqs)):
                if j in used_peaks:
                    continue
                harmonic_num = is_harmonic_of(peak_freqs[j], freq, config.harmonic_tolerance)
                if harmonic_num:
                    note.harmonics.append((peak_freqs[j], peak_amps[j], harmonic_num))
                    used_peaks.add(j)
            
            notes.append(note)
            used_peaks.add(i)
            
            # Limit number of notes per frame
            if len(notes) >= config.num_peaks:
                break
        
        all_frame_notes.append(notes)
    
    return all_frame_notes


def extract_dominant_frequencies_improved(
    result: AnalysisResult,
    config: AudioConfig,
    notes_per_frame: list[list[NoteDetection]]
) -> np.ndarray:
    """
    Convert note detections to frequency/amplitude array.
    
    Prioritizes:
    1. Notes with more detected harmonics (more confident detection)
    2. Higher amplitude fundamentals
    """
    num_frames = len(notes_per_frame)
    freq_amp = np.zeros((num_frames, config.num_peaks, 2))
    
    for i, notes in enumerate(notes_per_frame):
        # Score notes by: harmonic count * 2 + amplitude
        scored_notes = []
        for note in notes:
            score = len(note.harmonics) * 2 + note.fundamental_amp
            scored_notes.append((score, note))
        
        # Sort by score descending
        scored_notes.sort(key=lambda x: x[0], reverse=True)
        
        for j, (score, note) in enumerate(scored_notes[:config.num_peaks]):
            freq_amp[i, j, 0] = note.fundamental_freq
            freq_amp[i, j, 1] = note.fundamental_amp
    
    return freq_amp


# =============================================================================
# Improved Audio Reconstruction
# =============================================================================

def reconstruct_audio_with_harmonics(
    notes_per_frame: list[list[NoteDetection]],
    config: AudioConfig
) -> np.ndarray:
    """
    Reconstruct audio using detected fundamentals AND their harmonics.
    
    This produces much more natural-sounding output because real instruments
    have rich harmonic content.
    """
    num_frames = len(notes_per_frame)
    output_length = num_frames * config.hop_size + config.window_size
    reconstructed = np.zeros(output_length)
    
    # Create a window for overlap-add synthesis
    synth_window = get_window('hann', config.window_size)
    
    for i, notes in enumerate(notes_per_frame):
        start = i * config.hop_size
        t = np.arange(config.window_size) / config.sample_rate
        
        frame_signal = np.zeros(config.window_size)
        
        for note in notes[:6]:  # Limit to 6 notes per frame
            # Add fundamental
            amp = note.fundamental_amp
            freq = note.fundamental_freq
            frame_signal += amp * np.sin(2 * np.pi * freq * t)
            
            # Add harmonics with their detected amplitudes
            for harm_freq, harm_amp, harm_num in note.harmonics:
                frame_signal += harm_amp * np.sin(2 * np.pi * harm_freq * t)
        
        # Apply window and overlap-add
        reconstructed[start:start + config.window_size] += frame_signal * synth_window
    
    # Normalize
    max_val = np.abs(reconstructed).max()
    if max_val > 0:
        reconstructed = reconstructed / max_val * 0.9 * 32767
    
    return reconstructed.astype(np.int16)


def reconstruct_audio_simple(
    freq_amp: np.ndarray,
    config: AudioConfig
) -> np.ndarray:
    """
    Simple multi-note reconstruction (uses all detected fundamentals).
    """
    num_frames = freq_amp.shape[0]
    output_length = num_frames * config.hop_size + config.window_size
    reconstructed = np.zeros(output_length)
    
    synth_window = get_window('hann', config.window_size)
    
    for i in range(num_frames):
        start = i * config.hop_size
        t = np.arange(config.window_size) / config.sample_rate
        
        frame_signal = np.zeros(config.window_size)
        
        for j in range(freq_amp.shape[1]):
            freq = freq_amp[i, j, 0]
            amp = freq_amp[i, j, 1]
            
            if freq > 0 and amp > 0:
                # Add fundamental and first few harmonics for richer sound
                frame_signal += amp * np.sin(2 * np.pi * freq * t)
                frame_signal += amp * 0.5 * np.sin(2 * np.pi * freq * 2 * t)  # 2nd harmonic
                frame_signal += amp * 0.25 * np.sin(2 * np.pi * freq * 3 * t)  # 3rd harmonic
        
        reconstructed[start:start + config.window_size] += frame_signal * synth_window
    
    max_val = np.abs(reconstructed).max()
    if max_val > 0:
        reconstructed = reconstructed / max_val * 0.9 * 32767
    
    return reconstructed.astype(np.int16)


# =============================================================================
# Visualization
# =============================================================================

def plot_spectrogram_with_notes(
    result: AnalysisResult,
    notes_per_frame: list[list[NoteDetection]],
    output_path: str | Path,
    max_freq: float = 2000.0
) -> None:
    """Plot spectrogram with detected notes overlaid."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Limit frequency range for visibility
    max_freq_idx = np.searchsorted(result.frequencies, max_freq)
    
    spec_db = 20 * np.log10(result.spectrogram[:, :max_freq_idx].T + 1e-10)
    
    ax.imshow(
        spec_db,
        origin='lower',
        aspect='auto',
        extent=[
            result.times[0], result.times[-1],
            result.frequencies[0], result.frequencies[max_freq_idx]
        ],
        cmap='viridis'
    )
    
    # Overlay detected fundamentals
    for frame_idx, notes in enumerate(notes_per_frame):
        time = result.times[frame_idx] if frame_idx < len(result.times) else result.times[-1]
        for note in notes:
            if note.fundamental_freq < max_freq:
                ax.scatter(time, note.fundamental_freq, c='red', s=10, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram with Detected Notes (red dots = fundamentals)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_piano_roll(
    notes_per_frame: list[list[NoteDetection]],
    times: np.ndarray,
    output_path: str | Path
) -> None:
    """Create a piano roll visualization of detected notes."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Collect all (time, midi, amplitude) points
    points = []
    for frame_idx, notes in enumerate(notes_per_frame):
        if frame_idx >= len(times):
            break
        time = times[frame_idx]
        for note in notes:
            points.append((time, note.midi_note, note.fundamental_amp))
    
    if not points:
        print("No notes detected for piano roll")
        return
    
    points = np.array(points)
    
    # Normalize amplitudes for color
    amps = points[:, 2]
    amps_norm = (amps - amps.min()) / (amps.max() - amps.min() + 1e-10)
    
    scatter = ax.scatter(
        points[:, 0], points[:, 1],
        c=amps_norm, cmap='YlOrRd',
        s=20, alpha=0.7
    )
    
    # Y-axis: show note names
    midi_min = int(points[:, 1].min()) - 1
    midi_max = int(points[:, 1].max()) + 1
    
    tick_positions = list(range(midi_min, midi_max + 1, 2))
    tick_labels = [midi_to_note_name(m) for m in tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Note')
    ax.set_title('Piano Roll - Detected Notes')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, label='Relative Amplitude')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_detected_notes_summary(
    notes_per_frame: list[list[NoteDetection]],
    times: np.ndarray,
    sample_interval: int = 20
) -> None:
    """Print a summary of detected notes at regular intervals."""
    print("\n" + "=" * 60)
    print("DETECTED NOTES SUMMARY")
    print("=" * 60)
    
    for i in range(0, len(notes_per_frame), sample_interval):
        if i >= len(times):
            break
        time = times[i]
        notes = notes_per_frame[i]
        
        if notes:
            note_strs = [f"{n.note_name}({n.fundamental_freq:.1f}Hz, {len(n.harmonics)}h)" 
                        for n in notes[:6]]
            print(f"  {time:6.2f}s: {', '.join(note_strs)}")
    
    print("=" * 60)


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Main analysis pipeline with improved bass detection."""
    
    # Configuration optimized for full-range detection
    config = AudioConfig(
        window_size=8192,       # Larger window = ~5.4 Hz bins (better bass)
        hop_size=2048,          # 75% overlap
        amplitude_threshold=0.02,
        num_peaks=12,
        min_freq=60.0,          # Capture bass notes
        max_freq=3000.0
    )
    
    # Convert and load
    mp3_to_wav_ffmpeg("best_part.mp3", "best_part.wav")
    audio_data, sample_rate = load_audio("best_part.wav")
    config.sample_rate = sample_rate
    
    print(f"Loaded audio: {len(audio_data)/sample_rate:.2f}s at {sample_rate}Hz")
    print(f"FFT bin resolution: {sample_rate/config.window_size:.2f} Hz")
    
    # Compute spectrogram with zero-padding for better peak localization
    print("Computing spectrogram...")
    result = compute_spectrogram(audio_data, config, zero_pad_factor=2)
    
    # Extract notes with harmonic grouping
    print("Detecting notes with harmonic analysis...")
    notes_per_frame = extract_notes_with_harmonics(result, config)
    
    # Print summary
    print_detected_notes_summary(notes_per_frame, result.times, sample_interval=30)
    
    # Convert to frequency array for compatibility
    freq_amp = extract_dominant_frequencies_improved(result, config, notes_per_frame)
    
    # Save frequency data
    num_frames = freq_amp.shape[0]
    flat_data = freq_amp.reshape(num_frames, -1)
    np.savetxt('freq_amp_improved.csv', flat_data, fmt='%.6g', delimiter=',')
    
    # Reconstruct with harmonics
    print("Reconstructing audio...")
    reconstructed = reconstruct_audio_with_harmonics(notes_per_frame, config)
    wavfile.write('reconstructed_improved.wav', config.sample_rate, reconstructed)
    
    # Also create simpler reconstruction for comparison
    reconstructed_simple = reconstruct_audio_simple(freq_amp, config)
    wavfile.write('reconstructed_simple.wav', config.sample_rate, reconstructed_simple)
    
    # # Visualizations ( commented out as they take too long )
    # print("Creating visualizations...")
    # plot_spectrogram_with_notes(result, notes_per_frame, 'spectrogram_notes.png')
    # plot_piano_roll(notes_per_frame, result.times, 'piano_roll.png')
    
    print("\nDone! Created:")
    print("  - reconstructed_improved.wav (with detected harmonics)")
    print("  - reconstructed_simple.wav (fundamentals + synthetic harmonics)")
    print("  - spectrogram_notes.png")
    print("  - piano_roll.png")


if __name__ == '__main__':
    main()