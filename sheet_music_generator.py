"""
Sheet Music Generator from Transcription
Author: Andre Lim

Converts transcription.txt to musical notation with:
- Treble clef (melody + upper chords)
- Bass clef (bass + lower chords)
- Automatic BPM and time signature detection from file
- Bar/beat alignment

Outputs: MusicXML file (viewable in MuseScore, Finale, etc.)
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import music21
from music21 import stream, note, chord, meter, clef, key, tempo, metadata


# =============================================================================
# Note Parsing
# =============================================================================

NOTE_PATTERN = re.compile(r'^([A-G])([#b]?)(-?\d+)$')

def parse_note_name(note_str: str) -> Optional[tuple[str, int]]:
    """
    Parse note name like 'C#4' or 'Bb3' to (pitch_name, octave).
    
    Returns (pitch_name, octave) or None if invalid.
    """
    note_str = note_str.strip()
    match = NOTE_PATTERN.match(note_str)
    
    if not match:
        return None
    
    letter = match.group(1)
    accidental = match.group(2)
    octave = int(match.group(3))
    
    # Convert to music21 format
    if accidental == '#':
        pitch_name = f"{letter}#"
    elif accidental == 'b':
        pitch_name = f"{letter}-"  # music21 uses - for flat
    else:
        pitch_name = letter
    
    return (pitch_name, octave)


def note_to_midi(note_str: str) -> int:
    """Convert note name to MIDI number for sorting."""
    parsed = parse_note_name(note_str)
    if not parsed:
        return 0
    
    pitch_name, octave = parsed
    
    # Base values for each note
    base = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    letter = pitch_name[0]
    midi = (octave + 1) * 12 + base.get(letter, 0)
    
    if '#' in pitch_name:
        midi += 1
    elif '-' in pitch_name:
        midi -= 1
    
    return midi


# =============================================================================
# Transcription Parsing (Extended for Bar/Beat format)
# =============================================================================

@dataclass
class TimeSliceData:
    time: float
    bar: int = 1
    beat: int = 1
    beat_fraction: float = 0.0
    chord_name: str = "-"
    bass_notes: list[str] = field(default_factory=list)
    chord_notes: list[str] = field(default_factory=list)
    melody_notes: list[str] = field(default_factory=list)


@dataclass
class TranscriptionMetadata:
    """Metadata extracted from transcription file header."""
    source_file: str = ""
    duration: float = 0.0
    bpm: float = 120.0
    time_signature: str = "4/4"
    total_bars: int = 0
    first_downbeat: float = 0.0
    bpm_confidence: float = 0.0


def parse_transcription_header(lines: list[str]) -> TranscriptionMetadata:
    """Parse header comments to extract metadata."""
    meta = TranscriptionMetadata()
    
    for line in lines:
        line = line.strip()
        if not line.startswith('#'):
            break
        
        # Remove # and parse key: value
        content = line[1:].strip()
        
        if content.startswith('Transcription:'):
            meta.source_file = content.split(':', 1)[1].strip()
        elif content.startswith('Duration:'):
            try:
                meta.duration = float(content.split(':')[1].strip().replace('s', ''))
            except:
                pass
        elif content.startswith('BPM:'):
            try:
                meta.bpm = float(content.split(':')[1].strip())
            except:
                pass
        elif content.startswith('Time Signature:'):
            meta.time_signature = content.split(':')[1].strip()
        elif content.startswith('Total Bars:'):
            try:
                meta.total_bars = int(content.split(':')[1].strip())
            except:
                pass
        elif content.startswith('First Downbeat:'):
            try:
                meta.first_downbeat = float(content.split(':')[1].strip().replace('s', ''))
            except:
                pass
        elif content.startswith('BPM Confidence:'):
            try:
                meta.bpm_confidence = float(content.split(':')[1].strip())
            except:
                pass
    
    return meta


def parse_transcription_file_v2(filepath: str | Path) -> tuple[list[TimeSliceData], TranscriptionMetadata]:
    """
    Parse transcription.txt with new format including Bar.Beat column.
    
    New format:
    Bar.Beat   Time     Chord      Bass         Chords                       Melody
    
    Returns (slices, metadata)
    """
    slices = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header metadata
    meta = parse_transcription_header(lines)
    
    # Find column positions from header
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # New format has Bar.Beat column
        if 'Bar.Beat' in stripped or (stripped.startswith('Time') and 'Chord' in stripped):
            header_idx = i
            break
    
    if header_idx is None:
        print("Could not find header line")
        return slices, meta
    
    header = lines[header_idx]
    
    # Detect format (old vs new)
    has_bar_beat = 'Bar.Beat' in header
    
    if has_bar_beat:
        # New format: Bar.Beat   Time     Chord      Bass         Chords       Melody
        bar_beat_col = header.find('Bar.Beat')
        time_col = header.find('Time')
        chord_col = header.find('Chord', time_col + 1) if time_col >= 0 else header.find('Chord')
        bass_col = header.find('Bass')
        chords_col = header.find('Chords')
        melody_col = header.find('Melody')
    else:
        # Old format: Time     Chord        Bass         Chords       Melody
        bar_beat_col = -1
        time_col = header.find('Time')
        chord_col = header.find('Chord')
        bass_col = header.find('Bass')
        chords_col = header.find('Chords')
        melody_col = header.find('Melody')
    
    # Parse data lines
    for line in lines[header_idx + 2:]:  # Skip header and separator
        if not line.strip() or line.strip().startswith('#'):
            continue
        
        # Pad line to ensure we can slice it
        line = line.ljust(melody_col + 100)
        
        try:
            if has_bar_beat:
                bar_beat_str = line[bar_beat_col:time_col].strip()
                time_str = line[time_col:chord_col].strip()
                
                # Parse bar.beat (e.g., "1.2" or "1.2.50")
                bar_parts = bar_beat_str.split('.')
                bar = int(bar_parts[0]) if bar_parts else 1
                beat = int(bar_parts[1]) if len(bar_parts) > 1 else 1
                beat_fraction = int(bar_parts[2]) / 100.0 if len(bar_parts) > 2 else 0.0
            else:
                time_str = line[time_col:chord_col].strip()
                bar, beat, beat_fraction = 1, 1, 0.0
            
            time = float(time_str)
        except (ValueError, IndexError):
            continue
        
        chord_name = line[chord_col:bass_col].strip() or "-"
        bass_str = line[bass_col:chords_col].strip()
        chords_str = line[chords_col:melody_col].strip()
        melody_str = line[melody_col:].strip()
        
        def parse_notes(s):
            if s == '-' or not s:
                return []
            return [n for n in s.split() if n != '-' and parse_note_name(n)]
        
        slices.append(TimeSliceData(
            time=time,
            bar=bar,
            beat=beat,
            beat_fraction=beat_fraction,
            chord_name=chord_name,
            bass_notes=parse_notes(bass_str),
            chord_notes=parse_notes(chords_str),
            melody_notes=parse_notes(melody_str)
        ))
    
    return slices, meta


# Legacy function for backward compatibility
def parse_transcription_file_columnar(filepath: str | Path) -> list[TimeSliceData]:
    """Parse transcription.txt (legacy wrapper)."""
    slices, _ = parse_transcription_file_v2(filepath)
    return slices


def parse_transcription_file(filepath: str | Path) -> list[TimeSliceData]:
    """Parse transcription.txt file (legacy)."""
    slices, _ = parse_transcription_file_v2(filepath)
    return slices


# =============================================================================
# Quantization
# =============================================================================

def quantize_to_beats(
    slices: list[TimeSliceData],
    bpm: float = 120.0,
    beats_per_bar: int = 4,
    quantize_resolution: float = 0.25  # Quarter note = 1, eighth = 0.5, sixteenth = 0.25
) -> list[dict]:
    """
    Quantize time slices to beat positions.
    
    Returns list of dicts with:
    - beat: beat number (0-indexed from start)
    - bar: bar number (0-indexed)
    - beat_in_bar: beat within bar (0-3 for 4/4)
    - bass_notes, chord_notes, melody_notes
    """
    seconds_per_beat = 60.0 / bpm
    
    # Group slices by quantized beat
    beat_data = defaultdict(lambda: {
        'bass_notes': set(),
        'chord_notes': set(),
        'melody_notes': set(),
        'chord_names': []
    })
    
    for sl in slices:
        # Convert time to beat
        beat_float = sl.time / seconds_per_beat
        
        # Quantize to nearest resolution
        beat_quantized = round(beat_float / quantize_resolution) * quantize_resolution
        
        # Add notes
        beat_data[beat_quantized]['bass_notes'].update(sl.bass_notes)
        beat_data[beat_quantized]['chord_notes'].update(sl.chord_notes)
        beat_data[beat_quantized]['melody_notes'].update(sl.melody_notes)
        if sl.chord_name and sl.chord_name != '-':
            beat_data[beat_quantized]['chord_names'].append(sl.chord_name)
    
    # Convert to list
    result = []
    for beat in sorted(beat_data.keys()):
        data = beat_data[beat]
        
        bar = int(beat // beats_per_bar)
        beat_in_bar = beat % beats_per_bar
        
        result.append({
            'beat': beat,
            'bar': bar,
            'beat_in_bar': beat_in_bar,
            'bass_notes': sorted(data['bass_notes'], key=note_to_midi),
            'chord_notes': sorted(data['chord_notes'], key=note_to_midi),
            'melody_notes': sorted(data['melody_notes'], key=note_to_midi),
            'chord_name': data['chord_names'][0] if data['chord_names'] else None
        })
    
    return result


# =============================================================================
# Sheet Music Generation
# =============================================================================

def create_music21_note(note_str: str, duration_quarters: float = 1.0) -> Optional[note.Note]:
    """Create a music21 Note from note string."""
    parsed = parse_note_name(note_str)
    if not parsed:
        return None
    
    pitch_name, octave = parsed
    
    n = note.Note()
    n.pitch.name = pitch_name
    n.pitch.octave = octave
    n.duration.quarterLength = duration_quarters
    
    return n


def create_music21_chord(note_strs: list[str], duration_quarters: float = 1.0) -> Optional[chord.Chord]:
    """Create a music21 Chord from note strings."""
    if not note_strs:
        return None
    
    pitches = []
    for note_str in note_strs:
        parsed = parse_note_name(note_str)
        if parsed:
            pitch_name, octave = parsed
            p = music21.pitch.Pitch()
            p.name = pitch_name
            p.octave = octave
            pitches.append(p)
    
    if not pitches:
        return None
    
    c = chord.Chord(pitches)
    c.duration.quarterLength = duration_quarters
    
    return c


def generate_sheet_music(
    quantized_beats: list[dict],
    title: str = "Transcription",
    composer: str = "Auto-generated",
    bpm: float = 120.0,
    time_signature: str = "4/4",
    key_signature: str = "C"
) -> stream.Score:
    """
    Generate sheet music from quantized beat data.
    
    Creates a Score with:
    - Treble clef part (melody + high chords)
    - Bass clef part (bass + low chords)
    """
    # Create score
    score = stream.Score()
    
    # Add metadata
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = composer
    
    # Create parts
    treble_part = stream.Part()
    treble_part.id = 'Treble'
    treble_part.insert(0, clef.TrebleClef())
    treble_part.insert(0, meter.TimeSignature(time_signature))
    treble_part.insert(0, tempo.MetronomeMark(number=bpm))
    
    bass_part = stream.Part()
    bass_part.id = 'Bass'
    bass_part.insert(0, clef.BassClef())
    bass_part.insert(0, meter.TimeSignature(time_signature))
    
    # Process beats
    if not quantized_beats:
        return score
    
    # Determine note durations by looking at gaps between beats
    beat_positions = [b['beat'] for b in quantized_beats]
    
    for i, beat_data in enumerate(quantized_beats):
        beat_pos = beat_data['beat']
        
        # Calculate duration to next beat (or default to quarter note)
        if i + 1 < len(quantized_beats):
            duration = quantized_beats[i + 1]['beat'] - beat_pos
        else:
            duration = 1.0  # Default quarter note
        
        # Clamp duration to reasonable values
        duration = max(0.25, min(duration, 4.0))
        
        # Treble clef: melody + high chord notes
        treble_notes = []
        
        # Add melody notes
        for note_str in beat_data['melody_notes']:
            treble_notes.append(note_str)
        
        # Add chord notes that are in treble range (>= C4, MIDI 60)
        for note_str in beat_data['chord_notes']:
            if note_to_midi(note_str) >= 60:  # C4 and above
                treble_notes.append(note_str)
        
        # Remove duplicates and sort
        treble_notes = sorted(set(treble_notes), key=note_to_midi)
        
        # Create treble element
        if len(treble_notes) == 0:
            treble_elem = note.Rest()
            treble_elem.duration.quarterLength = duration
        elif len(treble_notes) == 1:
            treble_elem = create_music21_note(treble_notes[0], duration)
        else:
            treble_elem = create_music21_chord(treble_notes, duration)
        
        if treble_elem:
            treble_part.insert(beat_pos, treble_elem)
        
        # Bass clef: bass + low chord notes
        bass_notes = []
        
        # Add bass notes
        for note_str in beat_data['bass_notes']:
            bass_notes.append(note_str)
        
        # Add chord notes that are in bass range (< C4)
        for note_str in beat_data['chord_notes']:
            if note_to_midi(note_str) < 60:
                bass_notes.append(note_str)
        
        # Remove duplicates and sort
        bass_notes = sorted(set(bass_notes), key=note_to_midi)
        
        # Create bass element
        if len(bass_notes) == 0:
            bass_elem = note.Rest()
            bass_elem.duration.quarterLength = duration
        elif len(bass_notes) == 1:
            bass_elem = create_music21_note(bass_notes[0], duration)
        else:
            bass_elem = create_music21_chord(bass_notes, duration)
        
        if bass_elem:
            bass_part.insert(beat_pos, bass_elem)
    
    # Add parts to score
    score.insert(0, treble_part)
    score.insert(0, bass_part)
    
    return score


# =============================================================================
# Main Pipeline
# =============================================================================

def transcription_to_sheet_music(
    input_file: str | Path,
    output_dir: str | Path,
    bpm: float = 120.0,
    time_signature: str = "4/4",
    max_bars: int = None,  # Limit output length
    title: str = "Transcription"
) -> Path:
    """
    Convert transcription.txt to sheet music.
    
    Args:
        input_file: Path to transcription.txt
        output_dir: Output directory
        bpm: Tempo in beats per minute
        time_signature: Time signature (e.g., "4/4", "3/4")
        max_bars: Maximum number of bars to include (None = all)
        title: Title for the sheet music
    
    Returns:
        Path to output MusicXML file
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading transcription: {input_file}")
    
    # Parse transcription
    slices = parse_transcription_file_columnar(input_file)
    print(f"  Parsed {len(slices)} time slices")
    
    if not slices:
        print("  ERROR: No data found!")
        return None
    
    # Get duration
    duration = slices[-1].time if slices else 0
    print(f"  Duration: {duration:.2f}s")
    
    # Quantize to beats
    print(f"\nQuantizing to beats (BPM={bpm}, {time_signature})...")
    
    beats_per_bar = int(time_signature.split('/')[0])
    quantized = quantize_to_beats(slices, bpm=bpm, beats_per_bar=beats_per_bar)
    print(f"  Quantized to {len(quantized)} beats")
    
    # Limit bars if specified
    if max_bars:
        max_beat = max_bars * beats_per_bar
        quantized = [b for b in quantized if b['beat'] < max_beat]
        print(f"  Limited to {max_bars} bars ({len(quantized)} beats)")
    
    num_bars = (quantized[-1]['beat'] // beats_per_bar + 1) if quantized else 0
    print(f"  Total bars: {num_bars}")
    
    # Generate sheet music
    print("\nGenerating sheet music...")
    score = generate_sheet_music(
        quantized,
        title=title,
        bpm=bpm,
        time_signature=time_signature
    )
    
    # Save as MusicXML
    output_path = output_dir / 'sheet_music.musicxml'
    score.write('musicxml', fp=str(output_path))
    print(f"  Saved: {output_path}")
    
    # Also save as MIDI for playback
    midi_path = output_dir / 'sheet_music.mid'
    score.write('midi', fp=str(midi_path))
    print(f"  Saved: {midi_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input: {input_file}")
    print(f"  BPM: {bpm}")
    print(f"  Time signature: {time_signature}")
    print(f"  Total bars: {num_bars}")
    print(f"  Output: {output_path}")
    print("=" * 60)
    
    # Print first few bars for preview
    print("\nPREVIEW (first 8 beats):")
    print("-" * 60)
    print(f"{'Beat':<8} {'Bar.Beat':<10} {'Treble':<25} {'Bass'}")
    print("-" * 60)
    
    for b in quantized[:8]:
        bar_beat = f"{b['bar']+1}.{b['beat_in_bar']+1}"
        
        # Treble notes
        treble = b['melody_notes'] + [n for n in b['chord_notes'] if note_to_midi(n) >= 60]
        treble_str = " ".join(treble) if treble else "-"
        
        # Bass notes
        bass = b['bass_notes'] + [n for n in b['chord_notes'] if note_to_midi(n) < 60]
        bass_str = " ".join(bass) if bass else "-"
        
        print(f"{b['beat']:<8.2f} {bar_beat:<10} {treble_str:<25} {bass_str}")
    
    print("-" * 60)
    
    return output_path


# =============================================================================
# Sheet Music from Fused Transcription (with proper note durations)
# =============================================================================

def fused_to_sheet_music(
    fused_json_path: str | Path,
    output_dir: str | Path,
    title: str = "Transcription",
    max_bars: int = None
) -> Path:
    """
    Generate sheet music from fused transcription JSON.
    
    This uses the proper note durations from the fusion process,
    resulting in more accurate sheet music.
    
    Args:
        fused_json_path: Path to transcription_fused.json
        output_dir: Output directory
        title: Title for the sheet music
        max_bars: Maximum bars to include (None = all)
    
    Returns:
        Path to output MusicXML file
    """
    from note_event_fusion import FusedTranscription
    
    fused_json_path = Path(fused_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading fused transcription: {fused_json_path}")
    
    fused = FusedTranscription.from_json_file(fused_json_path)
    
    print(f"  BPM: {fused.bpm}")
    print(f"  Time Signature: {fused.time_signature}")
    print(f"  Total Bars: {fused.num_bars}")
    print(f"  Note Events: {len(fused.note_events)}")
    
    # Parse time signature
    ts_parts = fused.time_signature.split('/')
    ts_num = int(ts_parts[0])
    ts_denom = int(ts_parts[1])
    
    # Create score
    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    
    # Create parts
    treble_part = stream.Part()
    treble_part.id = 'Treble'
    treble_part.insert(0, clef.TrebleClef())
    treble_part.insert(0, meter.TimeSignature(fused.time_signature))
    treble_part.insert(0, tempo.MetronomeMark(number=fused.bpm))
    
    bass_part = stream.Part()
    bass_part.id = 'Bass'
    bass_part.insert(0, clef.BassClef())
    bass_part.insert(0, meter.TimeSignature(fused.time_signature))
    
    # Filter events by max_bars if specified
    events = fused.note_events
    if max_bars:
        events = [e for e in events if e.bar <= max_bars]
    
    # Add notes to appropriate parts
    for event in events:
        # Calculate position in beats from start
        # bar is 1-indexed, beat is 1-indexed
        beat_position = (event.bar - 1) * ts_num + (event.beat - 1)
        
        # Create note
        parsed = parse_note_name(event.note_name)
        if not parsed:
            continue
        
        pitch_name, octave = parsed
        
        n = note.Note()
        n.pitch.name = pitch_name
        n.pitch.octave = octave
        n.duration.quarterLength = event.duration_beats
        
        # Assign to treble or bass based on pitch
        midi = event.midi_note
        if midi >= 60 or event.role == 'melody':
            treble_part.insert(beat_position, n)
        else:
            bass_part.insert(beat_position, n)
    
    # Add parts to score
    score.insert(0, treble_part)
    score.insert(0, bass_part)
    
    # Save
    output_path = output_dir / 'sheet_music_fused.musicxml'
    score.write('musicxml', fp=str(output_path))
    print(f"  Saved: {output_path}")
    
    # midi_path = output_dir / 'sheet_music_fused.mid'
    # score.write('midi', fp=str(midi_path))
    # print(f"  Saved: {midi_path}")
    
    print(f"\n✓ Sheet music generated with proper note durations!")
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    output = transcription_to_sheet_music(
        input_file="transcription_output_bars_beats/transcription.txt",
        output_dir=Path("transcription_output_bars_beats"),
        bpm=67.0,
        time_signature="4/4",
        max_bars=32,  # First 32 bars for testing
        title="Best Part - Transcription"
    )
    
    fused_to_sheet_music(
        fused_json_path="transcription_output_bars_beats/transcription_fused.json",
        output_dir="transcription_output_bars_beats/",
        title="My Song"
    )
    if output:
        print(f"\n✓ Sheet music generated!")
        print(f"  Open {output} in MuseScore, Finale, or any MusicXML viewer")


if __name__ == '__main__':
    main()