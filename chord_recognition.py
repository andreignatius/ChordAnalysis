"""
Chord Recognition Module
Author: Andre Lim

Detects chord names from sets of notes.
Supports: major, minor, diminished, augmented, sus2, sus4,
dominant 7th, major 7th, minor 7th, and common extensions.
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Chord Templates
# =============================================================================

# Intervals from root (in semitones)
CHORD_TEMPLATES = {
    # Triads
    'maj':      [0, 4, 7],
    'min':      [0, 3, 7],
    'dim':      [0, 3, 6],
    'aug':      [0, 4, 8],
    'sus2':     [0, 2, 7],
    'sus4':     [0, 5, 7],
    
    # Seventh chords
    'maj7':     [0, 4, 7, 11],
    'min7':     [0, 3, 7, 10],
    '7':        [0, 4, 7, 10],      # Dominant 7th
    'dim7':     [0, 3, 6, 9],
    'min7b5':   [0, 3, 6, 10],      # Half-diminished
    'minMaj7':  [0, 3, 7, 11],
    'aug7':     [0, 4, 8, 10],
    
    # Extended chords (common voicings)
    'add9':     [0, 4, 7, 14],
    'madd9':    [0, 3, 7, 14],
    '9':        [0, 4, 7, 10, 14],
    'min9':     [0, 3, 7, 10, 14],
    'maj9':     [0, 4, 7, 11, 14],
    '11':       [0, 4, 7, 10, 14, 17],
    '13':       [0, 4, 7, 10, 14, 21],
    
    # Sixth chords
    '6':        [0, 4, 7, 9],
    'min6':     [0, 3, 7, 9],
    
    # Power chord
    '5':        [0, 7],
}

# Note names for display
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enharmonic equivalents for nicer display
ENHARMONIC_DISPLAY = {
    'C#': 'Db',
    'D#': 'Eb', 
    'F#': 'Gb',
    'G#': 'Ab',
    'A#': 'Bb'
}


# =============================================================================
# Chord Detection
# =============================================================================

@dataclass
class ChordMatch:
    """Result of chord matching."""
    root: str           # Root note name (e.g., "C", "F#")
    root_midi: int      # Root MIDI number
    quality: str        # Chord quality (e.g., "maj", "min7")
    name: str           # Full chord name (e.g., "Cmaj7", "F#min")
    matched_notes: int  # How many input notes matched
    total_notes: int    # Total input notes
    match_score: float  # 0-1, how well it matched
    missing_notes: list[int]  # Template notes not found
    extra_notes: list[int]    # Input notes not in template
    

def midi_to_pitch_class(midi: int) -> int:
    """Convert MIDI note to pitch class (0-11)."""
    return midi % 12


def pitch_class_to_name(pc: int) -> str:
    """Convert pitch class to note name."""
    return NOTE_NAMES[pc % 12]


def get_intervals_from_bass(midi_notes: list[int]) -> tuple[int, list[int]]:
    """
    Get intervals relative to the bass (lowest) note.
    
    Returns (bass_midi, intervals_in_semitones)
    """
    if not midi_notes:
        return 0, []
    
    sorted_notes = sorted(midi_notes)
    bass = sorted_notes[0]
    
    # Get unique pitch classes relative to bass
    intervals = set()
    for note in sorted_notes:
        interval = (note - bass) % 12
        intervals.add(interval)
    
    return bass, sorted(intervals)


def match_chord_template(
    intervals: list[int],
    template_name: str,
    template_intervals: list[int]
) -> Optional[ChordMatch]:
    """
    Check how well intervals match a chord template.
    
    Allows for:
    - Missing notes (especially 5th)
    - Extra notes (extensions, bass notes)
    - Octave equivalence
    """
    if not intervals:
        return None
    
    # Normalize template to pitch classes
    template_pcs = set(i % 12 for i in template_intervals)
    input_pcs = set(i % 12 for i in intervals)
    
    # Must have the root (0)
    if 0 not in input_pcs:
        return None
    
    # Calculate matches
    matched = template_pcs & input_pcs
    missing = template_pcs - input_pcs
    extra = input_pcs - template_pcs
    
    # Must have at least the root and one other template note
    if len(matched) < 2:
        return None
    
    # Score calculation
    # - Reward matched notes
    # - Penalize missing notes (but 5th is less important)
    # - Small penalty for extra notes
    
    match_score = len(matched) / len(template_pcs)
    
    # Missing 5th (interval 7) is okay
    missing_penalty = 0
    for m in missing:
        if m == 7:
            missing_penalty += 0.1  # 5th is optional
        elif m in [0, 3, 4]:  # Root or 3rd is important
            missing_penalty += 0.4
        else:
            missing_penalty += 0.2
    
    # Extra notes are okay if they're common extensions
    extra_penalty = 0
    common_extensions = {2, 9, 14}  # 9th
    common_extensions.update({5, 17})  # 11th  
    common_extensions.update({9, 21})  # 13th, 6th
    
    for e in extra:
        if e in common_extensions or (e % 12) in {2, 5, 9}:
            extra_penalty += 0.05
        else:
            extra_penalty += 0.1
    
    final_score = max(0, match_score - missing_penalty - extra_penalty)
    
    return ChordMatch(
        root='',  # Will be filled in later
        root_midi=0,
        quality=template_name,
        name='',
        matched_notes=len(matched),
        total_notes=len(input_pcs),
        match_score=final_score,
        missing_notes=list(missing),
        extra_notes=list(extra)
    )


def detect_chord(
    midi_notes: list[int],
    prefer_bass_as_root: bool = True,
    min_score: float = 0.3
) -> Optional[ChordMatch]:
    """
    Detect chord from a list of MIDI notes.
    
    Args:
        midi_notes: List of MIDI note numbers
        prefer_bass_as_root: If True, prefer chords where bass = root
        min_score: Minimum match score to accept
    
    Returns:
        Best matching ChordMatch, or None if no good match
    """
    if not midi_notes or len(midi_notes) < 2:
        return None
    
    # Remove duplicates and sort
    unique_notes = sorted(set(midi_notes))
    
    best_match = None
    best_score = min_score
    
    # Try each note as potential root
    for root_midi in unique_notes:
        root_pc = midi_to_pitch_class(root_midi)
        
        # Get intervals relative to this root
        intervals = []
        for note in unique_notes:
            interval = (note - root_midi) % 12
            intervals.append(interval)
        
        intervals = sorted(set(intervals))
        
        # Try each chord template
        for template_name, template_intervals in CHORD_TEMPLATES.items():
            match = match_chord_template(intervals, template_name, template_intervals)
            
            if match is None:
                continue
            
            # Bonus if root is the bass note
            is_root_bass = (root_midi == unique_notes[0])
            if prefer_bass_as_root and is_root_bass:
                match.match_score += 0.15
            
            # Prefer simpler chords (triads over 7ths over extensions)
            simplicity_bonus = {
                'maj': 0.1, 'min': 0.1, '5': 0.05,
                'dim': 0.08, 'aug': 0.08,
                'sus2': 0.07, 'sus4': 0.07,
                '7': 0.05, 'maj7': 0.05, 'min7': 0.05,
            }.get(template_name, 0)
            match.match_score += simplicity_bonus
            
            if match.match_score > best_score:
                best_score = match.match_score
                
                # Fill in root info
                match.root = pitch_class_to_name(root_pc)
                match.root_midi = root_midi
                match.name = f"{match.root}{template_name}"
                
                # Clean up name
                if template_name == 'maj':
                    match.name = match.root  # "C" instead of "Cmaj"
                elif template_name == 'min':
                    match.name = f"{match.root}m"
                
                best_match = match
    
    return best_match


def detect_chord_from_names(note_names: list[str]) -> Optional[ChordMatch]:
    """
    Detect chord from note names like ["C4", "E4", "G4"].
    """
    midi_notes = []
    for name in note_names:
        midi = note_name_to_midi(name)
        if midi > 0:
            midi_notes.append(midi)
    
    return detect_chord(midi_notes)


def note_name_to_midi(name: str) -> int:
    """Convert note name like 'C4' or 'F#3' to MIDI number."""
    if not name:
        return 0
    
    # Parse note and octave
    note_part = ''
    octave_part = ''
    
    for i, char in enumerate(name):
        if char.isdigit() or (char == '-' and i > 0):
            octave_part = name[i:]
            break
        note_part += char
    
    if not note_part or not octave_part:
        return 0
    
    # Find note index
    note_upper = note_part.upper()
    try:
        note_idx = NOTE_NAMES.index(note_upper)
    except ValueError:
        # Try without sharp
        if len(note_upper) > 1 and note_upper[1] == '#':
            try:
                base_idx = NOTE_NAMES.index(note_upper[0])
                note_idx = (base_idx + 1) % 12
            except ValueError:
                return 0
        else:
            return 0
    
    try:
        octave = int(octave_part)
    except ValueError:
        return 0
    
    return (octave + 1) * 12 + note_idx


# =============================================================================
# Integration with Transcription
# =============================================================================

def analyze_slice_chord(
    bass_notes: list,  # list[DetectedNote]
    chord_notes: list,
    melody_notes: list = None,
    include_melody: bool = False
) -> Optional[ChordMatch]:
    """
    Analyze chord from a TimeSlice's notes.
    
    Args:
        bass_notes: Bass notes from the slice
        chord_notes: Chord notes from the slice
        melody_notes: Melody notes (optional)
        include_melody: Whether to include melody in chord analysis
    """
    midi_notes = []
    
    # Add bass
    for note in bass_notes:
        midi_notes.append(note.midi_note)
    
    # Add chords
    for note in chord_notes:
        midi_notes.append(note.midi_note)
    
    # Optionally add melody
    if include_melody and melody_notes:
        for note in melody_notes:
            midi_notes.append(note.midi_note)
    
    return detect_chord(midi_notes)


def format_chord_for_display(chord: Optional[ChordMatch], width: int = 12) -> str:
    """Format chord name for display in transcription output."""
    if chord is None:
        return "-".ljust(width)
    
    # Confidence indicator
    if chord.match_score >= 0.8:
        confidence = ""
    elif chord.match_score >= 0.5:
        confidence = "?"
    else:
        confidence = "??"
    
    name = f"{chord.name}{confidence}"
    return name.ljust(width)


# =============================================================================
# Enhanced Transcription Export
# =============================================================================

def export_transcription_with_chords(
    transcription,  # TwoPassTranscription
    output_path,
    include_melody_in_chord: bool = False
) -> None:
    """
    Export transcription with chord analysis column.
    """
    from pathlib import Path
    
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write(f"# Two-Pass Transcription with Chord Analysis\n")
        f.write(f"# Source: {transcription.source_file}\n")
        f.write(f"# Duration: {transcription.duration_seconds:.2f}s\n\n")
        
        header = f"{'Time':<8} {'Chord':<12} {'Bass':<12} {'Chords':<28} {'Melody'}\n"
        f.write(header)
        f.write("-" * 90 + "\n")
        
        for sl in transcription.slices:
            # Detect chord
            chord = analyze_slice_chord(
                sl.bass_notes, 
                sl.chord_notes,
                sl.melody_notes,
                include_melody=include_melody_in_chord
            )
            
            # Format columns
            chord_str = format_chord_for_display(chord, width=12)
            bass = " ".join(n.note_name for n in sl.bass_notes) or "-"
            chords = " ".join(n.note_name for n in sl.chord_notes) or "-"
            melody = " ".join(n.note_name for n in sl.melody_notes) or "-"
            
            f.write(f"{sl.time:<8.2f} {chord_str} {bass:<12} {chords:<28} {melody}\n")
    
    print(f"Saved: {output_path}")


def print_transcription_with_chords(
    transcription,  # TwoPassTranscription
    max_slices: int = 30,
    include_melody_in_chord: bool = False
) -> None:
    """
    Print transcription with chord analysis.
    """
    print("\n" + "=" * 100)
    print("TRANSCRIPTION WITH CHORD ANALYSIS")
    print("=" * 100)
    
    print(f"{'Time':<8} {'Chord':<12} {'Bass':<12} {'Chords':<28} {'Melody'}")
    print("-" * 100)
    
    for sl in transcription.slices[:max_slices]:
        chord = analyze_slice_chord(
            sl.bass_notes,
            sl.chord_notes, 
            sl.melody_notes,
            include_melody=include_melody_in_chord
        )
        
        chord_str = format_chord_for_display(chord, width=12)
        bass = " ".join(n.note_name for n in sl.bass_notes)[:11] or "-"
        chords = " ".join(n.note_name for n in sl.chord_notes)[:27] or "-"
        melody = " ".join(n.note_name for n in sl.melody_notes) or "-"
        
        print(f"{sl.time:<8.2f} {chord_str} {bass:<12} {chords:<28} {melody}")
    
    if len(transcription.slices) > max_slices:
        print(f"... and {len(transcription.slices) - max_slices} more slices")
    
    print("=" * 100)
    
    # Chord progression summary
    print("\nCHORD PROGRESSION SUMMARY:")
    print("-" * 50)
    
    prev_chord = None
    chord_sequence = []
    
    for sl in transcription.slices:
        chord = analyze_slice_chord(sl.bass_notes, sl.chord_notes)
        
        if chord is not None:
            chord_name = chord.name
            if chord_name != prev_chord:
                chord_sequence.append((sl.time, chord_name, chord.match_score))
                prev_chord = chord_name
    
    for time, name, score in chord_sequence[:40]:
        conf = "✓" if score >= 0.6 else "?"
        print(f"  {time:>6.2f}s: {name:<10} {conf}")
    
    if len(chord_sequence) > 40:
        print(f"  ... and {len(chord_sequence) - 40} more chord changes")


# =============================================================================
# Test
# =============================================================================

def test_chord_detection():
    """Test chord detection with various inputs."""
    
    test_cases = [
        # (midi_notes, expected_chord)
        ([60, 64, 67], "C"),           # C major
        ([60, 63, 67], "Cm"),          # C minor
        ([60, 64, 67, 71], "Cmaj7"),   # C major 7
        ([60, 64, 67, 70], "C7"),      # C dominant 7
        ([60, 63, 67, 70], "Cm7"),     # C minor 7
        ([62, 65, 69], "Dm"),          # D minor
        ([65, 69, 72], "F"),           # F major
        ([67, 71, 74], "G"),           # G major
        ([67, 71, 74, 77], "Gmaj7"),   # G major 7
        ([60, 65, 67], "Csus4"),       # C sus4
        ([60, 62, 67], "Csus2"),       # C sus2
        ([60, 64, 68], "Caug"),        # C augmented
        ([60, 63, 66], "Cdim"),        # C diminished
        ([60, 64, 67, 69], "C6"),      # C 6
        ([60, 64, 67, 70, 74], "C9"),  # C 9
        ([57, 60, 64, 67], "Am"),      # A minor (with bass)
    ]
    
    print("CHORD DETECTION TEST")
    print("=" * 60)
    print(f"{'Input MIDI':<20} {'Detected':<15} {'Expected':<15} {'Score'}")
    print("-" * 60)
    
    for midi_notes, expected in test_cases:
        chord = detect_chord(midi_notes)
        detected = chord.name if chord else "None"
        score = f"{chord.match_score:.2f}" if chord else "-"
        
        match = "✓" if detected.replace('maj', '') == expected.replace('maj', '') else "✗"
        
        notes_str = str(midi_notes)[:18]
        print(f"{notes_str:<20} {detected:<15} {expected:<15} {score} {match}")
    
    print("=" * 60)


if __name__ == '__main__':
    test_chord_detection()