"""
Note Event Fusion: Combining Slice-Based Notes with Onset Timing
Author: Andre Lim

This module fuses two sources of information:
- Transcription A: Fine-grained note detection from 0.1s slices
- Transcription B: Onset/beat detection for timing

Output: NoteEvent objects with actual start times and durations

Key insight:
- Onset = something NEW happens (note attack)
- Between onsets = notes are sustained/tied
- Duration = time until next onset (or note disappears)
"""

import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, Counter
from pathlib import Path
import json


# =============================================================================
# Playability Constraints
# =============================================================================

# Maximum hand span in semitones (10 semitones = minor 7th, comfortable for most hands)
MAX_HAND_SPAN_BASS = 10  # Left hand max span
MAX_HAND_SPAN_TREBLE = 10  # Right hand max span

# Maximum notes per hand
MAX_NOTES_PER_HAND = 5  # Typical hand can play 5 notes max comfortably

# MIDI note for middle C (C4)
MIDDLE_C = 60

# Split point between bass and treble clef
BASS_TREBLE_SPLIT = 60  # C4

# Role weighting for filtering decisions
ROLE_WEIGHTS = {
    'melody': 1.0,   # Neutral - prone to noise but important
    'chord': 1.5,    # Higher weight - more reliable, emphasize
    'bass': 0.7,     # Lower weight - often has artifacts, de-emphasize
    'unknown': 0.8
}


def filter_by_hand_span(
    notes: list[int],
    max_span: int,
    max_notes: int = MAX_NOTES_PER_HAND,
    prefer: str = 'outer'
) -> list[int]:
    """
    Filter notes to fit within a playable hand span and note count.
    
    Args:
        notes: List of MIDI notes
        max_span: Maximum span in semitones
        max_notes: Maximum number of notes (typically 5 for one hand)
        prefer: Which notes to keep - 'outer' (highest/lowest), 'lowest', 'highest'
    
    Returns:
        Filtered list of MIDI notes
    """
    if len(notes) <= 1:
        return notes
    
    notes = sorted(set(notes))  # Remove duplicates and sort
    
    # If already fits both constraints, return as-is
    if notes[-1] - notes[0] <= max_span and len(notes) <= max_notes:
        return notes
    
    # Strategy: Find the best contiguous window, then select notes within it
    best_window_notes = []
    best_score = -1
    
    # Try each note as the starting point of a window
    for i, start_note in enumerate(notes):
        # Find all notes within max_span of this start note
        window = [n for n in notes if start_note <= n <= start_note + max_span]
        
        if not window:
            continue
        
        # If window has more notes than max_notes, select the best subset
        if len(window) > max_notes:
            if prefer == 'lowest':
                # Take the lowest max_notes
                window = window[:max_notes]
            elif prefer == 'highest':
                # Take the highest max_notes
                window = window[-max_notes:]
            else:  # 'outer' - keep extremes and fill middle
                selected = [window[0], window[-1]]  # Always keep lowest and highest in window
                # Fill remaining slots from middle, preferring evenly spaced
                middle = window[1:-1]
                remaining = max_notes - 2
                if remaining > 0 and middle:
                    step = max(1, len(middle) // (remaining + 1))
                    for j in range(0, len(middle), step):
                        if len(selected) < max_notes:
                            selected.append(middle[j])
                window = sorted(selected)
        
        # Score this window
        if prefer == 'lowest':
            # Lower average pitch = better
            score = len(window) * 100 - sum(window) / len(window)
        elif prefer == 'highest':
            # Higher average pitch = better
            score = len(window) * 100 + sum(window) / len(window)
        else:  # 'outer'
            # Prefer windows that include the original extremes
            score = len(window) * 100
            if notes[0] in window:
                score += 50
            if notes[-1] in window:
                score += 30
        
        if score > best_score:
            best_score = score
            best_window_notes = window
    
    return best_window_notes if best_window_notes else [notes[0]]


def apply_hand_span_constraint(
    note_events: list['NoteEvent'],
    max_bass_span: int = MAX_HAND_SPAN_BASS,
    max_treble_span: int = MAX_HAND_SPAN_TREBLE,
    max_notes_per_hand: int = MAX_NOTES_PER_HAND,
    split_point: int = BASS_TREBLE_SPLIT
) -> list['NoteEvent']:
    """
    Filter notes to ensure they're playable by human hands.
    
    Separates into bass (left hand) and treble (right hand) based on split point,
    then ensures each hand's notes fit within playable span and note count.
    
    Args:
        note_events: List of NoteEvent objects
        max_bass_span: Max semitones for left hand
        max_treble_span: Max semitones for right hand
        max_notes_per_hand: Max notes per hand (default 5)
        split_point: MIDI note to split bass/treble
    """
    if not note_events:
        return []
    
    print(f"\n  Applying hand span constraint (max {max_notes_per_hand} notes, {max_bass_span} semitones per hand)...")
    
    # Group by start time
    by_time = defaultdict(list)
    for event in note_events:
        time_key = round(event.start_time, 3)
        by_time[time_key].append(event)
    
    filtered = []
    removed = 0
    
    for time_key, events in sorted(by_time.items()):
        # Separate into bass and treble
        # Bass: notes below split point OR explicitly marked as bass
        bass_events = [e for e in events if e.midi_note < split_point or e.role == 'bass']
        # Treble: notes at/above split point that aren't bass
        treble_events = [e for e in events if e.midi_note >= split_point and e.role != 'bass']
        
        # Handle bass notes that are above split (e.g., bass line playing C4)
        # Move them to bass if they're close to other bass notes
        if bass_events and treble_events:
            bass_top = max(e.midi_note for e in bass_events) if bass_events else 0
            for e in list(treble_events):
                if e.role == 'bass' or (e.midi_note - bass_top <= 12 and e.midi_note < split_point + 12):
                    bass_events.append(e)
                    treble_events.remove(e)
        
        # Filter bass hand (left hand)
        if bass_events:
            bass_notes = [e.midi_note for e in bass_events]
            playable_bass = set(filter_by_hand_span(
                bass_notes, max_bass_span, max_notes_per_hand, prefer='lowest'
            ))
            
            for event in bass_events:
                if event.midi_note in playable_bass:
                    filtered.append(event)
                    playable_bass.discard(event.midi_note)  # Remove to handle duplicates
                else:
                    removed += 1
        
        # Filter treble hand (right hand)
        if treble_events:
            treble_notes = [e.midi_note for e in treble_events]
            playable_treble = set(filter_by_hand_span(
                treble_notes, max_treble_span, max_notes_per_hand, prefer='outer'
            ))
            
            for event in treble_events:
                if event.midi_note in playable_treble:
                    filtered.append(event)
                    playable_treble.discard(event.midi_note)  # Remove to handle duplicates
                else:
                    removed += 1
    
    filtered.sort(key=lambda e: (e.start_time, -e.midi_note))
    
    print(f"    Removed {removed} notes (hand span/count constraint)")
    print(f"    Remaining: {len(filtered)} notes")
    
    return filtered


# =============================================================================
# Role-Based Filtering
# =============================================================================

def filter_by_role_weight(
    note_events: list['NoteEvent'],
    role_weights: dict = None,
    confidence_threshold: float = 0.3
) -> list['NoteEvent']:
    """
    Filter notes based on role reliability weights.
    
    Bass notes are more prone to artifacts, so we apply stricter filtering.
    Chord notes are more reliable, so we're more lenient.
    
    Args:
        note_events: List of NoteEvent objects
        role_weights: Dict mapping role -> weight multiplier
        confidence_threshold: Base confidence threshold (adjusted by weight)
    
    Returns:
        Filtered list of NoteEvent objects
    """
    if role_weights is None:
        role_weights = ROLE_WEIGHTS
    
    print(f"\n  Applying role-based filtering...")
    print(f"    Weights: {role_weights}")
    
    filtered = []
    removed_by_role = defaultdict(int)
    
    for event in note_events:
        weight = role_weights.get(event.role, 1.0)
        
        # Adjust threshold inversely to weight
        # Higher weight = lower threshold = more notes kept
        # Lower weight = higher threshold = fewer notes kept
        adjusted_threshold = confidence_threshold / weight
        
        if event.velocity >= adjusted_threshold:
            filtered.append(event)
        else:
            removed_by_role[event.role] += 1
    
    total_removed = sum(removed_by_role.values())
    print(f"    Removed {total_removed} notes by role:")
    for role, count in sorted(removed_by_role.items()):
        print(f"      {role}: {count}")
    print(f"    Remaining: {len(filtered)} notes")
    
    return filtered


def rebalance_by_role(
    note_events: list['NoteEvent'],
    max_bass_notes: int = 2,
    max_chord_notes: int = 4,
    max_melody_notes: int = 2,
    prefer_chord_roots: bool = True
) -> list['NoteEvent']:
    """
    Rebalance notes by role - fewer bass, more chords.
    
    This runs per time slice and ensures a good balance:
    - Bass: Keep 1-2 notes (usually root and fifth)
    - Chords: Keep up to 4 notes (the harmony)
    - Melody: Keep 1-2 notes (main melody line)
    
    Args:
        note_events: List of NoteEvent objects
        max_bass_notes: Max bass notes per time slice
        max_chord_notes: Max chord notes per time slice  
        max_melody_notes: Max melody notes per time slice
        prefer_chord_roots: If True, prefer chord root/fifth for bass
    """
    if not note_events:
        return []
    
    print(f"\n  Rebalancing by role (bass:{max_bass_notes}, chord:{max_chord_notes}, melody:{max_melody_notes})...")
    
    # Group by start time
    by_time = defaultdict(list)
    for event in note_events:
        time_key = round(event.start_time, 3)
        by_time[time_key].append(event)
    
    filtered = []
    removed = 0
    
    for time_key, events in sorted(by_time.items()):
        # Separate by role
        bass = [e for e in events if e.role == 'bass']
        chords = [e for e in events if e.role == 'chord']
        melody = [e for e in events if e.role == 'melody']
        other = [e for e in events if e.role not in ['bass', 'chord', 'melody']]
        
        # Select bass notes (prefer lowest, limit count)
        if bass:
            bass_sorted = sorted(bass, key=lambda e: e.midi_note)  # Lowest first
            selected_bass = bass_sorted[:max_bass_notes]
            filtered.extend(selected_bass)
            removed += len(bass) - len(selected_bass)
        
        # Select chord notes (prefer by confidence/velocity)
        if chords:
            chords_sorted = sorted(chords, key=lambda e: -e.velocity)  # Highest confidence first
            selected_chords = chords_sorted[:max_chord_notes]
            filtered.extend(selected_chords)
            removed += len(chords) - len(selected_chords)
        
        # Select melody notes (prefer highest pitched, typical melody behavior)
        if melody:
            melody_sorted = sorted(melody, key=lambda e: -e.midi_note)  # Highest first
            selected_melody = melody_sorted[:max_melody_notes]
            filtered.extend(selected_melody)
            removed += len(melody) - len(selected_melody)
        
        # Keep other notes as-is (they'll be filtered elsewhere)
        filtered.extend(other)
    
    filtered.sort(key=lambda e: (e.start_time, -e.midi_note))
    
    print(f"    Removed {removed} notes (role rebalancing)")
    print(f"    Remaining: {len(filtered)} notes")
    
    return filtered


# =============================================================================
# Majority Vote Filtering
# =============================================================================

def filter_by_majority_vote(
    note_events: list['NoteEvent'],
    slices: list,  # TimeSlice objects
    min_presence_ratio: float = 0.4,
    min_slice_count: int = 2
) -> list['NoteEvent']:
    """
    Filter notes using majority vote within onset segments.
    
    If a note only appears in a small fraction of slices within its duration,
    it's likely noise/artifact and should be filtered.
    
    Args:
        note_events: List of NoteEvent objects
        slices: TimeSlice objects from transcription
        min_presence_ratio: Minimum fraction of slices note must appear in (0-1)
        min_slice_count: Minimum number of slices note must appear in
    
    Returns:
        Filtered list of NoteEvent objects
    """
    if not note_events or not slices:
        return note_events
    
    print(f"\n  Applying majority vote filter...")
    
    # Build lookup: time -> slice
    slice_by_time = {}
    for sl in slices:
        time_key = round(sl.time, 2)
        slice_by_time[time_key] = sl
    
    filtered = []
    removed = 0
    
    for event in note_events:
        # Find all slices within this note's duration
        start = event.start_time
        end = event.end_time
        
        # Count slices where this note appears
        total_slices = 0
        present_slices = 0
        
        for time_key, sl in slice_by_time.items():
            if start <= sl.time < end:
                total_slices += 1
                
                # Check if note is in this slice
                note_in_slice = False
                
                # Check in the appropriate role
                if event.role == 'melody':
                    note_in_slice = any(n.midi_note == event.midi_note for n in sl.melody_notes)
                elif event.role == 'bass':
                    note_in_slice = any(n.midi_note == event.midi_note for n in sl.bass_notes)
                elif event.role == 'chord':
                    note_in_slice = any(n.midi_note == event.midi_note for n in sl.chord_notes)
                else:
                    # Check all
                    all_notes = sl.melody_notes + sl.chord_notes + sl.bass_notes
                    note_in_slice = any(n.midi_note == event.midi_note for n in all_notes)
                
                if note_in_slice:
                    present_slices += 1
        
        # Decide whether to keep
        if total_slices == 0:
            # No slices found - keep the note (edge case)
            filtered.append(event)
        else:
            presence_ratio = present_slices / total_slices
            
            # Keep if meets threshold OR is melody (more lenient for melody)
            keep = (presence_ratio >= min_presence_ratio and present_slices >= min_slice_count)
            
            # Be more lenient for melody
            if event.role == 'melody':
                keep = keep or (presence_ratio >= min_presence_ratio * 0.5)
            
            # Be more lenient for short notes
            if event.duration_beats <= 0.5:
                keep = keep or (present_slices >= 1)
            
            if keep:
                filtered.append(event)
            else:
                removed += 1
    
    filtered.sort(key=lambda e: (e.start_time, -e.midi_note))
    
    print(f"    Removed {removed} notes (majority vote)")
    print(f"    Remaining: {len(filtered)} notes")
    
    return filtered


# =============================================================================
# Combined Playability Filter
# =============================================================================

def apply_playability_filters(
    note_events: list['NoteEvent'],
    slices: list = None,
    apply_hand_span: bool = True,
    apply_majority_vote: bool = True,
    apply_role_rebalance: bool = True,
    max_bass_span: int = MAX_HAND_SPAN_BASS,
    max_treble_span: int = MAX_HAND_SPAN_TREBLE,
    max_notes_per_hand: int = MAX_NOTES_PER_HAND,
    min_presence_ratio: float = 0.4,
    max_bass_notes: int = 2,
    max_chord_notes: int = 4,
    max_melody_notes: int = 2
) -> list['NoteEvent']:
    """
    Apply all playability filters in sequence.
    
    Order:
    1. Majority vote (remove notes that appear briefly)
    2. Role rebalancing (fewer bass, more chords)
    3. Hand span constraint (ensure playable by human hands)
    """
    result = note_events
    
    if apply_majority_vote and slices:
        result = filter_by_majority_vote(
            result, slices,
            min_presence_ratio=min_presence_ratio,
            min_slice_count=2
        )
    
    if apply_role_rebalance:
        result = rebalance_by_role(
            result,
            max_bass_notes=max_bass_notes,
            max_chord_notes=max_chord_notes,
            max_melody_notes=max_melody_notes
        )
    
    if apply_hand_span:
        result = apply_hand_span_constraint(
            result,
            max_bass_span=max_bass_span,
            max_treble_span=max_treble_span,
            max_notes_per_hand=max_notes_per_hand
        )
    
    return result

# Common chord intervals (in semitones from root)
CHORD_TEMPLATES = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
    '7': [0, 4, 7, 10],
    'dim7': [0, 3, 6, 9],
    'hdim7': [0, 3, 6, 10],  # half-diminished
    'add9': [0, 4, 7, 14],
    '9': [0, 4, 7, 10, 14],
    '6': [0, 4, 7, 9],
    'min6': [0, 3, 7, 9],
}

# Intervals that sound consonant together
CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9, 12, 15, 16}  # unison, m3, M3, P4, P5, m6, M6, octave, etc.

# Dissonant intervals to avoid (unless passing tones)
DISSONANT_INTERVALS = {1, 2, 6, 10, 11}  # m2, M2, tritone, m7, M7


# =============================================================================
# Harmonic Analysis Functions
# =============================================================================

def get_pitch_class(midi_note: int) -> int:
    """Get pitch class (0-11) from MIDI note."""
    return midi_note % 12


def get_interval(note1: int, note2: int) -> int:
    """Get interval in semitones between two notes (always positive, mod 12)."""
    return abs(note1 - note2) % 12


def compute_harmonic_tension(notes: list[int]) -> float:
    """
    Compute harmonic tension of a set of notes.
    Lower = more consonant, Higher = more dissonant.
    
    Returns value between 0 (perfectly consonant) and 1 (highly dissonant).
    """
    if len(notes) < 2:
        return 0.0
    
    tension = 0.0
    pairs = 0
    
    for i, n1 in enumerate(notes):
        for n2 in notes[i+1:]:
            interval = get_interval(n1, n2)
            pairs += 1
            
            if interval in DISSONANT_INTERVALS:
                tension += 1.0
            elif interval not in CONSONANT_INTERVALS:
                tension += 0.5
    
    return tension / max(pairs, 1)


def find_chord_root(notes: list[int]) -> tuple[int, str, float]:
    """
    Find the most likely root and chord type for a set of notes.
    
    Returns (root_pitch_class, chord_type, confidence).
    """
    if not notes:
        return (0, 'unknown', 0.0)
    
    pitch_classes = set(get_pitch_class(n) for n in notes)
    
    best_root = 0
    best_type = 'unknown'
    best_score = 0
    
    # Try each pitch class as potential root
    for root in range(12):
        for chord_type, intervals in CHORD_TEMPLATES.items():
            expected_pcs = set((root + i) % 12 for i in intervals)
            
            # Score = how many expected notes are present
            matches = len(pitch_classes & expected_pcs)
            extras = len(pitch_classes - expected_pcs)
            
            # Penalize extra notes slightly
            score = matches - extras * 0.3
            
            # Bonus if root is present
            if root in pitch_classes:
                score += 0.5
            
            # Bonus for simpler chords
            if chord_type in ['major', 'minor']:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_root = root
                best_type = chord_type
    
    confidence = best_score / (len(pitch_classes) + 1)
    return (best_root, best_type, confidence)


def note_fits_chord(midi_note: int, chord_root: int, chord_type: str, 
                    tolerance: int = 2) -> bool:
    """
    Check if a note fits within a chord (with some tolerance for passing tones).
    
    Args:
        midi_note: MIDI note to check
        chord_root: Root pitch class (0-11)
        chord_type: Chord type string
        tolerance: How many semitones away is still acceptable (for extensions)
    """
    if chord_type not in CHORD_TEMPLATES:
        return True  # Unknown chord, accept all
    
    pc = get_pitch_class(midi_note)
    chord_pcs = set((chord_root + i) % 12 for i in CHORD_TEMPLATES[chord_type])
    
    # Direct match
    if pc in chord_pcs:
        return True
    
    # Check for common extensions (9th, 11th, 13th)
    extensions = [(chord_root + 14) % 12,  # 9th
                  (chord_root + 17) % 12,  # 11th
                  (chord_root + 21) % 12]  # 13th
    
    if pc in extensions:
        return True
    
    # Check tolerance (passing tone)
    for chord_pc in chord_pcs:
        if min(abs(pc - chord_pc), 12 - abs(pc - chord_pc)) <= tolerance:
            return True
    
    return False


# =============================================================================
# Harmonic Smoothing / Cleaning
# =============================================================================

def smooth_notes_by_harmony(
    note_events: list['NoteEvent'],
    segments_per_bar: int = 4,
    min_notes_for_chord: int = 3,
    dissonance_threshold: float = 0.5,
    remove_isolated: bool = True,
    min_duration_beats: float = 0.25
) -> list['NoteEvent']:
    """
    Filter notes based on harmonic coherence.
    
    Strategy:
    1. Group notes by bar/beat segment
    2. Find likely chord for each segment
    3. Remove notes that don't fit the chord
    4. Remove isolated "stray" notes
    
    Args:
        note_events: List of NoteEvent objects
        segments_per_bar: How many segments to divide each bar into
        min_notes_for_chord: Minimum notes needed to determine a chord
        dissonance_threshold: Max tension allowed (0-1)
        remove_isolated: Remove notes that appear alone briefly
        min_duration_beats: Minimum duration to keep a note
    
    Returns:
        Filtered list of NoteEvent objects
    """
    if not note_events:
        return []
    
    print(f"\n  Harmonic smoothing ({len(note_events)} notes)...")
    
    # Group notes by time segment (bar + beat subdivision)
    def get_segment_key(event):
        beat_segment = int((event.beat - 1) * segments_per_bar / 4)
        return (event.bar, beat_segment)
    
    segments = defaultdict(list)
    for event in note_events:
        key = get_segment_key(event)
        segments[key].append(event)
    
    # Analyze each segment
    filtered_events = []
    removed_count = 0
    
    for key, events in sorted(segments.items()):
        bar, seg = key
        
        # Separate by role
        melody = [e for e in events if e.role == 'melody']
        chords = [e for e in events if e.role == 'chord']
        bass = [e for e in events if e.role == 'bass']
        
        # Find chord from chord+bass notes
        harmony_notes = [e.midi_note for e in chords + bass]
        
        if len(harmony_notes) >= min_notes_for_chord:
            root, chord_type, confidence = find_chord_root(harmony_notes)
        else:
            root, chord_type, confidence = (0, 'unknown', 0)
        
        # Filter chord notes for harmonic coherence
        filtered_chords = []
        for event in chords:
            # Check if note fits the detected chord
            if chord_type != 'unknown':
                fits = note_fits_chord(event.midi_note, root, chord_type, tolerance=2)
            else:
                fits = True
            
            # Check duration
            long_enough = event.duration_beats >= min_duration_beats
            
            # Check for dissonance with other notes
            other_notes = [e.midi_note for e in filtered_chords]
            tension = compute_harmonic_tension(other_notes + [event.midi_note])
            consonant = tension < dissonance_threshold
            
            if fits and long_enough and consonant:
                filtered_chords.append(event)
            else:
                removed_count += 1
        
        # Filter melody notes - more lenient (melody can have passing tones)
        filtered_melody = []
        for event in melody:
            long_enough = event.duration_beats >= min_duration_beats
            
            if long_enough:
                filtered_melody.append(event)
            else:
                removed_count += 1
        
        # Filter bass notes - keep strongest/lowest
        filtered_bass = []
        if bass:
            # Sort by pitch (lowest first) then by confidence
            sorted_bass = sorted(bass, key=lambda e: (e.midi_note, -e.velocity))
            # Keep only 1-2 bass notes per segment
            filtered_bass = sorted_bass[:2]
            removed_count += len(bass) - len(filtered_bass)
        
        filtered_events.extend(filtered_melody)
        filtered_events.extend(filtered_chords)
        filtered_events.extend(filtered_bass)
    
    # Remove isolated notes (appear in only one segment, very short)
    if remove_isolated:
        note_occurrences = Counter(e.midi_note for e in filtered_events)
        
        non_isolated = []
        for event in filtered_events:
            # Keep if appears multiple times OR is long enough OR is melody
            keep = (note_occurrences[event.midi_note] > 1 or 
                   event.duration_beats >= 0.5 or
                   event.role == 'melody')
            
            if keep:
                non_isolated.append(event)
            else:
                removed_count += 1
        
        filtered_events = non_isolated
    
    # Sort by time
    filtered_events.sort(key=lambda e: (e.start_time, -e.midi_note))
    
    print(f"    Removed {removed_count} notes (harmonic filtering)")
    print(f"    Remaining: {len(filtered_events)} notes")
    
    return filtered_events


def reduce_chord_density(
    note_events: list['NoteEvent'],
    max_chord_notes: int = 4,
    max_total_notes: int = 6
) -> list['NoteEvent']:
    """
    Reduce the number of simultaneous notes to make output cleaner.
    
    Keeps the most important notes:
    - All melody notes
    - Lowest bass note
    - Top N chord notes by confidence/energy
    """
    if not note_events:
        return []
    
    print(f"\n  Reducing chord density...")
    
    # Group by exact start time
    by_time = defaultdict(list)
    for event in note_events:
        # Round to avoid floating point issues
        time_key = round(event.start_time, 3)
        by_time[time_key].append(event)
    
    filtered = []
    removed = 0
    
    for time_key, events in sorted(by_time.items()):
        melody = [e for e in events if e.role == 'melody']
        chords = [e for e in events if e.role == 'chord']
        bass = [e for e in events if e.role == 'bass']
        
        # Keep all melody
        kept = list(melody)
        
        # Keep lowest bass note
        if bass:
            bass_sorted = sorted(bass, key=lambda e: e.midi_note)
            kept.append(bass_sorted[0])
            removed += len(bass) - 1
        
        # Keep top chord notes (by velocity/confidence)
        remaining_slots = max_total_notes - len(kept)
        chord_slots = min(max_chord_notes, remaining_slots)
        
        if chords and chord_slots > 0:
            # Sort by velocity (confidence), keep highest
            chords_sorted = sorted(chords, key=lambda e: -e.velocity)
            kept.extend(chords_sorted[:chord_slots])
            removed += len(chords) - min(len(chords), chord_slots)
        
        filtered.extend(kept)
    
    filtered.sort(key=lambda e: (e.start_time, -e.midi_note))
    
    print(f"    Removed {removed} notes (density reduction)")
    print(f"    Remaining: {len(filtered)} notes")
    
    return filtered


def apply_temporal_smoothing(
    note_events: list['NoteEvent'],
    min_gap_beats: float = 0.25,
    merge_threshold_beats: float = 0.125
) -> list['NoteEvent']:
    """
    Smooth notes temporally:
    - Merge very short gaps between same notes
    - Remove notes that are too short
    """
    if not note_events:
        return []
    
    # Group by MIDI note
    by_note = defaultdict(list)
    for event in note_events:
        by_note[event.midi_note].append(event)
    
    smoothed = []
    
    for midi, events in by_note.items():
        # Sort by start time
        events = sorted(events, key=lambda e: e.start_time)
        
        merged = []
        current = None
        
        for event in events:
            if current is None:
                current = event
            else:
                # Check gap between current end and new start
                gap = event.start_time - current.end_time
                gap_beats = gap * event.duration_beats / max(event.duration, 0.001)
                
                if gap_beats < merge_threshold_beats:
                    # Merge: extend current note
                    new_duration = event.end_time - current.start_time
                    current = NoteEvent(
                        midi_note=current.midi_note,
                        note_name=current.note_name,
                        start_time=current.start_time,
                        duration=new_duration,
                        bar=current.bar,
                        beat=current.beat,
                        duration_beats=current.duration_beats + event.duration_beats,
                        velocity=max(current.velocity, event.velocity),
                        role=current.role,
                        is_tied=current.is_tied
                    )
                else:
                    merged.append(current)
                    current = event
        
        if current is not None:
            merged.append(current)
        
        smoothed.extend(merged)
    
    smoothed.sort(key=lambda e: (e.start_time, -e.midi_note))
    return smoothed


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NoteEvent:
    """A musical note with start time and duration."""
    midi_note: int
    note_name: str
    start_time: float      # Absolute time in seconds
    duration: float        # Duration in seconds
    bar: int               # Bar number (1-indexed)
    beat: float            # Beat position within bar (1-indexed, can be fractional)
    duration_beats: float  # Duration in beats
    velocity: float = 0.8  # 0.0-1.0
    role: str = 'unknown'  # 'melody', 'chord', 'bass'
    is_tied: bool = False  # True if tied from previous note
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration
    
    def to_dict(self) -> dict:
        return {
            'midi': self.midi_note,
            'name': self.note_name,
            'start': round(self.start_time, 4),
            'duration': round(self.duration, 4),
            'bar': self.bar,
            'beat': round(self.beat, 3),
            'duration_beats': round(self.duration_beats, 3),
            'velocity': round(self.velocity, 2),
            'role': self.role,
            'is_tied': self.is_tied
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'NoteEvent':
        return cls(
            midi_note=d['midi'],
            note_name=d['name'],
            start_time=d['start'],
            duration=d['duration'],
            bar=d['bar'],
            beat=d['beat'],
            duration_beats=d['duration_beats'],
            velocity=d.get('velocity', 0.8),
            role=d.get('role', 'unknown'),
            is_tied=d.get('is_tied', False)
        )


@dataclass
class OnsetSegment:
    """A segment between two onsets."""
    start_time: float
    end_time: float
    onset_strength: float = 0.0
    bar: int = 1
    beat: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class FusedTranscription:
    """Complete transcription with note events and timing."""
    duration_seconds: float
    source_file: str
    bpm: float
    time_signature: str
    first_beat_time: float
    note_events: list[NoteEvent] = field(default_factory=list)
    
    @property
    def melody_events(self) -> list[NoteEvent]:
        return [e for e in self.note_events if e.role == 'melody']
    
    @property
    def chord_events(self) -> list[NoteEvent]:
        return [e for e in self.note_events if e.role == 'chord']
    
    @property
    def bass_events(self) -> list[NoteEvent]:
        return [e for e in self.note_events if e.role == 'bass']
    
    @property
    def num_bars(self) -> int:
        if not self.note_events:
            return 0
        return max(e.bar for e in self.note_events)
    
    def to_dict(self) -> dict:
        return {
            'metadata': {
                'duration_seconds': round(self.duration_seconds, 2),
                'source_file': self.source_file,
                'bpm': self.bpm,
                'time_signature': self.time_signature,
                'first_beat_time': round(self.first_beat_time, 4),
                'num_bars': self.num_bars,
                'num_notes': len(self.note_events)
            },
            'note_events': [e.to_dict() for e in self.note_events]
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: Path) -> None:
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        print(f"Saved: {filepath}")
    
    @classmethod
    def from_dict(cls, d: dict) -> 'FusedTranscription':
        meta = d['metadata']
        events = [NoteEvent.from_dict(e) for e in d.get('note_events', [])]
        return cls(
            duration_seconds=meta['duration_seconds'],
            source_file=meta['source_file'],
            bpm=meta['bpm'],
            time_signature=meta['time_signature'],
            first_beat_time=meta['first_beat_time'],
            note_events=events
        )
    
    @classmethod
    def from_json_file(cls, filepath: Path) -> 'FusedTranscription':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Onset Segmentation
# =============================================================================

def create_onset_segments(
    onset_times: list[float],
    duration: float,
    min_segment_duration: float = 0.05
) -> list[OnsetSegment]:
    """
    Create segments between consecutive onsets.
    
    Args:
        onset_times: List of onset times in seconds
        duration: Total duration of the audio
        min_segment_duration: Minimum segment length (merge shorter ones)
    
    Returns:
        List of OnsetSegment objects
    """
    if not onset_times:
        return [OnsetSegment(start_time=0.0, end_time=duration)]
    
    # Ensure sorted
    onset_times = sorted(onset_times)
    
    # Add boundaries if needed
    if onset_times[0] > 0.1:
        onset_times = [0.0] + onset_times
    if onset_times[-1] < duration - 0.1:
        onset_times = onset_times + [duration]
    
    segments = []
    
    for i in range(len(onset_times) - 1):
        start = onset_times[i]
        end = onset_times[i + 1]
        
        # Skip very short segments (likely noise)
        if end - start < min_segment_duration:
            continue
        
        segments.append(OnsetSegment(
            start_time=start,
            end_time=end
        ))
    
    return segments


def assign_segment_bar_beats(
    segments: list[OnsetSegment],
    bpm: float,
    time_signature_numerator: int,
    first_beat_time: float
) -> list[OnsetSegment]:
    """Assign bar and beat positions to each segment."""
    seconds_per_beat = 60.0 / bpm
    
    for seg in segments:
        relative_time = seg.start_time - first_beat_time
        
        if relative_time < 0:
            seg.bar = 0
            seg.beat = 1.0
        else:
            total_beats = relative_time / seconds_per_beat
            seg.bar = int(total_beats // time_signature_numerator) + 1
            beat_in_bar = total_beats % time_signature_numerator
            seg.beat = beat_in_bar + 1  # 1-indexed
    
    return segments


# =============================================================================
# Note Clustering Within Segments
# =============================================================================

def cluster_notes_in_segment(
    segment: OnsetSegment,
    slices: list,  # TimeSlice objects from chord_transcription
    role: str = 'all'
) -> dict[int, dict]:
    """
    Cluster notes that appear within a segment.
    
    Args:
        segment: The onset segment
        slices: List of TimeSlice objects
        role: 'melody', 'chord', 'bass', or 'all'
    
    Returns:
        Dict mapping MIDI note -> {count, total_amp, avg_confidence, first_seen, last_seen}
    """
    note_stats = defaultdict(lambda: {
        'count': 0,
        'total_amp': 0.0,
        'total_confidence': 0.0,
        'first_seen': float('inf'),
        'last_seen': 0.0,
        'note_name': '',
        'role': ''
    })
    
    for sl in slices:
        if sl.time < segment.start_time or sl.time >= segment.end_time:
            continue
        
        # Collect notes based on role filter
        notes = []
        if role in ['all', 'melody']:
            notes.extend([(n, 'melody') for n in sl.melody_notes])
        if role in ['all', 'chord']:
            notes.extend([(n, 'chord') for n in sl.chord_notes])
        if role in ['all', 'bass']:
            notes.extend([(n, 'bass') for n in sl.bass_notes])
        
        for note, note_role in notes:
            stats = note_stats[note.midi_note]
            stats['count'] += 1
            stats['total_amp'] += note.fundamental_amp
            stats['total_confidence'] += note.confidence
            stats['first_seen'] = min(stats['first_seen'], sl.time)
            stats['last_seen'] = max(stats['last_seen'], sl.time)
            stats['note_name'] = note.note_name
            stats['role'] = note_role
    
    return dict(note_stats)


def determine_note_onset_type(
    note_midi: int,
    current_segment: OnsetSegment,
    prev_segment_notes: dict[int, dict],
    current_segment_notes: dict[int, dict],
    onset_threshold: float = 0.3
) -> tuple[bool, bool]:
    """
    Determine if a note is a new attack or tied from previous segment.
    
    Returns:
        (is_new_note, is_tied)
    """
    was_in_prev = note_midi in prev_segment_notes
    is_in_current = note_midi in current_segment_notes
    
    if not is_in_current:
        return False, False
    
    if not was_in_prev:
        # Note wasn't playing before - definitely new
        return True, False
    
    # Note was playing before - check if it's a re-attack or tie
    # Heuristic: If there's an onset and the note appears early in this segment,
    # it's likely a re-attack
    
    curr_stats = current_segment_notes[note_midi]
    first_seen_relative = curr_stats['first_seen'] - current_segment.start_time
    
    # If note appears within 10% of segment start, likely re-attacked
    segment_duration = current_segment.duration
    if first_seen_relative < segment_duration * 0.1:
        return True, False  # New attack
    else:
        return False, True  # Tied from previous


# =============================================================================
# Duration Quantization
# =============================================================================

def quantize_duration_to_beats(
    duration_seconds: float,
    bpm: float,
    allowed_durations: list[float] = None
) -> float:
    """
    Quantize a duration in seconds to the nearest musical duration in beats.
    
    Args:
        duration_seconds: Duration in seconds
        bpm: Beats per minute
        allowed_durations: List of allowed durations in beats 
                          (e.g., [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    
    Returns:
        Quantized duration in beats
    """
    if allowed_durations is None:
        # Standard note values: 16th, 8th, dotted 8th, quarter, dotted quarter, half, dotted half, whole
        allowed_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    seconds_per_beat = 60.0 / bpm
    duration_beats = duration_seconds / seconds_per_beat
    
    # Find nearest allowed duration
    nearest = min(allowed_durations, key=lambda d: abs(d - duration_beats))
    
    return nearest


def quantize_start_to_beat(
    start_time: float,
    bpm: float,
    first_beat_time: float,
    quantize_resolution: float = 0.25  # 16th note
) -> float:
    """
    Quantize start time to nearest beat subdivision.
    
    Returns quantized time in seconds.
    """
    seconds_per_beat = 60.0 / bpm
    relative_time = start_time - first_beat_time
    
    if relative_time < 0:
        return start_time  # Before first beat, don't quantize
    
    # Convert to beats
    beat_position = relative_time / seconds_per_beat
    
    # Quantize to nearest subdivision
    quantized_beats = round(beat_position / quantize_resolution) * quantize_resolution
    
    # Convert back to seconds
    return first_beat_time + quantized_beats * seconds_per_beat


# =============================================================================
# Main Fusion Algorithm
# =============================================================================

def fuse_transcription_with_onsets(
    slices: list,  # TimeSlice objects from chord_transcription
    onset_times: list[float],
    duration: float,
    bpm: float,
    time_signature_numerator: int = 4,
    time_signature_denominator: int = 4,
    first_beat_time: float = 0.0,
    source_file: str = "",
    min_note_confidence: float = 0.3,
    quantize_durations: bool = True,
    quantize_starts: bool = True
) -> FusedTranscription:
    """
    Main fusion algorithm: combine slice-based notes with onset timing.
    
    Args:
        slices: TimeSlice objects from chord_transcription.py
        onset_times: Detected onset times in seconds
        duration: Total duration in seconds
        bpm: Beats per minute
        time_signature_numerator: Top number of time signature
        time_signature_denominator: Bottom number of time signature
        first_beat_time: Time of first downbeat
        source_file: Source audio filename
        min_note_confidence: Minimum confidence to include note
        quantize_durations: Whether to snap durations to musical values
        quantize_starts: Whether to snap start times to beat grid
    
    Returns:
        FusedTranscription with NoteEvent list
    """
    print(f"\n  Fusing transcription with {len(onset_times)} onsets...")
    
    seconds_per_beat = 60.0 / bpm
    
    # Create onset segments
    segments = create_onset_segments(onset_times, duration)
    segments = assign_segment_bar_beats(
        segments, bpm, time_signature_numerator, first_beat_time
    )
    
    print(f"    Created {len(segments)} segments")
    
    # Process each segment
    note_events = []
    prev_segment_notes = {}
    active_notes = {}  # Track notes that are still sounding
    
    for i, segment in enumerate(segments):
        # Cluster notes in this segment by role
        melody_notes = cluster_notes_in_segment(segment, slices, 'melody')
        chord_notes = cluster_notes_in_segment(segment, slices, 'chord')
        bass_notes = cluster_notes_in_segment(segment, slices, 'bass')
        
        all_notes = {}
        for midi, stats in melody_notes.items():
            all_notes[midi] = stats
        for midi, stats in chord_notes.items():
            if midi not in all_notes:
                all_notes[midi] = stats
        for midi, stats in bass_notes.items():
            if midi not in all_notes:
                all_notes[midi] = stats
        
        # For each note in this segment
        for midi, stats in all_notes.items():
            if stats['count'] < 1:
                continue
            
            avg_confidence = stats['total_confidence'] / stats['count']
            if avg_confidence < min_note_confidence:
                continue
            
            # Determine if this is a new note or tied
            is_new, is_tied = determine_note_onset_type(
                midi, segment, prev_segment_notes, all_notes
            )
            
            if is_new or midi not in active_notes:
                # Calculate duration - how long until this note stops appearing
                # Look ahead to find when note disappears
                note_end_time = segment.end_time
                
                for future_seg in segments[i+1:]:
                    future_notes = cluster_notes_in_segment(future_seg, slices, 'all')
                    if midi in future_notes:
                        note_end_time = future_seg.end_time
                    else:
                        break
                
                note_duration = note_end_time - segment.start_time
                
                # Quantize if requested
                start_time = segment.start_time
                if quantize_starts:
                    start_time = quantize_start_to_beat(
                        segment.start_time, bpm, first_beat_time
                    )
                
                if quantize_durations:
                    duration_beats = quantize_duration_to_beats(note_duration, bpm)
                else:
                    duration_beats = note_duration / seconds_per_beat
                
                note_duration = duration_beats * seconds_per_beat
                
                # Calculate bar/beat position
                relative_time = start_time - first_beat_time
                if relative_time < 0:
                    bar = 0
                    beat = 1.0
                else:
                    total_beats = relative_time / seconds_per_beat
                    bar = int(total_beats // time_signature_numerator) + 1
                    beat = (total_beats % time_signature_numerator) + 1
                
                event = NoteEvent(
                    midi_note=midi,
                    note_name=stats['note_name'],
                    start_time=start_time,
                    duration=note_duration,
                    bar=bar,
                    beat=beat,
                    duration_beats=duration_beats,
                    velocity=min(1.0, avg_confidence),
                    role=stats['role'],
                    is_tied=is_tied
                )
                
                note_events.append(event)
                active_notes[midi] = event
        
        # Update previous segment notes
        prev_segment_notes = all_notes
        
        # Remove notes that ended in this segment
        ended_notes = []
        for midi, event in active_notes.items():
            if event.end_time <= segment.end_time:
                ended_notes.append(midi)
        for midi in ended_notes:
            del active_notes[midi]
    
    # Sort by start time, then by pitch
    note_events.sort(key=lambda e: (e.start_time, -e.midi_note))
    
    print(f"    Generated {len(note_events)} note events")
    
    return FusedTranscription(
        duration_seconds=duration,
        source_file=source_file,
        bpm=bpm,
        time_signature=f"{time_signature_numerator}/{time_signature_denominator}",
        first_beat_time=first_beat_time,
        note_events=note_events
    )


# =============================================================================
# Export Functions
# =============================================================================

def export_fused_txt(
    transcription: FusedTranscription,
    output_path: Path,
    group_by_bar: bool = True
) -> None:
    """Export fused transcription to human-readable text."""
    with open(output_path, 'w') as f:
        f.write(f"# Fused Transcription: {transcription.source_file}\n")
        f.write(f"# Duration: {transcription.duration_seconds:.2f}s\n")
        f.write(f"# BPM: {transcription.bpm:.0f}\n")
        f.write(f"# Time Signature: {transcription.time_signature}\n")
        f.write(f"# Total Bars: {transcription.num_bars}\n")
        f.write(f"# Total Notes: {len(transcription.note_events)}\n\n")
        
        if group_by_bar:
            f.write(f"{'Bar':<6} {'Beat':<8} {'Dur':<6} {'Note':<8} {'Role':<8} {'Vel'}\n")
            f.write("-" * 50 + "\n")
            
            current_bar = 0
            for event in transcription.note_events:
                if event.bar != current_bar:
                    if current_bar > 0:
                        f.write("\n")
                    current_bar = event.bar
                
                dur_str = f"{event.duration_beats:.2f}"
                vel_str = f"{event.velocity:.2f}"
                tied_str = " (tie)" if event.is_tied else ""
                
                f.write(f"{event.bar:<6} {event.beat:<8.2f} {dur_str:<6} "
                       f"{event.note_name:<8} {event.role:<8} {vel_str}{tied_str}\n")
        else:
            f.write(f"{'Time':<10} {'Bar.Beat':<10} {'Dur':<8} {'Note':<8} {'Role'}\n")
            f.write("-" * 55 + "\n")
            
            for event in transcription.note_events:
                bar_beat = f"{event.bar}.{event.beat:.2f}"
                f.write(f"{event.start_time:<10.3f} {bar_beat:<10} "
                       f"{event.duration_beats:<8.2f} {event.note_name:<8} {event.role}\n")
    
    print(f"Saved: {output_path}")


def print_fused_summary(transcription: FusedTranscription, max_events: int = 30) -> None:
    """Print summary of fused transcription."""
    print("\n" + "=" * 70)
    print("FUSED TRANSCRIPTION SUMMARY")
    print("=" * 70)
    
    print(f"Duration: {transcription.duration_seconds:.2f}s | "
          f"BPM: {transcription.bpm:.0f} | "
          f"Time Sig: {transcription.time_signature} | "
          f"Bars: {transcription.num_bars}")
    
    melody_count = len(transcription.melody_events)
    chord_count = len(transcription.chord_events)
    bass_count = len(transcription.bass_events)
    
    print(f"Notes - Melody: {melody_count} | Chords: {chord_count} | Bass: {bass_count}")
    print("-" * 70)
    
    print(f"{'Bar':<6} {'Beat':<8} {'Duration':<10} {'Note':<8} {'Role':<8} {'Tied'}")
    print("-" * 70)
    
    for event in transcription.note_events[:max_events]:
        dur_str = f"{event.duration_beats:.2f} beats"
        tied_str = "yes" if event.is_tied else "-"
        
        print(f"{event.bar:<6} {event.beat:<8.2f} {dur_str:<10} "
              f"{event.note_name:<8} {event.role:<8} {tied_str}")
    
    if len(transcription.note_events) > max_events:
        print(f"... and {len(transcription.note_events) - max_events} more events")
    
    print("=" * 70)


# =============================================================================
# Integration with chord_transcription.py
# =============================================================================

def fuse_from_transcription_result(
    transcription,  # TwoPassTranscription from chord_transcription.py
    onsets: list[dict],  # Onset dicts with 'time' and 'strength'
    quantize_durations: bool = True,
    quantize_starts: bool = True,
    apply_harmonic_smoothing: bool = True,
    apply_density_reduction: bool = True,
    apply_playability: bool = True,
    max_chord_notes: int = 4,
    max_total_notes: int = 8,
    max_bass_span: int = MAX_HAND_SPAN_BASS,
    max_treble_span: int = MAX_HAND_SPAN_TREBLE,
    max_notes_per_hand: int = MAX_NOTES_PER_HAND,
    min_presence_ratio: float = 0.4,
    max_bass_notes: int = 2,
    max_melody_notes: int = 2
) -> FusedTranscription:
    """
    Create fused transcription from chord_transcription.py output.
    
    Args:
        transcription: TwoPassTranscription object
        onsets: List of onset dicts from detect_onsets()
        quantize_durations: Snap durations to musical values
        quantize_starts: Snap starts to beat grid
        apply_harmonic_smoothing: Filter notes for harmonic coherence
        apply_density_reduction: Limit simultaneous notes
        apply_playability: Apply hand span, majority vote, and role filters
        max_chord_notes: Max chord notes per segment
        max_total_notes: Max total notes per segment
        max_bass_span: Max semitones for left hand (default 10)
        max_treble_span: Max semitones for right hand (default 10)
        max_notes_per_hand: Max notes per hand (default 5)
        min_presence_ratio: Min slice presence for majority vote
        max_bass_notes: Max bass notes per time slice (default 2)
        max_melody_notes: Max melody notes per time slice (default 2)
    
    Returns:
        FusedTranscription
    """
    onset_times = [o['time'] for o in onsets]
    
    fused = fuse_transcription_with_onsets(
        slices=transcription.slices,
        onset_times=onset_times,
        duration=transcription.duration_seconds,
        bpm=transcription.timing.bpm,
        time_signature_numerator=transcription.timing.time_signature_numerator,
        time_signature_denominator=transcription.timing.time_signature_denominator,
        first_beat_time=transcription.timing.first_beat_time,
        source_file=transcription.source_file,
        quantize_durations=quantize_durations,
        quantize_starts=quantize_starts
    )
    
    # Apply playability filters FIRST (majority vote uses slice data)
    if apply_playability:
        fused.note_events = apply_playability_filters(
            fused.note_events,
            slices=transcription.slices,
            apply_hand_span=True,
            apply_majority_vote=True,
            apply_role_rebalance=True,
            max_bass_span=max_bass_span,
            max_treble_span=max_treble_span,
            max_notes_per_hand=max_notes_per_hand,
            min_presence_ratio=min_presence_ratio,
            max_bass_notes=max_bass_notes,
            max_chord_notes=max_chord_notes,
            max_melody_notes=max_melody_notes
        )
    
    # Apply harmonic smoothing
    if apply_harmonic_smoothing:
        fused.note_events = smooth_notes_by_harmony(
            fused.note_events,
            segments_per_bar=4,
            min_notes_for_chord=3,
            dissonance_threshold=0.5,
            remove_isolated=True,
            min_duration_beats=0.25
        )
    
    # Apply density reduction (final cleanup)
    if apply_density_reduction:
        fused.note_events = reduce_chord_density(
            fused.note_events,
            max_chord_notes=max_chord_notes,
            max_total_notes=max_total_notes
        )
    
    return fused