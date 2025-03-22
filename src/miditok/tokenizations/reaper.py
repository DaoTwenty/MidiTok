"""REAPer (Revamped Expressive And Performed) tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from symusic import (
    Note,
    Pedal,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

from miditok.classes import Event, TokenizerConfig, TokSequence
from miditok.constants import (
    ADD_TRAILING_BARS,
    DEFAULT_VELOCITY,
    MIDI_INSTRUMENTS,
    TIME_SIGNATURE,
    USE_BAR_END_TOKENS,
)
from miditok.midi_tokenizer import MusicTokenizer
from miditok.utils import compute_ticks_per_bar, compute_ticks_per_beat

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from miditok.tokenizations import REMI

class REAP(REMI):

    r"""
    
    REAPer (Revamped Expressive And Performed) tokenizer.

    Introducted in `MIDI-GPT (Pasquier et al.) <https://arxiv.org/abs/2501.17011>`_,
    this tokenization extends the REMI tokenization by adding microtiming to 
    *Position* for more timewise precision allowing to capture expressiveness 
    and performance.

    New tokens include *Delta* which encodes the absolute difference between the raw 
    time position and the downsampled *Postion* value, and *DeltaDirection* whose presence 
    indicated that the difference is negative.

    New TokenizerConfig Options:

    * microtiming_factor: indicates the factor of added resolution with the *Delta* tokens
    
    **Note:** To achieve the tokenization from the paper, this class must be used within
    the MMM tokenization.

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.classes.TokenizerConfig` object.
        REAPer accepts additional_params for tokenizer configuration:
        | - max_bar_embedding (desciption below);
        | - use_bar_end_tokens -- if set will add Bar_End tokens at the end of each Bar;
        | - add_trailing_bars -- will add tokens for trailing empty Bars, if they are
        | - microtiming_factor -- factor of added resolution with the *Delta* tokens
        present in source symbolic music data. Applicable to :ref:`REAPer`. This flag is
        very useful in applications where we need bijection between Bars is source and
        tokenized representations, same lengths, anacrusis detection etc.
        False by default, thus trailing bars are omitted.
    :param max_bar_embedding: Maximum number of bars ("Bar_0", "Bar_1",...,
        "Bar_{num_bars-1}"). If None passed, creates "Bar_None" token only in
        vocabulary for Bar token. Has less priority than setting this parameter via
        TokenizerConfig additional_params
    :param params: path to a tokenizer config file. This will override other arguments
        and load the tokenizer based on the config file. This is particularly useful
        if the tokenizer learned Byte Pair Encoding. (default: None)
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig = None,
        max_bar_embedding: int | None = None,
        params: str | Path | None = None,
    ) -> None:
        super().__init__(tokenizer_config, max_bar_embedding, params)
        if "use_microtiming" not in self.config.additional_params:
            msg = "Tokenizer config must have a value for use_microtiming"
            raise ValueError(msg)
        if "microtiming_factor" not in self.config.additional_params:
            msg = "Tokenizer config must have a value for microtiming_factor"
            raise ValueError(msg)
        self.use_microtiming = bool(self.config.additional_params["use_microtiming"])
        self.df = int(self.config.additional_params["microtiming_factor"])

        # Workaround to not downsample and loose microtiming information
        # Resampling is done later when time events are added
        for range, res in self.config.beat_res.items():
            self.config.beat_res[range] = self.df * res

    def _add_position_event(
        self,
        event: Event,
        all_events: list[Event],
        tick_at_current_bar: int,
        ticks_per_pos: int,
    ) -> None:
        pos_index = self._units_between(
            tick_at_current_bar // self.df, 
            event.time // self.df,
            ticks_per_pos // self.df
        )
        all_events.append(
            Event(
                type_="Position",
                value=pos_index,
                time=event.time,
                desc=event.time,
            )
        )

        if self.use_microtiming:
            self._add_delta_event(event, all_events, tick_at_current_bar, ticks_per_pos)
        
    def _add_delta_event(
        self,
        event: Event,
        all_events: list[Event],
        tick_at_current_bar: int,
        ticks_per_pos: int,
    ) -> None:
        pos_index = self._units_between(tick_at_current_bar, event.time, ticks_per_pos)
        pos_index_time = pos_index * ticks_per_pos
        delta = pos_index_time - event.time
        if delta < 0:
            all_events.append(
                Event(
                    type_="DeltaDirection",
                    value=0,
                    time=event.time,
                    desc=event.time,
                )
            )
        all_events.append(
            Event(
                type_="Delta",
                value=abs(delta),
                time=event.time,
                desc=event.time,
            )
        )

    def _create_base_vocabulary(self) -> list[str]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MusicTokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MusicTokenizer`.

        **Attribute control tokens are added when creating the tokenizer by the**
        ``MusicTokenizer.add_attribute_control`` **method.**

        :return: the vocabulary as a list of string.
        """
        vocab = super()._create_base_vocabulary()
        
        if not self.use_microtiming:
            return vocab
        
        # Microtiming
        vocab += [
            f"Delta_{i}" for i in range(1,self.delta_range)
        ]
        vocab.append("DeltaDirection_0")

        return vocab

    @property
    def delta_range(self) -> list[int]:
        """
        Returns the max range of *Delta* tokens

        :return: range of the microtiming tokens
        """
        return self.config.max_num_pos_per_beat
    
    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic = super(). _create_token_types_graph()

        if not self.use_microtiming:
            return dic
        
        # Microtiming
        dic["Position"].add("Delta")
        dic["Position"].add("DeltaDirection")
        dic["DeltaPosition"] = {"Delta"}

        return dic
    
    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        r"""
        Convert tokens (:class:`miditok.TokSequence`) into a ``symusic.Score``.

        This is an internal method called by ``self.decode``, intended to be
        implemented by classes inheriting :class:`miditok.MusicTokenizer`.

        :param tokens: tokens to convert. Can be either a list of
            :class:`miditok.TokSequence` or a list of :class:`miditok.TokSequence`s.
        :param programs: programs of the tracks. If none is given, will default to
            piano, program 0. (default: ``None``)
        :return: the ``symusic.Score`` object.
        """
        # Unsqueeze tokens in case of one_token_stream
        if self.config.one_token_stream_for_programs:  # ie single token seq
            tokens = [tokens]
        for i, tokens_i in enumerate(tokens):
            tokens[i] = tokens_i.tokens
        score = Score(self.time_division)
        dur_offset = 2 if self.config.use_velocities else 1

        # RESULTS
        tracks: dict[int, Track] = {}
        tempo_changes, time_signature_changes = [], []

        def check_inst(prog: int) -> None:
            if prog not in tracks:
                tracks[prog] = Track(
                    program=0 if prog == -1 else prog,
                    is_drum=prog == -1,
                    name="Drums" if prog == -1 else MIDI_INSTRUMENTS[prog]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return (
                len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0
            )

        current_track = None
        for si, seq in enumerate(tokens):
            # First look for the first time signature if needed
            if si == 0:
                if self.config.use_time_signatures:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type == "TimeSig":
                            time_signature_changes.append(
                                TimeSignature(
                                    0, *self._parse_token_time_signature(tok_val)
                                )
                            )
                            break
                        if tok_type in [
                            "Pitch",
                            "PitchDrum",
                            "Velocity",
                            "Duration",
                            "PitchBend",
                            "Pedal",
                        ]:
                            break
                if len(time_signature_changes) == 0:
                    time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))
            current_time_sig = time_signature_changes[-1]
            ticks_per_bar = compute_ticks_per_bar(
                current_time_sig, score.ticks_per_quarter
            )
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)

            # Set tracking variables
            current_tick = tick_at_last_ts_change = tick_at_current_bar = 0
            current_bar = -1
            bar_at_last_ts_change = 0
            current_program = 0
            previous_note_end = 0
            current_delta_direction = 1
            previous_pitch_onset = {prog: -128 for prog in self.config.programs}
            previous_pitch_chord = {prog: -128 for prog in self.config.programs}
            active_pedals = {}

            # Set track / sequence program if needed
            if not self.config.one_token_stream_for_programs:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[si]
                elif self.config.use_programs:
                    for token in seq:
                        tok_type, tok_val = token.split("_")
                        if tok_type.startswith("Program"):
                            current_program = int(tok_val)
                            if current_program == -1:
                                is_drum, current_program = True, 0
                            break
                current_track = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums"
                    if current_program == -1
                    else MIDI_INSTRUMENTS[current_program]["name"],
                )
            current_track_use_duration = (
                current_program in self.config.use_note_duration_programs
            )

            # Decode tokens
            for ti, token in enumerate(seq):
                tok_type, tok_val = token.split("_")
                if token == "Bar_None":
                    current_bar += 1
                    if current_bar > 0:
                        current_tick = tick_at_current_bar + ticks_per_bar
                    tick_at_current_bar = current_tick
                elif tok_type == "Rest":
                    current_tick = max(previous_note_end, current_tick)
                    current_tick += self._tpb_rests_to_ticks[ticks_per_beat][tok_val]
                    real_current_bar = bar_at_last_ts_change + self._units_between(
                        tick_at_last_ts_change, current_tick, ticks_per_bar
                    )
                    if real_current_bar > current_bar:
                        # In case we instantly begin with a Rest,
                        # we need to update current_bar
                        if current_bar == -1:
                            current_bar = 0
                        tick_at_current_bar += (
                            real_current_bar - current_bar
                        ) * ticks_per_bar
                        current_bar = real_current_bar
                elif tok_type == "Position":
                    if current_bar == -1:
                        # as this Position token occurs before any Bar token
                        current_bar = 0
                    current_tick = tick_at_current_bar + int(tok_val) * ticks_per_pos
                    current_delta_direction = 1
                elif tok_type == "DeltaDirection":
                    current_delta_direction = -1
                elif tok_type == "Delta":
                    delta_tick = current_delta_direction * int(tok_val)
                    current_tick += delta_tick
                elif tok_type in {
                    "Pitch",
                    "PitchDrum",
                    "PitchIntervalTime",
                    "PitchIntervalChord",
                }:
                    if tok_type in {"Pitch", "PitchDrum"}:
                        pitch = int(tok_val)
                    elif tok_type == "PitchIntervalTime":
                        pitch = previous_pitch_onset[current_program] + int(tok_val)
                    else:  # PitchIntervalChord
                        pitch = previous_pitch_chord[current_program] + int(tok_val)
                    if (
                        not self.config.pitch_range[0]
                        <= pitch
                        <= self.config.pitch_range[1]
                    ):
                        continue

                    # We update previous_pitch_onset and previous_pitch_chord even if
                    # the try fails.
                    if tok_type != "PitchIntervalChord":
                        previous_pitch_onset[current_program] = pitch
                    previous_pitch_chord[current_program] = pitch

                    try:
                        if self.config.use_velocities:
                            vel_type, vel = seq[ti + 1].split("_")
                        else:
                            vel_type, vel = "Velocity", DEFAULT_VELOCITY
                        if current_track_use_duration:
                            dur_type, dur = seq[ti + dur_offset].split("_")
                        else:
                            dur_type = "Duration"
                            dur = int(
                                self.config.default_note_duration * ticks_per_beat
                            )
                        if vel_type == "Velocity" and dur_type == "Duration":
                            if isinstance(dur, str):
                                dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                            new_note = Note(
                                current_tick,
                                dur,
                                pitch,
                                int(vel),
                            )
                            if self.config.one_token_stream_for_programs:
                                check_inst(current_program)
                                tracks[current_program].notes.append(new_note)
                            else:
                                current_track.notes.append(new_note)
                            previous_note_end = max(
                                previous_note_end, current_tick + dur
                            )
                    except IndexError:
                        # A well constituted sequence should not raise an exception
                        # However with generated sequences this can happen, or if the
                        # sequence isn't finished
                        pass
                elif tok_type == "Program":
                    current_program = int(tok_val)
                    current_track_use_duration = (
                        current_program in self.config.use_note_duration_programs
                    )
                    if (
                        not self.config.one_token_stream_for_programs
                        and self.config.program_changes
                    ):
                        if current_program != -1:
                            current_track.program = current_program
                        else:
                            current_track.program = 0
                            current_track.is_drum = True
                elif tok_type == "Tempo":
                    if si == 0:
                        tempo_changes.append(Tempo(current_tick, float(tok_val)))
                    previous_note_end = max(previous_note_end, current_tick)
                elif tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    if (
                        num != current_time_sig.numerator
                        or den != current_time_sig.denominator
                    ):
                        current_time_sig = TimeSignature(current_tick, num, den)
                        if si == 0:
                            time_signature_changes.append(current_time_sig)
                        tick_at_last_ts_change = tick_at_current_bar  # == current_tick
                        bar_at_last_ts_change = current_bar
                        ticks_per_bar = compute_ticks_per_bar(
                            current_time_sig, score.ticks_per_quarter
                        )
                        ticks_per_beat = self._tpb_per_ts[den]
                        ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)
                elif tok_type == "Pedal":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if self.config.sustain_pedal_duration and ti + 1 < len(seq):
                        if seq[ti + 1].split("_")[0] == "Duration":
                            duration = self._tpb_tokens_to_ticks[ticks_per_beat][
                                seq[ti + 1].split("_")[1]
                            ]
                            # Add instrument if it doesn't exist, can happen for the
                            # first tokens
                            new_pedal = Pedal(current_tick, duration)
                            if self.config.one_token_stream_for_programs:
                                check_inst(pedal_prog)
                                tracks[pedal_prog].pedals.append(new_pedal)
                            else:
                                current_track.pedals.append(new_pedal)
                    elif pedal_prog not in active_pedals:
                        active_pedals[pedal_prog] = current_tick
                elif tok_type == "PedalOff":
                    pedal_prog = (
                        int(tok_val) if self.config.use_programs else current_program
                    )
                    if pedal_prog in active_pedals:
                        new_pedal = Pedal(
                            active_pedals[pedal_prog],
                            current_tick - active_pedals[pedal_prog],
                        )
                        if self.config.one_token_stream_for_programs:
                            check_inst(pedal_prog)
                            tracks[pedal_prog].pedals.append(new_pedal)
                        else:
                            current_track.pedals.append(new_pedal)
                        del active_pedals[pedal_prog]
                elif tok_type == "PitchBend":
                    new_pitch_bend = PitchBend(current_tick, int(tok_val))
                    if self.config.one_token_stream_for_programs:
                        check_inst(current_program)
                        tracks[current_program].pitch_bends.append(new_pitch_bend)
                    else:
                        current_track.pitch_bends.append(new_pitch_bend)

            # Add current_inst to score and handle notes still active
            if not self.config.one_token_stream_for_programs and not is_track_empty(
                current_track
            ):
                score.tracks.append(current_track)

        # Add global events to the score
        if self.config.one_token_stream_for_programs:
            score.tracks = list(tracks.values())
        score.tempos = tempo_changes
        score.time_signatures = time_signature_changes

        return score