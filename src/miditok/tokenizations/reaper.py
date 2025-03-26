"""REAPer (Revamped Expressive And Performed) tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from math import ceil
import numpy as np

from symusic import (
    Note,
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
    QUARTER_NOTE_RES,
    MICROTIMING_FACTOR,
    USE_MICROTIMING_REAPER,
    USE_DUR_MICROTIMING_REAPER,
    SIGNED_MICROTIMING,
    JOINT_MICROTIMING,
)

from miditok.attribute_controls import (
    BarAttributeControl,
)

from miditok.utils import (
    compute_ticks_per_bar,
    compute_ticks_per_beat,
)

from miditok.midi_tokenizer import MusicTokenizer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from symusic.score import TimeSignatureTickList

from miditok.tokenizations import REMI

class REAPER(REMI):

    r"""
    
    REAPer (Revamped Expressive And Performed) tokenizer.

    Introducted in `MIDI-GPT (Pasquier et al.) <https://arxiv.org/abs/2501.17011>`_,
    this tokenization extends the REMI tokenization by adding microtiming to 
    *Position* for more timewise precision allowing to capture expressiveness 
    and performance.

    New tokens include *Delta* which encodes the absolute difference between the raw 
    time position and the downsampled *Postion* value, and *DeltaDirection* whose presence 
    indicated that the difference is negative.

    Additionally, *Duration* encode time identtically to *Position* tokens. The resolution
    for both these tokens is based on quarter-notes, and not beats like in REMI.

    New TokenizerConfig Options:

    * use_microtiming: indicates whether microtiming tokens are used for position
    * dur_use_microtiming: indicates whether microtiming tokens are used for duration
    * microtiming_factor: indicates the factor of added resolution with the positional microtiming tokens
    * microtiming_factor: indicates the factor of added resolution with the durational microtiming tokens
    * quarter_note_res: replaces beat_res, resolution is relative to quarter-notes
    * dur_quarter_note_res: same as quarter_note_res, but for durations
    * signed_microtiming: if True, *Delta* values can be negative. If false, a direction token is use
    * joint_microtiming: indicates if a singe *Delta* token is used for both position and duration
    
    **Note:** To achieve the tokenization from the paper, this class must be used within
    the MMM tokenization.

    :param tokenizer_config: the tokenizer's configuration, as a
        :class:`miditok.classes.TokenizerConfig` object.
        REAPer accepts additional_params for tokenizer configuration:
        | - max_bar_embedding (desciption below);
        | - use_bar_end_tokens -- if set will add Bar_End tokens at the end of each Bar;
        | - add_trailing_bars -- will add tokens for trailing empty Bars, if they are
        | - microtiming_factor -- factor of added resolution with the *Delta* tokens
        | - quarter_note_res -- positions per quarter-note
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

        if self.use_microtiming:
            self._tpb_per_ts = self.__create_tpb_per_ts()
            self._tpb_to_time_array = self.__create_tpb_to_ticks_array()
            self._tpb_tokens_to_ticks = self.__create_tpb_tokens_to_ticks()
            self._tpb_ticks_to_tokens = self.__create_tpb_ticks_to_tokens()

    # Methods to override base MusicTokenizer versions
    # To handle MicroTiming and multiple beat_res resolutions
    # This is accomplished by removing the downsampling methods
    # As a result, many time-based methods need to be redesigned

    def _tweak_config_before_creating_voc(self) -> None:
        super()._tweak_config_before_creating_voc()

        if "use_microtiming" not in self.config.additional_params:
            self.config.additional_params["use_microtiming"] = USE_MICROTIMING_REAPER
        if "use_dur_microtiming" not in self.config.additional_params:
            self.config.additional_params["use_dur_microtiming"] = USE_DUR_MICROTIMING_REAPER
        if "microtiming_factor" not in self.config.additional_params:
            self.config.additional_params["microtiming_factor"] = MICROTIMING_FACTOR
        if "quarter_note_res" not in self.config.additional_params:
            self.config.additional_params["quarter_note_res"] = QUARTER_NOTE_RES
        if "dur_quarter_note_res" not in self.config.additional_params:
            res = self.config.additional_params["quarter_note_res"]
            self.config.additional_params["dur_quarter_note_res"] = res
        if "dur_microtiming_factor" not in self.config.additional_params:
            factor = self.config.additional_params["microtiming_factor"]
            self.config.additional_params["dur_microtiming_factor"] = factor
        if "signed_microtiming" not in self.config.additional_params:
            self.config.additional_params["signed_microtiming"] = SIGNED_MICROTIMING
        if "joint_microtiming" not in self.config.additional_params:
            self.config.additional_params["joint_microtiming"] = JOINT_MICROTIMING
        else:
            pos_factor = self.config.additional_params["microtiming_factor"]
            dur_factor = self.config.additional_params["dur_microtiming_factor"]
            if (dur_factor > pos_factor) or (pos_factor % dur_factor != 0):
                msg = """Positional microtiming factor must be divisible by 
                    durational microtiming factor"""
                raise ValueError(msg)
        if (
            self.res_dur > self.res_pos or 
            self.res_pos % self.res_dur != 0
        ):
                msg = """note position resolution must me divisible
                by the note duration resolution"""
                raise ValueError(msg)


        self.use_microtiming = self.config.additional_params["use_microtiming"] 
        self.use_dur_microtiming = self.config.additional_params["use_dur_microtiming"]
        # Implementation requires positional microtiming in order to use 
        # durational microtiming. This condition can be bypassed by removing
        # the following line
        self.use_dur_microtiming &= self.use_microtiming
        self.delta_factor = self.config.additional_params["microtiming_factor"]
        self.dur_delta_factor = self.config.additional_params["dur_microtiming_factor"]
        self.dur_to_pos_delta_factor = self.delta_factor // self.dur_delta_factor
        self.signed_microtiming = self.config.additional_params["signed_microtiming"]
        self.joint_microtiming = self.config.additional_params["joint_microtiming"]

        self.dur_delta_token = "Delta"
        if not self.joint_microtiming:
            self.dur_delta_token = "DurationDelta"

        # We don't need these, may change later
        self.config.use_chords = False
        self.config.use_pitch_bends = False
        self.config.use_pitch_intervals = False
        self.config.use_sustain_pedals = False
        self.config.use_rests = False

    @property
    def longest_ts_qn(self) ->tuple[int]:
        """
        Returns the longest time signature in quarter-notes.

        :return: longest time signature in quarter-notes.
        """
        ts_lengths = np.array([ts[0]/ts[1] for ts in self.time_signatures])
        return self.time_signatures[np.argmax(ts_lengths)]

    @property
    def max_quarter_notes(self) -> int:
        """
        Returns the maximum number of quarter-note per bar.

        :return: maximum number of quarter-note per bar.
        """
        longest_ts = self.longest_ts_qn
        return ceil(4 * longest_ts[0]/longest_ts[1])

    @property
    def res_pos(self) -> int:
        """
        Returns the maximum number of positions per quarter-note covered by the config.

        :return: maximum number of positions per quarter-note covered by the config.
        """
        return max(self.config.additional_params["quarter_note_res"].values())
    
    @property
    def res_dur(self) -> int:
        """
        Returns the maximum number of durations per quarter-note covered by the config.

        :return: maximum number of durations per quarter-note covered by the config.
        """
        return max(self.config.additional_params["dur_quarter_note_res"].values())

    @property
    def tpq(self) -> int:
        """
        Returns the resolution at which score is resampled. 
        Is also resolution of *Delta* tokens.

        :return: resampling resolution.
        """
        return self.delta_factor * self.res_pos
    
    @property
    def dur_tpq(self) -> int:
        """
        Returns the resolution of *DurationDelta* tokens.

        :return: duration delta tokens resolution.
        """
        return self.dur_delta_factor * self.res_pos
    
    @property
    def max_pos(self) -> int:
        """
        Returns the max *Position* value.

        :return: max position token value.
        """
        return self.res_pos * self.max_quarter_notes
    
    @property
    def max_dur(self) -> int:
        """
        Returns the max *Duration* value.

        :return: max duration token value.
        """
        return self.res_dur * self.max_quarter_notes
    
    @property
    def delta_range(self) -> int:
        """
        Returns the max range of *Delta* tokens

        :return: range of the microtiming tokens
        """
        return self.delta_factor // 2
    
    @property
    def dur_delta_range(self) -> int:
        """
        Returns the max range of *DurationDelta* tokens

        :return: range of the duration microtiming tokens
        """
        return self.dur_delta_factor // 2

    def _resample_score(
        self, score: Score, _new_tpq: int, _time_signatures_copy: TimeSignatureTickList
    ) -> Score:
        tpq_to_resample = _new_tpq
        if self.use_microtiming:
            tpq_to_resample = self.tpq

        return super()._resample_score(
            score, 
            tpq_to_resample,
            _time_signatures_copy
        )
    
    # For position token value, we want to round to the nearest integer,
    # not round to the integer closest to zero.
    # Because we may be rounding up at the last possible position, therefore
    # rounding to what sould be Position_0 of the next bar, we max at max_pos
    
    def _units_between_pos(
            self, 
            start_tick: int, 
            end_tick: int, 
            ticks_per_unit: int
        ) -> int:
        return min(
                    int((end_tick - start_tick) / ticks_per_unit + 0.5), 
                    self.max_pos
                )
    
    def _units_dur(self, duration_tick: int, ticks_per_unit: int) -> int:
        return max(
            1,
            min(
                int(duration_tick / ticks_per_unit + 0.5), 
                self.max_dur - 1
            )
        )
    
    def _create_duration_event(
        self, note: Note, _program: int, _ticks_per_pos: np.ndarray
    ) -> list[Event]:
        res = []
        dur = self._units_dur(
            note.duration,
            _ticks_per_pos
        )
        res.append(Event(
                type_="Duration",
                value=dur,
                time=note.start,
                program=_program,
                desc=f"{note.duration} ticks",
            )
        )
        if not self.use_dur_microtiming:
            return res
        
        delta_dir, delta_tok = self._create_dur_delta_event(
            note.start,
            note.duration,
            dur * _ticks_per_pos
        )
        if delta_dir:
            res.append(delta_dir)
        if delta_tok:
            res.append(delta_tok)
        
        return res

    def _add_position_event(
        self,
        event: Event,
        all_events: list[Event],
        tick_at_current_bar: int,
        ticks_per_pos: int,
    ) -> None:
        pos_index = self._units_between_pos(
            tick_at_current_bar, 
            event.time,
            ticks_per_pos 
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
            self._add_delta_event(
                event,
                event.time - tick_at_current_bar, 
                pos_index * ticks_per_pos, 
                all_events
            )
        
    def _add_delta_event(
        self,
        event: Event,
        tick_event: int,
        tick_pos: int,
        all_events: list[Event],
    ) -> None:
        delta = tick_event - tick_pos
        if delta !=0:
            if delta < 0 and not self.signed_microtiming:
                all_events.append(
                    Event(
                        type_="DeltaDirection",
                        value=0,
                        time=event.time,
                        desc="Negative Delta",
                    )
                )
            if not self.signed_microtiming:
                delta = min(abs(delta),self.delta_range)
            else:
                delta = max(min(delta,self.delta_range), -self.delta_range)
            all_events.append(
                Event(
                    type_="Delta",
                    value=delta,
                    time=event.time,
                    desc=f"Delta {delta}",
                )
            )

    def _create_dur_delta_event(
        self,
        time: int,
        tick_event: int,
        tick_dur: int,
    ) -> tuple[Event]:
        # Downsampling to the durational microtiming resolution
        delta_tok = None
        delta_dir = None
        delta = (tick_event - tick_dur) // self.dur_to_pos_delta_factor * self.dur_to_pos_delta_factor
        if delta !=0:
            if delta < 0 and not self.signed_microtiming:
                delta_dir = Event(
                    type_=f"{self.dur_delta_token}Direction",
                    value=0,
                    time=time,
                    desc="Negative Duration Delta",
                )
            if not self.signed_microtiming:
                delta = min(abs(delta),self.dur_delta_range)
            else:
                delta = max(min(delta,self.dur_delta_range), -self.dur_delta_range)
            delta_tok = Event(
                type_=self.dur_delta_token,
                value=delta,
                time=time,
                desc=f"Delta Duration {delta}",
            )

        return delta_dir, delta_tok

    def __create_tpb_per_ts(self) -> dict[int, int]:
        """
        Return the dictionary of the possible ticks per beat values per time signature.

        The set of possible values depends on the tokenizer's maximum number of
        positions per beat (`self.config.max_num_pos_per_beat`) and the time signatures
        it supports.

        :return: dictionary of the possible ticks per beat values per time signature,
            keys are time signature denominators, values the ticks/beat values.
        """
        max_denom = max(ts[1] for ts in self.time_signatures)
        return {
            denom: self.delta_factor * self.config.max_num_pos_per_beat * (max_denom // denom)
            for denom in self.config.time_signature_range
        }
    
    def __create_tpb_to_ticks_array(self, rest: bool = False) -> dict[int, np.ndarray]:
        r"""
        Create arrays of the times in ticks of the time tokens of the vocabulary.

        These time values following the ticks/beat value, which depends on the time
        signature.

        The arrays are used during tokenization to efficiently find the closest values.

        :param rest: will use rest values if given ``True``, otherwise durations.
            (default: ``False``)
        :return: dictionary of the durations in tick depending on the ticks per beat
            resolution.
        """
        values = self.rests if rest else self.durations
        return {
            tpb: np.array(
                [self._time_token_to_ticks(time_tuple, tpb) for time_tuple in values],
                dtype=np.intc,
            )
            for tpb in self._tpb_per_ts.values()
        }
    
    def __create_tpb_ticks_to_tokens(self) -> dict[int, dict[int, str]]:
        r"""
        Create the correspondences between times in tick and token value (str).

        These correspondences vary following the ticks/beat value, which depends on the
        time signature.

        The returned dictionary is used during tokenization to get the values of
        *Duration*/*TimeShift*/*Rest* tokens while taking the time signature into
        account.

        :return: ticks per beat + duration in ticks to token value.
        """
        return {
            tpb: {v: k for k, v in tokens_to_ticks.items()}
            for tpb, tokens_to_ticks in self._tpb_tokens_to_ticks.items()
        }
    
    def __create_tpb_tokens_to_ticks(
        self, rest: bool = False
    ) -> dict[int, dict[str, int]]:
        r"""
        Create the correspondences between times in tick and token value (str).

        These correspondences vary following the ticks/beat value, which depends on the
        time signature.

        The returned dictionary is used when decoding *Duration*/*TimeShift*/*Rest*
        tokens while taking the time signature into account.

        :param rest: will use rest values if given ``True``, otherwise durations.
            (default: ``False``)
        :return: ticks per beat + token value to duration in tick.
        """
        values = self.rests if rest else self.durations
        return {
            tpb: {
                ".".join(map(str, duration_tuple)): self._time_token_to_ticks(
                    duration_tuple, tpb
                )
                for duration_tuple in values
            }
            for tpb in self._tpb_per_ts.values()
        }

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
        vocab = []

        # Bar
        if self.config.additional_params["max_bar_embedding"] is not None:
            vocab += [
                f"Bar_{i}"
                for i in range(self.config.additional_params["max_bar_embedding"])
            ]
        else:
            vocab += ["Bar_None"]
        if self.config.additional_params["use_bar_end_tokens"]:
            vocab.append("Bar_End")

        # NoteOn/NoteOff/Velocity
        self._add_note_tokens_to_vocab_list(vocab)

        # Position
        # self.time_division is equal to the maximum possible ticks/beat value.
        vocab += [f"Position_{i}" for i in range(self.max_pos)]

        # Add additional tokens
        self._add_additional_tokens_to_vocab_list(vocab)
        
        if not self.use_microtiming:
            return vocab
        
        # Microtiming
        vocab += [
            f"Delta_{i}" for i in range(1,self.delta_range+1)
        ]
        if self.signed_microtiming:
            vocab += [
                f"Delta_{i}" for i in range(-self.delta_range-1,0)
            ]
        else:
            vocab.append("DeltaDirection_0")

        if self.use_dur_microtiming and not self.joint_microtiming:
            vocab += [
                f"DurationDelta_{i}" for i in range(1,self.dur_delta_range+1)
            ]
            if self.signed_microtiming:
                vocab += [
                    f"DurationDelta_{i}" for i in range(-self.dur_delta_range-1,0)
                ]
            else:
                vocab.append("DurationDeltaDirection_0")

        return vocab
    
    def _add_note_tokens_to_vocab_list(self, vocab: list[str]) -> None:
        # NoteOn + NoteOff
        if self._note_on_off:
            vocab += [f"NoteOn_{i}" for i in range(*self.config.pitch_range)]
            if len(self.config.use_note_duration_programs) > 0:
                vocab += [f"NoteOff_{i}" for i in range(*self.config.pitch_range)]
        # Pitch + Duration (later done after velocity)
        else:
            vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # Velocity
        if self.config.use_velocities:
            vocab += [f"Velocity_{i}" for i in self.velocities]

        # Duration
        if (
            not self._note_on_off and self.config.using_note_duration_tokens
        ) or self.config.sustain_pedal_duration:
            vocab += [
                f"Duration_{duration}"
                for duration in range(1, self.max_dur)
            ]
    
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

        if self.use_dur_microtiming:
            dic["Duration"].add(self.dur_delta_token)
            dic["Duration"].add(f"{self.dur_delta_token}Direction")
            dic[f"{self.dur_delta_token}Direction"] = {self.dur_delta_token}

        return dic
    
    def _compute_ticks_per_units(
        self, time: int, current_time_sig: Sequence[int], time_division: int
    ) -> tuple[int, int, int]:
        ticks_per_bar = compute_ticks_per_bar(
            TimeSignature(time, *current_time_sig), time_division
        )
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = time_division // self.res_pos
        return ticks_per_bar, ticks_per_beat, ticks_per_pos
    
    def _add_time_events(self, events: list[Event], time_division: int) -> list[Event]:
        r"""
        Create the time events from a list of global and track events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """
        # Add time events
        all_events = []
        current_bar = -1
        bar_at_last_ts_change = 0
        previous_tick = -1
        previous_note_end = 0
        tick_at_last_ts_change = tick_at_current_bar = 0

        # Determine time signature and compute ticks per entites
        current_time_sig, time_sig_time = TIME_SIGNATURE, 0

        # First look for a TimeSig token, if any is given at tick 0, to update
        # current_time_sig
        if self.config.use_time_signatures:
            for event in events:
                # There should be a TimeSig token at tick 0
                if event.type_ == "TimeSig":
                    current_time_sig = self._parse_token_time_signature(event.value)
                    time_sig_time = event.time
                    break
        ticks_per_bar, ticks_per_beat, ticks_per_pos = self._compute_ticks_per_units(
            time_sig_time, current_time_sig, time_division
        )
        ticks_per_qn = time_division

        # Add the time events
        for ei, event in enumerate(events):
            if event.type_.startswith("ACTrack"):
                all_events.append(event)
                continue
            if event.time != previous_tick:
                        
                # Bar
                current_bar, tick_at_current_bar = self._add_new_bars(
                    event.time,
                    event.type_,
                    all_events,
                    current_bar,
                    bar_at_last_ts_change,
                    tick_at_last_ts_change,
                    tick_at_current_bar,
                    current_time_sig,
                    ticks_per_bar,
                )

                # Position
                if event.type_ != "TimeSig" and not event.type_.startswith("ACBar"):
                    self._add_position_event(
                        event, all_events, tick_at_current_bar, ticks_per_pos
                    )

                previous_tick = event.time

            # Update time signature time variables, after adjusting the time (above)
            if event.type_ == "TimeSig":
                bar_at_last_ts_change += self._units_between(
                    tick_at_last_ts_change, event.time, ticks_per_bar
                )
                tick_at_last_ts_change = event.time
                current_time_sig = self._parse_token_time_signature(event.value)
                ticks_per_bar, ticks_per_beat, ticks_per_pos = (
                    self._compute_ticks_per_units(
                        event.time, current_time_sig, time_division
                    )
                )
                # We decrease the previous tick so that a Position token is enforced
                # for the next event
                previous_tick -= 1

            all_events.append(event)
            # Adds a Position token if the current event is a bar-level attribute
            # control and the next one is at the same position, as the position token
            # wasn't added previously.
            if (
                event.type_.startswith("ACBar")
                and not events[ei + 1].type_.startswith("ACBar")
                and event.time == events[ei + 1].time
            ):
                self._add_position_event(
                    event, all_events, tick_at_current_bar, ticks_per_pos
                )

            # Update max offset time of the notes encountered
            previous_note_end = self._previous_note_end_update(event, previous_note_end)

        if (
            previous_note_end > previous_tick
            and self.config.additional_params["add_trailing_bars"]
        ):
            # there are some trailing bars
            _ = self._add_new_bars(
                previous_note_end,
                event.type_,
                all_events,
                current_bar,
                bar_at_last_ts_change,
                tick_at_last_ts_change,
                tick_at_current_bar,
                current_time_sig,
                ticks_per_bar,
            )
        return all_events
    
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
        score = Score(self.tpq)
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
            ticks_per_pos = self.delta_factor
            ticks_per_dur = self.tpq // self.res_dur

            # Set tracking variables
            current_tick = tick_at_current_bar = 0
            current_bar = -1
            current_program = 0
            previous_note_end = 0
            current_delta_direction = 1
            current_dur_delta_direction = 1
            dur_delta_tick = 0
            previous_pitch_onset = {prog: -128 for prog in self.config.programs}
            previous_pitch_chord = {prog: -128 for prog in self.config.programs}
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
            prev_tok_type = None
            # Decode tokens
            for ti, token in enumerate(seq):
                tok_type, tok_val = token.split("_")
                if token == "Bar_None":
                    current_bar += 1
                    if current_bar > 0:
                        current_tick = tick_at_current_bar + ticks_per_bar
                    tick_at_current_bar = current_tick
                elif tok_type == "Position":
                    if current_bar == -1:
                        # as this Position token occurs before any Bar token
                        current_bar = 0
                    current_tick = tick_at_current_bar + int(tok_val) * ticks_per_pos
                    current_delta_direction = 1
                elif tok_type == "DeltaDirection":
                    if self.joint_microtiming and prev_tok_type == "Duration":
                        current_dur_delta_direction = -1
                    else:
                        current_delta_direction = -1
                elif tok_type == "Delta":
                    if self.joint_microtiming and prev_tok_type == "Duration":
                        dur_delta_tick = current_dur_delta_direction * int(tok_val) * self.dur_to_pos_delta_factor
                    else:
                        delta_tick = current_delta_direction * int(tok_val)
                        current_tick += delta_tick
                elif tok_type == "DurationDeltaDirection" and not self.joint_microtiming:
                    current_dur_delta_direction = -1
                elif tok_type == "DurationDelta" and not self.joint_microtiming:
                    dur_delta_tick = current_delta_direction * int(tok_val) * self.dur_to_pos_delta_factor
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
                                self.config.default_note_duration
                            )
                        if vel_type == "Velocity" and dur_type == "Duration":
                            dur = int(dur) * ticks_per_dur + dur_delta_tick
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
                            dur_delta_tick = 0
                            current_dur_delta_direction = 1
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
                        ticks_per_bar = compute_ticks_per_bar(
                            current_time_sig, score.ticks_per_quarter
                        )
                        ticks_per_pos = self.delta_factor
                        ticks_per_dur = self.tpq // self.res_dur
                    
                prev_tok_type = tok_type

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
    
    def _create_track_events(
        self,
        track: Track,
        ticks_per_beat: np.ndarray,
        time_division: int,
        ticks_bars: Sequence[int],
        ticks_beats: Sequence[int],
        attribute_controls_indexes: Mapping[int, Sequence[int] | bool] | None = None,
    ) -> list[Event]:
        r"""
        Extract the tokens/events from a track (``symusic.Track``).

        Concerned events are: *Pitch*, *Velocity*, *Duration*, *NoteOn*, *NoteOff* and
        optionally *Chord*, *Pedal* and *PitchBend*.
        **If the tokenizer is using pitch intervals, the notes must be sorted by time
        then pitch values. This is done in**
        :py:func:`miditok.MusicTokenizer.preprocess_score`.

        :param track: ``symusic.Track`` to extract events from.
        :param ticks_per_beat: array indicating the number of ticks per beat per
            section. The numbers of ticks per beat depend on the time signatures of
            the Score being parsed. The array has a shape ``(N,2)``, for ``N`` changes
            of ticks per beat, and the second dimension representing the end tick of
            each portion and the number of ticks per beat respectively.
            This argument is not required if the tokenizer is not using *Duration*,
            *PitchInterval* or *Chord* tokens. (default: ``None``)
        :param time_division: time division in ticks per quarter note of the file.
        :param ticks_bars: ticks indicating the beginning of each bar.
        :param ticks_beats: ticks indicating the beginning of each beat.
        :param attribute_controls_indexes: indices of the attribute controls to compute
            This argument has to be provided as a dictionary mapping attribute control
            indices (indexing ``tokenizer.attribute_controls``) to a sequence of
            bar indexes if the AC is "bar-level" or anything if it is "track-level".
            Its structure is as: ``{ac_idx: Any (track ac) | [bar_idx, ...] (bar ac)}``
            This argument is meant to be used when training a model in order to make it
            learn to generate tokens accordingly to the attribute controls.
        :return: sequence of corresponding ``Event``s.
        """
        program = track.program if not track.is_drum else -1
        use_durations = program in self.config.use_note_duration_programs
        events = []
        # max_time_interval is adjusted depending on the time signature denom / tpb
        max_time_interval = 0
        if self.config.use_pitch_intervals:
            max_time_interval = (
                ticks_per_beat[0, 1] * self.config.pitch_intervals_max_time_dist
            )

        # Attribute controls
        if attribute_controls_indexes:
            for ac_idx, tracks_bars_idx in attribute_controls_indexes.items():
                if (
                    isinstance(self.attribute_controls[ac_idx], BarAttributeControl)
                    and len(tracks_bars_idx) == 0
                ):
                    continue
                events += self.attribute_controls[ac_idx].compute(
                    track,
                    time_division,
                    ticks_bars,
                    ticks_beats,
                    tracks_bars_idx,
                )

        # Creates the Note On, Note Off and Velocity events
        tpb_idx = 0
        for note in track.notes:
            # Program
            if self.config.use_programs and not self.config.program_changes:
                events.append(
                    Event(
                        type_="Program",
                        value=program,
                        time=note.start,
                        program=program,
                        desc=note.end,
                    )
                )

            if self.config.use_pitchdrum_tokens and track.is_drum:
                note_token_name = "DrumOn" if self._note_on_off else "PitchDrum"
            else:
                note_token_name = "NoteOn" if self._note_on_off else "Pitch"
            events.append(
                Event(
                    type_=note_token_name,
                    value=note.pitch,
                    time=note.start,
                    program=program,
                    desc=note.end,
                )
            )

            # Velocity
            if self.config.use_velocities:
                events.append(
                    Event(
                        type_="Velocity",
                        value=note.velocity,
                        time=note.start,
                        program=program,
                        desc=f"{note.velocity}",
                    )
                )

            # Duration / NoteOff
            if use_durations:
                events += self._create_duration_event(
                    note=note,
                    _program=program,
                    _ticks_per_pos = time_division // self.res_dur
                )

        return events