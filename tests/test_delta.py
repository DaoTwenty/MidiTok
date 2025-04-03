from miditok import TokenizerConfig, REMI, REAPER, MMM
from miditok.attribute_controls import create_random_ac_indexes
from pathlib import Path
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from tqdm import tqdm
import os 

from symusic import Score

HERE = Path(__file__).parent
MIDI_PATHS_ONE_TRACK = sorted((HERE / "MIDIs_one_track").rglob("*.mid"))
MIDI_PATHS_MULTITRACK = sorted((HERE / "MIDIs_multitrack").rglob("*.mid"))
MIDI_PATHS_CORRUPTED = sorted((HERE / "MIDIs_corrupted").rglob("*.mid"))
MIDI_PATHS_ALL = sorted(
    deepcopy(MIDI_PATHS_ONE_TRACK) + deepcopy(MIDI_PATHS_MULTITRACK)
)
ABC_PATHS = sorted((HERE / "abc_files").rglob("*.abc"))
TEST_LOG_DIR = HERE / "test_logs"
# MIDI files known to contain tricky contents (time sig, pedals...) and edge case
# situations, likely to make some tests fail.
MIDIS_ONE_TRACK_HARD_NAMES = [
    "6338816_Etude No. 4.mid",
    "6354774_Macabre Waltz.mid",
    "Maestro_9.mid",
    "POP909_191.mid",
]
MIDI_PATHS_ONE_TRACK_HARD = [
    path for path in MIDI_PATHS_ONE_TRACK if path.name in MIDIS_ONE_TRACK_HARD_NAMES
]
from pathlib import Path

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
    "num_velocities": 127,
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_pitch_intervals": False,  # cannot be used as extracting tokens in data loading
    "use_programs": True,
    "num_tempos": 48,
    "tempo_range": (50, 200),
    "programs": list(range(-1, 127)),
    "use_microtiming": False,
    "base_tokenizer": "REAPER",
    "ac_nomml_track": True,
    "ac_loops_track":False,
    "ac_note_density_track": True,
    "ac_polyphony_bar": True,
    "one_token_stream_for_programs": False,
}

TOKENIZER_PARAMS_DELTA = TOKENIZER_PARAMS.copy()
TOKENIZER_PARAMS_DELTA["use_microtiming"] = True
TOKENIZER_PARAMS_DELTA_LOOPS = TOKENIZER_PARAMS_DELTA.copy()
TOKENIZER_PARAMS_DELTA_LOOPS["use_loops"] = True
TOKENIZER_PARAMS_DELTA_LOOPS["ac_loops_track"] = True
TOKENIZER_PARAMS_DUR = TOKENIZER_PARAMS_DELTA.copy()
TOKENIZER_PARAMS_DUR["use_dur_microtiming"] = True
TOKENIZER_PARAMS_SIGNED = TOKENIZER_PARAMS_DELTA.copy()
TOKENIZER_PARAMS_SIGNED["signed_microtiming"] = True
TOKENIZER_PARAMS_JOINT = TOKENIZER_PARAMS_DELTA.copy()
TOKENIZER_PARAMS_JOINT["joint_microtiming"] = True
# Removing "hard" MIDIs from the list
MIDI_PATHS_ONE_TRACK_EASY = [
    p for p in MIDI_PATHS_ONE_TRACK if p not in MIDI_PATHS_ONE_TRACK_HARD
]

config = TokenizerConfig(**TOKENIZER_PARAMS)
config_delta = TokenizerConfig(**TOKENIZER_PARAMS_DELTA)
config_loops = TokenizerConfig(**TOKENIZER_PARAMS_DELTA_LOOPS)
config_dur = TokenizerConfig(**TOKENIZER_PARAMS_DUR)
config_joint = TokenizerConfig(**TOKENIZER_PARAMS_JOINT)
config_signed = TokenizerConfig(**TOKENIZER_PARAMS_SIGNED)
tokenizers = [
    (REAPER(config), 'quant'), 
    (REAPER(config_delta), 'expressive'),
    (REAPER(config_signed), 'signed'),
    (REAPER(config_dur), 'duration'),
    (REAPER(config_joint), 'joint'),
    (REAPER(config_loops), 'loops'),
]

mmm_tokenizers = [
    (MMM(config), 'mmm_quant'),
    (MMM(config_delta), 'mmm_delta'),
    (MMM(config_loops), 'mmm_delta_loops')
]

mmm_tokenizers_loops = [
    (MMM(config_loops), 'mmm_delta_loops')
]

res_size = 0
res_equal = 0
total = 0

test = [Path('mtest.mid')]

print_all = True
print_none = False

if not os.path.exists("MIDIs_decoded"):
    os.makedirs("MIDIs_decoded")

tracks_idx_ratio = 1
bars_idx_ratio = 1

for mf in tqdm(MIDI_PATHS_MULTITRACK[:1]):
    res = []
    print(f" ----- {mf.stem} ----- ")
    for tokenizer, name in mmm_tokenizers_loops:
        score = Score(mf)
        metadata = {"loops":[
            {
                "track_idx":0,
                "start_tick":0,
                "end_tick":3*score.tpq
            },
            {
                "track_idx":0,
                "start_tick":score.tpq,
                "end_tick":2*score.tpq
            },
            {
                "track_idx":1,
                "start_tick":score.tpq,
                "end_tick":2*score.tpq
            }
        ]}
        metadata["tpq"] = score.tpq
        score = tokenizer.preprocess_score(score)
        ac_ind = create_random_ac_indexes(
            score,
            tokenizer.attribute_controls,
            tracks_idx_ratio,
            bars_idx_ratio,
        )
        # Tokenize a MIDI file
        tokens = tokenizer(
            score, 
            no_preprocess_score = True, 
            attribute_controls_indexes = ac_ind,
            metadata = metadata
        )  # automatically detects Score objects, paths, tokens
        res.append(tokens.tokens)
        # Convert to MIDI and save it
        for tok in tokens.tokens:
            if ('Delta' in tok or print_all) and not print_none:
                print(f"    {name} - {tok}")
        generated_midi, metadata = tokenizer(tokens)
        print(metadata)
        generated_midi.dump_midi(Path("MIDIs_decoded", f"{mf.stem}_{type(tokenizer).__name__}_{name}.mid"))
        