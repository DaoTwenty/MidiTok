from miditok import TokenizerConfig, REMI, REAPER, MMM
from pathlib import Path
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from tqdm import tqdm
import os 

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
    "base_tokenizer": "REAPER"
}

TOKENIZER_PARAMS_DELTA = TOKENIZER_PARAMS.copy()
TOKENIZER_PARAMS_DELTA["use_microtiming"] = True
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
config_dur = TokenizerConfig(**TOKENIZER_PARAMS_DUR)
config_joint = TokenizerConfig(**TOKENIZER_PARAMS_JOINT)
config_signed = TokenizerConfig(**TOKENIZER_PARAMS_SIGNED)
tokenizers = [
    (REAPER(config), 'quant'), 
    (REAPER(config_delta), 'expressive'),
    (REAPER(config_signed), 'signed'),
    (REAPER(config_dur), 'duration'),
    (REAPER(config_joint), 'joint'),
]

mmm_tokenizers = [
    (MMM(config), 'mmm_quant'),
    (MMM(config_delta), 'mmm_delta')
]

res_size = 0
res_equal = 0
total = 0

test = [Path('mtest.mid')]

print_all = True

if not os.path.exists("MIDIs_decoded"):
    os.makedirs("MIDIs_decoded")

for mf in tqdm(MIDI_PATHS_MULTITRACK):
    res = []
    print(f" ----- {mf.stem} ----- ")
    for tokenizer, name in mmm_tokenizers:
        # Tokenize a MIDI file
        tokens = tokenizer(mf)  # automatically detects Score objects, paths, tokens
        res.append(tokens.tokens)
        # Convert to MIDI and save it
        for tok in tokens.tokens:
            if 'Delta' in tok or print_all:
                print(f"    {name} - {tok}")
        generated_midi = tokenizer(tokens)
        generated_midi.dump_midi(Path("MIDIs_decoded", f"{mf.stem}_{type(tokenizer).__name__}_{name}.mid"))
    total += 1
    res_size += int((len(res[0]) == len(res[1])))
    res_equal += int((res[0] == res[1]))

print(f"Same size: {100 * res_size/total} %")
print(f"Same sequence: {100 * res_equal/total} %")
        