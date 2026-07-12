# EE3D-S split of the DOWNLOADED units (own split, not the author-shipped one)

The author-shipped `configs/ee3d_s_splits/` is degenerate (train=855 / val=4 / test=1
poses — they publish EE3D-S train-only). These files instead split ONLY the units that are
actually present on disk at `/fs/nexus-projects/DVS_Encodings/EE3D-S` (units with
`events.h5`), and do so **subject-disjoint** so no actor leaks across splits (matches this
project's LOSO / cross-subject-generalization theme).

Downloaded: 36 units across 7 subjects — 18(8) 19(6) 24(1) 26(5) 27(4) 28(5) 32(7).

| split | subjects            | takes |
|-------|---------------------|-------|
| train | 18, 24, 27, 28, 32  | 25    |
| val   | 19                  | 6     |
| test  | 26                  | 5     |

Subject 24 (single take) goes to train (too small to be a held-out set). Val/test are whole
held-out subjects. Regenerate after copying more units by re-running the enumeration and
keeping subjects disjoint. Point a config at this dir via `dataset_init_args.split_dir`.
