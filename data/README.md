# Dataset

15 short English clips (10–30 s, one speaker, no music) from **6 speakers (3 female, 3 male)**, cut from public-domain NASA interview videos.

| | Clips | Speakers |
|---|---|---|
| **dev** (5) | `crippen_01`, `greer_01`, `jones_01`, `talaat_01`, `willing_01` | one clip from each of 5 speakers |
| **test** (10) | `crippen_02`, `crippen_03`, `greer_02`, `greer_03`, `jones_02`, `talaat_02`, `talaat_03`, `willing_02`, `mhanna_01`, `mhanna_02` | all 6; `mhanna` appears only in test (unseen speaker) |

The **dev** split is for tuning prompts, stretch limits and tolerances. The **test** split stays untouched until the parameters of M0/M1/M2 are frozen (blind evaluation, KM4).

## Files

Only the description of the dataset is kept in git. The media are rebuilt from the NASA sources with one command:

```bash
uv run python scripts/build_dataset.py
```

| Path | In git | Content |
|---|---|---|
| `dataset.csv` | yes | clips: id, split, speaker, NASA source id, start/end in the source (s) |
| `speakers.csv` | yes | speakers: name, gender, role, where the voice sample is cut from |
| `metadata.json` | yes | everything in one file, incl. source URLs and the final cut points |
| `references/<clip>.json` | yes | reference English transcript and segment boundaries (from NASA's closed captions) |
| `clips/<clip>.mp4` | no | the clips (1280×720 H.264, AAC mono) |
| `voices/<speaker>.wav` | no | 11–22 s voice sample per speaker for XTTS-v2, 22.05 kHz mono |
| `raw/` | no | downloaded source videos, captions and NASA metadata (cache) |

Voice samples come from parts of the source videos that are **not used in any clip**, so the model never hears the test speech when cloning the voice. The build script checks this.

Cut points are snapped to the quietest 20 ms within ±0.25 s, so no clip starts or ends mid-word.

## Sources and licence

All sources were produced by NASA and are in the **public domain** in the US (NASA media usage guidelines). The guidelines ask that NASA material is not used to imply endorsement.

| Speaker | Source (images.nasa.gov) |
|---|---|
| Robert L. Crippen (M) | [Bob Crippen Interview on STS-1: Soundbites](https://images.nasa.gov/details/KSC-20210325-MH-JRS01_0001-Bob_Crippen_Interview_on_STS-1_Soundbites) |
| Katelynn Greer (F) | [GOLD Resources: Katelynn Greer](https://images.nasa.gov/details/GSFC_20180124_m12825_Katelynn_Greer) |
| Sarah Jones (F) | [GOLD Resources: Sarah Jones](https://images.nasa.gov/details/GSFC_20180124_m12825_SarahJones) |
| Elsayed Talaat (M) | [GOLD Resources: Elsayed Talaat](https://images.nasa.gov/details/GSFC_20180124_m12825_ElsayedTalaat) |
| Gerold Willing (M) | [Professor Gerold Willing Explains the ACE-H2 Experiment](https://images.nasa.gov/details/GRC-2017-CM-0147.4) |
| Ramona Mhanna (F) | [Dr. Ramona Mhanna on the Three Different Types of ACET-9 Experiments](https://images.nasa.gov/details/GRC-2022-CM-0188.2) |

The clips were screened for background music with an AudioSet classifier (AST). Only the intro of the Willing video has music, and `willing_01` starts after it.

> **Voice cloning of real people.** The speakers are real, identifiable people who did not consent to having their voices cloned. Use the dubbed outputs only inside the project and the listening test, and do not publish them. The original plan (self-recorded clips with consent) avoids this. These clips can be swapped for self-recorded ones later by editing the two CSV files.

## To do by hand

- [ ] Check every `references/*.json` against the audio and set `"verified": true`. The captions are good, but GOLD captions were re-typed and the Mhanna captions were ALL CAPS (case restored automatically).
- [ ] Add reference Polish translations.
