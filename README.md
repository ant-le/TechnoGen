# TechnoGen: An AI Techno Music Generator

> Author: `Anton Lechuga` <br>
> Project Type: `Beat the classics | Bring your own data` <br>
> Domain `Audio Processing | Music Generation`

The goal is to build a model that that generates Electronic Dance Music (EDM) indistinguishable from human creators.
In order to limit the scope of the project, I decided to concentrate on the subgeneres of
[Industrial](https://en.wikipedia.org/wiki/Industrial_techno) and/or [Acid](https://en.wikipedia.org/wiki/Acid_techno) Techno, which are very similar in structure, melody and rhythm.

This project is a part of the course 'Applied Deep Learning' at the Technical University Vienna,
in which I want to investigate to what degree it is possible to downscale current state of the art
architectu

## Table of contents

- [1. Usage](#1-usage)
- [2. Approach](#2-appraoch)
  - [2.1 Dataset](#-dataset)
  - [2.2 Model](#-model)
- [3. Training & Results](#3-training-and-results)
- [4. Planned Extensions](#3-results)


## 1. Usage
Installation of the required dependencies using *Python 3.9* onwards with

```bash
pip install -r requirements.txt
```

The codebase in designed for training of a model end-to-end after specifying the directory with the audio file location.

For running the training loop, run:
```bash
python3 training/train.py --config config/<config_name>.yml
```




## 2. Appraoch
  > [!NOTE]<br>
  > The chosen procedure greatly differs from my initial plans. Some of the major changes are:
  > - 
  
  Deviations from planned milestone 1 ...
  The initial goal was to implement a music generator by the deadline of milestone 2. For now, this goal was reduced due to constraints in time and hardware and is planned to be added in future releases.


#### Dataset
- pipeline is designed to work on any audio data, currently supported file types are .wav and .mp3 extensions 
- all data operations are in the 'dataset/*' and can be configurated:
    ```yaml
    dataset:
      offline_data_path: '<path_to_audio>' 
      sample_rate: 44_100 
      hop_size: 8 # number of seconds of training samples
      channels: 1 # number of audio channels to use
      split: [.8, .1, .1] 
      batch_size: 16 
      shuffle: true
      limit: None
    ```



dataset considerations:
- raw audio data 
- selection of tracks from freely downloadable **.wav** files from music platform [soundcloud](https://soundcloud.com/discover). 
- I opted for high quality audio over quantity


The pipeline is designed for processing audio files end-to-end taking raw audio samples as an input.  

In this case, data quality is much more important than quantity and genre specific information can be incorporated

I constructed my own dataset of tracks which were free to download on soundcloud

it consituted much time as there were many factors to be considered

- how to split data (beats, time, not at all...)
- how to transform data (raw, spec, mel_spec)
- how to store for usage independent of model
  - sequences should still be mapped to track
  - easily readables
- number of channels
- sample lenght

#### Model


## 3. Training and Results





## Time Plan

The time plan consists of several stages, where the building & fine-tuning of the model are expected
to be the most time consuming tasks. Note that I plan to include considerations about the final report
and the presentation in each part of the other stages. As I am further quite new to the topic of audio processing
and music generation in particular, I expect the project to be a lot more time consuming as indicated by the
ECTS, which is however a circumstance I am willing to take.

| Task | Hours | Deadline |
| --- | --- | --- |
| dataset collection | 15 | 31.10.2023 |
| Model Design & Building| > 25  | 25.11.2023 |
| Training & Fine-Tuning| > 25 | 15.12.2023 |
| Application Building| 10 | 10.01.2024 |
| Final Report | 5 | 15.01.2024 |
| Presentation Preparation| 5 | 15.01.2024 |

## References

[1] Ji, S., Luo, J., & Yang, X. (2020).
A comprehensive survey on deep music generation: Multi-level representations, algorithms, evaluations, and future directions. arXiv preprint arXiv:2011.06801.

[2] Huang, C. Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., ... & Eck, D. (2018).
Music transformer. arXiv preprint arXiv:1809.04281.

[3] Briot, J. P., Hadjeres, G., & Pachet, F. D. (2020). Deep learning techniques for music generation (Vol. 1).
Heidelberg: Springer.


```

├── 📂 config
├── 📂 docs
│
├──📂 inference
│
├──📂 dataset
│
├──📂 model
│
├──📂 tests
│
├──📂 training

├── 📂 existing
│   ├── 📂 musicxml
│   │   ├── Mozart
│   │   │   ...
│   │   │   └── 📜 Very Famous Composition.mxl
│   │   ...
│   │   └── 📜 Mr. Brightside – The Killers.mxl
│   ├── 📂 render_png
│   └── 📂 render_svg
├── 📂 generated
│   ├── 📂 musicxml
│   ├── 📂 render_png
│   └── 📂 render_svg
└── 📂 pairs
    ├── 📂 clean
    ├── 📜 clean_dirty_index.csv
    └── 📂 dirty
```