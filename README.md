# TechnoGen: An AI Techno Music Generator

> Author: `Anton Lechuga` <br>
> Project Type: `Bring your own method | Beat the classics` <br>
> Domain `Audio Processing`

In this project, the goal is to build a model that is able to generate Electronic Dance Music (EDM), mainly the subgeneres of [Industrial](https://en.wikipedia.org/wiki/Industrial_techno) &amp; [Acid](https://en.wikipedia.org/wiki/Acid_techno) Techno. It is a project for the course 'Applied Deep Learning' at the Technical University Vienna in which I want to investigate to which degree an AI system can be trained in order to capture the structure of specific music generes which don't rely too much on vocals and follow rather clear structures.

## 1) Introduction

[![Hello](http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)
Short description of your project idea and the approach you intend to use

- Generate music based on a transformer model which is build/fine-tuned on specific genres of EDM music.
- Use transformer architectures in order to to maintain structure‚

## 2) Data

For example, the python package [Music21](https://web.mit.edu/music21/) can be used for handling MIDI information.

- in addition to my personal collection MIDI-files, there are several [sources](https://colinraffel.com/projects/lmd/) to get free MIDI files on the web, too.
  - divided by Beat, or longer
- Often follow certain shape and structures

## 3) Approach

### 3.1 Model Design

Transformer-based model
Investigate whether such a model is capable of capturing some
of the key concepts in music

- OpenAI's model [Jukebox](https://openai.com/research/jukebox)
- Meta's [AudioCraft](https://about.fb.com/news/2023/08/audiocraft-generative-ai-for-music-and-audio/#:~:text=AudioCraft%20consists%20of%20three%20models,generates%20audio%20from%20text%20prompts.) with some interesting [articles](https://www.theverge.com/2023/8/2/23816431/meta-generative-ai-music-audio)  already written about it

### 3.2 Training

### 3.3 Evaluation

# TODO for A1

1. References to at least two scientific papers that are related to your topic
2. A decision of a topic of your choice (see below for inspiration)
3. A decision of which type of project you want to do (see below)
4. A written summary that should contain:
    1. Short description of your project idea and the approach you intend to use
    2. Description of the dataset you are about to use (or collect)
    3. A work-breakdown structure for the individual tasks with time estimates (hours or days) for dataset collection; designing and building an appropriate network; training and fine-tuning that network; building an application to present the results; writing the final report; preparing the presentation of your work.

# Ideas

- Use Transformer Models
  - Predict next bars/beats
  - Next Beat, Bar generation
  - Get Model to find concepts such as key, major/minor, etc. [Google](https://magenta.tensorflow.org/music-transformer) has implemented a first transformer model and there are other interesting [articles](https://www.rootstrap.com/blog/how-to-generate-mu◊sic-with-ai) writing about it

## Time Plan

The time plan consists of several stages, where the building & fine-tuning of the model are expected to be the most time consuming tasks. Note that I plan to include considerations about the final report and the presentation in each part of the other stages. As I am futher quite new to the topic of audio processing and music generation in particular, I expect the project to be a lot more time consuming as indicated by the ECTS, which is however a circumstance I am willing to take.

| Task | Hours | Deadline |
| --- | --- | --- |
| dataset collection | 10 | 31.10.2023 |
| Model Design & Building| > 25  | 25.11.2023 |
| Training & Fine-Tuning| > 25 | 15.12.2023 |
| Application Building| 10 | 10.01.2024 |
| Final Report | 5 | 15.01.2024 |
| Presentation Preperation| 5 | 15.01.2024 |

## References

[1] Ji, S., Luo, J., & Yang, X. (2020). A comprehensive survey on deep music generation: Multi-level representations, algorithms, evaluations, and future directions. arXiv preprint arXiv:2011.06801.

[2] Huang, C. Z. A., Vaswani, A., Uszkoreit, J., Shazeer, N., Simon, I., Hawthorne, C., ... & Eck, D. (2018).
Music transformer. arXiv preprint arXiv:1809.04281.

[3] Briot, J. P., Hadjeres, G., & Pachet, F. D. (2020). Deep learning techniques for music generation (Vol. 1).
Heidelberg: Springer.
