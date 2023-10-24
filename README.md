# TechnoGen: An AI Techno Music Generator

> Author: `Anton Lechuga` <br>
> Project Type: `Beat the classics| Bring your own data` <br>
> Domain `Audio Processing`

In this project, the goal is to build a model that is able to generate Electronic Dance Music (EDM). Since music as a broad concepts seems to difficult to capture with the estimated scope of this project, I decided to opt for creating music in the subgeneres of [Industrial](https://en.wikipedia.org/wiki/Industrial_techno) and/or [Acid](https://en.wikipedia.org/wiki/Acid_techno) Techno. It is a project for the course 'Applied Deep Learning' at the Technical University Vienna in which I want to investigate to what degree recent advances in the creation of classical music can transfer to other genres of music.

## 1) Introduction

Techno music greatly relies on repeating elements and structures. For example, most tracks are organised in sequences of 8/16 beats and often incorporate concepts derived from music theory.
The goal of this project will be to design a model that is able to capture these concepts in order to generate new content following similar structures.

Very basic musical elements can be already created by fairly simple LSTM architectures, as [generated](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE) music from a [blog post](https://medium.com/@leesurkis/how-to-generate-techno-music-using-deep-learning-17c06910e1b3) illustrates. My goal will be to use more advanced model architectures like Transformer Models, potentially in combination with Others, which have already been successfully employed in classical music settings, and adapt them if needed.

## 2) Data

My main source of data will consist of MIDI recordings which are [open](https://www.partnersinrhyme.com/blog/) [source](https://colinraffel.com/projects/lmd/) and a self-owned collection of favourite tracks.  I did not collect the data yet, so information will follow. Depending on the performance and need of training data, [data augmentation](https://music-classification.github.io/tutorial/part3_supervised/data-augmentation.html#:~:text=Data%20augmentations%20are%20a%20set,reduce%20the%20problem%20of%20overfitting.) techniques can also be applied.
One of the main challanges in my approach will therefore not lie in the collection of (MIDI) data, but in the decision how to stucture the data to feed it into the model.

## 3) Approach

### 3.1 Model Design

As a starting point, I decided to take the [music transformer](https://arxiv.org/abs/1809.04281) of Huang et. al (2018) which which was already successfully deployed on classical music data.
Inspired by transformer models for NLP, the authors built a model suited to
capture the structure of music by introducing attention mechanisms to capture
reoccurring elements in relative distance, which of course is also an important
aspect for techno music. As the authors of course had lots of data and computing
capacity, I might downscale this approach to fit my needs or approach the idea
different, for example using a [LSTM](https://github.com/Skuldur/Classical-Piano-Composer) architecture.

### 3.2 Training

As the authors of my target paper stated, "different genres call for different ways of serializing polyphonic music into a single stream and also discretizing time." (p.3). Hence, the first important step in training will be to define how the genre of techno should be fitted into the model for attributes (beat, kick-drum, ..., ) and rows (1 beat, 8 beats, ...). For the rest, I will need to adjust the size of training data depending onj training times which will be discussed in more detail in the next stage of the project.

### 3.3 Evaluation

Listening Test Study (with the participants of the course)
Real data, my generation and maybe some reference model

## Time Plan

The time plan consists of several stages, where the building & fine-tuning of the model are expected to be the most time consuming tasks. Note that I plan to include considerations about the final report and the presentation in each part of the other stages. As I am futher quite new to the topic of audio processing and music generation in particular, I expect the project to be a lot more time consuming as indicated by the ECTS, which is however a circumstance I am willing to take.

| Task | Hours | Deadline |
| --- | --- | --- |
| dataset collection | 15‚ | 31.10.2023 |
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
