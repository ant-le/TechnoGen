# TechnoGen: An AI Techno Music Generator

> Author: `Anton Lechuga` <br>
> Project Type: `Beat the classics| Bring your own data` <br>
> Domain `Audio Processing`

The goal is to build a model that that generates Electronic Dance Music (EDM) indistinguishable from human creators.
In order to limit the scope of the project, I decided to concentrate on the subgeneres of
[Industrial](https://en.wikipedia.org/wiki/Industrial_techno) and/or [Acid](https://en.wikipedia.org/wiki/Acid_techno)
Techno, which are very similar in structure, melody and rhythm.

This project is a part of the course 'Applied Deep Learning' at the Technical University Vienna,
in which I want to investigate to what degree recent advances in the creation of classical music can
transfer to other genres of music.

## 1) Introduction

(Techno) music greatly relies on repeating elements and structures and there are many
interesting [theories](https://www.youtube.com/watch?v=JcjT7zgs6cs) around this topic.
Just as an example for the techno genre, many tracks are structured in sequences of 8-16 beats
and often incorporate music theoretical concepts such as of harmony and dissonance.
Very basic techno elements can be already created by fairly simple LSTM architectures,
as [generated](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE) music from this
[blog post](https://medium.com/@leesurkis/how-to-generate-techno-music-using-deep-learning-17c06910e1b3)
illustrates. My goal will be to extend on this by using more recent model architectures
(Transformer Models, GAN) which have already been successfully employed in classical music settings.
This project's end goal is to obtain a model that is able to capture these concepts in order to
generate new pieces of art following similar structures.

## 2) Data

As input data, the current plan is to use musical instrument digital interface
([MDID](https://en.wikipedia.org/wiki/MIDI)) files as symbolic representations of
music [3]. Without getting into too much detail, MIDI-files incorporate information
about the structure of music over time (bar, beats, etc.) which are of relevance for techno music.

As there exist many collections of [open](https://www.partnersinrhyme.com/blog/)
[source](https://colinraffel.com/projects/lmd/) MIDI recordings for each (sub) genre [2,3] and
I have a collection of favourite tracks in MIDI format, I do not worry to much about the data
collection process. However, the data was not collected yet, so more information on that will follow.
Depending on the performance and need of training data,
[data augmentation](https://music-classification.github.io/tutorial/part3_supervised/data-augmentation.html#:~:text=Data%20augmentations%20are%20a%20set,reduce%20the%20problem%20of%20overfitting.)
techniques can also be applied [1]. Hence, the main challenges will not lie in the collection of
(MIDI) data, but in the decision of how to structure the MIDI files in order to feed it them into the
model. For instance, my model of reference uses a vector representation for a fixed sequence and attribute length.

## 3) Approach

### 3.1 Model Design

As a starting point, I decided to take the music transformer which which was already successfully deployed on classical music data [2].
Inspired by transformer models for NLP, the authors built a model which is able to
capture long-term structures of music by introducing relative attention mechanisms. These mulit-head attentions
'remember' reoccurring elements and timed structures in (classical) music [2].

Such mechanisms can naturally also be relevant for techno music. As the authors had lots of data and computing
capacity, I might downscale this approach to fit my needs or opt for other models ([LSTM](https://github.com/Skuldur/Classical-Piano-Composer)) in case
I encounter unexpected difficulties.

### 3.2 Training

In the words of the authors of my reference article, *different genres call for different ways of
serializing polyphonic music into a single stream and also discretizing time.* [2, p.3].
Hence, the first important step in training will be to experiment how the genre of techno
can be fitted into the model for attributes (e.g. beat, kick-drum, ..., ) and rows (e.g. 1 beat, 8 beats, ...) [2].
In related papers and works, the training time often exceeded 24 hours. Hence, I will need
to take this into account and experiment with different training data sizes in order to be
able to finish the project with the given resources as hyperparameter settings are an important aspect to optimise for.

### 3.3 Evaluation

One big question mark lies in how to evaluate the model, since it is generally difficult to
evaluate generative art. While objective methods might comprise music theoretical consideration,
I am currently planning to opt for a subjective evaluation. The current plan is to conduct a
Listening Test Study with the participants of the course, where participants will be asked to
take part in a Turing test, where they need to judge wether a techno piece was generated by the
model or human-created [3]. Weather this will be implemented in the application or conducted differently
(f.e. during the project presentation) is still undecided.

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
