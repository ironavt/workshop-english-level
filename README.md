# workshop-english-level
Determining CEFR English level for subtitle files

# Intro
**Yandex.Practicum's English department** is a customer for this project https://practicum.yandex.ru/english/

One of the most effective ways to study foreighn languages (including English) is to watch movies. It's considered it's best for the student to watch videos in which they can understand from 50% to 70% of all dialogs in order to maximise their learning rate. Thus it's vital how movie contents matches an English level. We will use **CEFR** to define English level.

A dataset containig information on some movies' English level is provided by Yandex.Prackticum experts.

**Objective** is to build a model that can evaluate English level of movies based on their subtitles content.

## Project composition

This project is devided on three notebooks:
* `english_level_dataset.ipynb`: forms a dataset from all the data and saves it into `text_labels.csv` file
* `english_level_modeling.ipynb`: takes `text_labels.csv` file, does text processing, modeling and saves model `english_labels_model.pkl` file
* `english_level_servise.ipynb`: allows to label provided `.srt` file using the saved model

# Project status
 This project is now finished yet
