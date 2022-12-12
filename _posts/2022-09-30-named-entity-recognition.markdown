---
layout: post
title:  "Named Entity Recognition (NER)"
date:   2022-09-30 16:15:21 +0200
categories: IE
---

In natural language processing, Named Entity Recognition (NER) is a common sub-tasks of Information Extraction that consists of identifying and classifing named entities in untructured texts, into pre-defined categories such as **persons**, **locations**, **organizations**, etc. An entity can be any word or series of words that consistently refers to the same thing. Every detected entity is classified into a predetermined category. For example, an NER model might detect the word “Google” in a text and classify it as a “Company”.

<!--We need sentences labeled with entities of interest where the labeling of each sentence is done either manually or by some automated method (often using heuristics to create a noisy/weakly labeled data set). These labeled sentences are then used to train a model to recognize those entities as a supervised learning task.-->
