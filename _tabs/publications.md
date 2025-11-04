---
title: Publications
icon: fas fa-book
order: 1
---

<details>
<summary><b style="font-family: 'Arial'; color: blue; font-size: 18px;"> 2025 </b></summary>

<summary> <a href="https://arxiv.org/abs/2507.07539"><b>CEA-LIST at CheckThat! 2025: evaluating LLMs as detectors of bias and opinion in text</b></a></summary>
<p><i>Akram Elbouanani, Evan Dufraisse, Aboubacar Tuo, Adrian Popescu. arXiv preprint arXiv:2507.07539, 2025.</i></p>

<blockquote>
This paper presents a competitive approach to multilingual subjectivity detection using large language models (LLMs) with few-shot prompting. We participated in Task 1: Subjectivity of the CheckThat! 2025 evaluation campaign. We show that LLMs, when paired with carefully designed prompts, can match or outperform fine-tuned smaller language models (SLMs), particularly in noisy or low-quality data settings. Despite experimenting with advanced prompt engineering techniques, such as debating LLMs and various example selection strategies, we found limited benefit beyond well-crafted standard few-shot prompts. Our system achieved top rankings across multiple languages in the CheckThat! 2025 subjectivity detection task, including first place in Arabic and Polish, and top-four finishes in Italian, English, German, and multilingual tracks. Notably, our method proved especially robust on the Arabic dataset, likely due to its resilience to annotation inconsistencies. These findings highlight the effectiveness and adaptability of LLM-based few-shot learning for multilingual sentiment tasks, offering a strong alternative to traditional fine-tuning, particularly when labeled data is scarce or inconsistent.
</blockquote>

</details>


<details>
<summary><b style="font-family: 'Arial'; color: blue; font-size: 18px;"> 2024 </b></summary>

<summary> <a href="https://aclanthology.org/2024.naacl-srw.17/"><b>A Meta-Learning Approach for Few-Shot Event Argument Extraction</b></a></summary>
<p><i>Aboubacar Tuo, Romaric Besançon, Olivier Ferret, Julien Tourille. JEP-TALN, 2024.</i></p>

<blockquote>
Few-shot learning techniques for Event Extraction are developed to alleviate the cost of data annotation. However, most studies on few-shot event extraction only focus on event trigger detection and no study has been proposed on argument extraction in a meta-learning context. In this paper, we investigate few-shot event argument extraction using prototypical networks, casting the task as a relation classification problem. Furthermore, we propose to enhance the relation embeddings by injecting syntactic knowledge into the model using graph convolutional networks. Our experimental results show that our proposed approach achieves strong performance on ACE 2005 in several few-shot configurations and highlight the importance of syntactic knowledge for this task.
</blockquote>

<summary> <a href="https://inria.hal.science/hal-04623011v1/document"><b>Extraction des arguments d'événements à partir de peu d'exemples par méta-apprentissage</b></a></summary>
<p><i>Aboubacar Tuo, Romaric Besançon, Olivier Ferret, Julien Tourille. JEP-TALN, 2024.</i></p>

<blockquote>
Les méthodes d'apprentissage avec peu d'exemples pour l'extraction d'événements sont développées pour réduire le coût d'annotation des données. Cependant, la plupart des études sur cette tâche se concentrent uniquement sur la détection des déclencheurs d'événements et aucune étude n'a été proposée sur l'extraction d'arguments dans un contexte de méta-apprentissage. Dans cet article, nous étudions l'extraction d'arguments d'événements avec peu d'exemples en exploitant des réseaux prototypiques et en considérant la tâche comme un problème de classification de relations. De plus, nous proposons d'améliorer les représentations des relations en injectant des connaissances syntaxiques dans le modèle par le biais de réseaux de convolution sur les graphes. Nos évaluations montrent que cette approche obtient de bonnes performances sur ACE 2005 dans plusieurs configurations avec peu d'exemples et soulignent l'importance des connaissances syntaxiques pour cette tâche.
</blockquote>

</details>


<details>
<summary><b style="font-family: 'Arial'; color: blue; font-size: 18px;"> 2023 </b></summary>
<details>
<summary> <a href="https://link.springer.com/chapter/10.1007/978-3-031-08473-7_26"><b>Trigger or not Trigger: Dynamic Thresholding for Few Shot Event Detection</b></a></summary>
<p><i>Aboubacar Tuo, Romaric Besançon, Olivier Ferret, Julien Tourille. ECIR, 2023.</i></p>

<blockquote>
Recent studies in few-shot event trigger detection from text address the task as a word sequence annotation task using prototypical networks. In this context, the classification of a word is based on the similarity of its representation to the prototypes built for each event type and for the “non-event” class (also named null class). However, the “non-event” prototype aggregates by definition a set of semantically heterogeneous words, which hurts the discrimination between trigger and non-trigger words. We address this issue by handling the detection of non-trigger words as an out-of-domain (OOD) detection problem and propose a method for dynamically setting a similarity threshold to perform this detection. Our approach increases f-score by about 10 points on average compared to the state-of-the-art methods on three datasets.
</blockquote>

</details>
<details>
<summary> <a href="https://aclanthology.org/2023.jeptalnrecital-international.18/"><b>Détection d’événements à partir de peu d’exemples par seuillage dynamique</b></a></summary>
<p><i>Aboubacar Tuo, Romaric Besançon, Olivier Ferret, Julien Tourille. RECITAL-TALN, 2023.</i></p>

<blockquote>
Les études récentes abordent la détection d’événements à partir de peu de données comme une tâche d’annotation de séquences en utilisant des réseaux prototypiques. Dans ce contexte, elles classifient chaque mot d’une phrase donnée en fonction de leurs similarités avec des prototypes construits pour chaque type d’événement et pour la classe nulle “non-événement”. Cependant, le prototype de la classe nulle agrège par définition un ensemble de mots sémantiquement hétérogènes, ce qui nuit à la discrimination entre les mots déclencheurs et non déclencheurs. Dans cet article, nous abordons ce problème en traitant la détection des mots non-déclencheurs comme un problème de détection d’exemples “hors-domaine” et proposons une méthode pour fixer dynamiquement un seuil de similarité pour cette détection.
</blockquote>
</details>
</details>

<details>
<summary><b style="font-family: 'Arial'; color: blue; font-size: 18px;"> 2022 </b></summary>
<details>
<summary> <a href="https://link.springer.com/chapter/10.1007/978-3-031-08473-7_26"><b>Better Exploiting BERT for Few-shot Event Detection</b></a></summary>
<p><i>Aboubacar Tuo, Romaric Besançon, Olivier Ferret, Julien Tourille. NLDB, 2022.</i></p>

<blockquote>
Recent approaches for event detection rely on deep supervised learning, which requires large annotated corpora. Few-shot learning approaches, such as the meta-learning paradigm, can be used to address this issue. We focus in this paper on the use of prototypical networks with a BERT encoder for event detection. More specifically, we optimize the use of the information contained in the different layers of a pre-trained BERT model and show that simple strategies for combining BERT layers can outperform the current state-of-the-art for this task.
</blockquote>

</details>
<details>
<summary> <a href="https://hal.archives-ouvertes.fr/hal-03701491/file/3792.pdf"><b>Mieux utiliser BERT pour la détection d’évènements à partir de peu d’exemples</b></a></summary>
<p><i>Aboubacar Tuo, Romaric Besançon, Olivier Ferret, Julien tourille. TALN, 2022.</i></p>

<blockquote>
Les méthodes actuelles pour la détection d’évènements, qui s’appuient essentiellement sur l’apprentissage supervisé profond, s’avèrent très coûteuses en données annotées. Parmi les approches pourl’apprentissage à partir de peu de données, nous exploitons dans cet article le méta-apprentissage et l’utilisation de l’encodeur BERT pour cette tâche. Plus particulièrement, nous explorons plusieurs stratégies pour mieux exploiter les informations présentes dans les différentes couches d’un modèle BERT pré-entraîné et montrons que ces stratégies simples permettent de dépasser les résultats de l’état de l’art pour cette tâche en anglais.
</blockquote>
</details>
</details>
