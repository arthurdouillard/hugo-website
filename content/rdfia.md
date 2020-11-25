+++
title = "TP Deep Learning RDFIA"

date = 2020-10-05T00:00:00
draft = false

type = "mylayouts"
layout = "blank"
+++

RDFIA / Master DAC & IMA / Sorbonne

Le cours est organisé par le professeur Matthieu Cord. Vos assistants de TPs auquels
vous devrez envoyer vos travaux sont Asya Grechka (asya.grechka@lip6.fr), Alexandre Rame (alexandre.rame@lip6.fr) et moi-même Arthur
Douillard (arthur.douillard@lip6.fr).

Pour simplifier notre tâche vous êtes priés de nous adresser les mails avec pour objet
`[RDFIA][TP-<numero>]`.

## Rappels

Les TPs seront en Python3 et plusieurs bibliothèques seront utilisées. Voici
quelques liens pour rappel, ou pour vous familiariser en avance:

- [Rappel de Python](https://learnxinyminutes.com/docs/python/)
- [Rappel de Numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html)
- [Introduction de Scikit-Learn](https://scikit-learn.org/stable/tutorial/basic/tutorial.html). L'api est très similaire quelque soit l'algorithme (init / fit / predict)
- [Introduction de Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

Les cours seront ajoutés au fur et à mesure.

## TP 1 - 2 : SIFT / Bag of words
*7 & 14 & 21 Octobre 2020*

~~A rendre pour avant le 21 Octobre 2020 à 23h59.~~
**MaJ: A rendre pour avant le 30 Octobre 2020 à 23h59.**

**Suite à la crise du Covid vous êtes présents en demi-groupes.  Ce TP 1-2
sera fait en 3 semaines (7, 14, et 21 Octobre). Le deuxième groupe ne nous aura donc que une fois en
présentiel mais nous serons plus indulgents, de plus nous pourrons répondre aux questions le 28 Octobre.**

**Pour les prochains TPs (3 -> $\infty$) nous continuerons au rythme normal des années précédentes, il
faudra donc commencer le TP en remote.**

- Énoncé: [TP1-2.pdf](/files/rdfia_resources/tp1-2.pdf)
- Colab: [colab](https://colab.research.google.com/drive/1jL8yy91z6RI0JJIxMQi6Odkh3uMeb7-H?usp=sharing) (faire "File -> Save a copy in Drive")

NB: There is a mistake in our conv_separable, you should see that the images with horizontal and vertical gradients have been inversed. This only impair the visualization of the function compute_grad, but no worry the quality of the final sift stay the same.

Pour aller plus loin, TP sur les SVMs (non noté!):

- Énoncé: [TP2-bis.pdf](/files/rdfia_resources/tp2-bis.pdf)
- Colab: [colab](https://colab.research.google.com/drive/1xkgV6yz2E6_41aYdIC8uSro6gl7eLHn6?usp=sharing)

## TP 3 - 4: Introduction aux réseaux de neurones
*28 Octobre & 4 Novembre 2020*

- Énoncé: [TP3-4.pdf](/files/rdfia_resources/tp3-4.pdf)
- Colab: [colab](https://colab.research.google.com/drive/1MrenVA2opTP0zgut8_Q4O-VUa978Y2DI?usp=sharing)

- Pour le second groupe, tp en visio à 13h45 ici: [Zoom Link](https://zoom.us/j/97871580043?pwd=NXB6Z29sUGtESTJQOXYvNkp4U0dFZz09), password **rdfia**

- Solution gradient: [tp3-4_math.pdf](/files/rdfia_resources/tp3-4_math.pdf)


## TP 5 - 6: Réseaux convolutionnels pour l'image

*18 & 25 Novembre 2020*

Cette fois-ci, les deux groupes auront TPs en même temps, par zoom.

- Énoncé: [TP5-6.pdf](/files/rdfia_resources/tp5-6.pdf)
- Colab: [colab](https://colab.research.google.com/drive/1ZfD37TsJudcyUjff-D8Gagylf-R5NuHd?usp=sharing)
- Zoom 18 Novembre 16h: [zoom link](https://zoom.us/j/98138914049?pwd=ZTVkbWZsc21vS2tKeUF0cWFVeStOdz09), password **rdfia**
- Zoom 25 Novembre 16h: [zoom link](https://zoom.us/j/91842902968?pwd=S3ZxQ1N0TFl6Y0syMyticGdlTHlodz09), password **rdfia**


#### Mises à jour:

2020-10-05, 22:34: Add TP 1-2.

2020-10-07, 13:35: Add Colab link.

2020-10-07, 14:15: Change Colab link.

2020-10-14, 10:54: Update de date de rendu et précision covid.

2020-10-21, 11:17: Add bonus on svm.

2020-10-23, 14:32: Add warning about minor mistake.

2020-10-27, 15:50: Add TP 3-4.

2020-10-28, 09:27: Fix deadline of TP3-4.

2020-11-03, 21:14: Ajout d'un lien zoom pour le prochain tp du 4 Nov.

2020-11-03, 14:25: Ajout d'une cheatsheet gradient.

2020-11-16, 12:42: Ajout du TP5-6 conv.
