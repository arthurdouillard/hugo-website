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

Calendrier où les cours de RDFIA sont indiqués: [calendrier](https://sucal.aminedjeghri.tk/calendar/M2_IMA)

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

- Enregistrement du zoom du 25 Nov: [video](https://drive.google.com/drive/folders/1lBxnQ3Yh4_Q-P2A26F5Uv12ooa5BvP2r?usp=sharing)


**Le rendu est déplacé au 11 Décembre, 23h59.**

**Vu que certains ont mal compris, nous considerons pour le premier rendu (TP1-2) que les questions
avec étoiles sont des bonus. Pour le rendu 2 (TP3-4-5-6) et les suivants, ces questions sont des questions
avec plus de points, les questions bonus seront, comme précisé dès le départ, dénotés par le mot "bonus".**


## TP 7: Transfer Learning

*9 Décembre 2020*

- Énoncé: [TP7.pdf](/files/rdfia_resources/tp7.pdf)
- Colab: [colab](https://colab.research.google.com/drive/1_RnEZX1Fp1z6seRM2c3mpHHowvDeQ1ka?usp=sharing)
- Zoom: [zoom link](https://zoom.us/j/93572572065?pwd=STlWM0JwLzZJUTE5aXNldEhUeFUyZz09), password **rdfia**
- [Alex](https://alexrame.github.io/)'s slides: [slides.pdf](/files/rdfia_resources/alex_slides_tp7.pdf)

## TP 8: Visualization

*16 Décembre 2020*

- Énoncé: [TP8.pdf](/files/rdfia_resources/tp8.pdf)
- Colab: [colab](https://colab.research.google.com/drive/1ubvt5LuJ0SNqsw3rnHRbYHL3Ycp3f3rk?usp=sharing)
- Zoom: [zoom link](https://zoom.us/j/96910655374?pwd=eWt2ZWoydXEvZHJwSEFEZHY1U3c1QT09), password **rdfia**

## TP 9-10: GAN

*6 & 13 Janvier 2021*

- Énoncé: [TP9-10.pdf](/files/rdfia_resources/tp9-10.pdf)

- Colab 1ère séance: [colab 1](https://colab.research.google.com/drive/1judQvIGv965KBdmVRgrkHzGQICVzyA5s?usp=sharing)
- Zoom du 6 Janvier: [zoom link](https://zoom.us/j/98963023208?pwd=RC9tQ2hXY28zVG9LVXd2N053ZjljZz09), password **rdfia**
- Enregistrement du 6 Janvier par Asya: [video.mp4](https://drive.google.com/file/d/1y_wTD5xAURFLthZX37vSqce9hypaqrdm/view?usp=sharing)
- **FIX**: the link for the celeba and celeba64 datasets have been changed (the latter was broken). They are now
http://webia.lip6.fr/~douillard/rdfia/celeba.zip and http://webia.lip6.fr/~douillard/rdfia/celeba64.zip

- Colab 2ème séance: [colab 2](https://colab.research.google.com/drive/1t1N3-EtzWu6mY_-5Hr7Bb0jTgMTC-S4I?usp=sharing)
- Zoom du 13 Janvier: [zoom link](https://zoom.us/j/91509429294?pwd=Y3VEV2xwUlZkdnVub01DV211NVpFdz09)
- Enregistrement du 13 Janvier par Alex + slides: [Google Drive](https://drive.google.com/drive/folders/14RSgh5ik8qV5bhdAf53MMWzzi5mXiCAW?usp=sharing)


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

2020-11-26, 10:28: Mise à jour de la date de rendu du second TP + précision sur les bonus.

2020-12-08, 11:11: Ajout du TP 7 transfer.

2020-12-15, 16:36: Ajout des TPs 8 - 9 - 10.

2021-01-07, 13:46: Ajout du lien vers le cours du 6 Janvier, fix des urls pour celeba.

2021-01-25, 20:50: Ajout du lien vers l'enregistrement et les slides du TP 10.
