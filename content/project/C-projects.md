+++
# Date this page was created.
date = 2030-04-27T00:00:00

# Project title.
title = "C/C++ low-level projects"

# Project summary to display on homepage.
summary = "Low level C/C++ projects done during the first semester of Engineering school"

# Optional image to display on homepage (relative to `static/img/` folder).
image_preview = ""

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["programming", "C/C++"]

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Optional featured image (relative to `static/img/` folder).
[header]
image = ""
caption = ""

+++

Engineering schools last 3 years in France. Before these years students learn
for two years a broad science curriculum covering mathematics, algo,
physics, electronics...
The three years of engineering cover a more specialized subject, for me it was
computer science.

The first year of my engineering school, EPITA, is intensive. We learn
many different subjects about low-level programming and almost weekly have hackathons the
whole weekend. During the days we study the theory of computer science (language
theory, compiler, graph, algo, some maths, etc.), and during the nights we code.

I'll list in this page, only some of the projects done during the first semester.

# ISO reader

In this C project I had to learn how to use `mmap` to read ISO files (such as
a Ubuntu image). The goal was to provide a CLI tool to navigate, and read files
stored in the ISO.

While it may sound obvious to me now, it was interesting to discover how the
data can be stored in memory, and how simple pointer arithmetic can go a long way.

# Make-like

For this C project I wrote a `make`-like CLI tool. `make` is a tool to maintain
program dependencies and to facilitate a program compilation.

It was interesting for both reasons:

- Learn to parse file, the structure is simple but still harder than a csv
file of course.
- Learn all the tricks `make` have, and it has many.

# Naive Malloc

For this C project I wrote a naive implementation of `malloc`. For those who don't
know, `malloc` is used in C program (and a lot of binaries like `ls`) to
allocate memory.

To allocate memory, I had to map one or several pages. Those pages could be fully
filled or split into several chunks. It's enlightening to understand
how a program uses the memory, and to see all the tricks to allocate it more
efficiently than a simple first-fit algo.

Along side `malloc`, `realloc`, `calloc`, and `free` were obviously implemented.

# Misc

I've also implemented:

- Dijkstra algorithm for a real-time formula one competition. Other tricks were
used as Bezier curves to produce smooth turning.
- A `bash`-like in C in a group of four students, from the command line reading,
to the execution with custom builtins and instructions tree building.
- A fast implementation of a calculator in C++. It supported any base (2, 3, 10, etc.)
with any ASCII characters. It was also a "*big num*" implementation were numbers
could be bigger than a `long long`. Finally some interesting algos were coded
like the Karatsuba multiplication.
