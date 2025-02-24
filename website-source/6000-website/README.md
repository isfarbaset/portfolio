# DSAN 6000 Project

[Review the project overview here](https://gu-dsan.github.io/6000-fall-2023/project/project.html)

## Milestones

The project will be executed over several milestones, and each one has a specific set of requirements and instructions. These instructions are located on the [course website](https://gu-dsan.github.io/6000-fall-2023/project/project.html)

There are four major milestones (click on each one for the approprate instructions and description):

1. [Milestone 1: Define the questions and Exploratory Data Analysis](https://gu-dsan.github.io/6000-fall-2023/project/eda.html)
1. [Milestone 2: NLP and external data overlay](https://gu-dsan.github.io/6000-fall-2023/project/eda.html)
1. [Peer Feedback: Give and receive peer feedback](https://gu-dsan.github.io/6000-fall-2023/project/feedback.html)
1. [Milestone 3: Machine Learning](https://gu-dsan.github.io/6000-fall-2023/project/eda.html)
1. [Milestone 4: Final delivery](https://gu-dsan.github.io/6000-fall-2023/project/eda.html)

All of your work will be done within this team GitHub repository, and each milestone will be tagged with a specific [release tag](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) by the due date.  

1. Milestone 1 (EDA): will be tagged `v0.1-eda`
1. Milestone 2 (NLP): will be tagged `v0.2-nlp`
1. Milestone 3 (ML): will be tagged `v0.3-ml`
1. Milestone 4 (Final): will be tagged `v1.0-final`

## Repository structure

You will work within an **organized** repository and apply coding and development best practices. The repository has the following structure:

```.
├── LICENSE
├── README.md
├── code/
├── data/
├── website/
└── website-source/
```
### Description

* The `code/` directory is where you will write all of your scripts. You will have a combination of Pyspark and Python notebooks, and one sub-directory per major task area. You may add additional sub-directories as needed to modularize your development.
* The `data/` directory contains.
* The `website/` directory where the final website will be built. Any website asset (image, html, etc.) must be added to this directory. **You do not need to serve the website, although you can do so if you wish. However, this naming convention may not work.**
* The `website-source/` is where you will develop the website using your preferred method. It must render in `website/`.

## Code

* Your code files must be well organized
* Do not work in a messy repository and then try to clean it up
* In notebooks, use Markdown cells to explain what you are doing and the decisions you are making
* Do not write monolithic Notebooks or scripts
* Modularize your code (a script should do a single task)
* Use code comments
* Use functions to promote code reuse

## Delivery mechanism

The output of the project will be delivered through a self-contained website in the `website/` subdirectory, having `index.html` as the starting point. You will build the website incrementally over the milestones.

[Read the website requirements.](https://gu-dsan.github.io/6000-fall-2023/project/website.html)

## Evaluation

The project will be evaluated using the following high-level criteria:

* Level of analytical rigor at the graduate student level
* Level of technical approach
* Quality and clarity of your writing and overall presentation


### Grading rubric

- If a deliverable exceeds the requirements and expectations, that is considered A level work.
- If a deliverable just meets the requirements and expectations, that is considered A-/B+ level work.
- If a deliverable does not meet the requirements, that is considered B or lesser level work.

Deductions will be made for any of the following reasons:

- There is lack of analytical rigor:
    - Analytical decisions are not justified
    - Analysis is too simplistic
- Big data files included in the repository
- Instruction are not followed
- There are missing sections of the deliverable
- The overall presentation and/or writing is sloppy
- There are no comments in your code
- There are absolute filename links in your code
- The repository structure is sloppy
- Files are named incorrectly (wrong extensions, wrong case, etc.)
