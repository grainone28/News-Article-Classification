# News Article Categorization via Linear SVC

##  Authors
* **Davide D'Amico** (s353778) - [Politecnico di Torino]
* **Gerardo Rainone** (s354059) - [Politecnico di Torino]

##  Overview
This project implements an automated system to classify news articles into seven distinct categories (Business, Technology, Sports, etc.). By leveraging Natural Language Processing (NLP) and supervised learning, the model identifies the topic of an article based on its title, content, and available metadata.

##  Technical Pipeline
The project follows a standard Data Science workflow:
1. **Feature Engineering**: Combined title and article body into a single text feature.
2. **Preprocessing**: Utilized `TfidfVectorizer` with n-gram ranges (1, 2) and English stop-word removal to convert text into high-dimensional sparse vectors.
3. **Metadata Integration**: Incorporated categorical features (Source) via One-Hot Encoding and numerical features (Page Rank) via Standard Scaling.
4. **Model Selection**: Evaluated various classifiers, selecting **Linear Support Vector Classification (LinearSVC)** for its efficiency with high-dimensional text data.

##  Performance
The optimized model achieves a **Macro F1-score of 0.72**. 
* Detailed methodology and hyperparameter tuning results are available in the [Technical Report](./Report.pdf).

##  Repository Structure
* `main.py`: Production-ready classification pipeline.
* `script.py`: Exploratory analysis and model benchmarking scripts.
* `Report_DAmico_Rainone.pdf`: Comprehensive project documentation.
* `requirements.txt`: Environment dependencies.

##  Data Availability Note
The dataset files (`development.csv` and `evaluation.csv`) are **not included** in this repository. These files are restricted educational materials provided by the professors at Politecnico di Torino. To run the code, the original datasets must be placed in the project's root directory.
