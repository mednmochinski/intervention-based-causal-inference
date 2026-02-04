# Intervention-Based Causal Inference  
## Identification, Estimation, and Application  
**Experiments — Chapter 3**

This repository contains the code and data pipeline used in **Chapter 3** of my master’s dissertation:

> **_Intervention-Based Causal Inference: Identification, Estimation, and Application_**

- Author: Maria Eduarda Mochinski
- Supervisor: Prof. Dr. Cristiano Torezzan
- Program: Mestrado Profissional em Matemática Aplicada e Computacional, IMECC - UNICAMP

The chapter combines **synthetic experiments** and **observational data analysis** to study causal effect estimation under different identification strategies, with a specific application to climate exposure and early neonatal outcomes in Brazil.

---

## Repository Overview

The codebase is organized into two main experiment tracks:

- **Synthetic experiments (`01-*`)**  
  Controlled data-generating processes used to validate causal estimators under known ground truth.

- **Observational experiments (`02-*`)**  
  Real-world analysis using Brazilian climate, birth, and death records from the Climaterna platform.

The naming convention encodes **execution order and experiment type**, described below.


Also included are supporting modules, the main one being `causal_estimators.py` which implements the causal estimators described in the text.  

---

## File Naming Convention and Execution Order

All experiment scripts follow the pattern:

`<experiment>-<step>-<description>.py`



### Experiment index
- `01-*` → Synthetic experiments  
- `02-*` → Observational data experiments  

### Step index
- The number **after the dash (`-`) indicates execution order**
- Scripts with **the same step number can be run in any order**
- Scripts with **higher step numbers must be run after lower ones**

Example:
- `02-01-*` must be executed before any `02-02-*`
- `02-02-*` before `02-03-*`, and so on

---

## Synthetic Experiments (`01-*`)

These scripts generate and analyze simulated datasets with known causal structure.

### Files
- `01-01-synthetic_experiments.py`  
  Runs causal estimators on discrete and continuous synthetic datasets.

- `01-02-format_results.py`  
  Prints results as latex tables and generates plots.

### Data and Outputs
- Generated datasets are stored in: `synthetic_data/datasets/`
- Results and figures are stored in: `synthetic_data/results/`

Synthetic experiments are used to benchmark estimator behavior against the true Average Treatment Effect (ATE).

---

## Observational Data Experiments (`02-*`)

These scripts process real-world observational data and estimate causal effects of heat exposure on early neonatal mortality.

### Data Sources

Raw observational data comes from the [**Climaterna** platform](https://redu.unicamp.br/dataset.xhtml?persistentId=doi:10.25824/redu/ZE4IJM), combining:
- Climate variables
- Birth records (SINASC)
- Death records (SIM)

Raw files taken from the CLIMATERNA dataset are stored in: `observational_data/raw_CLIMATERNA_data/`, within the folders `health` and `climate`



#### Climaterna References

If you use or cite this data, please reference:

**Dataset**
```bibtex
@misc{climaternadata,
  author = {Soares, Camila Ferreira and Fran{\c{c}}a, Breno Bernard Nicolau de and Coltri, Priscila Pereira and Lima, Everton Emanuel Campos de and Torezzan, Cristiano and Xavier, Alexandre Candido and Nichi, Jaqueline and Gallardo Alvarado, Negli Ren{\'e} and Charles, Charles M'poca and Sales, Sergio Floquet and Motta, Gabriel Moreira and Andrade, Matheus Alves de and Risso, Mateus Samuel and Pereira, Malcolm dos Reis Alves and Torres, Guilherme Almussa Leite and Hyslop, Kevin and Silva, Dimitri de Oliveira and Awe, Oluwafunmilola Deborah and Arantes, Caio Simplicio and Andrade J{\'u}nior, Valter Lacerda de and Pacagnella, Rodolfo de Carvalho},
  title = {{Climaterna: integrated platform for climate and maternal-perinatal health data in Brazil}},
  year = {2024},
  version = {DRAFT VERSION},
  publisher = {Reposit{\'o}rio de Dados de Pesquisa da Unicamp},
  doi = {10.25824/redu/ZE4IJM},
  url = {https://doi.org/10.25824/redu/ZE4IJM}
}
```
**Associated Article**

```bibtex
@article{climaternapaper,
  title = {Climaterna: A decade of daily data on births, deaths, pollution and climate variables for all municipalities in Brazil},
  journal = {Data in Brief},
  volume = {62},
  pages = {111920},
  year = {2025},
  issn = {2352-3409},
  doi = {10.1016/j.dib.2025.111920},
  url = {https://www.sciencedirect.com/science/article/pii/S2352340925006444},
  author = {Torezzan, Cristiano and Soares, Camila Ferreira and de Fran{\c{c}}a, Breno Bernard Nicolau and Coltri, Priscila Pereira and de Lima, Everton Emanuel Campos and Xavier, Alexandre C{\^a}ndido and Nichi, Jaqueline and Charles, Charles M'poca and Gallardo-Alvarado, Negli Ren{\'e} and Floquet, Sergio and Motta, Gabriel Moreira and de Andrade, Matheus Alves and Risso, Mateus Samuel and Pereira, Malcolm dos Reis Alves and Torres, Guilherme Almussa Leite and Hyslop, Kevin and de Oliveira Silva, Dimitri and Awe, Oluwafunmilola Deborah and Arantes, Caio Simplicio and de Andrade J{\'u}nior, Valter Lacerda and Pacagnella, Rodolfo de Carvalho}
}

```

### Observational Pipeline Overview
#### Step 1 — Data formatting (`02-01-*`)
- `02-01-format_births_data.py`
- `02-01-format_deaths_data.py`
- `02-01-format_climate_data.py`

  Outputs are written to: `observational_data/processed_data/`

#### Step 2 — Record linkage (`02-02-*`)

- `02-02-match_births_deaths.py`

  Matches birth and death records using demographic and temporal information, ensuring one-to-one. Outputs are written to: `observational_data/processed_data/`

#### Step 3 — Dataset assembly (`02-03-*`)

- `02-03-full_dataset.py`

  Builds the final analysis dataset by merging: births, death outcomes, climate exposure indicators. Outputs are written to: `observational_data/processed_data/`

#### Step 4 — Causal estimation (`02-04-*`)

- `02-04-causal_search.py`

  Applies the causal estimators to the data, using bootstrap. Outputs are written to: `observational_data/results/`

#### Step 5 — Result formatting (`02-05-*`)

- `02-05-format_results.py`

  Prints results as latex tables and generates plots. Outputs are written to: `observational_data/results/`

## Supporting Modules

- `generate_data.py`: Synthetic data generators.
- `causal_estimators.py`: Implementations of causal estimators.
- `aux_functions.py`: Bootstrap, utilities, and result formatting.
- `output_results.py`: Helper routines for exporting figures and tables.

## Disclaimer

This code was developed for academic research purposes as part of a master’s dissertation.

It is **not intended for clinical or policy decision-making without further validation**.
