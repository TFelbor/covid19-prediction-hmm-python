# CPSC444 : AI Project 3 - COVID-19 Case Prediction with Hidden Markov Models

This repository contains a Python implementation of a Hidden Markov Model (HMM) for analyzing and predicting COVID-19 case trends based on historical monthly data.

## Overview

This project uses a Hidden Markov Model to:
- Analyze patterns in COVID-19 case data
- Predict future case levels based on historical trends
- Visualize transition probabilities between different case levels
- Evaluate the accuracy of predictions

The model categorizes COVID-19 cases per million into 10 different states (ranging from "<1k" to ">9k") and uses monthly observations to predict future states.

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/covid-hmm-prediction.git
cd covid-hmm-prediction
pip install -r requirements.txt
```

## Data

The model uses a `covid_data.csv` file that contains COVID-19 data with the following columns:
- `date`: Date in MM/DD/YYYY format
- `new_cases_per_million`: Number of new COVID-19 cases per million population
- (potentially other columns that aren't used in this analysis)

> **Note:** The dataset file is not included in this repository due to its size. You need to download it separately and place it in the project directory.

## Usage

Run the main script to perform the analysis:

```bash
python HMM.py
```

This will:
1. Process the COVID-19 data
2. Build and train the Hidden Markov Model
3. Run predictions for example months
4. Evaluate the model's accuracy
5. Generate visualizations of the transition and emission matrices

## Model Details

The HMM uses the following components:

- **States (10)**: Represent different levels of COVID-19 cases per million:
  - "<1k", "1k-2k", "2k-3k", "3k-4k", "4k-5k", "5k-6k", "6k-7k", "7k-8k", "8k-9k", ">9k"

- **Observations (12)**: Represent months of the year:
  - "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"

- **Algorithms Implemented**:
  - Forward algorithm (for filtering)
  - Viterbi algorithm (for finding most likely state sequence)
  - Prediction for future months

## Functions

- `forward(obs_seq)`: Implements the forward algorithm for filtering
- `viterbi(obs_seq)`: Implements the Viterbi algorithm for finding the most likely state sequence
- `predict_next_3(month)`: Predicts case levels for the next 3 months
- `filter_month(month)`: Estimates the most likely state for a specific month

## Example Output

The script outputs:
- Current state estimate for the specified month
- Predictions for the next three months
- Most likely path of states leading to the current month
- Overall prediction accuracy across the dataset
- Visualization of transition and emission probability matrices

## Visualizations

The script generates two heatmaps:
1. **Transition Matrix**: Shows the probability of moving from one state to another
2. **Emission Matrix**: Shows the relationship between states and observations (months)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- This project uses COVID-19 data for educational and research purposes
- The HMM implementation is based on standard algorithms for sequential data analysis
