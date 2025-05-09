import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define model parameters
states = ["<1k", "1k-2k", "2k-3k", "3k-4k", "4k-5k", "5k-6k", "6k-7k", "7k-8k", "8k-9k", ">9k"]
n_states = len(states)
observations = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
n_observations = len(observations)

# Read and process CSV
df = pd.read_csv('covid_data.csv')
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['new_cases_per_million'] = df['new_cases_per_million'].fillna(0)  # Handle NaN (Not a Number)

# Group by year and month
monthly_sums = df.groupby(['year', 'month'])['new_cases_per_million'].sum().reset_index()
monthly_data = [(int(row['month']), row['new_cases_per_million']) for _, row in monthly_sums.iterrows()]

# Map to states
state_sequence = []
for month, total in monthly_data:
    if total < 1000:
        state_sequence.append(0)
    elif total < 2000:
        state_sequence.append(1)
    elif total < 3000:
        state_sequence.append(2)
    elif total < 4000:
        state_sequence.append(3)
    elif total < 5000:
        state_sequence.append(4)
    elif total < 6000:
        state_sequence.append(5)
    elif total < 7000:
        state_sequence.append(6)
    elif total < 8000:
        state_sequence.append(7)
    elif total < 9000:
        state_sequence.append(8)
    else:
        state_sequence.append(9)

# Verify lengths match
assert len(state_sequence) == len(monthly_data), f"Length mismatch: state_sequence={len(state_sequence)}, monthly_data={len(monthly_data)}"

# Initial probabilities
pi = np.zeros(n_states)
pi[state_sequence[0]] = 1.0

# Transition matrix
A = np.zeros((n_states, n_states))
for i in range(len(state_sequence) - 1):
    A[state_sequence[i], state_sequence[i+1]] += 1
A = (A + 1) / (A.sum(axis=1, keepdims=True) + n_states)

# Emission matrix
B = np.zeros((n_states, n_observations))
for i, (month, _) in enumerate(monthly_data):
    month_idx = int(month) - 1  # Ensure integer index
    B[state_sequence[i], month_idx] += 1
B = (B + 1) / (B.sum(axis=0, keepdims=True) + n_states)

# Forward algorithm
def forward(obs_seq):
    T = len(obs_seq)
    alpha = np.zeros((T, n_states))
    alpha[0] = pi * B[:, obs_seq[0]]
    alpha[0] /= alpha[0].sum() or 1
    for t in range(1, T):
        alpha[t] = np.dot(alpha[t-1], A) * B[:, obs_seq[t]]
        alpha[t] /= alpha[t].sum() or 1
    return alpha

# Viterbi algorithm
def viterbi(obs_seq):
    T = len(obs_seq)
    V = np.zeros((T, n_states))
    path = np.zeros((T, n_states), dtype=int)
    V[0] = pi * B[:, obs_seq[0]]
    V[0] /= V[0].sum() or 1
    for t in range(1, T):
        for j in range(n_states):
            prob = V[t-1] * A[:, j] * B[j, obs_seq[t]]
            V[t, j] = np.max(prob)
            path[t, j] = np.argmax(prob)
        V[t] /= V[t].sum() or 1
    best_path = [np.argmax(V[-1])]
    for t in range(T-1, 0, -1):
        best_path.insert(0, path[t, best_path[0]])
    return best_path

# Predict next 3 months
def predict_next_3(month):
    current = forward([month-1])[-1]
    preds = []
    prob = current
    for _ in range(3):
        prob = np.dot(prob, A)
        preds.append(np.argmax(prob))
    return preds

# Filter current month
def filter_month(month):
    probs = forward([month-1])[-1]
    return np.argmax(probs)

# Example usage
month = 5 # May
print(f"Filter May: {states[filter_month(month)]}")
print(f"Predict next 3 from May: {[states[i] for i in predict_next_3(month)]}")
obs_seq = list(range(month))  # Jan to May
best_path = viterbi(obs_seq)
print(f"Most likely path to May: {[states[i] for i in best_path]}")

# Accuracy Test
correct = 0
total = 0
for start_idx in range(0, len(state_sequence) - 3):
    month = (start_idx % 12) + 1  # Map to 1-12
    actual = state_sequence[start_idx + 1:start_idx + 4]
    predicted = predict_next_3(month)
    for a, p in zip(actual, predicted):
        if a == p:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0
print(f"Prediction Accuracy: {accuracy:.2%} (Correct: {correct}, Total: {total})")

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(A, annot=True, cmap="Blues", xticklabels=states, yticklabels=states)
plt.title("Transition Matrix")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(B, annot=True, cmap="Greens", xticklabels=observations, yticklabels=states)
plt.title("Emission Matrix")
plt.show()