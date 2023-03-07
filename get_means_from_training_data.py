from scipy.stats import entropy

def get_mean_entropy_from_training_data(num_sheets, probs0, labels, num_clusters):
    entropies = dict()
    formatted_data = []
    for i, sheets in enumerate(probs0[:num_sheets]):
        sequence = []
        for j, samples in enumerate(sheets):
            sequence.append(samples)
        formatted_data.append(sequence)

    for k in range(num_clusters):
        entropies[f"entropy_{k}"] = []

    for i in range(len(formatted_data)):
        for j in range(len(formatted_data[0])):
            entropies[f"entropy_{labels[i * 64 + j]}"].append(entropy(formatted_data[i][j]))

    return entropies  # returns a dictionary with the entropies, the keys are the different clusters
