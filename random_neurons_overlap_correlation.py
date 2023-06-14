import matplotlib.pyplot as plt
import numpy as np

input_neurons_size = 700
number_of_experiments = 50
fixed_neurons_percentage = 0.1
min_random_neurons_percentage = 0
max_random_neurons_percentage = 90

overlap_percentages = []
random_neurons_percentages = []


def experiment(stimulus_1, stimulus_2, random_neurons_percentage):
    stimulus_1 = fill_stimulus_with_random_neurons(stimulus_1, random_neurons_percentage)
    stimulus_2 = fill_stimulus_with_random_neurons(stimulus_2, random_neurons_percentage)

    numerator = 0
    denominator = int(input_neurons_size * random_neurons_percentage)

    for i in range(stimulus_1.size):
        stim_1_val = stimulus_1[i]
        stim_2_val = stimulus_2[i]

        if stim_1_val == 0.5:
            if stim_2_val == 0.5:
                numerator += 1
            elif stim_2_val == 1:
                numerator += 0.5
        elif stim_2_val == 1:
            if stim_1_val == 0.5:
                numerator += 0.5

    if denominator == 0:
        return 0

    return numerator / denominator


def init_model():
    stimulus_1, stimulus_1_fixed_indices = create_stimulus_with_fixed_neurons()
    stimulus_2, stimulus_2_fixed_indices = create_stimulus_with_fixed_neurons()

    return stimulus_1, stimulus_2,


def create_stimulus_with_fixed_neurons():
    stimulus = np.zeros(input_neurons_size)
    num_changes = int(input_neurons_size * fixed_neurons_percentage)
    fixed_indices = np.random.choice(input_neurons_size, size=num_changes, replace=False)
    stimulus[fixed_indices] = 1

    return np.copy(stimulus), fixed_indices


def fill_stimulus_with_random_neurons(stimulus, random_neurons_percentage):
    stimulus_copy = np.copy(stimulus)
    num_changes = int(input_neurons_size * random_neurons_percentage)
    zero_indices = np.where(stimulus == 0)[0]
    random_indices = np.random.choice(zero_indices, size=num_changes, replace=False)
    stimulus_copy[random_indices] = 0.5
    return stimulus_copy


def plot_chart():
    plt.plot(random_neurons_percentages, overlap_percentages)
    x_interp = np.interp(0.2, overlap_percentages, random_neurons_percentages)
    plt.plot(x_interp, 0.2, marker='o', markersize=8, color='red')

    plt.annotate(f'({round(x_interp,2)}, 0.2)',
                 xy=(x_interp, 0.2),
                 xytext=(x_interp, 0.3),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 horizontalalignment='center')

    plt.xlabel('random_neurons_percentages')
    plt.ylabel('overlap_percentages')
    # plt.grid(True)
    plt.show()


def run_all_experiments():
    stimulus_1, stimulus_2 = init_model()

    for i in range(min_random_neurons_percentage, max_random_neurons_percentage):
        random_neurons_percentage = i / 100
        results_for_specific_percentage = []

        for j in range(number_of_experiments):
            overlap_pct = experiment(stimulus_1, stimulus_2, random_neurons_percentage)
            results_for_specific_percentage.append(overlap_pct)
            # print(f"overlap_percentages for exp no. {j} with {i} random neurons = {str(overlap_pct)} overlap")
            # print("---")

        average_overlap_pct = sum(results_for_specific_percentage) / len(results_for_specific_percentage)
        overlap_percentages.append(average_overlap_pct)
        random_neurons_percentages.append(random_neurons_percentage)

    print(overlap_percentages)
    print(len(overlap_percentages))

    plot_chart()


run_all_experiments()
