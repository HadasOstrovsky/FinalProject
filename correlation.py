import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

input_neurons_size = 20
number_of_experiments = 10
fixed_neurons_percentage = 0.1
random_neurons_percentage = 0.15

stimulus_1_random_indices_selected = np.zeros(input_neurons_size, dtype=int)
stimulus_1_fixed_indices_selected = np.zeros(input_neurons_size, dtype=int)
stimulus_2_random_indices_selected = np.zeros(input_neurons_size, dtype=int)
stimulus_2_fixed_indices_selected = np.zeros(input_neurons_size, dtype=int)

def experiment(stimulus_1, stimulus_2):
    stimulus_1, stimulus_1_random_indices = fill_stimulus_with_random_neurons(stimulus_1)
    stimulus_1_random_indices_selected[stimulus_1_random_indices] += 1

    stimulus_2, stimulus_2_random_indices = fill_stimulus_with_random_neurons(stimulus_2)
    stimulus_2_random_indices_selected[stimulus_2_random_indices] += 1



def init_model():
    stimulus_1, stimulus_1_fixed_indices = create_stimulus_with_fixed_neurons()
    stimulus_2, stimulus_2_fixed_indices = create_stimulus_with_fixed_neurons()

    for i in range(number_of_experiments):
        stimulus_1_fixed_indices_selected[stimulus_1_fixed_indices] += 1
        stimulus_2_fixed_indices_selected[stimulus_2_fixed_indices] += 1

    return stimulus_1, stimulus_2,

def create_stimulus_with_fixed_neurons():
    stimulus = np.zeros(input_neurons_size)
    num_changes = int(input_neurons_size * fixed_neurons_percentage)
    fixed_indices = np.random.choice(input_neurons_size, size=num_changes, replace=False)
    stimulus[fixed_indices] = 1

    return np.copy(stimulus), fixed_indices

def fill_stimulus_with_random_neurons(stimulus):
    stimulus_copy = np.copy(stimulus)
    num_changes = int(input_neurons_size * random_neurons_percentage)
    zero_indices = np.where(stimulus == 0)[0]
    random_indices = np.random.choice(zero_indices, size=num_changes, replace=False)
    stimulus_copy[random_indices] = 0.5
    return stimulus_copy, random_indices


def run_all_experiments():
    stimulus_1, stimulus_2 = init_model()

    for i in range(number_of_experiments):
        print("experiment number " + str(i))
        experiment(stimulus_1, stimulus_2)
        print("------------")
        print(stimulus_1, stimulus_2)

    return(stimulus_1,stimulus_2)

run_all_experiments()
