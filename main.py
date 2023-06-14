import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

input_neurons_size = 700
plot_array_shape = (35, 20)

output_neurons_size = 1
number_of_experiments = 10
fixed_neurons_percentage = 0.1
random_neurons_percentage = 0.15

stimulus_1_random_indices_selected = np.zeros(input_neurons_size, dtype=int)
stimulus_1_fixed_indices_selected = np.zeros(input_neurons_size, dtype=int)

stimulus_2_random_indices_selected = np.zeros(input_neurons_size, dtype=int)
stimulus_2_fixed_indices_selected = np.zeros(input_neurons_size, dtype=int)

all_initial_learning_stimulus_1 = []
all_second_stage_learning_stimulus_1 = []
all_second_stage_learning_stimulus_2 = []


def experiment(stimulus_1, stimulus_2, weights):
    stimulus_1, stimulus_1_random_indices = fill_stimulus_with_random_neurons(stimulus_1)
    stimulus_1_random_indices_selected[stimulus_1_random_indices] += 1

    stimulus_2, stimulus_2_random_indices = fill_stimulus_with_random_neurons(stimulus_2)
    stimulus_2_random_indices_selected[stimulus_2_random_indices] += 1

    initial_learning_stimulus_1 = np.dot(stimulus_1, weights)
    initial_learning_stimulus_2 = np.dot(stimulus_2, weights)

    all_initial_learning_stimulus_1.append(initial_learning_stimulus_1)

    weights_all_zeros = np.zeros(input_neurons_size)
    first_stage_learning_stimulus_1 = np.dot(stimulus_1, weights_all_zeros)
    first_stage_learning_stimulus_2 = np.dot(stimulus_2, weights_all_zeros)

    weights_for_random_neurons_1 = np.square(stimulus_1)
    fixed_neurons_indices_1 = np.where(stimulus_1 == 1)
    weights_for_random_neurons_1[fixed_neurons_indices_1] = 0.2
    weights_for_random_neurons_2 = np.square(stimulus_1)
    fixed_neurons_indices_2 = np.where(stimulus_2 == 1)
    weights_for_random_neurons_2[fixed_neurons_indices_2] = 1

    print("weights_for_random_neurons_1 = " + str(weights_for_random_neurons_1))
    print("weights_for_random_neurons_2 = " + str(weights_for_random_neurons_2))

    second_stage_learning_stimulus_1 = np.dot(stimulus_1, weights_for_random_neurons_1)
    second_stage_learning_stimulus_2 = np.dot(stimulus_2, weights_for_random_neurons_2)

    all_second_stage_learning_stimulus_1.append(second_stage_learning_stimulus_1)
    all_second_stage_learning_stimulus_2.append(second_stage_learning_stimulus_2)


def init_model():
    stimulus_1, stimulus_1_fixed_indices = create_stimulus_with_fixed_neurons()
    stimulus_2, stimulus_2_fixed_indices = create_stimulus_with_fixed_neurons()

    for i in range(number_of_experiments):
        stimulus_1_fixed_indices_selected[stimulus_1_fixed_indices] += 1
        stimulus_2_fixed_indices_selected[stimulus_2_fixed_indices] += 1

    weights = np.ones(input_neurons_size)

    return stimulus_1, stimulus_2, weights


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
    stimulus_copy[random_indices] = np.random.uniform(low=0.4, high=0.8, size=num_changes)
    rounded_stimulus = np.round(stimulus_copy, decimals=1)
    #print(rounded_stimulus)
    return rounded_stimulus, random_indices


def plot_heatmap():
    fig, (ax, ax2) = plt.subplots(1, 2)
    custom_palette = sns.color_palette("coolwarm", 11)
    cmap = sns.color_palette(custom_palette, as_cmap=True, )

    heatmap_1 = stimulus_1_random_indices_selected + stimulus_1_fixed_indices_selected
    heatmap_1 = np.reshape(heatmap_1, plot_array_shape)
    heatmap_2 = stimulus_2_random_indices_selected + stimulus_2_fixed_indices_selected
    heatmap_2 = np.reshape(heatmap_2, plot_array_shape)

    sns.heatmap(heatmap_2, ax=ax, cmap=cmap, annot=True, fmt="d", linewidth=0.5,
                cbar_kws={'label': 'Activations Number'})
    fig.tight_layout()
    sns.heatmap(heatmap_1, ax=ax2, cmap=cmap, annot=True, fmt="d", linewidth=0.5,
                cbar_kws={'label': 'Activations Number'})
    fig.tight_layout()

    plt.show()


def plot_xy_chart():
    experiments = list(range(1, number_of_experiments + 1))

    plt.plot(experiments, all_initial_learning_stimulus_1, color='red', label='Baseline - odor 1 + odor 2')
    plt.plot(experiments, all_second_stage_learning_stimulus_1, color='green', label='Weights reset - odor 1')
    plt.plot(experiments, all_second_stage_learning_stimulus_2, color='blue', label='Weights reset - odor 2')

    plt.xlabel('Experiment Number')
    plt.ylabel('Output')
    plt.title('Experiment Results')

    plt.legend()

    plt.show()


def run_all_experiments():
    stimulus_1, stimulus_2, weights = init_model()

    for i in range(number_of_experiments):
        print("experiment number " + str(i))
        experiment(stimulus_1, stimulus_2, weights)
        print("------------")

    plot_heatmap()
    plot_xy_chart()


run_all_experiments()

