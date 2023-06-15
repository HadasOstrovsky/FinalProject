import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

input_neurons_size = 700
plot_array_shape = (35, 20)

inhibition_group_size = 180

output_neurons_size = 1
number_of_experiments = 100
fixed_neurons_percentage = 0.1
random_neurons_percentage = 0.15

stimulus_1_random_indices_selected = np.zeros(input_neurons_size, dtype=int)
stimulus_1_fixed_indices_selected = np.zeros(input_neurons_size, dtype=int)

stimulus_2_random_indices_selected = np.zeros(input_neurons_size, dtype=int)
stimulus_2_fixed_indices_selected = np.zeros(input_neurons_size, dtype=int)

all_initial_learning_stimulus_1_with_inhibition = []
all_initial_learning_stimulus_2_with_inhibition = []
all_initial_learning_stimulus_1_without_inhibition = []
all_initial_learning_stimulus_2_without_inhibition = []
all_second_stage_learning_stimulus_1_with_inhibition = []
all_second_stage_learning_stimulus_2_with_inhibition = []
all_second_stage_learning_stimulus_1_without_inhibition = []
all_second_stage_learning_stimulus_2_without_inhibition = []


def calculate_inhibition_effect(stimulus, inhibition_effect, index):
    connected_neurons_indices = np.random.choice(input_neurons_size, size=inhibition_group_size, replace=False)
    connected_neurons = stimulus[connected_neurons_indices]
    effect = np.sum(connected_neurons) / inhibition_group_size
    inhibition_effect[index] = effect


def experiment_with_inhibition(stimulus_1, stimulus_2, weights):
    stimulus_1, stimulus_1_random_indices = fill_stimulus_with_random_neurons(stimulus_1)
    stimulus_1_random_indices_selected[stimulus_1_random_indices] += 1

    stimulus_2, stimulus_2_random_indices = fill_stimulus_with_random_neurons(stimulus_2)
    stimulus_2_random_indices_selected[stimulus_2_random_indices] += 1

    inhibition_effect_stimulus_1 = np.zeros(input_neurons_size)
    inhibition_effect_stimulus_2 = np.zeros(input_neurons_size)

    for i in range(input_neurons_size):
        calculate_inhibition_effect(stimulus_1, inhibition_effect_stimulus_1, i)
        calculate_inhibition_effect(stimulus_2, inhibition_effect_stimulus_2, i)

    updated_weights_stimulus1 = weights - inhibition_effect_stimulus_1
    updated_weights_stimulus2 = weights - inhibition_effect_stimulus_2

    initial_learning_stimulus_1 = np.dot(stimulus_1, updated_weights_stimulus1)
    initial_learning_stimulus_2 = np.dot(stimulus_2, updated_weights_stimulus2)

    all_initial_learning_stimulus_1_with_inhibition.append(initial_learning_stimulus_1)
    all_initial_learning_stimulus_2_with_inhibition.append(initial_learning_stimulus_2)

    weights_all_zeros = np.zeros(input_neurons_size)
    first_stage_learning_stimulus_1 = np.dot(stimulus_1, weights_all_zeros)
    first_stage_learning_stimulus_2 = np.dot(stimulus_2, weights_all_zeros)

    weights = np.ones(input_neurons_size)
    fixed_neurons_indices_1 = np.where(stimulus_1 == 1)
    weights[fixed_neurons_indices_1] = 0.2
    random_neurons_indices_1 = np.where((stimulus_1 != 0) & (stimulus_1 != 1))
    weights[random_neurons_indices_1] = stimulus_1[random_neurons_indices_1]

    updated_weights_stimulus1 = weights - inhibition_effect_stimulus_1
    updated_weights_stimulus2 = weights - inhibition_effect_stimulus_2

    non_negative_weights_stimulus1 = np.where(updated_weights_stimulus1 < 0, 0, updated_weights_stimulus1)
    non_negative_weights_stimulus2 = np.where(updated_weights_stimulus2 < 0, 0, updated_weights_stimulus2)

    second_stage_learning_stimulus_1 = np.dot(stimulus_1, non_negative_weights_stimulus1)
    second_stage_learning_stimulus_2 = np.dot(stimulus_2, non_negative_weights_stimulus2)

    all_second_stage_learning_stimulus_1_with_inhibition.append(second_stage_learning_stimulus_1)
    all_second_stage_learning_stimulus_2_with_inhibition.append(second_stage_learning_stimulus_2)


def experiment_without_inhibition(stimulus_1, stimulus_2, weights):
    stimulus_1, stimulus_1_random_indices = fill_stimulus_with_random_neurons(stimulus_1)
    stimulus_1_random_indices_selected[stimulus_1_random_indices] += 1

    stimulus_2, stimulus_2_random_indices = fill_stimulus_with_random_neurons(stimulus_2)
    stimulus_2_random_indices_selected[stimulus_2_random_indices] += 1

    initial_learning_stimulus_1 = np.dot(stimulus_1, weights)
    initial_learning_stimulus_2 = np.dot(stimulus_2, weights)

    all_initial_learning_stimulus_1_without_inhibition.append(initial_learning_stimulus_1)
    all_initial_learning_stimulus_2_without_inhibition.append(initial_learning_stimulus_2)

    weights_all_zeros = np.zeros(input_neurons_size)
    first_stage_learning_stimulus_1 = np.dot(stimulus_1, weights_all_zeros)
    first_stage_learning_stimulus_2 = np.dot(stimulus_2, weights_all_zeros)

    weights = np.ones(input_neurons_size)
    fixed_neurons_indices_1 = np.where(stimulus_1 == 1)
    weights[fixed_neurons_indices_1] = 0.2
    random_neurons_indices_1 = np.where((stimulus_1 != 0) & (stimulus_1 != 1))
    weights[random_neurons_indices_1] = np.ones(input_neurons_size)[random_neurons_indices_1] - stimulus_1[
        random_neurons_indices_1]

    second_stage_learning_stimulus_1 = np.dot(stimulus_1, weights)
    second_stage_learning_stimulus_2 = np.dot(stimulus_2, weights)

    all_second_stage_learning_stimulus_1_without_inhibition.append(second_stage_learning_stimulus_1)
    all_second_stage_learning_stimulus_2_without_inhibition.append(second_stage_learning_stimulus_2)


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
    # print(rounded_stimulus)
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


def plot_xy_chart_with_inhibition():
    experiments = list(range(1, number_of_experiments + 1))

    plt.plot(experiments, all_initial_learning_stimulus_1_with_inhibition, color='red', label='Baseline - odor 1 + odor 2')
    plt.plot(experiments, all_second_stage_learning_stimulus_1_with_inhibition, color='green',
             label='Weights reset - odor 1')
    plt.plot(experiments, all_second_stage_learning_stimulus_2_with_inhibition, color='blue',
             label='Weights reset - odor 2')

    average_value_stimulus_1_initial = np.mean(all_initial_learning_stimulus_1_with_inhibition)
    plt.axhline(y=np.nanmean(all_initial_learning_stimulus_1_with_inhibition), color='red', linestyle='--', linewidth=1)
    plt.yticks(list(plt.yticks()[0]) + [average_value_stimulus_1_initial])

    average_value_stimulus_1_final = np.mean(all_second_stage_learning_stimulus_1_with_inhibition)
    plt.axhline(y=np.nanmean(all_second_stage_learning_stimulus_1_with_inhibition), color='green', linestyle='--',
                linewidth=1)
    plt.yticks(list(plt.yticks()[0]) + [average_value_stimulus_1_final])

    average_value_stimulus_2_final = np.mean(all_second_stage_learning_stimulus_2_with_inhibition)
    plt.axhline(y=np.nanmean(all_second_stage_learning_stimulus_2_with_inhibition), color='blue', linestyle='--',
                linewidth=1)
    plt.yticks(list(plt.yticks()[0]) + [average_value_stimulus_2_final])

    plt.xlabel('Experiment Number')
    plt.ylabel('Output')
    plt.title('Experiment Results')

    plt.legend(loc='upper right')

    plt.show()


def plot_xy_chart_without_inhibition():
    experiments = list(range(1, number_of_experiments + 1))

    plt.plot(experiments, all_initial_learning_stimulus_1_without_inhibition, color='red', label='Baseline - odor 1 + odor 2')
    plt.plot(experiments, all_second_stage_learning_stimulus_1_without_inhibition, color='green',
             label='Weights reset - odor 1')
    plt.plot(experiments, all_second_stage_learning_stimulus_2_without_inhibition, color='blue',
             label='Weights reset - odor 2')

    average_value_stimulus_1_initial = np.mean(all_initial_learning_stimulus_1_without_inhibition)
    plt.axhline(y=np.nanmean(all_initial_learning_stimulus_1_without_inhibition), color='red', linestyle='--', linewidth=1)
    plt.yticks(list(plt.yticks()[0]) + [average_value_stimulus_1_initial])

    average_value_stimulus_1_final = np.mean(all_second_stage_learning_stimulus_1_without_inhibition)
    plt.axhline(y=np.nanmean(all_second_stage_learning_stimulus_1_without_inhibition), color='green', linestyle='--',
                linewidth=1)
    plt.yticks(list(plt.yticks()[0]) + [average_value_stimulus_1_final])

    average_value_stimulus_2_final = np.mean(all_second_stage_learning_stimulus_2_without_inhibition)
    plt.axhline(y=np.nanmean(all_second_stage_learning_stimulus_2_without_inhibition), color='blue', linestyle='--',
                linewidth=1)
    plt.yticks(list(plt.yticks()[0]) + [average_value_stimulus_2_final])

    plt.xlabel('Experiment Number')
    plt.ylabel('Output')
    plt.title('Experiment Results')

    plt.legend(loc='upper right')

    plt.show()


def plot_bars_chart():
    average_all_initial_learning_stimulus_1_with_inhibition = np.mean(all_initial_learning_stimulus_1_with_inhibition)
    average_all_initial_learning_stimulus_2_with_inhibition = np.mean(all_initial_learning_stimulus_2_with_inhibition)
    average_all_initial_learning_stimulus_1_without_inhibition = np.mean(all_initial_learning_stimulus_1_without_inhibition)
    average_all_initial_learning_stimulus_2_without_inhibition = np.mean(all_initial_learning_stimulus_2_without_inhibition)


    average_value_stimulus_1_inhibition = average_all_initial_learning_stimulus_1_with_inhibition - np.mean(
        all_second_stage_learning_stimulus_1_with_inhibition)
    average_value_stimulus_1_without_inhibition = average_all_initial_learning_stimulus_1_without_inhibition - np.mean(
        all_second_stage_learning_stimulus_1_without_inhibition)
    average_value_stimulus_2_inhibition = average_all_initial_learning_stimulus_2_with_inhibition - np.mean(
        all_second_stage_learning_stimulus_2_with_inhibition)
    average_value_stimulus_2_without_inhibition = average_all_initial_learning_stimulus_2_without_inhibition - np.mean(
        all_second_stage_learning_stimulus_2_without_inhibition)

    values = [average_value_stimulus_1_inhibition, average_value_stimulus_1_without_inhibition,
              average_value_stimulus_2_inhibition, average_value_stimulus_2_without_inhibition]
    labels = ['stimulus_1_with_inhibition', 'stimulus_1_without_inhibition', 'stimulus_2_with_inhibition',
              'stimulus_2_without_inhibition']
    colors = ['blue', 'blue', 'red', 'red']

    plt.bar(labels, values, color=colors)

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart')

    # Show the chart
    plt.show()


def run_all_experiments():
    stimulus_1, stimulus_2, weights = init_model()

    for i in range(number_of_experiments):
        print("experiment number " + str(i))
        experiment_with_inhibition(stimulus_1, stimulus_2, weights)
        experiment_without_inhibition(stimulus_1, stimulus_2, weights)
        print("------------")

    # print(all_initial_learning_stimulus_1)
    # print(all_second_stage_learning_stimulus_1)
    # print(all_second_stage_learning_stimulus_2)
    # plot_heatmap()
    # plot_xy_chart_with_inhibition()
    # plot_xy_chart_without_inhibition()
    plot_bars_chart()


run_all_experiments()
