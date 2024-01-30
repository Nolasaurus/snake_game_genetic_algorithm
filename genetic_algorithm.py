import pandas as pd
import torch
import matplotlib.pyplot as plt
import random

import controller

GAME_TICK = 3600
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
GRID_SIZE = (8, 8)
RUN_HEADLESS = True
NUM_TRIALS = 200
NUM_EPOCHS = 10000
MUTATION_RATE = 0.2
MUTATION_STDDEV = 0.25
ELITE_SELECTIVITY = 0.2

def main():
    ga = GeneticAlgorithm()
    for epoch in range(NUM_EPOCHS):
        ga.run_trial()

    # Lists to store the data for plotting
    epochs = []
    max_num_steps = []
    mean_num_steps = []
    max_snake_lengths = []
    mean_snake_length = []
    trials_record = ga.trials_record

    # Extract data from the trials_record
    for epoch, record in trials_record.items():
        epochs.append(epoch)
        mean_num_steps.append(record['game_record']['num_steps'].mean())
        mean_snake_length.append(record['game_record']['snake_length'].mean())
        max_num_steps.append(record['max_num_steps'])
        max_snake_lengths.append(record['max_snake_length'])

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # First subplot for max values
    ax1.plot(epochs, max_num_steps, label='Max Number of Steps', color='blue', marker='o')
    ax1.plot(epochs, max_snake_lengths, label='Max Snake Length', color='red', marker='x')
    ax1.set_title('Max Performance Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Max Values')
    ax1.legend()

    # Second subplot for mean values
    ax2.plot(epochs, mean_num_steps, label='Mean Number of Steps', color='green', marker='o')
    ax2.plot(epochs, mean_snake_length, label='Mean Snake Length', color='purple', marker='x')
    ax2.set_title('Average Performance Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Values')
    ax2.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

class GeneticAlgorithm:
    def __init__(self):
        self.epoch = 0
        self.trials_record = {}
        self.curr_results = None

    def run_trial(self):
        print('Epoch:', self.epoch)
        # first run
        if self.epoch == 0:
            self.tester = TrialRunner(NUM_TRIALS, GRID_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH, GAME_TICK, RUN_HEADLESS)
            self.tester.run_trial()
            self.curr_results = self.tester.output_table()
        else:
            weights = self.create_weights_for_next_trial(self.curr_results)
            self.tester = TrialRunner(NUM_TRIALS, GRID_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH, GAME_TICK, RUN_HEADLESS, weights)
            self.tester.run_trial()

        self.curr_results = self.tester.output_table()
        self.trials_record[self.epoch] = {'game_record': self.curr_results,
                                          'max_num_steps': self.curr_results['num_steps'].max(),
                                          'max_snake_length': self.curr_results['snake_length'].max()}

        print('Steps:', self.curr_results['num_steps'].max(), '\n', 'Length:', self.curr_results['snake_length'].max())
        self.epoch += 1

    def create_weights_for_next_trial(self, results_df):
        elite = results_df.sort_values('num_steps', ascending=False).head(int(NUM_TRIALS*ELITE_SELECTIVITY))
        num_remaining = len(results_df) - len(elite)
        # extend weights by randomly choosing weights to duplicate
        best_fc2_weights = elite['fc2_weights'].tolist()
        best_fc3_weights = elite['fc3_weights'].tolist()

        # Extend the best weights lists by resampling
        while len(best_fc2_weights) < len(results_df):
            # Randomly select an item from the list and append it
            best_fc2_weights.append(random.choice(best_fc2_weights))

        while len(best_fc3_weights) < len(results_df):
            # Randomly select an item from the list and append it
            best_fc3_weights.append(random.choice(best_fc3_weights))

        mutated_fc2_weights = self.mutate_weights(pd.Series(best_fc2_weights), MUTATION_RATE).tolist()
        mutated_fc3_weights = self.mutate_weights(pd.Series(best_fc3_weights), MUTATION_RATE).tolist()
        # Stack the weights
        new_fc2_weights = torch.stack(best_fc2_weights + mutated_fc2_weights, dim=0)
        new_fc3_weights = torch.stack(best_fc3_weights + mutated_fc3_weights, dim=0)

        return (new_fc2_weights, new_fc3_weights)


    def mutate_weights(self, weights, mutation_rate):
        mutated_weights = []
        for run_weights in weights:
            # Create a random mask with the same shape as weights
            run_weights = torch.as_tensor(run_weights)
            mutation_mask = torch.rand_like(run_weights) <= mutation_rate

            # Create random mutations
            mutations = torch.randn_like(run_weights) * MUTATION_STDDEV + 1  # normalvariate with mean 1 and std dev 0.2

            # Apply mutations where the mask is true
            mutated_weights.append(torch.where(mutation_mask, run_weights * mutations, run_weights))

        return pd.Series(mutated_weights)

class TrialRunner:
    def __init__(self, num_trials, grid_size, window_height, window_width, game_tick, run_headless, weights=None):
        self.num_trials = num_trials
        self.grid_size = grid_size
        self.window_height = window_height
        self.window_width = window_width
        self.game_tick = game_tick
        self.weights = weights
        self.run_headless = run_headless
        self.game_controller = None
        self.runs = {}

    def run_trial(self):
        for run_number in range(self.num_trials):
            self.game_controller = controller.GameController(self.grid_size, self.window_height, self.window_width, self.game_tick, self.run_headless)
            
            if self.weights:
                fc2_weights, fc3_weights = self.weights
                self.game_controller.model.set_weights((torch.as_tensor(fc2_weights[run_number]), torch.as_tensor(fc3_weights[run_number])))

            self.runs[run_number] = self.game_controller.play_game()


    def output_table(self):
        return pd.DataFrame.from_dict(self.runs, orient='index')

    def print_table(self):
        print(self.output_table())

if __name__=='__main__':
    main()
