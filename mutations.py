from controller import SnakeNN
import torch
def main():

    P_REROLL = 0.1
    P_DROPOUT = 0.1
    P_TRANSPOSE = 0.1
    P_INVERSION = 0.1

    snek = SnakeNN(65, 4, 4)
    weights = snek.fc2.weight.data

    print(weights)
    mutation = MutationBuilder(weights)
    mutation.reroll_weights(P_REROLL)
    mutation.transpose_weights(P_TRANSPOSE)
    mutation.sign_invert_weights(P_INVERSION)
    mutation.dropout_weights(P_DROPOUT)
    print(mutation.weights)
    

class MutationBuilder:
    def __init__(self, weights):
        self.weights = weights
        self.mutations=[]

    @staticmethod
    def manipulate_weights_by_index(weights_tensor, coordinates, new_value):
        if torch.is_tensor(weights_tensor):
            x, y = coordinates
            weights_tensor[x, y] = new_value
            return weights_tensor
        else:
            raise TypeError('Input weights are not of type torch.tensor')

    
    # Point Mutations:
    def reroll_weights(self, reroll_probability: float):
        curr_weights = self.weights
        # Create a random mask with the same shape as weights
        reroll_mask = torch.rand_like(curr_weights) <= reroll_probability

        # New value where the mask is true
        mutations = torch.randn_like(curr_weights)
        self.weights = torch.where(reroll_mask, mutations, curr_weights)

    
    # Zero weight at random (dropout)
    def dropout_weights(self, dropout_probability: float):
        curr_weights = self.weights
        # Create a random mask with the same shape as weights
        dropout_mask = torch.rand_like(curr_weights) <= dropout_probability

        # New value where the mask is true
        deletions = torch.zeros_like(curr_weights)
        self.weights = torch.where(dropout_mask, deletions, curr_weights)


    # Transpose two values (inversion): ABCDE -> ACBDE
    def transpose_weights(self, transpose_probability: float):
        curr_weights = self.weights
        x_dim, y_dim = curr_weights.shape
        # Create a mask for transposition based on the probability
        transpose_mask = torch.rand_like(curr_weights) <= transpose_probability

        # Generate indices for transposition
        x_indices, y_indices = torch.where(transpose_mask)

        # Ensure an even number of indices for pairwise swapping
        if len(x_indices) % 2 != 0:
            x_indices = x_indices[:-1]
            y_indices = y_indices[:-1]

        # Pairwise transposition
        for i in range(0, len(x_indices), 2):
            x1, y1 = x_indices[i], y_indices[i]
            x2, y2 = x_indices[i + 1], y_indices[i + 1]

            # Swap the elements
            curr_weights[x1, y1], curr_weights[x2, y2] = curr_weights[x2, y2], curr_weights[x1, y1]


    def sign_invert_weights(self, inversion_probability: float):
        curr_weights = self.weights
        # Create a random mask with the same shape as weights
        reroll_mask = torch.rand_like(curr_weights) <= inversion_probability

        # Opposite sign where the mask is true
        flipped_sign = curr_weights * -1
        self.weights = torch.where(reroll_mask, flipped_sign, curr_weights)

    # Parent Pairs
        # Choose parent pairs
        # Cross-over
        # Random swap single weights
    


if __name__ == '__main__':
    main()