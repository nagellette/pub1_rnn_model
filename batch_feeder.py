from support_functions import math_support_functions
import random


class batch_feeder:

    def __init__(self, cols, rows, col_size, row_size):
        self.cols = cols
        self.rows = rows
        self.col_size = col_size
        self.row_size = row_size

        # rounding column, row counts down to floor
        self.new_cols = math_support_functions.round_to_floor(cols, col_size)
        self.new_rows = math_support_functions.round_to_floor(rows, row_size)

        # creating the edge pairs list for sample pictures
        self.example_pool = []
        for i in range(0, self.new_cols):
            for j in range(0, self.new_rows):
                self.example_pool.append((i, j))

        # creating ordered list to process the examples randomly
        self.ordered_example_pool = list(range(0, len(self.example_pool)))

        # random seed
        random.seed(6)

    def get_next_interval(self):

        # get random value in the range
        rand_idx = random.randint(0, len(self.ordered_example_pool) - 1)

        # get the boundary pair from example pair pool
        example = self.example_pool[rand_idx]

        return_cols = example[0] * self.col_size
        return_rows = example[1] * self.row_size

        return return_cols, return_rows, return_cols + self.col_size, return_rows + self.row_size

    def get_total_train_data(self):
        return len(self.ordered_example_pool)
