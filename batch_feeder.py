from support_functions import math_support_functions
import random


class batch_feeder:

    def __init__(self, cols, rows, col_size, row_size, shuffle):
        self.cols = cols
        self.rows = rows
        self.col_size = col_size
        self.row_size = row_size
        self.shuffle = shuffle
        self.counter = 0

        self.new_cols = math_support_functions.round_to_floor(cols, col_size)
        self.new_rows = math_support_functions.round_to_floor(rows, row_size)

        self.example_pool = []
        for i in range(0, self.new_cols):
            for j in range(0, self.new_rows):
                self.example_pool.append((i, j))

        self.ordered_example_pool = list(range(0, len(self.example_pool)))

    def get_next_interval(self):
        if self.shuffle:
            rand_idx = random.randint(0, len(self.ordered_example_pool) - 1)
            example = self.ordered_example_pool[rand_idx]
            self.ordered_example_pool.remove(example)
            ## TODO: go on from here, print returning 2299 instead of len 2 list or dict.
            print(example)

            return_cols = example[0] * self.col_size
            return_rows = example[1] * self.row_size

            return return_cols, return_rows, return_cols + self.col_size, return_rows + self.row_size

        else:
            if self.counter + 1 == len(self.example_pool):
                return None
            else:
                counter += 1
                example = elf.example_pool[counter -1]

                return_cols = example[0] * self.col_size
                return_rows = example[1] * self.row_size

                return return_cols, return_rows, return_cols + self.col_size, return_rows + self.row_size
