este modulo hay que borrarlo entero

# import cplex
import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Dict

from kernel.data.helpers import CplexIds, Process


class BasicResult:  # (Result):
    def __init__(self, valid: bool, solution: List, cplex_ids: Dict[Process, CplexIds]):

        self.valid = valid
        self.cplex_solution_list = solution
        self.cplex_ids = cplex_ids

    def is_valid(self) -> bool:
        return self.valid

    def get_cplex_solution_list(self):
        return self.cplex_solution_list

    def get_cplex_ids(self):
        return self.cplex_ids


class ModelBuilder(ABC):
    pass
