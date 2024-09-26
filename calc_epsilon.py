import numpy as np


def int2base(x, base, size):
    x = np.asarray(x)
    powers = base ** np.arange(size - 1, -1, -1)
    digits = (x.reshape(x.shape + (1,)) // powers) % base
    return digits

from ortools.sat.python import cp_model

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
#        for v in self.__variables:
#            print('%s=%i' % (v, self.Value(v)), end=' ')
#        print()

    def solution_count(self):
        return self.__solution_count

def count(x1, x2, y1, y2, q, height, Sparsity="Full"):
    model = cp_model.CpModel()
    if not model:
        return

    n = len(x1)
    max_sum1 = sum(x1)
    max_sum2 = sum(x2)
    
    if Sparsity == "Full":
        x = [model.NewIntVar(0, q-1, f'x{i}') for i in range(n)]
        max_div1 = max_sum1*(q-1)//q
        max_div2 = max_sum2*(q-1)//q
        div1 = model.NewIntVar(0, max_div1, "div1")
        div2 = model.NewIntVar(0, max_div2, "div2")
    if Sparsity == "Binary":
        x = [model.NewIntVar(0, 1, f'x{i}') for i in range(n)]
        max_div1 = max_sum1//q
        max_div2 = max_sum2//q
        div1 = model.NewIntVar(0, max_div1, "div1")
        div2 = model.NewIntVar(0, max_div2, "div2")
    if Sparsity == "Ternary":
        x = [model.NewIntVar(-1, 1, f'x{i}') for i in range(n)]
        x_abs = [model.NewIntVar(0, 1, f'x{i}') for i in range(n)]
        min_div1 = (-max_sum1)//q
        max_div1 = max_sum1//q
        min_div2 = (-max_sum2)//q
        max_div2 = max_sum2//q
        div1 = model.NewIntVar(min_div1, max_div1, "div1")
        div2 = model.NewIntVar(min_div2, max_div2, "div2")

    model.Add(sum([xi*pi for xi, pi in zip(x, x1)]) == y1+div1*q)
    model.Add(sum([xi*pi for xi, pi in zip(x, x2)]) == y2+div2*q)
    if Sparsity == "Binary":
        model.Add(sum([xi for xi in x]) <= height)
    if Sparsity == "Ternary":
        for i in range(n):
            model.AddAbsEquality(x_abs[i], x[i])
        model.Add(sum([xi for xi in x_abs]) <= height)

    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(x)
    status = solver.SearchForAllSolutions(model, solution_printer)

    return solution_printer.solution_count()


def cardinality(n, q, height, Sparsity="Full"):
    model = cp_model.CpModel()
    if not model:
        return

    if Sparsity == "Full":
        x = [model.NewIntVar(0, q-1, f'x{i}') for i in range(n)]
    if Sparsity == "Binary":
        x = [model.NewIntVar(0, 1, f'x{i}') for i in range(n)]
    if Sparsity == "Ternary":
        x = [model.NewIntVar(-1, 1, f'x{i}') for i in range(n)]
        x_abs = [model.NewIntVar(0, 1, f'x{i}') for i in range(n)]
    if Sparsity == "Binary":
        model.Add(sum([xi for xi in x]) <= height)
    if Sparsity == "Ternary":
        for i in range(n):
            model.AddAbsEquality(x_abs[i], x[i])
        model.Add(sum([xi for xi in x_abs]) <= height)

    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(x)
    status = solver.SearchForAllSolutions(model, solution_printer)

    return solution_printer.solution_count()


def epsilon(x1, x2, q, height, cardinality, Sparsity):
    n = len(x1)
    total = 0.0
    for y1 in range(q):
        for y2 in range(q):
            total += abs(count(x1, x2, y1, y2, q, height, Sparsity)/cardinality-pow(q,-2))
    return total

def average_epsilon(n, a, q, height, Sparsity):
    c = pow(a,n)
    total = 0.0
    cardinality_H = cardinality(n, q, height, Sparsity)
    for i in range(c):
        x1 = int2base([i], a, n)[0]
        for j in range(i+1, c):
            x2 = int2base([j], a, n)[0]
            total += pow(epsilon(x1, x2, q, height, cardinality_H, Sparsity),2)
    return np.sqrt(total*2/(c*(c-1)))


Q = 3
for n in np.arange(1,10,1):
    for height in np.arange(1,n+1,1):
        cardinality_H = cardinality(n, Q, height, Sparsity="Binary")
        for a in np.arange(2,Q+1,1):
            eps = average_epsilon(n, a, Q, height, Sparsity="Binary")
            print("{}\t{}\t{}\t{}\t{}".format(n,height,a,cardinality_H,eps), flush=True)
