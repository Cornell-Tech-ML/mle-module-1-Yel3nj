from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_lst = list(vals)

    vals_lst[arg] += epsilon
    forward = f(*vals_lst)

    vals_lst[arg] -= 2 * epsilon
    backward = f(*vals_lst)

    difference = (forward - backward) / (2 * epsilon)
    return difference


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    traversal = []

    def DFS(var: Variable) -> None:
        if var.is_constant() or var.unique_id in visited:
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            DFS(parent)
        traversal.append(var)

    DFS(variable)
    return reversed(traversal)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topo_order = list(topological_sort(variable))

    gradient_map = {variable.unique_id: deriv}

    for var in topo_order:
        current_gradient = gradient_map.get(var.unique_id)

        if current_gradient is None or var.is_leaf():
            continue

        for parent, local_gradient in var.chain_rule(current_gradient):
            if parent.is_constant():
                continue
            if parent.unique_id in gradient_map:
                gradient_map[parent.unique_id] += local_gradient
            else:
                gradient_map[parent.unique_id] = local_gradient

    for var in topo_order:
        if var.is_leaf():
            var.accumulate_derivative(gradient_map.get(var.unique_id, 0.0)) 


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
