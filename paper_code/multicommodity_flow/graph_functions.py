"""
Here functions for graph processing are implemented
"""
from typing import Any, Callable
from numpy._typing import NDArray
import numpy.typing as npt
from dataclasses import dataclass, field

import numpy as np
import cvxpy as cp

from paper_code.multicommodity_flow.process_graph import read_graph, read_traffic,\
            solve_problem_with_flow, solve_max_flow, solve_min_cost
import paper_code.multicommodity_flow.lambda2_metrics as l2

# clip functions
from paper_code.altorithm import clip_default, clip_func_decrease, clip_func_increase


# graph_handler
class GraphFunc:    
    def __init__(self, graph, cost, traffic_mat, additional_cost) -> None:
        self.graph = graph
        self.cost = cost
        self.traffic_mat = traffic_mat
        self.additional_cost = additional_cost
    def __call__(self, bandwidth, *args: Any, **solver_kwargs: Any) -> Any:
        dat = {'graph': self.graph, 'cost': self.cost, 'bandwidth': bandwidth}
        try:
            rez = solve_problem_with_flow(dat, self.traffic_mat, 
                                          self.additional_cost,
                                          **solver_kwargs)                                        
            self.rez = rez
            return rez['flow_cost']
        except Exception as e:
            print(e)
            return float("inf")
    def max_flow(self, bandwidth, source, target, **solver_kwargs: Any):
        dat = {'graph': self.graph, 'cost': self.cost, 'bandwidth': bandwidth}
        rez = solve_max_flow(dat, source, target, **solver_kwargs)                                        
        return rez['flow_amount']

    # def min_cost_flow(self, bandwidth, source, target, min_bandwidth, **solver_kwargs: Any):
    #     dat = {'graph': self.graph, 'cost': self.cost, 'bandwidth': bandwidth}
    #     rez = solve_min_cost(dat, source, target, min_bandwidth, **solver_kwargs)                                        
    #     return rez

    def lambda2_val(self, bandwidth, **kwargs):
        rez = l2.lambda2_metric(self.graph, bandwidth)
        return rez

    

# result_handlers
@dataclass
class result_value:
    """
    default class for functions
    """
    graph_func: GraphFunc
    def __call__(self, bandwidth: npt.NDArray, **kwargs) -> Any:
        """
        all functions are called from the bandwidth
        the necessary generation is carried out internally
        """
        raise NotImplementedError("implement this function")
    def get_v2L_clip(self, **kwargs) -> tuple[npt.NDArray, list[Callable[..., Any]]]:
        """
        returns v/L for values, that computed in this function
        and clip functions
        """
        raise NotImplementedError("implement this function")
    def reset_copy(self):
        return result_value(self.graph_func)

@dataclass 
class flow_result(result_value):
    quantile: float = 0.05
    positions: list[npt.NDArray, npt.NDArray] = field(default= None)
    flow_values: list[float] = field(default_factory= lambda: []) 

    def __post_init__(self):
        assert 0. <= self.quantile <= 1., "quantile should be in (0, 1)"
    def __call__(self, bandwidth: npt.NDArray, **kwargs) -> Any:
        """
        there we select (quantile * 100)% of most wanted flows w.r.t. traffic_matrix
        and then compute maximal flow for this pairs
        """
        positions = np.where(self.graph_func.traffic_mat >= \
                            np.quantile(self.graph_func.traffic_mat, 0.95))
        self.positions = positions
        for i, j in zip(*positions):
            max_flow = self.graph_func.max_flow(bandwidth, i, j, solver = cp.SCIPY)
            self.flow_values.append(max_flow)
        return
    def get_v2L_clip(self, **kwargs) -> tuple[npt.NDArray, list[Callable[..., Any]]]:
        """
        Lipchitz constants is 1/2, values is flow_values
        """
        v2L = [2 * min(self.flow_values)]
        clip_f = [clip_func_increase]
        return v2L, clip_f
    @property
    def value(self):
        return self.flow_values
    def reset_copy(self):
        return flow_result(self.graph_func, quantile= self.quantile)
        

@dataclass
class mccf_result(result_value):
    value: float = 0.
    def __call__(self, bandwidth: npt.NDArray, **kwargs) -> Any:
        self.value = self.graph_func(bandwidth, solver = cp.ECOS)
    def get_v2L_clip(self, **kwargs) -> tuple[npt.NDArray, list[Callable[..., Any]]]:
        v2L = [self.value/max(self.graph_func.additional_cost)]
        clip_f = [clip_func_decrease]
        return v2L, clip_f
    def reset_copy(self):
        return mccf_result(self.graph_func)
        

@dataclass
class lambda2_result(result_value):
    value: float = 0.
    def __call__(self, bandwidth: npt.NDArray, **kwargs) -> Any:
        self.value = self.graph_func.lambda2_val(bandwidth, solver = cp.ECOS)
    def get_v2L_clip(self, **kwargs) -> tuple[npt.NDArray, list[Callable[..., Any]]]:
        begin_cost = kwargs["begin_cost"]
        v2L = [self.value/np.max(begin_cost)]
        clip_f = [clip_default]
        return v2L, clip_f
    def reset_copy(self):
        return lambda2_result(self.graph_func)