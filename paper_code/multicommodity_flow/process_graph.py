import os
from pathlib import Path
import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd

def read_graph(filename):
    nodes = pd.read_csv(f"{filename}/Node.csv", sep = ",")
    topology = pd.read_csv(f"{filename}/topo.csv", sep=",")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.index)
    graph.add_edges_from(topology.loc[:, ["src", "dst"]].to_numpy())
    cost = topology["IGP cost"].astype(np.float32).to_numpy()
    build_cost = topology["TE cost"].astype(np.float32).to_numpy()
    bandwidth = topology["cap"].astype(np.float32).to_numpy()
    return {"graph": graph, "cost": cost, "build_cost": build_cost, "bandwidth": bandwidth / 1e6}

def read_traffic(filename):
    """
    считываем матрицу корреспонденций
    """
    tunnel = pd.read_csv(filename, sep=",")
    traffic_mat = np.zeros((tunnel["src"].max() + 1, tunnel["dst"].max() + 1), dtype=np.float32)
    traffic_mat[tunnel["src"], tunnel["dst"]] = tunnel["bandwidth"]
    return traffic_mat / 1e6

def solve_problem_with_flow(graph_data, traffic_mat, additional_cost, **solver_kwargs):
    # y отражает дополнтельную стоимость построения путей.
    y = cp.Variable(len(additional_cost))

    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph_data["graph"], oriented=True).todense()
    flow = cp.Variable((len(graph_data["graph"].edges), traffic_mat.shape[0]))
    constraints = [cp.sum(flow, axis=1) <= graph_data["bandwidth"] + y, (incidence_mat @ flow).T == traffic_lapl, flow >= 0, y >= 0] 
    prob = cp.Problem(
        cp.Minimize(cp.sum(flow, axis=1) @ graph_data["cost"] + additional_cost @ y),
        constraints = constraints,
        )
    # prob.solve(max_iters = 10000, reltol= 1e-3, **solver_kwargs)
    prob.solve( **solver_kwargs)
    # print(prob.solution.opt_val)
    return {
        "flow_cost": prob.solution.opt_val,
        "flow_amount": np.abs(flow.value * incidence_mat.T).sum() if flow.value is not None else 1.0,
        "flow": flow.value,
        "extra_bandwidth": y.value
    }

def solve_max_flow(graph_data, source, target, **solver_kwargs):
    f = cp.Variable(nonneg = True)
    bandwidth = graph_data['bandwidth']
    incidence_mat = nx.incidence_matrix(graph_data["graph"], oriented=True).todense()
    n = len(bandwidth)
    # print(incidence_mat.shape, n)
    assert n == incidence_mat.shape[1]
    m = incidence_mat.shape[0]

    flow = cp.Variable(shape = (n,), nonneg = True)
    
    e_vect = np.zeros(shape=(m,))
    e_vect[source] = 1
    e_vect[target] = -1

    constraints = [flow <= bandwidth, incidence_mat @ flow == e_vect * f]
    prob = cp.Problem(
        cp.Maximize(f),
        constraints = constraints
    )
    prob.solve(**solver_kwargs)
    # print()
    return {
        "flows": flow.value,
        "flow_amount": f.value,
    }

def solve_min_cost(graph_data, source, target, min_bandwidth, **solver_kwargs):
    """
    ищем поток минимальной стоимости source -> target по ребрам, 
    пропускная способность которых >= min_bandwidth

    интерпретиуреся как поиск самого дешевого пути,
    чтобы доставить сообщение объемом min_bandwidth
    """
    graph = graph_data["graph"]
    bandwidth = graph_data['bandwidth']
    costs = graph_data["cost"]

    for i, edge in enumerate(graph.edges()):
        
        weight = float("inf")
        if bandwidth[i] > min_bandwidth:
            weight = costs[i]
        graph.edges[edge]['weight'] = weight
    val = nx.shortest_path_length(graph, source, target, weight="weight")
    return val

