import numpy as np
import numpy.typing as npt
import networkx as nx
# from src.utils import calculate_laplacian_from_weights_matrix, get_weights_matrix

def get_weights_matrix(graph: nx.DiGraph, bandwidth: npt.NDArray) -> npt.NDArray:
    capacity_matrix = nx.adjacency_matrix(graph)
    capacity_matrix = capacity_matrix.toarray().astype(float)
    for edge, bw in zip(graph.edges(), bandwidth):
        capacity_matrix[(edge[1], edge[0])] = \
            capacity_matrix[edge] = bw
        
    return capacity_matrix

def lambda2_metric(graph: nx.DiGraph, bandwidth: npt.NDArray) -> float:
    """
    Calculate lambda_2 metric for weighted graph
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        lam: float, the second smallest eigen value of weight matrix
    """
    weights_matrix = get_weights_matrix(graph, bandwidth = bandwidth)
    # weights_matrix = 0.5 * (weights_matrix + weights_matrix.T)
    rez = lambda2_metric_from_weights_matrix(weights_matrix)
    # print(rez)
    return rez

def calculate_laplacian_from_weights_matrix(weights_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate laplacian matrix: L := D - W, where D is diagonal matrix with d_{ii} = sum_j w_ij
    :param weights_matrix: ndarray of shape (num_nodes, num_nodes), symmetric weights matrix of graph
    :return:
        L: ndarray of shape (num_nodes, num_nodes), laplacian matrix for input weight matrix
    """
    return np.diag(weights_matrix.sum(0)) - weights_matrix

def lambda2_metric_list_graph(graph_data, n) -> float:
    """
    Calculate lambda_2 metric for weighted graph
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        lam: float, the second smallest eigen value of weight matrix

    В этом случае bandwidth представлены в виде списка, поэтому нужно самим составить weight_matrix
    """
    weights_matrix = np.zeros((n,n))
    indices = np.array(list(graph_data['graph'].edges())).T
    weights_matrix[indices[0], indices[1]] = graph_data['bandwidth']

    return lambda2_metric_from_weights_matrix(weights_matrix)

def lambda2_metric_from_weights_matrix(weights_matrix: np.ndarray) -> float:
    """
    Calculate lambda_2 metric for weighted graph
    :param weights_matrix: ndarray of shape (num_nodes, num_nodes), symmetric weights matrix of graph
    :return:
        lam: float, the second smallest eigen value of weight matrix
    """
    laplacian = calculate_laplacian_from_weights_matrix(weights_matrix)
    eigvals, _ = np.linalg.eigh(laplacian)
    lam = sorted(eigvals)[1]
    return lam

def second_min(arr):
    a, b =  np.inf, np.inf
    for elem in arr:
        if elem < a:
            b = a
            a = elem
        elif elem < b:
            b = elem
    return b


def lambda2_mean_metric(graph: nx.DiGraph, key: str = "bandwidth", remaining_edge_portion: float = 0.) -> float:
    """
    Calculate the value lambda_{2,m}(G) = 1/|E| * sum_{e in E} lambda_2(G / {e})

    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :param remaining_edge_portion: float, default=0., the capacity of edge will be decreased from w_0 to w_0 * remaining_edge_portion. Deleting edge is equivalent
        to remaining_edge_portion=0.
    :return:
        lam: float, return the value lambda_{2,m}(G)
    """
    weights_matrix = get_weights_matrix(graph, key=key)
    return lambda2_mean_metric_from_weights_matrix(weights_matrix, remaining_edge_portion=remaining_edge_portion)


def lambda2_mean_metric_from_weights_matrix(weights_matrix: np.ndarray, remaining_edge_portion: float = 0.) -> float:
    """
    Calculate the value lambda_{2,m}(G) = 1/|E| * sum_{e in E} lambda_2(G / {e})
    :param weights_matrix: ndarray of shape (num_nodes, num_nodes), symmetric weights matrix of graph
    :param remaining_edge_portion: float, default=0., the capacity of edge will be decreased from w_0 to w_0 * remaining_edge_portion. Deleting edge is
        equivalent to remaining_edge_portion=0.
    :return:
        lam: float, return the value lambda_{2,m}(G)
    """
    num_nodes = weights_matrix.shape[0]
    lam_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if weights_matrix[i, j] == 0:
                continue
            weights_matrix_new = weights_matrix.copy()
            weights_matrix_new[i, j] = remaining_edge_portion * weights_matrix_new[i, j]
            weights_matrix_new[j, i] = remaining_edge_portion * weights_matrix_new[j, i]
            laplacian = calculate_laplacian_from_weights_matrix(weights_matrix=weights_matrix_new)
            eigvals, _ = np.linalg.eigh(laplacian)
            lam = sorted(eigvals)[1]
            lam_list.append(lam)
    return sum(lam_list) / len(lam_list)


def lambda2_robust_metric(graph: nx.DiGraph, key: str = "bandwidth", remaining_edge_portion: float = 0.) -> float:
    """
    Calculate the value lambda_{2,R}(G)=min_{e in E} lambda_2(G / {e})

    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :param remaining_edge_portion: float, default=0., the capacity of edge will be decreased from w_0 to w_0 * remaining_edge_portion. Deleting edge is equivalent
        to remaining_edge_portion=0.
    :return:
        lam: float, return lambda_{2,R} metric for weighted graph
    """
    weights_matrix = get_weights_matrix(graph, key=key)
    return lambda2_robust_metric_from_weights_matrix(weights_matrix, remaining_edge_portion=remaining_edge_portion)


def lambda2_robust_metric_from_weights_matrix(weights_matrix: np.ndarray, remaining_edge_portion: float = 0.) -> float:
    """
    Calculate the value lambda_{2,R}(G)=min_{e in E} lambda_2(G / {e})

    :param weights_matrix: ndarray of shape (num_nodes, num_nodes), symmetric weights matrix of graph
    :param remaining_edge_portion: float, default=0., the capacity of edge will be decreased from w_0 to w_0 * remaining_edge_portion. Deleting edge is
        equivalent to remaining_edge_portion=0.
    :return:
        lam: float, return lambda_{2,R} metric for weighted graph
    """
    num_nodes = weights_matrix.shape[0]
    lam_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if weights_matrix[i, j] == 0:
                continue
            weights_matrix_new = weights_matrix.copy()
            weights_matrix_new[i, j] = remaining_edge_portion * weights_matrix_new[i, j]
            weights_matrix_new[j, i] = remaining_edge_portion * weights_matrix_new[j, i]
            laplacian = calculate_laplacian_from_weights_matrix(weights_matrix=weights_matrix_new)
            eigvals, _ = np.linalg.eigh(laplacian)
            lam = (eigvals)[1]
            lam_list.append(lam)
    return min(lam_list)
