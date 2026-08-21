import json

from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class RegularGraph(Topology):
    def __init__(self, topo_input: TopologyParams):
        if not topo_input.graph_file:
            raise ValueError("RegularGraph requires TopologyParams.graph_file")
        super().__init__(topo_input)

    def construct_topology(self, topo_input: TopologyParams):
        with open(topo_input.graph_file, "r") as f:
            graph = json.load(f)

        node_count = int(graph["node_count"])
        self.node_per_chassis = node_count
        self.endpoint_indices = list(map(
            int, graph.get("endpoint_indices", list(range(node_count)))))
        self._graph_switch_indices = list(map(
            int, graph.get("switch_indices", [])))
        directed = bool(graph.get("directed", False))
        rate_unit = str(graph.get("rate_unit", "")).lower()
        rate_scale = 1.0 / 8.0 if rate_unit in ("gbps", "gbit/s") else 1.0
        speed = (
            float(graph.get("link_rate_gbps", topo_input.link_rate_gbps))
            * rate_scale / self.chunk_size
        )
        link_alpha = float(graph.get("alpha", topo_input.alpha[0]))
        node_tx_gbps = graph.get("node_tx_gbps", [])
        if node_tx_gbps and len(node_tx_gbps) != node_count:
            raise ValueError("node_tx_gbps length must match node_count")
        self.node_tx_capacity = [
            float(rate) * rate_scale / self.chunk_size for rate in node_tx_gbps
        ]

        self.capacity = [[0 for _ in range(node_count)] for _ in range(node_count)]
        self.alpha = [[-1 for _ in range(node_count)] for _ in range(node_count)]

        for node in self.endpoint_indices + self._graph_switch_indices:
            if node < 0 or node >= node_count:
                raise ValueError(f"node index {node} is outside node range 0..{node_count - 1}")

        for edge in graph["edges"]:
            left = int(edge[0])
            right = int(edge[1])
            if left < 0 or right < 0 or left >= node_count or right >= node_count:
                raise ValueError(f"edge {edge} is outside node range 0..{node_count - 1}")
            edge_speed = (
                float(edge[2]) * rate_scale / self.chunk_size
                if len(edge) > 2 else speed
            )
            edge_alpha = float(edge[3]) if len(edge) > 3 else link_alpha
            self.capacity[left][right] = edge_speed
            self.alpha[left][right] = edge_alpha
            if not directed:
                self.capacity[right][left] = edge_speed
                self.alpha[right][left] = edge_alpha

    def set_switch_indicies(self) -> None:
        self.switch_indices = list(self._graph_switch_indices)
