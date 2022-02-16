from models.dnn_model.dnn import DNN
from enum import Enum


class AdaptiveAttribute(Enum):
    """ run-time adaptive attributes of SBRS MoC """
    # input and output edges of CNN layers
    IO = 1,
    # parameters (weights and biases)
    WEIGHTS = 2
    # hyper-parameters (kernel size, stride, padding): currently unsupported
    # hyp = 3


class SBRSMoC:
    """
    Scenario-Based Runtime-adaptive Switching computational model (SBRS MoC):

    The SBRS MoC models a CNN-based application, associated with multiple scenarios:
     CNNs characterised with different
     * architecture or/and
     * parameters (weights) or/and
     * functionality or/and
     * way of execution (dnn_partitioning, mapping, schedule) or/and
     * characteristics (accuracy, throughput, memory and energy)

     The SBRS MoC captures all the scenarios, associated with the CNN-based application,
     and allows for run-time switching among these scenarios. Formally, the SBRS MoC is defined as a scenarios supergraph,
     augmented with a control node and a set of control edges. The scenarios supergraph,
     captures all components (layers and edges) in every scenario of a CNN-based application.
     Thereby, it captures functionality of every scenario, used by the application.
     To represent functionality of a specific scenario, the SBRS MoC uses a sub-graph of the scenarios supergraph.
     The control node of the SBRS MoC is a special node that communicates with the application environment,
     and determines the execution of scenarios in the application supergraph as well as the switching between these scenarios.
     Finally, the set of control edges specifies communication between the control node and the application supergraph.
    """
    def __init__(self, name, scenarios, adaptive_attributes: [AdaptiveAttribute]):
        self.name = name
        self.scenarios = scenarios
        self.supergraph = ScenariosSupergraph()
        self.control_node = ControlNode()
        # The adaptive attributes collection determines which attributes of
        # application scenarios are run-time adaptive and
        # controls the amount of components reuse exploited by the SBRS MoC
        self.__adaptive_attributes = adaptive_attributes

    ######################
    # getters and setters

    def get_adaptive_attributes(self):
        return self.__adaptive_attributes

    def get_sbrs_layer(self, scenario_id, scenario_layer_id):
        """
        Get SBRS layer, capturing layer of a scenario
        :param scenario_id: id of scenario
        :param scenario_layer_id: id of layer
        :return: SBRS layer (Layer), capturing layer of a scenario if
        such a sbrs layer is found, else None
        """
        sbrs_layer_id = self.get_sbrs_layer_id(scenario_id, scenario_layer_id)
        if sbrs_layer_id == -1:
            return None
        layers = self.supergraph.get_layers()
        return layers[sbrs_layer_id]

    def get_sbrs_layer_id(self, scenario_id, scenario_layer_id):
        """
        Get id of SBRS layer, capturing layer of a scenario
        :param scenario_id: id of scenario
        :param scenario_layer_id: id of layer within scenario
        :return: id (int) of SBRS layer, capturing layer of a scenario if
        such a sbrs layer is found, else -1
        """
        for captured_layer in self.control_node.layers_capturing.keys():
            if captured_layer.dnn_id == scenario_id and captured_layer.component_id == scenario_layer_id:
                return self.control_node.layers_capturing[captured_layer]
        return -1

    def get_sbrs_connection(self, scenario_id, scenario_connection_id):
        """
        Get SBRS layer, SBRS connection, capturing connection of a scenario
        :param scenario_id: id of scenario
        :param scenario_connection_id: id of layer
        :return: SBRS Connection (Connection), capturing connection of a scenario if
        such a sbrs connection is found, else None
        """
        sbrs_connection_id = self.get_sbrs_connection_id(scenario_id, scenario_connection_id)
        if sbrs_connection_id == -1:
            return None
        connections = self.supergraph.get_connections()
        return connections[sbrs_connection_id]

    def get_sbrs_connection_id(self, scenario_id, connection_id):
        """
        Get id of SBRS connection, capturing connection of a scenario
        :param scenario_id: id of scenario
        :param connection_id: id of connection within scenario
        :return: id (int) of connection, capturing connection of a scenario if
        such a sbrs connection is found, else -1
        """
        for captured_connection in self.control_node.edges_capturing.keys():
            if captured_connection.dnn_id == scenario_id and captured_connection.component_id == connection_id:
                return self.control_node.layers_capturing[captured_connection]
        return -1

    def get_captured_scenario_layers(self, sbrs_layer, scenario_id):
        """
        Get list of all layers of a specific scenario, captured by the sbrs layer
        :param sbrs_layer: sbrs layer
        :param scenario_id id of scenario
        :return: list of all scenario connections, captured by the sbrs connection
        """
        captured_layers = []
        sbrs_layer_id = sbrs_layer.id

        for captured_layer_record in self.control_node.edges_capturing.items():
            k, v = captured_layer_record
            if v == sbrs_layer_id:
                if k.dnn_id == scenario_id:
                    scenario = self.scenarios[k.dnn_id]
                    scenario_layers = scenario.get_layers()
                    layer = scenario_layers[k.component_id]
                    captured_layers.append(layer)
        return captured_layers

    def get_captured_scenario_connections(self, sbrs_connection, scenario_id):
        """
        Get list of all connections of a specific scenario, captured by the sbrs connection
        :param sbrs_connection: sbrs connection
        :param scenario_id id of scenario
        :return: list of all scenario connections, captured by the sbrs connection
        """
        captured_scenario_connections = []
        sbrs_connection_id = self.supergraph.get_connection_id(sbrs_connection)

        for captured_connection_record in self.control_node.edges_capturing.items():
            k, v = captured_connection_record
            if v == sbrs_connection_id:
                if k.dnn_id == scenario_id:
                    scenario_connection = self.scenarios[scenario_id].get_connections()[k.component_id]
                    captured_scenario_connections.append(scenario_connection)
        return captured_scenario_connections

    ##################
    # print functions

    def __str__(self):
        return "{" + self.name + " scenarios: " + str(len(self.scenarios)) + "}"

    def print_details(self, print_supergraph=True,
                      print_supergraph_details=False,
                      print_control_node=True,
                      print_control_node_details=False):
        print(self)
        if print_supergraph:
            print("supergraph: ")
            self.supergraph.print_details(print_supergraph_details,
                                          print_supergraph_details,
                                          print_supergraph_details)
            print("")
        if print_control_node:
            print("control node: ")
            self.control_node.print_details(print_control_node_details,
                                            print_control_node_details,
                                            print_control_node_details)


class ScenariosSupergraph(DNN):
    """
     Scenarios supergraph,
     captures all components (layers and edges) in every scenario of a CNN-based application,
     represented as the SBRS MoC. Thereby, scenarios supergraph captures functionality of
     every scenario, used by the application.
    """
    def __init__(self):
        super(ScenariosSupergraph, self).__init__("supergraph")


class ComponentInDnnId:
    """ Identifier of a component (layer or edge) within a DNN (scenario)
     Used to distinguish components belonging to different DNNs (scenarios)
     Attributes:
         dnn_id: unique DNN id
         component_id: unique id of component (layer or edge) within the DNN
     """
    def __init__(self, dnn_id: int, component_id: int):
        self.dnn_id = dnn_id
        self.component_id = component_id

    def __str__(self):
        return "{dnn id: " + str(self.dnn_id) + ", component_id: " + str(self.component_id) + "}"


class ControlParameter:
    """ Control parameter """
    def __init__(self, name, par_type: AdaptiveAttribute):
        self.name = name
        self.par_type = par_type

    def __str__(self):
        return "{" + self.name + "," + str(self.par_type) + "}"


class ControlNode:
    """
    The control node of the SBRS MoC is a special node that communicates with the application environment,
     and determines the execution of scenarios in the application supergraph as well as the switching between these scenarios.
    """
    def __init__(self):
        # capturing of scenario layers by scenario supergraph:
        # a dictionary, where key = ComponentInDnnId of scenario layer, captured within the scenario supergraph layer
        # value = id (int) of layer within scenario supergraph
        self.layers_capturing = {}
        # capturing of scenario edges by scenario supergraph:
        # a dictionary, where key = ComponentInDnnId of scenario layer, captured within the scenario supergraph layer
        # value = id (int) of layer within scenario supergraph
        self.edges_capturing = {}
        self.control_parameters_per_sbrs_layer = {}

    def capture_layer(self, supergraph_layer_id, scenario_dnn_id, scenario_layer_id):
        scenario_layer_in_dnn_id = ComponentInDnnId(scenario_dnn_id, scenario_layer_id)
        self.layers_capturing[scenario_layer_in_dnn_id] = supergraph_layer_id

    def capture_edge(self, supergraph_edge_id, scenario_dnn_id, scenario_edge_id):
        scenario_edge_in_dnn_id = ComponentInDnnId(scenario_dnn_id, scenario_edge_id)
        self.edges_capturing[scenario_edge_in_dnn_id] = supergraph_edge_id

    def count_control_par(self):
        control_par_num = 0
        for par_per_layer in self.control_parameters_per_sbrs_layer.values():
            control_par_num += len(par_per_layer)
        return control_par_num

    def add_control_parameter(self, sbrs_layer, par_type: AdaptiveAttribute):
        par_id = self.count_control_par()
        par_name = "par" + str(par_id)
        new_control_parameter = ControlParameter(par_name, par_type)
        if sbrs_layer not in self.control_parameters_per_sbrs_layer.keys():
            self.control_parameters_per_sbrs_layer[sbrs_layer] = []
        self.control_parameters_per_sbrs_layer[sbrs_layer].append(new_control_parameter)

    def find_sbrs_layer_id(self, scenario_id, scenario_layer_id):
        """ TODO: optimize ( search only in interval that belongs to requried scenario"""
        for item in self.layers_capturing.items():
            k, v = item
            if k.dnn_id == scenario_id and k.component_id == scenario_layer_id:
                return v
        return -1

    def __str__(self):
        return "layers captured: " + str(len(self.layers_capturing.items())) + \
               ", edges captured: " + str(len(self.edges_capturing.items()))

    def print_details(self, print_captured_layers, print_captured_edges, print_control_parameters):
        print(self)
        if print_captured_layers:
            print("captured layers {scenario dnn id, scenario layer id} : supergraph layer id")
            for item in self.layers_capturing.items():
                k, v = item
                print(" ", k, ":", v)

        if print_captured_edges:
            print("captured edges: {scenario dnn id, scenario edge id} : supergraph edge id")
            for item in self.edges_capturing.items():
                k, v = item
                print(" ", k, ":", v)

        if print_control_parameters:
            print("control parameters:")
            for item in self.control_parameters_per_sbrs_layer.items():
                k, v = item
                print(" ", k, ":", v)







