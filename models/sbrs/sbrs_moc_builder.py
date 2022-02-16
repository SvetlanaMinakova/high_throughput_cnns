from models.sbrs.sbrs_moc import SBRSMoC, AdaptiveAttribute, ScenariosSupergraph
from models.dnn_model.dnn import DNN, Layer, Connection
from models.dnn_model.param_shapes_generator import generate_param_shapes_dict
import copy


def build_sbrs_moc(scenarios: [DNN], adaptive_attributes: [AdaptiveAttribute]):
    """
    Algorithm that automatically builds SBRS model
    :param scenarios: set of application scenarios (DNNs)
    :param adaptive_attributes: collection of run-time adaptive attributes of SBRS MoC
    :return: SBRS model
    """
    def __add_layers():
        """Add layers into SBRS MoC supergraph"""
        scenario_id = 0
        # visit every scenario
        for scenario in scenarios:
            # add layers of scenario
            for scenario_layer in scenario.get_layers():
                supergraph_layer = find_suitable_sbrs_layer(sbrs_moc, scenario_layer)
                # add new supergraph layer
                if supergraph_layer is None:
                    supergraph_layer = copy.deepcopy(scenario_layer)
                    sbrs_moc.supergraph.add_layer(supergraph_layer)

                # add record into layers capturing
                sbrs_moc.control_node.capture_layer(supergraph_layer.id, scenario_id, scenario_layer.id)

            # increment scenario id
            scenario_id += 1

    def __add_edges():
        """Add edges into SBRS MoC supergraph"""
        scenario_id = 0
        # visit every scenario
        for scenario in scenarios:
            # add edges of scenario
            scenario_edges = scenario.get_connections()
            scenario_edge_id = 0
            for scenario_edge in scenario_edges:
                supergraph_edge = find_suitable_sbrs_connection(sbrs_moc, scenario_id, scenario_edge)
                # add new super-graph connection
                if supergraph_edge is None:
                    sbrs_src = sbrs_moc.get_sbrs_layer(scenario_id, scenario_edge.src.id)
                    sbrs_dst = sbrs_moc.get_sbrs_layer(scenario_id, scenario_edge.dst.id)
                    supergraph_edge = Connection(sbrs_src, sbrs_dst)
                    sbrs_moc.supergraph.add_connection(supergraph_edge)

                supergraph_edge_id = sbrs_moc.supergraph.get_connection_id(supergraph_edge)
                # add record into edges capturing
                sbrs_moc.control_node.capture_edge(supergraph_edge_id, scenario_id, scenario_edge_id)

                # increment scenario edge id
                scenario_edge_id += 1

            # increment scenario id
            scenario_id += 1

    def __param_vary(captured_layers, scenario_id_per_layer):
        """ Check if parameters vary among captured (scenario) layers"""
        # parameters can only vary within a list containing more than one layer
        if len(captured_layers) > 1:
            first_layer = captured_layers[0]
            for i in range(1, len(captured_layers)):
                cur_layer = captured_layers[i]
                if __have_different_par(first_layer, cur_layer):
                    return True
        return False

    def __have_different_par(layer1, layer2):
        """
        Check if two captured (scenario) layers have different parameters
        :param layer1: first layer
        :param layer2: second layer
        :return: True if two captured (scenario) layers have different parameters,
        and False otherwise
        """
        layer1_par_shapes = generate_param_shapes_dict(layer1)
        layer2_par_shapes = generate_param_shapes_dict(layer2)
        # layers have matching (not different) parameters shapes if they
        # both don't have parameters
        if len(layer1_par_shapes.items()) == len(layer2_par_shapes.items()) == 0:
            return False
        # TODO: explicitly specify parameters reuse!
        # FOR NOW in all other cases parameters mismatch
        return True

    def __par_shapes_differ(layer1_par_shapes, layer2_par_shapes):
        """ Check if two layers have different parameters shapes"""
        # layers have different parameters shapes if they have different
        # number of parameters
        if len(layer1_par_shapes.items()) != len(layer2_par_shapes.items()):
            return True

        # compare parameters per-item
        for item in layer1_par_shapes.items():
            k, v = item
            # layer2 does not have the same parameter (name) as layer1
            if k not in layer2_par_shapes.keys():
                return True
            else:
                # layer2 gives different value to parameter (name) as layer1
                if v != layer2_par_shapes[k]:
                    return True

        return False

    def __ios_vary(captured_layers, scenario_id_per_layer):
        """ Check if sbrs inputs/outputs vary among captured (scenario) layers"""

        def __get_sbrs_input_connections(scn_layer, scn_id):
            """
            Get list of input SBRS connections used by a layer of a scenario
            :param scn_layer: layer of a scenario
            :param scn_id: id of scenario
            :return: list of input SBRS connections used by a layer of a scenario
            """
            scenario = sbrs_moc.scenarios[scn_id]
            scenario_input_connections = scenario.get_layer_input_connections(scn_layer)
            sbrs_input_connections = __get_sbrs_connections(scenario_input_connections, scn_id)

            return sbrs_input_connections

        def __get_sbrs_output_connections(scn_layer, scn_id):
            """
            Get list of input SBRS connections used by a layer of a scenario
            :param scn_layer: layer of a scenario
            :param scn_id: id of scenario
            :return: list of input SBRS connections used by a layer of a scenario
            """
            scenario = sbrs_moc.scenarios[scn_id]
            scenario_output_connections = scenario.get_layer_output_connections(scn_layer)
            sbrs_output_connections = __get_sbrs_connections(scenario_output_connections, scn_id)
            return sbrs_output_connections

        def __get_sbrs_connections(scn_connections, scn_id):
            """
            Get list of SBRS connections capturing scenario connections
            :param scn_connections list of scenario connections
            :param scn_id: id of scenario
            :return: list of SBRS connections capturing scenario connections
            """
            scenario = sbrs_moc.scenarios[scn_id]
            sbrs_connections = []
            for scenario_connection in scn_connections:
                scenario_connection_id = scenario.get_connection_id(scenario_connection)
                sbrs_connection = sbrs_moc.get_sbrs_connection(scn_id, scenario_connection_id)
                sbrs_connections.append(sbrs_connection)
            return sbrs_connections

        def __sbrs_connections_vary(sbrs_connections1, sbrs_connections2):
            """ check if lists of sbrs connections vary"""
            # lists vary if the number of connections mismatch between the lists
            if len(sbrs_connections1) != len(sbrs_connections2):
                return True
            # lists vary if, provided the same number of connections (see condition above)
            # the first list has a connection that the second list does not have
            for sbrs_con1 in sbrs_connections1:
                if sbrs_con1 not in sbrs_connections2:
                    return True
            # if none above is true, lists do not vary
            return False

        def __inputs_vary():
            """ check if lists of input connections vary"""
            first_layer_input_connections = __get_sbrs_input_connections(captured_layers[0],
                                                                         scenario_id_per_layer[0])
            for i in range(1, len(captured_layers)):
                layer_input_connections = __get_sbrs_input_connections(captured_layers[i],
                                                                       scenario_id_per_layer[i])
                if __sbrs_connections_vary(first_layer_input_connections, layer_input_connections):
                    return True
            return False

        def __outputs_vary():
            """ check if lists of output connections vary"""
            first_layer_output_connections = __get_sbrs_output_connections(captured_layers[0],
                                                                           scenario_id_per_layer[0])
            for i in range(1, len(captured_layers)):
                layer_output_connections = __get_sbrs_output_connections(captured_layers[i],
                                                                         scenario_id_per_layer[i])
                if __sbrs_connections_vary(first_layer_output_connections, layer_output_connections):
                    return True
            return False

        # main script
        # inputs/outputs can only vary within a list containing more than one layer
        if len(captured_layers) > 1:
            if __inputs_vary():
                return True
            if __outputs_vary():
                return True

        return False

    def __add_control_parameters():
        """ Add control parameters into the SBRS MoC """
        # SBRS moc does not need control parameters if it does not have
        # runtime-adaptive attributes
        if not adaptive_attributes:
            return

        for sbrs_layer in sbrs_moc.supergraph.get_layers():
            # list of all captured layers
            captured_layers = []
            # list of scenario ids per captured layer
            scenario_ids_per_captured_layer = []
            # list of all captured layers per scenario
            # captured_layers_per_scenario = {}
            for scenario_id in range(len(sbrs_moc.scenarios)):
                captured_scenario_layers = sbrs_moc.get_captured_scenario_layers(sbrs_layer, scenario_id)
                if len(captured_scenario_layers) > 0:
                    # captured_layers_per_scenario[scenario_id] = captured_scenario_layers
                    for scenario_layer in captured_scenario_layers:
                        captured_layers.append(scenario_layer)
                        scenario_ids_per_captured_layer.append(scenario_id)
            # sbrs layer does not need control parameters if it captures only one scenario layer
            if len(captured_layers) > 1:
                # check if I/O connections of this layer vary among captured layers
                # and therefore have to be runtime-adaptive
                if __ios_vary(captured_layers, scenario_ids_per_captured_layer):
                    sbrs_moc.control_node.add_control_parameter(sbrs_layer, AdaptiveAttribute.IO)

                if __param_vary(captured_layers, scenario_ids_per_captured_layer):
                    sbrs_moc.control_node.add_control_parameter(sbrs_layer, AdaptiveAttribute.WEIGHTS)

    #############
    # main script
    sbrs_moc = SBRSMoC("sbrs", scenarios, adaptive_attributes)
    __add_layers()
    __add_edges()
    __add_control_parameters()
    # TODO: add execution sequences!

    return sbrs_moc


####################
# suitable layer search


def find_suitable_sbrs_layer(sbrs_moc: SBRSMoC, scenario_layer):
    """
    Find layer (node) in a scenarios supergraph of the sbrs moc that can capture new scenario layer
    :param sbrs_moc: sbrs computational model
    :param scenario_layer: new scenario layer
    :return: layer (node) in a scenarios supergraph that can capture new scenario layer
    """
    sbrs_moc_adaptive_attributes = sbrs_moc.get_adaptive_attributes()
    for sbrs_layer in sbrs_moc.supergraph.get_layers():
        if layers_match(sbrs_layer, scenario_layer, sbrs_moc_adaptive_attributes):
            return sbrs_layer
    return None


def layers_match(layer1: Layer, layer2: Layer, adaptive_attributes: [AdaptiveAttribute], weights_match=False):
    """
    Checks if layers match with provided adaptive attributes, i.e.,
    if all layer attributes except of adaptive attributes match
    :param layer1 first layer
    :param layer2 second layer
    :param adaptive_attributes collection of run-time adaptive attributes
    :param weights_match: (flag) if True, parameters (weights) match between layer 1 and layer 2
    :return true if layers are equal with provided adaptive attributes and false otherwise
    """
    def __op_match():
        op_match = layer1.op == layer2.op and layer1.subop == layer2.subop
        return op_match

    def __par_match():
        # if parameters matching is explicitly specified, parameters match
        if weights_match is True:
            return True

        # if parameters are adaptive they match
        if AdaptiveAttribute.WEIGHTS in adaptive_attributes:
            return True

        # if layer 1 and layer 2 have no parameters, their parameters match
        par_shapes_1 = generate_param_shapes_dict(layer1)
        par_shapes_2 = generate_param_shapes_dict(layer2)
        if not par_shapes_1.items() and not par_shapes_2.items():
            return True

        return False

    def __hyp_match():
        hyp_match = layer1.stride == layer2.stride and layer1.fs == layer2.fs and layer1.pads == layer2.pads
        return hyp_match

    # main script
    l_match = __op_match() and __par_match() and __hyp_match()
    return l_match


############################
# suitable connection search

def find_suitable_sbrs_connection(sbrs_moc: SBRSMoC, scenario_id, scenario_connection):
    """
    Find connection (edge) in a scenarios supergraph of the sbrs moc that can capture new scenario connection
    :param sbrs_moc: sbrs computational model
    :param scenario_id: id of scenario, which contains the scenario connection
    :param scenario_connection: new scenario layer
    :return: layer (node) in a scenarios supergraph that can capture new scenario layer
    """
    def __captures_io_of_scenario_connection(sbrs_connection):
        """ Checks if sbrs connection src (input) and dst (output) capture src and dst of new scenario connection"""
        captures_io = False

        # check if source is matching
        sbrs_src_id = sbrs_connection.src.id
        required_sbrs_src_id = sbrs_moc.control_node.find_sbrs_layer_id(scenario_id, scenario_src_id)

        # if source is matching
        if sbrs_src_id == required_sbrs_src_id:
            # check if destination is matching
            sbrs_dst_id = sbrs_connection.dst.id
            required_sbrs_dst_id = sbrs_moc.control_node.find_sbrs_layer_id(scenario_id, scenario_dst_id)
            # destination matches too
            if sbrs_dst_id == required_sbrs_dst_id:
                captures_io = True

        return captures_io

    def __occupied_by_a_necessary_duplicate(sbrs_connection):
        """ Checks if connection is already occupied by a necessary duplicate. Necessary duplicates of input
        connections may exist for multi-input DNN layers. E.g. if a DNN layer has 2x inputs,
        and both inputs are captured by the same SBRS MoC layer, the DNN layer will have
        two input connections with same source and destination. These two input connections have to
        be represented as two different connections to preserve the scenario functionality
        """
        occupied_by_necessary_duplicate = False
        # get list of (other) connections from the same scenario as the (new)
        # scenario connection, that occupy sbrs connection
        captured_scenario_connections = sbrs_moc.get_captured_scenario_connections(sbrs_connection, scenario_id)

        # if such connections exist, they are necessary duplicates,
        # occupying the connections
        if len(captured_scenario_connections) > 0:
            occupied_by_necessary_duplicate = True

        return occupied_by_necessary_duplicate

    #############
    # main script
    scenario_src_id = scenario_connection.src.id
    scenario_dst_id = scenario_connection.dst.id

    for sbrs_con in sbrs_moc.supergraph.get_connections():
        if __captures_io_of_scenario_connection(sbrs_con):
            if not __occupied_by_a_necessary_duplicate(sbrs_con):
                return sbrs_con
    return None



