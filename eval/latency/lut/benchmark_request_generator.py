"""
Module analyzes DNNs and generates request for LUT-benchmarks
"""


def generate_bm_request(dnns, group_by_op=True):
    """
    Request benchmark for a set of DNNs
    :param dnns: set of DNNs
    :param group_by_op: group request by dnn operator, performed by the layer
    :return: benchmark for the set of DNNs
    """
    from eval.latency.lut.LUT_builder import build_lut_tree
    dnns_lut = build_lut_tree(dnns)
    dnns_lut_as_table = dnns_lut.get_as_table()
    # dnns_lut.print_as_table()
    prev_op = "none"
    for record in dnns_lut_as_table:
        cur_op = record["op"]
        bm_config = generate_bm_layer_config(record)
        if cur_op != prev_op:
            print("")
            print("   ", "// operator: ", cur_op)
            print("    Config", bm_config)
        else:
            print("   ", bm_config)
        print("   ", "configs.push_back(config);")
        prev_op = cur_op


def generate_bm_layer_config(lut_record):
    """
    Generate config for evaluating a LUT record
    :param lut_record: record in LUT
    :return: config for evaluating a LUT record
    """

    if lut_record["op"] == "conv":
        bm_config = "config = Config(" + \
                    str(lut_record['iw']) + ", " + \
                    str(lut_record['ifm']) + ", " + \
                    str(lut_record['ofm']) + ", " + \
                    str(lut_record['fs']) + ", " + \
                    str(int(lut_record['hpad'])) + ", " + \
                    str(lut_record['stride']) + ");"
        return bm_config

    if lut_record["op"] in ["matmul", "gemm", "fc"]:
        bm_config = "config = Config(" + \
                    str(lut_record['iw']) + ", " + \
                    str(lut_record['ifm']) + ", " + \
                    str(lut_record['ofm']) + ", " + \
                    str(lut_record['fs']) + ");"
        return bm_config

    # default
    bm_config = "config = Config(" + \
                str(lut_record['iw']) + ", " + \
                str(lut_record['ifm']) + ", " + \
                str(lut_record['ofm']) + ", " + \
                str(lut_record['fs']) + ", " + \
                str(lut_record['hpad']) + ", " + \
                str(lut_record['stride']) + ");"

    return bm_config




