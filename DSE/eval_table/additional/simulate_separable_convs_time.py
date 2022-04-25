from models.TaskGraph import parse_app_graph_json
from models.edge_platform import get_jetson


def is_grouped(layer_name, grouped_convs:{}):
    return layer_name in grouped_convs.keys()


def get_group(layer_name, grouped_convs:{}):
    if layer_name in grouped_convs.keys():
        return grouped_convs[layer_name]
    return 1


def get_layer_id(eval_name, app_graph):
    for layer_id in range(len(app_graph.jobs_per_task)):
        layer = app_graph.jobs_per_task[layer_id]
        if eval_name.startswith(layer):
            return layer_id
    return -1


def test():
    evals = ["node_Conv1", "node_Sigmoid2", "node_Mul3", "node_Conv4", "node_Sigmoid5", "node_Mul6",
               "node_ReduceMean7", "node_Conv8", "node_Sigmoid9", "node_Mul10", "node_Conv11", "node_Sigmoid12",
               "node_Mul13", "node_Conv14", "node_Conv15", "node_Sigmoid17", "node_Mul18", "node_Conv19",
               "node_Sigmoid20", "node_Mul21", "node_ReduceMean22", "node_Conv23", "node_Sigmoid24", "node_Mul25",
               "node_Conv26", "node_Sigmoid27", "node_Mul28", "node_Conv29", "node_Conv30", "node_Sigmoid32",
               "node_Mul33", "node_Conv34", "node_Sigmoid35", "node_Mul36", "node_ReduceMean37", "node_Conv38",
               "node_Sigmoid39", "node_Mul40", "node_Conv41", "node_Sigmoid42", "node_Mul43", "node_Conv44_node_Add45",
               "node_Conv46", "node_Sigmoid47", "node_Mul48", "node_Conv49", "node_Sigmoid50", "node_Mul51",
               "node_ReduceMean52", "node_Conv53", "node_Sigmoid54", "node_Mul55", "node_Conv56", "node_Sigmoid57",
               "node_Mul58", "node_Conv59", "node_Conv60", "node_Sigmoid62", "node_Mul63", "node_Conv64",
               "node_Sigmoid65", "node_Mul66", "node_ReduceMean67", "node_Conv68", "node_Sigmoid69", "node_Mul70",
               "node_Conv71", "node_Sigmoid72", "node_Mul73", "node_Conv74_node_Add75", "node_Conv76", "node_Sigmoid77",
               "node_Mul78", "node_Conv79", "node_Sigmoid80", "node_Mul81", "node_ReduceMean82", "node_Conv83",
               "node_Sigmoid84", "node_Mul85", "node_Conv86", "node_Sigmoid87", "node_Mul88", "node_Conv89",
               "node_Conv90", "node_Sigmoid92", "node_Mul93", "node_Conv94", "node_Sigmoid95", "node_Mul96",
               "node_ReduceMean97", "node_Conv98", "node_Sigmoid99", "node_Mul100", "node_Conv101", "node_Sigmoid102",
               "node_Mul103", "node_Conv104_node_Add105", "node_Conv106", "node_Sigmoid107", "node_Mul108",
               "node_Conv109", "node_Sigmoid110", "node_Mul111", "node_ReduceMean112", "node_Conv113",
               "node_Sigmoid114", "node_Mul115", "node_Conv116", "node_Sigmoid117", "node_Mul118",
               "node_Conv119_node_Add120", "node_Conv121", "node_Sigmoid122", "node_Mul123", "node_Conv124",
               "node_Sigmoid125", "node_Mul126", "node_ReduceMean127", "node_Conv128", "node_Sigmoid129", "node_Mul130",
               "node_Conv131", "node_Sigmoid132", "node_Mul133", "node_Conv134", "node_Conv135", "node_Sigmoid137",
               "node_Mul138", "node_Conv139", "node_Sigmoid140", "node_Mul141", "node_ReduceMean142", "node_Conv143",
               "node_Sigmoid144", "node_Mul145", "node_Conv146", "node_Sigmoid147", "node_Mul148",
               "node_Conv149_node_Add150", "node_Conv151", "node_Sigmoid152", "node_Mul153", "node_Conv154",
               "node_Sigmoid155", "node_Mul156", "node_ReduceMean157", "node_Conv158", "node_Sigmoid159", "node_Mul160",
               "node_Conv161", "node_Sigmoid162", "node_Mul163", "node_Conv164_node_Add165", "node_Conv166",
               "node_Sigmoid167", "node_Mul168", "node_Conv169", "node_Sigmoid170", "node_Mul171", "node_ReduceMean172",
               "node_Conv173", "node_Sigmoid174", "node_Mul175", "node_Conv176", "node_Sigmoid177", "node_Mul178",
               "node_Conv179", "node_Conv180", "node_Sigmoid182", "node_Mul183", "node_Conv184", "node_Sigmoid185",
               "node_Mul186", "node_ReduceMean187", "node_Conv188", "node_Sigmoid189", "node_Mul190", "node_Conv191",
               "node_Sigmoid192", "node_Mul193", "node_Conv194_node_Add195", "node_Conv196", "node_Sigmoid197",
               "node_Mul198", "node_Conv199", "node_Sigmoid200", "node_Mul201", "node_ReduceMean202", "node_Conv203",
               "node_Sigmoid204", "node_Mul205", "node_Conv206", "node_Sigmoid207", "node_Mul208",
               "node_Conv209_node_Add210", "node_Conv211", "node_Sigmoid212", "node_Mul213", "node_Conv214",
               "node_Sigmoid215", "node_Mul216", "node_ReduceMean217", "node_Conv218", "node_Sigmoid219", "node_Mul220",
               "node_Conv221", "node_Sigmoid222", "node_Mul223", "node_Conv224_node_Add225", "node_Conv226",
               "node_Sigmoid227", "node_Mul228", "node_Conv229", "node_Sigmoid230", "node_Mul231", "node_ReduceMean232",
               "node_Conv233", "node_Sigmoid234", "node_Mul235", "node_Conv236", "node_Sigmoid237", "node_Mul238",
               "node_Conv239", "node_Conv240", "node_Sigmoid242", "node_Mul243", "node_ReduceMean244", "node_MatMul245",
               "node_Softmax246"]
    gpu_time = [0.213751, 0.127316, 0.122121, 0.424153, 0.130088, 0.121743, 0.166245, 0.011015, 0.00559424, 0.00371712,
            0.0118554, 0.00304128, 0.123531, 0.141006, 0.376892, 0.368019, 0.351748, 1.04418, 0.0811475, 0.0900583,
            0.0950368, 0.0130477, 0.00584704, 0.00375872, 0.00785728, 0.00303104, 0.0915443, 0.0805658, 0.168881,
            0.118659, 0.132083, 1.70239, 0.116524, 0.132207, 0.105228, 0.0130144, 0.00580864, 0.00376064, 0.0141798,
            0.00342784, 0.135974, 0.104053, 0.169814, 0.118374, 0.132047, 1.95708, 0.0320864, 0.0356806, 0.0499814,
            0.0125722, 0.00597632, 0.00383232, 0.0147917, 0.00344704, 0.036896, 0.0513357, 0.0688211, 0.0536512,
            0.0573242, 4.04209, 0.0526099, 0.0567712, 0.058384, 0.0172282, 0.0058784, 0.00382592, 0.0137626, 0.00346624,
            0.0565517, 0.0722483, 0.0789114, 0.053433, 0.0568666, 0.492536, 0.013479, 0.0094912, 0.0194278, 0.0176218,
            0.00488576, 0.00353216, 0.013351, 0.00350208, 0.015511, 0.054487, 0.0664659, 0.0259923, 0.0300179, 1.05258,
            0.0254458, 0.0284486, 0.0320762, 0.0222246, 0.00585088, 0.00374336, 0.014569, 0.00402368, 0.0276134,
            0.0926509, 0.065975, 0.0256384, 0.0295667, 1.05839, 0.0253766, 0.0286893, 0.031769, 0.0225376, 0.00590528,
            0.00375232, 0.0101491, 0.00380992, 0.0277427, 0.092745, 0.065792, 0.0245075, 0.0292275, 5.33296, 0.0254342,
            0.0295258, 0.0319098, 0.0225926, 0.00597952, 0.00369344, 0.0100166, 0.00384064, 0.0277965, 0.0952096,
            0.105932, 0.0402963, 0.0393485, 10.8396, 0.0387674, 0.0395738, 0.0371878, 0.0276493, 0.00610304, 0.00373376,
            0.0124429, 0.00514432, 0.0378989, 0.127336, 0.104997, 0.0399149, 0.0389446, 10.8568, 0.0377344, 0.0393952,
            0.037399, 0.0285517, 0.00594688, 0.003792, 0.0123238, 0.00516352, 0.0379507, 0.126106, 0.106222, 0.0374618,
            0.0389574, 4.40728, 0.0151155, 0.00704512, 0.0171354, 0.0274938, 0.00336448, 0.00268288, 0.012448,
            0.0051712, 0.0100774, 0.0549402, 0.129829, 0.0143149, 0.00984512, 7.89241, 0.0167277, 0.0101446, 0.0192717,
            0.0403373, 0.00532608, 0.00366656, 0.0198605, 0.00606656, 0.0159834, 0.172965, 0.1265, 0.0141805,
            0.00972544, 8.0601, 0.017033, 0.0101715, 0.0194195, 0.0421696, 0.00527104, 0.00365952, 0.0195469,
            0.00606848, 0.0158592, 0.174893, 0.132352, 0.0144723, 0.0145946, 8.01226, 0.0168915, 0.0103699, 0.0192595,
            0.0422656, 0.00531264, 0.00371904, 0.0197338, 0.00608192, 0.0160166, 0.174231, 0.129808, 0.0145658,
            0.0100941, 2.4836, 0.0166406, 0.0103309, 0.0193574, 0.0401165, 0.0052736, 0.00365184, 0.0198131, 0.00608256,
            0.015863, 0.129411, 0.199002, 0.0155674, 0.011945, 0.019017, 0.156987, 0.0169574]

    architecture = get_jetson()
    gpu_id = 5
    efficient_net = "/home/svetlana/Pycharm_old_HDD/ga_mapping/data/efficientnet_B0/efficientnet_B0Topology.json_converters", ""
    grouped_convs = {"node_Conv4": 32, "node_Conv19": 96, "node_Conv34": 144, "node_Conv49": 144, "node_Conv64": 240,
                     "node_Conv79": 240, "node_Conv94": 480, "node_Conv109": 480, "node_Conv124": 480,
                     "node_Conv139": 672, "node_Conv154": 672, "node_Conv169": 672, "node_Conv184": 1152,
                     "node_Conv199": 1152, "node_Conv214": 1152, "node_Conv229": 1152}


    app_path, eval_path = efficient_net
    app_graph = parse_app_graph_json(app_path)
    reduced_times = []
    for eval_id in range(len(evals)):
        eval = evals[eval_id]
        layer_id = get_layer_id(eval, app_graph)
        layer = app_graph.jobs_per_task[layer_id]
        group = get_group(layer, grouped_convs)
        # print(eval_table, "corresponds to layer", layer)
        original_time = gpu_time[eval_id]
        time = original_time/group
        time = time/10
        reduced_times.append(time)
        # print("time: ", original_time, "-->", time)


    # print("lens: ", len(gpu_time), len(reduced_times))
    print(reduced_times)
    gpu_time_full = sum(gpu_time)
    gpu_time = sum(reduced_times)
    print("GPU time (ms): ", gpu_time)





    """
    for layer in app_graph.layers:
        group = get_group(layer, grouped_convs)
        print(layer, ":",  group)
    """

    # time_eval_matrix = build_eval_table(eval_path, architecture.processors_types_distinct, app_graph)


test()