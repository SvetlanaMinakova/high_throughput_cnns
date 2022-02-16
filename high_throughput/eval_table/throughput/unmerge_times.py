import json


def unmerge(names, complexities, times):
    """
    Unmerge times evaluation for trt eval_table times
    """
    unmerged_names = []
    unmerged_times = []
    timeid = 0
    for name in names:
        if "||" in name:
           splitted_name = name.split("||")
           splitted_time = []

           complexity_start_id = unmerged_names.__len__()
           merged_time = times[timeid]
           merged_complexity = 0
           separate_complexity = []

           for i in range (0, splitted_name.__len__()):
               separate_complexity.append(complexities[complexity_start_id + i])
               merged_complexity += complexities[complexity_start_id + i]

           for i in range (0, splitted_name.__len__()):
               unmerged_names.append(splitted_name[i])
               separated_time = separate_complexity[i]/merged_complexity * merged_time
               unmerged_times.append(separated_time)

        else:
            #print(name)
            unmerged_names.append(name)
            unmerged_times.append(times[timeid])

        timeid = timeid + 1

    unmerged = []
    unmerged.append(unmerged_names)
    unmerged.append(unmerged_times)
    return unmerged

def unmerge_from_json(original_file):
    with open(original_file, 'r') as file:
        if file is None:
            raise FileNotFoundError
        evals = json.load(file)
        names = evals["layers"]
        complexities = evals["task_complexities_share"]
        times = evals["gpu_time"]
        unmerged = unmerge(names, complexities, times)
        print("\"names\"", unmerged[0], ",")
        print("\"times\"", unmerged[1])

def test():
    names = ["Conv0", "Conv1 || Conv2 || Conv3", "Conv4", "Conv5 || Conv6"]
    complexities = [1, 2, 3, 4, 5, 1, 2]
    times = [0.1, 0.9, 0.5, 0.3]

    unmerged = unmerge(names, complexities, times)
    print(unmerged)

    unmerge_from_json("/vol/home/minakovas/PycharmProjects/ga_mapping/data/densenet/e0_unmerged.json_converters")