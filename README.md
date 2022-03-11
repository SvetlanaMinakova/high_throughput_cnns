## Code for scientific paper Combining Task- and Data-level Parallelism for High-Throughput CNN Inference on Embedded CPUs-GPUs MPSoCs"
### authors: Svetlana Minakova, Erqian Tang, and Todor Stefanov
paper published In Proc. "20th International Conference on Embedded Computer Systems: Architectures, Modeling and Simulation (SAMOS'20)", pp. 18-35, Pythagoreio, Samos Island, Greece, July 05-09, 2020."



## Abstract
The code is aimed at increasing throughput of a Convolutional Neural Network (CNN), executed on an edge (mobile or embedded) platform.
It represents a methodology, based on the publication above. The methodology exploits two types of 
parallelism: data-level parallelism and task-level parallelism, available in a CNN, to efficiently 
distribute (map) the computations within the CNN to the computational resources of an edge (mobile or embedded) platform.
The CNN distribution (mapping) is considered efficient if it ensures high CNN throughput. 
To find an efficient CNN mapping, our proposed methodology performs an automated Design Space Exploration (DSE), 
based on a Genetic Algorithm (GA) as proposed in the original paper on a Greedy Algorithm (only added to the tool).
Exploitation of task-level (pipeline) parallelism together with data-level parallelism is the main novel feature of our proposed methodology.
This feature distinguishes our methodology from the existing DL frameworks and other methodologies 
that utilize only task-level (pipeline) parallelism or only data-level parallelism, available in a CNN, to ensure high CNN throughput. 
Thanks to the combined use of task- and data-level parallelism, our proposed methodology takes 
full advantage of all computational resources that are available on the edge platform, 
and ensures very high CNN throughput. To evaluate our proposed methodology, we perform experiments 
where we apply our methodology to real-world CNNs from the Open Neural Network Exchange Format (ONNX)
models zoo and execute the CNNs on the NVIDIA Jetson TX2 edge platform. 
We compare the throughput demonstrated by the CNNs mapped on the Jetson TX2 platform using:
our proposed methodology and the best-known and state-of-the-art TensorRT DL framework for the Jetson TX2 edge platform. 
The experimental results shown that ~20% higher throughput is achieved, when the CNNs are mapped using our methodology. 
We note  that, while the proposed methodology is applicable for a wide range of CNNs and platforms, in the current tool implementation teh code generations is only supported for ARM-CL (ARM CPU) + TensoRT (GPU) DL frameworks and is only meant to be executed on NVIDIA Jetson platforms.


## requirements
* python 3.6+
* onnx 1.8
* [optionally] keras (https://keras.io/) for keras cnn models

## inputs and outputs
*Examples of the tool inputs and outputs are located in ./input_examples folder*

The tool accepts as input:
* a DNN in ONNX (see https://onnx.ai/) format. *For an example see ./input_examples/dnn/mnist.onnx*
* a target platform architecture in json format. *For an example see ./input_examples/architecture/jetson.json*
* step-specific intermediate files in .json format, generated by the tool. *For examples see ./input_examples/intermediate folder*

The tool provides as output:
A final CNN inference model (in .json format), which describes how a CNN is partitioned,
mapped and scheduled for efficient CNN execution on the target edge platform.


## Toolflow

Tool consists of several steps, executed one after another. 
The steps are represented as API scripts, located in the root directory of the project.
Shortly, the steps are:
1) Generation of an SDF (task graph) from an input CNN (dnn_to_sdf_task_graph.py)
2) Generation of a per-layer execution time (latency) evaluation template (sdf_latency_eval_template.py)
3) Filling the generated per-layer execution time (latency) template with real latency estimations (this step is performed manually!)
4) Generation of efficient mapping of the input CNN onto target edge platform architecture (map_and_partition.py)
5) Generation of final (CSDF) application model (generate_final_app_model.py)
6) Generation of executable code from the input DNN and the final application model (see generate_code*.py)

Below, more detailed explanation for every step (script) is given.

### Step 1: Generation of an SDF (task graph) from an input CNN (dnn_to_sdf_task_graph.py)

This script generates a static dataflow (task graph) from the input CNN model. 
The task graph shows the tasks, performed to execute CNN inference as well as connections between these tasks.
One task corresponds to one or more layers of the input CNN. 

Example use

*$ python ./dnn_to_sdf_task_graph.py --cnn ./input_examples/dnn/mnist.onnx -o ./output/mnist*

Example output: see ./output/mnist/task_graph.json or ./input_examples/intermediate/mnist/task_graph.json

### Step 2: Generation of a per-layer execution time (latency) evaluation template (sdf_latency_eval_template.py)










