# High-throughput DNN inference

This module implements techniques for high throughput DNN inference at the edge (see Chapter 3).

## Theory

Nowadays Convolutional Neural Networks (CNNs) are widely used
to perform various tasks in areas such as computer vision or natural language
processing. Some of CNN applications require high-throughput execution of
the CNN inference, on embedded devices, and many modern embedded devices
are based on CPUs-GPUs multi-processor systems-on-chip (MPSoCs). Ensuring high-throughput execution of the CNN inference on embedded CPUs-GPUs
MPSoCs is a complex task, which requires efficient utilization of both task-level
(pipeline) and data-level parallelism, available in a CNN. However, the existing
Deep Learning frameworks utilize only task-level (pipeline) or only data-level
parallelism, available in a CNN, and do not take full advantage of all embedded
MPSoC computational resources. Therefore, in this paper, we propose a novel
methodology for efficient execution of the CNN inference on embedded CPUs-
GPUs MPSoCs. In our methodology, we ensure efficient utilization of both task-
level (pipeline) and data-level parallelism, available in a CNN, to achieve high-
throughput execution of the CNN inference on embedded CPUs-GPUs MPSoCs.

For more details see Chapter 3.

## Steps
Increasing throughput of throughout of a CNN involves following steps:

1) ***Conversion of DNN into a TaskGraph***: DNN model does not specify tasks, performed during the DNN inference. Depending on the DL framework, one task can correspond to a layer, a part of a layer or several layers at once. To efficiently exploit parallelism
within a CNN, we need to precisely specify tasks, executed during the CNN inference. To do so, we introduce a TaskGraph, where every node is a task. In our framework, every task corresponds to one or several CNN layers. Automated conversion of a DNN into the task graph is performed using converters.dnn_to_task_graph.py converter.
2) ***Estimation of execution time (latency) in milliseconds (ms) of every DNN task on every processor of the target platform***.
3) ***Search of efficient DNN mapping***, i.e., distribution of DNN tasks among the target edge platform processors.

### Estimation of execution time (latency) in milliseconds (ms) of every DNN task on every processor of the target platform
This can be done in several ways (from most precise to least precise):

* using direct measurements on the platform
* using look-up Tables (LUTs)
* using number of floating-point operations (FLOPS) as a merit of task execution time