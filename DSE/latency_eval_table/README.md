# LATENCY EVAL TABLE

Latency eval table shows execution time of application tasks on the processors of the target platform. It is a matrix [M]x[N] where
 * every element matrix[m][n] contains execution time of task n on processor of type m
 * m in [0, len (processor_types_distinct)-1] represents type of a processor, available on a target platform
 * n in [0, layers_num] represents a task in the application task graph


Latency eval table is required for exploring high-throughput execution of a DNN on a target edge platform. There are several ways to build eval matrix:
* from direct measurements on the platform, represented as a .json file of a specific format
* from look-up Table (LUT)
* moc (filled with random numbers, and used ONLY for test purposes)
* using number of floating-point operations (FLOPS) as a merit of task execution time

With respect to the ways of building the eval matrix, we offer several eval table (et) builders. For example, lut_et_builder builds the evaluation table, using the luuk-up table (LUT)