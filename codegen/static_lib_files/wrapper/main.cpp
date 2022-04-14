#include <iostream>
#include "example_apps/three_subnets_example/original/CNNAppOriginal.h"
#include "example_apps/three_subnets_example/abstract/CNNAppAbstract.h"
int main() {
    // CNNAppOriginal::run(100, 0, 0, 0);
    CNNAppAbstract::run(100,0,0,0);
    return 0;
}
