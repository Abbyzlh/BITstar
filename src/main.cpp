#include "../include/BatchInformedTree.h"
#include <iostream>
#include <cmath>

int main(int argc, char **argv) {
    std::cout << "Hello, World!" << std::endl;
    BatchInformedTree* bit=new BatchInformedTree(4,10);
    std::cout<<bit->dimension<<std::endl;

    return 0;
}


