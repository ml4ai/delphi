/*
    * A simple program to check whether Delphi has been compiled with OpenMP
 */

#include "AnalysisGraph.hpp"

int main() {
    AnalysisGraph::check_multithreading();
    return(0);
}
