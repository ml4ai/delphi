#include "PybindTester.hpp"

int main() {
    PybindTester pt = PybindTester(1, 2.5, true, "hello", make_pair(3, 4), make_pair(make_pair(5, 6), make_pair(7, 8)));

    pt.print_PybindTester();

    PybindTester pt2= PybindTester::from_something(1, 2.5, true, "hello", make_pair(3, 4), make_pair(make_pair(5, 6), make_pair(7, 8)));
    
    pt2.print_PybindTester();
}
