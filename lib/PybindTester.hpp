#include <string>
#include <utility>
#include <iostream>

using namespace std;

class PybindTester{
    private:
        int i;
        double d;
        bool b;
        string s;
        pair<int, int> p;
        pair<pair<int, int>, pair<int, int>> pp;

        void populate(int i, double d, bool b, string s, pair<int, int> p,
                                pair<pair<int, int>, pair<int, int>> pp) {
            this->i = i;
            this->d = d;
            this->b = b;
            this->s = s;
            this->p = p;
            this->pp = pp;
        };

    public:
        PybindTester(){};

        PybindTester(int i, double d, bool b, string s, pair<int, int> p,
                                pair<pair<int, int>, pair<int, int>> pp):
            i(i), d(d), b(b), s(s), p(p), pp(pp) {};

        static PybindTester from_something(int i, double d, bool b, string s, pair<int, int> p,
                                pair<pair<int, int>, pair<int, int>> pp) {
            PybindTester pt = PybindTester();
            pt.populate(i, d, b, s, p, pp);
            return pt;
        };

        void print_PybindTester() {
            cout << "i  : " << i << endl;
            cout << "d  : " << d << endl;
            cout << "b  : " << b << endl;
            cout << "s  : " << s << endl;
            cout << "p  : (" << p.first << ", " << p.second << ")" << endl;
            cout << "pp : ((" << pp.first.first << ", " << pp.first.second << "), (" << pp.second.first << ", " << pp.second.second << "))" << endl;
        };
};
