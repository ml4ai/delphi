#include "ibex.h"
#include <iostream>
//#include <fstream>

using namespace std;
using namespace ibex;

int main()
{

  int N = 2;
  float tmax_lb[N], tmax_ub[N], tmin_lb[N], tmin_ub[N];


  stack<IntervalVector> s;
  int num = 0;
  while(num < N)
  {

    tmax_lb[num] = 2.0 + num;
    tmax_ub[num] = tmax_lb[num] + 2;

    tmin_lb[num] = 2.0 + num;
    tmin_ub[num] = tmin_lb[num] + 2;

    double _box[2][2] = {{tmax_lb[num], tmax_ub[num]}, {tmin_lb[num], tmin_ub[num]}};
    IntervalVector box(2, _box);

    s.push(box);

    num++;
  }

  cout << s.pop() << endl;
  //Set s1(s);

  return 0;
}
