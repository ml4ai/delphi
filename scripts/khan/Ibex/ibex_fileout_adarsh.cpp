#include "ibex.h"
#include "vibes.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace ibex;

// int n=5;

void contract_and_draw(Ctc& c, IntervalVector& box, const char* color, ofstream& file) {

  IntervalVector savebox = box;
  IntervalVector* result;

  c.contract(box);

  int n = savebox.diff(box, result);


  for (int i = 0; i < n; i++) {

    if (strncmp(color, "k[r]", 4)) {
      vibes::drawBox(result[i][0].lb(),
                     result[i][0].ub(),
                     result[i][1].lb(),
                     result[i][1].ub(),
                     color);
      file << "tmax (lb):" << result[i][0].lb() << endl;
    }
  }
  delete[] result;
}

int main() {
  vibes::beginDrawing();
  vibes::newFigure("petpt");

  Variable tmax;
  Variable tmin;
  Variable msalb;
  Variable xhlai;
  Variable srad;

  Function td(tmax, tmin, 0.6 * tmax + 0.4 * tmin);

  Function albedo0(msalb, msalb);
  Function albedo1(msalb, xhlai, 0.23 - (0.23 - msalb) * exp(-0.75 * xhlai));
  Function albedo(
      msalb, xhlai, chi(xhlai, albedo0(msalb), albedo1(msalb, xhlai)));

  Function slang(srad, srad * 23.923);

  double _x0[2][2] = {{16.1, 36.7}, {0.0, 23.9}};
  IntervalVector x0(2, _x0);
  Interval z1 = td.eval(x0);

  double _y0[2][2] = {{0.0, 1.0}, {0.0, 4.77}};
  IntervalVector y0(2, _y0);
  Interval z2 = albedo.eval(y0);

  double _y1[1][2] = {2.45, 27.8};
  IntervalVector y1(1, _y1);
  Interval z3 = slang.eval(y1);

  cout << "td values :" << z1 << endl;
  cout << "albedo values :" << z2 << endl;
  cout << "slang values :" << z3 << endl;

  Function eeq1(tmax, tmin, td(tmax, tmin) + 29.0);
  // Interval z4 = eeq1.eval(x0);

  // cout << "eeq1 values :" << z4 << endl;

  // double _y2[4][2] = {{16.1,36.7},{0.0,23.9}, {0.0, 1.0}, {0.0, 4.77}};
  // IntervalVector y2(4,_y2);

  Function eeq2(tmax,
                tmin,
                msalb,
                xhlai,
                (0.00020410 - 0.000183 * albedo(msalb, xhlai)) *
                    eeq1(tmax, tmin));
  // Function eeq2(tmax, tmin, msalb, xhlai,
  // (2.04*pow(10,-4)-1.83*pow(10,-4)*albedo(msalb, xhlai))*eeq1(tmax, tmin));
  // Interval z5 = eeq2.eval(y2);

  // cout << "eeq2 values :" << z5 << endl;

  // double _y3[2][2] = {{0.0, 1.0}, {0.0, 4.77}};
  // IntervalVector y3(2,_y3);
  // Function eeq3(msalb, xhlai, (2.04*pow(10,-4)-1.83*pow(10,-4)*albedo(msalb,
  // xhlai))); Interval z6 = eeq3.eval(y3);

  // cout << "eeq3 values :" << z6 << endl;

  Function eeq(tmax,
               tmin,
               msalb,
               xhlai,
               srad,
               slang(srad) * eeq2(tmax, tmin, msalb, xhlai));

  double _y2[5][2] = {
      {16.1, 36.7}, {0.0, 23.9}, {0.0, 1.0}, {0.0, 4.77}, {2.45, 27.8}};
  IntervalVector y2(5, _y2);
  Interval z5 = eeq.eval(y2);

  cout << "eeq values :" << z5 << endl;

  Function eo0(tmax,
               tmin,
               msalb,
               xhlai,
               srad,
               eeq(tmax, tmin, msalb, xhlai, srad) * 1.1);

  Function eo1(tmax,
               tmin,
               msalb,
               xhlai,
               srad,
               chi(tmax - 35,
                   eo0(tmax, tmin, msalb, xhlai, srad),
                   eeq(tmax, tmin, msalb, xhlai, srad) *
                       ((tmax - 35.0) * 0.05 + 1.1))); // tmax > 35
  Function eo2(tmax,
               tmin,
               msalb,
               xhlai,
               srad,
               chi(5 - tmax,
                   eo0(tmax, tmin, msalb, xhlai, srad),
                   eeq(tmax, tmin, msalb, xhlai, srad) * 0.01 *
                       exp(0.18 * (tmax + 20.0)))); // tmax < 5

  //  Function condition(tmax, ibex::max(5-tmax, tmax -35));
  Function eonew(
      tmax,
      tmin,
      msalb,
      xhlai,
      srad,
      chi(tmax - 35,
          chi(5 - tmax,
              eo0(tmax, tmin, msalb, xhlai, srad),
              eeq(tmax, tmin, msalb, xhlai, srad) * 0.01 *
                  exp(0.18 * (tmax + 20.0))),
          eeq(tmax, tmin, msalb, xhlai, srad) *
              ((tmax - 35.0) * 0.05 + 1.1))); // tmax > 35 && tmin < 5

  Function eo(tmax,
              tmin,
              msalb,
              xhlai,
              srad,
              ibex::max(0.0001, eonew(tmax, tmin, msalb, xhlai, srad)));

  double _box[5][2] = {
      {16.1, 36.7}, {0.0, 23.9}, {0.0, 1.0}, {0.0, 4.77}, {2.45, 27.8}};
  IntervalVector box(5, _box);

  stack<IntervalVector> s;

  double eps = 1.0;

  s.push(box);

  NumConstraint c1(tmax,
                   tmin,
                   msalb,
                   xhlai,
                   srad,
                   eo(tmax, tmin, msalb, xhlai, srad) <= 10.0);
  NumConstraint c2(tmax,
                   tmin,
                   msalb,
                   xhlai,
                   srad,
                   eo(tmax, tmin, msalb, xhlai, srad) >= 5.5);

  NumConstraint c3(tmax,
                   tmin,
                   msalb,
                   xhlai,
                   srad,
                   eo(tmax, tmin, msalb, xhlai, srad) > 10.0);
  NumConstraint c4(
      tmax, tmin, msalb, xhlai, srad, eo(tmax, tmin, msalb, xhlai, srad) < 5.5);

  // CtcFwdBwd c(eo(tmax, tmin, msalb, xhlai, srad));
  CtcFwdBwd out1(c1);
  CtcFwdBwd out2(c2);
  CtcFwdBwd in1(c3);
  CtcFwdBwd in2(c4);

  CtcCompo outside(out1, out2);

  CtcUnion inside(in1, in2);

  ofstream myFile;
  myFile.open("tmax_tmin_intervals.txt");
  while (!s.empty()) {
    IntervalVector box = s.top();
    // cout << box << endl;
    s.pop();
    contract_and_draw(outside, box, "k[r]", myFile);
    contract_and_draw(inside, box, "k[g]", myFile);
    if (!box.is_empty() && box.max_diam() > eps) {
      int i = box.extr_diam_index(false);
      pair<IntervalVector, IntervalVector> p = box.bisect(i);
      s.push(p.first);
      s.push(p.second);
    }
  }
  myFile.close();

  // inside.contract(box);
  // cout << "parameter values:" << box << endl;

  // double deltax1 = box[0].diam()/n;
  // double deltax2 = box[1].diam()/n;
  // double deltax3 = box[2].diam()/n;
  // double deltax4 = box[3].diam()/n;
  // double deltax5 = box[4].diam()/n;

  // for(int i1=0; i1<n; i1++)
  // for(int i2=0; i2<n; i2++)
  // for(int i3=0; i3<n; i3++)
  // for(int i4=0; i4<n; i4++)
  // for(int i5=0; i5<n; i5++){
  // IntervalVector box2(5);
  // box2[0] = Interval(box[0].lb()+i1*deltax1,box[0].lb()+(i1+1)*deltax1);
  // box2[1] = Interval(box[1].lb()+i2*deltax2,box[1].lb()+(i2+1)*deltax2);
  // box2[2] = Interval(box[2].lb()+i3*deltax3,box[2].lb()+(i3+1)*deltax3);
  // box2[3] = Interval(box[3].lb()+i4*deltax4,box[3].lb()+(i4+1)*deltax4);
  // box2[4] = Interval(box[4].lb()+i5*deltax5,box[4].lb()+(i5+1)*deltax5);
  // Interval petpt_eo = eo.eval(box2);

  // cout << "petpt values :" << petpt_eo << endl;
  //}

  // Interval z = eo.eval(x);

  // Interval z7 = eo0.eval(x);
  // Interval z8 = eo1.eval(x);
  // Interval z9 = eo2.eval(x);

  // cout << "petpt values :" << z << endl;
  // cout << "petpt eo0 values :" << z7 << endl;
  // cout << "petpt eo1 values :" << z8 << endl;
  // cout << "petpt eo2 values :" << z9 << endl;

  vibes::endDrawing();
}
