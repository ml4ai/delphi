#include "ibex.h"
#include <iostream>
#include "vibes.cpp"

using namespace std;
using namespace ibex;

//int n=5;

void contract_and_draw(Ctc& c, IntervalVector& box, const char* color)
{

  IntervalVector savebox=box;
  IntervalVector *result;

  c.contract(box);

  int n = savebox.diff(box, result);

  for( int i=0; i<n; i++ ){

    if(strncmp(color,"k[r]",4)){

      vibes::drawBox(result[i][0].lb(), result[i][0].ub(), result[i][1].lb(), result[i][1].ub(), color);
      //vibes::drawBox(result[i][0].lb(), result[i][0].ub(), result[i][2].lb(), result[i][2].ub(), color);
      //vibes::drawBox(result[i][0].lb(), result[i][0].ub(), result[i][3].lb(), result[i][3].ub(), color);
      //vibes::drawBox(result[i][0].lb(), result[i][0].ub(), result[i][4].lb(), result[i][4].ub(), color);
      //vibes::drawBox(result[i][1].lb(), result[i][1].ub(), result[i][2].lb(), result[i][2].ub(), color);
      //vibes::drawBox(result[i][1].lb(), result[i][1].ub(), result[i][3].lb(), result[i][3].ub(), color);
      //vibes::drawBox(result[i][1].lb(), result[i][1].ub(), result[i][4].lb(), result[i][4].ub(), color);
      //vibes::drawBox(result[i][2].lb(), result[i][2].ub(), result[i][3].lb(), result[i][3].ub(), color);
      //vibes::drawBox(result[i][2].lb(), result[i][2].ub(), result[i][4].lb(), result[i][4].ub(), color);
      //vibes::drawBox(result[i][3].lb(), result[i][3].ub(), result[i][4].lb(), result[i][4].ub(), color);
      cout << "yes" << endl;

      cout << "tmax (lb):" << result[i][0].lb() << endl;
      cout << "tmax (ub):" << result[i][0].ub() << endl;
      cout << "tmin (lb):" << result[i][1].lb() << endl;
      cout << "tmin (ub):" << result[i][1].ub() << endl;
      cout << "msalb (lb):" << result[i][2].lb() << endl;
      cout << "msalb (ub):" << result[i][2].ub() << endl;
      cout << "color:" << color << endl;
      cout << '\n';
    
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
  Variable flag;

  
  //Variable tdew;
  //Variable xelev;
  //Variable doy;
  //Variable xlat;
  //Variable windrun;
  //Variable windht;
  //Variable meevp;
  //Variable canht;
  Interval tdew(20.0, 20.1);
  Interval xelev(10,10.1);
  Interval doy(1, 1.1);
  Interval xlat(26.63,26.64);
  Interval windrun(400, 400.1);
  Interval windht(3.0, 3.1);
  Interval canht(1.0, 1.01);

  
/*-------------------- PETPT MODEL ---------------*/

  Function td(tmax, tmin, 0.6*tmax + 0.4*tmin);

  Function albedo0(msalb, msalb);
  Function albedo1(msalb, xhlai, 0.23 - (0.23 - msalb)*exp(-0.75*xhlai));
  Function albedo(msalb, xhlai, chi(xhlai, albedo0(msalb), albedo1(msalb, xhlai)));

  Function slang(srad, srad*23.923);

  //double _x0[2][2] = {{16.1, 36.7}, {0.0, 23.9}};
  //IntervalVector x0(2, _x0);
  //Interval z1 = td.eval(x0);

  //double _y0[2][2] = {{0.0, 1.0}, {0.0, 4.77}};
  //IntervalVector y0(2, _y0);
  //Interval z2 = albedo.eval(y0);

  //double _y1[1][2] = {2.45, 27.8};
  //IntervalVector y1(1, _y1);
  //Interval z3 = slang.eval(y1);

  //cout << "td values :" << z1 << endl;
  //cout << "albedo values :" << z2 << endl;
  //cout << "slang values :" << z3 << endl;

  Function eeq1(tmax, tmin, td(tmax, tmin) + 29.0);
  //Interval z4 = eeq1.eval(x0);

  //cout << "eeq1 values :" << z4 << endl;

  //double _y2[4][2] = {{16.1,36.7},{0.0,23.9}, {0.0, 1.0}, {0.0, 4.77}};
  //IntervalVector y2(4,_y2);

  Function eeq2(tmax, tmin, msalb, xhlai, (0.00020410-0.000183*albedo(msalb, xhlai))*eeq1(tmax, tmin));
  //Function eeq2(tmax, tmin, msalb, xhlai, (2.04*pow(10,-4)-1.83*pow(10,-4)*albedo(msalb, xhlai))*eeq1(tmax, tmin));
  //Interval z5 = eeq2.eval(y2);

  //cout << "eeq2 values :" << z5 << endl;

  //double _y3[2][2] = {{0.0, 1.0}, {0.0, 4.77}};
  //IntervalVector y3(2,_y3);
  //Function eeq3(msalb, xhlai, (2.04*pow(10,-4)-1.83*pow(10,-4)*albedo(msalb, xhlai)));
  //Interval z6 = eeq3.eval(y3);

  //cout << "eeq3 values :" << z6 << endl;

  Function eeq(tmax, tmin, msalb, xhlai, srad, slang(srad)*eeq2(tmax, tmin, msalb, xhlai));

  //double _y2[5][2] = {{16.1,36.7},{0.0,23.9}, {0.0, 1.0}, {0.0, 4.77}, {2.45, 27.8}};
  //IntervalVector y2(5,_y2);
  //Interval z5 = eeq.eval(y2);

  //cout << "eeq values :" << z5 << endl;

  Function eo_petpt0(tmax, tmin, msalb, xhlai, srad, eeq(tmax, tmin, msalb, xhlai, srad)*1.1);

  Function eo_petpt1(tmax, tmin, msalb, xhlai, srad, chi(tmax - 35, eo_petpt0(tmax, tmin, msalb, xhlai, srad), eeq(tmax, tmin, msalb, xhlai, srad)*((tmax-35.0)*0.05 + 1.1))); // tmax > 35
  Function eo_petpt2(tmax, tmin, msalb, xhlai, srad, chi(5 - tmax, eo_petpt0(tmax, tmin, msalb, xhlai, srad) ,eeq(tmax, tmin, msalb, xhlai, srad)*0.01*exp(0.18*(tmax+20.0)))); // tmax < 5


  //  Function condition(tmax, ibex::max(5-tmax, tmax -35));
  Function eo_petptnew(tmax, tmin, msalb, xhlai, srad, chi(tmax - 35, chi(5 - tmax, eo_petpt0(tmax, tmin, msalb, xhlai, srad) ,eeq(tmax, tmin, msalb, xhlai, srad)*0.01*exp(0.18*(tmax+20.0))) , eeq(tmax, tmin, msalb, xhlai, srad)*((tmax-35.0)*0.05 + 1.1))); // tmax > 35 && tmin < 5

  Function eo_petpt(tmax, tmin, msalb, xhlai, srad, ibex::max(0.0001,eo_petptnew(tmax, tmin, msalb, xhlai, srad)));




/* -----------------  PETASCE MODEL ----------------- */




  Function tavg(tmax, tmin, (tmax + tmin)/2.0);
  double _x0[2][2] = {{16.1, 36.7}, {0.0, 23.9}};
  IntervalVector x0(2, _x0);
  Interval z1 = tavg.eval(x0);
  //cout << "tavg:" << z1 << endl;


  //Function patm(xelev, 101.3*pow((293.0-0.0065*xelev)/293.0, 5.26));
  Function patm(flag, flag*101.3*pow((293.0-0.0065*xelev)/293.0, 5.26));
  Interval x1 (1.000, 1.001);
  //cout << "patm:" << patm.eval(x1) << endl;


  //Function psycon(xelev, 0.00066*patm(xelev));
  Function psycon(flag, 0.00066*patm(flag));

  //double _x2[1][2]={{10.0, 10.1}};
  //IntervalVector x2(1, _x2);
  Interval z2 = psycon.eval(x1);
  //cout << "psycon:" << z2 << endl;


  //Function udelta(tmax, tmin, 2503.0*exp(17.27*tavg(tmax, tmin)/(tavg(tmax, tmin)+237.3))/((tavg(tmax, tmin)+237.3)*(tavg(tmax, tmin)+237.3)));
  Function udelta(tmax, tmin, 2503.0*exp(17.27*tavg(tmax, tmin)/(tavg(tmax, tmin)+237.3))/(pow((tavg(tmax, tmin)+237.3),2.0)));

  Interval z3 = udelta.eval(x0);
  //cout << "udelta:" << z3 << endl;

  Function emax(tmax, 0.6108*exp((17.27*tmax)/(tmax+237.3)));
  Function emin(tmin, 0.6108*exp((17.27*tmin)/(tmin+237.3)));
  Function es(tmax, tmin, (emax(tmax) + emin(tmin))/2.0);

  Interval z4 = es.eval(x0);
  //cout << "es:" << z4 << endl;
  
  //Function ea(tdew, 0.6108*exp((17.27*tdew)/(tdew+237.3)));
  Function ea(flag, flag*0.6108*exp((17.27*tdew)/(tdew+237.3)));

  Interval z5 = ea.eval(x1);
  //cout << "ea:" << z5 << endl;
  
  //Function rhmin(tmax, tdew, ibex::max(20.0, ibex::min(80.0, ea(tdew)/emax(tmax)*100.0)));
  Function rhmin(tmax, flag, ibex::max(20.0, ibex::min(80.0, ea(flag)/emax(tmax)*100.0)));

  double _x3[2][2]={{16.1, 36.7}, {1.000, 1.001}};
  IntervalVector x3(2, _x3);
  Interval z5_1 = rhmin.eval(x3);
  //cout << "rhmin:" << z5_1 << endl;
  

  Function albedo_petasce(msalb, xhlai, chi(xhlai, msalb,0.23*ibex::max(1.0,msalb)));

  double _x4[2][2] = {{0.0, 1.0}, {0.01, 4.77}};
  IntervalVector x4(2, _x4);
  Interval z6 = albedo_petasce.eval(x4);
  //cout << "albedo:" << z6 << endl;
  
  Function rns(msalb, xhlai, srad, (1.0-albedo_petasce(msalb, xhlai))*srad);

  double _x5[3][2] = {{0.0, 1.0}, {0.01, 4.77}, {2.45, 27.8}};
  IntervalVector x5(3, _x5);
  Interval z7 = rns.eval(x5);
  //cout << "rns:" << z7 << endl;
  //Interval pi(3.1415, 3.1416);

  //Function dr(doy, 1.0+0.033*cos(2.0*pi/365.0*doy));
  Function dr(flag, flag*(1.0+0.033*cos(2.0*pi/365.0*doy)));

  //Function ldelta(doy, 0.409*sin(2.0*pi/365.0*doy-1.39));
  Function ldelta(flag, flag*0.409*sin(2.0*pi/365.0*doy-1.39));
  //Function ws(xlat, doy, acos(-1.0*tan(xlat*pi/180.0)*tan(ldelta(doy))));
  Function ws(flag, flag*acos(-1.0*tan(xlat*pi/180.0)*tan(ldelta(flag))));

  //double _x2[2][2] = {{-90, 90}, {0, 365}};
  //IntervalVector x2(2, _x2);
  Interval z7_1 = ws.eval(x1);
  //cout << "ws:" << z7_1 << endl;

  //Function ra1(xlat, doy, ws(xlat, doy)*sin(xlat*pi/180.0)*sin(ldelta(doy)));
  Function ra1(flag, ws(flag)*sin(xlat*pi/180.0)*sin(ldelta(flag)));
  //Function ra2(xlat, doy, cos(xlat*pi/180.0)*cos(ldelta(doy))*sin(ws(xlat,doy)));
  Function ra2(flag, cos(xlat*pi/180.0)*cos(ldelta(flag))*sin(ws(flag)));
  //Function ra(xlat, doy, 24.0/pi*4.92*dr(doy)*(ra1(xlat,doy)+ra2(xlat,doy)));
  Function ra(flag, 24.0/pi*4.92*dr(flag)*(ra1(flag)+ra2(flag)));

  //Function rso(xelev, xlat, doy, (0.75 + 0.00002*xelev)*ra(xlat, doy));
  Function rso(flag, (0.75 + 0.00002*xelev)*ra(flag));

  //Function ratio(srad, xelev, xlat, doy, ibex::min(1.0, ibex::max(1.0, srad/rso(xelev, xlat, doy))));
  Function ratio(srad, flag, ibex::min(1.0, ibex::max(1.0, srad/rso(flag))));

  //Function fcd(srad, xelev, xlat, doy, 1.35*ratio(srad, xelev, xlat, doy) - 0.35);
  Function fcd(srad, flag, 1.35*ratio(srad, flag) - 0.35);

  Function tk4(tmax, tmin, (pow(tmax+273.16,4.0)+pow(tmin+273.16,4.0))/2.0);

  //Function rnl(tdew, tmax, tmin, srad, xelev, xlat, doy, 4.901*pow(10,-9)*fcd(srad, xelev, xlat, doy)*(0.34 - 0.14*pow(ea(tdew),0.5))*tk4(tmax, tmin));
  Function rnl(tmax, tmin, srad, flag, 4.901*pow(10,-9)*fcd(srad, flag)*(0.34 - 0.14*pow(ea(flag),0.5))*tk4(tmax, tmin));

  double _x6[4][2] = {{16.1, 36.7}, {0.0, 23.9}, {2.45, 27.8}, {1.000, 1.001}};
  IntervalVector x6(4, _x6);
  Interval z8 = rnl.eval(x6);
  //cout << "rnl:" << z8 << endl;
  
  //Function rn(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy, rns(msalb, xhlai, srad)-rnl(tdew, tmax, tmin, srad, xelev, xlat, doy));
  Function rn(tmax, tmin, srad, msalb, xhlai, flag, rns(msalb, xhlai, srad)-rnl(tmax, tmin, srad, flag));


  //Function windsp(windrun, windrun*1000/(24*3600));
  Function windsp(flag, flag*windrun*1000/(24*3600));


  //double _x7[1][2]={{0, 900}};
  //IntervalVector x7(1, _x7);
  Interval z9 = windsp.eval(x1);
  //cout << "windsp:" << z9 << endl;

  //Function wind2m(windrun, windht, windsp(windrun)*(4.87/log(67.8*windht - 5.42)));
  Function wind2m(flag, windsp(flag)*(4.87/log(67.8*windht - 5.42)));

  //double _x8[2][2] = {{0, 900}, {3.0, 3.01}};
  //IntervalVector x8(2, _x8);
  Interval z10 = wind2m.eval(x1);
  //cout << "wind2m:" << z10 << endl;


  char meevp = 'G';
  //int flag = 1;
  const float g = 0.0;
  double cn, cd;

  switch(meevp){
    case 'G':{
               cn = 900;
               cd = 0.34;
               //flag = -1;
               //Interval flag(-1,0);
               break;
             }
    case 'A':{
             cn = 1600.0;
             cd = 0.38;
             //Interval flag(0.1,1);
             //flag = -1;
             break;
             }
  }

  //cout << "cn and cd :" << cn << cd << endl;
  //cout << "g :" << g << endl;

  //Function refet0(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, 0.408*udelta(tmax, tmin)*(rn(tmax,tmin,tdew,srad,msalb,xhlai,xelev,xlat,doy)-g)+psycon(xelev)*(cn/(tavg(tmax,tmin)+273.0))*wind2m(windrun,windht)*(es(tmax,tmin)-ea(tdew)));
  Function refet0(tmax, tmin, srad, msalb, xhlai, flag, 0.408*udelta(tmax, tmin)*(rn(tmax,tmin,srad,msalb,xhlai,flag)-g)+psycon(flag)*(cn/(tavg(tmax,tmin)+273.0))*wind2m(flag)*(es(tmax,tmin)-ea(flag)));

  //Function refet1(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,  windrun, windht, refet0(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht)/(udelta(tmax,tmin)+psycon(xelev)*(1.0+cd*wind2m(windrun,windht))));
  Function refet1(tmax, tmin, srad, msalb, xhlai, flag, refet0(tmax, tmin, srad, msalb, xhlai, flag)/(udelta(tmax,tmin)+psycon(flag)*(1.0+cd*wind2m(flag))));

  //Function refet(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht,ibex::max(0.0001, refet1(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy, windrun, windht)));
  Function refet(tmax, tmin, srad, msalb, xhlai,flag, ibex::max(0.0001, refet1(tmax, tmin, srad, msalb, xhlai, flag)));



  //double _x9[9][2] = {{16.1, 36.7}, {0.0, 23.9}, {0.0, 36.7}, {2.45, 27.8},{0.0, 1.0}, {0.0, 4.77}, {10.0, 10.1}, {26.63, 26.64}, {1.0, 2.0}};
  double _x9[6][2] = {{16.1, 36.7}, {0.0, 23.9}, {2.45, 27.8},{0.0, 1.0}, {0.01, 4.77}, {1.000, 1.001}};
  IntervalVector x9(9, _x9);
  Interval z11 = rn.eval(x9);
  //cout << "rn:" << z11 << endl;

  
  
  //double _x10[11][2] = {{16.1, 36.7}, {0.0, 23.9}, {0.0, 36.7}, {2.45, 27.8},{0.0, 1.0}, {0.01, 4.77}, {10.0, 10.1}, {26.63, 26.64}, {1.0, 2.0}, {0, 900}, {3.0,3.01}};
  //IntervalVector x10(11, _x10);
  Interval z12 = refet.eval(x9);
  //cout << "refet:" << z12 << endl;
  
  const double kcbmax = 1.2;
  const double kcbmin = 0.3;
  const double skc = 0.8;

  // Lower bound should be equal to kcbmin = 0.3
  Function kcb(xhlai, chi(xhlai, 0.0*xhlai, ibex::max(0.0,kcbmin+(kcbmax-kcbmin)*(1.0-exp(-1.0*skc*xhlai)))));
  //Function kcb(xhlai, ibex::max(0.0,kcbmin+(kcbmax-kcbmin)*(1.0-exp(-1.0*skc*xhlai))));

  Interval x11(0.01, 4.77);
  Interval z13 = kcb.eval(x11);
  //cout << "kcb:" << z13 << endl;
  
  //Function wnd(windrun, windht, ibex::max(1.0, ibex::min(wind2m(windrun,windht),6.0)));
  Function wnd(flag, ibex::max(1.0, ibex::min(wind2m(flag),6.0)));

  //Function cht(canht, ibex::max(0.001, canht));
  Function cht(flag, ibex::max(0.001, flag));

  
  //switch(meevp){
    //case 'A':{
      //Function kcmax(tmax, tdew, xhlai, windrun, windht, canht, ibex::max(1.0, kcb(xhlai)+0.05));
      //break;
             //}
    //case 'G':{
      //Function kcmax(tmax, tdew, xhlai, windrun, windht, canht, ibex::max((1.2+(0.04*(wnd(windrun,windht)-2.0)-0.004*(rhmin(tmax,tdew)-45.0))*pow((cht(canht)/3.0),0.3)),kcb(xhlai)+0.05));    
             //}
  //}

  // meevp = 'A' :
  //Function kcmax0(xhlai, ibex::max(1.0, kcb(xhlai)+0.05));
  Function kcmax0(xhlai, ibex::max(1.0, kcb(xhlai)+0.05));

  // meevp = 'G' :
  //Function kcmax1(tmax, tdew, xhlai, windrun, windht, canht, ibex::max((1.2+(0.04*(wnd(windrun,windht)-2.0)-0.004*(rhmin(tmax,tdew)-45.0))*pow((cht(canht)/3.0),0.3)),kcb(xhlai)+0.05));    
  Function kcmax1(tmax, xhlai, flag, ibex::max((1.2+(0.04*(wnd(flag)-2.0)-0.004*(rhmin(tmax,flag)-45.0))*pow((cht(flag)/3.0),0.3)),kcb(xhlai)+0.05));    
  
  //Function kcmax(tmax, tdew, xhlai, windrun, windht, canht, chi(flag,kcmax0(tmax, tdew, xhlai, windrun, windht, canht),kcmax1(tmax, tdew, xhlai, windrun, windht, canht)));

  //cout << "Flag:" << flag << endl;

  double _x12[3][2] = {{16.1, 36.7}, {0.01, 4.77}, {1.000, 1.001}};
  IntervalVector x12(3, _x12);
  //Interval z14 = kcmax.eval(x12);
  //cout << "kcmax:" << z14 << endl;

  // fc bounds exceeding 1.00
  //Function fc(tmax, tdew, xhlai, windrun, windht, canht, flag, chi(kcb(xhlai)-kcbmin, 0.0*xhlai, pow((kcb(xhlai)-kcbmin)/(kcmax(tmax,tdew,xhlai,windrun,windht,canht,flag)-kcbmin), 1.0+0.5*canht)) );


  //Function fc0(tmax, tdew, xhlai, windrun, windht, canht, flag, pow((kcb(xhlai)-kcbmin)/(kcmax(tmax, tdew, xhlai, windrun, windht, canht, flag)-kcbmin),1.0+0.5*canht));

  //Function fc(tmax, tdew, xhlai, windrun, windht, canht, flag, ibex::min(1.0,fc0(tmax, tdew, xhlai, windrun, windht, canht, flag)));

  //Function fc0(tmax, tdew, xhlai, windrun, windht, canht, pow((kcb(xhlai)-kcbmin)/(kcmax1(tmax, tdew, xhlai, windrun, windht, canht)-kcbmin),1.0+0.5*canht));
  Function fc0(tmax, xhlai, flag, pow((kcb(xhlai)-kcbmin)/(kcmax1(tmax, xhlai, flag)-kcbmin),1.0+0.5*canht));

  //Function fc(tmax, tdew, xhlai, windrun, windht, canht, ibex::min(1.0,fc0(tmax, tdew, xhlai, windrun, windht, canht)));
  Function fc(tmax, xhlai, flag, ibex::min(1.0,fc0(tmax, xhlai, flag)));



  Interval z15 = fc.eval(x12);
  //cout << "fc:" << z15 << endl;
  
  const int fw = 1.0;
  //Function few(tmax, tdew, xhlai, windrun, windht, canht, flag, ibex::min(1.0-fc(tmax,tdew,xhlai,windrun,windht,canht,flag), fw));
  //Function few(tmax, tdew, xhlai, windrun, windht, canht, ibex::min(1.0-fc(tmax,tdew,xhlai,windrun,windht,canht), fw));
  Function few(tmax, xhlai, flag, ibex::min(1.0-fc(tmax,xhlai,flag), fw));
  
  Interval z16 = few.eval(x12);
  //cout << "few:" << z16 << endl;
  
  //Function ke(tmax, tdew, xhlai, windrun, windht, canht, flag, ibex::max(0.0, ibex::min(1.0*(kcmax(tmax, tdew, xhlai, windrun, windht, canht, flag)-kcb(xhlai)), few(tmax, tdew, xhlai, windrun, windht, canht, flag)*kcmax(tmax, tdew, xhlai, windrun, windht, canht, flag))));

  Function ke(tmax, xhlai, flag, ibex::max(0.0, ibex::min(kcmax1(tmax, xhlai, flag)-kcb(xhlai), few(tmax, xhlai, flag)*kcmax1(tmax, xhlai, flag))));
  //Function ke(tmax, tdew, xhlai, windrun, windht, canht, ibex::min(kcmax1(tmax, tdew, xhlai, windrun, windht, canht)-kcb(xhlai), few(tmax, tdew, xhlai, windrun, windht, canht)*kcmax1(tmax, tdew, xhlai, windrun, windht, canht)));
  
  //Function ke0(tmax, tdew, xhlai, windrun, windht, canht, kcmax1(tmax, tdew, xhlai, windrun, windht, canht)-kcb(xhlai));
  //Function ke1(tmax, tdew, xhlai, windrun, windht, canht, few(tmax, tdew, xhlai, windrun, windht, canht)*kcmax1(tmax, tdew, xhlai, windrun, windht, canht));
  //Function ke(tmax, tdew, xhlai, windrun, windht, canht, ibex::min(ke0(tmax, tdew, xhlai, windrun, windht, canht), ke1(tmax, tdew, xhlai, windrun, windht, canht)));
  


  Interval z17 = ke.eval(x12);
  //cout << "ke:" << z17 << endl;
  
  //Function kc(tmax, tdew, xhlai, windrun, windht, canht, flag, kcb(xhlai) + ke(tmax, tdew, xhlai, windrun, windht, canht, flag));
  //Function kc(tmax, tdew, xhlai, windrun, windht, canht, kcb(xhlai) + ke(tmax, tdew, xhlai, windrun, windht, canht));
  Function kc(tmax, xhlai, flag, kcb(xhlai) + ke(tmax, xhlai, flag));

  Interval z18 = kc.eval(x12);
  //cout << "kc:" << z18 << endl;
  
  //Function eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht, flag, (kcb(xhlai) + ke(tmax, tdew, xhlai, windrun, windht, canht, flag)*refet(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht)));
  //Function eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy, windrun, windht, canht, kc(tmax, tdew, xhlai, windrun, windht, canht)*refet(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy, windrun, windht));
  Function eo_petasce1(tmax, tmin, srad, msalb, xhlai, flag, kc(tmax, xhlai, flag)*refet(tmax, tmin, srad, msalb, xhlai, flag));
  
  //Function eonew(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht, flag, ibex::min(10.0, eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht,flag)));
  //Function eonew(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy, windrun, windht, canht, ibex::max(0.0001, eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy, windrun, windht, canht)));
  Function eo_petasce(tmax, tmin, srad, msalb, xhlai, flag, ibex::max(0.0001, eo_petasce1(tmax, tmin, srad, msalb, xhlai, flag)));

  
  //Function eo_diff(tmax, tmin, srad, msalb, xhlai, flag, pow(eo_petasce(tmax, tmin, srad, msalb, xhlai, flag) - eo_petpt(tmax, tmin, msalb, xhlai, srad), 2.0));
  Function eo_diff(tmax, tmin, srad, msalb, xhlai, flag, abs(eo_petasce(tmax, tmin, srad, msalb, xhlai, flag) - eo_petpt(tmax, tmin, msalb, xhlai, srad)));

  double _box[6][2] = {{16.1, 36.7}, {0.0, 23.9}, {2.45, 27.8}, {0.0, 1.0}, {0.01, 4.77}, {1.000, 1.001}};
  IntervalVector box(6, _box);
  
  //double _box[12][2] = {{16.1, 36.7}, {0.0, 23.9}, {0.0, 36.7}, {2.45, 27.8},{0.0, 1.0}, {0.01, 4.77}, {10.0, 10.1}, {26.63, 26.64}, {1.0, 2.0}, {0, 900}, {3.0,3.01}, {0.0, 3.0}};
  //IntervalVector box(12, _box);
  Interval z19 = eo_petasce.eval(x9);
  //cout << "petasce eo:" << z19 << endl;

  //double _box[5][2] = {{16.1,36.7},{0.0,23.9}, {0.0, 1.0}, {0.0, 4.77}, {2.45, 27.8}};
  //IntervalVector box(5,_box);

  stack<IntervalVector> s;

  double eps=0.5;

  s.push(box);

  NumConstraint c1(tmax, tmin, srad, msalb, xhlai, flag, eo_diff(tmax, tmin, srad, msalb, xhlai, flag) <= 0.01);
  NumConstraint c2(tmax, tmin, srad, msalb, xhlai, flag, eo_diff(tmax, tmin, srad, msalb, xhlai, flag) >= 0.0);

  NumConstraint c3(tmax, tmin, srad, msalb, xhlai, flag, eo_diff(tmax, tmin, srad, msalb, xhlai, flag) > 0.01);
  NumConstraint c4(tmax, tmin, srad, msalb, xhlai, flag, eo_diff(tmax, tmin, srad, msalb, xhlai, flag) < 0.0);

  //NumConstraint c1(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht, eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht) <= 10);
  //NumConstraint c2(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht, eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht) >= 3);

  //NumConstraint c3(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht, eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht) > 10);
  //NumConstraint c4(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht, eo(tmax, tmin, tdew, srad, msalb, xhlai, xelev, xlat, doy,windrun, windht, canht) < 3);
  
  //NumConstraint c1(tmax, tdew, xhlai, windrun, windht, canht, fc(tmax, tdew, xhlai, windrun, windht, canht) <= 1.0);
  //NumConstraint c2(tmax, tdew, xhlai, windrun, windht, canht, fc(tmax, tdew, xhlai, windrun, windht, canht) >= 0.0);

  //NumConstraint c3(tmax, tdew, xhlai, windrun, windht, canht, fc(tmax, tdew, xhlai, windrun, windht, canht) > 1.0);
  //NumConstraint c4(tmax, tdew, xhlai, windrun, windht, canht, fc(tmax, tdew, xhlai, windrun, windht, canht) < 0.0);

  //NumConstraint c1(tmax, tdew, xhlai, windrun, windht, canht, flag,  fc(tmax, tdew, xhlai, windrun, windht, canht, flag) <= 0.5);
  //NumConstraint c2(tmax, tdew, xhlai, windrun, windht, canht, flag, fc(tmax, tdew, xhlai, windrun, windht, canht, flag) >= 0.0);

  //NumConstraint c3(tmax, tdew, xhlai, windrun, windht, canht, flag,  fc(tmax, tdew, xhlai, windrun, windht, canht, flag) > 0.5);
  //NumConstraint c4(tmax, tdew, xhlai, windrun, windht, canht, flag, fc(tmax, tdew, xhlai, windrun, windht, canht, flag) < 0.0);
  
  //CtcFwdBwd c(eo(tmax, tmin, msalb, xhlai, srad));
  CtcFwdBwd out1(c1);
  CtcFwdBwd out2(c2);
  CtcFwdBwd in1(c3);
  CtcFwdBwd in2(c4);

  CtcCompo outside(out1, out2);

  CtcUnion inside(in1, in2);


  //inside.contract(box);
  //outside.contract(box);

  //cout << box << endl;


  while(!s.empty()){
    IntervalVector box = s.top();
    //cout << box << endl;
    s.pop();
    contract_and_draw(outside, box, "k[r]");
    contract_and_draw(inside, box, "k[g]");
    if(!box.is_empty() && box.max_diam()>eps){
      int i=box.extr_diam_index(false);
      pair<IntervalVector, IntervalVector> p=box.bisect(i);
      s.push(p.first);
      s.push(p.second);
    }
  }


  //inside.contract(box);
  //cout << "parameter values:" << box << endl;

  //double deltax1 = box[0].diam()/n;
  //double deltax2 = box[1].diam()/n;
  //double deltax3 = box[2].diam()/n;
  //double deltax4 = box[3].diam()/n;
  //double deltax5 = box[4].diam()/n;

  //for(int i1=0; i1<n; i1++)
  //for(int i2=0; i2<n; i2++)
  //for(int i3=0; i3<n; i3++)
  //for(int i4=0; i4<n; i4++)
  //for(int i5=0; i5<n; i5++){
  //IntervalVector box2(5);
  //box2[0] = Interval(box[0].lb()+i1*deltax1,box[0].lb()+(i1+1)*deltax1);
  //box2[1] = Interval(box[1].lb()+i2*deltax2,box[1].lb()+(i2+1)*deltax2);
  //box2[2] = Interval(box[2].lb()+i3*deltax3,box[2].lb()+(i3+1)*deltax3);
  //box2[3] = Interval(box[3].lb()+i4*deltax4,box[3].lb()+(i4+1)*deltax4);
  //box2[4] = Interval(box[4].lb()+i5*deltax5,box[4].lb()+(i5+1)*deltax5);
  //Interval petpt_eo = eo.eval(box2);

  //cout << "petpt values :" << petpt_eo << endl;
  //}

  //Interval z = eo.eval(x);  

  //Interval z7 = eo0.eval(x);  
  //Interval z8 = eo1.eval(x);  
  //Interval z9 = eo2.eval(x);  

  //cout << "petpt values :" << z << endl;
  //cout << "petpt eo0 values :" << z7 << endl;
  //cout << "petpt eo1 values :" << z8 << endl;
  //cout << "petpt eo2 values :" << z9 << endl;

  vibes::endDrawing();

}


