package com.company;

/**
 *
 * @author Xin
 */
public class StatFunctions {
    
    public static double pnorm(double z,boolean upper) {
    /* Reference:
       I. D. Hill 
       Algorithm AS 66: "The Normal Integral"
       Applied Statistics
    */
    double ltone=7.0,
           utzero=18.66,
           con=1.28,
           a1 = 0.398942280444,
           a2 = 0.399903438504,
           a3 = 5.75885480458,
           a4 =29.8213557808,
           a5 = 2.62433121679,
           a6 =48.6959930692,
           a7 = 5.92885724438,
           b1 = 0.398942280385,
           b2 = 3.8052e-8,
           b3 = 1.00000615302,
           b4 = 3.98064794e-4,
           b5 = 1.986153813664,
           b6 = 0.151679116635,
           b7 = 5.29330324926,
           b8 = 4.8385912808,
           b9 =15.1508972451,
           b10= 0.742380924027,
           b11=30.789933034,
           b12= 3.99019417011;
    double y,alnorm;

    if(z<0) {
      upper=!upper;
      z=-z;
    }
    if(z<=ltone || upper && z<=utzero) {
      y=0.5*z*z;
      if(z>con) {
        alnorm=b1*Math.exp(-y)/(z-b2+b3/(z+b4+b5/(z-b6+b7/(z+b8-b9/(z+b10+b11/(z+b12))))));
      }
      else {
        alnorm=0.5-z*(a1-a2*y/(y+a3-a4/(y+a5+a6/(y+a7))));
      }
    }
    else {
      alnorm=0;
    }
    if(!upper) alnorm=1-alnorm;
    return(alnorm);
  }

  public static double pnorm(double x,boolean upper,double mu,double sigma2) {
    return(pnorm((x-mu)/Math.sqrt(sigma2),upper));
  }
}
