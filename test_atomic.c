#include <stdio.h>
#include <omp.h>
#include <complex.h>
#include <math.h>

#define MAX 500000000

#define update_momenta(m) \
_Pragma("omp atomic") \
    m.d1 += conj(a2)*a1; \
_Pragma("omp atomic") \
    m.d2 += creal(a3)+cimag(a5); \
_Pragma("omp atomic") \
    m.d3 += cimag(a3)+creal(a5); \
_Pragma("omp atomic") \
    m.d4 += cimag(a4)-creal(a1); \
_Pragma("omp atomic") \
    m.d5 += conj(a3)*a5-creal(a2)+creal(a3); \
_Pragma("omp atomic") \
    m.d6 += creal(a3)+creal(a4); \
_Pragma("omp atomic") \
    m.d7 += cimag(a1)+cimag(a2); \
_Pragma("omp atomic") \
    m.d8 += cimag(a4)-creal(a4); \
 
inline void waste_cycles(double *in) {
  double pp = pow(*in,7);
  *in = pp/pow(*in,6);
}

typedef struct {
  double d1;
  double d2;
  double d3;
  double d4;
  double d5;
  double d6;
  double d7;
  double d8;
} su3adj;

su3adj momenta = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

int main(void) {
  printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",momenta.d1,momenta.d2,momenta.d3,momenta.d4,momenta.d5,momenta.d6,momenta.d7,momenta.d8);
 
  #pragma omp parallel
  {
    
  if( omp_get_thread_num() == 0 ) {
    printf("%d OpenMP threads!\n",omp_get_num_threads());
  }
     
  double in = 33;
  double out = 0;
  double res = 0;

  complex double a1 = 1.0 + I*2.0; // {1.0,2.0};
  complex double a2 = 3.0 + I*4.0; // {3.0,4.0};
  complex double a3 = 5.0 - I*6.0; // {5.0,6.0};
  complex double a4 = 7.0 + I*8.0; //{7.0;8.0};
  complex double a5 = 9.0 + I*10.0;// ; {9.0,10.0};
  
  #pragma omp for
  for(int i = 0; i < MAX; ++i) {
    waste_cycles(&in);
    update_momenta(momenta);
    waste_cycles(&in);
    update_momenta(momenta);
    waste_cycles(&in);
    update_momenta(momenta);
    waste_cycles(&in);
    update_momenta(momenta);
    waste_cycles(&in);
    update_momenta(momenta);
    waste_cycles(&in);
    update_momenta(momenta);
    waste_cycles(&in);
    update_momenta(momenta);
  }

  printf("thread %d %lf \n",omp_get_thread_num(),in);

  } /* OpenMP closing brace */

  printf("%lf %lf %lf %lf %lf %lf %lf %lf\n",momenta.d1,momenta.d2,momenta.d3,momenta.d4,momenta.d5,momenta.d6,momenta.d7,momenta.d8);
  return(0); 
}
