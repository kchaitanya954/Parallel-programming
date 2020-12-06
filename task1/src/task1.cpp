
#include <omp.h> 
#include<iostream>
#include<cmath>

using namespace std;
double f(double x);
double integral(double a , double b);
double integral_critical(double a, double b);
double integral_atomic(double a, double b);
double integral_lock(double a, double b);
double integral_reduction(double a, double b);
static int thread_count = 4;
static int n=10000000;

int main()
{
  double I, start, end, I_C, start_c, end_c, I_A, start_a, end_a, \
  I_L, start_l, end_l, I_R, start_r, end_r ;
  
  double a[]={0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10};
  double b[]={0.0001, 0.001, 0.01, 0.1, 1, 10, 100};
  
  int size= sizeof(a)/sizeof(a[0]);
  cout<< "Number of steps: "<< n<< "\n Number of threads: "<< thread_count<< endl;
  for (int i=0; i<size; i++)
  {
    start=omp_get_wtime();
    for(int j=0; j<5; j++)
    {
      I=integral(a[i],b[i]);
    }
    end=omp_get_wtime();
    cout<< "a="<<a[i]<<" :b="<<b[i]<<"\n";
    cout<<"Integral: "<< I<<"\n"<<"Execution time: "<<(end-start)/5<<endl;

    start_c=omp_get_wtime();
    for(int j=0; j<5; j++)
    {
      I_C=integral_critical(a[i],b[i]);
    }
    
    end_c=omp_get_wtime();
    cout<<"Integral_critical: "<< I_C<<"\n"<<"Execution time: "<<(end_c-start_c)/5<<endl;

    start_a=omp_get_wtime();
    for(int j=0; j<5; j++)
    {
      I_A=integral_atomic(a[i],b[i]);
    }
    end_a=omp_get_wtime();
    cout<<"Integral_atomic: "<< I_A<<"\n"<<"Execution time: "<<(end_a-start_a)/5<<endl;

    start_l=omp_get_wtime();
    for(int j=0; j<5; j++)
    {
      I_L=integral_lock(a[i],b[i]);
    }
    end_l=omp_get_wtime();
    cout<<"Integral_lock: "<< I_L<<"\n"<<"Execution time: "<<(end_l-start_l)/5<<endl;

    start_r=omp_get_wtime();
    for(int j=0; j<5; j++)
    {
      I_R=integral_reduction(a[i],b[i]);
    }
    end_r=omp_get_wtime();
    cout<<"Integral_reduction: "<< I_R<<"\n"<<"Execution time: "<<(end_r-start_r)/5<<endl;
    cout<< "\n";
  }
  return 0;

}

double f(double x)
{
  return (1/pow(x,2))*pow(sin(1/x),2);
}

double integral(double a , double b)
{
  double step=(b-a)/n ;
  double I= (f(a)+f(b))/2;
  double x;
  for (int i=1; i<n; i++ )
  {
    I=I+f(a+step*i);
  }
  I=I*step;
  return I;
}


double integral_critical(double a, double b)
{
  double step= (b-a)/n;
  double I=0;
  double J=0;
  
  #pragma omp parallel num_threads(thread_count)
  {
    double I=0;
  #pragma omp for 
    for (int i=1; i<n; i++)
        I+=f(a+step*i);
  #pragma omp critical
    {
        J+=I;
    }
  }
  J=(J+0.5*(f(a)+f(b)))*step;
  return J;
}

double integral_atomic(double a, double b)
{
  double step= (b-a)/n;
  double I=0;
  double J=0;
  
  #pragma omp parallel num_threads(thread_count)
  {
    double I=0;
  #pragma omp for 
    for (int i=1; i<n; i++)
        I+=f(a+step*i);
  #pragma omp atomic
        J+=I;
    
  }
  J=(J+0.5*(f(a)+f(b)))*step;
  return J;
}

double integral_lock(double a, double b)
{
  double step= (b-a)/n;
  double I=0;
  double J=0;
  omp_lock_t lock;
  omp_init_lock(&lock);

  #pragma omp parallel num_threads(thread_count) 
  {
  #pragma omp for 
    for (int i=1; i<n; i++)
    {
        omp_set_lock(&lock);
        I+=f(a+step*i);
        omp_unset_lock(&lock);
    }
  }
  omp_destroy_lock(&lock);

  J=(I+0.5*(f(a)+f(b)))*step;
  return J;
}

double integral_reduction(double a, double b)
{
  double step= (b-a)/n;
  double I=0;
  double J=0;
  #pragma omp parallel num_threads(thread_count)
  {
    #pragma omp for reduction(+: I)
    for (int i=1; i<n; i++)
    {
      I=I+f(a+step*i);
    }
  }
  J=(I+0.5*(f(a)+f(b)))*step; 
  return J;
}


