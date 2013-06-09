void saxpy( int n, float a, float* x, float* __restrict__ y )
{
  #pragma acc kernels
  for( int i =0; i<n; ++i )
    y[i] = a*x[i] + y[i];
}

int main()
{

  const int N = (1<<20);
  float x[N];
  float y[N];

  // perform on 1M elements
  saxpy(N, 2.0, x, y);

  return 0;
}
