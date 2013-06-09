#include <iostream>
#include <iomanip>
#include <cmath>

const double m_dbl = 2.0e3;
const float  m     = static_cast<float>(m_dbl);

const double c_dbl = 1.0;
const float  c     = static_cast<float>(c_dbl);

int main()
{

  float  x     = 0.f;
  double x_dbl = 0.;
  float  g     = 1.0f;
  double g_dbl = 1.0f;
  float  p     = 0.f;
  float  v     = 0.f;

  for( double p_dbl = 0.1; p_dbl < 100.; p_dbl*=3. )
  {
    g_dbl = sqrt(1. + p_dbl*p_dbl/(m_dbl*m_dbl*c_dbl*c_dbl));
    x_dbl = p_dbl / (m_dbl*g_dbl);

    p = static_cast<float>(p_dbl);

    // Method 1
    g = sqrtf(1.f + p*p/(m*m*c*c));
    x = p / (m*g);

    // Method 2
    //v = p / sqrtf(m*m + p*p/(c*c));
    //x = v;

    std::cout << p_dbl << ": " << std::setprecision(10) << x << " " << x_dbl << std::endl;
  }
  std::cout << "test" << std::endl;
}
