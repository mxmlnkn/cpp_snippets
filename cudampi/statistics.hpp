#ifndef STATISTICS_H

#include <cmath>

template <typename T> 
float Mean( T *array, unsigned int count ) {
	float sum = 0;
	for (int i=0; i<count; i++)
		sum += array[i];
	return sum/count;
}

template <typename T> 
float StdDev( T *array, unsigned int count ) {
	if (count <= 1)
		return 0;
	float sum = 0, sum2 = 0;
	for (int i=0; i<count; i++) {
		sum  += array[i];
		sum2 += array[i]*array[i];
	}
	return sqrt((std::abs(sum2-sum*sum/count))/(count-1));
}

#endif