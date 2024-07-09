#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long myCPUTimer()
{
  struct timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec);
}