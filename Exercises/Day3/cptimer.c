#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

double seconds()

/* Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)                                                                     
 */

{

  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;

}
