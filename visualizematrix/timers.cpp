/**
	timers.cpp

	Created on: Mar 10, 2010
		Author: kuestner
*/

#include <sys/time.h>

#include "timers.hpp"

/// Return the current time (in microseconds)
/**
 * This function uses gettimeofday()
 * NOTE POSIX.1-2008 marks gettimeofday() as obsolete, recommending the use of
 * clock_gettime() instead.
 */
double musecs() {
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double)tv.tv_sec * 1.0e6 + (double)tv.tv_usec;
}

/// Return the current time (in seconds)
/**
 * This function uses gettimeofday()
 * NOTE POSIX.1-2008 marks gettimeofday() as obsolete, recommending the use of
 * clock_gettime() instead.
 */
double secs() {
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}
