#ifndef __UAV_UTILS_H
#define __UAV_UTILS_H

#include <cassert>
#include <sstream>

#include <uav_utils/converters.h>
#include <uav_utils/geometry_utils.h>

namespace uav_utils
{

/* judge if value belongs to [low,high] */
template <typename T, typename T2>
bool
in_range(T value, const T2& low, const T2& high)
{
  assert(low < high);
  return (low <= value) && (value <= high);
}

/* judge if value belongs to [-limit, limit] */
template <typename T, typename T2>
bool
in_range(T value, const T2& limit)
{
  assert(limit > 0);
  return in_range(value, -limit, limit);
}

template <typename T, typename T2>
void
limit_range(T& value, const T2& low, const T2& high)
{
  assert(low < high);
  if (value < low)
  {
    value = low;
  }

  if (value > high)
  {
    value = high;
  }

  return;
}

template <typename T, typename T2>
void
limit_range(T& value, const T2& limit)
{
  assert(limit > 0);
  limit_range(value, -limit, limit);
}

typedef std::stringstream DebugSS_t;
} // end of namespace uav_utils

#endif
