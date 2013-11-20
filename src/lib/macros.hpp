#ifndef MACROS_HPP_
#define MACROS_HPP_

#ifndef NDEBUG
#define USE_ASSERTIONS
#define PRINT_DEBUG_MSGS
#include <cstdlib>
#include <iostream>
#endif

// http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)

#ifdef PRINT_DEBUG_MSGS
  #define PRINT(x) do { (std::cout << x << std::endl); } while (0)
#else
  #define PRINT(x)
#endif

#ifdef USE_ASSERTIONS
  #define ASSERT(cond,msg)                               \
  do {                                                   \
    if (!(cond)) {                                       \
        std::cerr << __FILE__ << ":" << __LINE__ << ": " \
                  << __PRETTY_FUNCTION__ << ": "         \
                  << "Assertion `" #cond "` failed: "    \
                  << msg << std::endl;                   \
        std::abort();                                    \
      }                                                  \
   } while (0)
#else
  #define ASSERT(cond, msg)
#endif


#endif  // MACROS_HPP_