/* harness.h -- a test harness small enough to not need explaining.
 *
 * No framework. The firmware layer has no dependencies by design, and adding
 * a test dependency that the build must fetch would undermine the property
 * that makes this directory testable in the first place: it compiles with a
 * bare C compiler and nothing else.
 */

#ifndef WISENSE_TEST_HARNESS_H
#define WISENSE_TEST_HARNESS_H

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_checks = 0;
static int g_failures = 0;
static const char *g_current = "";

#define RUN(fn)                                                    \
    do {                                                           \
        g_current = #fn;                                           \
        fn();                                                      \
    } while (0)

#define CHECK(cond)                                                \
    do {                                                           \
        g_checks++;                                                \
        if (!(cond)) {                                             \
            g_failures++;                                          \
            printf("  FAIL %s:%d  %s\n    %s\n",                   \
                   __FILE__, __LINE__, g_current, #cond);          \
        }                                                          \
    } while (0)

#define CHECK_EQ(a, b)                                             \
    do {                                                           \
        g_checks++;                                                \
        const long long _a = (long long)(a);                       \
        const long long _b = (long long)(b);                       \
        if (_a != _b) {                                            \
            g_failures++;                                          \
            printf("  FAIL %s:%d  %s\n    %s == %s  (%lld vs %lld)\n", \
                   __FILE__, __LINE__, g_current, #a, #b, _a, _b);  \
        }                                                          \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                      \
    do {                                                           \
        g_checks++;                                                \
        const double _a = (double)(a);                             \
        const double _b = (double)(b);                             \
        if (!(fabs(_a - _b) <= (tol))) {                           \
            g_failures++;                                          \
            printf("  FAIL %s:%d  %s\n    %s ~= %s  (%.9g vs %.9g, tol %.3g)\n", \
                   __FILE__, __LINE__, g_current, #a, #b, _a, _b, (double)(tol)); \
        }                                                          \
    } while (0)

static int harness_report(const char *suite)
{
    if (g_failures == 0) {
        printf("%-16s %3d checks   OK\n", suite, g_checks);
        return 0;
    }
    printf("%-16s %3d checks   %d FAILED\n", suite, g_checks, g_failures);
    return 1;
}

#endif /* WISENSE_TEST_HARNESS_H */
