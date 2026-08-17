/* Hand-written config.h for the vendored libdivsufsort build (MSVC).
 * Replaces config.h.cmake. Values: static, Windows, 32+64-bit APIs. */
#ifndef _CONFIG_H
#define _CONFIG_H 1

#ifdef __cplusplus
extern "C" {
#endif

#define PROJECT_VERSION_FULL "2.0.3-snidewider"

#define HAVE_STDDEF_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_MEMORY_H 1
#define HAVE_SYS_TYPES_H 1

#define HAVE_IO_H 1
#define HAVE_FCNTL_H 1
#define HAVE__SETMODE 1
#define HAVE__FILENO 1
#define HAVE_FOPEN_S 1
#define HAVE__O_BINARY 1

#ifndef INLINE
#define INLINE __inline
#endif

#ifdef _MSC_VER
#pragma warning(disable: 4127)
#endif

#ifdef __cplusplus
}
#endif

#endif /* _CONFIG_H */
