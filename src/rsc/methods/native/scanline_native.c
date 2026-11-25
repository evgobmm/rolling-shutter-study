#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static int clamp_index(int x, int size) {
    return x < 0 ? 0 : (x >= size ? size - 1 : x);
}

EXPORT void thumbnail(const uint8_t *src, float *dst, const float *window,
                      int width, int height, int small_w, int small_h, int sample) {
    for (int y = 0; y < small_h; ++y) {
        int row = ((2 * y + 1) * height) / (2 * small_h);
        for (int x = 0; x < small_w; ++x) {
            int col = ((2 * x + 1) * width) / (2 * small_w);
            const uint8_t *pixel = src + (row * width + col) * 3;
            int offset = y * small_w + x;
            if (sample) {
                const uint8_t *above = pixel - width * 3;
                int green = pixel[1] + pixel[-2] + above[1] + above[-2];
                dst[offset] = (green * .25f) * window[offset];
            } else {
                dst[offset] = pixel[1] * window[offset];
            }
        }
    }
}

static void copy_shifted_row(uint8_t *out, const uint8_t *in, int width, int shift) {
    int from = shift < 0 ? -shift : 0;
    int until = shift > 0 ? width - shift : width;
    from = from > width ? width : from;
    until = until < from ? from : until;
    for (int x = 0; x < from; ++x) memcpy(out + 3 * x, in, 3);
    if (until > from) memcpy(out + 3 * from, in + 3 * (from + shift), (until - from) * 3);
    for (int x = until; x < width; ++x) memcpy(out + 3 * x, in + 3 * (width - 1), 3);
}

static __m256i muladd(__m256i sum, __m256i values, __m256i weight, int high) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i words = high ? _mm256_unpackhi_epi8(values, zero) : _mm256_unpacklo_epi8(values, zero);
    return _mm256_add_epi16(sum, _mm256_mullo_epi16(words, weight));
}

static void filter_row(uint8_t *out, const uint8_t *a, const uint8_t *b,
                       int width, int shift, int wx, int wy) {
    int weight[4];
    weight[1] = ((256 - wy) * wx + 128) >> 8;
    weight[3] = (wy * wx + 128) >> 8;
    weight[0] = 256 - wy - weight[1];
    weight[2] = wy - weight[3];
    int first = shift < 0 ? -shift : 0;
    int last = width - 1 - shift;
    first = first > width ? width : first;
    last = last > width ? width : last;
    last = last < first ? first : last;
    int x = 0;
    for (; x < first; ++x) {
        for (int k = 0; k < 3; ++k)
            out[3 * x + k] = (uint8_t)(((256 - wy) * a[k] + wy * b[k] + 128) >> 8);
    }
    const __m256i rounding = _mm256_set1_epi16(128);
    const __m256i w0 = _mm256_set1_epi16((short)weight[0]);
    const __m256i w1 = _mm256_set1_epi16((short)weight[1]);
    const __m256i w2 = _mm256_set1_epi16((short)weight[2]);
    const __m256i w3 = _mm256_set1_epi16((short)weight[3]);
    int index = 3 * x;
    int stop = 3 * last;
    for (; index + 32 <= stop; index += 32) {
        int input = index + 3 * shift;
        __m256i p0 = _mm256_loadu_si256((const __m256i *)(a + input));
        __m256i p1 = _mm256_loadu_si256((const __m256i *)(a + input + 3));
        __m256i p2 = _mm256_loadu_si256((const __m256i *)(b + input));
        __m256i p3 = _mm256_loadu_si256((const __m256i *)(b + input + 3));
        __m256i lo = muladd(muladd(muladd(muladd(rounding, p0, w0, 0), p1, w1, 0), p2, w2, 0), p3, w3, 0);
        __m256i hi = muladd(muladd(muladd(muladd(rounding, p0, w0, 1), p1, w1, 1), p2, w2, 1), p3, w3, 1);
        __m256i packed = _mm256_packus_epi16(_mm256_srli_epi16(lo, 8), _mm256_srli_epi16(hi, 8));
        _mm256_storeu_si256((__m256i *)(out + index), packed);
    }
    for (; index < stop; ++index) {
        int input = index + 3 * shift;
        int value = weight[0] * a[input] + weight[1] * a[input + 3]
                  + weight[2] * b[input] + weight[3] * b[input + 3];
        out[index] = (uint8_t)((value + 128) >> 8);
    }
    for (x = last; x < width; ++x) {
        for (int k = 0; k < 3; ++k) {
            int edge = 3 * (width - 1) + k;
            out[3 * x + k] = (uint8_t)(((256 - wy) * a[edge] + wy * b[edge] + 128) >> 8);
        }
    }
}

static void warp_range(const uint8_t *src, uint8_t *dst, int width, int height,
                       float slope_x, float slope_y, int mode, int first_row, int last_row) {
    for (int y = first_row; y < last_row; ++y) {
        float time = y - .5f * (height - 1);
        float x_shift = time * slope_x;
        float y_source = y + time * slope_y;
        uint8_t *out = dst + (size_t)y * width * 3;
        if (mode == 0) {
            int at_y = clamp_index((int)floorf(y_source + .5f), height);
            copy_shifted_row(out, src + (size_t)at_y * width * 3, width, (int)floorf(x_shift + .5f));
            continue;
        }
        if (mode == 2) y_source = floorf(y_source + .5f);
        if (mode == 3) x_shift = floorf(x_shift + .5f);
        int floor_x = (int)floorf(x_shift), floor_y = (int)floorf(y_source);
        int wx = (int)floorf(256 * (x_shift - floor_x) + .5f);
        int wy = (int)floorf(256 * (y_source - floor_y) + .5f);
        const uint8_t *top = src + (size_t)clamp_index(floor_y, height) * width * 3;
        const uint8_t *bottom = src + (size_t)clamp_index(floor_y + 1, height) * width * 3;
        if (wx == 0 && (top == bottom || wy == 0)) {
            copy_shifted_row(out, top, width, floor_x);
        } else {
            filter_row(out, top, bottom, width, floor_x, wx, wy);
        }
    }
}

EXPORT void warp_rows(const uint8_t *src, uint8_t *dst, int width, int height,
                      float slope_x, float slope_y, int mode) {
    warp_range(src, dst, width, height, slope_x, slope_y, mode, 0, height);
}

static void shift_same_row(uint8_t *row, int width, int shift) {
    if (shift == 0) return;
    int count = shift < 0 ? -shift : shift;
    if (count > width) count = width;
    uint8_t edge[3];
    if (shift < 0) {
        memcpy(edge, row, 3);
        memmove(row + 3 * count, row, 3 * (width - count));
        for (int x = 0; x < count; ++x) memcpy(row + 3 * x, edge, 3);
    } else {
        memcpy(edge, row + 3 * (width - 1), 3);
        memmove(row, row + 3 * count, 3 * (width - count));
        for (int x = width - count; x < width; ++x) memcpy(row + 3 * x, edge, 3);
    }
}

static void warp_inplace_range(uint8_t *image, int width, int height, float slope_x, float slope_y,
                               int row, int stop, int step) {
    for (; row != stop; row += step) {
        float time = row - .5f * (height - 1);
        int shift = (int)floorf(time * slope_x + .5f);
        int source_row = clamp_index((int)floorf(row + time * slope_y + .5f), height);
        uint8_t *destination = image + (size_t)row * width * 3;
        if (row == source_row) shift_same_row(destination, width, shift);
        else copy_shifted_row(destination, image + (size_t)source_row * width * 3, width, shift);
    }
}

EXPORT void warp_inplace(uint8_t *image, int width, int height, float slope_x, float slope_y) {
    int middle = height / 2;
    /* With a monotone row map, visit every source row before it is overwritten. */
    if (slope_y >= 0) {
        warp_inplace_range(image, width, height, slope_x, slope_y, middle - 1, -1, -1);
        warp_inplace_range(image, width, height, slope_x, slope_y, middle, height, 1);
    } else {
        warp_inplace_range(image, width, height, slope_x, slope_y, 0, middle, 1);
        warp_inplace_range(image, width, height, slope_x, slope_y, height - 1, middle - 1, -1);
    }
}

#ifdef _WIN32
typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    int width, height, mode, first, last;
    float slope_x, slope_y;
} WarpJob;

typedef struct {
    int workers;
    WarpJob jobs[3];
    PTP_WORK work[3];
} ParallelWarp;

static void CALLBACK warp_callback(PTP_CALLBACK_INSTANCE instance, void *context, PTP_WORK work) {
    (void)instance;
    (void)work;
    WarpJob *j = context;
    if (j->mode < 0) {
        int step = j->slope_y < 0 ? 1 : -1;
        int begin = step > 0 ? j->first : j->last - 1;
        int stop = step > 0 ? j->last : j->first - 1;
        warp_inplace_range(j->dst,j->width,j->height,j->slope_x,j->slope_y,begin,stop,step);
    } else {
        warp_range(j->src, j->dst, j->width, j->height, j->slope_x, j->slope_y, j->mode, j->first, j->last);
    }
}

EXPORT void delete_parallel(ParallelWarp *state) {
    if (!state) return;
    for (int i = 0; i < state->workers; ++i) if (state->work[i]) CloseThreadpoolWork(state->work[i]);
    free(state);
}

EXPORT ParallelWarp *make_parallel(int workers) {
    if (workers < 1 || workers > 3) return NULL;
    ParallelWarp *state = calloc(1, sizeof(*state));
    if (!state) return NULL;
    state->workers = workers;
    for (int i = 0; i < workers; ++i) {
        state->work[i] = CreateThreadpoolWork(warp_callback, &state->jobs[i], NULL);
        if (!state->work[i]) {
            delete_parallel(state);
            return NULL;
        }
    }
    return state;
}

EXPORT void warp_parallel(ParallelWarp *state, const uint8_t *src, uint8_t *dst,
                          int width, int height, float slope_x, float slope_y, int mode) {
    int chunk = height / (state->workers + 1);
    for (int i = 0; i < state->workers; ++i) {
        state->jobs[i] = (WarpJob){src, dst, width, height, mode, i * chunk, (i + 1) * chunk, slope_x, slope_y};
        SubmitThreadpoolWork(state->work[i]);
    }
    warp_range(src, dst, width, height, slope_x, slope_y, mode, state->workers * chunk, height);
    for (int i = 0; i < state->workers; ++i) WaitForThreadpoolWorkCallbacks(state->work[i], FALSE);
}

EXPORT void warp_inplace_parallel(ParallelWarp *state, uint8_t *image, int width, int height,
                                  float slope_x, float slope_y) {
    int middle = height / 2;
    state->jobs[0] = (WarpJob){image, image, width, height, -1, 0, middle, slope_x, slope_y};
    SubmitThreadpoolWork(state->work[0]);
    if (slope_y >= 0) warp_inplace_range(image,width,height,slope_x,slope_y,middle,height,1);
    else warp_inplace_range(image,width,height,slope_x,slope_y,height-1,middle-1,-1);
    WaitForThreadpoolWorkCallbacks(state->work[0],FALSE);
}
#endif
