/*
 polynomial_decoder.c
 Compile: gcc -O2 -std=c11 -o polynomial_decoder polynomial_decoder.c -lgmp

 Reads JSON-like input (see problem statement), decodes "base"/"value" strings
 into big integers, uses first k points (keys.k) and Lagrange interpolation
 evaluated at x=0 to compute constant term C (exact arithmetic via GMP).

 Outputs: C (decimal). If exact integer, prints integer. Otherwise prints
 a high-precision decimal approximation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <gmp.h>

#define MAX_POINTS 2048

// Read whole file into a malloc'd buffer (caller frees)
char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open"); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long len = ftell(f);
    rewind(f);
    char *buf = malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, 1, len, f);
    fclose(f);
    buf[r] = '\0';
    return buf;
}

// Find integer k in "keys": { ... "k": <num> ... }
int parse_k(const char *s) {
    const char *p = strstr(s, "\"keys\"");
    if (!p) p = strstr(s, "'keys'"); // just in case
    if (!p) return -1;
    p = strchr(p, '{');
    if (!p) return -1;
    const char *q = strstr(p, "\"k\"");
    if (!q) q = strstr(p, "'k'");
    if (!q) return -1;
    const char *colon = strchr(q, ':');
    if (!colon) return -1;
    int k = -1;
    if (sscanf(colon + 1, " %d", &k) == 1) return k;
    return -1;
}

// Utility: skip whitespace
const char *skip_ws(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) p++;
    return p;
}

// Extract next quoted string starting at p (returns malloc'd string, or NULL)
char *extract_quoted(const char *p) {
    p = strchr(p, '"');
    if (!p) p = strchr(p, '\'');
    if (!p) return NULL;
    char quote = *p;
    p++;
    const char *end = p;
    while (*end && *end != quote) {
        if (*end == '\\' && *(end+1)) end += 2;
        else end++;
    }
    if (!*end) return NULL;
    size_t len = end - p;
    char *out = malloc(len + 1);
    if (!out) return NULL;
    strncpy(out, p, len);
    out[len] = '\0';
    return out;
}

// Parse numbered points: looks for patterns like "1": { ... "base":"10", "value":"4" ... }
// Stores up to MAX_POINTS. Returned number of points stored; x_keys[] and base/value strings are set (valueStr/baseStr malloc'd).
int parse_points(const char *s, int *x_keys, char **baseStrs, char **valueStrs) {
    int count = 0;
    const char *p = s;
    while (p && *p) {
        // find opening quote for a key (digits)
        const char *q = strchr(p, '"');
        if (!q) break;
        q++;
        // check if digits follow
        if (!isdigit((unsigned char)*q)) { p = q; continue; }
        // read number
        char *endptr;
        long idx = strtol(q, &endptr, 10);
        if (endptr == q) { p = q; continue; }
        // confirm there's a closing quote after the digits
        if (*endptr != '"') { p = endptr; continue; }
        // now find the opening brace after the colon
        const char *brace = strchr(endptr, '{');
        if (!brace) break;
        // find the matching closing brace for this object (simple scan)
        const char *scan = brace + 1;
        int depth = 1;
        while (*scan && depth > 0) {
            if (*scan == '{') depth++;
            else if (*scan == '}') depth--;
            scan++;
        }
        if (depth != 0) break;
        size_t body_len = (scan - brace - 1);
        char *body = malloc(body_len + 1);
        strncpy(body, brace + 1, body_len);
        body[body_len] = '\0';

        // find base and value inside body
        char *bpos = strstr(body, "\"base\"");
        if (!bpos) bpos = strstr(body, "'base'");
        char *vpos = strstr(body, "\"value\"");
        if (!vpos) vpos = strstr(body, "'value'");

        char *base = NULL;
        char *value = NULL;
        if (bpos) {
            char *tmp = extract_quoted(bpos);
            if (tmp) {
                // either it returned "base" itself; find the next quoted (the value)
                free(tmp);
                // move to colon and extract next quoted string
                char *colon = strchr(bpos, ':');
                if (colon) base = extract_quoted(colon);
            }
        }
        if (vpos) {
            char *tmp = extract_quoted(vpos);
            if (tmp) {
                free(tmp);
                char *colon = strchr(vpos, ':');
                if (colon) value = extract_quoted(colon);
            }
        }

        if (base && value) {
            if (count < MAX_POINTS) {
                x_keys[count] = (int)idx;
                baseStrs[count] = base;
                valueStrs[count] = value;
                count++;
            } else {
                free(base); free(value);
            }
        } else {
            if (base) free(base);
            if (value) free(value);
        }

        free(body);
        // continue searching after the object
        p = scan;
    }
    return count;
}

// decode value string with base into mpz_t (supports base up to 62 via mpz_set_str)
int decode_to_mpz(mpz_t out, const char *baseStr, const char *valueStr) {
    long base = strtol(baseStr, NULL, 10);
    if (base < 2 || base > 62) return -1;
    // GMP expects digits in 0-9, A-Z, a-z. valueStr may be lower-case; convert to upper for digits>9 is acceptable up to 36.
    // But mpz_set_str accepts both lower and upper for bases beyond 36? GMP accepts letters a-z as digits > 35.
    // So use the string as-is but trim whitespace.
    while (isspace((unsigned char)*valueStr)) valueStr++;
    // mpz_set_str returns 0 on success
    int rc = mpz_set_str(out, valueStr, (int)base);
    if (rc != 0) {
        // fallback: manual decode (base up to 36 assumed)
        mpz_set_ui(out, 0);
        mpz_t b; mpz_init_set_ui(b, (unsigned int)base);
        const char *p = valueStr;
        while (*p) {
            if (isspace((unsigned char)*p)) { p++; continue; }
            char c = *p++;
            int d;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'A' && c <= 'Z') d = 10 + (c - 'A');
            else if (c >= 'a' && c <= 'z') d = 10 + (c - 'a');
            else { mpz_clear(b); return -2; }
            if (d >= base) { mpz_clear(b); return -3; }
            mpz_mul(out, out, b);
            mpz_add_ui(out, out, d);
        }
        mpz_clear(b);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.json\n", argv[0]);
        return 2;
    }
    char *buf = read_file(argv[1]);
    if (!buf) { fprintf(stderr, "Failed to read file\n"); return 2; }

    int k = parse_k(buf);
    if (k <= 0) { fprintf(stderr, "Failed to find keys.k or invalid k\n"); free(buf); return 2; }

    int x_keys[MAX_POINTS];
    char *bases[MAX_POINTS];
    char *values[MAX_POINTS];
    int npoints = parse_points(buf, x_keys, bases, values);
    if (npoints < k) {
        fprintf(stderr, "Not enough points: found %d, need %d\n", npoints, k);
        // free resources
        for (int i=0;i<npoints;i++){ free(bases[i]); free(values[i]); }
        free(buf);
        return 2;
    }

    // Use the first k points in ascending numeric order by key
    // The parse_points likely finds them in textual order; normalise by mapping
    // Build arrays of selected x_i and y_i
    // We'll select by smallest numeric key values among parsed points
    int order_idx[MAX_POINTS];
    for (int i = 0; i < npoints; ++i) order_idx[i] = i;
    // simple selection sort by x_keys to get first k smallest indices
    for (int i = 0; i < npoints; ++i) {
        for (int j = i+1; j < npoints; ++j) {
            if (x_keys[order_idx[j]] < x_keys[order_idx[i]]) {
                int tmp = order_idx[i]; order_idx[i] = order_idx[j]; order_idx[j] = tmp;
            }
        }
    }

    // prepare arrays for xs (int) and ys (mpz_t)
    int xs[MAX_POINTS];
    mpz_t ys[MAX_POINTS];
    for (int i = 0; i < k; ++i) {
        int idx = order_idx[i];
        xs[i] = x_keys[idx];
        mpz_init(ys[i]);
        if (decode_to_mpz(ys[i], bases[idx], values[idx]) != 0) {
            fprintf(stderr, "Failed to decode value for point %d (base=%s, value=%s)\n", x_keys[idx], bases[idx], values[idx]);
            // cleanup
            for (int t = 0; t <= i; ++t) mpz_clear(ys[t]);
            for (int t = 0; t < npoints; ++t) { free(bases[t]); free(values[t]); }
            free(buf);
            return 2;
        }
    }

    // free base/value strings (no longer needed)
    for (int i = 0; i < npoints; ++i) { free(bases[i]); free(values[i]); }
    free(buf);

    // Prepare numerator products N[i] = product_{j != i} (-x_j)  and denominators D[i] = product_{j != i} (x_i - x_j)
    mpz_t N[MAX_POINTS], D[MAX_POINTS];
    for (int i = 0; i < k; ++i) { mpz_init(N[i]); mpz_init(D[i]); mpz_set_ui(N[i], 1); mpz_set_ui(D[i], 1); }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (i == j) continue;
            // N[i] *= -x_j
            mpz_t tmp; mpz_init(tmp);
            mpz_set_si(tmp, - (long) xs[j]);
            mpz_mul(N[i], N[i], tmp);
            mpz_clear(tmp);
            // D[i] *= (x_i - x_j)
            mpz_t tmp2; mpz_init(tmp2);
            mpz_set_si(tmp2, (long)(xs[i] - xs[j]));
            mpz_mul(D[i], D[i], tmp2);
            mpz_clear(tmp2);
        }
    }

    // Sum terms as rationals: total = sum_i ( y_i * N[i] / D[i] )
    mpq_t total; mpq_init(total); mpq_set_ui(total, 0, 1);
    for (int i = 0; i < k; ++i) {
        mpz_t numer; mpz_init(numer);
        mpz_mul(numer, ys[i], N[i]); // numer = y_i * N[i]
        mpq_t term; mpq_init(term);
        mpq_set_num(term, numer);
        mpq_set_den(term, D[i]);
        mpq_canonicalize(term);
        mpq_add(total, total, term);
        mpz_clear(numer);
        mpq_clear(term);
    }

    // Clean up N/D/ys
    for (int i = 0; i < k; ++i) { mpz_clear(N[i]); mpz_clear(D[i]); mpz_clear(ys[i]); }

    // If denominator is 1, print numerator. Otherwise print a decimal approximation with high precision.
    mpz_t num, den;
    mpz_init(num); mpz_init(den);
    mpq_get_num(num, total);
    mpq_get_den(den, total);

    if (mpz_cmp_ui(den, 1) == 0) {
        // integer
        mpz_out_str(stdout, 10, num);
        printf("\n");
    } else {
        // decimal approximation with high precision
        // convert to mpf_t with sufficient precision (bits ~ #digits*3.33)
        size_t denom_digits = mpz_sizeinbase(den, 10);
        size_t num_digits = mpz_sizeinbase(num, 10);
        size_t want_digits = (num_digits > denom_digits ? num_digits : denom_digits) + 60;
        mpf_set_default_prec((mp_bitcnt_t)(want_digits * 4)); // coarse multiplier to get plenty bits
        mpf_t f; mpf_init(f);
        mpf_set_q(f, total);
        // print with GMP high-precision formatting
        // we use gmp_printf with format "%.Ff" which prints the full precision of the mpf_t
        gmp_printf("%.Ff\n", f);
        mpf_clear(f);
    }

    mpz_clear(num); mpz_clear(den);
    mpq_clear(total);

    return 0;
}