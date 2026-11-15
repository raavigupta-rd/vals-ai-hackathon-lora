#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353LL;
const long long G = 3LL;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

void ntt(vector<long long>& a, bool invert) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = modpow(G, (MOD - 1) / len);
        if (invert) wlen = modpow(wlen, MOD - 2);
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            int half = len >> 1;
            for (int j = 0; j < half; j++) {
                long long u = a[i + j];
                long long v = a[i + j + half] * w % MOD;
                a[i + j] = u + v < MOD ? u + v : u + v - MOD;
                a[i + j + half] = u - v >= 0 ? u - v : u - v + MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (invert) {
        long long inv_n = modpow(n, MOD - 2);
        for (long long& x : a) x = x * inv_n % MOD;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int MAXM = 200000;
    int sz = 1;
    while (sz <= 2 * MAXM + 2) sz <<= 1;
    vector<long long> poly(sz, 0), diag(2 * MAXM + 2, 0);
    for (int i = 0; i < n; i++) {
        int m; long long w;
        cin >> m >> w;
        w %= MOD;
        poly[m] = (poly[m] + w) % MOD;
        diag[2 * m] = (diag[2 * m] + w * w) % MOD;
    }
    ntt(poly, false);
    for (long long& x : poly) x = x * x % MOD;
    ntt(poly, true);
    long long inv2 = (MOD + 1) / 2;
    while (q--) {
        int s;
        cin >> s;
        long long ans = 0;
        if (s <= 2 * MAXM) {
            ans = (poly[s] - diag[s] + MOD) % MOD * inv2 % MOD;
        }
        cout << ans << '\n';
    }
    return 0;
}
