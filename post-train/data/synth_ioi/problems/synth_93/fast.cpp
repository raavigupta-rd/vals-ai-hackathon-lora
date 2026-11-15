#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 998244353;
static constexpr int G = 3;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

void ntt(vector<int>& a, bool invert) {
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
                int u = a[i + j];
                int v = (int)(a[i + j + half] * w % MOD);
                a[i + j] = u + v < MOD ? u + v : u + v - MOD;
                a[i + j + half] = u - v >= 0 ? u - v : u - v + MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (invert) {
        long long inv_n = modpow(n, MOD - 2);
        for (int& x : a) x = (int)(x * inv_n % MOD);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> A(n + 1, 0), W(n + 1, 0), sum_conv(n + 1, 0), F(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> A[i];
    for (int i = 1; i <= n; i++) cin >> W[i];
    F[0] = 1;

    vector<int> fa, fb;
    function<void(int, int)> cdq = [&](int l, int r) {
        if (l == r) {
            if (l > 0) F[l] = (sum_conv[l] % MOD) * A[l] % MOD;
            return;
        }
        int mid = (l + r) >> 1;
        cdq(l, mid);

        int lenL = mid - l + 1;
        int lenW = r - l;
        int sz = 1;
        while (sz < lenL + lenW) sz <<= 1;
        fa.assign(sz, 0);
        fb.assign(sz, 0);
        for (int i = 0; i < lenL; i++) fa[i] = (int)F[l + i];
        for (int k = 1; k <= lenW; k++) fb[k - 1] = (int)W[k];

        ntt(fa, false);
        ntt(fb, false);
        for (int i = 0; i < sz; i++) fa[i] = (int)(1LL * fa[i] * fb[i] % MOD);
        ntt(fa, true);

        // fa[idx] represents sum_{i + (k - 1) = idx} F[l + i] * W[k] -> target pos = l + idx + 1
        for (int pos = mid + 1; pos <= r; pos++) {
            int idx = pos - l - 1;
            sum_conv[pos] += fa[idx];
            if (sum_conv[pos] >= MOD) sum_conv[pos] -= MOD;
        }

        cdq(mid + 1, r);
    };

    cdq(0, n);
    for (int i = 1; i <= n; i++) {
        cout << F[i] << (i == n ? '\n' : ' ');
    }
    return 0;
}
