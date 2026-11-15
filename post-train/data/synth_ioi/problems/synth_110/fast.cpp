#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;
static const int G = 3;

long long mod_pow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

void ntt(vector<int>& a, bool inv) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = mod_pow(G, (MOD - 1) / len);
        if (inv) wlen = mod_pow(wlen, MOD - 2);
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
    if (inv) {
        long long inv_n = mod_pow(n, MOD - 2);
        for (int& x : a) x = (int)(x * inv_n % MOD);
    }
}

vector<int> poly_mul(const vector<int>& a, const vector<int>& b) {
    if (a.empty() || b.empty()) return {};
    int need = (int)a.size() + (int)b.size() - 1;
    int sz = 1;
    while (sz < need) sz <<= 1;
    vector<int> fa(sz, 0), fb(sz, 0);
    copy(a.begin(), a.end(), fa.begin());
    copy(b.begin(), b.end(), fb.begin());
    ntt(fa, false);
    ntt(fb, false);
    for (int i = 0; i < sz; i++) fa[i] = (int)((long long)fa[i] * fb[i] % MOD);
    ntt(fa, true);
    fa.resize(need);
    return fa;
}

vector<int> poly_inv(const vector<int>& a, int n) {
    vector<int> b = {(int)mod_pow(a[0], MOD - 2)};
    int k = 1;
    while (k < n) {
        k <<= 1;
        int sz = k << 1;
        vector<int> fa(sz, 0), fb(sz, 0);
        int take = min((int)a.size(), k);
        for (int i = 0; i < take; i++) fa[i] = a[i];
        for (int i = 0; i < (int)b.size(); i++) fb[i] = b[i];
        ntt(fa, false);
        ntt(fb, false);
        for (int i = 0; i < sz; i++) {
            long long term = (2 - 1LL * fa[i] * fb[i] % MOD + MOD) % MOD;
            fb[i] = (int)(1LL * fb[i] * term % MOD);
        }
        ntt(fb, true);
        fb.resize(k);
        b.swap(fb);
    }
    b.resize(n);
    return b;
}

vector<int> poly_deriv(const vector<int>& a) {
    if ((int)a.size() <= 1) return {0};
    vector<int> r(a.size() - 1);
    for (size_t i = 1; i < a.size(); i++) r[i - 1] = (int)(1LL * a[i] * i % MOD);
    return r;
}

vector<int> poly_integ(const vector<int>& a, const vector<int>& invs) {
    vector<int> r(a.size() + 1, 0);
    for (size_t i = 0; i < a.size(); i++) r[i + 1] = (int)(1LL * a[i] * invs[i + 1] % MOD);
    return r;
}

vector<int> poly_ln(const vector<int>& a, int n, const vector<int>& invs) {
    // Assumes a[0] == 1
    vector<int> d = poly_deriv(a);
    if ((int)d.size() > n) d.resize(n);
    vector<int> ia = poly_inv(a, n);
    vector<int> prod = poly_mul(d, ia);
    if ((int)prod.size() > n - 1) prod.resize(max(0, n - 1));
    vector<int> res = poly_integ(prod, invs);
    res.resize(n, 0);
    return res;
}

vector<int> poly_exp(const vector<int>& a, int n, const vector<int>& invs) {
    // Assumes a[0] == 0
    vector<int> b = {1};
    int k = 1;
    while (k < n) {
        k <<= 1;
        vector<int> lb = poly_ln(b, k, invs);
        vector<int> rhs(k, 0);
        int take = min((int)a.size(), k);
        for (int i = 0; i < k; i++) {
            int av = (i < take ? a[i] : 0);
            rhs[i] = (av - lb[i] + MOD) % MOD;
        }
        rhs[0] = (rhs[0] + 1) % MOD;
        vector<int> nb = poly_mul(b, rhs);
        nb.resize(k, 0);
        b.swap(nb);
    }
    b.resize(n);
    return b;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, m;
    if (!(cin >> n >> k >> m)) return 0;
    vector<int> s(m);
    for (int i = 0; i < m; i++) cin >> s[i];
    sort(s.begin(), s.end());
    s.erase(unique(s.begin(), s.end()), s.end());
    m = (int)s.size();

    int s0 = s[0];
    if (1LL * k * s0 > n) {
        cout << 0 << '\n';
        return 0;
    }

    vector<int> invs(n + 2, 1);
    for (int i = 2; i <= n + 1; i++) {
        invs[i] = (int)(1LL * (MOD - MOD / i) * invs[MOD % i] % MOD);
    }
    long long fact_n = 1, fact_k = 1;
    for (int i = 1; i <= n; i++) fact_n = fact_n * i % MOD;
    for (int i = 1; i <= k; i++) fact_k = fact_k * i % MOD;

    int deg = n - k * s0;
    int len = deg + 1;
    // B(x) = sum_{s in S} (1/s) x^{s - s0}, normalized by B(0)^{-1} = s0
    vector<int> B(len, 0);
    for (int len_c : s) {
        if (len_c - s0 <= deg) {
            B[len_c - s0] = (int)(1LL * invs[len_c] * s0 % MOD);
        }
    }
    // B[0] == 1; compute exp(k * ln(B)) mod x^len
    vector<int> lnB = poly_ln(B, len, invs);
    for (int i = 0; i < len; i++) lnB[i] = (int)(1LL * lnB[i] * k % MOD);
    vector<int> pB = poly_exp(lnB, len, invs);

    long long coeff = pB[deg] * mod_pow(invs[s0], k) % MOD;
    long long ans = coeff * fact_n % MOD * mod_pow(fact_k, MOD - 2) % MOD;
    cout << ans << '\n';
    return 0;
}
