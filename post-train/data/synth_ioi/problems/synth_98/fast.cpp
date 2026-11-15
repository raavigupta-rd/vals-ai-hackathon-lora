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

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int d;
    long long k;
    if (!(cin >> d >> k)) return 0;
    vector<int> a(d), c(d + 1, 0);
    for (int i = 0; i < d; i++) {
        cin >> a[i];
        a[i] %= MOD;
    }
    for (int j = 1; j <= d; j++) {
        cin >> c[j];
        c[j] %= MOD;
    }
    if (k < d) {
        cout << a[k] << '\n';
        return 0;
    }
    // Q(x) = 1 - sum_{j=1}^d c_j x^j
    vector<int> Q(d + 1, 0);
    Q[0] = 1;
    for (int j = 1; j <= d; j++) Q[j] = (MOD - c[j]) % MOD;
    // P(x) = (A(x) * Q(x)) mod x^d
    vector<int> P = poly_mul(a, Q);
    P.resize(d);

    while (k > 0) {
        vector<int> Qneg = Q;
        for (int i = 1; i < (int)Qneg.size(); i += 2) {
            if (Qneg[i]) Qneg[i] = MOD - Qneg[i];
        }
        vector<int> U = poly_mul(P, Qneg);
        vector<int> V = poly_mul(Q, Qneg);
        int parity = (int)(k & 1LL);
        vector<int> nP, nQ;
        nP.reserve((U.size() + 1) / 2);
        nQ.reserve((V.size() + 1) / 2);
        for (int i = parity; i < (int)U.size(); i += 2) nP.push_back(U[i]);
        for (int i = 0; i < (int)V.size(); i += 2) nQ.push_back(V[i]);
        P.swap(nP);
        Q.swap(nQ);
        k >>= 1;
    }
    int ans = P.empty() ? 0 : (int)((long long)P[0] * mod_pow(Q[0], MOD - 2) % MOD);
    cout << ans << '\n';
    return 0;
}
