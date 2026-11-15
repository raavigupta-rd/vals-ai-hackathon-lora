#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

long long mod_pow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, m;
    cin >> n >> k >> m;
    vector<int> s(m);
    for (int i = 0; i < m; i++) cin >> s[i];
    sort(s.begin(), s.end());
    s.erase(unique(s.begin(), s.end()), s.end());

    vector<int> invs(n + 1, 1);
    for (int i = 2; i <= n; i++) invs[i] = (int)(1LL * (MOD - MOD / i) * invs[MOD % i] % MOD);

    vector<int> base(n + 1, 0), res(n + 1, 0), tmp(n + 1, 0);
    for (int x : s) if (x <= n) base[x] = invs[x];
    res[0] = 1;

    auto mul_trunc = [&](const vector<int>& A, const vector<int>& B) {
        fill(tmp.begin(), tmp.end(), 0);
        for (int i = 0; i <= n; i++) {
            if (!A[i]) continue;
            long long ai = A[i];
            for (int j = 0; i + j <= n; j++) {
                if (B[j]) tmp[i + j] = (int)((tmp[i + j] + ai * B[j]) % MOD);
            }
        }
        return tmp;
    };

    int e = k;
    while (e > 0) {
        if (e & 1) res = mul_trunc(res, base);
        e >>= 1;
        if (e > 0) base = mul_trunc(base, base);
    }

    long long fn = 1, fk = 1;
    for (int i = 1; i <= n; i++) fn = fn * i % MOD;
    for (int i = 1; i <= k; i++) fk = fk * i % MOD;
    cout << 1LL * res[n] * fn % MOD * mod_pow(fk, MOD - 2) % MOD << '\n';
    return 0;
}
