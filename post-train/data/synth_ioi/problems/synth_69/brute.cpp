#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

long long modpow(long long a, long long e) {
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
    int n;
    cin >> n;
    vector<int> A(n), B(n);
    for (int i = 0; i < n; i++) cin >> A[i] >> B[i];

    const int MAXF = 8010;
    vector<long long> fac(MAXF, 1), ifac(MAXF, 1);
    for (int i = 1; i < MAXF; i++) fac[i] = fac[i - 1] * i % MOD;
    ifac[MAXF - 1] = modpow(fac[MAXF - 1], MOD - 2);
    for (int i = MAXF - 2; i >= 0; i--) ifac[i] = ifac[i + 1] * (i + 1) % MOD;

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int x = A[i] + A[j];
            int y = B[i] + B[j];
            ans = (ans + fac[x + y] * ifac[x] % MOD * ifac[y]) % MOD;
        }
    }
    cout << ans << '\n';
    return 0;
}
