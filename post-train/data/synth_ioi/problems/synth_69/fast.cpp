#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;
static constexpr int INV2 = (MOD + 1) / 2;

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
    int mx = 0;
    for (int i = 0; i < n; i++) {
        cin >> A[i] >> B[i];
        mx = max({mx, A[i], B[i]});
    }

    const int OFF = mx + 1;
    const int LIM = 2 * OFF + 1;
    vector<int> dp(LIM * LIM, 0);
    auto idx = [&](int x, int y) { return x * LIM + y; };

    for (int i = 0; i < n; i++) {
        int p = idx(OFF - A[i], OFF - B[i]);
        dp[p]++;
        if (dp[p] >= MOD) dp[p] -= MOD;
    }

    for (int x = 1; x < LIM; x++) {
        int row = x * LIM, prow = (x - 1) * LIM;
        for (int y = 1; y < LIM; y++) {
            long long s = (long long)dp[row + y] + dp[prow + y] + dp[row + y - 1];
            dp[row + y] = (int)(s % MOD);
        }
    }

    int MAXF = 4 * mx + 5;
    vector<long long> fac(MAXF, 1), ifac(MAXF, 1);
    for (int i = 1; i < MAXF; i++) fac[i] = fac[i - 1] * i % MOD;
    ifac[MAXF - 1] = modpow(fac[MAXF - 1], MOD - 2);
    for (int i = MAXF - 2; i >= 0; i--) ifac[i] = ifac[i + 1] * (i + 1) % MOD;
    auto C = [&](int nn, int kk) -> long long {
        return fac[nn] * ifac[kk] % MOD * ifac[nn - kk] % MOD;
    };

    long long total = 0;
    for (int i = 0; i < n; i++) {
        total += dp[idx(OFF + A[i], OFF + B[i])];
        total -= C(2 * A[i] + 2 * B[i], 2 * A[i]);
        total %= MOD;
    }
    if (total < 0) total += MOD;
    cout << (total * INV2) % MOD << '\n';
    return 0;
}
