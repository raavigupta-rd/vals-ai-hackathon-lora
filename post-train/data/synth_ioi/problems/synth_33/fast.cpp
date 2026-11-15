#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

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
    int H, W, K;
    cin >> H >> W >> K;
    int LIM = H + W + 5;
    vector<long long> fact(LIM, 1), invf(LIM, 1);
    for (int i = 1; i < LIM; i++) fact[i] = fact[i - 1] * i % MOD;
    invf[LIM - 1] = modpow(fact[LIM - 1], MOD - 2);
    for (int i = LIM - 2; i >= 0; i--) invf[i] = invf[i + 1] * (i + 1) % MOD;

    auto ways = [&](int dr, int dc) -> long long {
        if (dr < 0 || dc < 0) return 0;
        return fact[dr + dc] * invf[dr] % MOD * invf[dc] % MOD;
    };

    vector<pair<int, int>> pts(K);
    for (int i = 0; i < K; i++) cin >> pts[i].first >> pts[i].second;
    pts.push_back({H, W});
    sort(pts.begin(), pts.end());

    int m = (int)pts.size();
    vector<long long> dp(m, 0);
    for (int i = 0; i < m; i++) {
        long long v = ways(pts[i].first - 1, pts[i].second - 1);
        for (int j = 0; j < i; j++) {
            if (pts[j].second <= pts[i].second) {
                v = (v - dp[j] * ways(pts[i].first - pts[j].first, pts[i].second - pts[j].second)) % MOD;
            }
        }
        if (v < 0) v += MOD;
        dp[i] = v;
    }
    cout << dp[m - 1] << '\n';
    return 0;
}
