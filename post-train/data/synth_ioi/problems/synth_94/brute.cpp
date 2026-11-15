#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

// 40-pt subtask solver: DAG Grundy + O(K * SZ^2) direct XOR convolution DP
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int v, e;
    long long K;
    cin >> v >> e >> K;
    vector<int> B(v + 1);
    for (int i = 1; i <= v; i++) cin >> B[i];
    vector<vector<int>> g(v + 1);
    for (int i = 0; i < e; i++) {
        int x, y; cin >> x >> y;
        g[x].push_back(y);
    }

    vector<int> gr(v + 1, 0), seen(v + 5, 0);
    int stamp = 0;
    for (int u = v; u >= 1; u--) {
        ++stamp;
        for (int nb : g[u]) seen[gr[nb]] = stamp;
        int mex = 0;
        while (seen[mex] == stamp) mex++;
        gr[u] = mex;
    }

    int SZ = 1;
    for (int u = 1; u <= v; u++) while (SZ <= (gr[u] ^ B[u])) SZ <<= 1;
    vector<long long> cnt(SZ, 0), dp(SZ, 0), ndp(SZ, 0);
    for (int u = 1; u <= v; u++) cnt[gr[u] ^ B[u]]++;
    dp[0] = 1;
    for (long long step = 0; step < K; step++) {
        fill(ndp.begin(), ndp.end(), 0);
        for (int a = 0; a < SZ; a++) if (dp[a]) {
            for (int b = 0; b < SZ; b++) if (cnt[b]) {
                ndp[a ^ b] = (ndp[a ^ b] + dp[a] * cnt[b]) % MOD;
            }
        }
        dp.swap(ndp);
    }
    long long tot = 0;
    for (int x = 1; x < SZ; x++) tot = (tot + dp[x]) % MOD;
    cout << tot << '\n';
    return 0;
}
