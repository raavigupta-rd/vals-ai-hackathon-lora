#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<long long>> g(n, vector<long long>(n, 0));
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u - 1][v - 1] = (g[u - 1][v - 1] + 1) % MOD;
    }
    while (q--) {
        int u, v; long long k;
        cin >> u >> v >> k;
        vector<long long> dp(n, 0), ndp(n, 0);
        dp[u - 1] = 1;
        for (long long step = 0; step < k; step++) {
            fill(ndp.begin(), ndp.end(), 0);
            for (int i = 0; i < n; i++) if (dp[i]) {
                for (int j = 0; j < n; j++) if (g[i][j]) {
                    ndp[j] = (ndp[j] + dp[i] * g[i][j]) % MOD;
                }
            }
            dp.swap(ndp);
        }
        cout << dp[v - 1] << '\n';
    }
    return 0;
}
