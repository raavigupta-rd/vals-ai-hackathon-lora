#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<long long> S(n + 1, 0), P(n + 1, 0), dist(n + 1, 0), dp(n + 1, 0);
    for (int u = 2; u <= n; u++) cin >> S[u] >> P[u];
    vector<int> par(n + 1, 0), ord = {1};
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, w] : adj[u]) if (v != par[u]) {
            par[v] = u;
            dist[v] = dist[u] + w;
            ord.push_back(v);
        }
    }
    for (int i = 1; i < n; i++) {
        int u = ord[i];
        __int128 best = (__int128)1 << 120;
        for (int a = par[u]; a != 0; a = par[a]) {
            __int128 cand = (__int128)dp[a] + (__int128)(dist[u] - dist[a]) * P[u] + S[u];
            if (cand < best) best = cand;
        }
        dp[u] = (long long)best;
    }
    for (int u = 2; u <= n; u++) cout << dp[u] << (u == n ? '\n' : ' ');
    return 0;
}
