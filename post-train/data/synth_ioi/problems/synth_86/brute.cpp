#include <bits/stdc++.h>
using namespace std;

static constexpr long long INF = (1LL << 60);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> par(n + 1, 0), ord = {1};
    vector<long long> pw(n + 1, 0);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, w] : adj[u]) if (v != par[u]) {
            par[v] = u;
            pw[v] = w;
            ord.push_back(v);
        }
    }
    vector<char> mark(n + 1, 0);
    vector<long long> dp0(n + 1), dp1(n + 1);
    while (q--) {
        int k; cin >> k;
        vector<int> ks(k);
        for (int i = 0; i < k; i++) { cin >> ks[i]; mark[ks[i]] = 1; }
        for (int i = n - 1; i >= 0; i--) {
            int u = ord[i];
            long long sumA = 0, best_d = 0;
            for (auto [v, w] : adj[u]) if (v != par[u]) {
                long long Av = min(dp0[v], dp1[v] + w);
                sumA += Av;
                best_d = min(best_d, dp1[v] - Av);
            }
            if (mark[u]) {
                dp0[u] = INF;
                dp1[u] = sumA;
            } else {
                dp0[u] = sumA;
                dp1[u] = sumA + best_d;
            }
        }
        cout << dp1[1] << '\n';
        for (int x : ks) mark[x] = 0;
    }
    return 0;
}
