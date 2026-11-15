#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<long long> a(n + 1, 0), dist(n + 1, 0);
    vector<int> par(n + 1, 0), que(n + 1);
    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u;
            long long d;
            cin >> u >> d;
            a[u] += d;
        } else {
            int src;
            cin >> src;
            int head = 0, tail = 0;
            que[tail++] = src;
            par[src] = 0;
            dist[src] = 0;
            long long ans = 0;
            while (head < tail) {
                int u = que[head++];
                ans += a[u] * dist[u];
                for (auto [v, w] : adj[u]) {
                    if (v != par[u]) {
                        par[v] = u;
                        dist[v] = dist[u] + w;
                        que[tail++] = v;
                    }
                }
            }
            cout << ans << '\n';
        }
    }
    return 0;
}
