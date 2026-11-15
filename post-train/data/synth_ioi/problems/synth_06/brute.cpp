#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    const int INF = 2e9;
    vector<int> dist(n + 1);
    while (q--) {
        int s, t;
        cin >> s >> t;
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        dist[s] = 0;
        pq.push({0, s});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (u == t) break;
            if (d != dist[u]) continue;
            for (auto [v, w] : adj[u]) {
                int nd = max(d, w);
                if (nd < dist[v]) {
                    dist[v] = nd;
                    pq.push({nd, v});
                }
            }
        }
        cout << dist[t] << '\n';
    }
    return 0;
}
