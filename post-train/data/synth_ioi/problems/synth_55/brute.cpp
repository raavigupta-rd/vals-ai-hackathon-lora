#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1);
    while (q--) {
        int a, b; cin >> a >> b;
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        dist[a] = 0;
        pq.push({0, a});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (u == b) break;
            if (du != dist[u]) continue;
            for (auto [v, w] : adj[u]) if (du + w < dist[v]) {
                dist[v] = du + w;
                pq.push({dist[v], v});
            }
        }
        cout << dist[b] << '\n';
    }
    return 0;
}
