#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    int sz = 1;
    while (sz < n) sz <<= 1;
    // Tree nodes: 1 .. 2*sz - 1; leaf for fort i (1-based) is sz + (i - 1)
    int V = 2 * sz;
    vector<vector<pair<int, long long>>> adj(V);
    for (int u = 1; u < sz; u++) {
        adj[u].push_back({2 * u, 0});
        adj[u].push_back({2 * u + 1, 0});
    }
    for (int i = 0; i < m; i++) {
        int u, l, r; long long w;
        cin >> u >> l >> r >> w;
        int src = sz + u - 1;
        for (l += sz - 1, r += sz; l < r; l >>= 1, r >>= 1) {
            if (l & 1) adj[src].push_back({l++, w});
            if (r & 1) adj[src].push_back({--r, w});
        }
    }

    const long long INF = (1LL << 60);
    vector<long long> dist(V, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[sz] = 0;
    pq.push({0, sz});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) if (d + w < dist[v]) {
            dist[v] = d + w;
            pq.push({dist[v], v});
        }
    }
    for (int i = 1; i <= n; i++) {
        long long d = dist[sz + i - 1];
        cout << (d >= INF ? -1LL : d) << (i == n ? '\n' : ' ');
    }
    return 0;
}
