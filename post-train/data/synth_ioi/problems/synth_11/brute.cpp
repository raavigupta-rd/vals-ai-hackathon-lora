#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, g;
    cin >> n >> m >> g;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    for (int j = 0; j < g; j++) {
        int k; long long c;
        cin >> k >> c;
        vector<int> mem(k);
        for (int i = 0; i < k; i++) cin >> mem[i];
        for (int a = 0; a < k; a++) {
            for (int b = a + 1; b < k; b++) {
                adj[mem[a]].push_back({mem[b], c});
                adj[mem[b]].push_back({mem[a], c});
            }
        }
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[1] = 0;
    pq.push({0, 1});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (d + w < dist[v]) {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << (dist[i] == INF ? -1LL : dist[i]) << (i == n ? '\n' : ' ');
    }
    return 0;
}
