#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<int> clan(n + 1);
    for (int i = 1; i <= n; i++) cin >> clan[i];
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1);
    for (int s = 1; s <= n; s++) {
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        dist[s] = 0;
        pq.push({0, s});
        long long ans = -1;
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du != dist[u]) continue;
            if (clan[u] != clan[s]) { ans = du; break; }
            for (auto [v, w] : adj[u]) if (du + w < dist[v]) {
                dist[v] = du + w;
                pq.push({dist[v], v});
            }
        }
        cout << ans << (s == n ? '\n' : ' ');
    }
    return 0;
}
