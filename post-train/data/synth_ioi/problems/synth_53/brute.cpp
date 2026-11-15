#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    struct Edge { int u, v; long long w; };
    vector<Edge> es(m + 1);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= m; i++) {
        cin >> es[i].u >> es[i].v >> es[i].w;
        adj[es[i].u].push_back({es[i].v, i});
        adj[es[i].v].push_back({es[i].u, i});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1);
    while (q--) {
        int ban; cin >> ban;
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        dist[1] = 0;
        pq.push({0, 1});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (u == n) break;
            if (du != dist[u]) continue;
            for (auto [v, eid] : adj[u]) if (eid != ban) {
                if (du + es[eid].w < dist[v]) {
                    dist[v] = du + es[eid].w;
                    pq.push({dist[v], v});
                }
            }
        }
        cout << (dist[n] >= INF ? -1LL : dist[n]) << '\n';
    }
    return 0;
}
