#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    struct Cat { int l, r; long long w; };
    vector<vector<Cat>> cats(n + 1);
    for (int i = 0; i < m; i++) {
        int u, l, r; long long w;
        cin >> u >> l >> r >> w;
        cats[u].push_back({l, r, w});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[1] = 0;
    pq.push({0, 1});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [l, r, w] : cats[u]) {
            long long nd = d + w;
            for (int v = l; v <= r; v++) if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << (dist[i] >= INF ? -1LL : dist[i]) << (i == n ? '\n' : ' ');
    }
    return 0;
}
