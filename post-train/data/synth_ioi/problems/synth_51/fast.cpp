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
    vector<array<long long, 2>> d(n + 1, {INF, INF});
    vector<array<int, 2>> src(n + 1, {0, 0});
    using T = tuple<long long, int, int>; // dist, node, clan
    priority_queue<T, vector<T>, greater<>> pq;

    for (int i = 1; i <= n; i++) {
        d[i][0] = 0;
        src[i][0] = clan[i];
        pq.push({0, i, clan[i]});
    }

    while (!pq.empty()) {
        auto [du, u, cu] = pq.top(); pq.pop();
        if (!((src[u][0] == cu && d[u][0] == du) || (src[u][1] == cu && d[u][1] == du))) continue;
        for (auto [v, w] : adj[u]) {
            long long nd = du + w;
            if (src[v][0] == cu) {
                if (nd < d[v][0]) { d[v][0] = nd; pq.push({nd, v, cu}); }
            } else if (nd < d[v][0]) {
                d[v][1] = d[v][0]; src[v][1] = src[v][0];
                d[v][0] = nd; src[v][0] = cu;
                pq.push({nd, v, cu});
            } else if (src[v][1] == cu) {
                if (nd < d[v][1]) { d[v][1] = nd; pq.push({nd, v, cu}); }
            } else if (nd < d[v][1]) {
                d[v][1] = nd; src[v][1] = cu;
                pq.push({nd, v, cu});
            }
        }
    }

    for (int i = 1; i <= n; i++) {
        long long ans = (src[i][0] != clan[i] ? d[i][0] : d[i][1]);
        cout << (ans >= INF ? -1LL : ans) << (i == n ? '\n' : ' ');
    }
    return 0;
}
