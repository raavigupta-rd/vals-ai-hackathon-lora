#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> eu(n), ev(n);
    vector<long long> ew(n);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= n - 1; i++) {
        cin >> eu[i] >> ev[i] >> ew[i];
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }

    vector<int> par(n + 1), que(n + 1);
    vector<long long> dist(n + 1);
    auto farthest = [&](int src) -> pair<int, long long> {
        int head = 0, tail = 0;
        que[tail++] = src;
        par[src] = 0;
        dist[src] = 0;
        int best_u = src;
        long long best_d = 0;
        while (head < tail) {
            int u = que[head++];
            if (dist[u] > best_d) {
                best_d = dist[u];
                best_u = u;
            }
            for (auto [v, id] : adj[u]) {
                if (v != par[u]) {
                    par[v] = u;
                    dist[v] = dist[u] + ew[id];
                    que[tail++] = v;
                }
            }
        }
        return {best_u, best_d};
    };

    while (q--) {
        int eid;
        long long nw;
        cin >> eid >> nw;
        ew[eid] = nw;
        auto [x, _] = farthest(1);
        auto [y, diam] = farthest(x);
        cout << diam << '\n';
    }
    return 0;
}
