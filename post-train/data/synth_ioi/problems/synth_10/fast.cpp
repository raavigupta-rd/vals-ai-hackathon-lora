#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    vector<int> tin(n + 1, 0), first_pos(n + 1, 0), dep(n + 1, 0), par(n + 1, 0);
    vector<long long> dist(n + 1, 0);
    vector<int> euler;
    euler.reserve(2 * n);
    int timer = 0;

    vector<pair<int, int>> st = {{1, 0}};
    while (!st.empty()) {
        auto& [u, idx] = st.back();
        if (idx == 0) {
            tin[u] = ++timer;
            first_pos[u] = (int)euler.size();
            euler.push_back(u);
        }
        if (idx < (int)adj[u].size()) {
            auto [v, w] = adj[u][idx++];
            if (v == par[u]) continue;
            par[v] = u;
            dep[v] = dep[u] + 1;
            dist[v] = dist[u] + w;
            st.push_back({v, 0});
        } else {
            st.pop_back();
            if (!st.empty()) euler.push_back(st.back().first);
        }
    }

    int m = (int)euler.size();
    int LOG = __lg(m) + 1;
    vector<vector<int>> rmq(LOG, vector<int>(m));
    rmq[0] = euler;
    for (int k = 1; k < LOG; k++) {
        int half = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= m; i++) {
            int a = rmq[k - 1][i], b = rmq[k - 1][i + half];
            rmq[k][i] = (dep[a] < dep[b] ? a : b);
        }
    }
    auto lca = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        return dep[a] < dep[b] ? a : b;
    };
    auto tree_dist = [&](int u, int v) {
        int w = lca(u, v);
        return dist[u] + dist[v] - 2LL * dist[w];
    };

    while (q--) {
        int k;
        cin >> k;
        vector<int> v(k);
        for (int i = 0; i < k; i++) cin >> v[i];
        if (k <= 1) { cout << 0 << '\n'; continue; }
        sort(v.begin(), v.end(), [&](int a, int b) { return tin[a] < tin[b]; });
        long long sum = 0;
        for (int i = 0; i < k; i++) sum += tree_dist(v[i], v[(i + 1) % k]);
        cout << (sum / 2) << '\n';
    }
    return 0;
}
