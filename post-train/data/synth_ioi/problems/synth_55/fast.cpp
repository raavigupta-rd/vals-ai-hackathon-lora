#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n + 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        p[b] = a; return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, long long>>> adj(n + 1), tadj(n + 1);
    DSU dsu(n);
    vector<int> specials;
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        if (dsu.unite(u, v)) {
            tadj[u].push_back({v, w});
            tadj[v].push_back({u, w});
        } else {
            specials.push_back(u);
        }
    }
    sort(specials.begin(), specials.end());
    specials.erase(unique(specials.begin(), specials.end()), specials.end());

    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
    vector<long long> tdist(n + 1, 0);
    euler.reserve(2 * n);
    {
        vector<pair<int, int>> st = {{1, 0}};
        while (!st.empty()) {
            auto& [u, idx] = st.back();
            if (idx == 0) {
                first_pos[u] = (int)euler.size();
                euler.push_back(u);
            }
            if (idx < (int)tadj[u].size()) {
                auto [v, w] = tadj[u][idx++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] + 1;
                tdist[v] = tdist[u] + w;
                st.push_back({v, 0});
            } else {
                st.pop_back();
                if (!st.empty()) euler.push_back(st.back().first);
            }
        }
    }
    int em = (int)euler.size(), LOG = __lg(em) + 1;
    vector<vector<int>> rmq(LOG, vector<int>(em));
    rmq[0] = euler;
    for (int k = 1; k < LOG; k++) {
        int h = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= em; i++) {
            int a = rmq[k - 1][i], b = rmq[k - 1][i + h];
            rmq[k][i] = dep[a] < dep[b] ? a : b;
        }
    }
    auto tree_d = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        int w = dep[a] < dep[b] ? a : b;
        return tdist[u] + tdist[v] - 2LL * tdist[w];
    };

    const long long INF = (1LL << 60);
    int S = (int)specials.size();
    vector<vector<long long>> sdist(S, vector<long long>(n + 1, INF));
    for (int i = 0; i < S; i++) {
        int src = specials[i];
        auto& d = sdist[i];
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        d[src] = 0;
        pq.push({0, src});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du != d[u]) continue;
            for (auto [v, w] : adj[u]) if (du + w < d[v]) {
                d[v] = du + w;
                pq.push({d[v], v});
            }
        }
    }

    while (q--) {
        int a, b; cin >> a >> b;
        long long ans = tree_d(a, b);
        for (int i = 0; i < S; i++) ans = min(ans, sdist[i][a] + sdist[i][b]);
        cout << ans << '\n';
    }
    return 0;
}
