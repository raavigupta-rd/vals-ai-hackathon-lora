#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n + 1), sz(n + 1, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a; sz[a] += sz[b];
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    struct Edge { int u, v, w; };
    vector<Edge> es(m);
    for (int i = 0; i < m; i++) cin >> es[i].u >> es[i].v >> es[i].w;
    sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) { return a.w < b.w; });

    DSU dsu(n);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (const auto& e : es) {
        if (dsu.unite(e.u, e.v)) {
            adj[e.u].push_back({e.v, e.w});
            adj[e.v].push_back({e.u, e.w});
        }
    }

    int LOG = 1;
    while ((1 << LOG) <= n) LOG++;
    vector<int> dep(n + 1, 0);
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<vector<int>> mx(LOG, vector<int>(n + 1, 0));

    vector<int> st = {1};
    dep[1] = 1;
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        for (auto [v, w] : adj[u]) {
            if (!dep[v]) {
                dep[v] = dep[u] + 1;
                up[0][v] = u;
                mx[0][v] = w;
                st.push_back(v);
            }
        }
    }
    for (int k = 1; k < LOG; k++) {
        for (int i = 1; i <= n; i++) {
            int mid = up[k - 1][i];
            up[k][i] = up[k - 1][mid];
            mx[k][i] = max(mx[k - 1][i], mx[k - 1][mid]);
        }
    }

    while (q--) {
        int u, v;
        cin >> u >> v;
        if (u == v) { cout << 0 << '\n'; continue; }
        int ans = 0;
        if (dep[u] < dep[v]) swap(u, v);
        int diff = dep[u] - dep[v];
        for (int k = 0; k < LOG; k++) if ((diff >> k) & 1) {
            ans = max(ans, mx[k][u]);
            u = up[k][u];
        }
        if (u != v) {
            for (int k = LOG - 1; k >= 0; k--) {
                if (up[k][u] != up[k][v]) {
                    ans = max({ans, mx[k][u], mx[k][v]});
                    u = up[k][u];
                    v = up[k][v];
                }
            }
            ans = max({ans, mx[0][u], mx[0][v]});
        }
        cout << ans << '\n';
    }
    return 0;
}
