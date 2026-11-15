#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    vector<pair<int, int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }

    vector<int> tin(n + 1, 0), low(n + 1, 0);
    vector<char> is_bridge(m, 0);
    int timer = 0;

    // Iterative Tarjan DFS
    struct Frame { int u, pe, idx; };
    vector<Frame> st = {{1, -1, 0}};
    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.idx == 0) tin[u] = low[u] = ++timer;
        if (f.idx < (int)adj[u].size()) {
            auto [v, eid] = adj[u][f.idx++];
            if (eid == f.pe) continue;
            if (tin[v]) {
                low[u] = min(low[u], tin[v]);
            } else {
                st.push_back({v, eid, 0});
            }
        } else {
            int pe = f.pe;
            st.pop_back();
            if (!st.empty()) {
                int p = st.back().u;
                low[p] = min(low[p], low[u]);
                if (low[u] > tin[p]) is_bridge[pe] = 1;
            }
        }
    }

    vector<int> comp(n + 1, 0);
    int ccnt = 0;
    for (int i = 1; i <= n; i++) if (!comp[i]) {
        ++ccnt;
        vector<int> qv = {i};
        comp[i] = ccnt;
        while (!qv.empty()) {
            int u = qv.back(); qv.pop_back();
            for (auto [v, eid] : adj[u]) {
                if (!is_bridge[eid] && !comp[v]) {
                    comp[v] = ccnt;
                    qv.push_back(v);
                }
            }
        }
    }

    vector<vector<int>> tree(ccnt + 1);
    for (int i = 0; i < m; i++) if (is_bridge[i]) {
        int a = comp[edges[i].first], b = comp[edges[i].second];
        tree[a].push_back(b);
        tree[b].push_back(a);
    }

    int LOG = 1;
    while ((1 << LOG) <= ccnt + 1) LOG++;
    vector<vector<int>> up(LOG, vector<int>(ccnt + 1, 1));
    vector<int> dep(ccnt + 1, 0);
    {
        vector<int> qv = {1};
        dep[1] = 1;
        for (int i = 0; i < (int)qv.size(); i++) {
            int u = qv[i];
            for (int v : tree[u]) if (!dep[v]) {
                dep[v] = dep[u] + 1;
                up[0][v] = u;
                qv.push_back(v);
            }
        }
    }
    for (int k = 1; k < LOG; k++)
        for (int i = 1; i <= ccnt; i++)
            up[k][i] = up[k - 1][up[k - 1][i]];

    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int k = 0; k < LOG; k++) if ((d >> k) & 1) u = up[k][u];
        if (u == v) return u;
        for (int k = LOG - 1; k >= 0; k--)
            if (up[k][u] != up[k][v]) { u = up[k][u]; v = up[k][v]; }
        return up[0][u];
    };

    while (q--) {
        int a, b;
        cin >> a >> b;
        int ca = comp[a], cb = comp[b];
        int w = lca(ca, cb);
        cout << (dep[ca] + dep[cb] - 2 * dep[w]) << '\n';
    }
    return 0;
}
