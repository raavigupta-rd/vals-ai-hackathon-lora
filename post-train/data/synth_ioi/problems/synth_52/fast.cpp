#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<int> tin(n + 1, 0), low(n + 1, 0), stk;
    vector<vector<int>> bct(2 * n + 5);
    int timer = 0, bcnt = 0;

    // Iterative Tarjan for vertex biconnected components
    struct Frame { int u, p, idx, child_v; bool returning; };
    vector<Frame> call = {{1, 0, 0, 0, false}};
    while (!call.empty()) {
        auto& f = call.back();
        int u = f.u;
        if (f.idx == 0 && !f.returning) {
            tin[u] = low[u] = ++timer;
            stk.push_back(u);
        }
        if (f.returning) {
            int v = f.child_v;
            f.returning = false;
            low[u] = min(low[u], low[v]);
            if (low[v] >= tin[u]) {
                int bnode = n + (++bcnt);
                if ((int)bct.size() <= bnode) bct.resize(bnode + 5);
                while (true) {
                    int x = stk.back(); stk.pop_back();
                    bct[bnode].push_back(x);
                    bct[x].push_back(bnode);
                    if (x == v) break;
                }
                bct[bnode].push_back(u);
                bct[u].push_back(bnode);
            }
        }
        bool pushed = false;
        while (f.idx < (int)g[u].size()) {
            int v = g[u][f.idx++];
            if (v == f.p) continue;
            if (tin[v]) {
                low[u] = min(low[u], tin[v]);
            } else {
                f.child_v = v;
                f.returning = true;
                call.push_back({v, u, 0, 0, false});
                pushed = true;
                break;
            }
        }
        if (!pushed) call.pop_back();
    }

    int V = n + bcnt;
    int LOG = 1;
    while ((1 << LOG) <= V + 1) LOG++;
    vector<vector<int>> up(LOG, vector<int>(V + 1, 1));
    vector<int> dep(V + 1, 0), qv = {1};
    dep[1] = 1;
    for (size_t i = 0; i < qv.size(); i++) {
        int u = qv[i];
        for (int v : bct[u]) if (!dep[v]) {
            dep[v] = dep[u] + 1;
            up[0][v] = u;
            qv.push_back(v);
        }
    }
    for (int b = 1; b < LOG; b++)
        for (int i = 1; i <= V; i++)
            up[b][i] = up[b - 1][up[b - 1][i]];

    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int b = 0; b < LOG; b++) if ((d >> b) & 1) u = up[b][u];
        if (u == v) return u;
        for (int b = LOG - 1; b >= 0; b--)
            if (up[b][u] != up[b][v]) { u = up[b][u]; v = up[b][v]; }
        return up[0][u];
    };

    while (q--) {
        int a, b;
        cin >> a >> b;
        int w = lca(a, b);
        int d = dep[a] + dep[b] - 2 * dep[w];
        cout << (d / 2 - 1) << '\n';
    }
    return 0;
}
