#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
    }

    // Iterative DFS preorder from 1
    vector<int> dfn(n + 1, 0), rev_dfn(n + 2, 0), par(n + 2, 0);
    int cnt = 0;
    {
        struct Frame { int u, idx; };
        vector<Frame> st = {{1, 0}};
        while (!st.empty()) {
            auto& f = st.back();
            int u = f.u;
            if (f.idx == 0) {
                dfn[u] = ++cnt;
                rev_dfn[cnt] = u;
            }
            bool pushed = false;
            while (f.idx < (int)g[u].size()) {
                int v = g[u][f.idx++];
                if (!dfn[v]) {
                    par[cnt + 1] = dfn[u]; // assigned right on entry of v!
                    st.push_back({v, 0});
                    pushed = true;
                    break;
                }
            }
            if (!pushed) st.pop_back();
        }
    }

    // Build reverse graph in DFS-number coordinates 1..cnt
    vector<vector<int>> rg(cnt + 2), bucket(cnt + 2);
    for (int u = 1; u <= n; u++) if (dfn[u]) {
        int du = dfn[u];
        for (int v : g[u]) if (dfn[v]) {
            rg[dfn[v]].push_back(du);
        }
    }

    vector<int> sdom(cnt + 2), idom(cnt + 2, 0), dsu_p(cnt + 2), mn(cnt + 2);
    for (int i = 1; i <= cnt; i++) {
        sdom[i] = dsu_p[i] = mn[i] = i;
    }

    vector<int> chain;
    chain.reserve(cnt + 2);
    auto eval_dsu = [&](int x) -> int {
        if (dsu_p[x] == x) return x;
        chain.clear();
        int cur = x;
        while (dsu_p[cur] != cur) {
            chain.push_back(cur);
            cur = dsu_p[cur];
        }
        for (int idx = (int)chain.size() - 1; idx >= 0; idx--) {
            int v = chain[idx];
            int p = dsu_p[v];
            if (sdom[mn[p]] < sdom[mn[v]]) mn[v] = mn[p];
            dsu_p[v] = cur;
        }
        return mn[x];
    };

    for (int i = cnt; i >= 2; i--) {
        for (int v : rg[i]) {
            int u = eval_dsu(v);
            sdom[i] = min(sdom[i], sdom[u]);
        }
        bucket[sdom[i]].push_back(i);
        int p = par[i];
        dsu_p[i] = p;
        for (int w : bucket[p]) {
            int u = eval_dsu(w);
            idom[w] = (sdom[u] < sdom[w]) ? u : p;
        }
        bucket[p].clear();
    }

    vector<int> depth(cnt + 2, 0);
    depth[1] = 1;
    for (int i = 2; i <= cnt; i++) {
        if (idom[i] != sdom[i]) idom[i] = idom[idom[i]];
        depth[i] = depth[idom[i]] + 1;
    }

    for (int u = 1; u <= n; u++) {
        int ans = dfn[u] ? depth[dfn[u]] : 0;
        cout << ans << (u == n ? '\n' : ' ');
    }
    return 0;
}
