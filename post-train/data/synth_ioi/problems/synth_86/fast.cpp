#include <bits/stdc++.h>
using namespace std;

static constexpr long long INF = (1LL << 60);

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

    int LOG = __lg(n) + 2;
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<vector<long long>> mnw(LOG, vector<long long>(n + 1, INF));
    vector<int> dep(n + 1, 0), tin(n + 1, 0);
    int timer = 0;
    {
        vector<pair<int, int>> st = {{1, 0}};
        dep[1] = 1;
        while (!st.empty()) {
            auto& [u, idx] = st.back();
            if (idx == 0) tin[u] = ++timer;
            if (idx < (int)adj[u].size()) {
                auto [v, w] = adj[u][idx++];
                if (v == up[0][u] && u != 1) continue;
                if (dep[v]) continue;
                dep[v] = dep[u] + 1;
                up[0][v] = u;
                mnw[0][v] = w;
                for (int b = 1; b < LOG; b++) {
                    int mid = up[b - 1][v];
                    up[b][v] = up[b - 1][mid];
                    mnw[b][v] = min(mnw[b - 1][v], mnw[b - 1][mid]);
                }
                st.push_back({v, 0});
            } else {
                st.pop_back();
            }
        }
    }

    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int b = 0; b < LOG; b++) if ((d >> b) & 1) u = up[b][u];
        if (u == v) return u;
        for (int b = LOG - 1; b >= 0; b--)
            if (up[b][u] != up[b][v]) { u = up[b][u]; v = up[b][v]; }
        return up[0][u];
    };
    auto path_min = [&](int anc, int u) -> long long {
        long long res = INF;
        int d = dep[u] - dep[anc];
        for (int b = 0; b < LOG; b++) if ((d >> b) & 1) {
            res = min(res, mnw[b][u]);
            u = up[b][u];
        }
        return res;
    };

    vector<char> mark(n + 1, 0);
    vector<vector<pair<int, long long>>> vt(n + 1);
    vector<long long> dp0(n + 1, 0), dp1(n + 1, 0);
    vector<int> nodes, all_vt, stk;

    while (q--) {
        int k; cin >> k;
        nodes.resize(k);
        for (int i = 0; i < k; i++) {
            cin >> nodes[i];
            mark[nodes[i]] = 1;
        }
        sort(nodes.begin(), nodes.end(), [&](int a, int b) { return tin[a] < tin[b]; });

        all_vt.clear();
        stk.clear();
        auto push_vt = [&](int x) { all_vt.push_back(x); vt[x].clear(); };

        push_vt(nodes[0]);
        stk.push_back(nodes[0]);
        for (int i = 1; i < k; i++) {
            int u = nodes[i];
            int w = lca(u, stk.back());
            if (w != stk.back()) {
                while (stk.size() >= 2 && dep[stk[stk.size() - 2]] >= dep[w]) {
                    int child = stk.back(); stk.pop_back();
                    int par = stk.back();
                    vt[par].push_back({child, path_min(par, child)});
                }
                if (stk.back() != w) {
                    int child = stk.back(); stk.pop_back();
                    push_vt(w);
                    vt[w].push_back({child, path_min(w, child)});
                    stk.push_back(w);
                }
            }
            push_vt(u);
            stk.push_back(u);
        }
        while (stk.size() >= 2) {
            int child = stk.back(); stk.pop_back();
            int par = stk.back();
            vt[par].push_back({child, path_min(par, child)});
        }
        int root = stk[0];

        // Iterative post-order over virtual tree rooted at `root`
        vector<int> vord = {root};
        for (size_t i = 0; i < vord.size(); i++) {
            for (auto [c, _] : vt[vord[i]]) vord.push_back(c);
        }
        for (int i = (int)vord.size() - 1; i >= 0; i--) {
            int u = vord[i];
            long long sumA = 0, best_delta = 0;
            for (auto [c, mw] : vt[u]) {
                long long Ac = min(dp0[c], dp1[c] + mw);
                sumA += Ac;
                best_delta = min(best_delta, dp1[c] - Ac);
            }
            if (mark[u]) {
                dp0[u] = INF;
                dp1[u] = sumA;
            } else {
                dp0[u] = sumA;
                dp1[u] = sumA + best_delta;
            }
        }

        cout << dp1[root] << '\n';
        for (int x : nodes) mark[x] = 0;
    }
    return 0;
}
