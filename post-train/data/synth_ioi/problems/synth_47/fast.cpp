#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Euler tour RMQ LCA for O(1) tree distance
    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
    euler.reserve(2 * n);
    {
        vector<pair<int, int>> st = {{1, 0}};
        while (!st.empty()) {
            auto& [u, idx] = st.back();
            if (idx == 0) {
                first_pos[u] = (int)euler.size();
                euler.push_back(u);
            }
            if (idx < (int)adj[u].size()) {
                int v = adj[u][idx++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] + 1;
                st.push_back({v, 0});
            } else {
                st.pop_back();
                if (!st.empty()) euler.push_back(st.back().first);
            }
        }
    }
    int m = (int)euler.size(), LOG = __lg(m) + 1;
    vector<vector<int>> rmq(LOG, vector<int>(m));
    rmq[0] = euler;
    for (int k = 1; k < LOG; k++) {
        int h = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= m; i++) {
            int a = rmq[k - 1][i], b = rmq[k - 1][i + h];
            rmq[k][i] = dep[a] < dep[b] ? a : b;
        }
    }
    auto dist = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        int w = dep[a] < dep[b] ? a : b;
        return dep[u] + dep[v] - 2 * dep[w];
    };

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), cpar(n + 1, 0), tmp_ord, tmp_par(n + 1, 0);

    function<void(int, int)> build_cd = [&](int start, int p_cent) {
        tmp_ord.clear();
        tmp_ord.push_back(start);
        tmp_par[start] = 0;
        for (size_t i = 0; i < tmp_ord.size(); i++) {
            int u = tmp_ord[i];
            sz[u] = 1;
            for (int v : adj[u]) if (!dead[v] && v != tmp_par[u]) {
                tmp_par[v] = u;
                tmp_ord.push_back(v);
            }
        }
        int tot = (int)tmp_ord.size();
        for (int i = tot - 1; i >= 1; i--) sz[tmp_par[tmp_ord[i]]] += sz[tmp_ord[i]];
        int c = start;
        while (true) {
            int nxt = 0;
            for (int v : adj[c]) if (!dead[v] && v != tmp_par[c] && sz[v] > tot / 2) {
                nxt = v; break;
            }
            if (!nxt) break;
            c = nxt;
        }
        dead[c] = 1;
        cpar[c] = p_cent;
        for (int v : adj[c]) if (!dead[v]) build_cd(v, c);
    };
    build_cd(1, 0);

    const int INF = 1e9;
    vector<int> best(n + 1, INF);
    auto activate = [&](int u) {
        for (int c = u; c != 0; c = cpar[c]) best[c] = min(best[c], dist(u, c));
    };
    auto query = [&](int u) {
        int ans = INF;
        for (int c = u; c != 0; c = cpar[c]) ans = min(ans, best[c] + dist(u, c));
        return ans;
    };

    activate(1);
    while (q--) {
        int t, u;
        cin >> t >> u;
        if (t == 1) activate(u);
        else cout << query(u) << '\n';
    }
    return 0;
}
