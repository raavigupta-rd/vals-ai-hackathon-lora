#include <bits/stdc++.h>
using namespace std;

struct RangeBIT {
    int n;
    vector<long long> t;
    RangeBIT(int n) : n(n), t(n + 2, 0) {}
    void add_pt(int i, long long v) {
        for (; i <= n + 1; i += i & -i) t[i] += v;
    }
    void add_range(int l, int r, long long v) {
        add_pt(l, v);
        add_pt(r + 1, -v);
    }
    long long get(int i) const {
        long long s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int LOG = 1;
    while ((1 << LOG) <= n) LOG++;
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<int> par(n + 1, 0), dep(n + 1, 0), tin(n + 1, 0), tout(n + 1, 0);
    vector<long long> base_f(n + 1, 0);

    int timer = 0;
    // Iterative DFS for Euler tour
    vector<pair<int, int>> st = {{1, 0}};
    dep[1] = 1;
    base_f[1] = a[1];
    while (!st.empty()) {
        auto& [u, idx] = st.back();
        if (idx == 0) tin[u] = ++timer;
        if (idx < (int)adj[u].size()) {
            int v = adj[u][idx++];
            if (v == par[u]) continue;
            par[v] = u;
            up[0][v] = u;
            dep[v] = dep[u] + 1;
            base_f[v] = base_f[u] + a[v];
            st.push_back({v, 0});
        } else {
            tout[u] = timer;
            st.pop_back();
        }
    }
    for (int k = 1; k < LOG; k++)
        for (int i = 1; i <= n; i++)
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

    RangeBIT slope(n), cst(n);
    auto root_sum = [&](int u) -> long long {
        if (u == 0) return 0LL;
        return base_f[u] + slope.get(tin[u]) * dep[u] + cst.get(tin[u]);
    };

    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int u; long long x;
            cin >> u >> x;
            slope.add_range(tin[u], tout[u], x);
            cst.add_range(tin[u], tout[u], x * (1LL - dep[u]));
        } else {
            int u, v;
            cin >> u >> v;
            int w = lca(u, v);
            cout << (root_sum(u) + root_sum(v) - root_sum(w) - root_sum(par[w])) << '\n';
        }
    }
    return 0;
}
