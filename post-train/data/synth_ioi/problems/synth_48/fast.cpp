#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), dep(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0), ord = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] + 1;
            ord.push_back(v);
        }
    }
    for (int i = n - 1; i >= 1; i--) {
        int u = ord[i];
        sz[par[u]] += sz[u];
        if (sz[u] > sz[heavy[par[u]]]) heavy[par[u]] = u;
    }

    vector<int> head(n + 1, 0), pos(n + 1, 0);
    int cur_pos = 0;
    {
        vector<pair<int, int>> st = {{1, 1}};
        while (!st.empty()) {
            auto [u, h] = st.back(); st.pop_back();
            for (int x = u; x != 0; x = heavy[x]) {
                head[x] = h;
                pos[x] = cur_pos++;
                for (int v : adj[x]) if (v != par[x] && v != heavy[x]) st.push_back({v, v});
            }
        }
    }

    int S = 1;
    while (S < n) S <<= 1;
    vector<int> seg(2 * S, 0);
    for (int i = 1; i <= n; i++) seg[S + pos[i]] = a[i];
    for (int i = S - 1; i >= 1; i--) seg[i] = max(seg[2 * i], seg[2 * i + 1]);

    auto seg_set = [&](int idx, int val) {
        int u = S + idx;
        seg[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = max(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_q = [&](int l, int r) {
        int res = 0;
        for (l += S, r += S + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = max(res, seg[l++]);
            if (r & 1) res = max(res, seg[--r]);
        }
        return res;
    };

    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int u, x; cin >> u >> x;
            seg_set(pos[u], x);
        } else {
            int u, v; cin >> u >> v;
            int ans = 0;
            while (head[u] != head[v]) {
                if (dep[head[u]] < dep[head[v]]) swap(u, v);
                ans = max(ans, seg_q(pos[head[u]], pos[u]));
                u = par[head[u]];
            }
            if (dep[u] > dep[v]) swap(u, v);
            ans = max(ans, seg_q(pos[u], pos[v]));
            cout << ans << '\n';
        }
    }
    return 0;
}
