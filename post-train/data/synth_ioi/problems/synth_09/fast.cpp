#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> col(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), dep(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0);
    vector<int> tin(n + 1, 0), tout(n + 1, 0), ver(n + 1, 0);
    int timer = 0;
    {
        vector<pair<int, int>> st = {{1, 0}};
        while (!st.empty()) {
            auto& [u, i] = st.back();
            if (i == 0) {
                tin[u] = ++timer;
                ver[timer] = u;
            }
            if (i < (int)adj[u].size()) {
                int v = adj[u][i++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] ^ 1;
                st.push_back({v, 0});
            } else {
                tout[u] = timer;
                st.pop_back();
                if (par[u]) {
                    sz[par[u]] += sz[u];
                    if (sz[u] > sz[heavy[par[u]]]) heavy[par[u]] = u;
                }
            }
        }
    }

    vector<array<int, 2>> cnt(n + 1, {0, 0});
    long long cur_pairs = 0;
    vector<long long> ans(n + 1, 0);

    auto add_node = [&](int u) {
        int c = col[u], p = dep[u];
        cur_pairs += cnt[c][p];
        cnt[c][p]++;
    };
    auto rem_node = [&](int u) {
        int c = col[u], p = dep[u];
        cnt[c][p]--;
        cur_pairs -= cnt[c][p];
    };

    // Iterative DSU on tree
    struct Frame { int u; bool keep; int stage; int idx; };
    vector<Frame> st = {{1, true, 0, 0}};
    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.stage == 0) {
            if (f.idx < (int)adj[u].size()) {
                int v = adj[u][f.idx++];
                if (v == par[u] || v == heavy[u]) continue;
                st.push_back({v, false, 0, 0});
            } else {
                f.stage = 1;
                if (heavy[u]) st.push_back({heavy[u], true, 0, 0});
            }
        } else if (f.stage == 1) {
            for (int v : adj[u]) {
                if (v == par[u] || v == heavy[u]) continue;
                for (int t = tin[v]; t <= tout[v]; t++) add_node(ver[t]);
            }
            add_node(u);
            ans[u] = cur_pairs;
            if (!f.keep) {
                for (int t = tin[u]; t <= tout[u]; t++) rem_node(ver[t]);
            }
            st.pop_back();
        }
    }

    for (int i = 1; i <= n; i++) cout << ans[i] << '\n';
    return 0;
}
