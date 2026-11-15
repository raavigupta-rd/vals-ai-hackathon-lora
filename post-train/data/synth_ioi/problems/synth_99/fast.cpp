#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    static const int LOG = 19;
    vector<array<int, LOG>> anc(n + 1);
    vector<array<long long, LOG>> d_anc(n + 1);
    vector<int> len_anc(n + 1, 0);

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), par_tmp(n + 1, 0), nodes;
    nodes.reserve(n);

    auto build_cd = [&](auto& self, int entry) -> void {
        nodes.clear();
        par_tmp[entry] = 0;
        nodes.push_back(entry);
        for (size_t i = 0; i < nodes.size(); i++) {
            int u = nodes[i];
            sz[u] = 1;
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_tmp[u]) {
                    par_tmp[v] = u;
                    nodes.push_back(v);
                }
            }
        }
        int tot = (int)nodes.size();
        for (int i = tot - 1; i >= 0; i--) {
            int u = nodes[i];
            if (par_tmp[u]) sz[par_tmp[u]] += sz[u];
        }
        int cent = entry;
        for (int u : nodes) {
            int mx = tot - sz[u];
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_tmp[u]) mx = max(mx, sz[v]);
            }
            if (mx * 2 <= tot) {
                cent = u;
                break;
            }
        }

        // Iterative BFS from cent to record distance to cent
        nodes.clear();
        par_tmp[cent] = 0;
        int lvl_c = len_anc[cent];
        anc[cent][lvl_c] = cent;
        d_anc[cent][lvl_c] = 0;
        len_anc[cent]++;
        nodes.push_back(cent);
        for (size_t i = 0; i < nodes.size(); i++) {
            int u = nodes[i];
            long long du = d_anc[u][len_anc[u] - 1];
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_tmp[u]) {
                    par_tmp[v] = u;
                    int lv = len_anc[v];
                    anc[v][lv] = cent;
                    d_anc[v][lv] = du + w;
                    len_anc[v]++;
                    nodes.push_back(v);
                }
            }
        }

        dead[cent] = 1;
        for (auto [v, w] : adj[cent]) {
            if (!dead[v]) self(self, v);
        }
    };

    build_cd(build_cd, 1);

    vector<long long> sum_a(n + 1, 0), sum_ad(n + 1, 0), sub_ad(n + 1, 0);

    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u;
            long long delta;
            cin >> u >> delta;
            int k = len_anc[u];
            for (int l = 0; l < k; l++) {
                int c = anc[u][l];
                long long d = d_anc[u][l];
                sum_a[c] += delta;
                sum_ad[c] += delta * d;
                if (l + 1 < k) {
                    int ch = anc[u][l + 1];
                    sub_ad[ch] += delta * d;
                }
            }
        } else {
            int u;
            cin >> u;
            long long ans = 0;
            int k = len_anc[u];
            for (int l = 0; l < k; l++) {
                int c = anc[u][l];
                long long d = d_anc[u][l];
                if (l == k - 1) {
                    ans += sum_ad[c];
                } else {
                    int ch = anc[u][l + 1];
                    ans += (sum_ad[c] - sub_ad[ch]) + (sum_a[c] - sum_a[ch]) * d;
                }
            }
            cout << ans << '\n';
        }
    }
    return 0;
}
