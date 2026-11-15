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
    int LOG = 1;
    while ((1 << LOG) <= n) LOG++;
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<int> dep(n + 1, 0), qv = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = qv[i];
        for (int v : adj[u]) if (!dep[v]) {
            dep[v] = dep[u] + 1;
            up[0][v] = u;
            qv.push_back(v);
        }
    }
    for (int b = 1; b < LOG; b++)
        for (int i = 1; i <= n; i++)
            up[b][i] = up[b - 1][up[b - 1][i]];

    auto kth_anc = [&](int u, int k) {
        for (int b = 0; b < LOG; b++) if ((k >> b) & 1) u = up[b][u];
        return u;
    };
    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        u = kth_anc(u, dep[u] - dep[v]);
        if (u == v) return u;
        for (int b = LOG - 1; b >= 0; b--)
            if (up[b][u] != up[b][v]) { u = up[b][u]; v = up[b][v]; }
        return up[0][u];
    };

    while (q--) {
        int u, v, k;
        cin >> u >> v >> k;
        int w = lca(u, v);
        int du = dep[u] - dep[w], dv = dep[v] - dep[w];
        if (k > du + dv) cout << -1 << '\n';
        else if (k <= du) cout << kth_anc(u, k) << '\n';
        else cout << kth_anc(v, du + dv - k) << '\n';
    }
    return 0;
}
