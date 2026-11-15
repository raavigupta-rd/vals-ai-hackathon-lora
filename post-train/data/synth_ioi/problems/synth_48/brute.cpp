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
    vector<int> par(n + 1, 0), dep(n + 1, 0), ord = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] + 1;
            ord.push_back(v);
        }
    }
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int u, x; cin >> u >> x;
            a[u] = x;
        } else {
            int u, v; cin >> u >> v;
            int mx = 0;
            while (dep[u] > dep[v]) { mx = max(mx, a[u]); u = par[u]; }
            while (dep[v] > dep[u]) { mx = max(mx, a[v]); v = par[v]; }
            while (u != v) { mx = max({mx, a[u], a[v]}); u = par[u]; v = par[v]; }
            mx = max(mx, a[u]);
            cout << mx << '\n';
        }
    }
    return 0;
}
