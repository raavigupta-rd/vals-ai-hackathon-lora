#include <bits/stdc++.h>
using namespace std;

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
    vector<int> par(n + 1, 0), dep(n + 1, 0);
    vector<int> qv = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = qv[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] + 1;
            qv.push_back(v);
        }
    }
    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int u; long long x;
            cin >> u >> x;
            vector<int> st = {u};
            while (!st.empty()) {
                int c = st.back(); st.pop_back();
                a[c] += x;
                for (int v : adj[c]) if (v != par[c]) st.push_back(v);
            }
        } else {
            int u, v;
            cin >> u >> v;
            long long s = 0;
            while (dep[u] > dep[v]) { s += a[u]; u = par[u]; }
            while (dep[v] > dep[u]) { s += a[v]; v = par[v]; }
            while (u != v) { s += a[u] + a[v]; u = par[u]; v = par[v]; }
            s += a[u];
            cout << s << '\n';
        }
    }
    return 0;
}
