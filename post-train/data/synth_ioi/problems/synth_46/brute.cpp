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
    vector<int> par(n + 1, 0), dep(n + 1, 0), qv = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = qv[i];
        for (int v : adj[u]) if (!dep[v]) {
            dep[v] = dep[u] + 1;
            par[v] = u;
            qv.push_back(v);
        }
    }
    while (q--) {
        int u, v, k;
        cin >> u >> v >> k;
        vector<int> L, R;
        int a = u, b = v;
        while (dep[a] > dep[b]) { L.push_back(a); a = par[a]; }
        while (dep[b] > dep[a]) { R.push_back(b); b = par[b]; }
        while (a != b) { L.push_back(a); R.push_back(b); a = par[a]; b = par[b]; }
        L.push_back(a);
        L.insert(L.end(), R.rbegin(), R.rend());
        cout << (k < (int)L.size() ? L[k] : -1) << '\n';
    }
    return 0;
}
