#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<int>> adj(n + 1);
    vector<int> indeg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        indeg[v]++;
    }
    vector<int> topo;
    for (int i = 1; i <= n; i++) if (!indeg[i]) topo.push_back(i);
    for (size_t i = 0; i < topo.size(); i++) {
        int u = topo[i];
        for (int v : adj[u]) if (--indeg[v] == 0) topo.push_back(v);
    }
    vector<int> g(n + 1, 0), seen(n + 2, 0);
    for (int i = n - 1; i >= 0; i--) {
        int u = topo[i];
        for (int v : adj[u]) seen[g[v]] = u;
        int mex = 0;
        while (seen[mex] == u) mex++;
        g[u] = mex;
    }
    while (q--) {
        int k; cin >> k;
        vector<int> p(k);
        int X = 0;
        for (int i = 0; i < k; i++) { cin >> p[i]; X ^= g[p[i]]; }
        if (X == 0) { cout << "SECOND 0\n"; continue; }
        long long ways = 0;
        for (int u : p) {
            int need = X ^ g[u];
            for (int v : adj[u]) if (g[v] == need) ways++;
        }
        cout << "FIRST " << ways << '\n';
    }
    return 0;
}
