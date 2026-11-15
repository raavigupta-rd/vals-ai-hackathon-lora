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
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        indeg[v]++;
    }

    vector<int> topo;
    topo.reserve(n);
    for (int i = 1; i <= n; i++) if (!indeg[i]) topo.push_back(i);
    for (size_t i = 0; i < topo.size(); i++) {
        int u = topo[i];
        for (int v : adj[u]) if (--indeg[v] == 0) topo.push_back(v);
    }

    vector<int> g(n + 1, 0), seen(n + 2, 0);
    vector<vector<pair<int, int>>> nfreq(n + 1);
    for (int i = n - 1; i >= 0; i--) {
        int u = topo[i];
        vector<int> vals;
        vals.reserve(adj[u].size());
        for (int v : adj[u]) {
            seen[g[v]] = u;
            vals.push_back(g[v]);
        }
        int mex = 0;
        while (seen[mex] == u) mex++;
        g[u] = mex;
        sort(vals.begin(), vals.end());
        for (int x : vals) {
            if (nfreq[u].empty() || nfreq[u].back().first != x) nfreq[u].push_back({x, 1});
            else nfreq[u].back().second++;
        }
    }

    while (q--) {
        int k;
        cin >> k;
        vector<int> p(k);
        int X = 0;
        for (int i = 0; i < k; i++) {
            cin >> p[i];
            X ^= g[p[i]];
        }
        if (X == 0) {
            cout << "SECOND 0\n";
            continue;
        }
        long long ways = 0;
        for (int u : p) {
            int need = X ^ g[u];
            const auto& fv = nfreq[u];
            auto it = lower_bound(fv.begin(), fv.end(), make_pair(need, 0));
            if (it != fv.end() && it->first == need) ways += it->second;
        }
        cout << "FIRST " << ways << '\n';
    }
    return 0;
}
