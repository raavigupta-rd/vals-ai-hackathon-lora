#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    // Find one shortest BFS path A->B; any vertex separating A and B MUST lie on this path!
    // Test deleting each internal vertex on that path via BFS -> O(Q * (N + M)) worst-case on chains!
    vector<int> par(n + 1, 0), vis(n + 1, 0);
    int stamp = 0;
    while (q--) {
        int a, b; cin >> a >> b;
        ++stamp;
        vector<int> qv = {a};
        vis[a] = stamp;
        par[a] = 0;
        for (size_t i = 0; i < qv.size(); i++) {
            int u = qv[i];
            if (u == b) break;
            for (int v : g[u]) if (vis[v] != stamp) {
                vis[v] = stamp;
                par[v] = u;
                qv.push_back(v);
            }
        }
        vector<int> cands;
        for (int x = par[b]; x != a && x != 0; x = par[x]) cands.push_back(x);
        int ans = 0;
        for (int ban : cands) {
            ++stamp;
            vis[a] = stamp;
            vis[ban] = stamp;
            qv.assign(1, a);
            bool reach = false;
            for (size_t i = 0; i < qv.size(); i++) {
                int u = qv[i];
                if (u == b) { reach = true; break; }
                for (int v : g[u]) if (vis[v] != stamp) {
                    vis[v] = stamp;
                    qv.push_back(v);
                }
            }
            if (!reach) ans++;
        }
        cout << ans << '\n';
    }
    return 0;
}
