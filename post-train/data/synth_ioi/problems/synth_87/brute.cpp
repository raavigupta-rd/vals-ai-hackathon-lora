#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
    }

    vector<char> reach0(n + 1, 0), vis(n + 1, 0);
    vector<int> qv = {1};
    reach0[1] = 1;
    for (size_t i = 0; i < qv.size(); i++) {
        for (int v : g[qv[i]]) if (!reach0[v]) {
            reach0[v] = 1;
            qv.push_back(v);
        }
    }

    vector<int> dom_cnt(n + 1, 0);
    for (int u = 1; u <= n; u++) if (reach0[u]) dom_cnt[u] = 1; // depot 1 dominates all reachable

    for (int ban = 2; ban <= n; ban++) {
        if (!reach0[ban]) continue;
        fill(vis.begin(), vis.end(), 0);
        vis[1] = 1;
        qv.assign(1, 1);
        for (size_t i = 0; i < qv.size(); i++) {
            int u = qv[i];
            for (int v : g[u]) if (v != ban && !vis[v]) {
                vis[v] = 1;
                qv.push_back(v);
            }
        }
        for (int v = 1; v <= n; v++) {
            if (reach0[v] && !vis[v]) dom_cnt[v]++;
        }
    }

    for (int u = 1; u <= n; u++) {
        cout << dom_cnt[u] << (u == n ? '\n' : ' ');
    }
    return 0;
}
