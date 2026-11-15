#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v, h;
        cin >> u >> v >> h;
        adj[u].push_back({v, h});
        adj[v].push_back({u, h});
    }
    vector<int> vis(n + 1, 0);
    for (int step = 1; step <= q; step++) {
        int s, t;
        cin >> s >> t;
        vector<int> st = {s};
        vis[s] = step;
        int cnt = 0;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            cnt++;
            for (auto [v, h] : adj[u]) if (h > t && vis[v] != step) {
                vis[v] = step;
                st.push_back(v);
            }
        }
        cout << cnt << '\n';
    }
    return 0;
}
