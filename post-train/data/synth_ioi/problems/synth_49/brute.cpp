#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<int>> adj(n + 1);
    vector<int> d(n + 1, -1), qv;
    auto bfs = [&](int s) -> pair<int, int> {
        qv.assign(1, s);
        d[s] = 0;
        int far = s;
        for (size_t i = 0; i < qv.size(); i++) {
            int u = qv[i];
            if (d[u] > d[far]) far = u;
            for (int v : adj[u]) if (d[v] == -1) {
                d[v] = d[u] + 1;
                qv.push_back(v);
            }
        }
        int mx = d[far];
        for (int x : qv) d[x] = -1;
        return {far, mx};
    };
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        auto [f, _] = bfs(u);
        auto [__, dm] = bfs(f);
        cout << dm << '\n';
    }
    return 0;
}
