#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }
    // For each query, find ONE path A->B, then test removing each edge on that path!
    // Still O(Q * (N + M) + Q * path_len * M) -> make small N,M,Q <= 500 or O(Q*(N+M)) via per-query Tarjan!
    // Per-query DFS from A tracking bridges on the ancestor chain to B is O(N + M) per query => O(Q*(N+M))!
    vector<int> tin(n + 1), low(n + 1), par_e(n + 1), par_u(n + 1);
    while (q--) {
        int a, b;
        cin >> a >> b;
        if (a == b) { cout << 0 << '\n'; continue; }
        fill(tin.begin(), tin.end(), 0);
        int timer = 0;
        function<void(int, int)> dfs = [&](int u, int pe) {
            tin[u] = low[u] = ++timer;
            for (auto [v, eid] : adj[u]) {
                if (eid == pe) continue;
                if (tin[v]) low[u] = min(low[u], tin[v]);
                else {
                    par_u[v] = u;
                    par_e[v] = eid;
                    dfs(v, eid);
                    low[u] = min(low[u], low[v]);
                }
            }
        };
        dfs(a, -1);
        int ans = 0, cur = b;
        while (cur != a) {
            int p = par_u[cur];
            if (low[cur] > tin[p]) ans++;
            cur = p;
        }
        cout << ans << '\n';
    }
    return 0;
}
