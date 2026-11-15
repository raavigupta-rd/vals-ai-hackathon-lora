#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, T;
    cin >> n >> m >> T;
    struct Pact { int u, v, l, r; };
    vector<Pact> ps(m);
    for (int i = 0; i < m; i++) cin >> ps[i].u >> ps[i].v >> ps[i].l >> ps[i].r;

    vector<vector<int>> adj(n + 1);
    vector<int> col(n + 1, -1), touched;
    for (int t = 1; t <= T; t++) {
        touched.clear();
        for (const auto& p : ps) if (p.l <= t && t <= p.r) {
            if (adj[p.u].empty()) touched.push_back(p.u);
            if (adj[p.v].empty()) touched.push_back(p.v);
            adj[p.u].push_back(p.v);
            adj[p.v].push_back(p.u);
        }
        bool ok = true;
        for (int s : touched) if (col[s] == -1 && ok) {
            col[s] = 0;
            vector<int> st = {s};
            while (!st.empty() && ok) {
                int u = st.back(); st.pop_back();
                for (int v : adj[u]) {
                    if (col[v] == -1) { col[v] = col[u] ^ 1; st.push_back(v); }
                    else if (col[v] == col[u]) { ok = false; break; }
                }
            }
        }
        for (int u : touched) { adj[u].clear(); col[u] = -1; }
        cout << (ok ? "YES\n" : "NO\n");
    }
    return 0;
}
