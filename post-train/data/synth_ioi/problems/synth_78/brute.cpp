#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    struct E { int to, col; long long w; };
    vector<vector<E>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, c; long long w;
        cin >> u >> v >> c >> w;
        adj[u].push_back({v, c, w});
        adj[v].push_back({u, c, w});
    }
    vector<int> par(n + 1, 0), dep(n + 1, 0), pcol(n + 1, 0), ord = {1};
    vector<long long> pw(n + 1, 0);
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto e : adj[u]) if (e.to != par[u]) {
            par[e.to] = u;
            dep[e.to] = dep[u] + 1;
            pcol[e.to] = e.col;
            pw[e.to] = e.w;
            ord.push_back(e.to);
        }
    }
    while (q--) {
        int a, b, k; long long nw;
        cin >> a >> b >> k >> nw;
        long long tot = 0;
        auto step_up = [&](int& x) {
            tot += (pcol[x] == k ? nw : pw[x]);
            x = par[x];
        };
        while (dep[a] > dep[b]) step_up(a);
        while (dep[b] > dep[a]) step_up(b);
        while (a != b) { step_up(a); step_up(b); }
        cout << tot << '\n';
    }
    return 0;
}
