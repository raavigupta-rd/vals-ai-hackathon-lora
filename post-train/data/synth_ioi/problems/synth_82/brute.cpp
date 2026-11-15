#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1, 0), ord = {1};
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ord.push_back(v);
        }
    }
    vector<long long> dp0(n + 1), dp1(n + 1);
    while (q--) {
        int u; long long x;
        cin >> u >> x;
        a[u] = x;
        for (int i = n - 1; i >= 0; i--) {
            int v = ord[i];
            long long s0 = 0, s1 = a[v];
            for (int nb : adj[v]) if (nb != par[v]) {
                s0 += max(dp0[nb], dp1[nb]);
                s1 += dp0[nb];
            }
            dp0[v] = s0;
            dp1[v] = s1;
        }
        cout << max(dp0[1], dp1[1]) << '\n';
    }
    return 0;
}
