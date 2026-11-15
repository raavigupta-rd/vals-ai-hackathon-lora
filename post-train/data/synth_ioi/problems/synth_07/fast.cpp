#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> w(n + 1);
    long long total_w = 0;
    for (int i = 1; i <= n; i++) { cin >> w[i]; total_w += w[i]; }
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long l;
        cin >> u >> v >> l;
        adj[u].push_back({v, l});
        adj[v].push_back({u, l});
    }
    vector<int> par(n + 1, 0), ord;
    vector<long long> plen(n + 1, 0), dist1(n + 1, 0);
    ord.reserve(n);
    ord.push_back(1);
    par[1] = -1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, l] : adj[u]) if (v != par[u]) {
            par[v] = u;
            plen[v] = l;
            dist1[v] = dist1[u] + l;
            ord.push_back(v);
        }
    }
    vector<long long> subw = w, ans(n + 1, 0);
    for (int i = 1; i <= n; i++) ans[1] += w[i] * dist1[i];
    for (int i = n - 1; i >= 1; i--) {
        int u = ord[i];
        subw[par[u]] += subw[u];
    }
    for (int i = 1; i < n; i++) {
        int u = ord[i];
        ans[u] = ans[par[u]] + (total_w - 2LL * subw[u]) * plen[u];
    }
    for (int i = 1; i <= n; i++) cout << ans[i] << '\n';
    return 0;
}
