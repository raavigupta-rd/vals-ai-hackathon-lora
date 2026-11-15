#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long mod;
    cin >> n >> mod;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1), ord;
    vector<long long> dp(n + 1);
    ord.reserve(n);
    for (int root = 1; root <= n; root++) {
        ord.clear();
        fill(par.begin(), par.end(), 0);
        ord.push_back(root);
        par[root] = -1;
        for (int i = 0; i < n; i++) {
            int u = ord[i];
            for (int v : adj[u]) if (v != par[u]) {
                par[v] = u;
                ord.push_back(v);
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            int u = ord[i];
            long long prod = 1;
            for (int v : adj[u]) if (v != par[u]) {
                prod = (prod * (dp[v] + 1)) % mod;
            }
            dp[u] = prod;
        }
        cout << dp[root] % mod << '\n';
    }
    return 0;
}
