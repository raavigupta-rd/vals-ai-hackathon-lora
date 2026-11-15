#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long mod;
    cin >> n >> mod;
    vector<vector<int>> adj(n + 1), ch(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), ord;
    ord.reserve(n);
    ord.push_back(1);
    par[1] = -1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ch[u].push_back(v);
            ord.push_back(v);
        }
    }

    vector<long long> down(n + 1, 1), up_fac(n + 1, 1);
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        long long prod = 1;
        for (int c : ch[u]) prod = (prod * (down[c] + 1)) % mod;
        down[u] = prod;
    }

    for (int u : ord) {
        int deg = (int)ch[u].size();
        vector<long long> pref(deg + 1, 1), suff(deg + 1, 1);
        for (int i = 0; i < deg; i++) pref[i + 1] = (pref[i] * (down[ch[u][i]] + 1)) % mod;
        for (int i = deg - 1; i >= 0; i--) suff[i] = (suff[i + 1] * (down[ch[u][i]] + 1)) % mod;
        for (int i = 0; i < deg; i++) {
            int c = ch[u][i];
            long long u_without_c = up_fac[u] * pref[i] % mod * suff[i + 1] % mod;
            up_fac[c] = (u_without_c + 1) % mod;
        }
    }

    for (int u = 1; u <= n; u++) {
        cout << (down[u] * up_fac[u]) % mod << '\n';
    }
    return 0;
}
