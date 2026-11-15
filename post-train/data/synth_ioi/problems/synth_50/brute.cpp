#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, K;
    cin >> n >> K;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    long long ans = 0;
    for (int s = 1; s <= n; s++) {
        vector<tuple<int, int, int>> st = {{s, 0, 0}};
        while (!st.empty()) {
            auto [u, p, d] = st.back(); st.pop_back();
            if (d == K) { if (u > s) ans++; continue; }
            for (int v : adj[u]) if (v != p) st.push_back({v, u, d + 1});
        }
    }
    cout << ans << '\n';
    return 0;
}
