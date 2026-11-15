#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> col(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1, 0), dep(n + 1, 0), ord = {1};
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] ^ 1;
            ord.push_back(v);
        }
    }
    vector<array<int, 2>> cnt(n + 1, {0, 0});
    for (int root = 1; root <= n; root++) {
        vector<int> st = {root}, vis;
        long long pairs = 0;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            vis.push_back(u);
            pairs += cnt[col[u]][dep[u]];
            cnt[col[u]][dep[u]]++;
            for (int v : adj[u]) if (v != par[u]) st.push_back(v);
        }
        for (int u : vis) cnt[col[u]] = {0, 0};
        cout << pairs << '\n';
    }
    return 0;
}
