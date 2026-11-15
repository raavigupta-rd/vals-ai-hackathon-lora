#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i];
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long l;
        cin >> u >> v >> l;
        adj[u].push_back({v, l});
        adj[v].push_back({u, l});
    }
    for (int root = 1; root <= n; root++) {
        long long total = 0;
        vector<tuple<int, int, long long>> st = {{root, 0, 0}};
        while (!st.empty()) {
            auto [u, p, d] = st.back(); st.pop_back();
            total += w[u] * d;
            for (auto [v, l] : adj[u]) if (v != p) st.push_back({v, u, d + l});
        }
        cout << total << '\n';
    }
    return 0;
}
