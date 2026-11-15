#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> par(n + 1, 0), ord = {1};
    vector<long long> pw(n + 1, 0);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, w] : adj[u]) if (v != par[u]) {
            par[v] = u;
            pw[v] = w;
            ord.push_back(v);
        }
    }
    vector<int> cnt(n + 1, 0);
    while (q--) {
        int k;
        cin >> k;
        fill(cnt.begin(), cnt.end(), 0);
        for (int i = 0; i < k; i++) {
            int x; cin >> x;
            cnt[x] = 1;
        }
        long long ans = 0;
        for (int i = n - 1; i >= 1; i--) {
            int u = ord[i];
            if (cnt[u] > 0 && cnt[u] < k) ans += pw[u];
            cnt[par[u]] += cnt[u];
        }
        cout << ans << '\n';
    }
    return 0;
}
