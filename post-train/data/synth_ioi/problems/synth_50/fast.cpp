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

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), tmp_par(n + 1, 0), ord, cnt(K + 1, 0), sub_d;
    long long ans = 0;

    function<void(int)> solve = [&](int start) {
        ord.clear();
        ord.push_back(start);
        tmp_par[start] = 0;
        for (size_t i = 0; i < ord.size(); i++) {
            int u = ord[i];
            sz[u] = 1;
            for (int v : adj[u]) if (!dead[v] && v != tmp_par[u]) {
                tmp_par[v] = u;
                ord.push_back(v);
            }
        }
        int tot = (int)ord.size();
        if (tot <= K) return;
        for (int i = tot - 1; i >= 1; i--) sz[tmp_par[ord[i]]] += sz[ord[i]];
        int c = start;
        while (true) {
            int nxt = 0;
            for (int v : adj[c]) if (!dead[v] && v != tmp_par[c] && sz[v] > tot / 2) {
                nxt = v; break;
            }
            if (!nxt) break;
            c = nxt;
        }
        dead[c] = 1;
        cnt[0] = 1;
        int max_d = 0;
        for (int nb : adj[c]) if (!dead[nb]) {
            sub_d.clear();
            vector<tuple<int, int, int>> st = {{nb, c, 1}};
            while (!st.empty()) {
                auto [u, p, d] = st.back(); st.pop_back();
                sub_d.push_back(d);
                if (d < K) {
                    for (int v : adj[u]) if (!dead[v] && v != p) st.push_back({v, u, d + 1});
                }
            }
            for (int d : sub_d) ans += cnt[K - d];
            for (int d : sub_d) { cnt[d]++; max_d = max(max_d, d); }
        }
        for (int d = 0; d <= max_d; d++) cnt[d] = 0;
        for (int nb : adj[c]) if (!dead[nb]) solve(nb);
    };

    solve(1);
    cout << ans << '\n';
    return 0;
}
