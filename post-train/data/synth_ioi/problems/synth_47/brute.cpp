#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<char> open(n + 1, 0);
    open[1] = 1;
    vector<int> d(n + 1, -1), qv;
    while (q--) {
        int t, u; cin >> t >> u;
        if (t == 1) open[u] = 1;
        else {
            if (open[u]) { cout << 0 << '\n'; continue; }
            qv.assign(1, u);
            d[u] = 0;
            int ans = -1;
            for (size_t i = 0; i < qv.size(); i++) {
                int x = qv[i];
                for (int y : adj[x]) if (d[y] == -1) {
                    d[y] = d[x] + 1;
                    qv.push_back(y);
                    if (open[y]) { ans = d[y]; break; }
                }
                if (ans != -1) break;
            }
            for (int x : qv) d[x] = -1;
            cout << ans << '\n';
        }
    }
    return 0;
}
